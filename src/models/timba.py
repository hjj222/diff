import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["pointCoder", "pointwhCoder"]
from src.models.layers_pristi import *
from src.utils_pristi import *

from src.models.layers import CustomMamba


class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1., 1.), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        # self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2. / self.patch_count
        for i in range(self.patch_count):
            x = -1 + (0.5 + i) * patch_stride_x
            anchors.append([x])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        # self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1. / self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel if self.tanh else rel_codes[:, :, 0] * pixel / wx
        dy = F.tanh(rel_codes[:, :, 1] / wy) * pixel if self.tanh else rel_codes[:, :, 1] * pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:, 0].unsqueeze(0)
        ref_y = boxes[:, 1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size


class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1., 1.), pts=1, tanh=True, wh_bias=None,
                 deform_range=0.25):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        # print(boxes.shape)
        # print(self.wh_bias.shape)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x = 2. / self.patch_count  # patch_count=in_size//stride 这里应该用2除而不是1除 得到pixel_x是两个patch中点的原本距离
        wx, ww1, ww2 = self.weights

        dx = F.tanh(rel_codes[:, :, 0] / wx) * pixel_x / 4 if self.tanh else rel_codes[:, :,
                                                                             0] * pixel_x / wx  # 中心点不会偏移超过patch_len

        dw1 = F.relu(F.tanh(rel_codes[:, :,
                            1] / ww1)) * pixel_x * self.deform_range + pixel_x  # 中心点左边长度在[stride,stride+1/4*stride]，右边同理
        dw2 = F.relu(F.tanh(rel_codes[:, :, 2] / ww2)) * pixel_x * self.deform_range + pixel_x  #
        # dw =

        pred_boxes = torch.zeros((rel_codes.shape[0], rel_codes.shape[1], rel_codes.shape[2] - 1)).to(rel_codes.device)

        ref_x = boxes[:, 0].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs = boxes
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        results = xs
        results = results.reshape(B, self.patch_count, self.patch_pixel, 1)
        # print((1+results[0])/2*336)
        return results
class SideInfo(nn.Module):
    def __init__(self, time_steps, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.embed_layer = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=16)

        self.arange = torch.arange(self.num_nodes)
    
    def get_time(self, B):
        observed_tp = torch.arange(self.time_steps).unsqueeze(0)
        pos = torch.cat([observed_tp for _ in range(B)], dim=0)
        self.div_term = 1 / torch.pow(
            10000.0, torch.arange(0, 128, 2) / 128
        )
        pe = torch.zeros(pos.shape[0], pos.shape[1], 128)
        position = pos.unsqueeze(2)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe

    def forward(self, cond_mask):
        B, _, K, L = cond_mask.shape

        observed_tp= torch.arange(L).to(cond_mask.device).unsqueeze(0)
        observed_tp = torch.cat([observed_tp for _ in range(B)], dim=0)
        time_embed = self.get_time(B).unsqueeze(2).expand(-1, -1, K, -1).to(cond_mask.device)
        self.arange = self.arange.to(cond_mask.device)
        feature_embed = self.embed_layer(self.arange)  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info.to(cond_mask.device)
class HdMixer1(nn.Module):
    def __init__(self, time_steps, num_nodes,channels):
        super().__init__()
        self.channels=channels
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.patch_len = 207
        self.stride = 207
        context_window =24*207
        self.patch_num  = context_window // self.stride
        self.patch_shift_linear = nn.Linear(207*24, self.patch_num * 3)
        self.embed_layer = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=16)
        self.n_vars = 2
        self.box_coder = pointwhCoder(input_size=context_window, patch_count=self.patch_num, weights=(1., 1., 1.),
                                      pts=self.patch_len, tanh=True, wh_bias=torch.tensor(5. / 3.).sqrt().log(),
                                      deform_range=1)
    def forward(self, x):
        B, _, K, L = x.shape
        batch_size = x.shape[0]
        x=x.reshape(B,_,K*L)
        seq_len =x.shape[-1]
        observed_tp = torch.arange(L).to(x.device).unsqueeze(0)
        anchor_shift = self.patch_shift_linear(x).view(batch_size * self.n_vars, self.patch_num, 3)
        sampling_location_1d = self.box_coder(anchor_shift)  # B*C, self.patch_num,self.patch_len, 1
        add1d = torch.ones(size=(batch_size * self.n_vars, self.patch_num, self.patch_len, 1)).float().to(
            sampling_location_1d.device)
        sampling_location_2d = torch.cat([sampling_location_1d, add1d], dim=-1)
        x = x.reshape(batch_size * self.n_vars, 1,1,seq_len)
        patch= F.grid_sample(x, sampling_location_2d, mode='bilinear', padding_mode='border',
                              align_corners=False).squeeze(1)  # B*C, self.patch_num,self.patch_len
        # x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        #
        # B, _, K, num_patches, patch_len = x.shape
        #
        # x = x.reshape(B, _, K, num_patches * patch_len)

        # [bs x nvars x patch_len x patch_num]
        x = patch.reshape(batch_size,self.n_vars,self.patch_num,self.patch_len).permute(0,1,3,2)
        return x

    class HdMixer2(nn.Module):
        def __init__(self, time_steps, num_nodes, channels):
            super().__init__()
            self.channels = channels
            self.num_nodes = num_nodes
            self.time_steps = time_steps
            self.patch_len = 207
            self.stride = 207
            context_window = 24 * 207
            self.patch_num = context_window // self.stride
            self.patch_shift_linear = nn.Linear(207 * 24, self.patch_num * 3)
            self.embed_layer = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=16)
            self.n_vars = 2
            self.box_coder = pointwhCoder(input_size=context_window, patch_count=self.patch_num, weights=(1., 1., 1.),
                                          pts=self.patch_len, tanh=True, wh_bias=torch.tensor(5. / 3.).sqrt().log(),
                                          deform_range=1)

        def forward(self, x):
            B, _, K, L = x.shape
            batch_size = x.shape[0]
            x = x.reshape(B, _, K * L)
            seq_len = x.shape[-1]

            anchor_shift = self.patch_shift_linear(x).view(batch_size * self.n_vars, self.patch_num, 3)
            sampling_location_1d = self.box_coder(anchor_shift)  # [B*C, patch_num, patch_len, 1]

            # 转换为整数索引并限制范围
            sampling_indices = sampling_location_1d.squeeze(-1).long()  # [B*C, patch_num, patch_len]
            sampling_indices = torch.clamp(sampling_indices, 0, seq_len - 1)

            # 重塑输入用于索引 [B*C, seq_len]
            x_flat = x.reshape(batch_size * self.n_vars, seq_len)

            # 使用 gather 进行批量索引提取
            # 扩展维度以便使用 gather
            batch_indices = torch.arange(batch_size * self.n_vars, device=x.device)
            batch_indices = batch_indices.view(-1, 1, 1).expand(-1, self.patch_num, self.patch_len)

            # 提取补丁 [B*C, patch_num, patch_len]
            patch = x_flat[batch_indices, sampling_indices]

            # 重新组织补丁结构 [B, n_vars, patch_len, patch_num]
            x = patch.reshape(batch_size, self.n_vars, self.patch_num, self.patch_len).permute(0, 1, 3, 2)

            return x
class NoiseProject(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, target_dim, proj_t, order=2, include_self=True,
                 device=None, is_adp=False, adj_file=None, is_cross_t=False, is_cross_s=True, num_nodes=None, time_steps=None, bidirectional=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.forward_time = CustomMamba(channels=channels, t=time_steps, n=num_nodes, bidirectional=bidirectional)
        self.forward_feature = SpatialLearning(channels=channels, nheads=nheads//2, target_dim=target_dim,
                                               order=order, include_self=include_self, device=device, is_adp=is_adp,
                                               adj_file=adj_file, proj_t=proj_t, is_cross=is_cross_s)

    def forward(self, x, side_info, diffusion_emb, itp_info, support):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, itp_info)
        y = self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        return (x + residual) / math.sqrt(2.0), skip

class TIMBA(nn.Module):
    def __init__(self, inputdim=2, is_itp=True, config=None):
        super().__init__()
        
        self.num_nodes = config['num_nodes']
        self.channels = config["channels"]
        self.time_steps = config["time_steps"]
        self.batch_size = config["batch_size"]
        self.bidirectional = config["bidirectional"]

        self.side_info = SideInfo(self.time_steps, self.num_nodes)
        self.hdmixer = HdMixer1(self.time_steps, self.num_nodes,self.channels)
        self.is_itp = is_itp
        self.itp_channels = None
        if self.is_itp:
            self.itp_channels = config["channels"]
            self.itp_projection = Conv1d_with_init(inputdim-1, self.itp_channels, 1)

            self.itp_modeling = GuidanceConstructTimba(channels=self.itp_channels, nheads=config["nheads"], target_dim=self.num_nodes,
                                            order=2, include_self=True, device=None, is_adp=config["is_adp"],
                                            adj_file=config["adj_file"], proj_t=config["proj_t"], time_steps = config["time_steps"], num_nodes = config['num_nodes'], bidirectional=self.bidirectional)
            self.cond_projection = Conv1d_with_init(config["side_dim"], self.itp_channels, 1)

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        if config["adj_file"] == 'AQI36':
            self.adj = get_adj_AQI36()
        elif config["adj_file"] == 'metr-la':
            self.adj = get_similarity_metrla(thr=0.1)
        elif config["adj_file"] == 'pems-bay':
            self.adj = get_similarity_pemsbay(thr=0.1)
        elif config["adj_file"] == 'mimic-iii':
            self.adj = get_similarity_mimic(thr=0.1)

        self.support = compute_support_gwn(self.adj)
        self.is_adp = config["is_adp"]
        if self.is_adp:
            node_num = self.adj.shape[0]
            self.nodevec1 = nn.Parameter(torch.randn(node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, node_num), requires_grad=True)
            self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    target_dim=self.num_nodes,
                    proj_t=config["proj_t"],
                    is_adp=config["is_adp"],
                    device=None,
                    adj_file=config["adj_file"],
                    is_cross_t=config["is_cross_t"],
                    is_cross_s=config["is_cross_s"],
                    time_steps = config["time_steps"],
                    num_nodes = config['num_nodes'],
                    bidirectional=self.bidirectional
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, itp_x, u, diffusion_step):


        if self.is_itp:
            x = torch.cat([x, itp_x], dim=-1)

        x = x.permute(0, 3, 2, 1) # B, input_dim, K,
        x = self.hdmixer(x)
        side_info = self.side_info(x)
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        if self.is_itp:
            itp_x = itp_x.reshape(B, inputdim-1, K * L)
            itp_x = self.itp_projection(itp_x)
            itp_cond_info = side_info.reshape(B, -1, K * L)
            itp_cond_info = self.cond_projection(itp_cond_info)
            itp_x = itp_x + itp_cond_info
            itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
            itp_x = F.relu(itp_x)
            itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for i in range(len(self.residual_layers)):
            x, skip_connection = self.residual_layers[i](x, side_info, diffusion_emb, itp_x, self.support)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        # x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1, K*L)
        x = x.reshape(B, 1, K, L).permute(0, 3, 2, 1)

        return x