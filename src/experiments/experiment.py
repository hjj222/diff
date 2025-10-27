import os
import time
import random
import numpy as np
import pandas as pd
from omegaconf import open_dict
import sys
import torch
from schedulefree import AdamWScheduleFree
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from tsl.metrics import torch as torch_metrics
from src.data.traffic import MetrLADataset, PemsBayDataset
from src.data.airquality import AQI36Dataset
from src.data.mimiciii import MimicIIIDataset
from src.models.diffusion import DiffusionImputer
from pathlib import Path

from torch_geometric.data import Data
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from src.data.data_handlers import create_interpolation  # 数据插值工具函数
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class Experiment:
    def __init__(self, dataset, cfg, optimizer_type, epochs, accelerator='gpu', device=None, seed=42):
        # 核心参数初始化
        self.cfg = cfg
        self.dataset = dataset
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.accelerator = accelerator

        # 设备配置
        self.device = torch.device(f"cuda:{device}") if device is not None else \
            (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.seed = seed

        # 节点替换配置
        self.replace_test_node = 0  # 是否启用测试集节点替换
        self.replace_from = 8  # 被替换节点
        self.replace_to = 9  # 替换目标节点

        # 固定随机种子
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 可视化样式配置
        plt.rcParams["font.family"] = ["monospace", "sans-serif"]
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Courier New", "monospace"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 80
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams["axes.labelsize"] = 9
        plt.rcParams["legend.fontsize"] = 8
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8

        # 颜色配置
        self.line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 原始/缺失/补全数据颜色
        self.node_colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 节点区分色

    def _replace_test_nodes(self, batch):
        """测试集节点替换逻辑"""
        if not self.replace_test_node:
            return batch

        if isinstance(batch, list):
            return [self._replace_test_nodes(item) for item in batch]

        if isinstance(batch, dict):
            batch_copy = batch.copy()
            if 'x' in batch_copy and batch_copy['x'].size(-2) > max(self.replace_from, self.replace_to):
                batch_copy['x'][..., self.replace_from, :] = batch_copy['x'][..., self.replace_to, :].clone()
            if 'mask' in batch_copy and batch_copy['mask'].size(-2) > max(self.replace_from, self.replace_to):
                batch_copy['mask'][..., self.replace_from, :] = batch_copy['mask'][..., self.replace_to, :].clone()
            if 'y' in batch_copy and batch_copy['y'].size(-2) > max(self.replace_from, self.replace_to):
                batch_copy['y'][..., self.replace_from, :] = batch_copy['y'][..., self.replace_to, :].clone()
            return batch_copy

        if hasattr(batch, 'input') and hasattr(batch.input, 'x'):
            if batch.input.x.size(-2) > max(self.replace_from, self.replace_to):
                batch.input.x[..., self.replace_from, :] = batch.input.x[..., self.replace_to, :].clone()
                if hasattr(batch.input, 'mask'):
                    batch.input.mask[..., self.replace_from, :] = batch.input.mask[..., self.replace_to, :].clone()
                if hasattr(batch.target, 'y'):
                    batch.target.y[..., self.replace_from, :] = batch.target.y[..., self.replace_to, :].clone()

        return batch

    def prepare_data(self):
        """数据加载与准备"""
        dm_params = {
            'batch_size': self.cfg.config.batch_size,
            'scale_window_factor': self.cfg.config.scale_window_factor
        }

        # 数据集选择
        if self.dataset == 'metr-la':
            data_class = MetrLADataset
            dm_params['point'] = self.cfg['dataset']['scenario'] == 'point'
        elif self.dataset == 'pems-bay':
            data_class = PemsBayDataset
            dm_params['point'] = self.cfg['dataset']['scenario'] == 'point'
        elif self.dataset == 'aqi-36':
            data_class = AQI36Dataset
        elif self.dataset == 'mimic-iii':
            data_class = MimicIIIDataset
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # 数据加载器初始化
        self.dm = data_class(**dm_params).get_dm()
        self.dm_stride = data_class(stride='window_size', **dm_params).get_dm()

        # 历史模式加载（如果需要）
        if self.cfg.missing_pattern.strategy1 == 'historical' or self.cfg.missing_pattern.strategy2 == 'historical':
            self.hist_patterns = data_class(test_months=(2, 5, 8, 11), **dm_params).get_historical_patterns()
        else:
            self.hist_patterns = None

        # 数据集划分
        self.dm.setup()
        self.dm_stride.setup()
        print(f"Dataset loaded | Train: {self.dm.train_len} | Val: {self.dm.val_len} | Test: {self.dm.test_len}")

        # 更新配置参数
        with open_dict(self.cfg):
            self.cfg.config.time_steps = self.dm.window
            self.cfg.config.num_nodes = self.dm.n_nodes

        # 数据加载器
        self.train_dataloader = self.dm.train_dataloader()
        self.val_dataloader = self.dm_stride.val_dataloader()
        self.test_dataloader = self.dm_stride.test_dataloader()

        # 节点替换提示
        if self.replace_test_node:
            print(f"测试集节点替换: {self.replace_from} -> {self.replace_to}")
            print(f"测试集加载器类型: {type(self.test_dataloader).__name__}")
        else:
            print("未启用测试集节点替换")

    def _wrap_dataloader_for_replacement(self, dataloader):
        """包装数据加载器以支持节点替换"""
        if not self.replace_test_node:
            return dataloader

        class ReplacementIterator:
            def __init__(self, iterator, replace_fn):
                self.iterator = iterator
                self.replace_fn = replace_fn

            def __iter__(self):
                return self

            def __next__(self):
                batch = next(self.iterator)
                return self.replace_fn(batch)

        class ReplacementDataLoader(type(dataloader)):
            def __init__(self, dataloader, replace_fn):
                self.__dict__ = dataloader.__dict__.copy()
                self.replace_fn = replace_fn

            def __iter__(self):
                original_iterator = super().__iter__()
                return ReplacementIterator(original_iterator, self.replace_fn)

            def __len__(self):
                return len(super())

        return ReplacementDataLoader(dataloader, self._replace_test_nodes)

    def prepare_optimizer(self):
        """优化器与调度器配置"""
        if self.optimizer_type == 0:
            self.optimizer = Adam
            self.optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 1e-6}
            p1 = int(0.75 * self.epochs)
            p2 = int(0.9 * self.epochs)
            self.scheduler = MultiStepLR
            self.scheduler_kwargs = {'milestones': [p1, p2], 'gamma': 0.1}
        elif self.optimizer_type == 1:
            steps_per_epoch = self.dm.train_len // self.dm.batch_size
            self.optimizer = AdamWScheduleFree
            self.optimizer_kwargs = {
                'lr': 5e-3, 'weight_decay': 0,
                'warmup_steps': int(steps_per_epoch * 0.75),
                'betas': (0.98, 0.999), 'eps': 1e-8
            }
            self.scheduler = None
            self.scheduler_kwargs = None
        elif self.optimizer_type == 2:
            self.optimizer = Adam
            self.optimizer_kwargs = {'lr': 1e-3, 'weight_decay': 1e-6}
            steps_per_epoch = self.dm.train_len // self.dm.batch_size
            self.scheduler = CosineAnnealingLR
            self.scheduler_kwargs = {'T_max': steps_per_epoch}
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def prepare_model(self):
        """模型初始化与配置"""
        torch.cuda.empty_cache()  # 清空CUDA缓存
        import gc
        gc.collect()
        cfg = dict(self.cfg)
        cfg['hist_patterns'] = self.hist_patterns

        # 模型实例化
        self.model = DiffusionImputer(
            model_kwargs=cfg,
            optim_class=self.optimizer,
            optim_kwargs=self.optimizer_kwargs,
            whiten_prob=None,
            scheduler_class=self.scheduler,
            scheduler_kwargs=self.scheduler_kwargs,
            metrics={
                'mae': torch_metrics.MaskedMAE(),
                'mse': torch_metrics.MaskedMSE(),
                'mre': torch_metrics.MaskedMRE()
            }
        ).to(self.device)
        self.model = self.model.to(self.device)
        print(f"Model device: {next(self.model.parameters()).device}")

        # 确保缓冲区在正确设备上
        for name, buf in self.model.named_buffers():
            if buf.device != self.device:
                print(f"Warning: Buffer {name} is on {buf.device}, moving to {self.device}")
                buf.data = buf.data.to(self.device)

        # 日志配置
        logger = TensorBoardLogger(
            save_dir='./logs',
            name=f'{self.dataset}_imputation_{time.strftime("%Y%m%d_%H%M%S")}'
        )

        # 辅助信息设备转移
        if hasattr(self.model, 'side_info'):
            model_device = next(self.model.parameters()).device
            self.model.side_info = self.model.side_info.to(model_device)

        # 训练回调
        self.callbacks = [
            ModelCheckpoint(
                monitor='val_loss',
                filename='best_model_epoch{epoch:02d}_valLoss{val_loss:.5f}',
                save_top_k=1,
                mode='min',
                verbose=True
            )
        ]

        # 训练器配置
        self.trainer = Trainer(
            max_epochs=self.epochs,
            default_root_dir='./logs',
            logger=logger,
            accelerator=self.accelerator,
            devices=[self.device.index] if self.device.type == 'cuda' else None,
            callbacks=self.callbacks,
            gradient_clip_val=1.0,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=True
        )

    def move_scaler_to_device(self, scaler, device):
        """将scaler移动到目标设备"""
        if hasattr(scaler, 'bias') and isinstance(scaler.bias, torch.Tensor):
            scaler.bias = scaler.bias.to(device, non_blocking=True)
        if hasattr(scaler, 'scale') and isinstance(scaler.scale, torch.Tensor):
            scaler.scale = scaler.scale.to(device, non_blocking=True)
        return scaler

    def _recursive_move_to_device(self, obj, device):
        """递归将数据移动到目标设备"""
        if isinstance(obj, Data):
            data = obj.clone()
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device, non_blocking=True)
            return data
        elif isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif hasattr(obj, '__dict__'):
            obj_copy = deepcopy(obj)
            for attr_name in obj_copy.__dict__:
                attr_value = getattr(obj_copy, attr_name)
                setattr(obj_copy, attr_name, self._recursive_move_to_device(attr_value, device))
            return obj_copy
        elif isinstance(obj, (list, tuple)):
            return [self._recursive_move_to_device(item, device) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._recursive_move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj

    def _get_sample_data(self, dataloader, is_train=True, sample_idx=0, max_timesteps=None):
        """获取样本数据用于可视化"""
        if not is_train and self.replace_test_node:
            dataloader = self._wrap_dataloader_for_replacement(dataloader)

        # 获取批次数据
        batch = next(iter(dataloader))
        if isinstance(batch, list) and len(batch) > 0:
            batch = batch[0]

        # 数据处理与设备转移
        batch = create_interpolation(batch)
        batch = self._recursive_move_to_device(batch, self.device)

        # 处理scaler
        if hasattr(batch, 'transform'):
            if 'x' in batch.transform:
                batch.transform['x'] = self.move_scaler_to_device(batch.transform['x'], self.device)
            if 'y' in batch.transform:
                batch.transform['y'] = self.move_scaler_to_device(batch.transform['y'], self.device)

        # 生成补全结果
        with torch.no_grad():
            if is_train:
                imputed = self.model.get_imputation(batch)
            else:
                imputed = self.model.generate_median_imputation(batch)

        # 转移到CPU用于可视化
        batch_cpu = self._recursive_move_to_device(batch, torch.device('cpu'))
        imputed_cpu = imputed.cpu().numpy()

        # 提取样本数据
        x_raw = np.squeeze(batch_cpu.input.x[sample_idx].numpy())
        mask = np.squeeze(batch_cpu.input.mask[sample_idx].numpy())
        y_true = np.squeeze(batch_cpu.target.y[sample_idx].numpy())
        imputed_data = np.squeeze(imputed_cpu)[sample_idx]

        # 截断时间步（如果需要）
        total_timesteps = y_true.shape[0]
        if max_timesteps is not None and max_timesteps < total_timesteps:
            start_idx = 0
            end_idx = start_idx + max_timesteps
            x_raw = x_raw[start_idx:end_idx]
            mask = mask[start_idx:end_idx]
            y_true = y_true[start_idx:end_idx]
            imputed_data = imputed_data[start_idx:end_idx]
            time_steps = np.arange(start_idx, end_idx)
        else:
            time_steps = np.arange(total_timesteps)

        return {
            'x_raw': x_raw, 'mask': mask, 'y_true': y_true,
            'imputed': imputed_data, 'time_steps': time_steps,
            'is_train': is_train
        }

    def _plot_visualization(self, sample_data, node_idx=0, save_path=None, tick_interval=None):
        """单节点时序可视化"""
        time_steps = sample_data['time_steps']
        x_raw = sample_data['x_raw'][:, node_idx] if sample_data['x_raw'].ndim > 1 else sample_data['x_raw']
        mask = sample_data['mask'][:, node_idx] if sample_data['mask'].ndim > 1 else sample_data['mask']
        y_true = sample_data['y_true'][:, node_idx] if sample_data['y_true'].ndim > 1 else sample_data['y_true']
        imputed = sample_data['imputed'][:, node_idx] if sample_data['imputed'].ndim > 1 else sample_data['imputed']

        # 分离数据类型
        existing_data = np.where(mask == 1, y_true, np.nan)
        missing_true = np.where(mask == 0, y_true, np.nan)
        missing_imputed = np.where(mask == 0, imputed, np.nan)

        # 计算MAE
        missing_mae = np.nanmean(np.abs(missing_true - missing_imputed))

        # 图像配置
        num_timesteps = len(time_steps)
        fig_width = min(8 + num_timesteps / 50, 16)
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        phase = "Training" if sample_data['is_train'] else "Testing"
        ax.set_title(
            f'{self.dataset} (Node {node_idx}) | Timesteps: {num_timesteps} | MAE: {missing_mae:.4f}',
            fontsize=9, pad=10
        )

        # 绘制曲线
        ax.plot(time_steps, existing_data, 'o-', color=self.line_colors[0], alpha=0.7,
                markersize=2, linewidth=1, label='Original', zorder=3)
        ax.plot(time_steps, missing_true, 'x--', color=self.line_colors[1], alpha=0.7,
                markersize=3, linewidth=0.8, label='True Missing', zorder=4)
        ax.plot(time_steps, missing_imputed, '^-', color=self.line_colors[2], alpha=0.7,
                markersize=2.5, linewidth=0.9, label='Imputed', zorder=5)

        # 标记缺失区域
        missing_indices = np.where(mask == 0)[0]
        for i in missing_indices:
            ax.axvspan(i - 0.5, i + 0.5, color='#f0f0f0', alpha=0.5, zorder=1)

        # 坐标轴配置
        ax.set_xlabel('Time Step', fontsize=7, labelpad=5)
        ax.set_ylabel('Value', fontsize=7, labelpad=5)

        # 时间轴刻度
        if tick_interval is None:
            if num_timesteps <= 50:
                tick_interval = 5
            elif num_timesteps <= 200:
                tick_interval = 20
            elif num_timesteps <= 500:
                tick_interval = 50
            else:
                tick_interval = 100

        ax.set_xticks(np.arange(min(time_steps), max(time_steps) + 1, tick_interval))
        plt.xticks(rotation=45, ha='right', fontsize=6)

        # 图例与网格
        ax.legend(loc='upper right', fontsize=6, frameon=False)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlim(min(time_steps) - 0.5, max(time_steps) + 0.5)

        # 保存图像
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout(pad=0.5)
            plt.savefig(
                save_path,
                dpi=200,
                bbox_inches='tight',
                format='png'
            )
            if save_path.exists() and save_path.stat().st_size > 1000:
                print(f"✅ Saved: {save_path}")
            else:
                print(f"⚠️ 可能保存失败: {save_path}")

        plt.close(fig)

    def _plot_horizontal_merged(self, sample_data, num_nodes=3, save_path=None, tick_interval=None):
        """多节点横向拼接长图（共享时间轴）"""
        time_steps = sample_data['time_steps']
        num_timesteps = len(time_steps)
        phase = "Training" if sample_data['is_train'] else "Testing"

        # 图像尺寸配置
        node_width = 6  # 每个节点子图宽度
        total_width = node_width * num_nodes  # 总宽度
        fig_height = 4  # 固定高度

        # 创建子图
        fig, axes = plt.subplots(1, num_nodes, figsize=(total_width, fig_height), sharey=False)
        if num_nodes == 1:
            axes = [axes]

        # 统一时间轴范围
        x_min, x_max = min(time_steps) - 0.5, max(time_steps) + 0.5

        # 时间轴刻度
        if tick_interval is None:
            if num_timesteps <= 50:
                tick_interval = 5
            elif num_timesteps <= 200:
                tick_interval = 20
            elif num_timesteps <= 500:
                tick_interval = 50
            else:
                tick_interval = 100
        xticks = np.arange(min(time_steps), max(time_steps) + 1, tick_interval)

        # 绘制每个节点
        for i, ax in enumerate(axes):
            node_idx = i  # 节点索引

            # 提取数据
            x_raw = sample_data['x_raw'][:, node_idx] if sample_data['x_raw'].ndim > 1 else sample_data['x_raw']
            mask = sample_data['mask'][:, node_idx] if sample_data['mask'].ndim > 1 else sample_data['mask']
            y_true = sample_data['y_true'][:, node_idx] if sample_data['y_true'].ndim > 1 else sample_data['y_true']
            imputed = sample_data['imputed'][:, node_idx] if sample_data['imputed'].ndim > 1 else sample_data['imputed']

            # 分离数据类型
            existing_data = np.where(mask == 1, y_true, np.nan)
            missing_true = np.where(mask == 0, y_true, np.nan)
            missing_imputed = np.where(mask == 0, imputed, np.nan)

            # 计算MAE
            missing_mae = np.nanmean(np.abs(missing_true - missing_imputed))

            # 绘制曲线
            ax.plot(time_steps, existing_data, 'o-', color=self.line_colors[0], alpha=0.7,
                    markersize=1.5, linewidth=0.8, label='Original' if i == 0 else "", zorder=3)
            ax.plot(time_steps, missing_true, 'x--', color=self.line_colors[1], alpha=0.7,
                    markersize=2, linewidth=0.6, label='True Missing' if i == 0 else "", zorder=4)
            ax.plot(time_steps, missing_imputed, '^-', color=self.line_colors[2], alpha=0.7,
                    markersize=2, linewidth=0.7, label='Imputed' if i == 0 else "", zorder=5)

            # 标记缺失区域
            missing_indices = np.where(mask == 0)[0]
            for idx in missing_indices:
                ax.axvspan(idx - 0.5, idx + 0.5, color='#f0f0f0', alpha=0.5, zorder=1)

            # 子图标题
            ax.set_title(f'Node {node_idx} (MAE: {missing_mae:.4f})',
                         fontsize=8, pad=5,
                         color=self.node_colors[node_idx % len(self.node_colors)])

            # 坐标轴配置
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(xticks)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)

            # 仅显示左侧y轴标签
            if i > 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel('Value', fontsize=7, labelpad=5)

            ax.grid(True, alpha=0.2, linewidth=0.3)

        # 全局标题和图例
        fig.suptitle(f'{self.dataset} {phase} - {num_nodes} Nodes (Shared Time Axis) | Timesteps: {num_timesteps}',
                     fontsize=10, y=1.02)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7,
                   bbox_to_anchor=(0.5, -0.05), bbox_transform=fig.transFigure)

        # 调整布局
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])

        # 保存图像
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                format='png'
            )
            if save_path.exists() and save_path.stat().st_size > 1000:
                print(f"✅ Saved horizontal merged plot: {save_path}")
            else:
                print(f"⚠️ 可能保存失败: {save_path}")

        plt.close(fig)

    def _plot_time_step_raw_values(self, sample_data, time_step=10, num_nodes=8, save_path=None):
        """新增：第10时刻原始值按节点顺序连线可视化"""
        # 检查时间步是否存在
        time_steps = sample_data['time_steps']
        if time_step not in time_steps:
            print(f"警告：时间步 {time_step} 不存在，最大时间步为 {max(time_steps)}")
            return

        # 获取时间步索引
        step_idx = np.where(time_steps == time_step)[0][0]
        phase = "Training" if sample_data['is_train'] else "Testing"

        # 提取第10时刻各节点的原始真实值（y_true）
        nodes = np.arange(num_nodes)
        raw_values = []  # 存储原始值
        for node_idx in nodes:
            # 提取当前节点在第10时刻的原始值（无论是否缺失）
            val = sample_data['y_true'][step_idx, node_idx] if sample_data['y_true'].ndim > 1 else \
            sample_data['y_true'][step_idx]
            raw_values.append(val)

        # 可视化配置
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制折线图（点+线连接）
        ax.plot(nodes, raw_values, 'o-', color=self.line_colors[0],
                markersize=6, linewidth=2, alpha=0.8,
                markerfacecolor='white', markeredgewidth=1.5)  # 白色填充的点

        # 标记每个点的数值（可选，密集时可注释）
        for i, val in enumerate(raw_values):
            ax.text(i, val + 0.02 * (max(raw_values) - min(raw_values)),
                    f'{val:.2f}', ha='center', fontsize=7)

        # 图表配置
        ax.set_title(f'{self.dataset} {phase} - Raw Values at Time Step {time_step} (by Node Order)',
                     fontsize=12, pad=10)
        ax.set_xlabel('Node Index', fontsize=10, labelpad=8)
        ax.set_ylabel('Raw Value', fontsize=10, labelpad=8)
        ax.set_xticks(nodes)
        ax.set_xticklabels([f'Node {i}' for i in nodes], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')

        # 调整y轴范围，避免点靠近边界
        y_min, y_max = min(raw_values), max(raw_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight',
                format='png'
            )
            if save_path.exists() and save_path.stat().st_size > 1000:
                print(f"✅ Saved time step {time_step} raw values (line plot): {save_path}")
            else:
                print(f"⚠️ 可能保存失败: {save_path}")

        plt.close(fig)

    def generate_visualizations(self, num_nodes=3, sample_idx=0, target_timesteps=300, plot_horizontal=True):
        """生成所有可视化结果（包含第10时刻原始值连线图）"""
        save_dir = Path(f'./visualizations/{self.dataset}_{time.strftime("%Y%m%d_%H%M%S")}')
        save_dir.mkdir(parents=True, exist_ok=True)

        # 训练集可视化
        print(f"\n生成训练集可视化（{target_timesteps} 时间步）...")
        train_data = self._get_sample_data(
            self.train_dataloader,
            is_train=True,
            sample_idx=sample_idx,
            max_timesteps=target_timesteps
        )
        # 单节点图像
        for node_idx in range(num_nodes):
            self._plot_visualization(
                train_data,
                node_idx=node_idx,
                save_path=save_dir / f'train_node_{node_idx}.png'
            )
        # 横向拼接长图
        if plot_horizontal:
            self._plot_horizontal_merged(
                train_data,
                num_nodes=num_nodes,
                save_path=save_dir / f'train_horizontal_merged_{num_nodes}_nodes.png'
            )
        # 第10时刻原始值连线图
        self._plot_time_step_raw_values(
            train_data,
            time_step=11,
            num_nodes=50,
            save_path=save_dir / f'train_time_step_10_raw_values_line.png'
        )

        # 测试集可视化
        print(f"生成测试集可视化（{target_timesteps} 时间步）...")
        test_data = self._get_sample_data(
            self.test_dataloader,
            is_train=False,
            sample_idx=sample_idx,
            max_timesteps=target_timesteps
        )
        # 单节点图像
        for node_idx in range(num_nodes):
            self._plot_visualization(
                test_data,
                node_idx=node_idx,
                save_path=save_dir / f'test_node_{node_idx}.png'
            )
        # 横向拼接长图
        if plot_horizontal:
            self._plot_horizontal_merged(
                test_data,
                num_nodes=num_nodes,
                save_path=save_dir / f'test_horizontal_merged_{num_nodes}_nodes.png'
            )
        # 第10时刻原始值连线图
        self._plot_time_step_raw_values(
            test_data,
            time_step=10,
            num_nodes=num_nodes,
            save_path=save_dir / f'test_time_step_10_raw_values_line.png'
        )

        print(f"所有可视化结果已保存至: {save_dir}")

    def run(self):
        """运行实验主流程"""
        try:
            # 数据准备
            self.prepare_data()
            if self.replace_test_node:
                self.test_dataloader = self._wrap_dataloader_for_replacement(self.test_dataloader)

            # 优化器与模型准备
            self.prepare_optimizer()
            self.prepare_model()
            self.model = self.model.to(self.device)
            self.generate_visualizations(num_nodes=8)
            # 训练模型
            train_start = time.time()
            self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
            train_duration = time.time() - train_start
            print(f"训练完成，耗时 {train_duration:.2f}s")
            self.model = self.model.to(self.device)

            # 评估与可视化
            print("\n开始模型评估...")
            self.model = self.model.to(self.device)
            self.generate_visualizations(num_nodes=8)  # 生成8节点可视化

            # 加载最佳模型
            checkpoint_callback = self.callbacks[0]
            if checkpoint_callback.best_model_path:
                print(f"加载最佳模型: {checkpoint_callback.best_model_path}")
                self.model.load_model(checkpoint_callback.best_model_path)

            # 测试集评估
            self.model.freeze()
            test_start = time.time()
            results = self.trainer.test(self.model, self.test_dataloader)
            test_duration = time.time() - test_start
            print(f"评估完成，耗时 {test_duration:.2f}s")

            # 补充可视化（最佳模型结果）
            self.model = self.model.to(self.device)
            self.generate_visualizations(num_nodes=8)

            # 整理结果
            results[0]['training_time'] = train_duration
            results[0]['testing_time'] = test_duration
            return results[0]

        except Exception as e:
            print(f"实验失败: {str(e)}")
            raise


class AverageExperiment:
    """多轮实验结果平均类"""

    def __init__(self, dataset, cfg, optimizer_type, seed, epochs, accelerator='gpu', device=None, n=5):
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer_type = optimizer_type
        self.seed = seed
        self.epochs = epochs
        self.accelerator = accelerator
        self.device = device
        self.n = n  # 实验重复次数
        self.folder = Path('./metrics/')

        # 实验参数
        self.kwargs_experiment = {
            'dataset': self.dataset,
            'cfg': self.cfg,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'accelerator': self.accelerator,
            'device': self.device,
            'seed': seed,
        }

        print("实验参数:", self.kwargs_experiment)
        self.init_result_folder()

    def init_result_folder(self):
        """初始化结果存储目录"""
        self.folder.mkdir(parents=True, exist_ok=True)
        if not (self.folder / 'results_by_experiment.csv').exists():
            results = pd.DataFrame(columns=[
                'mae', 'mse', 'mre',
                'training_time', 'testing_time'
            ])
            results.to_csv(self.folder / 'results_by_experiment.csv')

    def save_results(self, results, i):
        """保存单轮实验结果"""
        results_df = pd.read_csv(self.folder / 'results_by_experiment.csv', index_col='Unnamed: 0')
        results_df.loc[i] = [
            results['test_mae'],
            results['test_mse'],
            results['test_mre'],
            results['training_time'],
            results['testing_time'],
        ]
        results_df.to_csv(self.folder / 'results_by_experiment.csv')

    def average_results(self):
        """计算多轮实验平均结果"""
        average_results = pd.DataFrame(columns=[
            'mae_mean', 'mae_std',
            'mse_mean', 'mse_std',
            'mre_mean', 'mre_std',
            'training_time_mean', 'training_time_std',
            'testing_time_mean', 'testing_time_std',
        ])

        results_by_experiment = pd.read_csv(self.folder / 'results_by_experiment.csv', index_col='Unnamed: 0')

        average_results.loc[0] = [
            results_by_experiment['mae'].mean(),
            results_by_experiment['mae'].std(),
            results_by_experiment['mse'].mean(),
            results_by_experiment['mse'].std(),
            results_by_experiment['mre'].mean(),
            results_by_experiment['mre'].std(),
            results_by_experiment['training_time'].mean(),
            results_by_experiment['training_time'].std(),
            results_by_experiment['testing_time'].mean(),
            results_by_experiment['testing_time'].std(),
        ]

        average_results.to_csv(self.folder / 'results.csv')

    def run(self):
        """运行多轮实验并计算平均值"""
        n_done = pd.read_csv(self.folder / 'results_by_experiment.csv').shape[0]
        for i in range(n_done, self.n):
            self.kwargs_experiment['seed'] = self.seed + i  # 不同种子保证独立性
            experiment = Experiment(**self.kwargs_experiment)
            results = experiment.run()
            self.save_results(results, i)

        self.average_results()
        print(f"多轮实验结果已保存至: {self.folder}")