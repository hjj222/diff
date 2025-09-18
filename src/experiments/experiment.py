import os
import time
import random
import numpy as np
import pandas as pd
import numpy as np
from omegaconf import open_dict

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
from src.data.data_handlers import create_interpolation  # 数据插值工具函数
import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import open_dict
import copy


class Experiment:
    def __init__(self, dataset, cfg, optimizer_type, epochs, accelerator='gpu', device=None, seed=42):
        # 初始化核心参数
        self.cfg = cfg
        self.dataset = dataset
        self.optimizer_type = optimizer_type
        self.epochs = 50
        self.accelerator = accelerator

        # 设备配置
        self.device = torch.device(f"cuda:{device}") if device is not None else \
            (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.seed = seed

        # 固定随机种子
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 【优化1：更基础的字体配置】确保所有环境兼容
        plt.rcParams["font.family"] = ["monospace", "sans-serif"]  # 最基础的字体族
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Courier New", "monospace"]  # 兼容性优先
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 80  # 降低默认分辨率
        plt.rcParams["savefig.dpi"] = 200  # 保存图像分辨率适中即可
        plt.rcParams["axes.titlesize"] = 10  # 缩小标题字体
        plt.rcParams["axes.labelsize"] = 8  # 缩小坐标轴标签
        plt.rcParams["legend.fontsize"] = 7  # 缩小图例字体

    def prepare_data(self):
        # 保持原有代码不变
        dm_params = {
            'batch_size': self.cfg.config.batch_size,
            'scale_window_factor': self.cfg.config.scale_window_factor
        }

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

        self.dm = data_class(**dm_params).get_dm()
        self.dm_stride = data_class(stride='window_size', **dm_params).get_dm()

        if self.cfg.missing_pattern.strategy1 == 'historical' or self.cfg.missing_pattern.strategy2 == 'historical':
            self.hist_patterns = data_class(test_months=(2, 5, 8, 11), **dm_params).get_historical_patterns()
        else:
            self.hist_patterns = None

        self.dm.setup()
        self.dm_stride.setup()
        print(f"Dataset loaded | Train: {self.dm.train_len} | Val: {self.dm.val_len} | Test: {self.dm.test_len}")

        with open_dict(self.cfg):
            self.cfg.config.time_steps = self.dm.window
            self.cfg.config.num_nodes = self.dm.n_nodes

        self.train_dataloader = self.dm.train_dataloader()
        self.val_dataloader = self.dm_stride.val_dataloader()
        self.test_dataloader = self.dm_stride.test_dataloader()

    def prepare_optimizer(self):
        # 保持原有代码不变
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
        # 保持原有代码不变
        cfg = dict(self.cfg)
        cfg['hist_patterns'] = self.hist_patterns

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
        # 【关键修复1：强制模型转移到目标设备】
        self.model = self.model.to(self.device)
        # 验证：打印模型设备，确保在cuda:0上（可选，用于调试）
        print(f"Model device: {next(self.model.parameters()).device}")  # 应输出cuda:0
        # 遍历所有buffer，验证设备（可选，用于调试）
        for name, buf in self.model.named_buffers():
            if buf.device != self.device:
                print(f"Warning: Buffer {name} is on {buf.device}, moving to {self.device}")
                buf.data = buf.data.to(self.device)  # 手动强制转移（防止注册失败）
            logger = TensorBoardLogger(
                save_dir='./logs',
                name=f'{self.dataset}_imputation_{time.strftime("%Y%m%d_%H%M%S")}'
            )
        if hasattr(self.model, 'side_info'):
            self.model.side_info = self.model.side_info.to(model_device)
        self.callbacks = [
            ModelCheckpoint(
                monitor='val_loss',
                filename='best_model_epoch{epoch:02d}_valLoss{val_loss:.5f}',
                save_top_k=1,
                mode='min',
                verbose=True
            )
        ]

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
        # 保持原有代码不变
        if hasattr(scaler, 'bias') and isinstance(scaler.bias, torch.Tensor):
            scaler.bias = scaler.bias.to(device, non_blocking=True)
        if hasattr(scaler, 'scale') and isinstance(scaler.scale, torch.Tensor):
            scaler.scale = scaler.scale.to(device, non_blocking=True)
        return scaler

    def _recursive_move_to_device(self, obj, device):
        # 保持原有代码不变
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
        """
        提取样本数据，支持指定最大时间步数量
        max_timesteps: 最大时间步数量，None则使用完整序列
        """
        # 获取批次数据
        batch = next(iter(dataloader))
        batch = create_interpolation(batch)
        batch = self._recursive_move_to_device(batch, self.device)

        # 同步scaler到设备
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

        # 移动到CPU处理
        batch_cpu = self._recursive_move_to_device(batch, torch.device('cpu'))
        imputed_cpu = imputed.cpu().numpy()

        # 提取样本数据
        x_raw = np.squeeze(batch_cpu.input.x[sample_idx].numpy())
        mask = np.squeeze(batch_cpu.input.mask[sample_idx].numpy())
        y_true = np.squeeze(batch_cpu.target.y[sample_idx].numpy())
        imputed_data = np.squeeze(imputed_cpu)[sample_idx]
        print(y_true.shape)
        # 【关键修改1：限制或扩展时间步数量】
        total_timesteps = y_true.shape[0]
        print(total_timesteps)
        if max_timesteps is not None and max_timesteps < total_timesteps:
            # 如果指定了最大时间步且小于总长度，则取一段连续的长序列
            start_idx = 0  # 从起始位置开始
            end_idx = start_idx + max_timesteps
            x_raw = x_raw[start_idx:end_idx]
            mask = mask[start_idx:end_idx]
            y_true = y_true[start_idx:end_idx]
            imputed_data = imputed_data[start_idx:end_idx]
            time_steps = np.arange(start_idx, end_idx)
        else:
            # 使用完整序列
            time_steps = np.arange(total_timesteps)

        return {
            'x_raw': x_raw, 'mask': mask, 'y_true': y_true,
            'imputed': imputed_data, 'time_steps': time_steps,
            'is_train': is_train
        }

    def _plot_visualization(self, sample_data, node_idx=0, save_path=None, tick_interval=None):
        """
        可视化函数，优化时间轴显示以适应更多时间步
        tick_interval: 时间轴刻度间隔，None则自动计算
        """
        # 提取节点数据
        time_steps = sample_data['time_steps']
        x_raw = sample_data['x_raw'][:, node_idx] if sample_data['x_raw'].ndim > 1 else sample_data['x_raw']
        mask = sample_data['mask'][:, node_idx] if sample_data['mask'].ndim > 1 else sample_data['mask']
        y_true = sample_data['y_true'][:, node_idx] if sample_data['y_true'].ndim > 1 else sample_data['y_true']
        imputed = sample_data['imputed'][:, node_idx] if sample_data['imputed'].ndim > 1 else sample_data['imputed']

        # 分离数据类型
        existing_data = np.where(mask == 1, y_true, np.nan)
        missing_true = np.where(mask == 0, y_true, np.nan)
        missing_imputed = np.where(mask == 0, imputed, np.nan)

        # 计算缺失区域MAE
        missing_mae = np.nanmean(np.abs(missing_true - missing_imputed))

        # 【关键修改2：根据时间步数量动态调整图像宽度】
        num_timesteps = len(time_steps)
        # 时间步越多，图像宽度越大（但限制最大宽度为16）
        fig_width = min(8 + num_timesteps / 50, 16)  # 每50个时间步增加1单位宽度
        fig, ax = plt.subplots(figsize=(fig_width, 4))

        phase = "Training" if sample_data['is_train'] else "Testing"
        ax.set_title(
            f'{self.dataset} (Node {node_idx}) | Timesteps: {num_timesteps} | MAE: {missing_mae:.4f}',
            fontsize=9, pad=10
        )

        # 绘制数据曲线
        ax.plot(time_steps, existing_data, 'o-', color='#1f77b4', alpha=0.7,
                markersize=2, linewidth=1, label='Original', zorder=3)
        ax.plot(time_steps, missing_true, 'x--', color='#ff7f0e', alpha=0.7,
                markersize=3, linewidth=0.8, label='True Missing', zorder=4)
        ax.plot(time_steps, missing_imputed, '^-', color='#2ca02c', alpha=0.7,
                markersize=2.5, linewidth=0.9, label='Imputed', zorder=5)

        # 标记缺失区域
        missing_indices = np.where(mask == 0)[0]
        for i in missing_indices:
            ax.axvspan(i - 0.5, i + 0.5, color='#f0f0f0', alpha=0.5, zorder=1)

        # 【关键修改3：优化时间轴刻度显示】
        ax.set_xlabel('Time Step', fontsize=7, labelpad=5)
        ax.set_ylabel('Value', fontsize=7, labelpad=5)

        # 自动计算合适的刻度间隔，避免刻度拥挤
        if tick_interval is None:
            # 根据时间步数量动态调整间隔
            if num_timesteps <= 50:
                tick_interval = 5
            elif num_timesteps <= 200:
                tick_interval = 20
            elif num_timesteps <= 500:
                tick_interval = 50
            else:
                tick_interval = 100

        # 设置x轴刻度
        ax.set_xticks(np.arange(min(time_steps), max(time_steps) + 1, tick_interval))
        # 旋转刻度标签，避免重叠
        plt.xticks(rotation=45, ha='right', fontsize=6)

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

    def generate_visualizations(self, num_nodes=3, sample_idx=0, target_timesteps=300):
        """
        生成可视化时指定目标时间步数量
        target_timesteps: 希望展示的时间步数量
        """
        save_dir = Path(f'./visualizations/{self.dataset}_{time.strftime("%Y%m%d_%H%M%S")}')
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating training visualizations with {target_timesteps} timesteps...")
        # 【关键修改4：生成时指定目标时间步】
        train_data = self._get_sample_data(
            self.train_dataloader,
            is_train=True,
            sample_idx=sample_idx,
            max_timesteps=target_timesteps  # 传递目标时间步参数
        )
        for node_idx in range(num_nodes):
            self._plot_visualization(
                train_data,
                node_idx=node_idx,
                save_path=save_dir / f'train_node_{node_idx}.png'
            )

        print(f"Generating testing visualizations with {target_timesteps} timesteps...")
        test_data = self._get_sample_data(
            self.test_dataloader,
            is_train=False,
            sample_idx=sample_idx,
            max_timesteps=target_timesteps
        )
        for node_idx in range(num_nodes):
            self._plot_visualization(
                test_data,
                node_idx=node_idx,
                save_path=save_dir / f'test_node_{node_idx}.png'
            )

        print(f"Visualizations directory: {save_dir}")

    def run(self):
        # 保持原有代码不变
        try:
            self.prepare_data()
            self.prepare_optimizer()
            self.prepare_model()
            self.generate_visualizations(num_nodes=3)
            print("\nStarting training...")
            train_start = time.time()
            self.trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
            train_duration = time.time() - train_start
            print(f"Training done in {train_duration:.2f}s")
            self.prepare_model()
            self.generate_visualizations(num_nodes=3)
            print("\nStarting evaluation...")
            checkpoint_callback = self.callbacks[0]
            if checkpoint_callback.best_model_path:
                print(f"Loading best model: {checkpoint_callback.best_model_path}")
                self.model.load_model(checkpoint_callback.best_model_path)

            self.model.freeze()
            test_start = time.time()
            results = self.trainer.test(self.model, self.test_dataloader)
            test_duration = time.time() - test_start
            print(f"Evaluation done in {test_duration:.2f}s")

            # self.generate_visualizations(num_nodes=3)

            results[0]['training_time'] = train_duration
            results[0]['testing_time'] = test_duration
            return results[0]

        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            raise


class AverageExperiment:
    def __init__(self, dataset, cfg, optimizer_type, seed, epochs, accelerator='gpu', device=None, n=5):
        self.dataset = dataset
        self.cfg = cfg
        self.optimizer_type = optimizer_type
        self.seed = seed
        self.epochs = epochs
        self.accelerator = accelerator
        self.device = device
        self.n = n
        self.folder = f'./metrics/'

        self.kwargs_experiment = {
            'dataset': self.dataset,
            'cfg': self.cfg,
            'optimizer_type': self.optimizer_type,
            'epochs': self.epochs,
            'accelerator': self.accelerator,
            'device': self.device,
            'seed': seed,
        }

        print(self.kwargs_experiment)
        self.init_result_folder()

    def init_result_folder(self):
        os.makedirs(self.folder, exist_ok=True)
        if len(os.listdir(self.folder)) == 0:
            results = pd.DataFrame(columns=[
                'mae',
                'mse',
                'mre',
                'training_time',
                'testing_time',
            ])
            results.to_csv(f'{self.folder}/results_by_experiment.csv')

    def save_results(self, results, i):
        results_df = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')
        results_df.loc[i] = [
            results['test_mae'],
            results['test_mse'],
            results['test_mre'],
            results['training_time'],
            results['testing_time'],
        ]
        results_df.to_csv(f'{self.folder}/results_by_experiment.csv')

    def average_results(self):
        average_results = pd.DataFrame(columns=[
            'mae_mean',
            'mae_std',
            'mse_mean',
            'mse_std',
            'mre_mean',
            'mre_std',
            'training_time_mean',
            'training_time_std',
            'testing_time_mean',
            'testing_time_std',
        ])

        results_by_experiment = pd.read_csv(f'{self.folder}/results_by_experiment.csv', index_col='Unnamed: 0')

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

        average_results.to_csv(f'{self.folder}/results.csv')

    def run(self):
        n_done = pd.read_csv(f'{self.folder}/results_by_experiment.csv').shape[0]
        for i in range(n_done, self.n):
            self.kwargs_experiment['seed'] = self.seed + i
            experiment = Experiment(**self.kwargs_experiment)
            results = experiment.run()
            self.save_results(results, i)

        self.average_results()

