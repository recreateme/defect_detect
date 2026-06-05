#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像缺陷分类模型训练管线
================================================================================

【架构概览】
  · 模型  : EfficientNet-B0（torchvision ImageNet 迁移学习，小样本场景首选）
  · 策略  : 两阶段微调
            阶段一 — 冻结 Backbone，仅训练 Dropout + Linear 分类头（快速对齐类别）
            阶段二 — 解冻全部层，Backbone 与 Head 分层学习率端到端微调
  · 增强  : 默认适配「离线已增强」数据（轻量在线增强）；原始图用 --no_pre_augmented
  · 不均衡: 四层策略（详见下方「四层不均衡学习」）
  · 导出  : best_model.pt（GPU/PyTorch）+ model.onnx（CPU/ONNXRuntime）+ 阈值 JSON

【四层不均衡学习】（类别样本数差异大时必须关注 Macro-F1，而非 Accuracy）
  第一层 · 采样   : WeightedRandomSampler（原始数据）或 shuffle（离线增强数据）
  第二层 · 损失   : CrossEntropy / FocalLoss 中传入逆频类别权重
  第三层 · Focal  : (1-p_t)^γ 聚焦难分样本，默认 γ=2
  第四层 · 评估   : 以 val_macro_f1 保存最优模型；验证集逐类搜索 class_thresholds.json

【路径约定】
  PROJECT_ROOT = 本文件所在目录。所有相对路径均相对项目根解析，
  可从任意工作目录启动，例如:
    python "D:/.../defects_classify/train.py" --data_dir data --img_size 128

【常用命令】
  从头训练:
    python train.py --data_dir data --img_size 128
  合并 corrections 增量微调:
    python train.py --finetune --extra_data_dirs corrections
  仅补跑阈值校准与 ONNX（不重训）:
    python train.py --postprocess_only
  指定 checkpoint 继续训练:
    python train.py --resume checkpoints/best_model.pt --epochs_phase1 0
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

# ── 项目根目录 ──────────────────────────────────────────────────────────────
# 无论用户在哪个 shell 目录执行 python，data/、checkpoints/ 等相对路径
# 都解析到此目录下，避免出现「No such file or directory」。
PROJECT_ROOT = Path(__file__).resolve().parent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image

# 可选依赖（缺失时跳过对应功能）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _setup_cjk_font():
        """自动选择支持中文的字体（Windows / macOS / Linux）。"""
        candidates = [
            "Microsoft YaHei", "SimHei", "SimSun",      # Windows
            "PingFang SC", "Heiti SC",                   # macOS
            "WenQuanYi Micro Hei", "Noto Sans CJK SC",  # Linux
        ]
        import matplotlib.font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        for font in candidates:
            if font in available:
                matplotlib.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return font
        matplotlib.rcParams["axes.unicode_minus"] = False
        return None

    _setup_cjk_font()
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════
# 路径解析（相对项目根，避免 No such file or directory）
# ═══════════════════════════════════════════════
def resolve_project_path(
    path: "str | Path",
    *,
    must_exist: bool = False,
    kind: str = "路径",
    create_parent: bool = False,
) -> Path:
    """
    将相对路径解析为基于 PROJECT_ROOT 的绝对路径。
    用户从任意目录执行 python .../train.py 时，--data_dir data 仍指向项目内 data/。
    """
    raw = str(path)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    try:
        p = p.resolve()
    except OSError as e:
        raise FileNotFoundError(
            f"无法解析{kind}: {raw}\n"
            f"  项目根目录: {PROJECT_ROOT}\n"
            f"  错误: {e}"
        ) from e

    if must_exist and not p.exists():
        hint = (
            f"\n  项目根目录: {PROJECT_ROOT}\n"
            f"  解析后路径: {p}\n"
            f"  启动前工作目录: {os.getcwd()}\n"
            f"提示: 使用相对项目根的路径，例如 --data_dir data；\n"
            f"  或: python \"{PROJECT_ROOT / 'train.py'}\" --data_dir data --img_size 128"
        )
        if kind == "数据目录":
            data_candidate = PROJECT_ROOT / "data"
            if data_candidate.is_dir():
                hint += f"\n  检测到 {data_candidate} 存在，请确认 --data_dir 参数。"
            else:
                hint += f"\n  未找到默认目录 {data_candidate}，请先准备数据集。"
        raise FileNotFoundError(f"{kind}不存在: {raw}{hint}")

    if create_parent and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_training_paths(args: argparse.Namespace) -> argparse.Namespace:
    """解析并校验所有路径参数，并将工作目录切换到项目根。"""
    try:
        os.chdir(PROJECT_ROOT)
    except OSError:
        pass

    args.data_dir = resolve_project_path(
        args.data_dir, must_exist=True, kind="数据目录"
    )
    args.save_dir = resolve_project_path(
        args.save_dir, create_parent=True, kind="输出目录"
    )

    resolved_extra: List[Path] = []
    for d in args.extra_data_dirs or []:
        try:
            resolved_extra.append(
                resolve_project_path(d, must_exist=True, kind="额外数据目录")
            )
        except FileNotFoundError as exc:
            print(f"  [跳过] {exc}")
    args.extra_data_dirs = resolved_extra

    resume_raw = (args.resume or "").strip()
    if resume_raw.lower() == "auto":
        cand = args.save_dir / "best_model.pt"
        args.resume_path = cand if cand.is_file() else None
    elif resume_raw:
        args.resume_path = resolve_project_path(
            resume_raw, must_exist=True, kind="Checkpoint"
        )
    else:
        args.resume_path = None

    if args.finetune and args.resume_path is None:
        auto_ckpt = args.save_dir / "best_model.pt"
        if auto_ckpt.is_file():
            args.resume_path = auto_ckpt
            print(f"  [--finetune] 自动加载 {auto_ckpt}")

    if args.finetune and args.epochs_phase1 > 0:
        print("  [--finetune] 跳过阶段一（仅端到端微调）")
        args.epochs_phase1 = 0

    if args.postprocess_only and not args.resume_path:
        auto_ckpt = args.save_dir / "best_model.pt"
        if auto_ckpt.is_file():
            args.resume_path = auto_ckpt

    return args


# ═══════════════════════════════════════════════
# 不均衡学习 · 第三层损失 + 第四层评估指标
# ═══════════════════════════════════════════════
# 与第一层（Sampler）、第二层（CE weight）配合使用，形成完整不均衡方案。
class FocalLoss(nn.Module):
    """
    Focal Loss — 自动聚焦难分样本，缓解类别不均衡。
    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    γ=0 退化为普通 CrossEntropy；γ=2 为经典设置。
    weight: 与 CrossEntropyLoss.weight 语义相同（类别逆频权重，第二层）。
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # log_p / p : 各类别 log-softmax 与概率
        log_p = F.log_softmax(logits, dim=1)
        p     = torch.exp(log_p)
        # p_t : 仅取「真实类别」对应的预测概率（难样本 p_t 小 → focal 权重大）
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal  = (1.0 - pt) ** self.gamma
        loss   = -focal * log_pt
        # 第二层类别权重：少数类算错时梯度进一步放大
        if self.weight is not None:
            w = self.weight.to(targets.device)
            loss = loss * w[targets]
        return loss.mean()

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, weighted={self.weight is not None}"


def compute_macro_f1(all_preds: np.ndarray, all_labels: np.ndarray) -> float:
    """
    Macro-F1：对每类 F1 取简单平均（第四层评估指标）。
    少数类 F1 偏低会直接拉低整体，比 Accuracy 更能反映不均衡数据表现。
    """
    classes_seen = np.unique(all_labels)
    f1_list = []
    for c in classes_seen:
        tp  = np.sum((all_preds == c) & (all_labels == c))
        fp  = np.sum((all_preds == c) & (all_labels != c))
        fn  = np.sum((all_preds != c) & (all_labels == c))
        pre = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1  = 2 * pre * rec / (pre + rec + 1e-9)
        f1_list.append(f1)
    return float(np.mean(f1_list)) if f1_list else 0.0


# ═══════════════════════════════════════════════
# 混合精度（兼容 PyTorch 2.x 新 API）
# ═══════════════════════════════════════════════
def create_grad_scaler(enabled: bool):
    """创建 GradScaler；仅 CUDA 混合精度训练时启用。"""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        from torch.cuda.amp import GradScaler as _GradScaler
        return _GradScaler()


def amp_autocast(enabled: bool):
    """返回 autocast 上下文；禁用时为空上下文。"""
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        from torch.cuda.amp import autocast as _autocast
        return _autocast()


# ═══════════════════════════════════════════════
# 命令行参数（分组说明见 README「训练参数建议」）
# ═══════════════════════════════════════════════
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="缺陷分类模型训练")
    # 数据
    p.add_argument("--data_dir",       type=str,   default="data")
    p.add_argument("--img_size",       type=int,   default=128)
    p.add_argument("--val_ratio",      type=float, default=0.15,
                   help="验证集比例（0~1）")
    # 训练
    p.add_argument("--batch_size",     type=int,   default=32,
                   help="批大小（128×128 下可适当增大以稳定梯度）")
    p.add_argument("--epochs_phase1",  type=int,   default=5,
                   help="阶段一：冻结 Backbone，只训练分类头")
    p.add_argument("--epochs_phase2",  type=int,   default=20,
                   help="阶段二：解冻全部层，端到端微调")
    p.add_argument("--lr_phase1",      type=float, default=1e-3)
    p.add_argument("--lr_phase2",      type=float, default=1e-4,
                   help="阶段二 Backbone 学习率（小数据集宜偏低）")
    p.add_argument("--label_smooth",   type=float, default=0.1,
                   help="Label Smoothing，小样本防过拟合")
    p.add_argument("--mixup_alpha",    type=float, default=0,
                   help="MixUp alpha，0 表示关闭（缺陷分类默认关闭）")
    p.add_argument("--patience",       type=int,   default=8,
                   help="Early Stopping 耐心轮数")
    p.add_argument("--pre_augmented",  action="store_true", default=True,
                   help="数据已离线增强：轻量在线增强 + shuffle（默认开启）")
    p.add_argument("--no_pre_augmented", dest="pre_augmented", action="store_false",
                   help="数据未增强：完整在线增强 + WeightedRandomSampler（第一层）")
    # 不均衡学习（第二～四层）
    p.add_argument("--use_class_weight", action="store_true", default=True,
                   help="损失函数中加入逆频类别权重（第二层，默认开启）")
    p.add_argument("--no_class_weight", dest="use_class_weight", action="store_false",
                   help="关闭损失函数类别加权")
    p.add_argument("--use_focal_loss", action="store_true", default=True,
                   help="使用 Focal Loss 替换 CrossEntropy（第三层，默认开启）")
    p.add_argument("--no_focal_loss", dest="use_focal_loss", action="store_false",
                   help="关闭 Focal Loss，改用加权 CrossEntropy")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal Loss γ 参数（越大越聚焦难样本，默认 2.0）")
    p.add_argument("--extra_data_dirs", nargs="*", default=[],
                   help="额外训练数据目录列表（如 corrections/），追加到主数据集")
    p.add_argument("--amp",            action="store_true", default=True,
                   help="启用混合精度训练（需 GPU）")
    # 输出
    p.add_argument("--save_dir",       type=str,   default="checkpoints")
    p.add_argument("--num_workers",    type=int,
                   default=0 if sys.platform == "win32" else 4,
                   help="DataLoader 工作进程数（Windows 建议 0）")
    p.add_argument("--seed",           type=int,   default=42)
    # 增量 / 继续训练
    p.add_argument("--resume", type=str, default="",
                   help="加载已有权重：checkpoint 路径，或 auto（save_dir/best_model.pt）")
    p.add_argument("--finetune", action="store_true",
                   help="增量微调：自动 resume + 跳过阶段一，在已有模型上继续训练")
    p.add_argument("--reuse_split", action="store_true", default=True,
                   help="复用 save_dir 内已保存的数据划分（默认开启）")
    p.add_argument("--no_reuse_split", dest="reuse_split", action="store_false",
                   help="忽略已保存划分，按 seed 重新划分")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="从头训练时使用 ImageNet 预训练 Backbone（默认开启）")
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                   help="随机初始化 Backbone（仅建议在无 resume 时尝试）")
    p.add_argument("--postprocess_only", action="store_true",
                   help="跳过训练，仅加载 best_model.pt 做阈值校准与 ONNX 导出")
    return p.parse_args()


# ═══════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DefectDataset(torch.utils.data.Dataset):
    """
    缺陷图像 Dataset：按「子文件夹名 = 类别名」组织的多源数据加载器。

    支持 data/ 与 corrections/ 等多个根目录合并；
    类别取各目录子文件夹名的并集，按字典序固定 idx（保证训练/推理 class_map 一致）。

    目录结构示例:
        data/
            局部破损/  img001.jpg ...
            断钻/      img002.jpg ...
    """

    def __init__(self, data_dirs, transform=None):
        """
        Args:
            data_dirs: 单个 Path/str 或列表；路径应为 normalize_training_paths 解析后的绝对路径。
            transform: torchvision Compose；训练/验证集各用不同 transform 的 Subset。
        """
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        # 统一转为绝对 Path 列表
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [Path(data_dirs)]
        else:
            data_dirs = [Path(d) for d in data_dirs]

        missing = [d for d in data_dirs if not d.is_dir()]
        if missing:
            raise FileNotFoundError(
                "以下数据目录不存在或不是文件夹:\n"
                + "\n".join(f"  · {d}" for d in missing)
                + f"\n  项目根目录: {PROJECT_ROOT}"
            )

        # 收集所有目录下的类别（取并集，字典序排序保证一致性）
        class_set: set = set()
        for d in data_dirs:
            class_set.update(
                sub.name for sub in d.iterdir() if sub.is_dir()
            )
        if not class_set:
            raise RuntimeError(
                f"在 {[str(d) for d in data_dirs]} 下未找到类别子文件夹。\n"
                f"期望结构: data/类别名/*.jpg"
            )

        self.classes      = sorted(class_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 遍历所有目录收集图像路径
        for d in data_dirs:
            for cls_name in self.classes:
                cls_dir = d / cls_name
                if cls_dir.exists():
                    for f in cls_dir.rglob("*"):
                        if f.suffix.lower() in IMG_EXTS:
                            self.samples.append(
                                (f, self.class_to_idx[cls_name])
                            )

        if not self.samples:
            raise RuntimeError(
                f"在 {[str(d) for d in data_dirs]} 下未找到任何图像文件。"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        逆频率类别权重（第二层损失用）。
        归一化使权重之和 = 类别数，避免整体 loss 尺度漂移。
        """
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts
        return weights / weights.sum() * len(self.classes)


def get_transforms(img_size: int, pre_augmented: bool = True) -> Tuple[T.Compose, T.Compose]:
    """
    返回训练增强 / 验证预处理的 transforms。

    pre_augmented=True（默认）:
      数据已在 data/ 中离线增强，在线只做等比缩放与极轻扰动，
      避免二次强增强导致缺陷特征失真（尤其 128×128 小图）。
    pre_augmented=False:
      原始数据训练，启用完整在线增强流水线。
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    if pre_augmented:
        # 离线已增强：直接缩放到目标分辨率，仅保留轻微翻转
        train_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        # 原始数据：适度在线增强（128 分辨率下裁剪余量不宜过大）
        margin = max(8, img_size // 8)
        train_tf = T.Compose([
            T.Resize((img_size + margin, img_size + margin)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            T.ToTensor(),
            T.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.0), value=0),
            T.Normalize(mean=mean, std=std),
        ])

    return train_tf, val_tf


def _split_cache_path(save_dir: Path, seed: int) -> Path:
    return save_dir / f"dataset_split_seed{seed}.json"


def _load_or_create_split(
    labels: List[int],
    classes: List[str],
    val_ratio: float,
    seed: int,
    save_dir: Optional[Path],
    reuse_split: bool,
) -> Tuple[List[int], List[int]]:
    """
    分层划分 train/val，并可选缓存到 save_dir/dataset_split_seed{seed}.json。

    增量训练时复用同一划分，保证验证集 F1 与历史 checkpoint 可比；
    若样本数或类别列表变化，自动重新划分。
    """
    n_total = len(labels)
    cache_path = _split_cache_path(save_dir, seed) if save_dir else None

    if cache_path and reuse_split and cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if (
            cached.get("n_samples") == n_total
            and cached.get("classes") == classes
        ):
            print(f"  复用数据划分 → {cache_path.name}")
            return cached["train_indices"], cached["val_indices"]
        print("  数据集或类别已变化，重新划分 train/val")

    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for cls_idx in range(len(classes)):
        cls_indices = np.where(np.array(labels) == cls_idx)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": seed,
                    "n_samples": n_total,
                    "classes": classes,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  数据划分已缓存 → {cache_path.name}")

    return train_indices, val_indices


def build_dataloaders(
    data_dir: Path,
    img_size: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    extra_data_dirs: List[Path] = None,
    pre_augmented: bool = True,
    save_dir: Optional[Path] = None,
    reuse_split: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """
    构建 train/val DataLoader（分层划分）。

    第一层 · WeightedRandomSampler（--no_pre_augmented 时启用）:
      训练时每个 batch 被强制采样为近似均衡分布，等效于将少数类过采样到与多数类相同频率。
    离线已增强数据（默认）使用 shuffle，避免对同一增强图重复过采样；由第二～四层补偿不均衡。
    """

    train_tf, val_tf = get_transforms(img_size, pre_augmented=pre_augmented)

    all_dirs = [data_dir]
    if extra_data_dirs:
        all_dirs += list(extra_data_dirs)

    full_dataset = DefectDataset(all_dirs, transform=None)
    classes      = full_dataset.classes
    labels       = [lbl for _, lbl in full_dataset.samples]

    train_indices, val_indices = _load_or_create_split(
        labels, classes, val_ratio, seed, save_dir, reuse_split
    )

    # 两份 Subset 各自用不同 transform
    train_set = Subset(DefectDataset(all_dirs, transform=train_tf), train_indices)
    val_set   = Subset(DefectDataset(all_dirs, transform=val_tf),   val_indices)

    train_labels = [labels[i] for i in train_indices]
    class_weights = full_dataset.get_class_weights()

    # 第一层：原始数据 → WeightedRandomSampler；离线增强 → shuffle
    if pre_augmented:
        sampler_mode = "shuffle（离线已增强，避免重复过采样同一张图）"
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        sampler_mode = "WeightedRandomSampler（batch 类别均衡）"
        sample_weights = class_weights[torch.tensor(train_labels)].tolist()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\n  数据集划分 ──")
    for i, cls in enumerate(classes):
        n_train_cls = sum(1 for lbl in train_labels if lbl == i)
        n_val_cls   = sum(1 for lbl in [labels[j] for j in val_indices] if lbl == i)
        print(f"    {cls:<20}  训练 {n_train_cls:>4}  验证 {n_val_cls:>3}")
    print(f"    {'合计':<20}  训练 {len(train_indices):>4}  验证 {len(val_indices):>3}")
    print(f"    训练采样 : {sampler_mode}")

    return train_loader, val_loader, classes, class_weights


# ═══════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    构建 EfficientNet-B0 + 自定义分类头。

    · Dropout(0.4) 抑制小样本过拟合
    · pretrained=True 时加载 ImageNet 权重（从头训练推荐）
    · resume/finetune 时由 TrainingPipeline 设 pretrained=False，从 checkpoint 加载
    · torchvision 过旧时回退 ResNet-18
    """
    try:
        weights = (models.EfficientNet_B0_Weights.IMAGENET1K_V1
                   if pretrained else None)
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        print(f"  模型   : EfficientNet-B0  (in_features={in_features})")
    except AttributeError:
        # torchvision < 0.11 fallback
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )
        print(f"  模型   : ResNet-18 (fallback)  (in_features={in_features})")
    return model


def freeze_backbone(model: nn.Module):
    """阶段一：冻结 features（Backbone），仅 classifier 参与反向传播。"""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """解冻全部参数。"""
    for param in model.parameters():
        param.requires_grad = True


def count_params(model: nn.Module) -> Tuple[int, int]:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(
    model: nn.Module,
    ckpt_path: Path,
    device: torch.device,
    expected_classes: List[str],
) -> Dict[str, Any]:
    """
    加载 checkpoint 做增量训练。类别数变化时 strict=False，仅复用 Backbone 权重。
    """
    print(f"\n  加载 Checkpoint → {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    ckpt_classes = ckpt.get("classes", [])

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"  未加载（新参数）: {len(incompatible.missing_keys)} 项")
    if incompatible.unexpected_keys:
        print(f"  忽略（旧 checkpoint）: {len(incompatible.unexpected_keys)} 项")

    if ckpt_classes and ckpt_classes != expected_classes:
        print(f"  [注意] 类别列表变化:")
        print(f"    checkpoint: {ckpt_classes}")
        print(f"    当前数据: {expected_classes}")
        print(f"    分类头已按当前类别数重新初始化，Backbone 权重已尽量复用。")
    elif ckpt_classes:
        print(f"  类别一致: {ckpt_classes}")

    meta = {
        "epoch": ckpt.get("epoch", 0),
        "val_acc": ckpt.get("val_acc"),
        "macro_f1": ckpt.get("macro_f1"),
        "img_size": ckpt.get("img_size"),
    }
    if meta["macro_f1"] is not None:
        print(f"  历史最佳 Macro-F1: {meta['macro_f1']:.4f}")
    return meta


# ═══════════════════════════════════════════════
# MixUp
# ═══════════════════════════════════════════════
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device):
    """生成 MixUp 混合样本。"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=device)
    mixed_x  = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════
# 训练 / 验证 Loop
# ═══════════════════════════════════════════════
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    """
    单 epoch 训练。返回 (平均 loss, 准确率)。

    · AMP: scaler 非 None 时启用混合精度（仅 CUDA）
    · MixUp: mixup_alpha>0 时混合样本与标签（缺陷分类默认关闭）
    · 梯度裁剪 max_norm=1.0 防止小 batch 下梯度爆炸
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(enabled=(scaler is not None)):
            if mixup_alpha > 0:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha, device)
                logits = model(imgs)
                loss   = mixup_loss(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        # MixUp 时准确率参考 y_a
        ref_labels = y_a if mixup_alpha > 0 else labels
        correct += (preds == ref_labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model, loader, criterion, device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    验证一轮，返回 (loss, accuracy, macro_f1, all_preds, all_labels)。
    macro_f1 为第四层指标，用于选取最优 checkpoint。
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with amp_autocast(enabled=False):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    macro_f1   = compute_macro_f1(all_preds, all_labels)

    return (
        total_loss / total,
        correct / total,
        macro_f1,
        all_preds,
        all_labels,
    )


# ═══════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════
def plot_training_curves(history: dict, save_dir: Path):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("训练曲线", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="训练损失", linewidth=1.5)
    ax.plot(history["val_loss"],   label="验证损失", linewidth=1.5)
    # 标记阶段分界
    if "phase_split" in history:
        ax.axvline(history["phase_split"], color="gray", linestyle="--",
                   linewidth=1, label="阶段一/二分界")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("损失")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy + Macro-F1
    ax = axes[1]
    ax.plot(history["train_acc"], label="训练准确率", linewidth=1.5)
    ax.plot(history["val_acc"],   label="验证准确率", linewidth=1.5)
    if history.get("val_f1"):
        ax.plot(history["val_f1"], label="验证 Macro-F1",
                linewidth=1.8, linestyle="--", color="#E64A19")
    if "phase_split" in history:
        ax.axvline(history["phase_split"], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("准确率 / Macro-F1（红虚线为选模型依据）")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  训练曲线已保存   → {out}")


def plot_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    classes: List[str],
    save_dir: Path,
):
    if not (HAS_MPL and HAS_SKLEARN):
        return
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.3)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵（归一化）", fontweight="bold")

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    out = save_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  混淆矩阵已保存   → {out}")


# ═══════════════════════════════════════════════
# 模型导出
# ═══════════════════════════════════════════════
def export_onnx(model: nn.Module, img_size: int, save_dir: Path, device) -> Optional[Path]:
    """
    导出 ONNX 模型，并用 ONNXRuntime 验证精度一致性。
    导出在 CPU 上进行，结束后将模型恢复到原 device（避免后续 GPU 推理 device 不一致）。
    """
    if not HAS_ONNX:
        print("  [WARNING] onnx / onnxruntime 未安装，跳过 ONNX 导出。")
        return None

    model.eval()
    orig_device = next(model.parameters()).device
    cpu = torch.device("cpu")
    out_path = save_dir / "model.onnx"
    dummy_cpu = torch.randn(1, 3, img_size, img_size, device=cpu)

    try:
        model.to(cpu)
        torch.onnx.export(
            model,
            dummy_cpu,
            str(out_path),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        ort_session = ort.InferenceSession(str(out_path))
        ort_out = ort_session.run(
            None, {"input": dummy_cpu.numpy()}
        )[0]
        torch_out = model(dummy_cpu).detach().numpy()
        max_diff = float(np.abs(ort_out - torch_out).max())
        print(f"  ONNX 导出完成    → {out_path}")
        print(f"    ORT vs Torch 最大误差: {max_diff:.2e}  "
              f"{'✓' if max_diff < 1e-4 else '⚠'}")
        return out_path
    except Exception as exc:
        print(f"  [WARNING] ONNX 导出失败，已跳过: {exc}")
        return None
    finally:
        model.to(orig_device)
        model.eval()


def calibrate_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    classes: List[str],
    device,
    save_dir: Path,
) -> Dict[str, float]:
    """
    第四层 · 阈值校准（One-vs-Rest 逐类搜索）。

    对每个类别 c，在验证集上扫描阈值 t∈[0.05, 0.95]，使「prob[c]≥t 判为 c」的 F1 最大。
    推理引擎读取 class_thresholds.json 后，仅在 score≥阈值的类别中取 argmax，
    可提升少数类（如「局部破损」）召回率，降低漏检。
    """
    model.eval()
    model.to(device)
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy().tolist())

    probs_mat  = np.vstack(all_probs)
    labels_arr = np.array(all_labels)

    thresholds: Dict[str, float] = {}
    print(f"\n  {'─'*58}")
    print(f"  阈值校准（验证集，搜索步长 0.05）")
    print(f"  {'─'*58}")
    print(f"  {'类别':<12}  {'最优阈值':>8}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  n")

    for idx, cls_name in enumerate(classes):
        n_pos = int(np.sum(labels_arr == idx))
        best_f1, best_thr = 0.0, 0.5
        best_p, best_r = 0.0, 0.0

        for thr in np.arange(0.05, 0.96, 0.05):
            pred_bin  = (probs_mat[:, idx] >= thr).astype(int)
            label_bin = (labels_arr == idx).astype(int)
            tp  = int(np.sum((pred_bin == 1) & (label_bin == 1)))
            fp  = int(np.sum((pred_bin == 1) & (label_bin == 0)))
            fn  = int(np.sum((pred_bin == 0) & (label_bin == 1)))
            pre = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1  = 2 * pre * rec / (pre + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
                best_p, best_r = pre, rec

        thresholds[cls_name] = round(best_thr, 2)
        print(f"  {cls_name:<12}  {best_thr:>8.2f}  "
              f"{best_f1:>7.3f}  {best_p:>7.3f}  {best_r:>7.3f}  {n_pos}")

    out = save_dir / "class_thresholds.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f"\n  阈值文件已保存  → {out}")
    print(f"  （部署时可读取此文件，替代纯 argmax 以提升少数类召回）")
    return thresholds


def _ensure_utf8_console():
    """Windows 终端默认 GBK，确保中文日志与分类报告正常输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


# ═══════════════════════════════════════════════
# 训练 Pipeline（路径 / 数据 / 模型 / 训练 / 导出）
# ═══════════════════════════════════════════════
class TrainingPipeline:
    """
    训练流水线：串联数据 → 模型 → 损失 → 两阶段训练 → 评估/导出。

    生命周期:
        run()
          ├─ setup_data()      构建 DataLoader、保存 class_map.json
          ├─ setup_model()     构建网络、可选 load_checkpoint
          ├─ setup_criterion() FocalLoss 或加权 CE + AMP Scaler
          ├─ _train_phases()   阶段一（冻 Backbone）→ 阶段二（全量微调）
          └─ finalize()        最佳权重评估、曲线/混淆矩阵、阈值校准、ONNX

    postprocess_only 模式跳过 _train_phases，仅执行 finalize 中的后处理步骤。
    """

    def __init__(self, args: argparse.Namespace):
        """args 应已通过 normalize_training_paths 解析为绝对 Path。"""
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.data_dir: Path = args.data_dir
        self.save_dir: Path = args.save_dir
        self.classes: List[str] = []
        self.class_weights: Optional[torch.Tensor] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler = None
        self.history: Dict[str, Any] = {}
        self.best_macro_f1 = 0.0
        self.no_improve = 0
        self.best_ckpt_path = self.save_dir / "best_model.pt"
        self.resume_meta: Dict[str, Any] = {}

    def run(self) -> None:
        self._set_seed()
        self._print_banner()
        self.setup_data()
        self.setup_model()
        if self.args.postprocess_only:
            if not self.best_ckpt_path.is_file():
                raise FileNotFoundError(
                    f"--postprocess_only 需要已有权重: {self.best_ckpt_path}"
                )
            if self.resume_meta.get("macro_f1") is not None:
                self.best_macro_f1 = float(self.resume_meta["macro_f1"])
            print(f"\n  [--postprocess_only] 跳过训练，仅后处理")
            self.setup_criterion()
            self.finalize()
            return
        self.setup_criterion()
        self._train_phases()
        self.finalize()

    def _set_seed(self) -> None:
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

    def _print_banner(self) -> None:
        a = self.args
        print(f"\n{'═'*58}")
        print(f"  缺陷分类模型训练")
        print(f"{'═'*58}")
        print(f"  项目根   : {PROJECT_ROOT}")
        print(f"  工作目录 : {os.getcwd()}")
        print(f"  数据目录 : {self.data_dir}")
        print(f"  输出目录 : {self.save_dir}")
        print(f"  设备     : {self.device}  "
              + (f"({torch.cuda.get_device_name(0)})"
                 if self.device.type == "cuda" else "(CPU 模式)"))
        print(f"  图像尺寸 : {a.img_size} × {a.img_size}")
        print(f"  批大小   : {a.batch_size}")
        print(f"  阶段一   : {a.epochs_phase1} epochs  (lr={a.lr_phase1})")
        print(f"  阶段二   : {a.epochs_phase2} epochs  (lr={a.lr_phase2})")
        mode = "增量微调" if a.resume_path else "从头训练"
        print(f"  训练模式 : {mode}"
              + (f"  ← {a.resume_path.name}" if a.resume_path else ""))
        print(f"  MixUp α  : {a.mixup_alpha}")
        print(f"  离线增强 : {'是' if a.pre_augmented else '否（+ Sampler）'}")
        print(f"  类别权重 : {'是' if a.use_class_weight else '否'}")
        print(f"  Focal    : {'是 γ=' + str(a.focal_gamma) if a.use_focal_loss else '否'}")
        print(f"  选模指标 : val_macro_f1")

    def setup_data(self) -> None:
        a = self.args
        self.train_loader, self.val_loader, self.classes, self.class_weights = (
            build_dataloaders(
                data_dir=self.data_dir,
                img_size=a.img_size,
                val_ratio=a.val_ratio,
                batch_size=a.batch_size,
                num_workers=a.num_workers,
                seed=a.seed,
                extra_data_dirs=a.extra_data_dirs,
                pre_augmented=a.pre_augmented,
                save_dir=self.save_dir,
                reuse_split=a.reuse_split,
            )
        )
        print(f"\n  类别({len(self.classes)})  : {self.classes}")

        all_dirs = [self.data_dir] + list(a.extra_data_dirs)
        full_scan = DefectDataset(all_dirs, transform=None)
        print(f"  样本总数 : {len(full_scan)}")
        for cls in self.classes:
            idx = full_scan.class_to_idx[cls]
            n = sum(1 for _, lbl in full_scan.samples if lbl == idx)
            print(f"    {cls:<12} {n:>5}  ({n/len(full_scan)*100:.1f}%)")

        print(f"\n  类别权重（逆频，损失第二层）:")
        for cls, w in zip(self.classes, self.class_weights.tolist()):
            bar = "█" * max(1, int(w * 8))
            print(f"    {cls:<12}  {w:>6.3f}  {bar}")

        cls_map = {
            "classes": self.classes,
            "class_to_idx": {c: i for i, c in enumerate(self.classes)},
        }
        with open(self.save_dir / "class_map.json", "w", encoding="utf-8") as f:
            json.dump(cls_map, f, ensure_ascii=False, indent=2)

    def setup_model(self) -> None:
        """构建模型；若 args.resume_path 存在则加载权重做增量训练。"""
        a = self.args
        use_pretrained = a.pretrained and (a.resume_path is None)
        if a.resume_path and a.pretrained:
            print(f"\n  Backbone: 从 checkpoint 加载（忽略 ImageNet 预训练）")
        elif use_pretrained:
            print(f"\n  Backbone: ImageNet-1K 预训练")
        else:
            print(f"\n  Backbone: 随机初始化")

        self.model = build_model(
            len(self.classes), pretrained=use_pretrained
        ).to(self.device)

        if a.resume_path:
            self.resume_meta = load_checkpoint(
                self.model, a.resume_path, self.device, self.classes
            )
            if a.resume_path == self.best_ckpt_path and a.resume_meta.get("macro_f1"):
                self.best_macro_f1 = float(a.resume_meta["macro_f1"])

    def setup_criterion(self) -> None:
        a = self.args
        cw = self.class_weights.to(self.device) if a.use_class_weight else None
        if a.use_focal_loss:
            self.criterion = FocalLoss(gamma=a.focal_gamma, weight=cw)
            print(f"  损失函数 : FocalLoss(γ={a.focal_gamma})"
                  + (" + 类别权重" if cw is not None else ""))
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=cw, label_smoothing=a.label_smooth,
            )
            print(f"  损失函数 : CrossEntropy(平滑={a.label_smooth})"
                  + (" + 类别权重" if cw is not None else ""))

        use_amp = a.amp and (self.device.type == "cuda")
        self.scaler = create_grad_scaler(use_amp)
        if use_amp:
            print(f"  混合精度: 启用 (AMP)")

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_f1": [], "phase_split": a.epochs_phase1,
        }

    def _run_epoch(
        self, epoch: int, total_epochs: int, phase_label: str,
        optimizer, scheduler,
    ) -> Tuple[np.ndarray, np.ndarray]:
        a = self.args
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            self.model, self.train_loader, self.criterion,
            optimizer, self.scaler, self.device,
            mixup_alpha=a.mixup_alpha,
        )
        val_loss, val_acc, macro_f1, val_preds, val_labels = validate(
            self.model, self.val_loader, self.criterion, self.device,
        )
        elapsed = time.time() - t0

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self.history["val_f1"].append(macro_f1)

        improved = macro_f1 > self.best_macro_f1
        if improved:
            self.best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "val_acc": val_acc,
                    "macro_f1": macro_f1,
                    "classes": self.classes,
                    "img_size": a.img_size,
                },
                self.best_ckpt_path,
            )
            self.no_improve = 0
        else:
            self.no_improve += 1

        marker = "★" if improved else " "
        print(
            f"  [{phase_label}] Ep {epoch:>3}/{total_epochs}  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"acc {train_acc:.3f}/{val_acc:.3f}  "
            f"F1 {macro_f1:.3f}  {elapsed:.1f}s  {marker}"
        )
        scheduler.step()
        return val_preds, val_labels

    def _train_phases(self) -> None:
        """
        两阶段训练 + Early Stopping（patience 轮 val_macro_f1 无提升则停止）。

        阶段二 Head 学习率为 Backbone 的 5 倍，使分类头更快适应新数据。
        """
        a = self.args
        if a.epochs_phase1 > 0:
            print(f"\n{'─'*58}")
            print(f"  阶段一：冻结 Backbone  ({a.epochs_phase1} epochs)")
            print(f"{'─'*58}")
            freeze_backbone(self.model)
            total, trainable = count_params(self.model)
            print(f"  可训练参数: {trainable:,} / {total:,}")

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=a.lr_phase1, weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=a.epochs_phase1, eta_min=a.lr_phase1 * 0.1,
            )
            for ep in range(1, a.epochs_phase1 + 1):
                self._run_epoch(ep, a.epochs_phase1, "P1", optimizer, scheduler)
                if self.no_improve >= a.patience:
                    print(f"  Early stopping @ epoch {ep}")
                    break
            self.no_improve = 0

        if a.epochs_phase2 > 0:
            print(f"\n{'─'*58}")
            print(f"  阶段二：端到端微调  ({a.epochs_phase2} epochs)")
            print(f"{'─'*58}")
            unfreeze_all(self.model)
            total, trainable = count_params(self.model)
            print(f"  可训练参数: {trainable:,} / {total:,}")

            backbone_params = [
                p for n, p in self.model.named_parameters() if "classifier" not in n
            ]
            head_params = [
                p for n, p in self.model.named_parameters() if "classifier" in n
            ]
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": a.lr_phase2},
                    {"params": head_params, "lr": a.lr_phase2 * 5},
                ],
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=a.epochs_phase2, eta_min=a.lr_phase2 * 0.05,
            )
            offset = a.epochs_phase1 if a.epochs_phase1 > 0 else 0
            for ep in range(1, a.epochs_phase2 + 1):
                self._run_epoch(
                    offset + ep, offset + a.epochs_phase2, "P2",
                    optimizer, scheduler,
                )
                if self.no_improve >= a.patience:
                    print(f"  Early stopping @ epoch {offset + ep}")
                    break

    def finalize(self) -> None:
        """
        训练收尾：加载 best_model.pt → 报告 → 可视化 → 阈值校准 → ONNX → train_config.json。

        顺序说明: 阈值校准在 GPU 上跑验证集；ONNX 导出会临时将模型移到 CPU，
        export_onnx 的 finally 块会恢复原 device。
        """
        a = self.args
        if not self.best_ckpt_path.is_file():
            print("\n  [警告] 未产生 best_model.pt，跳过评估与导出。")
            return

        print(f"\n{'═'*58}")
        print(f"  最佳 Macro-F1  : {self.best_macro_f1:.4f}")
        print(f"  模型已保存     → {self.best_ckpt_path}")
        print(f"{'═'*58}")

        ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        _, final_acc, final_f1, final_preds, final_labels = validate(
            self.model, self.val_loader, self.criterion, self.device,
        )
        print(f"  最终验证准确率 : {final_acc:.4f}")
        print(f"  最终 Macro-F1  : {final_f1:.4f}")

        if HAS_SKLEARN:
            print("\n" + classification_report(
                final_labels, final_preds,
                target_names=self.classes, digits=4, zero_division=0,
            ))

        plot_training_curves(self.history, self.save_dir)
        plot_confusion_matrix(
            final_preds, final_labels, self.classes, self.save_dir,
        )

        # 阈值校准需在 GPU 上跑验证集；放在 ONNX 导出之前，避免 export 临时切到 CPU
        print(f"\n{'─'*58}\n  阈值校准\n{'─'*58}")
        self.model.to(self.device)
        calibrate_thresholds(
            self.model, self.val_loader, self.classes,
            self.device, self.save_dir,
        )

        print(f"\n{'─'*58}\n  模型导出\n{'─'*58}")
        export_onnx(self.model, a.img_size, self.save_dir, self.device)

        train_cfg = {
            "project_root":   str(PROJECT_ROOT),
            "data_dir":       str(self.data_dir),
            "img_size":       a.img_size,
            "num_classes":    len(self.classes),
            "classes":        self.classes,
            "best_val_acc":   round(final_acc, 6),
            "best_macro_f1":  round(final_f1, 6),
            "model_arch":     "efficientnet_b0",
            "use_focal_loss": a.use_focal_loss,
            "use_class_weight": a.use_class_weight,
            "focal_gamma":    a.focal_gamma,
            "resumed_from":   str(a.resume_path) if a.resume_path else None,
        }
        with open(self.save_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(train_cfg, f, ensure_ascii=False, indent=2)
        print(f"  训练配置已保存  → {self.save_dir / 'train_config.json'}")
        print(f"\n  全部完成 ✓\n")


def main():
    """入口：解析参数 → 规范化路径 → 运行 TrainingPipeline。"""
    _ensure_utf8_console()
    try:
        args = normalize_training_paths(get_args())
        TrainingPipeline(args).run()
    except FileNotFoundError as e:
        print(f"\n[错误] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
