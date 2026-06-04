#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像缺陷分类模型训练管线
  · 模型  : EfficientNet-B0（torchvision 迁移学习）
  · 策略  : 两阶段微调 —— 先冻结 Backbone 训练分类头，再端到端微调
  · 增强  : 默认适配离线已增强数据（轻量在线增强）；原始数据可 --no_pre_augmented
  · 导出  : .pt（PyTorch 原生，GPU 推理）+ .onnx（ONNXRuntime，CPU 推理）

用法:
    python train.py
    python train.py --data_dir data --img_size 128 --batch_size 32 --epochs_phase1 5 --epochs_phase2 20
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as T
import torchvision.models as models
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
# 配置
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
                   help="数据已离线增强：使用轻量在线增强 + 加权损失（默认开启）")
    p.add_argument("--no_pre_augmented", dest="pre_augmented", action="store_false",
                   help="数据未增强：启用完整在线增强与 WeightedRandomSampler")
    p.add_argument("--weighted_loss",  action="store_true", default=True,
                   help="CrossEntropyLoss 使用类别反频率权重（默认开启）")
    p.add_argument("--no_weighted_loss", dest="weighted_loss", action="store_false",
                   help="关闭损失函数类别加权")
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
    return p.parse_args()


# ═══════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DefectDataset(torch.utils.data.Dataset):
    """
    从一个或多个目录加载缺陷图像数据集。
    支持同时传入 data/ 和 corrections/ 合并训练。
    期望结构:
        data_dir/
            class_A/  *.jpg ...
            class_B/  *.jpg ...
    """

    def __init__(self, data_dirs, transform=None):
        """
        data_dirs: Path / str / List[Path|str]
          单个目录或目录列表，类别以子文件夹名称区分，多目录取类别并集。
        """
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        # 统一转为 Path 列表
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [Path(data_dirs)]
        else:
            data_dirs = [Path(d) for d in data_dirs]

        # 收集所有目录下的类别（取并集，字典序排序保证一致性）
        class_set: set = set()
        for d in data_dirs:
            if d.exists():
                class_set.update(
                    sub.name for sub in d.iterdir() if sub.is_dir()
                )
        if not class_set:
            raise RuntimeError(
                f"在 {[str(d) for d in data_dirs]} 下未找到类别子文件夹。"
            )

        self.classes      = sorted(class_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # 遍历所有目录收集图像路径
        for d in data_dirs:
            if not d.exists():
                continue
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
        """计算反频率类别权重，用于不均衡数据集。"""
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / counts
        return weights / weights.sum() * len(self.classes)   # 归一化


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


def build_dataloaders(
    data_dir: Path,
    img_size: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    extra_data_dirs: List[str] = None,
    pre_augmented: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str], torch.Tensor]:
    """构建 train/val DataLoader，分层划分；离线增强数据用 shuffle，原始数据用 WeightedRandomSampler。"""

    train_tf, val_tf = get_transforms(img_size, pre_augmented=pre_augmented)

    # 合并主数据目录与额外数据目录（如 corrections/）
    all_dirs = [data_dir]
    if extra_data_dirs:
        all_dirs += [Path(d) for d in extra_data_dirs if Path(d).exists()]

    # 先用无增强的数据集扫描全部文件，获取标签列表
    full_dataset = DefectDataset(all_dirs, transform=None)
    classes      = full_dataset.classes
    n_total      = len(full_dataset)
    labels       = [lbl for _, lbl in full_dataset.samples]

    # 分层划分（按类别比例保留验证集）
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for cls_idx in range(len(classes)):
        cls_indices = np.where(np.array(labels) == cls_idx)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    # 两份 Subset 各自用不同 transform
    train_set = Subset(DefectDataset(all_dirs, transform=train_tf), train_indices)
    val_set   = Subset(DefectDataset(all_dirs, transform=val_tf),   val_indices)

    train_labels = [labels[i] for i in train_indices]
    class_weights = full_dataset.get_class_weights()

    # 离线已增强：shuffle 遍历全集 + 损失加权，避免对同一增强图过采样
    # 原始数据：WeightedRandomSampler 平衡类别
    if pre_augmented:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
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

    return train_loader, val_loader, classes, class_weights


# ═══════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    构建 EfficientNet-B0 模型，替换分类头为目标类别数。
    若 torchvision 版本较旧，自动回退到 resnet18。
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
    """冻结 EfficientNet features（Backbone），仅开放 classifier。"""
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
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
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

    # Accuracy
    ax = axes[1]
    ax.plot(history["train_acc"], label="训练准确率", linewidth=1.5)
    ax.plot(history["val_acc"],   label="验证准确率", linewidth=1.5)
    if "phase_split" in history:
        ax.axvline(history["phase_split"], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("准确率")
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
def export_onnx(model: nn.Module, img_size: int, save_dir: Path, device) -> Path:
    """导出 ONNX 模型，并用 ONNXRuntime 验证精度一致性。"""
    if not HAS_ONNX:
        print("  [WARNING] onnx / onnxruntime 未安装，跳过 ONNX 导出。")
        return None

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path = save_dir / "model.onnx"

    torch.onnx.export(
        model.cpu(),
        dummy.cpu(),
        str(out_path),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # 验证
    ort_session = ort.InferenceSession(str(out_path))
    ort_out = ort_session.run(None, {"input": dummy.cpu().numpy()})[0]
    torch_out = model.cpu()(dummy.cpu()).detach().numpy()
    max_diff = np.abs(ort_out - torch_out).max()
    print(f"  ONNX 导出完成    → {out_path}")
    print(f"    ORT vs Torch 最大误差: {max_diff:.2e}  {'✓' if max_diff < 1e-4 else '⚠'}")

    return out_path


def _ensure_utf8_console():
    """Windows 终端默认 GBK，确保中文日志与分类报告正常输出。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


# ═══════════════════════════════════════════════
# 主训练流程
# ═══════════════════════════════════════════════
def main():
    _ensure_utf8_console()
    args    = get_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n{'═'*58}")
    print(f"  缺陷分类模型训练")
    print(f"{'═'*58}")
    print(f"  设备     : {device}  "
          + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "(CPU 模式)"))
    print(f"  图像尺寸 : {args.img_size} × {args.img_size}")
    print(f"  批大小   : {args.batch_size}")
    print(f"  阶段一   : {args.epochs_phase1} epochs  (lr={args.lr_phase1})")
    print(f"  阶段二   : {args.epochs_phase2} epochs  (lr={args.lr_phase2})")
    print(f"  MixUp α  : {args.mixup_alpha}")
    print(f"  标签平滑 : {args.label_smooth}")
    print(f"  离线增强 : {'是（轻量在线增强）' if args.pre_augmented else '否（完整在线增强）'}")
    print(f"  损失加权 : {'是' if args.weighted_loss else '否'}")

    # ── 数据 ──────────────────────────────────
    train_loader, val_loader, classes, class_weights = build_dataloaders(
        data_dir        = data_dir,
        img_size        = args.img_size,
        val_ratio       = args.val_ratio,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
        extra_data_dirs = args.extra_data_dirs,
        pre_augmented   = args.pre_augmented,
    )
    num_classes = len(classes)
    print(f"\n  类别({num_classes})  : {classes}")

    # 打印类别样本分布，便于确认不均衡程度
    all_dirs = [data_dir] + [Path(d) for d in args.extra_data_dirs if Path(d).exists()]
    full_scan = DefectDataset(all_dirs, transform=None)
    print(f"  样本总数 : {len(full_scan)}")
    for cls in classes:
        idx = full_scan.class_to_idx[cls]
        n = sum(1 for _, lbl in full_scan.samples if lbl == idx)
        print(f"    {cls:<12} {n:>5}  ({n/len(full_scan)*100:.1f}%)")

    # 保存类别映射（供 PyQt 应用加载）
    cls_map = {"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}
    with open(save_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(cls_map, f, ensure_ascii=False, indent=2)

    # ── 模型 ──────────────────────────────────
    print(f"\n  预训练  : ImageNet-1K")
    model = build_model(num_classes, pretrained=True).to(device)

    # ── 损失 ──────────────────────────────────
    loss_weight = class_weights.to(device) if args.weighted_loss else None
    criterion = nn.CrossEntropyLoss(
        weight=loss_weight,
        label_smoothing=args.label_smooth,
    )

    # AMP Scaler（仅 CUDA）
    use_amp = args.amp and (device.type == "cuda")
    scaler  = create_grad_scaler(use_amp)
    if use_amp:
        print(f"  混合精度: 启用 (AMP)")

    # ── 训练历史 ──────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "phase_split": args.epochs_phase1,
    }

    best_val_acc  = 0.0
    no_improve    = 0
    best_ckpt_path = save_dir / "best_model.pt"

    def _run_epoch(epoch, total_epochs, phase_label):
        """执行单轮训练+验证，打印日志，更新历史。"""
        nonlocal best_val_acc, no_improve

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            mixup_alpha=args.mixup_alpha,
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device,
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_acc":    val_acc,
                    "classes":    classes,
                    "img_size":   args.img_size,
                },
                best_ckpt_path,
            )
            no_improve = 0
        else:
            no_improve += 1

        marker = "★" if improved else " "
        print(
            f"  [{phase_label}] Ep {epoch:>3}/{total_epochs}  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"acc {train_acc:.3f}/{val_acc:.3f}  "
            f"{elapsed:.1f}s  {marker}"
        )
        scheduler.step()
        return val_preds, val_labels

    # ════════════════════════════════════
    # 阶段一：冻结 Backbone，训练分类头
    # ════════════════════════════════════
    if args.epochs_phase1 > 0:
        print(f"\n{'─'*58}")
        print(f"  阶段一：冻结 Backbone，训练分类头  ({args.epochs_phase1} epochs)")
        print(f"{'─'*58}")

        freeze_backbone(model)
        total, trainable = count_params(model)
        print(f"  参数量: 总计={total:,}  可训练={trainable:,}  "
              f"({trainable/total*100:.1f}%)")

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_phase1, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_phase1, eta_min=args.lr_phase1 * 0.1,
        )

        for ep in range(1, args.epochs_phase1 + 1):
            _run_epoch(ep, args.epochs_phase1, "P1")
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {ep}（{args.patience} 轮无改善）")
                break

        no_improve = 0   # 重置，阶段二单独计算

    # ════════════════════════════════════
    # 阶段二：解冻全部，端到端微调
    # ════════════════════════════════════
    if args.epochs_phase2 > 0:
        print(f"\n{'─'*58}")
        print(f"  阶段二：解冻全部层，端到端微调  ({args.epochs_phase2} epochs)")
        print(f"{'─'*58}")

        unfreeze_all(model)
        total, trainable = count_params(model)
        print(f"  参数量: 总计={total:,}  可训练={trainable:,}  (100%)")

        # 对 Backbone 和 Head 使用不同学习率
        backbone_params = [p for n, p in model.named_parameters()
                           if "classifier" not in n]
        head_params     = [p for n, p in model.named_parameters()
                           if "classifier" in n]
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr_phase2},
                {"params": head_params,     "lr": args.lr_phase2 * 5},
            ],
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_phase2, eta_min=args.lr_phase2 * 0.05,
        )

        offset = args.epochs_phase1 if args.epochs_phase1 > 0 else 0
        last_preds = last_labels = None

        for ep in range(1, args.epochs_phase2 + 1):
            last_preds, last_labels = _run_epoch(
                offset + ep, offset + args.epochs_phase2, "P2"
            )
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {offset+ep}（{args.patience} 轮无改善）")
                break

    # ════════════════════════════════════
    # 训练结束，评估与导出
    # ════════════════════════════════════
    print(f"\n{'═'*58}")
    print(f"  最佳验证准确率 : {best_val_acc:.4f}")
    print(f"  模型已保存     → {best_ckpt_path}")
    print(f"{'═'*58}")

    # 加载最佳权重进行最终评估
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    _, final_acc, final_preds, final_labels = validate(
        model, val_loader, criterion, device
    )
    print(f"  最终验证准确率 : {final_acc:.4f}")

    if HAS_SKLEARN:
        print("\n" + classification_report(
            final_labels, final_preds,
            target_names=classes, digits=4,
            zero_division=0,
        ))

    # 可视化
    plot_training_curves(history, save_dir)
    plot_confusion_matrix(final_preds, final_labels, classes, save_dir)

    # ONNX 导出（将模型移到 CPU 以便通用部署）
    print(f"\n{'─'*58}")
    print("  模型导出")
    print(f"{'─'*58}")
    export_onnx(model, args.img_size, save_dir, device)

    # 保存训练配置（供再训练和 PyQt 应用读取）
    train_cfg = {
        "img_size":      args.img_size,
        "num_classes":   num_classes,
        "classes":       classes,
        "best_val_acc":  round(best_val_acc, 6),
        "model_arch":    "efficientnet_b0",
    }
    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_cfg, f, ensure_ascii=False, indent=2)
    print(f"  训练配置已保存  → {save_dir / 'train_config.json'}")
    print(f"\n  全部完成 ✓\n")


if __name__ == "__main__":
    main()
