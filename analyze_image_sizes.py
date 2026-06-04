#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像尺寸统计分析工具
扫描 data/ 下各缺陷类别子文件夹，统计图像尺寸分布，推荐统一输入分辨率。

用法:
    python analyze_image_sizes.py
    python analyze_image_sizes.py --data_dir path/to/data --save_dir outputs
    python analyze_image_sizes.py --no_save    # 直接弹窗显示图表，不保存文件
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 自动选择支持中文的字体（Windows / Linux 均适用）
def _setup_cjk_font():
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
            return
_setup_cjk_font()
from PIL import Image

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 标准候选分辨率（短边 / 正方形边长），按升序排列
CANDIDATE_SIZES = [128, 160, 192, 224, 256, 288, 320, 384, 448, 512]

# 判断宽高比是否为"近似正方形"的阈值
SQUARE_AR_MEAN_THRESH = 0.25   # |mean_ar - 1.0| < 此值时倾向正方形
SQUARE_AR_STD_THRESH  = 0.30   # ar_std < 此值时倾向正方形


# ─────────────────────────────────────────────
# 1. 数据收集
# ─────────────────────────────────────────────
def collect_image_stats(data_dir: Path) -> pd.DataFrame:
    """遍历 data_dir 下各类别子文件夹，收集每张图像的尺寸信息。"""

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"[ERROR] 在 {data_dir.resolve()} 下未找到子文件夹，请检查路径。")
        sys.exit(1)

    sep = "─" * 58
    print(f"\n{'═'*58}")
    print(f"  扫描目录 : {data_dir.resolve()}")
    print(f"  发现类别 : {len(class_dirs)} 个")
    print(f"{'═'*58}")

    records = []
    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        img_files = [
            f for f in cls_dir.rglob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not img_files:
            print(f"  [WARNING] 类别 '{cls_name}' 下未找到图像文件，已跳过。")
            continue

        print(f"\n  [{cls_name}]  共 {len(img_files)} 张，读取中 ...", end="", flush=True)

        failed = 0
        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size          # PIL 返回 (width, height)
                    channels = len(img.getbands())
                    mode = img.mode
                records.append(
                    {
                        "class":        cls_name,
                        "filename":     img_path.name,
                        "width":        w,
                        "height":       h,
                        "aspect_ratio": round(w / h, 4),
                        "short_side":   min(w, h),
                        "long_side":    max(w, h),
                        "channels":     channels,
                        "mode":         mode,
                        "megapixels":   round(w * h / 1e6, 3),
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"\n    [WARNING] 无法读取 {img_path.name}: {exc}", end="")

        status = f"  完成" + (f"（{failed} 张失败）" if failed else "")
        print(status)

    df = pd.DataFrame(records)
    print()
    return df


# ─────────────────────────────────────────────
# 2. 统计输出
# ─────────────────────────────────────────────
def _fmt(series: pd.Series) -> str:
    """将一列数值格式化为常用统计摘要字符串。"""
    return (
        f"最小 {series.min():>6.0f}  "
        f"最大 {series.max():>6.0f}  "
        f"均值 {series.mean():>7.1f}  "
        f"中位 {series.median():>7.1f}  "
        f"P90 {series.quantile(0.9):>7.1f}"
    )


def print_statistics(df: pd.DataFrame):
    """打印各类别及整体统计信息至控制台。"""
    print(f"{'═'*58}")
    print("  各类别图像尺寸统计")
    print(f"{'═'*58}")

    for cls_name, grp in df.groupby("class", sort=True):
        print(f"\n  ── {cls_name}  ({len(grp)} 张) ──")
        print(f"    宽(W) : {_fmt(grp['width'])}")
        print(f"    高(H) : {_fmt(grp['height'])}")
        ar = grp["aspect_ratio"]
        print(
            f"    AR    : 均值={ar.mean():.3f}  "
            f"中位={ar.median():.3f}  std={ar.std():.3f}"
        )
        mode_str = "  ".join(f"{k}:{v}" for k, v in grp["mode"].value_counts().items())
        print(f"    模式  : {mode_str}")

    print(f"\n{'─'*58}")
    print("  整体汇总")
    print(f"{'─'*58}")
    print(f"  总图像数 : {len(df)}")
    print(f"  宽(W)   : {_fmt(df['width'])}")
    print(f"  高(H)   : {_fmt(df['height'])}")
    ar_all = df["aspect_ratio"]
    print(
        f"  AR      : 均值={ar_all.mean():.3f}  "
        f"中位={ar_all.median():.3f}  std={ar_all.std():.3f}"
    )

    # 百分位数表
    pct_levels = [50, 75, 90, 95, 99]
    pw = df["width"].quantile([p / 100 for p in pct_levels])
    ph = df["height"].quantile([p / 100 for p in pct_levels])
    ps = df["short_side"].quantile([p / 100 for p in pct_levels])

    header = "  " + f"{'分位数':<10}" + "".join(f"  P{p:<5}" for p in pct_levels)
    print(f"\n{header}")
    print("  " + f"{'宽(W)':<10}" + "".join(f"  {pw[p/100]:>6.0f}" for p in pct_levels))
    print("  " + f"{'高(H)':<10}" + "".join(f"  {ph[p/100]:>6.0f}" for p in pct_levels))
    print("  " + f"{'短边':<10}" + "".join(f"  {ps[p/100]:>6.0f}" for p in pct_levels))


# ─────────────────────────────────────────────
# 3. 分辨率推荐
# ─────────────────────────────────────────────
def _snap_to_standard(target: float) -> int:
    """将目标像素数向上取最近的标准候选尺寸（确保不低于 target 的 85%）。"""
    best = min(CANDIDATE_SIZES, key=lambda s: abs(s - target))
    # 若最近值比目标低超过 15%，取下一档
    if best < target * 0.85:
        idx = CANDIDATE_SIZES.index(best)
        if idx + 1 < len(CANDIDATE_SIZES):
            best = CANDIDATE_SIZES[idx + 1]
    return best


def recommend_resolution(df: pd.DataFrame) -> dict:
    """基于尺寸统计数据推荐统一输入分辨率。"""
    mean_ar = df["aspect_ratio"].mean()
    ar_std  = df["aspect_ratio"].std()

    p90_w     = df["width"].quantile(0.9)
    p90_h     = df["height"].quantile(0.9)
    p90_short = df["short_side"].quantile(0.9)

    use_square = (
        abs(mean_ar - 1.0) < SQUARE_AR_MEAN_THRESH
        and ar_std < SQUARE_AR_STD_THRESH
    )

    if use_square:
        size = _snap_to_standard(p90_short)
        res_w, res_h = size, size
        shape_note = f"正方形  {size} × {size}"
        reason = (
            f"宽高比均值 {mean_ar:.3f}（≈ 1.0），std {ar_std:.3f}，"
            f"图像近似正方形；短边 P90 = {p90_short:.1f}px → 取标准尺寸 {size}。"
        )
    else:
        res_w = _snap_to_standard(p90_w)
        res_h = _snap_to_standard(p90_h)
        shape_note = f"矩形    {res_w} × {res_h}  (W × H)"
        reason = (
            f"宽高比均值 {mean_ar:.3f}，std {ar_std:.3f}，偏离正方形；"
            f"W P90 = {p90_w:.1f} → {res_w}，H P90 = {p90_h:.1f} → {res_h}。"
        )

    rec = {
        "res_w":      res_w,
        "res_h":      res_h,
        "is_square":  use_square,
        "shape_note": shape_note,
        "reason":     reason,
        "mean_ar":    mean_ar,
        "ar_std":     ar_std,
    }

    print(f"\n{'═'*58}")
    print("  推荐统一输入分辨率")
    print(f"{'═'*58}")
    print(f"\n  ► 推荐分辨率 :  {shape_note}")
    print(f"\n  分析依据 :")
    print(f"    {reason}")
    print(f"\n  附加建议 :")
    print(f"    · Resize 策略 : 等比缩放至短边 = 目标尺寸，再中心裁剪")
    print(f"    · 若缺陷常出现在边缘，改用 Letterbox Padding 保留全图")
    print(f"    · EfficientNet-B0 原生尺寸 224×224，上述推荐可直接传入")
    print(f"      transforms.Resize((res_h, res_w))")
    print(f"    · 如后续选择 EfficientNet-B2/B4，可相应上调至 260 / 380")

    return rec


# ─────────────────────────────────────────────
# 4. 可视化
# ─────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame, rec: dict, save_path: Path = None):
    """绘制 6 宫格尺寸分布分析图。"""
    classes = df["class"].unique()
    palette = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    color_map = dict(zip(sorted(classes), palette))

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle("图像尺寸分布分析", fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])   # W vs H 散点
    ax2 = fig.add_subplot(gs[0, 1])   # 宽度直方图
    ax3 = fig.add_subplot(gs[0, 2])   # 高度直方图
    ax4 = fig.add_subplot(gs[1, 0])   # 宽高比分布
    ax5 = fig.add_subplot(gs[1, 1])   # 各类样本数
    ax6 = fig.add_subplot(gs[1, 2])   # 短边 vs 长边

    # ── 图1: W vs H 散点 ──────────────────────
    for cls_name, grp in df.groupby("class"):
        ax1.scatter(
            grp["width"], grp["height"],
            c=[color_map[cls_name]], label=cls_name,
            alpha=0.65, s=18, edgecolors="none"
        )
    ax1.axvline(rec["res_w"], color="#E24B4A", linestyle="--", lw=1.2,
                label=f"推荐 W={rec['res_w']}")
    ax1.axhline(rec["res_h"], color="#EF9F27", linestyle="--", lw=1.2,
                label=f"推荐 H={rec['res_h']}")
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    ax1.set_title("宽 vs 高", fontweight="bold")
    ax1.legend(fontsize=7.5, markerscale=1.8, loc="upper left")

    # ── 图2: 宽度直方图 ───────────────────────
    for cls_name, grp in df.groupby("class"):
        ax2.hist(grp["width"], bins=25, alpha=0.55, color=color_map[cls_name],
                 label=cls_name, density=True)
    ax2.axvline(rec["res_w"], color="#E24B4A", linestyle="--", lw=1.5,
                label=f"推荐={rec['res_w']}")
    ax2.axvline(df["width"].median(), color="gray", linestyle=":", lw=1,
                label=f"中位数={df['width'].median():.0f}")
    ax2.set_xlabel("Width (px)")
    ax2.set_ylabel("密度")
    ax2.set_title("宽度分布", fontweight="bold")
    ax2.legend(fontsize=7.5)

    # ── 图3: 高度直方图 ───────────────────────
    for cls_name, grp in df.groupby("class"):
        ax3.hist(grp["height"], bins=25, alpha=0.55, color=color_map[cls_name],
                 label=cls_name, density=True)
    ax3.axvline(rec["res_h"], color="#EF9F27", linestyle="--", lw=1.5,
                label=f"推荐={rec['res_h']}")
    ax3.axvline(df["height"].median(), color="gray", linestyle=":", lw=1,
                label=f"中位数={df['height'].median():.0f}")
    ax3.set_xlabel("Height (px)")
    ax3.set_ylabel("密度")
    ax3.set_title("高度分布", fontweight="bold")
    ax3.legend(fontsize=7.5)

    # ── 图4: 宽高比分布 ──────────────────────
    for cls_name, grp in df.groupby("class"):
        ax4.hist(grp["aspect_ratio"], bins=25, alpha=0.55, color=color_map[cls_name],
                 label=cls_name, density=True)
    ax4.axvline(1.0, color="gray", linestyle=":", lw=1, label="AR=1.0 (正方形)")
    ax4.axvline(df["aspect_ratio"].mean(), color="black", linestyle="--", lw=1.3,
                label=f"均值={df['aspect_ratio'].mean():.2f}")
    ax4.set_xlabel("宽高比 (W/H)")
    ax4.set_ylabel("密度")
    ax4.set_title("宽高比分布", fontweight="bold")
    ax4.legend(fontsize=7.5)

    # ── 图5: 各类样本数 ──────────────────────
    counts = df.groupby("class").size().sort_index()
    bars = ax5.bar(
        counts.index, counts.values,
        color=[color_map[c] for c in counts.index],
        edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, counts.values):
        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(val), ha="center", va="bottom", fontsize=9
        )
    ax5.set_xlabel("类别")
    ax5.set_ylabel("图像数量")
    ax5.set_title("各类别样本数量", fontweight="bold")
    ax5.tick_params(axis="x", rotation=20)
    # 标出总均值
    mean_count = counts.mean()
    ax5.axhline(mean_count, color="gray", linestyle=":", lw=1,
                label=f"均值={mean_count:.1f}")
    ax5.legend(fontsize=8)

    # ── 图6: 短边 vs 长边 ────────────────────
    for cls_name, grp in df.groupby("class"):
        ax6.scatter(
            grp["long_side"], grp["short_side"],
            c=[color_map[cls_name]], label=cls_name,
            alpha=0.65, s=18, edgecolors="none"
        )
    max_val = df["long_side"].max()
    ax6.plot([0, max_val], [0, max_val], color="gray", linestyle=":", lw=0.8, label="正方形线")
    ax6.set_xlabel("长边 (px)")
    ax6.set_ylabel("短边 (px)")
    ax6.set_title("短边 vs 长边", fontweight="bold")
    ax6.legend(fontsize=7.5, markerscale=1.8)

    # 在图1右下角加上推荐框注释
    ax1.annotate(
        f" 推荐: {rec['shape_note']} ",
        xy=(0.98, 0.04), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FAECE7", ec="#D85A30", lw=1),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  可视化图表已保存 → {save_path}")
    else:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


# ─────────────────────────────────────────────
# 5. 保存 CSV
# ─────────────────────────────────────────────
def save_csv(df: pd.DataFrame, save_dir: Path):
    """将图像尺寸明细保存为 CSV 文件。"""
    csv_path = save_dir / "image_size_stats.csv"
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  明细数据已保存  → {csv_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="图像尺寸统计分析工具")
    parser.add_argument("--data_dir",  type=str, default="data",
                        help="数据集根目录（默认: data）")
    parser.add_argument("--save_dir",  type=str, default="outputs",
                        help="输出目录（默认: outputs）")
    parser.add_argument("--no_save",   action="store_true",
                        help="不保存文件，直接弹窗显示图表")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)

    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir.resolve()}")
        sys.exit(1)

    # Step 1: 收集图像信息
    df = collect_image_stats(data_dir)
    if df.empty:
        print("[ERROR] 未找到任何可读取的图像文件。")
        sys.exit(1)

    # Step 2: 打印统计
    print_statistics(df)

    # Step 3: 推荐分辨率
    rec = recommend_resolution(df)

    # Step 4: 可视化 & 保存
    if args.no_save:
        plot_distributions(df, rec, save_path=None)
    else:
        plot_path = save_dir / "image_size_distribution.png"
        plot_distributions(df, rec, save_path=plot_path)
        save_csv(df, save_dir)

    print(f"\n{'═'*58}")
    print(f"  分析完成！")
    print(f"  → 建议输入分辨率 :  {rec['shape_note']}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
