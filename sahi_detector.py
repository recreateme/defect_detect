#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slicing Aided Hyper Inference (SAHI) 大图检测 + 缺陷分类流水线
================================================================================

对 5120×5120（或其他大分辨率）图像执行：
  1. 切片目标检测（滑动窗口 → YOLO 批量推理）
  2. 后处理链：iou_nms → IoS 包含抑制 → 面积/长宽比过滤 → 边缘剔除
  3. 裁剪每个检测目标 → 缺陷分类引擎批量分类
  4. 保存裁剪图、可视化图、JSON 结果与统计信息

设计参考: data_process/pipeline/phase3_inference.py（产品级 24×24 网格推理）
本模块针对**单张大图**简化：无需网格坐标、跨图 NMS。

依赖:
  - ultralytics  (YOLO 推理，**仅开发环境**)
  - opencv-python (图像 I/O 与可视化)
  - numpy

使用示例:
  from sahi_detector import SahiDetector, SahiPipeline
  detector = SahiDetector("best.pt", device="auto")
  pipeline = SahiPipeline(detector, classifier_engine, output_dir="output")
  stats = pipeline.process_image("5120x5120.jpg")
"""

from __future__ import annotations

import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# 值对象
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """
    单个检测框（原图坐标系）。

    检测阶段填充 x1/y1/x2/y2/conf；
    分类阶段填充 defect_class / defect_conf / all_scores。
    """
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    cls_name: str = ""
    # ── 分类结果（SahiPipeline.process_image 中填充）──────────────
    defect_class: str = ""
    defect_conf: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    crop_path: str = ""

    @property
    def cx(self) -> float: return (self.x1 + self.x2) * 0.5
    @property
    def cy(self) -> float: return (self.y1 + self.y2) * 0.5
    @property
    def w(self) -> float: return self.x2 - self.x1
    @property
    def h(self) -> float: return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 1), "y1": round(self.y1, 1),
            "x2": round(self.x2, 1), "y2": round(self.y2, 1),
            "cx": round(self.cx, 1),  "cy": round(self.cy, 1),
            "w":  round(self.w, 1),   "h":  round(self.h, 1),
            "det_conf": round(self.conf, 4),
            "defect_class": self.defect_class,
            "defect_conf": round(self.defect_conf, 4),
            "crop_path": self.crop_path,
        }


# ════════════════════════════════════════════════════════════════════════
# 纯函数
# ════════════════════════════════════════════════════════════════════════

def slice_positions(H: int, W: int, size: int, step: int) -> List[Tuple[int, int, int, int]]:
    """
    生成滑动窗口切片坐标 (x1, y1, x2, y2)。

    末尾切片贴边处理（min(pos, dim-size)），确保无黑边、全覆盖。
    当图像小于切片尺寸时，返回单个覆盖全图的切片。

    示例: H=W=5120, size=1280, step=1024 → 每方向 5 片，共 25 片
    """
    if H <= size and W <= size:
        return [(0, 0, W, H)]

    coords: List[Tuple[int, int, int, int]] = []
    y = 0
    while True:
        y1 = min(y, H - size) if H > size else 0
        x = 0
        while True:
            x1 = min(x, W - size) if W > size else 0
            x2 = min(x1 + size, W)
            y2 = min(y1 + size, H)
            coords.append((x1, y1, x2, y2))
            if x1 + size >= W:
                break
            x += step
        if y1 + size >= H:
            break
        y += step
    return coords


def iou_nms(dets: List[Detection], threshold: float = 0.50) -> List[Detection]:
    """
    标准 IoU-NMS — 切片重叠区域对同一目标的重复检测去重。

    按置信度降序保留，IoU 超阈值的框被压制。
    复杂度 O(N²)，N = 单图检测数（通常 < 500）。
    """
    if len(dets) <= 1:
        return list(dets)

    boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    scores = np.array([d.conf for d in dets], dtype=np.float32)
    areas  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou < threshold]

    return [dets[i] for i in keep]


def crop_with_padding(
    img: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    padding: int = 15,
) -> np.ndarray:
    """
    从原图裁剪检测区域，四周扩展 padding 像素（边界自动裁剪到图像范围内）。

    padding 给分类模型提供少量上下文，但不会大到包含相邻目标。
    """
    H, W = img.shape[:2]
    cx1 = max(0, int(x1) - padding)
    cy1 = max(0, int(y1) - padding)
    cx2 = min(W, int(x2) + padding)
    cy2 = min(H, int(y2) + padding)

    # 确保有效尺寸
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(W, int(x2)), min(H, int(y2))

    return img[cy1:cy2, cx1:cx2]


# ════════════════════════════════════════════════════════════════════════
# 检测后处理 — IoS 包含抑制 / 面积过滤 / 边缘剔除
# ════════════════════════════════════════════════════════════════════════

def _pairwise_ios(dets: List[Detection]) -> np.ndarray:
    """
    计算两两框的 IoS（Intersection over Smaller）矩阵。

    IoS = 交集面积 / min(两框面积)。当小框几乎被大框包含时 IoS→1，
    而标准 IoU 会因并集偏大而偏小，故 IoS 更能捕捉「大框套小框」的冗余。
    """
    n = len(dets)
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    ios = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        smaller = np.minimum(areas[i], areas) + 1e-7
        ios[i] = inter / smaller
    return ios


def suppress_contained_boxes(
    dets: List[Detection],
    ios_thresh: float = 0.65,
) -> Tuple[List[Detection], List[Detection]]:
    """
    IoS 包含抑制 — 去除「被另一框大部分包含」的冗余小框。

    对任意两框，若 IoS >= ios_thresh，判为包含型冗余，丢弃其中
    置信度较低者（置信度相同则丢弃面积较小者）。

    Returns:
        (保留列表, 被剔除列表)
    """
    n = len(dets)
    if n <= 1 or ios_thresh >= 1.0:
        return list(dets), []

    ios = _pairwise_ios(dets)
    areas = [d.w * d.h for d in dets]
    removed = [False] * n

    for i in range(n):
        if removed[i]:
            continue
        for j in range(i + 1, n):
            if removed[j]:
                continue
            if ios[i, j] < ios_thresh:
                continue
            # 保留置信度高者；置信度相等保留面积大者
            if dets[i].conf > dets[j].conf or (
                dets[i].conf == dets[j].conf and areas[i] >= areas[j]
            ):
                removed[j] = True
            else:
                removed[i] = True
                break

    kept = [d for k, d in enumerate(dets) if not removed[k]]
    dropped = [d for k, d in enumerate(dets) if removed[k]]
    return kept, dropped


def filter_small_area(
    dets: List[Detection],
    min_area_ratio: float = 0.45,
    min_abs_area: Optional[float] = None,
) -> Tuple[List[Detection], List[Detection]]:
    """
    面积过滤 — 剔除面积明显偏小的孤立误检框。

    基准取本图检测框面积的中位数（对误检更鲁棒，避免被均值拉偏）：
      · area < 中位数 * min_area_ratio  → 剔除
      · area < min_abs_area（若提供）    → 剔除

    Returns:
        (保留列表, 被剔除列表)
    """
    if not dets:
        return [], []

    areas = np.array([d.w * d.h for d in dets], dtype=np.float32)
    thresholds: List[float] = []
    if min_area_ratio and min_area_ratio > 0:
        thresholds.append(float(np.median(areas)) * min_area_ratio)
    if min_abs_area and min_abs_area > 0:
        thresholds.append(float(min_abs_area))

    if not thresholds:
        return list(dets), []

    thr = max(thresholds)
    kept = [d for d, a in zip(dets, areas) if a >= thr]
    dropped = [d for d, a in zip(dets, areas) if a < thr]
    return kept, dropped


def filter_aspect_ratio(
    dets: List[Detection],
    max_aspect_ratio: float = 2.5,
) -> Tuple[List[Detection], List[Detection]]:
    """
    长宽比过滤 — 剔除过于细长的异常检测框。

    长宽比 = max(w, h) / min(w, h)（恒 ≥ 1）。钻石目标应接近正方形，
    比例过大常见于切片边界截断或粘连误检。max_aspect_ratio <= 0 时关闭。

    Returns:
        (保留列表, 被剔除列表)
    """
    if not dets or not max_aspect_ratio or max_aspect_ratio <= 0:
        return list(dets), []

    kept: List[Detection] = []
    dropped: List[Detection] = []
    for d in dets:
        w, h = d.w, d.h
        if w <= 0 or h <= 0:
            dropped.append(d)
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > max_aspect_ratio:
            dropped.append(d)
        else:
            kept.append(d)
    return kept, dropped


def filter_edge_boxes(
    dets: List[Detection],
    W: int,
    H: int,
    edge_margin_px: float = 20,
    drop_touching: bool = True,
) -> Tuple[List[Detection], List[Detection]]:
    """
    边缘剔除 — 去除贴近图像边界 / 被边界截断的不完整目标。

    这些边缘半颗钻石属分类模型的分布外输入，易把正常钻石误判为缺陷，
    故在缺陷统计前剔除（调用方可单列 edge_skipped 以便追溯）。

    edge_margin_px：距图像四边的像素边距带；框任一边落入该带内即剔除。
    设为 0 且 drop_touching=True 时仍按 0 边距（仅触边剔除）。

    Returns:
        (保留列表, 被剔除列表)
    """
    if not dets:
        return [], []

    margin = max(0.0, float(edge_margin_px))
    if margin <= 0 and not drop_touching:
        return list(dets), []

    kept: List[Detection] = []
    dropped: List[Detection] = []
    for d in dets:
        near_edge = (
            d.x1 <= margin
            or d.y1 <= margin
            or d.x2 >= W - margin
            or d.y2 >= H - margin
        )
        if near_edge:
            dropped.append(d)
        else:
            kept.append(d)
    return kept, dropped


# ════════════════════════════════════════════════════════════════════════
# 设备解析 — 避免 sm_120 等新 GPU 被 torch.cuda.is_available() 误报可用
# ════════════════════════════════════════════════════════════════════════

def resolve_yolo_device(requested: str) -> Tuple[str, Optional[str]]:
    """
    解析 YOLO 推理设备，并在 auto 模式下对不兼容 GPU 自动回退 CPU。

    torch.cuda.is_available() 在 RTX 50 系列（sm_120）等新型号上可能为 True，
    但实际执行 kernel 会报 cudaErrorUnknown；此处用 get_arch_list 与算力比对规避。

    Returns:
        (device, warning_or_none)
    """
    import torch

    req = (requested or "auto").strip().lower()
    want_cuda = req in ("auto", "cuda", "cuda:0") or req.startswith("cuda")

    if not want_cuda:
        return "cpu", None

    if not torch.cuda.is_available():
        msg = "CUDA 不可用"
        if req == "auto":
            return "cpu", f"{msg}，已自动使用 CPU 推理"
        raise RuntimeError(f"{msg}。请在「设置 → 切片推理配置」中将设备改为 cpu。")

    try:
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        archs = getattr(torch.cuda, "get_arch_list", lambda: [])() or []
        if archs and sm not in archs:
            detail = (
                f"GPU {name}（{sm}）不受当前 PyTorch {torch.__version__} 支持"
                f"（已编译: {', '.join(archs)}）"
            )
            if req == "auto":
                return "cpu", (
                    f"{detail}。"
                    "已自动回退到 CPU 推理（速度较慢）。"
                    "如需 GPU 加速，请安装支持 sm_120 的 PyTorch nightly/cu128+。"
                )
            raise RuntimeError(
                f"{detail}。"
                "请在「设置 → 切片推理配置」中将设备改为 auto 或 cpu，"
                "或升级 PyTorch 至支持 Blackwell 架构的版本。"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        if req == "auto":
            return "cpu", f"CUDA 检测失败（{exc}），已自动使用 CPU 推理"
        raise RuntimeError(f"CUDA 检测失败: {exc}") from exc

    device = req if req.startswith("cuda") else "cuda:0"
    return device, None


# ════════════════════════════════════════════════════════════════════════
# SAHI 检测器 — 封装 YOLO 切片推理
# ════════════════════════════════════════════════════════════════════════

class SahiDetector:
    """
    Slicing Aided Hyper Inference 检测器。

    流程:
      1. slice_positions() 将大图切为 size×size 子图（带 overlap 重叠）
      2. YOLO 按 batch_size 分批推理
      3. 检测框坐标 + 切片偏移 → 原图坐标
      4. iou_nms() 去除重叠区域的重复检测

    Args:
        model_path:     YOLO 权重路径（.pt）
        device:          'auto' / 'cuda:0' / 'cpu'
        slice_size:      切片边长（像素），5120 图推荐 1280
        overlap_ratio:   切片重叠比例 (0–1)，0.20 → 步长 = size×0.80
        conf:            检测置信度阈值
        batch_size:      每批推理的切片数（按显存调整）
        nms_iou:         IoU-NMS 阈值（重叠去重）

    后处理（检测后串联，去误检 + 剔除边缘不完整目标）:
        ios_thresh:        IoS 包含抑制阈值；<=0 或 >=1 关闭
        min_area_ratio:    面积过滤：相对本图中位面积的最小比例；<=0 关闭
        min_abs_area:      面积绝对下限（像素²，可选）
        max_aspect_ratio:  长宽比上限 max(w,h)/min(w,h)；<=0 关闭
        edge_filter:       是否启用边缘剔除
        edge_margin_px:    边缘边距（像素）；框触边距带内即剔除
        drop_touching:     触边框是否剔除
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        slice_size: int = 1280,
        overlap_ratio: float = 0.20,
        conf: float = 0.35,
        batch_size: int = 8,
        nms_iou: float = 0.50,
        ios_thresh: float = 0.65,
        min_area_ratio: float = 0.45,
        min_abs_area: Optional[float] = None,
        max_aspect_ratio: float = 2.5,
        edge_filter: bool = True,
        edge_margin_px: float = 20,
        drop_touching: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf = conf
        self.batch_size = batch_size
        self.nms_iou = nms_iou
        # ── 后处理参数 ──
        self.ios_thresh = ios_thresh
        self.min_area_ratio = min_area_ratio
        self.min_abs_area = min_abs_area
        self.max_aspect_ratio = max_aspect_ratio
        self.edge_filter = edge_filter
        self.edge_margin_px = edge_margin_px
        self.drop_touching = drop_touching
        # 最近一次 detect() 的后处理剔除统计（供流水线读取写入 stats）
        self.last_skip_stats: Dict[str, int] = {
            "contained_skipped": 0,
            "small_skipped": 0,
            "aspect_skipped": 0,
            "edge_skipped": 0,
        }
        self._model = None
        self._loaded = False

    @property
    def step(self) -> int:
        """滑动步长 = slice_size × (1 - overlap_ratio)"""
        return max(1, int(self.slice_size * (1.0 - self.overlap_ratio)))

    def load(self) -> str:
        """加载 YOLO 模型，返回状态描述字符串。"""
        from ultralytics import YOLO

        device, warn = resolve_yolo_device(self.device)
        self.device = device

        self._model = YOLO(self.model_path)
        try:
            self._model.to(self.device)
        except Exception as exc:
            if self.device != "cpu":
                self.device = "cpu"
                self._model.to("cpu")
                warn = f"YOLO 在 GPU 上加载失败（{exc}），已回退到 CPU"
            else:
                raise RuntimeError(f"YOLO 加载失败: {exc}") from exc

        self._loaded = True
        label = "GPU" if "cuda" in self.device else "CPU"
        msg = f"YOLO 加载成功 [{label}]  {self.model_path}"
        if warn:
            msg = f"[提示] {warn}\n{msg}"
        return msg

    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, img: np.ndarray) -> List[Detection]:
        """
        对单张大图执行 SAHI 切片检测 + 后处理。

        后处理顺序：iou_nms → suppress_contained_boxes → filter_small_area
        → filter_aspect_ratio → filter_edge_boxes（各步可由参数关闭）。

        Args:
            img: BGR 图像 (H, W, 3)

        Returns:
            原图坐标系下保留的检测框列表
        """
        if not self._loaded:
            raise RuntimeError("YOLO 模型未加载，请先调用 load()")

        H, W = img.shape[:2]
        coords = slice_positions(H, W, self.slice_size, self.step)
        tiles  = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in coords]

        # ── 分批推理 ──────────────────────────────────────────────
        raw: List[Detection] = []
        for i in range(0, len(tiles), self.batch_size):
            batch_tiles  = tiles[i: i + self.batch_size]
            batch_coords = coords[i: i + self.batch_size]
            batch_results = self._infer_batch(batch_tiles)

            for (ox1, oy1, ox2, oy2), tile_dets in zip(batch_coords, batch_results):
                for tx1, ty1, tx2, ty2, conf, cls_id, cls_name in tile_dets:
                    # 切片坐标 → 原图坐标
                    raw.append(Detection(
                        x1=tx1 + ox1, y1=ty1 + oy1,
                        x2=tx2 + ox1, y2=ty2 + oy1,
                        conf=conf,
                        cls_id=cls_id,
                        cls_name=cls_name,
                    ))

        # ── IoU-NMS 去重 ──────────────────────────────────────────
        result = iou_nms(raw, self.nms_iou)
        after_nms = len(result)

        # ── 后处理：IoS → 面积 → 长宽比 → 边缘剔除 ────────────────
        contained_n = small_n = aspect_n = edge_n = 0

        if self.ios_thresh and 0.0 < self.ios_thresh < 1.0:
            result, dropped = suppress_contained_boxes(result, self.ios_thresh)
            contained_n = len(dropped)

        if (self.min_area_ratio and self.min_area_ratio > 0) or (
            self.min_abs_area and self.min_abs_area > 0
        ):
            result, dropped = filter_small_area(
                result, self.min_area_ratio or 0.0, self.min_abs_area,
            )
            small_n = len(dropped)

        if self.max_aspect_ratio and self.max_aspect_ratio > 0:
            result, dropped = filter_aspect_ratio(result, self.max_aspect_ratio)
            aspect_n = len(dropped)

        if self.edge_filter:
            result, dropped = filter_edge_boxes(
                result, W, H,
                edge_margin_px=self.edge_margin_px,
                drop_touching=self.drop_touching,
            )
            edge_n = len(dropped)

        self.last_skip_stats = {
            "contained_skipped": contained_n,
            "small_skipped": small_n,
            "aspect_skipped": aspect_n,
            "edge_skipped": edge_n,
        }
        logger.debug(
            f"SAHI detect: 切片 {len(coords)} | 原始 {len(raw)} | "
            f"NMS 后 {after_nms} | 去包含 -{contained_n} | "
            f"去小框 -{small_n} | 去长宽比 -{aspect_n} | "
            f"去边缘 -{edge_n} | 最终 {len(result)}"
        )
        return result

    def _infer_batch(
        self, tiles: List[np.ndarray],
    ) -> List[List[Tuple[float, float, float, float, float, int, str]]]:
        """
        YOLO 批量推理，返回每个切片的检测列表。

        单个检测: (x1, y1, x2, y2, conf, cls_id, cls_name)
        """
        results = self._model(
            tiles,
            conf=self.conf,
            iou=0.40,
            max_det=500,
            imgsz=self.slice_size,
            device=self.device,
            verbose=False,
        )
        out: List[List[Tuple[float, float, float, float, float, int, str]]] = []
        for r in results:
            dets: List[Tuple[float, float, float, float, float, int, str]] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy   = r.boxes.xyxy.cpu().numpy()
                confs  = r.boxes.conf.cpu().numpy()
                clsids = r.boxes.cls.cpu().numpy().astype(int)
                names  = r.names
                for i in range(len(confs)):
                    cid = int(clsids[i])
                    dets.append((
                        float(xyxy[i, 0]), float(xyxy[i, 1]),
                        float(xyxy[i, 2]), float(xyxy[i, 3]),
                        float(confs[i]),
                        cid, names.get(cid, str(cid)),
                    ))
            out.append(dets)
        return out


# ════════════════════════════════════════════════════════════════════════
# 可视化 — 全分辨率双图 + PIL 中文标注
# ════════════════════════════════════════════════════════════════════════

# 输出文件名（保存在每张图的输出目录下）
VIS_DETECTION_NAME = "visualization_detection.jpg"   # 仅检测框，统一颜色
VIS_CLASSIFIED_NAME = "visualization_classified.jpg"   # 按缺陷类别着色 + 中文标签

# 缺陷类别可视化配色（BGR 框/标签底 + RGB 字体），保证类别可区分且文字可读
class DefectVisStyle(NamedTuple):
    box_bgr: Tuple[int, int, int]       # 锚框描边色
    label_bg_bgr: Tuple[int, int, int]   # 标签背景色
    text_rgb: Tuple[int, int, int]       # 标签字体色


_DEFAULT_VIS = DefectVisStyle(
    box_bgr=(128, 128, 128),
    label_bg_bgr=(70, 70, 70),
    text_rgb=(255, 255, 255),
)

# 五类缺陷：框用高饱和色便于定位，标签用更深底色 + 高对比字体（避免浅黄底+白字）
DEFECT_VIS_STYLES: Dict[str, DefectVisStyle] = {
    "局部破损": DefectVisStyle(
        box_bgr=(0, 140, 255),        # 亮橙框
        label_bg_bgr=(0, 90, 190),    # 深橙底
        text_rgb=(255, 255, 255),     # 白字
    ),
    "断钻": DefectVisStyle(
        box_bgr=(50, 50, 240),        # 亮红框
        label_bg_bgr=(25, 25, 165),   # 深红底
        text_rgb=(255, 255, 255),     # 白字
    ),
    "棱边朝上": DefectVisStyle(
        box_bgr=(220, 170, 0),        # 亮青框
        label_bg_bgr=(155, 105, 0),   # 深青底
        text_rgb=(255, 255, 255),     # 白字
    ),
    "点朝上": DefectVisStyle(
        box_bgr=(210, 0, 210),        # 亮紫框
        label_bg_bgr=(135, 0, 135),   # 深紫底
        text_rgb=(255, 255, 255),     # 白字
    ),
    "面朝上": DefectVisStyle(
        box_bgr=(0, 230, 255),        # 亮黄框（边框醒目）
        label_bg_bgr=(35, 35, 35),    # 深灰底 — 不用黄底，避免白字难辨
        text_rgb=(255, 228, 100),     # 浅黄字，深底上清晰
    ),
    "ERROR": DefectVisStyle(
        box_bgr=(80, 80, 80),
        label_bg_bgr=(40, 40, 40),
        text_rgb=(255, 180, 180),
    ),
    "未分类": DefectVisStyle(
        box_bgr=(160, 160, 160),
        label_bg_bgr=(95, 95, 95),
        text_rgb=(255, 255, 255),
    ),
}


def _get_vis_style(defect_class: str) -> DefectVisStyle:
    return DEFECT_VIS_STYLES.get(defect_class, _DEFAULT_VIS)

# 纯检测可视化：所有框统一颜色（BGR 绿）
DETECTION_ONLY_COLOR_BGR = (0, 255, 0)

# Windows / Linux / macOS 常见中文字体路径（按优先级）
_CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)


def _compute_vis_style(h: int, w: int) -> Tuple[int, int]:
    """按图像边长自适应线宽与字号（5120 图 → 线宽 8、字号 64）。"""
    side = max(h, w)
    line_w = max(2, side // 640)
    font_size = max(16, side // 80)
    return line_w, font_size


def _resolve_cjk_font(size: int):
    """加载支持中文的 TrueType 字体；失败时回退默认字体。"""
    from PIL import ImageFont

    for path in _CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    logger.warning("未找到中文字体，可视化标签可能无法正确显示中文")
    return ImageFont.load_default()


def _bgr_to_rgb(color_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = color_bgr
    return (r, g, b)


def _draw_cjk_label(
    draw,
    x: int,
    y: int,
    text: str,
    font,
    bg_bgr: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    line_w: int,
) -> None:
    """在 PIL ImageDraw 上绘制带背景的中文标签（背景色与字体色独立配置）。"""
    bbox = draw.textbbox((x, y), text, font=font)
    pad = max(2, line_w // 2)
    bg_rgb = _bgr_to_rgb(bg_bgr)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg_rgb,
    )
    draw.text((x, y), text, font=font, fill=text_rgb)


def draw_detection_boxes(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """
    全分辨率检测框可视化 — 所有目标统一颜色，不含中文标注。

    用于快速查看 SAHI 检测覆盖范围，不受分类结果影响。
    """
    vis = img.copy()
    if not dets:
        return vis

    line_w, _ = _compute_vis_style(img.shape[0], img.shape[1])
    color = DETECTION_ONLY_COLOR_BGR
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_w)
    return vis


def draw_classified_detections(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """
    全分辨率分类可视化 — 按缺陷类别着色，PIL 绘制中文类别名。

    OpenCV putText 不支持中文，故框用 cv2 绘制、文字用 PIL + 系统中文字体。
    """
    vis = img.copy()
    if not dets:
        return vis

    h, w = vis.shape[:2]
    line_w, font_size = _compute_vis_style(h, w)

    labels: List[Tuple[int, int, str, DefectVisStyle]] = []
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        style = _get_vis_style(d.defect_class or "")
        cv2.rectangle(vis, (x1, y1), (x2, y2), style.box_bgr, line_w)

        if d.defect_class and d.defect_class != "ERROR":
            label = f"{d.defect_class} {d.defect_conf:.2f}"
        elif d.defect_class == "ERROR":
            label = f"分类失败 {d.defect_conf:.2f}"
        else:
            label = f"检测 {d.conf:.2f}"
        ty = y1 - font_size - 6 if y1 > font_size + 10 else y2 + 4
        labels.append((x1, ty, label, style))

    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _resolve_cjk_font(font_size)
    for tx, ty, text, style in labels:
        _draw_cjk_label(
            draw, tx, ty, text, font,
            style.label_bg_bgr, style.text_rgb, line_w,
        )

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def save_visualizations(
    out_dir: Path,
    img: np.ndarray,
    dets: List[Detection],
    *,
    jpeg_quality: int = 95,
) -> Tuple[str, str]:
    """
    保存两张与原图同分辨率的 JPEG 可视化图。

    Returns:
        (检测框图路径, 分类着色图路径)
    """
    det_path = out_dir / VIS_DETECTION_NAME
    cls_path = out_dir / VIS_CLASSIFIED_NAME
    encode = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    vis_det = draw_detection_boxes(img, dets)
    vis_cls = draw_classified_detections(img, dets) if dets else img.copy()

    cv2.imwrite(str(det_path), vis_det, encode)
    cv2.imwrite(str(cls_path), vis_cls, encode)
    return str(det_path), str(cls_path)


# ════════════════════════════════════════════════════════════════════════
# 流水线 — 检测 + 分类 + 保存 + 统计
# ════════════════════════════════════════════════════════════════════════

class SahiPipeline:
    """
    SAHI 检测 + 缺陷分类完整流水线。

    流程 (process_image):
      1. 读取大图 → SahiDetector.detect() → 检测框列表
      2. 裁剪每个检测目标 → 保存到 crops/ 目录
      3. InferenceEngine.predict_batch() 批量分类所有裁剪图
      4. 绘制两张全分辨率可视化图（检测框 / 分类着色）→ 保存 JPEG
      5. 保存 result.json + statistics.json
      6. 返回统计字典

    Args:
        detector:    SahiDetector 实例（须已 load）
        classifier:  InferenceEngine 实例（须已 load）
        output_dir:  输出根目录
        crop_padding: 裁剪时四周扩展像素数
    """

    def __init__(
        self,
        detector: SahiDetector,
        classifier: Any,
        output_dir: str,
        crop_padding: int = 15,
    ):
        self.detector = detector
        self.classifier = classifier
        self.output_dir = Path(output_dir)
        self.crop_padding = crop_padding

    def process_image(
        self,
        img_path: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """
        处理单张大图：检测(+后处理) → 裁剪 → 分类 → 保存 → 统计。

        Returns:
            统计字典（含 total_diamonds、defect_counts、各类 skipped、耗时与 output_dir）
        """
        def _log(msg: str):
            if log_cb:
                log_cb(msg)
            else:
                logger.info(msg)

        path = Path(img_path)
        stem = path.stem
        img_out_dir = self.output_dir / stem
        crops_dir = img_out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        if should_stop and should_stop():
            return self._empty_stats(stem)

        # ── 1. 读取大图 ──────────────────────────────────────────
        _log(f"读取图像: {path.name}")
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"无法读取图像: {path}")
        H, W = img.shape[:2]
        _log(f"图像尺寸: {W}×{H}")

        if progress_cb:
            progress_cb(0, 4)

        # ── 2. SAHI 检测 ──────────────────────────────────────────
        t0 = time.time()
        _log(f"SAHI 检测: 切片 {self.detector.slice_size}px, "
             f"重叠 {self.detector.overlap_ratio:.0%}, 置信度 {self.detector.conf}")
        dets = self.detector.detect(img)
        det_time = time.time() - t0
        skip_stats = dict(self.detector.last_skip_stats)
        skipped_total = sum(skip_stats.values())
        if skipped_total:
            _log(
                f"后处理剔除: 包含冗余 {skip_stats.get('contained_skipped', 0)} · "
                f"小面积 {skip_stats.get('small_skipped', 0)} · "
                f"长宽比 {skip_stats.get('aspect_skipped', 0)} · "
                f"边缘 {skip_stats.get('edge_skipped', 0)}"
            )
        _log(f"检测完成: 保留 {len(dets)} 个目标（剔除 {skipped_total}）, 耗时 {det_time:.2f}s")

        if progress_cb:
            progress_cb(1, 4)

        if not dets:
            _log("未检测到任何目标")
            self._save_results(img_out_dir, stem, [], img, det_time, 0.0, skip_stats)
            return self._stats(stem, [], det_time, 0.0, str(img_out_dir), skip_stats)

        if should_stop and should_stop():
            return self._empty_stats(stem)

        # ── 3. 裁剪 + 保存裁剪图 ──────────────────────────────────
        _log(f"裁剪 {len(dets)} 个目标...")
        crop_paths: List[str] = []
        for i, d in enumerate(dets):
            if should_stop and should_stop():
                return self._empty_stats(stem)
            crop = crop_with_padding(img, d.x1, d.y1, d.x2, d.y2, self.crop_padding)
            crop_path = str(crops_dir / f"{i + 1:04d}.jpg")
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_paths.append(crop_path)
            d.crop_path = crop_path

        if progress_cb:
            progress_cb(2, 4)

        # ── 4. 批量分类 ───────────────────────────────────────────
        t1 = time.time()
        _log(f"批量分类 {len(crop_paths)} 个裁剪图...")
        cls_results = self.classifier.predict_batch(crop_paths)
        cls_time = time.time() - t1

        for d, r in zip(dets, cls_results):
            d.defect_class = r.get("class", "ERROR")
            d.defect_conf = r.get("confidence", 0.0)
            d.all_scores = r.get("all_scores", {})

        # 按分类结果重命名裁剪图（0001.jpg → 0001_局部破损.jpg）
        for i, d in enumerate(dets):
            old_path = crops_dir / f"{i + 1:04d}.jpg"
            if old_path.exists() and d.defect_class and d.defect_class != "ERROR":
                new_name = f"{i + 1:04d}_{d.defect_class}.jpg"
                new_path = crops_dir / new_name
                try:
                    old_path.rename(new_path)
                    d.crop_path = str(new_path)
                except OSError:
                    pass  # 重命名失败时保留原名

        if progress_cb:
            progress_cb(3, 4)

        # ── 5. 可视化 + 保存结果 ──────────────────────────────────
        _log("生成可视化与统计...")
        self._save_results(img_out_dir, stem, dets, img, det_time, cls_time, skip_stats)

        stats = self._stats(stem, dets, det_time, cls_time, str(img_out_dir), skip_stats)
        _log(f"统计: 钻石 {stats['total_diamonds']} 个 | "
             + " | ".join(f"{k}:{v}" for k, v in stats["defect_counts"].items())
             + f" | 总耗时 {stats['total_time_s']:.2f}s")

        if progress_cb:
            progress_cb(4, 4)

        return stats

    def process_images(
        self,
        img_paths: List[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[dict]:
        """批量处理多张大图，返回每张的统计字典列表。"""
        all_stats = []
        total = len(img_paths)
        for i, p in enumerate(img_paths):
            if should_stop and should_stop():
                break
            if log_cb:
                log_cb(f"\n━━━ [{i + 1}/{total}] {Path(p).name} ━━━")
            try:
                stats = self.process_image(p, log_cb=log_cb, should_stop=should_stop)
                all_stats.append(stats)
            except Exception as exc:
                if log_cb:
                    log_cb(f"处理失败: {exc}")
                all_stats.append({
                    "image": Path(p).name,
                    "error": str(exc),
                    "total_diamonds": 0,
                    "defect_counts": {},
                })
            if progress_cb:
                progress_cb(i + 1, total)

        # 保存汇总 CSV
        if all_stats:
            self._save_summary_csv(all_stats)
        return all_stats

    # ── 内部方法 ──────────────────────────────────────────────────

    def _save_results(
        self,
        out_dir: Path,
        stem: str,
        dets: List[Detection],
        img: np.ndarray,
        det_time: float,
        cls_time: float,
        skip_stats: Optional[Dict[str, int]] = None,
    ) -> None:
        """保存 result.json、statistics.json、两张全分辨率可视化图。"""
        det_vis, cls_vis = save_visualizations(out_dir, img, dets)

        result = {
            "image": f"{stem}.jpg",
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "detection_time_s": round(det_time, 3),
            "classification_time_s": round(cls_time, 3),
            "total_detections": len(dets),
            "postprocess_skipped": dict(skip_stats or {}),
            "visualization_detection": det_vis,
            "visualization_classified": cls_vis,
            "detections": [d.to_dict() for d in dets],
        }
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        stats = self._stats(stem, dets, det_time, cls_time, str(out_dir), skip_stats)
        with open(out_dir / "statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def _stats(
        self,
        stem: str,
        dets: List[Detection],
        det_time: float,
        cls_time: float,
        out_dir: str,
        skip_stats: Optional[Dict[str, int]] = None,
    ) -> dict:
        """构建统计字典。"""
        defect_counts: Dict[str, int] = {}
        for d in dets:
            cls = d.defect_class or "未分类"
            defect_counts[cls] = defect_counts.get(cls, 0) + 1
        skip = skip_stats or {}
        return {
            "image": f"{stem}.jpg",
            "total_diamonds": len(dets),
            "defect_counts": defect_counts,
            "contained_skipped": int(skip.get("contained_skipped", 0)),
            "small_skipped": int(skip.get("small_skipped", 0)),
            "aspect_skipped": int(skip.get("aspect_skipped", 0)),
            "edge_skipped": int(skip.get("edge_skipped", 0)),
            "detection_time_s": round(det_time, 3),
            "classification_time_s": round(cls_time, 3),
            "total_time_s": round(det_time + cls_time, 3),
            "output_dir": out_dir,
        }

    def _empty_stats(self, stem: str) -> dict:
        return {
            "image": f"{stem}.jpg",
            "total_diamonds": 0,
            "defect_counts": {},
            "contained_skipped": 0,
            "small_skipped": 0,
            "aspect_skipped": 0,
            "edge_skipped": 0,
            "detection_time_s": 0.0,
            "classification_time_s": 0.0,
            "total_time_s": 0.0,
            "output_dir": "",
        }

    def _save_summary_csv(self, all_stats: List[dict]) -> None:
        """保存所有图像的汇总 CSV。"""
        import csv
        csv_path = self.output_dir / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "图像", "钻石数", "缺陷类别分布",
                "去包含冗余", "去小面积", "去长宽比", "去边缘",
                "检测耗时(s)", "分类耗时(s)", "总耗时(s)",
            ])
            for s in all_stats:
                dist = " | ".join(f"{k}:{v}" for k, v in s.get("defect_counts", {}).items())
                writer.writerow([
                    s.get("image", ""),
                    s.get("total_diamonds", 0),
                    dist,
                    s.get("contained_skipped", 0),
                    s.get("small_skipped", 0),
                    s.get("aspect_skipped", 0),
                    s.get("edge_skipped", 0),
                    s.get("detection_time_s", 0),
                    s.get("classification_time_s", 0),
                    s.get("total_time_s", 0),
                ])
        logger.info(f"汇总 CSV: {csv_path}")


# ════════════════════════════════════════════════════════════════════════
# 依赖检查
# ════════════════════════════════════════════════════════════════════════

def check_ultralytics() -> Tuple[bool, str]:
    """检查 ultralytics 是否可用，返回 (是否可用, 状态描述)。"""
    try:
        import ultralytics  # noqa: F401
        return True, "ultralytics 已安装"
    except ImportError:
        return False, (
            "未安装 ultralytics（YOLO），大图检测功能不可用。\n"
            "安装: pip install ultralytics"
        )


def check_cv2() -> Tuple[bool, str]:
    """检查 opencv-python 是否可用。"""
    try:
        import cv2  # noqa: F401
        return True, "opencv-python 已安装"
    except ImportError:
        return False, (
            "未安装 opencv-python，大图检测功能不可用。\n"
            "安装: pip install opencv-python"
        )
