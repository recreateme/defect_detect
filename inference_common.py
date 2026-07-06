#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理公共模块 — 开发版 (inference_engine) 与机台版 (inference_engine_onnx) 共用。

集中放置：
  · 元数据读取（类别 / img_size / 阈值）
  · Softmax 与逐类阈值决策
  · run_batch_predict — 通用批量推理循环
  · logits_row_to_result / build_result_dict — 单张/批量统一结果结构
  · ImageNet 归一化常量

避免两处引擎逻辑漂移；修改预处理/阈值/批量逻辑时只改此处。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# 与 torchvision / train.py 一致的 ImageNet 统计量（HWC 布局，float32）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_model_meta(
    pt_path: Optional[str],
    onnx_path: Optional[str],
    *,
    load_pt_meta=None,
) -> Tuple[List[str], int]:
    """
    从模型目录读取类别与输入尺寸。

    优先级：
      1. load_pt_meta 回调（PyTorch checkpoint，仅开发版）
      2. class_map.json → 类别
      3. train_config.json → img_size（及类别兜底）

    注意：存在 class_map.json 时仍必须读 train_config.json 的 img_size，
    否则 ONNX 推理会错误使用默认 224。
    """
    classes: List[str] = []
    img_size = 224

    if load_pt_meta is not None:
        classes, img_size = load_pt_meta(pt_path)

    if not classes:
        for base in filter(None, [pt_path, onnx_path]):
            map_path = Path(base).parent / "class_map.json"
            if map_path.exists():
                with open(map_path, "r", encoding="utf-8") as f:
                    classes = json.load(f).get("classes", [])
                break

    for base in filter(None, [pt_path, onnx_path]):
        cfg_path = Path(base).parent / "train_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            img_size = cfg.get("img_size", img_size)
            if not classes:
                classes = cfg.get("classes", [])
            break

    if not classes:
        raise FileNotFoundError(
            "未能获取类别信息。请确保 class_map.json / train_config.json "
            "与模型在同一 checkpoints 目录。"
        )
    return classes, int(img_size)


def read_class_thresholds(
    pt_path: Optional[str],
    onnx_path: Optional[str],
) -> Optional[Dict[str, float]]:
    """读取 train.py 校准生成的 class_thresholds.json（可选）。"""
    for base in filter(None, [pt_path, onnx_path]):
        thr_path = Path(base).parent / "class_thresholds.json"
        if thr_path.exists():
            with open(thr_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def build_threshold_vector(
    classes: List[str],
    class_thresholds: Optional[Dict[str, float]],
) -> Optional[np.ndarray]:
    """将逐类阈值 dict 转为与 scores 对齐的 float32 向量；无阈值时返回 None。"""
    if not class_thresholds:
        return None
    return np.array(
        [class_thresholds.get(c, 0.5) for c in classes],
        dtype=np.float32,
    )


def softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的单样本 softmax（1D logits）。"""
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def softmax_batch(logits: np.ndarray) -> np.ndarray:
    """批量 softmax，logits 形状 (N, C)。"""
    x = logits.astype(np.float32, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def decide_class(scores: np.ndarray, thr_vec: Optional[np.ndarray]) -> int:
    """
    有逐类阈值时：仅在 score >= 阈值的类别中取最高分；
    若无一达标则回退 argmax（与 train.py 校准策略一致）。
    """
    if thr_vec is None:
        return int(np.argmax(scores))
    mask = scores >= thr_vec
    if mask.any():
        candidates = np.where(mask)[0]
        return int(candidates[np.argmax(scores[candidates])])
    return int(np.argmax(scores))


def build_result_dict(
    path: str,
    scores: np.ndarray,
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    *,
    elapsed_ms: float = 0.0,
) -> Dict:
    """
    由 softmax 后的 scores 构造 UI / 结果管理用 dict。

    confidence 为「预测类别」的概率（经逐类阈值决策后）；
    max_confidence 为 argmax 类别概率，二者可能不同。
    """
    pred_idx = decide_class(scores, thr_vec)
    argmax_idx = int(np.argmax(scores))
    return {
        "path": path,
        "class": classes[pred_idx],
        "class_idx": pred_idx,
        "confidence": float(scores[pred_idx]),
        "max_class": classes[argmax_idx],
        "max_confidence": float(scores[argmax_idx]),
        "all_scores": {c: float(s) for c, s in zip(classes, scores)},
        "elapsed_ms": elapsed_ms,
        "true_class": "",
        "flagged": False,
        "correction_saved": False,
    }


def make_error_result(path: str, err: str) -> Dict:
    """推理失败时的统一结果结构（开发版 / 机台版共用）。"""
    return {
        "path": str(path),
        "class": "ERROR",
        "class_idx": -1,
        "confidence": 0.0,
        "max_class": "ERROR",
        "max_confidence": 0.0,
        "all_scores": {},
        "elapsed_ms": 0.0,
        "error": err,
        "true_class": "",
        "flagged": False,
        "correction_saved": False,
    }


def logits_row_to_result(
    path: str,
    logits_row: np.ndarray,
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    *,
    elapsed_ms: float = 0.0,
) -> Dict:
    """单样本 logits → softmax → 结果 dict（ORT 输出为 raw logits，只 softmax 一次）。"""
    return build_result_dict(
        path, softmax(logits_row), classes, thr_vec, elapsed_ms=elapsed_ms,
    )


def run_batch_predict(
    image_paths: List[str],
    *,
    batch_size: int,
    preprocess_one: Callable[[str], Any],
    stack_batch: Callable[[List[Any]], Any],
    infer_batch: Callable[[Any], np.ndarray],
    classes: List[str],
    thr_vec: Optional[np.ndarray],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    result_cb: Optional[Callable[[int, Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """
    通用批量推理循环（开发 PyTorch / 开发 ONNX 回退 / 机台 ONNX 共用）。

    infer_batch 须返回 float32 logits，形状 (N, num_classes)；
    softmax 与阈值决策在 logits_row_to_result 内统一完成，避免重复或遗漏。
    """
    total = len(image_paths)
    results: List[Optional[Dict]] = [None] * total

    for start in range(0, total, batch_size):
        if should_stop and should_stop():
            break

        chunk_paths = image_paths[start : start + batch_size]
        tensors: List[Any] = []
        ok_indices: List[int] = []

        for offset, path in enumerate(chunk_paths):
            idx = start + offset
            try:
                tensors.append(preprocess_one(path))
                ok_indices.append(idx)
            except Exception as exc:
                err = make_error_result(str(path), str(exc))
                results[idx] = err
                if result_cb:
                    result_cb(idx, err)
                if progress_cb:
                    progress_cb(idx + 1, total)

        if not tensors:
            continue

        t0 = time.perf_counter()
        logits = infer_batch(stack_batch(tensors))
        chunk_ms = (time.perf_counter() - t0) * 1000
        per_ms = chunk_ms / len(ok_indices)

        for j, idx in enumerate(ok_indices):
            r = logits_row_to_result(
                str(image_paths[idx]),
                logits[j],
                classes,
                thr_vec,
                elapsed_ms=per_ms,
            )
            results[idx] = r
            if result_cb:
                result_cb(idx, r)
            if progress_cb:
                progress_cb(idx + 1, total)

    return [
        r if r is not None else make_error_result(str(p), "未知错误")
        for r, p in zip(results, image_paths)
    ]
