#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机台部署专用推理引擎（ONNX Runtime + NumPy + Pillow，不依赖 PyTorch）。

性能要点：
  · Session 图优化 (ORT_ENABLE_ALL)
  · 批量推理：多张图一次 session.run，显著降低 GPU 启动开销
  · 预处理：resize+crop 参数在 load 时缓存；归一化用 float32
  · 阈值向量在 load 时预计算，避免每张图重建数组

批量循环与 logits→结果 转换见 inference_common（与开发版 ONNX 回退共用，避免逻辑漂移）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from inference_common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_threshold_vector,
    logits_row_to_result,
    read_class_thresholds,
    read_model_meta,
    run_batch_predict,
)

ort: Any = None
HAS_ORT = False
_ORT_IMPORT_ERROR = ""

_DEFAULT_BATCH_GPU = 32
_DEFAULT_BATCH_CPU = 8


def _ensure_ort_import() -> bool:
    """延迟加载 onnxruntime，并在 import 前配置 DLL 路径（Windows 打包必需）。"""
    global ort, HAS_ORT, _ORT_IMPORT_ERROR
    if HAS_ORT and ort is not None:
        return True
    if _ORT_IMPORT_ERROR:
        return False
    try:
        from app_paths import setup_ort_dll_paths, ort_native_dirs, _preload_ort_core_dlls
        setup_ort_dll_paths()
        for d in ort_native_dirs():
            if d.name == "capi" and d.parent.name == "onnxruntime":
                _preload_ort_core_dlls(d)
                break
    except ImportError:
        pass
    try:
        import onnxruntime as ort_module
        ort = ort_module
        HAS_ORT = True
        return True
    except Exception as exc:
        import traceback
        _ORT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        HAS_ORT = False
        ort = None
        return False


def _ort_unavailable_message() -> str:
    if _ORT_IMPORT_ERROR:
        return (
            "onnxruntime 加载失败。\n"
            f"详情: {_ORT_IMPORT_ERROR}\n"
            "机台 GPU 包需 onnxruntime-gpu；请确认打包含 ORT DLL 或重新安装依赖。"
        )
    return "未安装 onnxruntime。机台 GPU 包需 onnxruntime-gpu。"


def ort_available_providers() -> List[str]:
    if not _ensure_ort_import():
        return []
    return ort.get_available_providers()


def pick_ort_providers(use_gpu: bool) -> Tuple[List[Any], str]:
    """选择 ORT ExecutionProvider 列表（GPU 优先 + CPU 回退）。"""
    available = ort_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        return (
            [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ],
            "CUDAExecutionProvider",
        )
    return (["CPUExecutionProvider"], "CPUExecutionProvider")


def _make_session_options() -> Any:
    """创建 ORT SessionOptions：图融合/常量折叠等优化可提升 CPU/GPU 推理速度。"""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    opts.inter_op_num_threads = 1
    return opts


class InferenceEngine:
    """ONNX 推理引擎（机台版，可选 GPU，支持批量推理）。"""

    def __init__(self):
        self.ort_session = None
        self.classes: List[str] = []
        self.img_size: int = 128
        self.device: str = "cpu"
        self.backend: str = "none"
        self.loaded: bool = False
        self._pt_path: Optional[str] = None
        self._onnx_path: Optional[str] = None
        self._use_gpu: bool = True
        self._ort_provider: str = "CPUExecutionProvider"
        self.class_thresholds: Optional[Dict[str, float]] = None
        self._input_name: str = "input"
        self._thr_vec: Optional[np.ndarray] = None
        self._side: int = 144
        self._crop_box: Tuple[int, int, int, int] = (0, 0, 128, 128)
        self._mean_hwc = IMAGENET_MEAN.reshape(1, 1, 3)
        self._std_hwc = IMAGENET_STD.reshape(1, 1, 3)
        self._batch_size: int = _DEFAULT_BATCH_CPU

    def load(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str] = None,
        use_gpu: bool = True,
    ) -> str:
        self.loaded = False
        self._use_gpu = use_gpu

        if not _ensure_ort_import():
            raise RuntimeError(_ort_unavailable_message())
        if not onnx_path or not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"未找到 ONNX 模型：{onnx_path or '(未指定)'}\n"
                "请确认 checkpoints/model.onnx（及 model.onnx.data）与 exe 同级。"
            )

        self.classes, self.img_size = read_model_meta(pt_path, onnx_path)
        self.class_thresholds = read_class_thresholds(pt_path, onnx_path)
        self._thr_vec = build_threshold_vector(self.classes, self.class_thresholds)
        self._refresh_preprocess_cache()

        providers, picked = pick_ort_providers(use_gpu)
        sess_opts = _make_session_options()
        self.ort_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
        )
        inputs = self.ort_session.get_inputs()
        self._input_name = inputs[0].name if inputs else "input"

        active = self.ort_session.get_providers()
        self._ort_provider = active[0] if active else picked
        self.device = "cuda" if "CUDA" in self._ort_provider.upper() else "cpu"
        self._batch_size = (
            _DEFAULT_BATCH_GPU if self.device == "cuda" else _DEFAULT_BATCH_CPU
        )
        self.backend = "onnx"
        self._pt_path = pt_path
        self._onnx_path = onnx_path
        self.loaded = True

        dev_label = "GPU" if self.device == "cuda" else "CPU"
        if use_gpu and self.device != "cuda":
            hint = (
                f"模型加载成功  [ONNX / {dev_label}]  {len(self.classes)} 类别 "
                f"· {self.img_size}px · batch={self._batch_size}\n"
                f"（已请求 GPU，但 CUDA 不可用，可用: "
                f"{', '.join(ort_available_providers())}）"
            )
        else:
            hint = (
                f"模型加载成功  [ONNX / {dev_label}]  {len(self.classes)} 类别 "
                f"· {self.img_size}px · batch={self._batch_size}"
            )
        return hint

    def reload(self) -> str:
        return self.load(self._pt_path, self._onnx_path, use_gpu=self._use_gpu)

    def _refresh_preprocess_cache(self) -> None:
        self._side = self.img_size + 16
        off = (self._side - self.img_size) // 2
        self._crop_box = (off, off, off + self.img_size, off + self.img_size)

    def predict(self, image_path: str) -> Dict:
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")
        t0 = time.perf_counter()
        logits = self.ort_session.run(
            None, {self._input_name: self._preprocess(image_path)},
        )[0][0]
        result = logits_row_to_result(
            str(image_path), logits, self.classes, self._thr_vec,
        )
        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def predict_batch(
        self,
        image_paths: List[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        result_cb: Optional[Callable[[int, Dict], None]] = None,
        batch_size: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Dict]:
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")

        session = self.ort_session
        input_name = self._input_name

        def _infer_batch(batch: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: batch})[0]

        return run_batch_predict(
            image_paths,
            batch_size=batch_size or self._batch_size,
            preprocess_one=self._preprocess_chw,
            stack_batch=lambda ts: np.stack(ts, axis=0).astype(np.float32, copy=False),
            infer_batch=_infer_batch,
            classes=self.classes,
            thr_vec=self._thr_vec,
            progress_cb=progress_cb,
            result_cb=result_cb,
            should_stop=should_stop,
        )

    def _preprocess(self, image_path: str) -> np.ndarray:
        return self._preprocess_chw(image_path)[np.newaxis]

    def _preprocess_chw(self, image_path: str) -> np.ndarray:
        return self._preprocess_image(self._load_rgb(image_path))

    def _load_rgb(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as im:
            return im.convert("RGB")

    def _preprocess_image(self, img: Image.Image) -> np.ndarray:
        img = img.resize((self._side, self._side), Image.BILINEAR)
        img = img.crop(self._crop_box)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean_hwc) / self._std_hwc
        return arr.transpose(2, 0, 1)
