#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开发环境推理引擎 — PyTorch GPU 优先，ONNX CPU 回退。

由 python app.py 默认选用；机台版见 inference_engine_onnx.py（仅 ORT）。

性能要点：
  · PyTorch 路径：GPU 上批量 forward，减少 kernel 启动次数
  · ONNX 回退：与机台版共用 inference_common.run_batch_predict
  · torch.inference_mode() 关闭 autograd，避免 .numpy() 报错
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from inference_common import (
    build_threshold_vector,
    logits_row_to_result,
    read_class_thresholds,
    read_model_meta,
    run_batch_predict,
)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None  # type: ignore

_DEFAULT_BATCH_GPU = 32
_DEFAULT_BATCH_CPU = 8


def _load_pt_meta(pt_path: Optional[str]) -> Tuple[List[str], int]:
    if pt_path and Path(pt_path).exists():
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        return ckpt.get("classes", []), int(ckpt.get("img_size", 224))
    return [], 224


def _ort_session_options():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    opts.inter_op_num_threads = 1
    return opts


class InferenceEngine:
    """图像缺陷分类推理引擎（开发版，支持热更新与批量推理）。"""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.ort_session = None
        self.classes: List[str] = []
        self.img_size: int = 128
        self.device: str = "cpu"
        self.backend: str = "none"
        self.transform: Optional[T.Compose] = None
        self.loaded: bool = False
        self._pt_path: Optional[str] = None
        self._onnx_path: Optional[str] = None
        self.class_thresholds: Optional[Dict[str, float]] = None
        self._thr_vec: Optional[np.ndarray] = None
        self._input_name: str = "input"
        self._batch_size: int = _DEFAULT_BATCH_CPU

    def load(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str] = None,
        use_gpu: bool = True,
    ) -> str:
        self.loaded = False

        gpu_ok = use_gpu and torch.cuda.is_available()
        onnx_ok = bool(onnx_path and Path(onnx_path).exists() and HAS_ORT)

        if gpu_ok:
            target_device, backend = "cuda", "pytorch"
        elif onnx_ok:
            target_device, backend = "cpu", "onnx"
        else:
            target_device, backend = "cpu", "pytorch"

        self.classes, self.img_size = read_model_meta(
            pt_path, onnx_path, load_pt_meta=_load_pt_meta,
        )
        self.class_thresholds = read_class_thresholds(pt_path, onnx_path)
        self._thr_vec = build_threshold_vector(self.classes, self.class_thresholds)

        if backend == "onnx":
            sess_opts = _ort_session_options()
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            inputs = self.ort_session.get_inputs()
            self._input_name = inputs[0].name if inputs else "input"
            self.model = None
        else:
            self.model = self._build_model(len(self.classes))
            if pt_path and Path(pt_path).exists():
                ckpt = torch.load(pt_path, map_location=target_device, weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state)
            self.model.to(target_device).eval()
            self.ort_session = None

        self.device = target_device
        self.backend = backend
        self._pt_path = pt_path
        self._onnx_path = onnx_path
        self._batch_size = (
            _DEFAULT_BATCH_GPU if target_device == "cuda" else _DEFAULT_BATCH_CPU
        )

        self.transform = T.Compose([
            T.Resize((self.img_size + 16, self.img_size + 16)),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.loaded = True
        if target_device == "cuda":
            dev_str = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            dev_str = "CPU"
        return (
            f"模型加载成功  [{self.backend.upper()} / {dev_str}]  "
            f"{len(self.classes)} 类别 · {self.img_size}px · batch={self._batch_size}"
        )

    def reload(self) -> str:
        use_gpu = self.device == "cuda"
        return self.load(self._pt_path, self._onnx_path, use_gpu=use_gpu)

    @torch.inference_mode()
    def predict(self, image_path: str) -> Dict:
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")

        t0 = time.perf_counter()
        if self.backend == "onnx":
            tensor = self._preprocess(image_path)
            logits = self.ort_session.run(
                None, {self._input_name: tensor.numpy()},
            )[0][0]
            result = logits_row_to_result(
                str(image_path), logits, self.classes, self._thr_vec,
            )
        else:
            tensor = self._preprocess(image_path)
            logits = self.model(tensor.to(self.device)).detach().cpu().numpy()[0]
            result = logits_row_to_result(
                str(image_path), logits, self.classes, self._thr_vec,
            )

        result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        return result

    @torch.inference_mode()
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

        if self.backend == "onnx":
            session = self.ort_session
            input_name = self._input_name

            def _preprocess_chw(path: str) -> np.ndarray:
                return self._preprocess(path)[0].numpy()

            def _infer_batch(batch: np.ndarray) -> np.ndarray:
                return session.run(None, {input_name: batch})[0]

            return run_batch_predict(
                image_paths,
                batch_size=batch_size or self._batch_size,
                preprocess_one=_preprocess_chw,
                stack_batch=lambda ts: np.stack(ts, axis=0).astype(np.float32, copy=False),
                infer_batch=_infer_batch,
                classes=self.classes,
                thr_vec=self._thr_vec,
                progress_cb=progress_cb,
                result_cb=result_cb,
                should_stop=should_stop,
            )

        model = self.model
        device = self.device

        def _preprocess_chw(path: str) -> torch.Tensor:
            return self._preprocess(path)[0]

        def _stack(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.stack(tensors).to(device)

        def _infer_batch(batch: torch.Tensor) -> np.ndarray:
            return model(batch).detach().cpu().numpy()

        return run_batch_predict(
            image_paths,
            batch_size=batch_size or self._batch_size,
            preprocess_one=_preprocess_chw,
            stack_batch=_stack,
            infer_batch=_infer_batch,
            classes=self.classes,
            thr_vec=self._thr_vec,
            progress_cb=progress_cb,
            result_cb=result_cb,
            should_stop=should_stop,
        )

    @torch.inference_mode()
    def predict_batch_images(
        self,
        images: List[Image.Image],
        progress_cb: Optional[Callable[[int, int], None]] = None,
        result_cb: Optional[Callable[[int, Dict], None]] = None,
        batch_size: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[Dict]:
        """
        内存图像批量推理（SAHI 裁剪用）。

        与 predict_batch(路径) 共用：self.transform、self.classes、self._thr_vec、
        run_batch_predict → logits_row_to_result / build_result_dict，
        保证与「缺陷检测」页同一套类别名与置信度语义。
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先在「设置」页面加载模型。")
        keys = [f"mem:{i}" for i in range(len(images))]
        rgb = [im.convert("RGB") if im.mode != "RGB" else im for im in images]

        if self.backend == "onnx":
            session = self.ort_session
            input_name = self._input_name

            def _preprocess_chw(key: str) -> np.ndarray:
                idx = int(key.split(":", 1)[1])
                return self.transform(rgb[idx]).numpy()

            def _infer_batch(batch: np.ndarray) -> np.ndarray:
                return session.run(None, {input_name: batch})[0]

            return run_batch_predict(
                keys,
                batch_size=batch_size or self._batch_size,
                preprocess_one=_preprocess_chw,
                stack_batch=lambda ts: np.stack(ts, axis=0).astype(np.float32, copy=False),
                infer_batch=_infer_batch,
                classes=self.classes,
                thr_vec=self._thr_vec,
                progress_cb=progress_cb,
                result_cb=result_cb,
                should_stop=should_stop,
            )

        model = self.model
        device = self.device

        def _preprocess_chw(key: str) -> torch.Tensor:
            idx = int(key.split(":", 1)[1])
            return self.transform(rgb[idx])

        return run_batch_predict(
            keys,
            batch_size=batch_size or self._batch_size,
            preprocess_one=_preprocess_chw,
            stack_batch=lambda ts: torch.stack(ts).to(device),
            infer_batch=lambda batch: model(batch).detach().cpu().numpy(),
            classes=self.classes,
            thr_vec=self._thr_vec,
            progress_cb=progress_cb,
            result_cb=result_cb,
            should_stop=should_stop,
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as im:
            return self.transform(im.convert("RGB")).unsqueeze(0)

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        try:
            m = models.efficientnet_b0(weights=None)
            in_f = m.classifier[1].in_features
            m.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_f, num_classes),
            )
        except AttributeError:
            m = models.resnet18(weights=None)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, num_classes),
            )
        return m
