#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理引擎
支持双后端：
  · PyTorch  —— GPU 优先，自动检测 CUDA
  · ONNXRuntime —— CPU 优化，需安装 onnxruntime
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


class InferenceEngine:
    """图像缺陷分类推理引擎（单例友好，可热更新模型）"""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.ort_session = None
        self.classes: List[str] = []
        self.img_size: int = 224
        self.device: str = "cpu"
        self.backend: str = "none"      # "pytorch" | "onnx"
        self.transform: Optional[T.Compose] = None
        self.loaded: bool = False
        self._pt_path: Optional[str] = None
        self._onnx_path: Optional[str] = None
        self.class_thresholds: Optional[Dict[str, float]] = None

    # ──────────────────────────────────────────────────
    # 加载
    # ──────────────────────────────────────────────────
    def load(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str] = None,
        use_gpu: bool = True,
    ) -> str:
        """
        加载模型，返回状态描述字符串。
        选择策略:
          1. use_gpu=True 且 CUDA 可用 → PyTorch GPU
          2. onnx_path 存在 且 onnxruntime 已安装 → ONNX CPU
          3. 兜底 → PyTorch CPU
        """
        self.loaded = False

        gpu_ok = use_gpu and torch.cuda.is_available()
        onnx_ok = (
            onnx_path and Path(onnx_path).exists() and HAS_ORT
        )

        if gpu_ok:
            target_device, backend = "cuda", "pytorch"
        elif onnx_ok:
            target_device, backend = "cpu", "onnx"
        else:
            target_device, backend = "cpu", "pytorch"

        # ── 读取类别信息 ─────────────────────────────
        self.classes, self.img_size = self._read_meta(pt_path, onnx_path)
        self.class_thresholds = self._read_class_thresholds(pt_path, onnx_path)

        # ── 加载权重 ─────────────────────────────────
        if backend == "onnx":
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            self.model = None
        else:
            self.model = self._build_model(len(self.classes))
            if pt_path and Path(pt_path).exists():
                ckpt = torch.load(pt_path, map_location=target_device)
                state = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state)
            self.model.to(target_device).eval()

        self.device  = target_device
        self.backend = backend
        self._pt_path   = pt_path
        self._onnx_path = onnx_path

        # ── 预处理 transform ─────────────────────────
        self.transform = T.Compose([
            T.Resize((self.img_size + 16, self.img_size + 16)),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.loaded = True
        if target_device == "cuda":
            dev_str = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            dev_str = "CPU"
        return (
            f"模型加载成功  [{self.backend.upper()} / {dev_str}]  "
            f"{len(self.classes)} 类别"
        )

    def reload(self) -> str:
        """用相同路径重新加载（训练完成后热更新调用）。"""
        use_gpu = (self.device == "cuda")
        return self.load(self._pt_path, self._onnx_path, use_gpu=use_gpu)

    # ──────────────────────────────────────────────────
    # 推理
    # ──────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        单张图像预测。
        返回:
            {
                "path":       str,
                "class":      str,
                "class_idx":  int,
                "confidence": float,           # 0~1
                "all_scores": {cls: float},    # softmax 全类别得分
                "elapsed_ms": float,
            }
        """
        if not self.loaded:
            raise RuntimeError('模型未加载，请先在「设置」页面加载模型。')

        t0 = time.perf_counter()
        tensor = self._preprocess(image_path)     # (1, 3, H, W)

        if self.backend == "onnx":
            ort_out = self.ort_session.run(
                None, {"input": tensor.numpy()}
            )[0]                                  # (1, C)
            scores = torch.softmax(
                torch.from_numpy(ort_out), dim=1
            )[0].numpy()
        else:
            tensor  = tensor.to(self.device)
            logits  = self.model(tensor)
            scores  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = self._decide_class(scores)
        elapsed  = (time.perf_counter() - t0) * 1000

        return {
            "path":       str(image_path),
            "class":      self.classes[pred_idx],
            "class_idx":  pred_idx,
            "confidence": float(scores[pred_idx]),
            "all_scores": {c: float(s)
                           for c, s in zip(self.classes, scores)},
            "elapsed_ms": elapsed,
            # 以下字段供 UI 使用，不由推理填写
            "true_class":    "",
            "flagged":       False,
            "correction_saved": False,
        }

    def predict_batch(
        self,
        image_paths: List[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict]:
        """批量预测；progress_cb(current, total) 每完成一张回调。"""
        results = []
        total   = len(image_paths)
        for i, path in enumerate(image_paths):
            try:
                r = self.predict(path)
            except Exception as exc:
                r = self._error_result(path, str(exc))
            results.append(r)
            if progress_cb:
                progress_cb(i + 1, total)
        return results

    # ──────────────────────────────────────────────────
    # 内部辅助
    # ──────────────────────────────────────────────────
    def _preprocess(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)   # (1, 3, H, W)

    def _read_meta(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str],
    ) -> Tuple[List[str], int]:
        """从 checkpoint 或 class_map.json 读取类别列表和图像尺寸。"""
        classes, img_size = [], 224

        if pt_path and Path(pt_path).exists():
            ckpt = torch.load(pt_path, map_location="cpu")
            classes  = ckpt.get("classes",  [])
            img_size = ckpt.get("img_size", 224)

        if not classes:
            # 尝试同目录的 class_map.json
            for base in filter(None, [pt_path, onnx_path]):
                map_path = Path(base).parent / "class_map.json"
                if map_path.exists():
                    with open(map_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    classes = d.get("classes", [])
                    break

        if not classes:
            # 尝试 train_config.json
            for base in filter(None, [pt_path, onnx_path]):
                cfg_path = Path(base).parent / "train_config.json"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    classes  = d.get("classes",  [])
                    img_size = d.get("img_size", 224)
                    break

        if not classes:
            raise FileNotFoundError(
                "未能获取类别信息。"
                "请确保 class_map.json 或 train_config.json 与模型文件在同一目录。"
            )
        return classes, img_size

    def _read_class_thresholds(
        self,
        pt_path: Optional[str],
        onnx_path: Optional[str],
    ) -> Optional[Dict[str, float]]:
        """读取 train.py 生成的 class_thresholds.json（可选）。"""
        for base in filter(None, [pt_path, onnx_path]):
            thr_path = Path(base).parent / "class_thresholds.json"
            if thr_path.exists():
                with open(thr_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    def _decide_class(self, scores: np.ndarray) -> int:
        """
        有 per-class 阈值时：仅在 score >= 阈值的类别中取最高分；
        否则回退 argmax。
        """
        if not self.class_thresholds:
            return int(np.argmax(scores))
        thr = np.array([
            self.class_thresholds.get(c, 0.5) for c in self.classes
        ], dtype=np.float32)
        mask = scores >= thr
        if mask.any():
            candidates = np.where(mask)[0]
            return int(candidates[np.argmax(scores[candidates])])
        return int(np.argmax(scores))

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
            m = models.resnet18(pretrained=False)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, num_classes),
            )
        return m

    @staticmethod
    def _error_result(path: str, err: str) -> Dict:
        return {
            "path":       str(path),
            "class":      "读取失败",
            "class_idx":  -1,
            "confidence": 0.0,
            "all_scores": {},
            "elapsed_ms": 0.0,
            "error":      err,
            "true_class": "",
            "flagged":    False,
            "correction_saved": False,
        }
