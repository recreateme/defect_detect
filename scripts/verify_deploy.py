#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收机台部署路径：ORT 导入 + 模型加载。用法: python scripts/verify_deploy.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ["DEFECTS_DEPLOY"] = "1"
sys.path.insert(0, str(ROOT))

from app_paths import resolve_path, setup_ort_dll_paths  # noqa: E402

setup_ort_dll_paths()

from inference_engine_onnx import (  # noqa: E402
    HAS_ORT,
    InferenceEngine,
    _ORT_IMPORT_ERROR,
    _ensure_ort_import,
    ort_available_providers,
)


def main() -> int:
    print("=== verify_deploy ===")
    ok = _ensure_ort_import()
    print("HAS_ORT:", HAS_ORT, "import_ok:", ok)
    if not ok:
        print("ERROR:", _ORT_IMPORT_ERROR)
        return 1
    print("providers:", ort_available_providers())

    onnx = str(resolve_path("checkpoints/model.onnx"))
    if not Path(onnx).exists():
        print("ERROR: missing", onnx)
        return 1

    eng = InferenceEngine()
    msg = eng.load(None, onnx, use_gpu=True)
    print(msg)
    print("device:", eng.device, "loaded:", eng.loaded)
    if eng.device != "cuda":
        print("[WARN] 未使用 GPU，请检查 cuDNN/CUDA DLL 是否已打入包或已安装。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
