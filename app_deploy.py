#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机台部署入口：PyInstaller 打包此文件。

分类：onnxruntime-gpu（inference_engine_onnx）
检测：SAHI + ultralytics YOLO（与 app.py 钻石检测分类页相同）
训练：机台页只读（无 train.py 热更新）

环境变量:
  DEFECTS_DEPLOY=1   — 由本文件设置；app.py 据此选用 ONNX 分类引擎
  DEFECTS_VERIFY=1   — 无 GUI 验收：测 ORT 导入 + model.onnx 加载后按码退出

开发调试: python app_deploy.py
验收模式: set DEFECTS_VERIFY=1 && 缺陷分类系统.exe
"""

import os
import sys

os.environ["DEFECTS_DEPLOY"] = "1"

from app_paths import resolve_path, setup_ort_dll_paths

setup_ort_dll_paths()


def _verify_and_exit() -> None:
    """打包验收：仅测试 ORT 导入与 GPU 模型加载，不启动 GUI。"""
    import inference_engine_onnx as ort_eng

    print("=== DEFECTS_VERIFY ===")
    if not ort_eng._ensure_ort_import():
        print("ORT FAIL:", ort_eng._ORT_IMPORT_ERROR)
        raise SystemExit(1)
    print("providers:", ort_eng.ort_available_providers())
    onnx = str(resolve_path("checkpoints/model.onnx"))
    eng = ort_eng.InferenceEngine()
    print(eng.load(None, onnx, use_gpu=True))
    print("device:", eng.device)
    if eng.device != "cuda":
        print("[WARN] 已回退 CPU，但程序仍可运行；如需 GPU 请检查驱动 / CUDA / cuDNN。")
    raise SystemExit(0)


if os.environ.get("DEFECTS_VERIFY") == "1":
    _verify_and_exit()

from app import main

if __name__ == "__main__":
    main()
