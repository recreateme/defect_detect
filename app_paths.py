#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用路径解析与 ORT 原生 DLL 搜索路径。

开发环境与 PyInstaller 冻结环境均可定位资源。
Windows 上 DLL 配置采用双点：
  · pyinstaller_hooks/rthook_ort_dll.py — 冻结进程最早执行（不 import 本模块）
  · setup_ort_dll_paths() — 业务入口 / import onnxruntime 前的统一入口（含 preload）
清除 CUDA_PATH/CUDA_HOME，避免机台旧 Toolkit 抢先加载错误 cudart。
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """可写目录：exe 所在目录（配置、corrections、外部 checkpoints）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    """只读资源目录：PyInstaller 解压的 _internal（或 _MEIPASS）。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return app_dir()


def resolve_path(rel: str) -> Path:
    """
    解析相对路径：优先 exe 同级，其次打包内置 _internal。
    绝对路径且存在则原样返回。
    """
    if not rel or not str(rel).strip():
        return Path(rel)
    p = Path(rel)
    if p.is_absolute():
        return p
    for base in (app_dir(), bundle_dir()):
        candidate = base / p
        if candidate.exists():
            return candidate
    return app_dir() / p


def chdir_app_root() -> None:
    """打包后工作目录设为 exe 同级，保证相对路径写入正确。"""
    if is_frozen():
        import os
        os.chdir(app_dir())


def _nvidia_site_bin_dirs() -> list[Path]:
    """开发环境：nvidia-* pip 包内的 bin 目录（CUDA runtime / cublas）。

    注意：排除 cudnn 目录 — torch cu132 自带 cudnn 9.x，若同时有 nvidia-cudnn-cu12
    的同名 DLL 在搜索路径中，会导致 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH。
    """
    try:
        import site
        import glob
        import os

        dirs: list[Path] = []
        seen: set[str] = set()
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            if not sp or not os.path.isdir(sp):
                continue
            for dll in glob.glob(os.path.join(sp, "nvidia", "*", "bin", "*.dll")):
                folder = Path(dll).parent
                key = str(folder.resolve())
                if key in seen:
                    continue
                # 跳过 cudnn：torch/lib 自带，避免同名版本冲突
                if folder.parent.name == "cudnn":
                    continue
                seen.add(key)
                dirs.append(folder)
        return dirs
    except Exception:
        return []


def _torch_lib_dir() -> Path | None:
    """torch/lib 目录（cu132 自带 cudnn/cudart/cublas，须优先于 nvidia cu12 避免同名冲突）。"""
    try:
        if is_frozen():
            cand = bundle_dir() / "torch" / "lib"
            if cand.is_dir():
                return cand
        else:
            import site
            for sp in site.getsitepackages() + [site.getusersitepackages()]:
                if not sp:
                    continue
                cand = Path(sp) / "torch" / "lib"
                if cand.is_dir():
                    return cand
    except Exception:
        pass
    return None


def ort_native_dirs() -> list[Path]:
    """ORT 及 CUDA 原生 DLL 搜索目录（顺序敏感：torch/lib 优先，避免 cudnn 同名冲突）。"""
    dirs: list[Path] = []
    # torch/lib 必须在最前面：torch cu132 自带 cudnn64_9.dll (CUDA13)，
    # 若 nvidia/cudnn/bin (CUDA12) 抢先加载同名 DLL，torch 会 WinError 127
    torch_lib = _torch_lib_dir()
    if torch_lib is not None:
        dirs.append(torch_lib)
    if is_frozen():
        base = bundle_dir()
        capi = base / "onnxruntime" / "capi"
        if capi.is_dir():
            dirs.append(capi)
        cuda_deps = base / "cuda_deps"
        if cuda_deps.is_dir():
            dirs.append(cuda_deps)
        for extra in (
            base / "PyQt5" / "Qt5" / "bin",
            base / "numpy.libs",
            base,
        ):
            if extra.is_dir() and extra not in dirs:
                dirs.append(extra)
    else:
        try:
            import onnxruntime as _ort  # noqa: WPS433
            import os

            capi = Path(os.path.dirname(_ort.__file__)) / "capi"
            if capi.is_dir():
                dirs.append(capi)
        except Exception:
            pass
        dirs.extend(_nvidia_site_bin_dirs())
    return [d for d in dirs if d.is_dir()]


def _preload_ort_core_dlls(capi: Path) -> None:
    """预加载 ORT 核心 DLL，避免 .pyd import 时初始化失败。"""
    if sys.platform != "win32" or not capi.is_dir():
        return
    import ctypes

    winmode = 8  # LOAD_WITH_ALTERED_SEARCH_PATH
    for name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
        dll_path = capi / name
        if not dll_path.is_file():
            continue
        try:
            ctypes.CDLL(str(dll_path), winmode=winmode)
        except OSError:
            pass


def setup_ort_dll_paths() -> None:
    """
    Windows 下将 onnxruntime / CUDA 原生 DLL 目录加入搜索路径，并预加载 ORT 核心 DLL。
    打包环境优先使用捆绑目录，保留系统 PATH（完全替换会导致 _ctypes 等无法加载）。
    须在 import onnxruntime 之前调用。
    """
    if sys.platform != "win32":
        return
    import os

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if is_frozen():
        os.environ.pop("CUDA_PATH", None)
        os.environ.pop("CUDA_HOME", None)

    dirs = ort_native_dirs()
    prepend_parts: list[str] = []
    capi_dir: Path | None = None

    for folder in dirs:
        p = str(folder)
        if folder.name == "capi" and folder.parent.name == "onnxruntime":
            capi_dir = folder
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(p)
        prepend_parts.append(p)

    if prepend_parts:
        # 冻结与开发均前置打包/站点 DLL，避免被系统旧库抢先
        os.environ["PATH"] = os.pathsep.join(prepend_parts) + os.pathsep + os.environ.get("PATH", "")

    if capi_dir is not None:
        _preload_ort_core_dlls(capi_dir)
