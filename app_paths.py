#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""应用路径解析：开发环境 / PyInstaller 打包后均可正确定位资源。"""

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
    """开发环境：nvidia-* pip 包内的 bin 目录（cuDNN / CUDA runtime 等）。"""
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
                if key not in seen:
                    seen.add(key)
                    dirs.append(folder)
        return dirs
    except Exception:
        return []


def ort_native_dirs() -> list[Path]:
    """ORT 及 CUDA 原生 DLL 搜索目录（顺序敏感：capi 优先于 cuda_deps）。"""
    dirs: list[Path] = []
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
    Windows 下将 onnxruntime / CUDA 原生 DLL 目录加入搜索路径。
    打包环境会隔离 PATH，避免机台已装旧版 CUDA 导致 DLL 冲突。
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

    if is_frozen():
        # 机台：优先使用打包目录，保留系统 PATH（完全替换会导致 _ctypes 等无法加载）
        os.environ["PATH"] = os.pathsep.join(prepend_parts) + os.pathsep + os.environ.get("PATH", "")
    elif prepend_parts:
        os.environ["PATH"] = os.pathsep.join(prepend_parts) + os.pathsep + os.environ.get("PATH", "")
