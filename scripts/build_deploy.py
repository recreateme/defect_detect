#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机台 ONNX-GPU 精简打包（不含 PyTorch）。

请在 defects-deploy 环境中运行:
  conda activate defects-deploy
  python scripts/build_deploy.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
STAGING = ROOT / "build_staging" / "checkpoints"
CUDA_STAGING = ROOT / "build_staging" / "cuda_deps"
DIST = ROOT / "dist" / "缺陷分类系统"
RTHOOK = ROOT / "pyinstaller_hooks" / "rthook_ort_dll.py"
ORT_GPU_VER = "1.20.1"
CKPT_FILES = (
    "model.onnx",
    "model.onnx.data",
    "class_map.json",
    "class_thresholds.json",
    "train_config.json",
)
CUDA_PIP = (
    "nvidia-cudnn-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
)

ORT_EXCLUDES = [
    "torch", "torchvision", "torchaudio",
    "scipy", "matplotlib", "pandas", "sklearn", "tensorboard", "ultralytics",
    "onnx", "onnxscript", "sympy",
    "onnxruntime.transformers",
    "onnxruntime.quantization",
    "nvidia",
]


def _patch_dist(dist: Path) -> None:
    """打包后补丁：补全 ORT Python 文件、移除重复 DLL/目录。"""
    internal = dist / "_internal"
    capi = internal / "onnxruntime" / "capi"
    ort_pkg = internal / "onnxruntime"
    if not capi.is_dir():
        print("[警告] 未找到 onnxruntime/capi，跳过补丁")
        return

    trt = capi / "onnxruntime_providers_tensorrt.dll"
    if trt.exists():
        trt.unlink()
        print("[补丁] 已移除 onnxruntime_providers_tensorrt.dll")

    # PyInstaller 可能从 site-packages 再收集 nvidia/，与 cuda_deps 重复
    nvidia_dir = internal / "nvidia"
    if nvidia_dir.is_dir():
        shutil.rmtree(nvidia_dir)
        print("[补丁] 已移除重复的 _internal/nvidia/")

    # 移除 _internal 根目录上与 capi/cuda_deps 重复的 DLL（历史扁平化残留）
    capi_dlls = {f.name for f in capi.glob("*.dll")}
    cuda_dir = internal / "cuda_deps"
    cuda_dlls = {f.name for f in cuda_dir.glob("*.dll")} if cuda_dir.is_dir() else set()
    for dll in internal.glob("*.dll"):
        if dll.name in capi_dlls or dll.name in cuda_dlls:
            dll.unlink()
            print(f"[补丁] 已移除重复根目录 DLL: {dll.name}")

    # 模型仅保留 exe 同级 checkpoints/，不重复打入 _internal
    internal_ckpt = internal / "checkpoints"
    if internal_ckpt.is_dir():
        shutil.rmtree(internal_ckpt)
        print("[补丁] 已移除 _internal/checkpoints/")

    # PyInstaller 可能误收集非必需 CUDA 组件（如 cufft）
    for optional in ("cufft64_11.dll",):
        stray = internal / optional
        if stray.is_file() and optional not in cuda_dlls:
            stray.unlink()
            print(f"[补丁] 已移除非必需 DLL: {optional}")

    import onnxruntime  # noqa: WPS433

    site_ort = Path(onnxruntime.__file__).parent
    site_capi = site_ort / "capi"
    ort_pkg.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py",):
        src = site_ort / name
        if src.exists() and not (ort_pkg / name).exists():
            shutil.copy2(src, ort_pkg / name)
            print(f"[补丁] 已补全 onnxruntime/{name}")
    if site_capi.is_dir():
        for py in site_capi.glob("*.py"):
            dst = capi / py.name
            if not dst.exists():
                shutil.copy2(py, dst)
                print(f"[补丁] 已补全 capi/{py.name}")

    py_root = Path(sys.executable).parent
    lib_bin = py_root / "Library" / "bin"
    for dll in ("ffi.dll", "liblzma.dll", "LIBBZ2.dll", "libexpat.dll", "zlib.dll"):
        if (internal / dll).exists():
            continue
        for src in (py_root / dll, lib_bin / dll):
            if src.exists():
                shutil.copy2(src, internal / dll)
                print(f"[补丁] 已复制 {dll} -> _internal/")
                break

    required = ("onnxruntime.dll", "onnxruntime_pybind11_state.pyd")
    missing = [n for n in required if not (capi / n).exists()]
    if missing:
        raise RuntimeError(f"打包不完整，capi 缺少: {missing}")


def _ensure_cuda_runtime() -> None:
    print("[依赖] 确保 CUDA 运行时 (cuDNN / cudart / cublas) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", *CUDA_PIP],
        cwd=ROOT,
        check=True,
    )


def _stage_cuda_dlls(dest: Path) -> int:
    import glob
    import site

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    count = 0
    seen: set[str] = set()
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if not sp:
            continue
        for src in glob.glob(str(Path(sp) / "nvidia" / "*" / "bin" / "*.dll")):
            name = Path(src).name
            if name in seen:
                continue
            seen.add(name)
            shutil.copy2(src, dest / name)
            count += 1
    print(f"[CUDA] 已收集 {count} 个 DLL -> {dest}")
    if count == 0:
        raise SystemExit("[错误] 未找到 nvidia CUDA DLL，请先 pip install -r requirements-deploy.txt")
    return count


def _test_cuda_provider() -> None:
    from app_paths import setup_ort_dll_paths
    setup_ort_dll_paths()
    import onnxruntime as ort  # noqa: WPS433

    providers = ort.get_available_providers()
    print(f"[ORT] 可用 Provider: {providers}")
    if "CUDAExecutionProvider" not in providers:
        print("[警告] 未检测到 CUDAExecutionProvider")
        return
    onnx = ROOT / "checkpoints" / "model.onnx"
    if not onnx.exists():
        return
    sess = ort.InferenceSession(
        str(onnx),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = sess.get_providers()
    if active and active[0] == "CUDAExecutionProvider":
        print("[ORT] CUDA Provider 实测 OK")
    else:
        print(f"[警告] CUDA Provider 未激活，当前: {active}")


def _ensure_ort(use_gpu: bool) -> None:
    if use_gpu:
        pkg_spec = f"onnxruntime-gpu=={ORT_GPU_VER}"
        other = "onnxruntime"
    else:
        pkg_spec = "onnxruntime"
        other = "onnxruntime-gpu"
    print(f"[依赖] 确保已安装 {pkg_spec} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", pkg_spec],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", other],
        cwd=ROOT,
    )
    from app_paths import setup_ort_dll_paths
    setup_ort_dll_paths()
    import onnxruntime as ort  # noqa: WPS433
    print(f"[ORT] 版本 {ort.__version__}  Provider: {ort.get_available_providers()}")


def _warn_if_wrong_env() -> None:
    try:
        import torch  # noqa: WPS433
        print(
            "[警告] 当前环境检测到 PyTorch，建议使用独立打包环境:\n"
            "       conda activate defects-deploy\n"
            "       或运行 scripts/setup_deploy_env.ps1"
        )
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="机台 ONNX-GPU 精简打包")
    parser.add_argument("--console", action="store_true", help="保留控制台窗口")
    parser.add_argument("--cpu-only", action="store_true", help="使用 CPU 版 onnxruntime")
    args = parser.parse_args()

    _warn_if_wrong_env()

    src_ckpt = ROOT / "checkpoints"
    if not (src_ckpt / "model.onnx").exists():
        print("[错误] 缺少 checkpoints/model.onnx")
        return 1
    if not (src_ckpt / "model.onnx.data").exists():
        print("[警告] 缺少 model.onnx.data")

    _ensure_ort(use_gpu=not args.cpu_only)
    if not args.cpu_only:
        _ensure_cuda_runtime()
        _test_cuda_provider()

    if STAGING.parent.exists():
        shutil.rmtree(STAGING.parent)
    STAGING.mkdir(parents=True)
    for name in CKPT_FILES:
        f = src_ckpt / name
        if f.exists():
            shutil.copy2(f, STAGING / name)
    if not args.cpu_only:
        _stage_cuda_dlls(CUDA_STAGING)

    sep = ";" if sys.platform == "win32" else ":"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", "缺陷分类系统",
        "--windowed" if not args.console else "--console",
        "--distpath", "dist", "--workpath", "build",
        f"--runtime-hook={RTHOOK}",
        "--hidden-import", "PyQt5.sip",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "onnxruntime.capi",
        "--hidden-import", "onnxruntime.capi._pybind_state",
        "--hidden-import", "onnxruntime.capi.onnxruntime_inference_collection",
        "--hidden-import", "onnxruntime.capi.onnxruntime_validation",
        "--hidden-import", "onnxruntime.capi.build_and_package_info",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--collect-submodules", "onnxruntime.capi",
        "--collect-binaries", "onnxruntime",
        "app_deploy.py",
    ]
    if not args.cpu_only and CUDA_STAGING.is_dir():
        cmd.insert(-1, f"--add-data={CUDA_STAGING}{sep}cuda_deps")
    for mod in ORT_EXCLUDES:
        cmd.extend(["--exclude-module", mod])

    print("[打包]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    _patch_dist(DIST)

    dst = DIST / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)
    for f in STAGING.iterdir():
        shutil.copy2(f, dst / f.name)

    internal = DIST / "_internal"
    buckets: dict[str, float] = {}
    for f in DIST.rglob("*"):
        if f.is_file():
            rel = f.relative_to(DIST)
            if len(rel.parts) > 1 and rel.parts[0] == "_internal":
                key = rel.parts[1] if len(rel.parts) > 2 else "(internal-root)"
            else:
                key = rel.parts[0]
            buckets[key] = buckets.get(key, 0) + f.stat().st_size

    size_mb = sum(buckets.values()) / 1024 / 1024
    mode = "CPU" if args.cpu_only else "ONNX-GPU"
    print(f"\n完成 [{mode}]: {DIST}")
    print(f"体积: {size_mb:.1f} MB")
    print("体积构成:")
    for k, v in sorted(buckets.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v / 1024 / 1024:6.1f} MB  {k}")
    print(f"运行: {DIST / '缺陷分类系统.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
