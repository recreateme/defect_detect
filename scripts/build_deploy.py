#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机台打包：ONNX 分类 + SAHI/YOLO 大图检测（-D onedir）。

正式打包唯一入口（勿用根目录手写 .spec）：
  1. 校验 checkpoints、确认 ORT 可用（已安装则保留当前版本，不降级）
  2. 收集 nvidia CUDA pip DLL → build_staging/cuda_deps/
  3. 校验 torch / ultralytics / opencv（YOLO 检测）
  4. PyInstaller --onedir 打包 app_deploy.py + rthook + SAHI 依赖
  5. _patch_dist；checkpoints 与 detect_weights 放到 exe 同级

可在 defects-deploy 或 cv-yolo 环境中运行（推荐 defects-deploy 避免污染开发环境）:
  conda activate defects-deploy   # 或: conda activate cv-yolo
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

# 分类用 ORT；检测用 torch+ultralytics+scipy，故不再排除这些
# ultralytics 运行时 import matplotlib，必须打入；仅排除训练/无关包
# 注意：sympy 不能排除 — ultralytics/torch 运行时链路需要它（YOLO 加载会 import sympy）
ORT_EXCLUDES = [
    "torchaudio",
    "pandas", "sklearn", "tensorboard",
    "onnx", "onnxscript",
    "onnxruntime.transformers",
    "onnxruntime.quantization",
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

    # ORT 用 cuda_deps/；torch/YOLO 可能依赖 _internal/nvidia/，不可再删

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
    """仅检查 CUDA 运行时 DLL 是否可用（cudart/cublas）；已安装则保留当前版本，
    不强制升级，避免污染开发环境。cudnn 由 torch/lib 提供，此处不检查。"""
    import glob
    import site

    found_cudart = False
    found_cublas = False
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        if not sp:
            continue
        for dll in glob.glob(str(Path(sp) / "nvidia" / "*" / "bin" / "*.dll")):
            name = Path(dll).name.lower()
            if name.startswith("cudart64"):
                found_cudart = True
            if name.startswith("cublas64") or name.startswith("cublaslt64"):
                found_cublas = True
    # torch/lib 也自带 cudart/cublas（CUDA13），同样算可用
    try:
        import torch  # noqa: WPS433
        tlib = Path(torch.__file__).parent / "lib"
        for dll in glob.glob(str(tlib / "*.dll")):
            name = Path(dll).name.lower()
            if name.startswith("cudart64"):
                found_cudart = True
            if name.startswith("cublas64") or name.startswith("cublaslt64"):
                found_cublas = True
    except Exception:
        pass

    print(f"[CUDA] cudart={'OK' if found_cudart else 'MISSING'}  cublas={'OK' if found_cublas else 'MISSING'}")
    if not (found_cudart and found_cublas):
        raise SystemExit(
            "[错误] 缺少 CUDA 运行时 DLL (cudart/cublas)。\n"
            "       pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12\n"
            "       或确认 torch 已安装（torch/lib 自带）"
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
            # 跳过 cudnn：torch cu132 自带 cudnn 9.x，混用会导致
            # CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
            if Path(src).parent.parent.name == "cudnn":
                continue
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
    try:
        sess = ort.InferenceSession(
            str(onnx),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = sess.get_providers()
        if active and active[0] == "CUDAExecutionProvider":
            print("[ORT] CUDA Provider 实测 OK")
        else:
            print(f"[警告] CUDA Provider 未激活，当前: {active}")
    except Exception as exc:
        print(f"[警告] CUDA Provider 试跑失败，运行时将回退 CPU：{type(exc).__name__}: {exc}")


def _ensure_ort(use_gpu: bool) -> None:
    """确保 ORT 可用：已安装则保留当前版本（不降级，避免污染开发环境）；
    未安装才按 ORT_GPU_VER 安装。同时清理对立包（仅当对方存在且本方已就绪时）。"""
    from app_paths import setup_ort_dll_paths
    setup_ort_dll_paths()
    try:
        import onnxruntime as ort  # noqa: WPS433
        cur = ort.__version__
        print(f"[ORT] 已安装 onnxruntime {cur}  Provider: {ort.get_available_providers()}")
        if use_gpu and "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"[警告] 当前 ORT 无 CUDA Provider，可能装的是 CPU 版 onnxruntime。")
        return
    except ImportError:
        pass

    if use_gpu:
        pkg_spec = f"onnxruntime-gpu=={ORT_GPU_VER}"
        other = "onnxruntime"
    else:
        pkg_spec = "onnxruntime"
        other = "onnxruntime-gpu"
    print(f"[依赖] 未检测到 ORT，安装 {pkg_spec} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", pkg_spec],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", other],
        cwd=ROOT,
    )
    setup_ort_dll_paths()
    import onnxruntime as ort  # noqa: WPS433
    print(f"[ORT] 版本 {ort.__version__}  Provider: {ort.get_available_providers()}")


def _warn_if_wrong_env() -> None:
    try:
        import torch  # noqa: WPS433
        print(f"[SAHI] torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    except ImportError:
        print(
            "[错误] 机台钻石检测需要 PyTorch + ultralytics + opencv-python。\n"
            "       conda activate cv-yolo  (或 defects-deploy)\n"
            "       pip install ultralytics opencv-python\n"
            "       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )
        raise SystemExit(1)


def _ensure_sahi_deps() -> None:
    missing: list[str] = []
    try:
        import torchvision  # noqa: WPS433
        print(f"[SAHI] torchvision {getattr(torchvision, '__version__', '?')}")
    except ImportError:
        missing.append("torchvision")
    try:
        import ultralytics  # noqa: WPS433
        print(f"[SAHI] ultralytics {getattr(ultralytics, '__version__', '?')}")
    except ImportError:
        missing.append("ultralytics")
    try:
        import cv2  # noqa: WPS433
        print(f"[SAHI] opencv {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")
    if missing:
        print("[错误] 缺少: " + ", ".join(missing))
        print("       pip install ultralytics opencv-python torchvision")
        raise SystemExit(1)


def _resolve_yolo_src() -> Path | None:
    """打包时复制 YOLO 权重：app_config / detect_weights / checkpoints。"""
    cfg = ROOT / "app_config.json"
    if cfg.is_file():
        try:
            import json
            data = json.loads(cfg.read_text(encoding="utf-8"))
            raw = str(data.get("yolo_path", "")).strip()
            if raw:
                p = Path(raw)
                if p.is_file():
                    return p
                rel = ROOT / raw
                if rel.is_file():
                    return rel
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    for cand in (
        ROOT / "detect_weights" / "best.pt",
        ROOT / "checkpoints" / "yolo.pt",
        ROOT / "checkpoints" / "best.pt",
    ):
        if cand.is_file() and cand.name != "best_model.pt":
            return cand
    return None


# 机台版默认配置（须与 app.py 的 _DEFAULT_CFG_DEPLOY 保持一致：相对路径，无开发机绝对路径）
_DEPLOY_APP_CONFIG = {
    "pt_path":          "",                       # 机台分类不使用 .pt
    "onnx_path":        "checkpoints/model.onnx",
    "data_dir":         "data",
    "corrections_dir":  "corrections",
    "use_gpu":          True,
    "yolo_path":        "detect_weights/best.pt",
    "sahi_device":      "auto",
    "sahi_slice_size":  1280,
    "sahi_overlap":     0.20,
    "sahi_det_conf":    0.35,
    "sahi_batch_size":  8,
    "sahi_crop_padding": 15,
    "sahi_output_dir":  "sahi_output",
    "sahi_ios_thresh":     0.60,
    "sahi_min_area_ratio": 0.45,
    "sahi_max_aspect_ratio": 1.5,
    "sahi_edge_filter":    True,
    "sahi_edge_margin_px": 20,
}


def _write_deploy_app_config(dist: Path) -> None:
    """写入机台版 app_config.json 到 exe 同级，使用相对路径。
    避免复制开发机 app_config.json（含绝对路径）污染机台。"""
    import json
    target = dist / "app_config.json"
    target.write_text(
        json.dumps(_DEPLOY_APP_CONFIG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[配置] 已写入机台版 app_config.json -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="机台打包：ONNX 分类 + SAHI/YOLO 检测")
    parser.add_argument("--console", action="store_true", help="保留控制台窗口")
    parser.add_argument("--cpu-only", action="store_true", help="使用 CPU 版 onnxruntime")
    args = parser.parse_args()

    _warn_if_wrong_env()
    _ensure_sahi_deps()

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
    icon = ROOT / "assets" / "app.ico"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir",  # -D：目录模式，机台须整包复制
        "--name", "缺陷分类系统",
        "--windowed" if not args.console else "--console",
        "--distpath", "dist", "--workpath", "build",
        f"--runtime-hook={RTHOOK}",
    ]
    if icon.is_file():
        cmd.extend(["--icon", str(icon)])
    cmd.extend([
        "--hidden-import", "PyQt5.sip",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "onnxruntime.capi",
        "--hidden-import", "onnxruntime.capi._pybind_state",
        "--hidden-import", "onnxruntime.capi.onnxruntime_inference_collection",
        "--hidden-import", "onnxruntime.capi.onnxruntime_validation",
        "--hidden-import", "onnxruntime.capi.build_and_package_info",
        "--hidden-import", "PIL",
        "--hidden-import", "numpy",
        "--hidden-import", "sahi_detector",
        "--hidden-import", "inference_engine_onnx",
        "--hidden-import", "inference_common",
        "--hidden-import", "ultralytics",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "cv2",
        "--hidden-import", "yaml",
        "--hidden-import", "sympy",
        "--collect-submodules", "onnxruntime.capi",
        "--collect-binaries", "onnxruntime",
        "--collect-all", "ultralytics",
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "cv2",
        "--collect-all", "matplotlib",
        "--collect-all", "sympy",
        "app_deploy.py",
    ])
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

    yolo_src = _resolve_yolo_src()
    yolo_dst_dir = DIST / "detect_weights"
    yolo_dst_dir.mkdir(parents=True, exist_ok=True)
    if yolo_src is not None and yolo_src.is_file():
        shutil.copy2(yolo_src, yolo_dst_dir / "best.pt")
        print(f"[YOLO] 已复制检测权重: {yolo_src} -> detect_weights/best.pt")
    else:
        print("[警告] 未找到 YOLO .pt，机台需在「设置 → 切片推理配置」中指定 detect_weights/best.pt")

    # 写入机台版 app_config.json（相对路径，避免开发机绝对路径污染）
    _write_deploy_app_config(DIST)

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
