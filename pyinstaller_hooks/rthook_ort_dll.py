# PyInstaller runtime hook：Windows 下配置 ORT / CUDA DLL 搜索路径（须最早执行）。
#
# 与 app_paths.setup_ort_dll_paths() 语义一致，故意独立实现：
# runtime hook 在业务模块 import 之前运行，不能依赖 app_paths。
# 业务代码统一走 app_paths；本文件只服务冻结进程冷启动。
import os
import sys

if getattr(sys, "frozen", False) and sys.platform == "win32":
    base = getattr(sys, "_MEIPASS", "")
    if base:
        # 延迟加载 CUDA 内核；清掉机台旧 CUDA_PATH，避免 ORT 初始化失败
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
        os.environ.pop("CUDA_PATH", None)
        os.environ.pop("CUDA_HOME", None)

        capi = os.path.join(base, "onnxruntime", "capi")
        folders = [
            capi,
            os.path.join(base, "cuda_deps"),
            base,
            os.path.join(base, "PyQt5", "Qt5", "bin"),
            os.path.join(base, "numpy.libs"),
        ]

        prepend = []
        for folder in folders:
            if os.path.isdir(folder):
                # Python 3.8+：仅改 PATH 不够，须 add_dll_directory
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(folder)
                prepend.append(folder)

        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend) + os.pathsep + os.environ.get("PATH", "")

        # 预加载须在 import onnxruntime 之前；winmode=8 = LOAD_WITH_ALTERED_SEARCH_PATH
        try:
            import ctypes

            for name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
                path = os.path.join(capi, name)
                if os.path.isfile(path):
                    ctypes.CDLL(path, winmode=8)
        except OSError:
            pass
