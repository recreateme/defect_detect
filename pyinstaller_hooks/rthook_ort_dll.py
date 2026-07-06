# PyInstaller runtime hook：Windows 下配置 ORT / CUDA DLL 搜索路径（须最早执行）。
import os
import sys

# 判断是否为 PyInstaller 打包后的冻结环境，且运行在 Windows 平台
if getattr(sys, "frozen", False) and sys.platform == "win32":
    # 获取 PyInstaller 解压后的临时目录（_MEIPASS）
    base = getattr(sys, "_MEIPASS", "")
    if base:
        # 设置 CUDA 模块延迟加载，避免启动时立即加载所有 CUDA 模块
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        # 避免机台 CUDA_PATH 指向旧版 CUDA，导致 ORT 核心 DLL 初始化失败
        os.environ.pop("CUDA_PATH", None)
        os.environ.pop("CUDA_HOME", None)

        # 构建 ONNX Runtime C API 的动态库路径
        capi = os.path.join(base, "onnxruntime", "capi")

        # 定义需要添加到 DLL 搜索路径的文件夹列表
        folders = [
            capi,  # ONNX Runtime C API 目录
            os.path.join(base, "cuda_deps"),  # CUDA 依赖库目录
            base,  # 程序根目录
            os.path.join(base, "PyQt5", "Qt5", "bin"),  # PyQt5 Qt5 二进制文件目录
            os.path.join(base, "numpy.libs"),  # NumPy 依赖库目录
        ]

        # 收集所有存在的目录，准备添加到 PATH
        prepend = []
        for folder in folders:
            if os.path.isdir(folder):
                # Windows Python 3.8+ 需要使用 add_dll_directory 注册 DLL 搜索目录
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(folder)
                prepend.append(folder)

        # 将配置的目录添加到 PATH 环境变量最前面，确保优先使用打包的 DLL
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend) + os.pathsep + os.environ.get("PATH", "")

        # 预加载 ORT 核心 DLL（须在 import onnxruntime 之前）
        try:
            import ctypes

            # 遍历需要预加载的 ONNX Runtime 核心动态库
            for name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
                path = os.path.join(capi, name)
                if os.path.isfile(path):
                    # 使用 winmode=8 (LOAD_WITH_ALTERED_SEARCH_PATH) 加载 DLL
                    # 这样可以确保 DLL 依赖项从正确的路径加载
                    ctypes.CDLL(path, winmode=8)
        except OSError:
            # 如果预加载失败则忽略，让 onnxruntime 模块自行处理加载
            pass
