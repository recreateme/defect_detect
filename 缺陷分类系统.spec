# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

binaries = []
hiddenimports = ['PyQt5.sip', 'onnxruntime', 'onnxruntime.capi', 'onnxruntime.capi._pybind_state', 
                    'onnxruntime.capi.onnxruntime_inference_collection', 'onnxruntime.capi.onnxruntime_validation', 
                    'onnxruntime.capi.build_and_package_info', 'PIL', 'numpy']

binaries += collect_dynamic_libs('onnxruntime')
hiddenimports += collect_submodules('onnxruntime.capi')


a = Analysis(
    ['app_deploy.py'],
    pathex=[],
    binaries=binaries,
    datas=[('D:\\工作记录\\3 人工智能相关课题\\张旭\\defects_classify\\build_staging\\cuda_deps', 'cuda_deps')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['D:\\工作记录\\3 人工智能相关课题\\张旭\\defects_classify\\pyinstaller_hooks\\rthook_ort_dll.py'],
    excludes=['torch', 'torchvision', 'torchaudio', 'scipy', 'matplotlib', 'pandas', 'sklearn', 'tensorboard', 'ultralytics', 'onnx', 'onnxscript', 'sympy', 'onnxruntime.transformers', 'onnxruntime.quantization', 'nvidia'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='缺陷分类系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='缺陷分类系统',
)
