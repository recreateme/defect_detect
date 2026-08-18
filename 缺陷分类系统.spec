# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('D:\\Develop\\defect_detect\\build_staging\\cuda_deps', 'cuda_deps')]
binaries = []
hiddenimports = ['PyQt5.sip', 'onnxruntime', 'onnxruntime.capi', 'onnxruntime.capi._pybind_state', 'onnxruntime.capi.onnxruntime_inference_collection', 'onnxruntime.capi.onnxruntime_validation', 'onnxruntime.capi.build_and_package_info', 'PIL', 'numpy', 'sahi_detector', 'inference_engine_onnx', 'inference_common', 'ultralytics', 'torch', 'torchvision', 'cv2', 'yaml', 'sympy']
binaries += collect_dynamic_libs('onnxruntime')
hiddenimports += collect_submodules('onnxruntime.capi')
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('matplotlib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sympy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app_deploy.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['D:\\Develop\\defect_detect\\pyinstaller_hooks\\rthook_ort_dll.py'],
    excludes=['torchaudio', 'pandas', 'sklearn', 'tensorboard', 'onnx', 'onnxscript', 'onnxruntime.transformers', 'onnxruntime.quantization'],
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
    icon=['D:\\Develop\\defect_detect\\assets\\app.ico'],
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
