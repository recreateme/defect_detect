# 创建机台打包专用 Conda 环境（与训练环境 cv-yolo 隔离）
# 用法: powershell -ExecutionPolicy Bypass -File scripts/setup_deploy_env.ps1
#
# 机台包含 ONNX 分类 + SAHI/YOLO 检测，需安装 CUDA 版 PyTorch。

$ErrorActionPreference = "Stop"
$EnvName = "defects-deploy"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=== 创建 Conda 环境: $EnvName ==="
conda create -n $EnvName python=3.12 -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 安装 CUDA 版 PyTorch（YOLO 检测）==="
conda run -n $EnvName python -m pip install -U pip
conda run -n $EnvName python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 安装 deploy 依赖（ORT + ultralytics + PyQt5）==="
conda run -n $EnvName python -m pip install -r "$Root\requirements-deploy.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "完成。后续打包:"
Write-Host "  conda activate $EnvName"
Write-Host "  cd `"$Root`""
Write-Host "  python scripts/build_deploy.py"
