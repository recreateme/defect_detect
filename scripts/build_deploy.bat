@echo off
REM 机台打包入口 — 转发至 scripts/build_deploy.py（推荐在 defects-deploy 环境中运行）
REM 用法:
REM   scripts\build_deploy.bat
REM   scripts\build_deploy.bat console

chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0.." || exit /b 1

set "EXTRA="
if /i "%~1"=="console" set "EXTRA=--console"
if /i "%~1"=="cpu-only" set "EXTRA=--cpu-only"

where python >nul 2>&1 || (
    echo [错误] 请先 conda activate defects-deploy
    exit /b 1
)

echo [提示] 推荐使用 defects-deploy 环境，详见 打包部署说明.md
python scripts/build_deploy.py %EXTRA%
exit /b %ERRORLEVEL%
