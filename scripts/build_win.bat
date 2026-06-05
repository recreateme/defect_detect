@echo off
REM ============================================================================
REM  缺陷分类系统 — Windows 打包脚本（PyInstaller）
REM  用法:
REM    scripts\build_win.bat              默认：窗口模式 + 内置 checkpoints
REM    scripts\build_win.bat slim         不打包模型（产线单独拷贝 checkpoints/）
REM    scripts\build_win.bat console      保留控制台（调试闪退）
REM ============================================================================

chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

set "ROOT=%~dp0.."
cd /d "%ROOT%" || (
    echo [错误] 无法进入项目目录: %ROOT%
    exit /b 1
)

set "MODE=full"
set "CONSOLE="
if /i "%~1"=="slim"    set "MODE=slim"
if /i "%~1"=="console" set "CONSOLE=--console"

echo.
echo ============================================================
echo   缺陷分类系统 — PyInstaller 打包
echo   项目根: %CD%
echo   模式  : %MODE%  %CONSOLE%
echo ============================================================
echo.

REM ── 检查 Python ─────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 python，请先激活 Conda 环境，例如:
    echo   conda activate cv-yolo
    exit /b 1
)
echo [1/5] Python: 
python --version

REM ── 检查 PyInstaller ────────────────────────────────────────
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [2/5] 安装 PyInstaller ...
    python -m pip install pyinstaller
) else (
    echo [2/5] PyInstaller 已安装
)

REM ── 检查入口与模型（full 模式）──────────────────────────────
if not exist "app.py" (
    echo [错误] 未找到 app.py，请在项目根目录执行本脚本。
    exit /b 1
)

if /i "%MODE%"=="full" (
    echo [3/5] 检查 checkpoints ...
    if not exist "checkpoints\best_model.pt" (
        echo [警告] 未找到 checkpoints\best_model.pt
        echo         可先训练: python train.py --data_dir data --img_size 128
        echo         或补跑后处理: python train.py --postprocess_only
        set /p CONTINUE=是否继续打包（不含有效模型）? [y/N]: 
        if /i not "!CONTINUE!"=="y" exit /b 1
    )
) else (
    echo [3/5] slim 模式 — 不打包 checkpoints，部署时需手动放置
)

REM ── 组装 PyInstaller 参数 ───────────────────────────────────
set "NAME=缺陷分类系统"
set "DIST=dist\%NAME%"
set "BUILD=build\%NAME%"

set "PYI=pyinstaller --noconfirm --clean --name "%NAME%" --windowed"
if defined CONSOLE set "PYI=pyinstaller --noconfirm --clean --name "%NAME%" --console"

set "PYI=%PYI% --hidden-import PyQt5.sip --hidden-import onnxruntime --hidden-import PIL"
set "PYI=%PYI% --collect-all onnxruntime"
set "PYI=%PYI% --distpath dist --workpath build"

if /i "%MODE%"=="full" (
    if exist "checkpoints" (
        set "PYI=!PYI! --add-data checkpoints;checkpoints"
    )
)

REM 可选图标（项目根放置 icon.ico 时自动使用）
if exist "icon.ico" (
    set "PYI=!PYI! --icon icon.ico"
    echo         使用图标: icon.ico
)

echo [4/5] 开始打包 ...
echo.
echo !PYI! app.py
echo.

!PYI! app.py
if errorlevel 1 (
    echo.
    echo [错误] PyInstaller 打包失败。
    echo        调试建议: scripts\build_win.bat console
    exit /b 1
)

echo.
echo [5/5] 打包完成
echo ============================================================
echo   输出目录: %CD%\%DIST%
echo   主程序  : %DIST%\%NAME%.exe
echo.
if /i "%MODE%"=="full" (
    echo   已内置 checkpoints\ — 可直接复制整个文件夹到产线机台
) else (
    echo   slim 模式: 请手动将 checkpoints\ 复制到 exe 同级目录
)
echo.
echo   更新模型: 替换 checkpoints\ 下 best_model.pt / model.onnx /
echo             class_thresholds.json 后重启应用即可
echo ============================================================
echo.

endlocal
exit /b 0
