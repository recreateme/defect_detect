#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钻石缺陷图像分类 — PyQt5 桌面应用
================================================================================

功能
  · 钻石检测分类：5120×5120 大图 SAHI 切片检测 + 缺陷分类（开发版与机台 exe 均提供）
  · 单图 / 文件夹批量缺陷检测
  · 检测结果管理与 CSV / 按类别文件夹导出
  · 误分类修正 → 归档至 corrections/（主动学习数据回收）
  · 模型再训练（开发版，子进程调用 train.py）与热加载
  · 设置页（分类配置 + 切片推理配置 Tab）

依赖
  开发版: PyQt5, torch, torchvision, onnxruntime, Pillow, ultralytics, opencv-python
  机台版: PyQt5, onnxruntime-gpu, Pillow, ultralytics, opencv-python, torch（YOLO 检测）
          分类仍走 ONNX；由 app_deploy.py 入口

架构（自顶向下）
  main()
    └─ QApplication + 全局 QSS
         └─ MainWindow（QMainWindow）
              ├─ sidebar — nav 按钮，objectName="nav"
              └─ QStackedWidget — 页面索引见 NAV_* 常量
                   0 DiamondDetectPage  钻石检测分类（SAHI）
                   1 DetectionPage      缺陷检测（单张 Tab + 批量 Tab）
                   2 ResultsPage        结果管理
                   3 CorrectionPage     误分类修正
                   4 RetrainPage        模型再训练（AdminLockBar 保护）
                   5 SettingsPage       分类配置 + 切片推理配置（AdminLockBar）

跨页面共享
  AppState — 推理引擎、内存中的检测历史、app_config.json、管理员解锁标志

耗时操作（禁止阻塞 GUI 主线程）
  InferenceWorker    — engine.predict_batch，信号回传逐条结果
  SahiPipelineWorker — SAHI 检测 + 分类流水线
  TrainWorker        — subprocess 运行 train.py，逐行读 stdout

双入口
  python app.py         → DEPLOY_ONNX_ONLY=False，分类用 inference_engine（PyTorch 优先）
  python app_deploy.py  → DEFECTS_DEPLOY=1，分类用 inference_engine_onnx；SAHI 检测仍可用

详见项目根目录「QT应用开发说明.md」。
"""

import os
import sys
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# PyQt5 三件套（详见 QT应用开发说明.md）:
#   QtWidgets — 窗口、按钮、表格、布局等可视控件
#   QtCore    — 信号槽、线程、定时器、事件循环
#   QtGui     — 字体、颜色、QPixmap 图像、光标
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from app_paths import app_dir, chdir_app_root, is_frozen, resolve_path, setup_ort_dll_paths

# ── 运行模式 ────────────────────────────────────────────────────────────────
# True  = 机台/打包：缺陷分类仅 ONNX（inference_engine_onnx），窗口标题带「· 机台版」
#         SAHI/YOLO 大图检测仍启用（需 ultralytics + torch + opencv）
# False = 开发环境：分类 PyTorch GPU 优先，ONNX 作回退
DEPLOY_ONNX_ONLY = is_frozen() or os.environ.get("DEFECTS_DEPLOY") == "1"

# 机台版须在 import onnxruntime 之前配置 DLL 搜索路径（Windows 打包必需）
if DEPLOY_ONNX_ONLY:
    setup_ort_dll_paths()

try:
    if DEPLOY_ONNX_ONLY:
        from inference_engine_onnx import InferenceEngine
    else:
        from inference_engine import InferenceEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    InferenceEngine = None  # type: ignore  — 缺依赖时 UI 仍可启动，设置页会提示

# ── 导航页索引（与 MainWindow.nav_items / QStackedWidget 顺序一致）──────────
NAV_DIAMOND   = 0   # 钻石检测分类（SAHI 大图检测 + 缺陷分类）
NAV_DETECT    = 1   # 缺陷检测（单张 / 批量分类）
NAV_RESULTS   = 2   # 结果管理
NAV_CORRECT   = 3   # 误分类修正
NAV_RETRAIN   = 4   # 模型再训练
NAV_SETTINGS  = 5   # 设置（分类配置 + 切片推理配置 Tab）

# ── 大图检测依赖检查（开发版 / 机台 exe 均可；缺包时页面提示安装）────────
_HAS_SAHI_DEPS = False
_SAHI_DEPS_MSG = ""
try:
    from sahi_detector import SahiDetector, SahiPipeline, check_ultralytics, check_cv2
    _ok_ultra, _msg_ultra = check_ultralytics()
    _ok_cv2,   _msg_cv2   = check_cv2()
    _HAS_SAHI_DEPS = _ok_ultra and _ok_cv2
    if not _HAS_SAHI_DEPS:
        _SAHI_DEPS_MSG = (_msg_ultra if not _ok_ultra else "") + "\n" + (_msg_cv2 if not _ok_cv2 else "")
except ImportError:
    _SAHI_DEPS_MSG = "sahi_detector 模块加载失败"

# ═══════════════════════════════════════════════════════════
# 常量 & 配置 — 全部设置参数持久化至项目根目录 app_config.json
# 启动时自动读取；文件缺失则使用下方代码默认值；设置页保存时重建文件
# ═══════════════════════════════════════════════════════════
APP_NAME = "钻石缺陷图像分类系统"
APP_VER  = "v1.0" + (" · 机台版" if DEPLOY_ONNX_ONLY else "")
# 窗口/任务栏图标：优先项目内 assets/app.ico，其次用户提供的素材路径
_APP_ICON_CANDIDATES = (
    app_dir() / "assets" / "app.ico",
    Path(r"C:\Users\12998\Pictures\素材\cat.ico"),
)
CFG_FILE = app_dir() / "app_config.json"   # 可写；开发=项目根，打包后=exe 同级
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_SCRIPT = app_dir() / "train.py"      # RetrainPage 子进程入口


def _load_app_icon() -> Optional[QIcon]:
    """加载窗口图标；文件不存在时返回 None，不影响启动。"""
    for p in _APP_ICON_CANDIDATES:
        if p.is_file():
            return QIcon(str(p))
    return None

# 开发版默认同时配置 .pt 与 .onnx；机台版由 FULL 派生后清空 pt_path。
# 分类阈值见 checkpoints/class_thresholds.json，不在此配置全局置信度旋钮。
_DEFAULT_CFG_FULL: Dict = {
    "pt_path":         "checkpoints/best_model.pt",
    "onnx_path":       "checkpoints/model.onnx",
    "data_dir":        "data",              # train.py --data_dir
    "corrections_dir": "corrections",       # 误分类归档根目录
    "use_gpu":         True,
    # ── SAHI 切片推理配置 ──────────────────────────────────────
    "yolo_path":          "",              # YOLO .pt 路径（在设置页配置）
    "sahi_device":        "auto",
    "sahi_slice_size":    1280,
    "sahi_overlap":       0.20,            # 过低易碎框；建议 0.20~0.25
    "sahi_det_conf":      0.35,            # 过低误检增多；建议 ≥0.30
    "sahi_batch_size":    8,
    "sahi_crop_padding":  15,
    "sahi_output_dir":    "sahi_output",
    # ── SAHI 检测后处理（去误检 + 剔除边缘不完整目标）──────────────
    "sahi_ios_thresh":       0.60,   # IoS；0.8 过严易漏抑制，建议 0.55~0.65
    "sahi_min_area_ratio":   0.45,   # 面积过滤：相对中位面积最小比例；<=0 关闭
    "sahi_max_aspect_ratio": 1.5,    # 长宽比上限 max(w,h)/min(w,h)；<=0 关闭；钻石近正方形默认 1.5
    "sahi_edge_filter":      True,   # 是否剔除边缘不完整钻石
    "sahi_edge_margin_px":   20,     # 边缘边距（像素）；过小几乎无效
}

_DEFAULT_CFG_DEPLOY: Dict = {
    **_DEFAULT_CFG_FULL,
    "pt_path": "",  # 机台分类不使用 .pt；YOLO 权重见 detect_weights/best.pt
    "yolo_path": "detect_weights/best.pt",
}


def _default_cfg() -> Dict:
    return dict(_DEFAULT_CFG_DEPLOY if DEPLOY_ONNX_ONLY else _DEFAULT_CFG_FULL)


DEFAULT_CFG = _default_cfg()


def _resolve_cfg_path(raw: str) -> str:
    """将配置中的相对路径解析为绝对路径（exe 同级优先于 _internal）。"""
    if not raw or not str(raw).strip():
        return ""
    return str(resolve_path(raw.strip()))


def _migrate_cfg(loaded: Dict, base: Dict) -> Dict:
    """旧配置字段迁移到当前 schema。"""
    out = dict(loaded)
    # 旧版「边缘边距比例」→ 像素（按 5120 短边估算）
    if "sahi_edge_margin_px" not in out and "sahi_edge_margin_ratio" in out:
        try:
            out["sahi_edge_margin_px"] = max(
                0, int(round(5120 * float(out["sahi_edge_margin_ratio"])))
            )
        except (TypeError, ValueError):
            out["sahi_edge_margin_px"] = base.get("sahi_edge_margin_px", 20)
    return out


def _load_cfg() -> Dict:
    """
    启动时读取项目根目录 app_config.json。
    · 文件缺失 / 损坏 → 使用代码中的默认参数（不自动写盘）
    · 文件存在 → 与默认合并（缺字段用默认补全，文件中的值优先）
    """
    base = _default_cfg()
    if not CFG_FILE.exists():
        return dict(base)
    try:
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            print(f"[配置] {CFG_FILE.name} 格式无效，已回退默认参数")
            return dict(base)
        loaded = _migrate_cfg(loaded, base)
        merged = {**base, **loaded}
        merged.pop("conf_threshold", None)  # 已废弃，忽略旧文件中的键
        return merged
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[配置] 读取 {CFG_FILE} 失败（{exc}），已回退默认参数")
        return dict(base)


def _save_cfg(cfg: Dict):
    """
    写入项目根目录 app_config.json（UTF-8，中文不转义）。
    始终以代码默认值为骨架补全全部已知键，文件不存在时重新创建。
    """
    base = _default_cfg()
    # 已知键按默认顺序写出；额外键（若有）追加在后；丢弃已废弃键
    cleaned = {k: v for k, v in cfg.items() if v is not None and k != "conf_threshold"}
    merged = {**base, **cleaned}
    ordered: Dict = {k: merged[k] for k in base.keys()}
    for k, v in merged.items():
        if k not in ordered:
            ordered[k] = v
    CFG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


# 模型再训练 / 设置页解除只读所需密码
ADMIN_PAGE_PASSWORD = "20250508"

# ── QSS（Qt Style Sheets）────────────────────────────────
# 类似 CSS，通过选择器（类名、objectName、伪状态）统一全局外观。
# 控件可通过 setProperty("class", "primary") 或 setObjectName("sidebar") 匹配规则。
STYLE = """
* { font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif; }
QMainWindow, QDialog { background: #F0F2F5; }
/* ── Sidebar（明亮背景，与白色主区有区分）── */
QWidget#sidebar  { background: #E8EAF0; }
QPushButton#nav  {
    color:#37474F; background:transparent; text-align:left;
    padding:14px 14px 14px 18px; border:none; font-size:17px; border-radius:0;
}
QPushButton#nav:hover    { background:#D5D9E2; color:#1565C0; }
QPushButton#nav[sel=true]{
    background:#DDE8F5; color:#1565C0;
    border-left:3px solid #1976D2; padding-left:17px; font-weight:bold;
}
QLabel#logo    { color:#263238; font-size:14px; font-weight:bold; padding:20px 16px 4px 20px; }
QLabel#logo_sub{ color:#78909C; font-size:11px; padding:0 0 18px 22px; }
/* ── Buttons ── */
QPushButton.primary{
    background:#1976D2; color:white; border:none;
    border-radius:5px; padding:7px 18px; font-weight:bold;
}
QPushButton.primary:hover { background:#1565C0; }
QPushButton.primary:disabled { background:#90A4AE; color:#CFD8DC; }
QPushButton.success{
    background:#2E7D32; color:white; border:none;
    border-radius:5px; padding:7px 18px; font-weight:bold;
}
QPushButton.success:hover { background:#1B5E20; }
QPushButton.success:disabled { background:#90A4AE; color:#CFD8DC; }
QPushButton.warning{
    background:#E65100; color:white; border:none;
    border-radius:5px; padding:7px 18px; font-weight:bold;
}
QPushButton.warning:hover { background:#BF360C; }
QPushButton.danger{
    background:#C62828; color:white; border:none;
    border-radius:5px; padding:7px 18px; font-weight:bold;
}
QPushButton.danger:hover   { background:#B71C1C; }
QPushButton.danger:disabled{ background:#90A4AE; color:#CFD8DC; }
QPushButton.flat{
    background:transparent; color:#1976D2; border:1px solid #1976D2;
    border-radius:5px; padding:6px 14px;
}
QPushButton.flat:hover { background:#E3F2FD; }
/* ── Cards / Frames ── */
QFrame#card{
    background:white; border:1px solid #CFD8DC; border-radius:8px;
}
QGroupBox{
    border:1px solid #CFD8DC; border-radius:6px;
    margin-top:16px; padding:8px 6px 8px 6px;
}
QGroupBox::title{
    subcontrol-origin:margin; subcontrol-position:top left;
    padding:0 8px; color:#455A64; font-weight:bold;
}
/* ── Table ── */
QTableWidget{
    background:white; gridline-color:#ECEFF1;
    border:1px solid #CFD8DC; border-radius:4px; outline:0;
}
QTableWidget::item:selected{ background:#E3F2FD; color:#0D47A1; }
QTableWidget::item:hover   { background:#F5F8FF; }
QHeaderView::section{
    background:#ECEFF1; padding:7px 6px; border:none;
    border-right:1px solid #CFD8DC; font-weight:bold; color:#37474F;
}
/* ── Input controls ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox{
    border:1px solid #CFD8DC; border-radius:4px;
    padding:5px 8px; background:white; color:#212121;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus{ border-color:#1976D2; }
/* ── Progress / Log ── */
QProgressBar{
    background:#E0E0E0; border:none; border-radius:4px;
    height:10px; text-align:center;
}
QProgressBar::chunk{ background:#42A5F5; border-radius:4px; }
QProgressBar#sahi_pbar{
    min-height:28px; height:28px; font-size:12px; color:#263238;
    text-align:center; padding:0 6px;
}
QProgressBar#sahi_pbar::chunk{ border-radius:4px; }
QTextEdit#log{
    background:#1A1A2E; color:#A8D8A8; font-size:12px;
    font-family:Consolas,'Courier New',monospace; border:none;
}
/* ── Misc ── */
QScrollArea{ border:none; }
QStatusBar{ background:#263238; color:#78909C; font-size:11px; padding:0 8px; }
QStatusBar QLabel{ color:#78909C; }
QTabWidget::pane{ border:1px solid #CFD8DC; border-radius:6px; background:white; }
QTabBar::tab{ padding:8px 16px; color:#546E7A; border:none; }
QTabBar::tab:selected{ color:#1976D2; border-bottom:2px solid #1976D2; font-weight:bold; }
"""


# ═══════════════════════════════════════════════════════════
# 共享状态 — 各 Page 通过同一 AppState 实例读写，避免全局变量
# ═══════════════════════════════════════════════════════════
class AppState:
    """
    应用级共享状态，由 MainWindow 创建一份实例并注入各 Page。

    Attributes:
        engine:         InferenceEngine 实例；HAS_ENGINE=False 时为 None
        results:        内存中的检测历史 List[Dict]，字段见 _ensure_result_meta
        config:         app_config.json 内容（路径、GPU 开关等）
        admin_unlocked: 是否已通过密码解锁「再训练 / 设置」页
    """

    def __init__(self):
        self.engine: Optional[InferenceEngine] = (
            InferenceEngine() if HAS_ENGINE else None
        )
        self.results: List[Dict] = []          # 进程内缓存，不自动落盘
        self.config:  Dict       = _load_cfg()
        self.admin_unlocked: bool = False
        if CFG_FILE.exists():
            print(f"[配置] 已加载: {CFG_FILE}")
        else:
            print(f"[配置] 未找到 {CFG_FILE.name}，使用代码默认参数；设置页保存后将创建该文件")


def _verify_admin_password(parent: QWidget) -> bool:
    """弹出密码框，验证通过返回 True。"""
    text, ok = QInputDialog.getText(
        parent,
        "管理员验证",
        "模型再训练与设置页面已锁定，请输入密码以解除只读：",
        QLineEdit.Password,
    )
    if not ok:
        return False
    if text != ADMIN_PAGE_PASSWORD:
        QMessageBox.warning(parent, "验证失败", "密码错误，仍保持只读模式。")
        return False
    return True


class AdminLockBar(QFrame):
    """
    再训练 / 设置页顶部的只读提示条。

    锁定态：橙色背景，显示「输入密码解锁」；
    解锁态：绿色背景，显示「重新锁定」。
    页面将 unlock_requested 转发为 admin_unlock_requested，由 MainWindow 统一验证密码。
    """

    unlock_requested = pyqtSignal()
    lock_requested   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.lbl = QLabel()
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl, 1)

        self.btn_unlock = _mk_btn("输入密码解锁", "warning")
        self.btn_unlock.clicked.connect(self.unlock_requested.emit)
        lay.addWidget(self.btn_unlock)

        self.btn_lock = _mk_btn("重新锁定", "flat", 96)
        self.btn_lock.clicked.connect(self.lock_requested.emit)
        lay.addWidget(self.btn_lock)

        self.set_locked(True)

    def set_locked(self, locked: bool) -> None:
        if locked:
            self.setStyleSheet(
                "AdminLockBar{background:#FFF3E0;border:1px solid #FFB74D;border-radius:6px;}"
            )
            self.lbl.setText("🔒 只读模式：不可修改训练参数或系统配置，如需操作请输入管理员密码。")
            self.lbl.setStyleSheet("color:#E65100;font-size:12px;")
            self.btn_unlock.setVisible(True)
            self.btn_lock.setVisible(False)
        else:
            self.setStyleSheet(
                "AdminLockBar{background:#E8F5E9;border:1px solid #81C784;border-radius:6px;}"
            )
            self.lbl.setText("🔓 已解除只读：可修改训练参数与系统设置。")
            self.lbl.setStyleSheet("color:#2E7D32;font-size:12px;")
            self.btn_unlock.setVisible(False)
            self.btn_lock.setVisible(True)


# ═══════════════════════════════════════════════════════════
# 后台工作线程 — 规则：耗时逻辑放 run()，UI 更新只通过 signal 回到主线程
# ═══════════════════════════════════════════════════════════
class InferenceWorker(QThread):
    """
    批量推理后台线程。

    Signals:
        result_item(int, dict) — 每完成一张图 emit (索引, 结果 dict)
        progress(int, int)     — (已完成数, 总数)
        done(list)             — 全部完成，携带结果列表（与 engine 返回值一致）

    注意:
        · 子线程内禁止直接操作 QTableWidget，仅 emit 信号
        · stop_flag=True 时 predict_batch 在下一批前退出（should_stop 回调）
        · 将 worker 挂到 self.worker 防止被 GC 回收导致崩溃
    """

    result_item = pyqtSignal(int, dict)
    progress    = pyqtSignal(int, int)
    done        = pyqtSignal(list)

    def __init__(self, engine, paths: List[str]):
        super().__init__()
        self.engine    = engine
        self.paths     = paths
        self.stop_flag = False

    def run(self):
        def _on_result(idx: int, r: dict):
            self.result_item.emit(idx, r)

        def _should_stop() -> bool:
            return self.stop_flag

        results = self.engine.predict_batch(
            self.paths,
            progress_cb=lambda c, t: self.progress.emit(c, t),
            result_cb=_on_result,
            should_stop=_should_stop,
        )
        self.done.emit(results)


    def stop(self):
        """请求中断：下一批推理前检查 stop_flag 并退出 run()。"""
        self.stop_flag = True


class TrainWorker(QThread):
    """
    训练子进程线程 — 避免在 GUI 主线程阻塞。

    使用 subprocess.Popen 捕获 stdout+stderr 合并流，逐行 emit log_line。
    Windows 下 train.py 使用 --num_workers 0 避免与 PyQt 多进程冲突。

    Signals:
        log_line(str)       — 单行训练日志
        finished(bool, str) — (成功与否, 摘要消息)
    """

    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, script: str, args: List[str]):
        super().__init__()
        self.script    = script
        self.args      = args
        self.stop_flag = False
        self._proc     = None

    def run(self):
        cmd = [sys.executable, self.script] + self.args
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1,
                encoding="utf-8", errors="replace",
            )
            for line in self._proc.stdout:
                if self.stop_flag:
                    self._proc.terminate()
                    break
                self.log_line.emit(line.rstrip())
            self._proc.wait()
            ok  = (self._proc.returncode == 0)
            msg = "训练完成 ✓" if ok else f"训练失败 (code={self._proc.returncode})"
            self.finished.emit(ok, msg)
        except Exception as exc:
            self.finished.emit(False, f"启动失败: {exc}")

    def stop(self):
        """终止训练子进程（用户点击「停止」或页面重新锁定时调用）。"""
        self.stop_flag = True
        if self._proc:
            self._proc.terminate()


    # ── 大图检测流水线线程 ──────────────────────────────────────
class SahiPipelineWorker(QThread):
    """
    SAHI 大图检测 + 缺陷分类后台线程。

    Signals:
        log_line(str)          — 单行日志
        progress(int, int)     — 细粒度进度 (当前步, 总步=图像数×4)
        stage_msg(str)         — 当前阶段文案
        image_done(int, dict)  — 单张完成 (序号, 统计字典)
        finished_all(list)     — 全部完成，携带所有统计字典列表
        error(str)             — 致命错误
    """

    log_line   = pyqtSignal(str)
    progress   = pyqtSignal(int, int)
    stage_msg  = pyqtSignal(str)
    image_done = pyqtSignal(int, dict)
    finished_all = pyqtSignal(list)
    error        = pyqtSignal(str)

    def __init__(
        self,
        yolo_path: str,
        img_paths: List[str],
        output_dir: str,
        classifier,
        device: str = "auto",
        slice_size: int = 1280,
        overlap: float = 0.20,
        det_conf: float = 0.35,
        batch_size: int = 8,
        crop_padding: int = 15,
        ios_thresh: float = 0.60,
        min_area_ratio: float = 0.45,
        max_aspect_ratio: float = 1.5,
        edge_filter: bool = True,
        edge_margin_px: int = 20,
    ):
        super().__init__()
        self.yolo_path    = yolo_path
        self.img_paths    = img_paths
        self.output_dir   = output_dir
        self.classifier   = classifier
        self.device       = device
        self.slice_size   = slice_size
        self.overlap      = overlap
        self.det_conf     = det_conf
        self.batch_size   = batch_size
        self.crop_padding = crop_padding
        self.ios_thresh        = ios_thresh
        self.min_area_ratio    = min_area_ratio
        self.max_aspect_ratio  = max_aspect_ratio
        self.edge_filter       = edge_filter
        self.edge_margin_px    = edge_margin_px
        self.stop_flag    = False

    def run(self):
        try:
            from sahi_detector import SahiDetector, SahiPipeline
        except ImportError as exc:
            self.error.emit(f"无法导入 sahi_detector: {exc}")
            return

        # ── 加载 YOLO 检测模型 ────────────────────────────────────
        detector = SahiDetector(
            model_path=self.yolo_path,
            device=self.device,
            slice_size=self.slice_size,
            overlap_ratio=self.overlap,
            conf=self.det_conf,
            batch_size=self.batch_size,
            ios_thresh=self.ios_thresh,
            min_area_ratio=self.min_area_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            edge_filter=self.edge_filter,
            edge_margin_px=self.edge_margin_px,
        )
        try:
            msg = detector.load()
            self.log_line.emit(msg)
        except Exception as exc:
            self.error.emit(f"YOLO 加载失败: {exc}")
            return

        if not self.classifier or not self.classifier.loaded:
            self.error.emit("分类引擎未加载，请先在「设置」页加载模型。")
            return

        pipeline = SahiPipeline(
            detector=detector,
            classifier=self.classifier,
            output_dir=self.output_dir,
            crop_padding=self.crop_padding,
        )

        # ── 逐张处理（每张 4 步：读图/检测/裁剪分类/保存）────────
        all_stats: List[dict] = []
        total = len(self.img_paths)
        steps_per_image = 4
        total_steps = max(1, total * steps_per_image)

        for i, img_path in enumerate(self.img_paths):
            if self.stop_flag:
                self.log_line.emit("已停止")
                self.stage_msg.emit("已停止")
                break

            name = Path(img_path).name
            self.stage_msg.emit(f"[{i + 1}/{total}] {name}")
            self.log_line.emit(f"━━━ [{i + 1}/{total}] {name} ━━━")

            def _log(msg, _i=i):
                self.log_line.emit(msg)

            def _should_stop():
                return self.stop_flag

            def _stage(msg: str, _i=i, _name=name):
                self.stage_msg.emit(f"[{_i + 1}/{total}] {msg}")

            def _progress(step: int, _total_stage: int, _i=i):
                # step 0..4 → 映射到该图已完成的步数
                done = _i * steps_per_image + min(max(step, 0), steps_per_image)
                self.progress.emit(done, total_steps)

            try:
                stats = pipeline.process_image(
                    img_path,
                    progress_cb=_progress,
                    log_cb=_log,
                    should_stop=_should_stop,
                    stage_cb=_stage,
                )
                all_stats.append(stats)
                self.image_done.emit(i, stats)
            except Exception as exc:
                err_stats = {
                    "image": Path(img_path).name,
                    "error": str(exc),
                    "total_diamonds": 0,
                    "defect_counts": {},
                }
                all_stats.append(err_stats)
                self.log_line.emit(f"处理失败: {exc}")
                self.progress.emit((i + 1) * steps_per_image, total_steps)

        self.stage_msg.emit("处理完成" if not self.stop_flag else "已停止")
        self.progress.emit(total_steps if all_stats and not self.stop_flag else
                           min(len(all_stats) * steps_per_image, total_steps),
                           total_steps)

        # ── 保存汇总 CSV ──────────────────────────────────────────
        if all_stats:
            try:
                import csv as _csv
                csv_path = Path(self.output_dir) / "summary.csv"
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = _csv.writer(f)
                    writer.writerow(["图像", "钻石数", "缺陷类别分布",
                                     "检测耗时(s)", "分类耗时(s)", "总耗时(s)"])
                    for s in all_stats:
                        dist = " | ".join(
                            f"{k}:{v}" for k, v in s.get("defect_counts", {}).items()
                        )
                        writer.writerow([
                            s.get("image", ""),
                            s.get("total_diamonds", 0),
                            dist,
                            s.get("detection_time_s", 0),
                            s.get("classification_time_s", 0),
                            s.get("total_time_s", 0),
                        ])
                self.log_line.emit(f"汇总 CSV: {csv_path}")
            except Exception as exc:
                self.log_line.emit(f"CSV 保存失败: {exc}")

        self.finished_all.emit(all_stats)

    def stop(self):
        self.stop_flag = True


# ═══════════════════════════════════════════════════════════
# 自定义小部件 — 拖放选图、缩略图悬停预览、右侧大图预览
# ═══════════════════════════════════════════════════════════
class ImageDropZone(QLabel):
    """
    图像拖放区：重写 mousePressEvent / dragEnterEvent / dropEvent。
    setAcceptDrops(True) 启用拖放；选图后 emit image_dropped(path)。
    """
    image_dropped = pyqtSignal(str)

    _IDLE_STYLE = ("QLabel{border:2px dashed #90A4AE;border-radius:8px;"
                   "color:#90A4AE;font-size:13px;background:#FAFAFA;}")
    _LOAD_STYLE = "QLabel{border:2px solid #42A5F5;border-radius:8px;background:#F5F8FF;}"

    def __init__(self, size=280, parent=None):
        super().__init__(parent)
        self._size     = size
        self._img_path: Optional[str] = None
        self.setAcceptDrops(True)
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setText("点击选择图像\n或拖入此处")
        self.setStyleSheet(self._IDLE_STYLE)

    # ── 事件 ──
    def mousePressEvent(self, _):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if path:
            self._load(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if Path(p).suffix.lower() in IMG_EXTS:
                self._load(p)

    def _load(self, path: str):
        self._img_path = path
        px = QPixmap(path).scaled(
            self._size - 8, self._size - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(px)
        self.setStyleSheet(self._LOAD_STYLE)
        self.image_dropped.emit(path)

    def reset(self):
        self._img_path = None
        self.clear()
        self.setText("点击选择图像\n或拖入此处")
        self.setStyleSheet(self._IDLE_STYLE)

    @property
    def image_path(self) -> Optional[str]:
        return self._img_path


def _thumb(path: str, size: int = 64) -> "ThumbnailLabel":
    """返回可悬停预览、单击打开的缩略图控件。"""
    return ThumbnailLabel(path, size)


def _open_image(path: str) -> None:
    """用系统默认程序打开图像文件。"""
    p = Path(path)
    if p.is_file():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))


def _open_folder(path: str) -> None:
    """用系统文件管理器打开文件夹。"""
    p = Path(path)
    if p.is_dir():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))


class ThumbnailLabel(QLabel):
    """缩略图：悬停右侧预览，单击打开原图文件。"""

    preview_hover = pyqtSignal(str)
    preview_leave = pyqtSignal()
    clicked_path  = pyqtSignal(str)

    def __init__(self, path: str, size: int = 64, parent=None):
        super().__init__(parent)
        self._path = path
        self._size = size
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("悬停预览 · 单击打开原图")
        px = QPixmap(path)
        if not px.isNull():
            px = px.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(px)

    def enterEvent(self, event):
        if self._path and Path(self._path).is_file():
            self.preview_hover.emit(self._path)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.preview_leave.emit()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._path:
            self.clicked_path.emit(self._path)
        super().mousePressEvent(event)


class HoverFilenameLabel(QLabel):
    """文件名列：悬停预览，单击固定右侧预览。"""

    preview_hover = pyqtSignal(str)
    preview_leave = pyqtSignal()
    pin_requested = pyqtSignal(str)

    def __init__(self, path: str, text: str, parent=None):
        super().__init__(text, parent)
        self._path = path
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("悬停预览 · 单击固定右侧预览")

    def enterEvent(self, event):
        if self._path and Path(self._path).is_file():
            self.preview_hover.emit(self._path)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.preview_leave.emit()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._path:
            self.pin_requested.emit(self._path)
        super().mousePressEvent(event)


class SidePreviewController(QObject):
    """
    右侧预览区控制器：协调「悬停临时预览」与「单击固定预览」。

    状态机:
      · show_transient — 鼠标悬停缩略图/文件名时显示，离开 300ms 后隐藏
      · pin            — 用户单击文件名固定；悬停离开时不隐藏，恢复固定图
      · unpin          — 取消固定，清空预览
    """

    def __init__(self, preview: "ImagePreviewSidePanel"):
        super().__init__(preview)
        self.preview = preview
        self.pinned_path: Optional[str] = None
        self.pinned_meta: str = ""
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._on_hide_timeout)

    def show_transient(self, path: str, meta: str = "") -> None:
        self._timer.stop()
        self.preview.show_image(path, meta)

    def pin(self, path: str, meta: str = "") -> None:
        self._timer.stop()
        self.pinned_path = path
        self.pinned_meta = meta
        self.preview.set_pinned(True)
        self.preview.show_image(path, meta)

    def unpin(self) -> None:
        self.pinned_path = None
        self.pinned_meta = ""
        self.preview.set_pinned(False)
        self.preview.clear_preview()

    def schedule_hide(self) -> None:
        self._timer.start()

    def cancel_hide(self) -> None:
        self._timer.stop()

    def _on_hide_timeout(self) -> None:
        if self.pinned_path and Path(self.pinned_path).is_file():
            self.preview.show_image(self.pinned_path, self.pinned_meta)
            self.preview.set_pinned(True)
        else:
            self.preview.clear_preview()


class ImagePreviewSidePanel(QFrame):
    """表格右侧原图预览区。"""

    def __init__(self, min_size: int = 300, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedWidth(min_size + 24)
        self._current_path = ""
        self._controller: Optional[SidePreviewController] = None
        self._min_size = min_size

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        hdr = QHBoxLayout()
        self.title_lbl = QLabel("原图预览")
        self.title_lbl.setStyleSheet("font-size:13px;font-weight:bold;color:#455A64;")
        hdr.addWidget(self.title_lbl)
        hdr.addStretch()
        self.pin_lbl = QLabel("")
        self.pin_lbl.setStyleSheet("color:#1976D2;font-size:11px;font-weight:bold;")
        hdr.addWidget(self.pin_lbl)
        lay.addLayout(hdr)

        self.image_lbl = QLabel("悬停缩略图或文件名\n查看原图")
        self.image_lbl.setAlignment(Qt.AlignCenter)
        self.image_lbl.setMinimumSize(min_size, min_size)
        self.image_lbl.setStyleSheet(
            "background:#FAFAFA;border:1px dashed #CFD8DC;border-radius:6px;"
            "color:#90A4AE;font-size:13px;"
        )
        self.image_lbl.setWordWrap(True)
        lay.addWidget(self.image_lbl)

        self.path_lbl = QLabel("")
        self.path_lbl.setStyleSheet("color:#37474F;font-size:12px;font-weight:bold;")
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.path_lbl)

        self.meta_lbl = QLabel("")
        self.meta_lbl.setStyleSheet("color:#78909C;font-size:11px;")
        self.meta_lbl.setWordWrap(True)
        self.meta_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.meta_lbl)

        btn_open = _mk_btn("打开原图", "primary")
        btn_open.clicked.connect(self._open_current)
        lay.addWidget(btn_open)

        btn_unpin = _mk_btn("取消固定", "flat")
        btn_unpin.clicked.connect(self._request_unpin)
        lay.addWidget(btn_unpin)
        lay.addStretch()

    def bind_controller(self, controller: SidePreviewController) -> None:
        self._controller = controller

    def set_pinned(self, pinned: bool) -> None:
        self.pin_lbl.setText("已固定" if pinned else "")

    def enterEvent(self, event):
        if self._controller:
            self._controller.cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._controller:
            self._controller.schedule_hide()
        super().leaveEvent(event)

    def show_image(self, path: str, meta: str = "") -> None:
        self._current_path = path
        px = QPixmap(path)
        if px.isNull():
            self.image_lbl.setText("无法加载图像")
            self.path_lbl.setText(Path(path).name)
            self.meta_lbl.setText(path)
            return
        side = self._min_size - 8
        scaled = px.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_lbl.setPixmap(scaled)
        self.image_lbl.setStyleSheet(
            "background:#1A1A2E;border:1px solid #37474F;border-radius:6px;"
        )
        self.path_lbl.setText(Path(path).name)
        self.meta_lbl.setText(meta or f"尺寸 {px.width()}×{px.height()} px")

    def clear_preview(self) -> None:
        self._current_path = ""
        self.image_lbl.clear()
        self.image_lbl.setText("悬停缩略图或文件名\n查看原图")
        self.image_lbl.setStyleSheet(
            "background:#FAFAFA;border:1px dashed #CFD8DC;border-radius:6px;"
            "color:#90A4AE;font-size:13px;"
        )
        self.path_lbl.setText("")
        self.meta_lbl.setText("")
        self.set_pinned(False)

    def _open_current(self) -> None:
        if self._current_path and Path(self._current_path).is_file():
            _open_image(self._current_path)

    def _request_unpin(self) -> None:
        if self._controller:
            self._controller.unpin()


# 兼容旧引用
PreviewHoverController = SidePreviewController
ImagePreviewBar = ImagePreviewSidePanel


def _connect_thumb_preview(
    thumb: ThumbnailLabel,
    controller: SidePreviewController,
    meta: str = "",
) -> None:
    """缩略图悬停预览；单击打开系统默认看图程序。"""
    thumb.preview_hover.connect(
        lambda p, m=meta: controller.show_transient(p, m)
    )
    thumb.preview_leave.connect(controller.schedule_hide)
    thumb.clicked_path.connect(_open_image)


def _connect_filename_preview(
    label: HoverFilenameLabel,
    controller: SidePreviewController,
    meta: str = "",
) -> None:
    """文件名悬停预览；单击固定右侧预览。"""
    label.preview_hover.connect(
        lambda p, m=meta: controller.show_transient(p, m)
    )
    label.preview_leave.connect(controller.schedule_hide)
    label.pin_requested.connect(
        lambda p, m=meta: controller.pin(p, m)
    )


def _mk_btn(text: str, cls: str = "flat", width: int = 0) -> QPushButton:
    """工厂：创建带 QSS class 属性的按钮（primary / success / flat 等）。"""
    btn = QPushButton(text)
    btn.setProperty("class", cls)  # 对应 STYLE 里 QPushButton.primary 等选择器
    if width:
        btn.setFixedWidth(width)
    return btn


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#E0E0E0;")
    return f


def _tune_form_layout(fl: QFormLayout, label_min_width: int = 136) -> None:
    """统一 FormLayout 行距与标签列宽，避免参数名/数值被截断。"""
    fl.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    fl.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
    fl.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    fl.setRowWrapPolicy(QFormLayout.DontWrapRows)
    fl.setHorizontalSpacing(14)
    fl.setVerticalSpacing(10)
    for row in range(fl.rowCount()):
        item = fl.itemAt(row, QFormLayout.LabelRole)
        if item and item.widget():
            item.widget().setMinimumWidth(label_min_width)


def _tune_spinbox(sb: QWidget, min_width: int = 112) -> None:
    """为数值控件设置最小宽度，保证完整显示。"""
    sb.setMinimumWidth(min_width)
    sb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


def _normalize_image_path(path: str) -> str:
    """统一结果中的图像路径（resolve + 大小写不敏感比较），避免重复记录。"""
    try:
        return str(Path(path).resolve())
    except (OSError, ValueError):
        return str(Path(path))


# ── 检测结果 dict 字段约定 ────────────────────────────────────────────────────
# 引擎产出（inference_common.build_result_dict / logits_row_to_result）:
#   path, class, confidence, max_class, max_confidence, all_scores, elapsed_ms
# UI / 工作流扩展（_ensure_result_meta 补全）:
#   true_class         — 用户标注的正确类别（列表查看模式）
#   flagged            — True 表示已送入误分类修正页，ResultsPage 不再显示
#   correction_saved   — True 表示已复制到 corrections/（归档完成）
#   _checked           — 结果管理表格勾选状态，导出时 only_checked 会读取
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_result_meta(r: Dict) -> Dict:
    """补全结果 dict 中的 UI 管理字段，并规范化 path。"""
    r.setdefault("true_class", "")
    r.setdefault("flagged", False)
    r.setdefault("correction_saved", False)
    r.setdefault("_checked", True)
    if "path" in r:
        r["path"] = _normalize_image_path(r["path"])
    return r


def _upsert_result(results: List[Dict], r: Dict) -> None:
    """
    按规范化路径 upsert：同一张图重复检测时更新记录而非追加。

    保留字段: true_class, flagged, correction_saved, _checked
    （避免重新检测覆盖用户已做的修正/勾选状态）
    """
    r = _ensure_result_meta(r)
    key = r["path"].lower()
    for i, existing in enumerate(results):
        if _normalize_image_path(existing["path"]).lower() == key:
            for field in ("true_class", "flagged", "correction_saved", "_checked"):
                r[field] = existing.get(field, r.get(field))
            results[i] = r
            return
    results.append(r)


def _is_results_list_item(r: Dict) -> bool:
    """
    是否仍在「结果管理」页展示。

    flagged=True 的条目转入误分类修正页；correction_saved=True 的已归档移除。
    """
    return not r.get("flagged") and not r.get("correction_saved")


def count_flagged_pending(results: List[Dict]) -> int:
    """误分类修正页待处理数量。"""
    return sum(
        1 for r in results
        if r.get("flagged") and not r.get("correction_saved")
    )


def _class_button_font_px(name: str) -> int:
    """按类别名长度估算字号，使文字尽量贴近按钮边框。"""
    n = len(name)
    if n <= 2:
        return 28
    if n <= 4:
        return 24
    if n <= 6:
        return 20
    return 17


def _class_button_stylesheet(selected: bool, font_px: int) -> str:
    if selected:
        return (
            f"QPushButton{{background:#E8F5E9;color:#1B5E20;border:3px solid #43A047;"
            f"border-radius:10px;font-size:{font_px}px;font-weight:bold;padding:4px 6px;}}"
        )
    return (
        f"QPushButton{{background:#FFFFFF;color:#0D47A1;border:3px solid #42A5F5;"
        f"border-radius:10px;font-size:{font_px}px;font-weight:bold;padding:4px 6px;}}"
        f"QPushButton:hover{{background:#E3F2FD;border-color:#1976D2;}}"
        f"QPushButton:pressed{{background:#BBDEFB;}}"
    )


def _save_correction_to_disk(state: AppState, r: Dict, true_cls: str) -> bool:
    """
    将误分类样本复制到 corrections_dir/true_cls/。

    不删除原图；重名时 _unique_dest_path 追加 _1、_2 后缀。
    成功后设置 r['correction_saved']=True（列表查看模式可能暂不 pop）。
    """
    src = Path(r["path"])
    if not src.is_file():
        return False
    corrections_dir = Path(state.config.get("corrections_dir", "corrections"))
    dst_dir = corrections_dir / true_cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique_dest_path(dst_dir, src.name)
    shutil.copy2(src, dst)
    r["true_class"] = true_cls
    r["correction_saved"] = True
    return True


def _archive_results_at_indices(
    state: AppState, indices: List[int], true_cls: str,
) -> int:
    """归档到 corrections/类别/ 并从检测结果列表移除；返回成功数。"""
    saved = 0
    for idx in sorted(set(indices), reverse=True):
        if not (0 <= idx < len(state.results)):
            continue
        r = state.results[idx]
        if not _save_correction_to_disk(state, r, true_cls):
            continue
        state.results.pop(idx)
        saved += 1
    return saved


def _unique_dest_path(dst_dir: Path, filename: str) -> Path:
    """目标目录下生成不重复的文件路径（重名时追加 _1、_2 …）。"""
    dst = dst_dir / filename
    if not dst.exists():
        return dst
    stem, suf = Path(filename).stem, Path(filename).suffix
    n = 1
    while dst.exists():
        dst = dst_dir / f"{stem}_{n}{suf}"
        n += 1
    return dst


def _pick_export_directory(parent: QWidget, default_name: str = "") -> Optional[Path]:
    """
    选择导出上级目录，并确认可编辑的子文件夹名（默认为批量检测所选文件夹名）。
    返回完整导出路径；用户取消则返回 None。
    """
    parent_dir = QFileDialog.getExistingDirectory(
        parent,
        "选择导出位置（上级目录）",
    )
    if not parent_dir:
        return None

    safe_default = (default_name or "导出结果").strip()
    for ch in '<>:"/\\|?*':
        safe_default = safe_default.replace(ch, "_")
    if not safe_default:
        safe_default = "导出结果"

    name, ok = QInputDialog.getText(
        parent,
        "导出文件夹名称",
        f"将在以下目录下创建子文件夹：\n{parent_dir}\n\n文件夹名称（可修改）：",
        QLineEdit.Normal,
        safe_default,
    )
    if not ok:
        return None
    name = name.strip()
    if not name:
        QMessageBox.warning(parent, "提示", "文件夹名称不能为空。")
        return None
    if any(c in name for c in '<>:"/\\|?*'):
        QMessageBox.warning(parent, "提示", "文件夹名称不能包含 \\ / : * ? \" < > | 等字符。")
        return None
    return Path(parent_dir) / name


def export_classified_images(
    parent: QWidget,
    results: List[Dict],
    *,
    only_checked: bool = True,
    default_export_name: str = "",
) -> bool:
    """
    按预测类别分文件夹导出图像（复制，非移动）。

    导出条件:
      · 未 flagged（非待修正）
      · 预测类别有效且非 ERROR
      · 源文件系统存在
      · only_checked=True 时须表格勾选（结果管理页）；单张/批量页传 False

    附带 export_manifest.csv 记录源路径与导出路径。
    """
    candidates: List[Dict] = []
    for r in results:
        if only_checked and not r.get("_checked", True):
            continue
        if r.get("flagged"):
            continue
        cls = r.get("class", "")
        if not cls or cls == "ERROR":
            continue
        if not Path(r["path"]).is_file():
            continue
        candidates.append(r)

    if not candidates:
        QMessageBox.warning(
            parent,
            "无法导出",
            "没有可导出的图像。\n\n"
            "请确保：条目已勾选、未标记为「待修正」、预测成功且源文件存在。\n"
            "误分类样本请先在「误分类修正」页归档。",
        )
        return False

    dest_path = _pick_export_directory(parent, default_export_name)
    if dest_path is None:
        return False

    reply = QMessageBox.question(
        parent,
        "确认导出",
        f"将把 {len(candidates)} 张确认识别无误的图像复制到：\n{dest_path}\n\n"
        "按预测类别分子文件夹存放，并生成 export_manifest.csv。\n\n是否继续？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    if reply != QMessageBox.Yes:
        return False

    try:
        dest_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        QMessageBox.warning(parent, "无法创建目录", f"导出目录创建失败：{exc}")
        return False
    manifest_rows: List[List] = []
    by_class: Dict[str, int] = {}
    copied = 0
    failed = 0

    for r in candidates:
        cls = r["class"]
        src = Path(r["path"])
        dst_dir = dest_path / cls
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = _unique_dest_path(dst_dir, src.name)
            shutil.copy2(src, dst)
            copied += 1
            by_class[cls] = by_class.get(cls, 0) + 1
            manifest_rows.append([
                str(src), src.name, cls,
                f"{r['confidence']:.4f}", str(dst),
            ])
        except OSError:
            failed += 1

    csv_path = dest_path / "export_manifest.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["源文件路径", "文件名", "预测类别", "置信度", "导出路径"])
            writer.writerows(manifest_rows)
    except OSError as exc:
        QMessageBox.warning(parent, "清单写入失败", f"图像已复制，但 CSV 保存失败：{exc}")

    summary = "\n".join(f"  · {c}：{n} 张" for c, n in sorted(by_class.items()))
    msg = f"已导出 {copied} 张图像至：\n{dest_path}\n\n{summary}\n\n清单：export_manifest.csv"
    if failed:
        msg += f"\n\n{failed} 张复制失败（权限或磁盘空间问题）。"
    QMessageBox.information(parent, "导出完成", msg)
    return copied > 0


def _export_results_csv(parent: QWidget, results: List[Dict]) -> None:
    """导出全部检测记录为 CSV（含待修正标记，不复制图像）。"""
    if not results:
        QMessageBox.warning(parent, "提示", "当前没有可导出的检测结果。")
        return
    path, _ = QFileDialog.getSaveFileName(
        parent, "导出 CSV", "results.csv", "CSV (*.csv)"
    )
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件路径", "文件名", "预测类别", "置信度", "修正类别", "已标记待修正"])
        for r in results:
            writer.writerow([
                r["path"], Path(r["path"]).name,
                r["class"], f"{r['confidence']:.4f}",
                r.get("true_class", ""), r.get("flagged", False),
            ])
    QMessageBox.information(parent, "导出成功", f"已保存至：{path}")


# ═══════════════════════════════════════════════════════════
# 页面 1 ── 缺陷检测（QTabWidget：单张 / 批量）
# ═══════════════════════════════════════════════════════════
class DetectionPage(QWidget):
    """
    缺陷检测页 — QTabWidget 两个 Tab。

    单张 Tab:
      · ImageDropZone 选图/拖放 → engine.predict（主线程，短任务）
      · 结果写入 AppState.results 并 upsert
      · 显示逐类得分条；confidence 为阈值决策后的预测类概率

    批量 Tab:
      · 递归扫描文件夹内 IMG_EXTS
      · InferenceWorker + predict_batch（子线程）
      · 注意: 开始批量时会 state.results.clear()，清空此前所有检测历史
    """

    results_ready  = pyqtSignal(list)  # MainWindow → ResultsPage.refresh
    navigate_to    = pyqtSignal(int)   # 批量完成后跳转结果管理页

    def __init__(self, state: AppState):
        super().__init__()
        self.state  = state
        self.worker: Optional[InferenceWorker] = None
        self._last_single_result: Optional[Dict] = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("缺陷检测")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        root.addWidget(title)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # ─── 单图检测 Tab ────────────────────────────
        single = QWidget()
        sl = QHBoxLayout(single)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(20)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.drop = ImageDropZone(280)
        self.drop.image_dropped.connect(lambda _: None)
        left.addWidget(self.drop)
        self.btn_detect = _mk_btn("▶  开始检测", "primary")
        self.btn_detect.setFixedHeight(36)
        self.btn_detect.clicked.connect(self._run_single)
        left.addWidget(self.btn_detect)
        self.btn_export_single = _mk_btn("导出分类结果", "success")
        self.btn_export_single.setFixedHeight(36)
        self.btn_export_single.setEnabled(False)
        self.btn_export_single.clicked.connect(self._export_single_result)
        left.addWidget(self.btn_export_single)
        sl.addLayout(left)

        # 结果面板
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMinimumWidth(270)
        rl = QVBoxLayout(panel)
        rl.setContentsMargins(18, 16, 18, 16)
        rl.setSpacing(8)

        lbl_h = QLabel("预测结果")
        lbl_h.setStyleSheet("color:#90A4AE;font-size:12px;")
        self.lbl_cls  = QLabel("—")
        self.lbl_cls.setStyleSheet("font-size:28px;font-weight:bold;color:#1565C0;")
        self.lbl_cls.setAlignment(Qt.AlignCenter)
        self.lbl_conf = QLabel("置信度: —")
        self.lbl_conf.setStyleSheet("color:#546E7A;font-size:13px;")
        self.lbl_conf.setAlignment(Qt.AlignCenter)

        rl.addWidget(lbl_h)
        rl.addWidget(self.lbl_cls)
        rl.addWidget(self.lbl_conf)
        rl.addWidget(_sep())

        lbl_all = QLabel("全类别得分")
        lbl_all.setStyleSheet("color:#90A4AE;font-size:12px;")
        rl.addWidget(lbl_all)

        self.score_box = QWidget()
        self.score_lay = QVBoxLayout(self.score_box)
        self.score_lay.setSpacing(5)
        self.score_lay.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.score_box)
        rl.addStretch()

        self.lbl_ms = QLabel("")
        self.lbl_ms.setStyleSheet("color:#B0BEC5;font-size:11px;")
        self.lbl_ms.setAlignment(Qt.AlignRight)
        rl.addWidget(self.lbl_ms)

        sl.addWidget(panel)
        sl.setStretch(0, 0)
        sl.setStretch(1, 1)
        tabs.addTab(single, "单张检测")

        # ─── 批量检测 Tab ────────────────────────────
        batch = QWidget()
        bl = QVBoxLayout(batch)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(10)

        # 文件夹选择行
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("选择图像文件夹...")
        self.folder_edit.setReadOnly(True)
        btn_browse = _mk_btn("浏览", "flat", 70)
        btn_browse.clicked.connect(self._browse)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(btn_browse)
        bl.addLayout(folder_row)

        # 控制行
        ctrl = QHBoxLayout()
        self.btn_batch = _mk_btn("▶  批量检测", "primary")
        self.btn_batch.setFixedHeight(34)
        self.btn_batch.clicked.connect(self._run_batch)
        self.btn_stop = _mk_btn("■  停止", "danger", 80)
        self.btn_stop.setFixedHeight(34)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_batch)
        self.lbl_cnt = QLabel("未选择文件夹")
        self.lbl_cnt.setStyleSheet("color:#546E7A;")
        ctrl.addWidget(self.btn_batch)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        ctrl.addWidget(self.lbl_cnt)
        bl.addLayout(ctrl)

        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        self.pbar.setFixedHeight(12)
        bl.addWidget(self.pbar)

        # 批量结果预览表
        self.btable = QTableWidget(0, 4)
        self.btable.setHorizontalHeaderLabels(["文件名", "预测类别", "置信度", "耗时(ms)"])
        self.btable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.btable.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.btable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.btable.setAlternatingRowColors(True)
        self.btable.setSelectionBehavior(QAbstractItemView.SelectRows)
        bl.addWidget(self.btable)

        nav_row = QHBoxLayout()
        self.btn_export_batch = _mk_btn("导出分类结果", "success")
        self.btn_export_batch.setEnabled(False)
        self.btn_export_batch.clicked.connect(self._export_batch_results)
        nav_row.addWidget(self.btn_export_batch)
        nav_row.addStretch()
        btn_go = _mk_btn("查看全部结果 →", "primary")
        btn_go.clicked.connect(lambda: self.navigate_to.emit(NAV_RESULTS))
        nav_row.addWidget(btn_go)
        bl.addLayout(nav_row)

        tabs.addTab(batch, "批量检测")

    # ── 单图检测（主线程同步推理，按钮禁用 + processEvents 防界面假死）──
    def _run_single(self):
        if not self.drop.image_path:
            QMessageBox.warning(self, "提示", "请先选择或拖入一张图像。")
            return
        if not (self.state.engine and self.state.engine.loaded):
            QMessageBox.warning(self, '提示', '模型未加载，请前往「设置」页面加载模型。')
            return
        self.btn_detect.setEnabled(False)
        self.btn_detect.setText("检测中…")
        QApplication.processEvents()
        try:
            r = self.state.engine.predict(self.drop.image_path)
            r = _ensure_result_meta(r)
            self._last_single_result = r
            _upsert_result(self.state.results, r)
            self._show_single(r)
            self.btn_export_single.setEnabled(True)
            self.results_ready.emit([r])
        except Exception as e:
            QMessageBox.critical(self, "检测出错", str(e))
        finally:
            self.btn_detect.setEnabled(True)
            self.btn_detect.setText("▶  开始检测")

    def _show_single(self, r: Dict):
        """渲染单张结果面板：类别颜色按 confidence 分档，得分条高亮预测类。"""
        conf  = r["confidence"]
        color = "#1B5E20" if conf >= 0.8 else ("#E65100" if conf >= 0.5 else "#B71C1C")
        self.lbl_cls.setText(r["class"])
        self.lbl_cls.setStyleSheet(f"font-size:28px;font-weight:bold;color:{color};")
        conf_text = f"置信度：{conf*100:.1f}%"
        max_cls = r.get("max_class", r["class"])
        max_conf = r.get("max_confidence", conf)
        if max_cls != r["class"]:
            # 逐类阈值决策后，预测类可能不是 argmax 最高分（见 class_thresholds.json）
            conf_text += f"（阈值决策；最高得分：{max_cls} {max_conf*100:.1f}%）"
        self.lbl_conf.setText(conf_text)
        self.lbl_ms.setText(f"推理耗时：{r['elapsed_ms']:.1f} ms")

        # 清空旧得分条
        while self.score_lay.count():
            w = self.score_lay.takeAt(0).widget()
            if w:
                w.deleteLater()

        for cls_name, score in sorted(r["all_scores"].items(), key=lambda x: -x[1]):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(6)
            nm = QLabel(cls_name)
            nm.setFixedWidth(88)
            nm.setStyleSheet("font-size:12px;color:#455A64;")
            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setValue(int(score * 100))
            pb.setTextVisible(False)
            pb.setFixedHeight(8)
            if cls_name == r["class"]:
                pb.setStyleSheet("QProgressBar::chunk{background:#1976D2;border-radius:4px;}")
            sc = QLabel(f"{score*100:.1f}%")
            sc.setFixedWidth(40)
            sc.setStyleSheet("font-size:11px;color:#78909C;")
            row_l.addWidget(nm)
            row_l.addWidget(pb)
            row_l.addWidget(sc)
            self.score_lay.addWidget(row_w)

    # ── 批量检测（子线程 InferenceWorker）────────────────────
    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.folder_edit.setText(folder)
            imgs = [f for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXTS]
            self.lbl_cnt.setText(f"共 {len(imgs)} 张图像")

    def _run_batch(self):
        folder = self.folder_edit.text()
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self, "提示", "请先选择有效的文件夹。")
            return
        if not (self.state.engine and self.state.engine.loaded):
            QMessageBox.warning(self, '提示', '模型未加载，请前往「设置」页面加载模型。')
            return
        imgs = sorted(
            _normalize_image_path(str(f))
            for f in Path(folder).rglob("*")
            if f.suffix.lower() in IMG_EXTS
        )
        if not imgs:
            QMessageBox.warning(self, "提示", "所选文件夹中没有支持的图像文件。")
            return

        self.btable.setRowCount(0)
        self.state.results.clear()   # 批量检测视为新会话，不保留旧结果
        self.btn_export_batch.setEnabled(False)
        self.pbar.setVisible(True)
        self.pbar.setMaximum(len(imgs))
        self.pbar.setValue(0)
        self.btn_batch.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = InferenceWorker(self.state.engine, imgs)
        self.worker.result_item.connect(self._on_item)
        self.worker.progress.connect(lambda c, _: self.pbar.setValue(c))
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_item(self, _: int, r: Dict):
        r = _ensure_result_meta(r)
        _upsert_result(self.state.results, r)
        row = self.btable.rowCount()
        self.btable.insertRow(row)
        items = [
            Path(r["path"]).name,
            r["class"],
            f"{r['confidence']*100:.1f}%",
            f"{r['elapsed_ms']:.1f}",
        ]
        for c, text in enumerate(items):
            it = QTableWidgetItem(text)
            it.setTextAlignment(Qt.AlignCenter if c else Qt.AlignLeft | Qt.AlignVCenter)
            self.btable.setItem(row, c, it)
        self.btable.scrollToBottom()

    def _on_done(self, results: List[Dict]):
        self.pbar.setVisible(False)
        self.btn_batch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_cnt.setText(f"完成 {len(results)} 张")
        self.btn_export_batch.setEnabled(len(results) > 0)
        self.results_ready.emit(results)

    def _export_single_result(self):
        if not self._last_single_result:
            QMessageBox.warning(self, "提示", "请先完成单张检测。")
            return
        src = Path(self._last_single_result["path"])
        default_name = src.parent.name if src.parent.name else src.stem
        export_classified_images(
            self, [self._last_single_result], only_checked=False,
            default_export_name=default_name,
        )

    def _export_batch_results(self):
        if not self.state.results:
            QMessageBox.warning(self, "提示", "请先完成批量检测。")
            return
        folder = self.folder_edit.text().strip()
        default_name = Path(folder).name if folder else "导出结果"
        export_classified_images(
            self, self.state.results, only_checked=False,
            default_export_name=default_name,
        )

    def _stop_batch(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag = True
            self.btn_stop.setEnabled(False)


# ═══════════════════════════════════════════════════════════
# 页面 2 ── 结果管理（QHBoxLayout：表格 + ImagePreviewSidePanel）
# ═══════════════════════════════════════════════════════════
class ResultsPage(QWidget):
    """
    检测结果管理页 — 表格 + 右侧预览 + 快捷归档按钮。

    表格列: 勾选 | 缩略图 | 文件名 | 预测类别 | 置信度 | 送修正
    筛选:   类别下拉 + 「置信度 ≤」滑块（聚焦低置信度样本）
    归档:   右侧大按钮 → 复制到 corrections/类别/ 并从 results 移除

    工作流标记:
      flagged=False, correction_saved=False → 本页可见
      flagged=True                          → 转入 CorrectionPage，本页隐藏
      correction_saved=True                 → 已归档，两页均不可见

    表格行通过 Qt.UserRole 存储 results 中的原始索引（筛选后仍指向正确条目）。
    """

    correction_updated = pyqtSignal(str)   # 状态栏提示 + 刷新修正页角标

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._conf_sort_order: Optional[str] = None  # None | "asc" | "desc"
        self._archive_class_btns: List[QPushButton] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        # 标题行
        hdr = QHBoxLayout()
        title = QLabel("检测结果管理")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_stat = QLabel("共 0 条")
        self.lbl_stat.setStyleSheet("color:#546E7A;")
        hdr.addWidget(self.lbl_stat)
        root.addLayout(hdr)

        # 工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        QLabel_f = QLabel("类别筛选:")
        QLabel_f.setStyleSheet("color:#546E7A;")
        toolbar.addWidget(QLabel_f)
        self.filter_cls = QComboBox()
        self.filter_cls.addItem("全部")
        self.filter_cls.setFixedWidth(120)
        self.filter_cls.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_cls)

        QLabel_c = QLabel("置信度 ≤")
        QLabel_c.setStyleSheet("color:#546E7A;")
        toolbar.addWidget(QLabel_c)
        self.filter_conf = QDoubleSpinBox()
        self.filter_conf.setRange(0.0, 1.0)
        self.filter_conf.setSingleStep(0.05)
        self.filter_conf.setValue(1.0)
        self.filter_conf.setFixedWidth(150)
        self.filter_conf.setToolTip(
            "显示置信度不超过该值的记录。先选类别，再调低此值可聚焦低置信度样本。"
        )
        self.filter_conf.valueChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_conf)

        toolbar.addStretch()

        self._all_checked = True
        self.btn_toggle_sel = _mk_btn("全不选", "flat", 88)
        self.btn_toggle_sel.clicked.connect(self._toggle_all_checks)
        toolbar.addWidget(self.btn_toggle_sel)

        btn_flag_sel = _mk_btn("送修正页", "warning")
        btn_flag_sel.setToolTip("将勾选的图像送入「误分类修正」页，并从本列表移除")
        btn_flag_sel.clicked.connect(self._flag_selected)
        toolbar.addWidget(btn_flag_sel)

        btn_flag_all = _mk_btn("当前筛选全部送修正", "warning")
        btn_flag_all.setToolTip("将当前筛选可见项全部送入误分类修正页")
        btn_flag_all.clicked.connect(self._flag_visible)
        toolbar.addWidget(btn_flag_all)

        btn_export = _mk_btn("导出分类结果", "success")
        btn_export.clicked.connect(self._export_classified)
        toolbar.addWidget(btn_export)

        btn_csv = _mk_btn("导出 CSV", "flat")
        btn_csv.clicked.connect(self._export_csv)
        toolbar.addWidget(btn_csv)

        root.addLayout(toolbar)

        lbl_export_hint = QLabel(
            "筛选：先选「类别」，再调「置信度 ≤」可聚焦低置信度样本。"
            "  点击表头「置信度 ↕」可排序。"
            "  右侧点击正确类别可直接归档；或「送修正页」后在误分类修正页处理。"
        )
        lbl_export_hint.setStyleSheet("color:#78909C;font-size:11px;")
        root.addWidget(lbl_export_hint)

        content = QHBoxLayout()
        content.setSpacing(12)

        self.table = QTableWidget(0, 6)
        self.table.setObjectName("resultsTable")
        self._set_table_header_labels()
        hdr = self.table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.sectionClicked.connect(self._on_table_header_clicked)
        hdr.setToolTip("点击「置信度 ↕」列标题，可在高→低 / 低→高之间切换排序")
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(1, 72)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 80)
        self.table.setColumnWidth(5, 100)
        self.table.setRowHeight(0, 72)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_check_changed)
        self.table.itemSelectionChanged.connect(self._on_row_selection_changed)
        content.addWidget(self.table, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        self.preview = ImagePreviewSidePanel(min_size=280)
        self._preview_ctrl = SidePreviewController(self.preview)
        self.preview.bind_controller(self._preview_ctrl)
        right_col.addWidget(self.preview, 1)

        archive_box = QFrame()
        archive_box.setObjectName("card")
        ab_lay = QVBoxLayout(archive_box)
        ab_lay.setContentsMargins(10, 10, 10, 10)
        ab_lay.setSpacing(8)
        self.lbl_archive_hint = QLabel("勾选或选中图像 → 点击正确类别归档")
        self.lbl_archive_hint.setStyleSheet(
            "color:#455A64;font-size:12px;font-weight:bold;"
        )
        self.lbl_archive_hint.setWordWrap(True)
        ab_lay.addWidget(self.lbl_archive_hint)
        self.archive_class_host = QWidget()
        self.archive_class_grid = QGridLayout(self.archive_class_host)
        self.archive_class_grid.setSpacing(8)
        self.archive_class_grid.setContentsMargins(0, 0, 0, 0)
        ab_lay.addWidget(self.archive_class_host)
        right_col.addWidget(archive_box)
        content.addLayout(right_col)

        self._rebuild_archive_class_buttons()

        root.addLayout(content, 1)

        # 底部状态
        self.lbl_bottom = QLabel("")
        self.lbl_bottom.setStyleSheet("color:#546E7A;font-size:12px;")
        root.addWidget(self.lbl_bottom)

    def refresh(self):
        """结果更新后重新填充表格。"""
        results = self.state.results
        self._rebuild_archive_class_buttons()

        # 更新类别筛选下拉（仅列表中仍显示的条目）
        visible = [r for r in results if _is_results_list_item(r)]
        classes = sorted({r["class"] for r in visible if r.get("class") and r["class"] != "ERROR"})
        self.filter_cls.blockSignals(True)
        self.filter_cls.clear()
        self.filter_cls.addItem("全部")
        for c in classes:
            self.filter_cls.addItem(c)
        self.filter_cls.blockSignals(False)

        self._fill_table(results)

    def _conf_header_label(self) -> str:
        if self._conf_sort_order == "desc":
            return "置信度 ▼"
        if self._conf_sort_order == "asc":
            return "置信度 ▲"
        return "置信度 ↕"

    def _set_table_header_labels(self) -> None:
        self.table.setHorizontalHeaderLabels([
            "", "缩略图", "文件名", "预测类别",
            self._conf_header_label(), "操作",
        ])
        conf_hdr = self.table.horizontalHeaderItem(4)
        if conf_hdr is None:
            conf_hdr = QTableWidgetItem(self._conf_header_label())
            self.table.setHorizontalHeaderItem(4, conf_hdr)
        else:
            conf_hdr.setText(self._conf_header_label())
        conf_hdr.setForeground(QColor("#1565C0"))
        conf_hdr.setToolTip("点击切换排序：高→低 ▼ / 低→高 ▲")

    def _rebuild_archive_class_buttons(self) -> None:
        while self.archive_class_grid.count():
            item = self.archive_class_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._archive_class_btns.clear()

        classes = self.state.engine.classes if self.state.engine else []
        if not classes:
            lbl = QLabel("模型未加载")
            lbl.setStyleSheet("color:#90A4AE;font-size:11px;")
            self.archive_class_grid.addWidget(lbl, 0, 0)
            return

        cols = min(2, max(1, len(classes)))
        for i, cls_name in enumerate(classes):
            fz = _class_button_font_px(cls_name)
            btn = QPushButton(cls_name)
            btn.setMinimumHeight(72)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(_class_button_stylesheet(False, fz))
            btn.clicked.connect(lambda _, c=cls_name: self._archive_as_class(c))
            self._archive_class_btns.append(btn)
            self.archive_class_grid.addWidget(btn, i // cols, i % cols)

    def _archive_target_indices(self) -> List[int]:
        """优先归档当前表格中勾选的行；若无勾选则归档当前选中行。"""
        self._sync_checks_from_table()
        checked: List[int] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.checkState() == Qt.Checked:
                idx = item.data(Qt.UserRole)
                if idx is not None and 0 <= idx < len(self.state.results):
                    if _is_results_list_item(self.state.results[idx]):
                        checked.append(idx)
        if checked:
            return checked
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        indices: List[int] = []
        for row in rows:
            item = self.table.item(row, 0)
            if not item:
                continue
            idx = item.data(Qt.UserRole)
            if idx is not None and 0 <= idx < len(self.state.results):
                if _is_results_list_item(self.state.results[idx]):
                    indices.append(idx)
        return indices

    def _archive_as_class(self, true_cls: str) -> None:
        indices = self._archive_target_indices()
        if not indices:
            QMessageBox.information(
                self, "提示",
                "请先勾选或选中要归档的图像，再点击正确类别。",
            )
            return
        if len(indices) > 1:
            reply = QMessageBox.question(
                self,
                "确认归档",
                f"将 {len(indices)} 张图像归档为「{true_cls}」并移出结果列表？\n"
                f"文件将复制到 corrections/{true_cls}/",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return

        saved = _archive_results_at_indices(self.state, indices, true_cls)
        if saved == 0:
            QMessageBox.warning(self, "归档失败", "未能保存任何图像，请检查文件是否存在。")
            return

        corr_dir = self.state.config.get("corrections_dir", "corrections")
        msg = f"已归档 {saved} 张至 {corr_dir}/{true_cls}/，已从结果列表移除"
        self.correction_updated.emit(msg)
        self._fill_table(self.state.results)

    def _on_row_selection_changed(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        if len(rows) != 1:
            return
        row = next(iter(rows))
        item = self.table.item(row, 0)
        if not item:
            return
        idx = item.data(Qt.UserRole)
        if idx is None or not (0 <= idx < len(self.state.results)):
            return
        r = self.state.results[idx]
        self.lbl_archive_hint.setText(
            f"当前：{Path(r['path']).name}\n预测 {r['class']} "
            f"({r['confidence']*100:.1f}%) → 点击正确类别归档"
        )
        if Path(r["path"]).exists():
            self.preview.show_image(r["path"], self._meta_for_result(r))

    def _on_table_header_clicked(self, col: int) -> None:
        if col != 4:
            return
        if self._conf_sort_order == "desc":
            self._conf_sort_order = "asc"
        else:
            self._conf_sort_order = "desc"
        self._set_table_header_labels()
        self._apply_filter()

    def _meta_for_result(self, r: Dict) -> str:
        text = f"预测：{r['class']}  ·  {r['confidence']*100:.1f}%"
        max_cls = r.get("max_class", r["class"])
        if max_cls != r["class"]:
            text += f"（最高：{max_cls} {r.get('max_confidence', 0)*100:.1f}%）"
        return text

    def _fill_table(self, results: List[Dict]):
        """根据筛选/排序条件重建表格；刷新前 blockSignals 避免勾选回调干扰。"""
        cls_f  = self.filter_cls.currentText()
        conf_f = self.filter_conf.value()

        filtered = [
            r for r in results
            if _is_results_list_item(r)
            and (cls_f == "全部" or r["class"] == cls_f)
            and r["confidence"] <= conf_f
        ]
        if self._conf_sort_order == "desc":
            filtered.sort(key=lambda r: r["confidence"], reverse=True)
        elif self._conf_sort_order == "asc":
            filtered.sort(key=lambda r: r["confidence"])

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(filtered))

        for row, r in enumerate(filtered):
            self.table.setRowHeight(row, 72)
            idx = self.state.results.index(r)   # 原始索引，存入 UserRole 供操作列使用

            # 勾选框
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if r.get("_checked", True) else Qt.Unchecked)
            chk.setData(Qt.UserRole, idx)
            chk.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, chk)

            # 缩略图：悬停预览，单击打开原图
            meta = self._meta_for_result(r)
            if Path(r["path"]).exists():
                thumb = _thumb(r["path"], 62)
                _connect_thumb_preview(thumb, self._preview_ctrl, meta)
                self.table.setCellWidget(row, 1, thumb)

            # 文件名：悬停预览，单击固定右侧预览
            fn_w = HoverFilenameLabel(r["path"], Path(r["path"]).name)
            fn_w.setStyleSheet(
                "font-size:13px;color:#263238;padding:2px 4px;"
            )
            _connect_filename_preview(fn_w, self._preview_ctrl, meta)
            self.table.setCellWidget(row, 2, fn_w)

            # 预测类别（带颜色标记）
            cls_item = QTableWidgetItem(r["class"])
            cls_item.setTextAlignment(Qt.AlignCenter)
            if r.get("flagged"):
                cls_item.setForeground(QColor("#C62828"))
            self.table.setItem(row, 3, cls_item)

            # 置信度（可排序列）
            conf_item = QTableWidgetItem(f"{r['confidence']*100:.1f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            conf_item.setData(Qt.UserRole, r["confidence"])
            self.table.setItem(row, 4, conf_item)

            # 操作：送入误分类修正页
            btn = QPushButton("送修正")
            btn.setToolTip("送入误分类修正页，并从本列表移除")
            btn.setStyleSheet(
                "background:#F57C00;color:white;border:none;border-radius:4px;"
                "padding:6px 10px;font-size:12px;"
            )
            btn.clicked.connect(lambda _, i=idx: self._send_to_correction(i))
            self.table.setCellWidget(row, 5, btn)

        n_list = len([r for r in results if _is_results_list_item(r)])
        n_queue = count_flagged_pending(results)
        self.lbl_stat.setText(f"本页 {n_list} 条  |  修正页待处理 {n_queue} 项")
        sort_hint = ""
        if self._conf_sort_order == "desc":
            sort_hint = "  |  排序：置信度 高→低"
        elif self._conf_sort_order == "asc":
            sort_hint = "  |  排序：置信度 低→高"
        self.lbl_bottom.setText(
            f"显示 {len(filtered)} 条{sort_hint}"
        )
        if self._preview_ctrl.pinned_path:
            p = self._preview_ctrl.pinned_path
            for r in results:
                if r["path"] == p:
                    self.preview.show_image(p, self._meta_for_result(r))
                    self.preview.set_pinned(True)
                    break
        self.table.blockSignals(False)

    def _on_check_changed(self, item: QTableWidgetItem):
        if item.column() != 0:
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self.state.results):
            self.state.results[idx]["_checked"] = (
                item.checkState() == Qt.Checked
            )

    def _apply_filter(self):
        self._sync_checks_from_table()
        self._fill_table(self.state.results)

    def _sync_checks_from_table(self):
        """将表格勾选状态写回 results，避免刷新后丢失。"""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item:
                continue
            idx = item.data(Qt.UserRole)
            if idx is not None and 0 <= idx < len(self.state.results):
                self.state.results[idx]["_checked"] = (
                    item.checkState() == Qt.Checked
                )

    def _toggle_all_checks(self):
        self._all_checked = not self._all_checked
        self._set_all_checks(self._all_checked)

    def _set_all_checks(self, checked: bool):
        self._all_checked = checked
        self.btn_toggle_sel.setText("全不选" if checked else "全选")
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setCheckState(state)
                idx = item.data(Qt.UserRole)
                if idx is not None and 0 <= idx < len(self.state.results):
                    self.state.results[idx]["_checked"] = checked

    def _send_to_correction(self, idx: int) -> None:
        """单行「送修正」：设 flagged=True，从本页表格消失，CorrectionPage 可见。"""
        if not (0 <= idx < len(self.state.results)):
            return
        self._sync_checks_from_table()
        r = self.state.results[idx]
        if not _is_results_list_item(r):
            return
        r["flagged"] = True
        name = Path(r["path"]).name
        n = count_flagged_pending(self.state.results)
        self.correction_updated.emit(f"「{name}」已送入误分类修正页（共 {n} 项待处理）")
        self._fill_table(self.state.results)

    def _flag_selected(self):
        self._sync_checks_from_table()
        count = 0
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if not item or item.checkState() != Qt.Checked:
                continue
            idx = item.data(Qt.UserRole)
            if idx is None or not (0 <= idx < len(self.state.results)):
                continue
            r = self.state.results[idx]
            if _is_results_list_item(r):
                r["flagged"] = True
                count += 1
        if count == 0:
            QMessageBox.information(self, "提示", "请先勾选要送入修正页的图像。")
            return
        n = count_flagged_pending(self.state.results)
        self.correction_updated.emit(
            f"已将 {count} 项送入误分类修正页（共 {n} 项待处理）"
        )
        self._fill_table(self.state.results)

    def _flag_visible(self):
        self._sync_checks_from_table()
        cls_f  = self.filter_cls.currentText()
        conf_f = self.filter_conf.value()
        count = 0
        for r in self.state.results:
            if (
                _is_results_list_item(r)
                and (cls_f == "全部" or r["class"] == cls_f)
                and r["confidence"] <= conf_f
            ):
                r["flagged"] = True
                count += 1
        if count == 0:
            QMessageBox.information(self, "提示", "当前筛选下没有可送入修正页的条目。")
            return
        n = count_flagged_pending(self.state.results)
        self.correction_updated.emit(
            f"已将当前筛选 {count} 项送入误分类修正页（共 {n} 项待处理）"
        )
        self._fill_table(self.state.results)

    def _export_classified(self):
        """导出勾选且未标记待修正的图像，按预测类别分文件夹存放。"""
        self._sync_checks_from_table()
        export_classified_images(
            self, self.state.results, only_checked=True,
            default_export_name="分类导出",
        )

    def _export_csv(self):
        _export_results_csv(self, self.state.results)


# ═══════════════════════════════════════════════════════════
# 页面 3 ── 误分类修正（重新标记 + 列表查看）
# ═══════════════════════════════════════════════════════════
class CorrectionPage(QWidget):
    """
    误分类修正页 — 处理 flagged=True 且未 correction_saved 的条目。

    数据来源:
      · 「结果管理」送入的误检
      · 「选择文件夹」导入的本地图像（待重新打标签）

    重新标记模式（默认）:
      · 左侧大图 + 类别大按钮，单击即归档到 corrections/ 并 pop 出 results
      · 右侧待修正列表，切换当前处理的图像

    列表查看模式（备用）:
      · 卡片 + 下拉框选类别，「保存选中修正」批量归档

    归档逻辑见 _save_correction_to_disk：复制文件，不移动原图。
    """

    saved_signal = pyqtSignal(str)   # 归档成功 → MainWindow 更新状态栏

    _MARK_VIEW = 0
    _LIST_VIEW = 1

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._card_checks: List[QCheckBox] = []
        self._flagged_indices: List[int] = []
        self._mark_pos: int = 0
        self._class_btns: List[QPushButton] = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 16, 12, 16)
        root.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel("误分类修正")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_count = QLabel("待修正：0 项")
        self.lbl_count.setStyleSheet("color:#E65100;font-weight:bold;")
        hdr.addWidget(self.lbl_count)

        btn_import = _mk_btn("选择文件夹", "flat", 118)
        btn_import.setMinimumWidth(118)
        btn_import.setToolTip(
            "选择文件夹，将其内图像导入为待重新打标签的训练样本"
            "（可含子文件夹；归档后进入 corrections/ 供再训练）"
        )
        btn_import.clicked.connect(self._import_folder)
        hdr.addWidget(btn_import)

        self.btn_mark_view = _mk_btn("重新标记", "primary", 118)
        self.btn_mark_view.setMinimumWidth(118)
        self.btn_mark_view.clicked.connect(lambda: self._switch_view(self._MARK_VIEW))
        hdr.addWidget(self.btn_mark_view)

        self.btn_list_view = _mk_btn("列表查看", "flat", 118)
        self.btn_list_view.setMinimumWidth(118)
        self.btn_list_view.clicked.connect(lambda: self._switch_view(self._LIST_VIEW))
        hdr.addWidget(self.btn_list_view)

        btn_save_all = _mk_btn("保存选中修正", "success", 128)
        btn_save_all.clicked.connect(self._save_all)
        self.btn_save_all = btn_save_all
        hdr.addWidget(btn_save_all)
        root.addLayout(hdr)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── 重新标记 ──────────────────────────────────
        mark_page = QWidget()
        mark_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mark_lay = QHBoxLayout(mark_page)
        mark_lay.setContentsMargins(0, 0, 0, 0)
        mark_lay.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(10)

        self.lbl_mark_hint = QLabel(
            "可从「结果管理」送入误检，或点「选择文件夹」导入本地图像后重新打标签。"
            "点击正确类别即归档至 corrections/ 对应文件夹，并移出待修正列表。"
        )
        self.lbl_mark_hint.setStyleSheet("color:#546E7A;font-size:12px;")
        self.lbl_mark_hint.setWordWrap(True)
        left.addWidget(self.lbl_mark_hint)

        self.img_large = QLabel("暂无待修正图像")
        self.img_large.setAlignment(Qt.AlignCenter)
        self.img_large.setMinimumSize(480, 400)
        self.img_large.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_large.setStyleSheet(
            "background:#FAFAFA;border:1px solid #CFD8DC;border-radius:8px;"
            "color:#90A4AE;font-size:14px;"
        )
        left.addWidget(self.img_large, 1)

        info_row = QHBoxLayout()
        self.lbl_mark_file = QLabel("")
        self.lbl_mark_file.setStyleSheet("font-weight:bold;color:#263238;font-size:14px;")
        self.lbl_mark_pred = QLabel("")
        self.lbl_mark_pred.setStyleSheet("color:#C62828;font-size:13px;")
        info_row.addWidget(self.lbl_mark_file, 1)
        info_row.addWidget(self.lbl_mark_pred)
        left.addLayout(info_row)

        class_title = QLabel("选择正确类别（单击即完成，无需确认）")
        class_title.setStyleSheet("color:#455A64;font-size:12px;font-weight:bold;")
        left.addWidget(class_title)

        self.class_grid_host = QWidget()
        self.class_grid_host.setMinimumWidth(520)
        self.class_grid = QGridLayout(self.class_grid_host)
        self.class_grid.setSpacing(10)
        self.class_grid.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.class_grid_host)

        mark_lay.addLayout(left, 5)

        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_panel.setMinimumWidth(360)
        right_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(12, 12, 12, 12)
        right.setSpacing(8)
        rlbl = QLabel("待修正列表")
        rlbl.setStyleSheet("font-size:13px;font-weight:bold;color:#455A64;")
        right.addWidget(rlbl)

        self.mark_list = QListWidget()
        self.mark_list.setAlternatingRowColors(True)
        self.mark_list.setIconSize(QSize(36, 36))
        self.mark_list.setMinimumWidth(320)
        self.mark_list.currentRowChanged.connect(self._on_mark_list_row)
        right.addWidget(self.mark_list, 1)

        self.lbl_mark_progress = QLabel("")
        self.lbl_mark_progress.setStyleSheet("color:#78909C;font-size:11px;")
        self.lbl_mark_progress.setAlignment(Qt.AlignCenter)
        self.lbl_mark_progress.setWordWrap(True)
        right.addWidget(self.lbl_mark_progress)

        mark_lay.addWidget(right_panel, 2)
        self.stack.addWidget(mark_page)

        # ── 列表查看（备用）──────────────────────────────
        list_page = QWidget()
        list_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_lay = QVBoxLayout(list_page)
        list_lay.setContentsMargins(0, 0, 0, 0)
        list_lay.setSpacing(8)

        lbl_hint = QLabel(
            "列表查看：勾选条目、下拉选择类别后批量保存。"
            "日常修正请优先使用「重新标记」。"
        )
        lbl_hint.setStyleSheet("color:#546E7A;font-size:12px;")
        lbl_hint.setWordWrap(True)
        list_lay.addWidget(lbl_hint)

        body = QHBoxLayout()
        body.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setMinimumWidth(720)
        self.card_container = QWidget()
        self.card_container.setMinimumWidth(700)
        self.card_lay = QVBoxLayout(self.card_container)
        self.card_lay.setSpacing(8)
        self.card_lay.setContentsMargins(4, 4, 4, 4)
        self.card_lay.addStretch()
        scroll.setWidget(self.card_container)
        body.addWidget(scroll, 3)

        self.preview = ImagePreviewSidePanel(min_size=260)
        self._preview_ctrl = SidePreviewController(self.preview)
        self.preview.bind_controller(self._preview_ctrl)
        body.addWidget(self.preview, 0)

        list_lay.addLayout(body, 1)

        list_hdr = QHBoxLayout()
        self.btn_toggle_sel = _mk_btn("全不选", "flat", 88)
        self.btn_toggle_sel.clicked.connect(self._toggle_all_checks)
        list_hdr.addWidget(self.btn_toggle_sel)
        self._all_checked = True
        list_hdr.addStretch()
        list_lay.addLayout(list_hdr)

        self.stack.addWidget(list_page)
        root.addWidget(self.stack, 1)

        self._rebuild_class_buttons()
        self._switch_view(self._MARK_VIEW)

    def _switch_view(self, idx: int):
        self.stack.setCurrentIndex(idx)
        active = idx == self._MARK_VIEW
        self.btn_mark_view.setProperty("class", "primary" if active else "flat")
        self.btn_list_view.setProperty("class", "primary" if not active else "flat")
        for btn in (self.btn_mark_view, self.btn_list_view):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.btn_save_all.setVisible(not active)
        if active:
            self._refresh_mark_view()
        else:
            self._refresh_list_view()

    def _rebuild_class_buttons(self):
        while self.class_grid.count():
            item = self.class_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._class_btns.clear()

        classes = self.state.engine.classes if self.state.engine else []
        if not classes:
            lbl = QLabel("模型未加载，无法显示类别按钮。")
            lbl.setStyleSheet("color:#C62828;")
            self.class_grid.addWidget(lbl, 0, 0)
            return

        cols = min(3, max(2, len(classes)))
        for i, cls_name in enumerate(classes):
            fz = _class_button_font_px(cls_name)
            btn = QPushButton(cls_name)
            btn.setMinimumHeight(88)
            btn.setMinimumWidth(168)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(_class_button_stylesheet(False, fz))
            btn.clicked.connect(lambda _, c=cls_name: self._apply_class_mark(c))
            self._class_btns.append(btn)
            self.class_grid.addWidget(btn, i // cols, i % cols)

    def refresh(self):
        """外部进入本页或保存后刷新。"""
        self._rebuild_class_buttons()
        if self.stack.currentIndex() == self._MARK_VIEW:
            self._refresh_mark_view()
        else:
            self._refresh_list_view()

    def _import_folder(self):
        """
        选择文件夹，扫描其中图像并加入待修正队列（flagged=True）。

        不跑推理：类别默认为「待标注」；若父目录名恰好是引擎五类之一，
        则用作提示（仍需用户点击确认归档）。归档逻辑与误检修正相同。
        """
        folder = QFileDialog.getExistingDirectory(
            self, "选择待重新打标签的图像文件夹"
        )
        if not folder:
            return

        root = Path(folder)
        imgs = sorted(
            f for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        )
        if not imgs:
            QMessageBox.information(
                self, "提示",
                f"未在以下目录找到图像：\n{folder}\n\n"
                f"支持格式：{', '.join(sorted(IMG_EXTS))}",
            )
            return

        known = list(self.state.engine.classes) if self.state.engine else []
        known_set = set(known)
        added = 0
        skipped_dup = 0

        existing_flagged = {
            _normalize_image_path(r["path"]).lower()
            for r in self.state.results
            if r.get("flagged") and not r.get("correction_saved")
        }

        for img_path in imgs:
            path_str = _normalize_image_path(str(img_path))
            key = path_str.lower()
            if key in existing_flagged:
                skipped_dup += 1
                continue

            parent = img_path.parent.name
            hint_cls = parent if parent in known_set else "待标注"
            r = _ensure_result_meta({
                "path": path_str,
                "class": hint_cls,
                "class_idx": known.index(hint_cls) if hint_cls in known_set else -1,
                "confidence": 0.0,
                "max_class": hint_cls,
                "max_confidence": 0.0,
                "all_scores": {},
                "elapsed_ms": 0.0,
                "flagged": True,
                "from_folder_import": True,
            })
            _upsert_result(self.state.results, r)
            # upsert 可能保留旧 flagged；强制进入待修正队列
            for existing in self.state.results:
                if _normalize_image_path(existing["path"]).lower() == key:
                    existing["flagged"] = True
                    existing["correction_saved"] = False
                    existing["from_folder_import"] = True
                    if not existing.get("class") or existing["class"] in ("ERROR",):
                        existing["class"] = hint_cls
                    break
            existing_flagged.add(key)
            added += 1

        self.refresh()
        self._switch_view(self._MARK_VIEW)

        parts = [f"已导入 {added} 张待标注图像"]
        if skipped_dup:
            parts.append(f"跳过已在队列中的 {skipped_dup} 张")
        msg = "，".join(parts) + f"。\n来源：{folder}"
        self.saved_signal.emit(parts[0] + (f"（跳过重复 {skipped_dup}）" if skipped_dup else ""))
        QMessageBox.information(self, "导入完成", msg)

    def _flagged_items(self) -> List[tuple]:
        return [
            (i, r) for i, r in enumerate(self.state.results)
            if r.get("flagged") and not r.get("correction_saved")
        ]

    def _update_count_label(self):
        flagged = self._flagged_items()
        self.lbl_count.setText(f"待修正：{len(flagged)} 项")

    def _refresh_mark_view(self):
        self._flagged_indices = [i for i, _ in self._flagged_items()]
        self._update_count_label()
        n = len(self._flagged_indices)

        self.mark_list.blockSignals(True)
        self.mark_list.clear()
        for pos, orig_idx in enumerate(self._flagged_indices):
            r = self.state.results[orig_idx]
            item = QListWidgetItem()
            item.setData(Qt.UserRole, pos)
            icon = self.style().standardIcon(QStyle.SP_FileDialogContentsView)
            item.setIcon(icon)
            name = Path(r["path"]).name
            if r.get("from_folder_import") or r.get("class") == "待标注":
                item.setText(f"{name}\n来源：文件夹导入")
            else:
                item.setText(f"{name}\n预测：{r['class']}")
            item.setSizeHint(QSize(0, 58))
            item.setToolTip(r["path"])
            self.mark_list.addItem(item)
        self.mark_list.blockSignals(False)

        if n == 0:
            self._mark_pos = 0
            self._show_mark_empty()
            return

        if self._mark_pos >= n:
            self._mark_pos = 0
        pending_pos = self._first_pending_pos()
        if pending_pos is not None:
            self._mark_pos = pending_pos

        self.mark_list.setCurrentRow(self._mark_pos)
        self._show_mark_at(self._mark_pos)

    def _first_pending_pos(self) -> Optional[int]:
        if self._flagged_indices:
            return 0
        return None

    def _show_mark_empty(self):
        self.img_large.clear()
        self.img_large.setText(
            "暂无待修正图像\n\n"
            "· 在「结果管理」中点击「送修正」\n"
            "· 或本页点击「选择文件夹」导入待打标签图像"
        )
        self.lbl_mark_file.setText("")
        self.lbl_mark_pred.setText("")
        self.lbl_mark_progress.setText("")
        for btn in self._class_btns:
            btn.setEnabled(False)

    def _show_mark_at(self, pos: int):
        if not self._flagged_indices or pos < 0 or pos >= len(self._flagged_indices):
            self._show_mark_empty()
            return

        self._mark_pos = pos
        orig_idx = self._flagged_indices[pos]
        r = self.state.results[orig_idx]
        path = r["path"]

        for btn in self._class_btns:
            btn.setEnabled(True)
            cls = btn.text()
            btn.setStyleSheet(
                _class_button_stylesheet(False, _class_button_font_px(cls))
            )

        self.lbl_mark_file.setText(Path(path).name)
        if r.get("from_folder_import") or r.get("class") == "待标注":
            hint = r.get("class") or "待标注"
            if hint != "待标注":
                self.lbl_mark_pred.setText(
                    f"来源：文件夹导入 · 目录提示类别：{hint}"
                )
            else:
                self.lbl_mark_pred.setText("来源：文件夹导入（待重新打标签）")
        else:
            self.lbl_mark_pred.setText(
                f"预测：{r['class']}  ({r['confidence']*100:.1f}%)"
            )

        self.lbl_mark_progress.setText(
            f"第 {pos + 1} / {len(self._flagged_indices)} 张"
        )

        if Path(path).is_file():
            side = max(self.img_large.width(), self.img_large.height(), 420)
            px = QPixmap(path).scaled(
                side - 16, side - 16,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.img_large.setPixmap(px)
            self.img_large.setStyleSheet(
                "background:#1A1A2E;border:1px solid #37474F;border-radius:8px;"
            )
        else:
            self.img_large.clear()
            self.img_large.setText("无法加载图像")
            self.img_large.setStyleSheet(
                "background:#FAFAFA;border:1px dashed #CFD8DC;border-radius:8px;"
                "color:#C62828;font-size:14px;"
            )

    def _on_mark_list_row(self, row: int):
        if row < 0 or row >= len(self._flagged_indices):
            return
        self._mark_pos = row
        self._show_mark_at(row)

    def _next_pending_pos(self, after_pos: int) -> Optional[int]:
        n = len(self._flagged_indices)
        if n <= 1:
            return None
        return (after_pos + 1) % n

    def _apply_class_mark(self, true_cls: str):
        """重新标记：单击类别 → 复制到 corrections/true_cls/ → 从 results 移除。"""
        if not self._flagged_indices:
            return
        pos = self._mark_pos
        if pos < 0 or pos >= len(self._flagged_indices):
            pos = 0
        orig_idx = self._flagged_indices[pos]
        r = self.state.results[orig_idx]

        if not Path(r["path"]).is_file():
            QMessageBox.warning(self, "文件不存在", f"找不到图像：\n{r['path']}")
            return

        try:
            ok = _save_correction_to_disk(self.state, r, true_cls)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        if not ok:
            QMessageBox.warning(self, "保存失败", "无法复制图像到 corrections 目录。")
            return

        corr_dir = self.state.config.get("corrections_dir", "corrections")
        name = Path(r["path"]).name
        self.state.results.pop(orig_idx)
        self.saved_signal.emit(
            f"已归档「{name}」→ {corr_dir}/{true_cls}/"
        )
        self._refresh_mark_view()

    # ── 列表查看（备用）──────────────────────────────────

    def _refresh_list_view(self):
        self._card_checks.clear()
        self.preview.clear_preview()
        while self.card_lay.count() > 1:
            it = self.card_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        flagged = self._flagged_items()
        self._update_count_label()
        classes = self.state.engine.classes if self.state.engine else []

        for orig_idx, r in flagged:
            card = self._make_card(orig_idx, r, classes)
            self.card_lay.insertWidget(self.card_lay.count() - 1, card)

    def _toggle_all_checks(self):
        self._all_checked = not self._all_checked
        self._set_all_checks(self._all_checked)

    def _set_all_checks(self, checked: bool):
        self._all_checked = checked
        self.btn_toggle_sel.setText("全不选" if checked else "全选")
        for cb in self._card_checks:
            cb.setChecked(checked)

    def _make_card(self, orig_idx: int, r: Dict, classes: List[str]) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(100)
        card.setMinimumWidth(660)
        cl = QHBoxLayout(card)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(12)

        chk = QCheckBox()
        chk.setChecked(True)
        chk.setProperty("orig_idx", orig_idx)
        self._card_checks.append(chk)
        cl.addWidget(chk)

        meta = f"预测：{r['class']}  ·  {r['confidence']*100:.1f}%"
        thumb_lbl = ThumbnailLabel(r["path"], 72) if Path(r["path"]).exists() else QLabel()
        if isinstance(thumb_lbl, ThumbnailLabel):
            _connect_thumb_preview(thumb_lbl, self._preview_ctrl, meta)
        else:
            thumb_lbl.setFixedSize(72, 72)
        cl.addWidget(thumb_lbl)

        info = QVBoxLayout()
        fn_lbl = HoverFilenameLabel(r["path"], Path(r["path"]).name)
        fn_lbl.setStyleSheet("font-weight:bold;color:#263238;font-size:15px;")
        fn_lbl.setMinimumWidth(260)
        _connect_filename_preview(fn_lbl, self._preview_ctrl, meta)
        pred_lbl = QLabel(f"预测：{r['class']}  ({r['confidence']*100:.1f}%)")
        pred_lbl.setStyleSheet("color:#C62828;font-size:13px;")
        info.addWidget(fn_lbl)
        info.addWidget(pred_lbl)
        cl.addLayout(info, 1)

        combo = QComboBox()
        for c in classes:
            combo.addItem(c)
        current_true = r.get("true_class", "")
        if current_true in classes:
            combo.setCurrentText(current_true)
        combo.setFixedWidth(160)
        combo.currentTextChanged.connect(
            lambda txt, i=orig_idx: self._set_true_class(i, txt)
        )
        cl.addWidget(QLabel("→ 正确类别："))
        cl.addWidget(combo)

        saved = r.get("correction_saved", False)
        status = QLabel("✓ 已归档" if saved else "")
        status.setStyleSheet("color:#2E7D32;font-weight:bold;")
        cl.addWidget(status)

        return card

    def _set_true_class(self, idx: int, text: str):
        if 0 <= idx < len(self.state.results):
            self.state.results[idx]["true_class"] = text

    def _save_all(self):
        """列表查看：批量保存勾选且已设类别的条目。"""
        corrections_dir = Path(self.state.config.get("corrections_dir", "corrections"))
        saved, skipped = 0, 0
        to_remove: List[int] = []

        checked_indices = {
            cb.property("orig_idx")
            for cb in self._card_checks
            if cb.isChecked() and cb.property("orig_idx") is not None
        }
        if not checked_indices:
            QMessageBox.warning(self, "提示", "请先勾选需要保存的条目。")
            return

        for i, r in enumerate(self.state.results):
            if not r.get("flagged") or i not in checked_indices:
                continue
            true_cls = r.get("true_class", "").strip()
            if not true_cls:
                skipped += 1
                continue
            try:
                if _save_correction_to_disk(self.state, r, true_cls):
                    saved += 1
                    to_remove.append(i)
                else:
                    skipped += 1
            except OSError:
                skipped += 1

        for i in sorted(to_remove, reverse=True):
            self.state.results.pop(i)

        msg = f"已归档 {saved} 张图像到 {corrections_dir}/，已从待修正列表移除。"
        if skipped:
            msg += f"\n{skipped} 项未设置类别或保存失败，已跳过。"
        QMessageBox.information(self, "保存完成", msg)
        self.refresh()
        self.saved_signal.emit(msg)


# ═══════════════════════════════════════════════════════════
# 页面 4 ── 模型再训练（TrainWorker + QTextEdit 日志 + AdminLockBar）
# ═══════════════════════════════════════════════════════════
class RetrainPage(QWidget):
    """
    模型再训练页（仅开发版可用；机台版 DEPLOY_ONNX_ONLY 时页面仍显示但训练不可用）。

    流程:
      1. 解锁 AdminLockBar → 填写参数 → TrainWorker 子进程跑 train.py
      2. 训练成功 → 「应用新模型」调用 engine.load 热更新
      3. corrections/ 若存在则自动追加 --extra_data_dirs（增量微调）

    默认 num_workers=0，避免 Windows 下 DataLoader 多进程与 PyQt 冲突。
    """

    model_updated = pyqtSignal()           # 热加载成功 → MainWindow 更新状态栏
    admin_unlock_requested = pyqtSignal()  # → MainWindow._try_unlock_admin_pages
    admin_lock_requested   = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state  = state
        self.worker: Optional[TrainWorker] = None
        self._admin_locked = True
        self._build()
        self.set_admin_locked(True)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("模型再训练")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        root.addWidget(title)

        self.lock_bar = AdminLockBar()
        self.lock_bar.unlock_requested.connect(self.admin_unlock_requested.emit)
        self.lock_bar.lock_requested.connect(self.admin_lock_requested.emit)
        root.addWidget(self.lock_bar)

        # ── 数据统计 ──────────────────────────────
        stat_grp = QGroupBox("数据统计")
        stat_lay = QVBoxLayout(stat_grp)
        self.stat_label = QLabel('点击「刷新统计」查看数据情况')
        self.stat_label.setStyleSheet("color:#455A64;font-size:12px;")
        self.stat_label.setWordWrap(True)
        stat_lay.addWidget(self.stat_label)
        btn_refresh = _mk_btn("刷新统计", "flat")
        btn_refresh.clicked.connect(self._refresh_stats)
        stat_lay.addWidget(btn_refresh)
        self.btn_refresh = btn_refresh
        root.addWidget(stat_grp)

        # ── 训练参数 ──────────────────────────────
        param_grp = QGroupBox("训练参数")
        form = QFormLayout(param_grp)
        form.setLabelAlignment(Qt.AlignRight)

        self.sp_img   = QSpinBox()
        self.sp_img.setRange(64, 512)
        self.sp_img.setSingleStep(32)
        self.sp_img.setValue(224)
        form.addRow("图像尺寸 (px):", self.sp_img)

        self.sp_ep1 = QSpinBox()
        self.sp_ep1.setRange(0, 50)
        self.sp_ep1.setValue(10)
        form.addRow("阶段一 Epochs:", self.sp_ep1)

        self.sp_ep2 = QSpinBox()
        self.sp_ep2.setRange(0, 100)
        self.sp_ep2.setValue(25)
        form.addRow("阶段二 Epochs:", self.sp_ep2)

        self.sp_bs = QSpinBox()
        self.sp_bs.setRange(4, 128)
        self.sp_bs.setSingleStep(4)
        self.sp_bs.setValue(16)
        form.addRow("批大小:", self.sp_bs)

        root.addWidget(param_grp)

        # ── 日志输出 ──────────────────────────────
        log_grp = QGroupBox("训练日志")
        log_lay = QVBoxLayout(log_grp)
        self.log_edit = QTextEdit()
        self.log_edit.setObjectName("log")
        self.log_edit.setReadOnly(True)
        self.log_edit.setFixedHeight(180)
        log_lay.addWidget(self.log_edit)
        root.addWidget(log_grp)

        # ── 控制按钮 ──────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_train = _mk_btn("▶  开始训练", "primary")
        self.btn_train.setFixedHeight(36)
        self.btn_train.clicked.connect(self._start_train)

        self.btn_stop = _mk_btn("■  停止", "danger")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_train)

        self.btn_apply = _mk_btn("✓  应用新模型", "success")
        self.btn_apply.setFixedHeight(36)
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_model)

        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_apply)
        root.addLayout(btn_row)
        root.addStretch()

    def set_admin_locked(self, locked: bool) -> None:
        """切换只读 / 可编辑（默认只读）。"""
        self._admin_locked = locked
        self.lock_bar.set_locked(locked)
        for w in (self.sp_img, self.sp_ep1, self.sp_ep2, self.sp_bs,
                  self.btn_train, self.btn_stop, self.btn_apply):
            w.setEnabled(not locked)
        # 刷新统计为只读操作，锁定状态下仍允许查看
        self.btn_refresh.setEnabled(True)
        if locked and self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def _refresh_stats(self):
        data_dir  = Path(self.state.config.get("data_dir", "data"))
        corr_dir  = Path(self.state.config.get("corrections_dir", "corrections"))
        lines = []

        for label, d in [("原始数据 data/", data_dir), ("修正数据 corrections/", corr_dir)]:
            if not d.exists():
                lines.append(f"{label}：目录不存在")
                continue
            total = 0
            class_counts = {}
            for cls_dir in sorted(d.iterdir()):
                if cls_dir.is_dir():
                    n = sum(1 for f in cls_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
                    class_counts[cls_dir.name] = n
                    total += n
            cls_str = "  ".join(f"{c}:{v}" for c, v in class_counts.items())
            lines.append(f"  {label}  共 {total} 张\n    {cls_str}")

        self.stat_label.setText("\n".join(lines))

        # 尝试从 train_config.json 读取上次训练参数
        cfg_path = Path(self.state.config.get("pt_path", "checkpoints/best_model.pt")
                        ).parent / "train_config.json"
        if cfg_path.exists() and not self._admin_locked:
            with open(cfg_path, encoding="utf-8") as f:
                tc = json.load(f)
            self.sp_img.setValue(tc.get("img_size", 224))

    def _start_train(self):
        """组装 train.py 命令行参数并启动 TrainWorker。"""
        if self._admin_locked:
            QMessageBox.warning(self, "只读模式", "请先输入密码解除只读后再训练。")
            return
        if not TRAIN_SCRIPT.exists():
            QMessageBox.critical(self, "错误", f"训练脚本不存在：{TRAIN_SCRIPT}")
            return

        data_dir = self.state.config.get("data_dir", "data")
        corr_dir = self.state.config.get("corrections_dir", "corrections")
        save_dir = str(Path(self.state.config.get("pt_path", "checkpoints/best_model.pt")).parent)

        args = [
            "--data_dir",       data_dir,
            "--img_size",       str(self.sp_img.value()),
            "--batch_size",     str(self.sp_bs.value()),
            "--epochs_phase1",  str(self.sp_ep1.value()),
            "--epochs_phase2",  str(self.sp_ep2.value()),
            "--save_dir",       save_dir,
            "--num_workers",    "0",   # Windows 多进程安全
        ]
        if Path(corr_dir).exists():
            args += ["--extra_data_dirs", corr_dir]

        self.log_edit.clear()
        self._log(f"$ python train.py {' '.join(args)}\n")
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_apply.setEnabled(False)

        self.worker = TrainWorker(str(TRAIN_SCRIPT), args)
        self.worker.log_line.connect(self._log)
        self.worker.finished.connect(self._on_train_done)
        self.worker.start()

    def _stop_train(self):
        if self.worker:
            self.worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_train_done(self, ok: bool, msg: str):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_apply.setEnabled(ok)
        color = "#A8D8A8" if ok else "#FF8A80"
        self._log(f"\n[{'完成' if ok else '失败'}] {msg}")
        self.log_edit.setTextColor(QColor(color))

    def _apply_model(self):
        """训练完成后从 checkpoints/ 重新 load 引擎（无需重启应用）。"""
        if self._admin_locked:
            QMessageBox.warning(self, "只读模式", "请先输入密码解除只读后再应用模型。")
            return
        if not (self.state.engine):
            return
        try:
            pt   = self.state.config.get("pt_path", "")
            onnx = self.state.config.get("onnx_path", "")
            msg  = self.state.engine.load(
                pt, onnx, self.state.config.get("use_gpu", True)
            )
            QMessageBox.information(self, "模型已更新", msg)
            self.model_updated.emit()
            self.btn_apply.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "更新失败", str(e))

    def _log(self, text: str):
        self.log_edit.setTextColor(QColor("#A8D8A8"))
        self.log_edit.append(text)
        self.log_edit.verticalScrollBar().setValue(
            self.log_edit.verticalScrollBar().maximum()
        )


# ═══════════════════════════════════════════════════════════
# 页面 5 ── 设置（路径、GPU、置信度阈值 → app_config.json）
# ═══════════════════════════════════════════════════════════
class SettingsPage(QWidget):
    """
    系统设置页 — QTabWidget 双 Tab。

    Tab 1 「分类配置」: 分类模型路径、数据路径、GPU、置信度阈值 → app_config.json
    Tab 2 「切片推理配置」: YOLO 模型、SAHI 参数 → app_config.json

    两个 Tab 共用 AdminLockBar 只读保护；保存时统一写入 app_config.json。
    分类配置保存后还会热加载分类引擎。
    """

    settings_saved = pyqtSignal(str)   # 加载结果消息 → 状态栏
    admin_unlock_requested = pyqtSignal()
    admin_lock_requested   = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._admin_locked = True
        self._browse_btns: List[QPushButton] = []
        self._sahi_browse_btns: List[QPushButton] = []
        self._build()
        self.set_admin_locked(True)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        title = QLabel("设置")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        root.addWidget(title)

        self.lock_bar = AdminLockBar()
        self.lock_bar.unlock_requested.connect(self.admin_unlock_requested.emit)
        self.lock_bar.lock_requested.connect(self.admin_lock_requested.emit)
        root.addWidget(self.lock_bar)

        # ── Tab 容器 ──────────────────────────────────────────────
        tabs = QTabWidget()
        root.addWidget(tabs)

        # ════════ Tab 1: 分类配置 ══════════════════════════════════
        tab_cls = QWidget()
        cl = QVBoxLayout(tab_cls)
        cl.setSpacing(16)

        # ── 分类模型文件 ──────────────────────────────────────────
        model_grp = QGroupBox("分类模型文件")
        mfl = QFormLayout(model_grp)
        mfl.setLabelAlignment(Qt.AlignRight)

        self.pt_edit = QLineEdit(self.state.config.get("pt_path", ""))
        btn_pt = _mk_btn("选择文件 …", "flat", 96)
        btn_pt.clicked.connect(
            lambda: self._browse_file(self.pt_edit, "PyTorch 模型", "*.pt *.pth")
        )
        pt_row = QHBoxLayout()
        pt_row.addWidget(self.pt_edit)
        pt_row.addWidget(btn_pt)
        self._browse_btns.append(btn_pt)
        if not DEPLOY_ONNX_ONLY:
            mfl.addRow("PyTorch 模型 (.pt):", pt_row)

        self.onnx_edit = QLineEdit(self.state.config.get("onnx_path", ""))
        btn_onnx = _mk_btn("选择文件 …", "flat", 96)
        btn_onnx.clicked.connect(
            lambda: self._browse_file(self.onnx_edit, "ONNX 模型", "*.onnx")
        )
        onnx_row = QHBoxLayout()
        onnx_row.addWidget(self.onnx_edit)
        onnx_row.addWidget(btn_onnx)
        self._browse_btns.append(btn_onnx)
        mfl.addRow("ONNX 模型 (.onnx):", onnx_row)
        cl.addWidget(model_grp)

        # ── 数据路径 ──────────────────────────────────────────────
        data_grp = QGroupBox("数据路径")
        dfl = QFormLayout(data_grp)
        dfl.setLabelAlignment(Qt.AlignRight)

        self.data_edit = QLineEdit(self.state.config.get("data_dir", "data"))
        btn_data = _mk_btn("选择路径 …", "flat", 96)
        btn_data.clicked.connect(
            lambda: self._browse_dir(self.data_edit, "训练数据目录")
        )
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_edit)
        data_row.addWidget(btn_data)
        self._browse_btns.append(btn_data)
        dfl.addRow("训练数据目录:", data_row)

        self.corr_edit = QLineEdit(self.state.config.get("corrections_dir", "corrections"))
        btn_corr = _mk_btn("选择路径 …", "flat", 96)
        btn_corr.clicked.connect(
            lambda: self._browse_dir(self.corr_edit, "修正数据目录")
        )
        corr_row = QHBoxLayout()
        corr_row.addWidget(self.corr_edit)
        corr_row.addWidget(btn_corr)
        self._browse_btns.append(btn_corr)
        dfl.addRow("修正数据目录:", corr_row)
        cl.addWidget(data_grp)

        # ── 推理设置 ──────────────────────────────────────────────
        infer_grp = QGroupBox("推理设置")
        ifl = QFormLayout(infer_grp)
        ifl.setLabelAlignment(Qt.AlignRight)

        self.chk_gpu = QCheckBox("使用 GPU（ONNX Runtime CUDA）")
        self.chk_gpu.setChecked(self.state.config.get("use_gpu", True))
        if DEPLOY_ONNX_ONLY:
            self.chk_gpu.setToolTip(
                "机台版通过 onnxruntime-gpu 调用 CUDA；"
                "若驱动/CUDA 不匹配将自动回退 CPU。"
            )
        ifl.addRow("加速设备:", self.chk_gpu)
        thr_hint = QLabel(
            "逐类置信度阈值由 checkpoints/class_thresholds.json 提供（训练校准生成）。"
        )
        thr_hint.setWordWrap(True)
        thr_hint.setStyleSheet("color:#78909C;font-size:11px;")
        ifl.addRow("", thr_hint)
        cl.addWidget(infer_grp)

        # ── 保存按钮 ──────────────────────────────────────────────
        btn_save = _mk_btn("保存并加载分类模型", "primary")
        btn_save.setFixedHeight(40)
        btn_save.clicked.connect(self._save_and_load)
        self.btn_save = btn_save
        cl.addWidget(btn_save)
        cl.addStretch()
        tabs.addTab(tab_cls, "分类配置")

        # ════════ Tab 2: 切片推理配置 ═════════════════════════════
        tab_sahi = QWidget()
        tab_sahi_outer = QVBoxLayout(tab_sahi)
        tab_sahi_outer.setContentsMargins(0, 0, 0, 0)

        sahi_scroll = QScrollArea()
        sahi_scroll.setWidgetResizable(True)
        sahi_scroll.setFrameShape(QFrame.NoFrame)
        sahi_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sahi_content = QWidget()
        sl = QVBoxLayout(sahi_content)
        sl.setSpacing(16)
        sl.setContentsMargins(4, 4, 12, 4)

        if not _HAS_SAHI_DEPS:
            warn = QFrame()
            warn.setStyleSheet(
                "QFrame{background:#FFF3E0;border:1px solid #FFB74D;border-radius:6px;}"
            )
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(12, 8, 12, 8)
            lbl = QLabel(_SAHI_DEPS_MSG.strip() or "切片推理依赖缺失")
            lbl.setStyleSheet("color:#E65100;font-size:12px;")
            lbl.setWordWrap(True)
            wl.addWidget(lbl)
            sl.addWidget(warn)

        # ── YOLO 检测模型 ─────────────────────────────────────────
        yolo_grp = QGroupBox("YOLO 检测模型")
        yl = QFormLayout(yolo_grp)
        yl.setLabelAlignment(Qt.AlignRight)

        self.yolo_edit = QLineEdit(self.state.config.get("yolo_path", ""))
        self.yolo_edit.setPlaceholderText("选择 YOLO .pt 权重文件")
        btn_yolo = _mk_btn("选择文件 …", "flat", 96)
        btn_yolo.clicked.connect(
            lambda: self._browse_file(self.yolo_edit, "YOLO 模型", "*.pt *.pth")
        )
        yolo_row = QHBoxLayout()
        yolo_row.addWidget(self.yolo_edit)
        yolo_row.addWidget(btn_yolo)
        self._sahi_browse_btns.append(btn_yolo)
        yl.addRow("模型路径:", yolo_row)

        self.cmb_sahi_device = QComboBox()
        self.cmb_sahi_device.addItems(["auto", "cuda:0", "cpu"])
        self.cmb_sahi_device.setMinimumWidth(112)
        self.cmb_sahi_device.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sah_dev = self.state.config.get("sahi_device", "auto")
        idx = self.cmb_sahi_device.findText(sah_dev)
        if idx >= 0:
            self.cmb_sahi_device.setCurrentIndex(idx)
        yl.addRow("设备:", self.cmb_sahi_device)
        sahi_dev_hint = QLabel(
            "auto：优先 GPU，若 GPU 架构不受当前 PyTorch 支持则自动回退 CPU。"
            "RTX 50 系列需安装支持 sm_120 的 PyTorch 才能使用 GPU。"
        )
        sahi_dev_hint.setWordWrap(True)
        sahi_dev_hint.setStyleSheet("color:#78909C;font-size:11px;")
        yl.addRow("", sahi_dev_hint)
        _tune_form_layout(yl, 140)
        sl.addWidget(yolo_grp)

        # ── SAHI 切片参数 ─────────────────────────────────────────
        sahi_grp = QGroupBox("SAHI 切片参数")
        spl = QFormLayout(sahi_grp)
        spl.setLabelAlignment(Qt.AlignRight)

        self.sp_slice = QSpinBox()
        self.sp_slice.setRange(320, 4096)
        self.sp_slice.setSingleStep(128)
        self.sp_slice.setValue(self.state.config.get("sahi_slice_size", 1280))
        _tune_spinbox(self.sp_slice)
        spl.addRow("切片大小 (px):", self.sp_slice)

        self.sp_overlap = QDoubleSpinBox()
        self.sp_overlap.setRange(0.0, 0.5)
        self.sp_overlap.setSingleStep(0.05)
        self.sp_overlap.setDecimals(2)
        self.sp_overlap.setValue(self.state.config.get("sahi_overlap", 0.20))
        _tune_spinbox(self.sp_overlap)
        spl.addRow("重叠率:", self.sp_overlap)

        self.sp_det_conf = QDoubleSpinBox()
        self.sp_det_conf.setRange(0.05, 0.95)
        self.sp_det_conf.setSingleStep(0.05)
        self.sp_det_conf.setDecimals(2)
        self.sp_det_conf.setValue(self.state.config.get("sahi_det_conf", 0.35))
        _tune_spinbox(self.sp_det_conf)
        spl.addRow("检测置信度:", self.sp_det_conf)

        self.sp_batch = QSpinBox()
        self.sp_batch.setRange(1, 64)
        self.sp_batch.setValue(self.state.config.get("sahi_batch_size", 8))
        _tune_spinbox(self.sp_batch)
        spl.addRow("切片批大小:", self.sp_batch)

        self.sp_padding = QSpinBox()
        self.sp_padding.setRange(0, 100)
        self.sp_padding.setValue(self.state.config.get("sahi_crop_padding", 15))
        _tune_spinbox(self.sp_padding)
        spl.addRow("裁剪边距 (px):", self.sp_padding)
        _tune_form_layout(spl, 140)
        sl.addWidget(sahi_grp)

        # ── 检测后处理（去误检 + 剔除边缘不完整目标）───────────────
        post_grp = QGroupBox("检测后处理")
        ppl = QFormLayout(post_grp)
        ppl.setLabelAlignment(Qt.AlignRight)

        self.sp_ios = QDoubleSpinBox()
        self.sp_ios.setRange(0.0, 1.0)
        self.sp_ios.setSingleStep(0.05)
        self.sp_ios.setDecimals(2)
        self.sp_ios.setValue(self.state.config.get("sahi_ios_thresh", 0.60))
        self.sp_ios.setToolTip(
            "交集/较小框面积 ≥ 此值判为包含冗余并去掉较小框；建议 0.55~0.65；0 关闭"
        )
        _tune_spinbox(self.sp_ios)
        ppl.addRow("IoS 包含抑制:", self.sp_ios)

        self.sp_min_area = QDoubleSpinBox()
        self.sp_min_area.setRange(0.0, 1.0)
        self.sp_min_area.setSingleStep(0.05)
        self.sp_min_area.setDecimals(2)
        self.sp_min_area.setValue(self.state.config.get("sahi_min_area_ratio", 0.45))
        self.sp_min_area.setToolTip("面积 < 本图中位面积 × 此比例的框判为误检；0 表示关闭")
        _tune_spinbox(self.sp_min_area)
        ppl.addRow("最小面积比例:", self.sp_min_area)

        self.sp_max_aspect = QDoubleSpinBox()
        self.sp_max_aspect.setRange(0.0, 20.0)
        self.sp_max_aspect.setSingleStep(0.1)
        self.sp_max_aspect.setDecimals(1)
        self.sp_max_aspect.setValue(float(self.state.config.get("sahi_max_aspect_ratio", 1.5)))
        self.sp_max_aspect.setToolTip(
            "长宽比 = max(宽,高)/min(宽,高)；超过此值的细长框视为异常并剔除；0 表示关闭"
        )
        _tune_spinbox(self.sp_max_aspect)
        ppl.addRow("最大长宽比:", self.sp_max_aspect)

        self.chk_edge = QCheckBox("剔除图像边缘不完整钻石")
        self.chk_edge.setChecked(bool(self.state.config.get("sahi_edge_filter", True)))
        ppl.addRow("边缘剔除:", self.chk_edge)

        self.sp_edge_margin = QSpinBox()
        self.sp_edge_margin.setRange(0, 500)
        self.sp_edge_margin.setSingleStep(1)
        self.sp_edge_margin.setValue(int(self.state.config.get("sahi_edge_margin_px", 20)))
        self.sp_edge_margin.setToolTip(
            "距图像四边的像素边距带；检测框任一边落入该带内即视为边缘不完整目标并剔除"
        )
        _tune_spinbox(self.sp_edge_margin)
        ppl.addRow("边缘边距 (px):", self.sp_edge_margin)
        _tune_form_layout(ppl, 140)
        sl.addWidget(post_grp)

        # ── 输出目录 ──────────────────────────────────────────────
        out_grp = QGroupBox("输出目录")
        ol = QHBoxLayout(out_grp)
        ol.setSpacing(10)
        self.sahi_out_edit = QLineEdit(
            self.state.config.get("sahi_output_dir", "sahi_output")
        )
        btn_sahi_out = _mk_btn("选择路径", "flat", 96)
        btn_sahi_out.setFixedHeight(34)
        btn_sahi_out.clicked.connect(
            lambda: self._browse_dir(self.sahi_out_edit, "SAHI 输出目录")
        )
        ol.addWidget(self.sahi_out_edit, 1)
        ol.addWidget(btn_sahi_out)
        self._sahi_browse_btns.append(btn_sahi_out)
        sl.addWidget(out_grp)

        # ── 保存按钮 ──────────────────────────────────────────────
        btn_sahi_save = _mk_btn("保存切片推理配置", "primary")
        btn_sahi_save.setFixedHeight(40)
        btn_sahi_save.clicked.connect(self._save_sahi)
        self.btn_sahi_save = btn_sahi_save
        sl.addWidget(btn_sahi_save)
        sl.addStretch()

        sahi_scroll.setWidget(sahi_content)
        tab_sahi_outer.addWidget(sahi_scroll)
        tabs.addTab(tab_sahi, "切片推理配置")

    def set_admin_locked(self, locked: bool) -> None:
        """切换只读 / 可编辑（默认只读，两个 Tab 同步）。"""
        self._admin_locked = locked
        self.lock_bar.set_locked(locked)
        # Tab 1 控件
        for edit in (self.pt_edit, self.onnx_edit, self.data_edit, self.corr_edit):
            edit.setReadOnly(locked)
        for btn in self._browse_btns:
            btn.setEnabled(not locked)
        self.chk_gpu.setEnabled(not locked)
        self.btn_save.setEnabled(not locked)
        # Tab 2 控件
        self.yolo_edit.setReadOnly(locked)
        self.cmb_sahi_device.setEnabled(not locked)
        for btn in self._sahi_browse_btns:
            btn.setEnabled(not locked)
        for w in (self.sp_slice, self.sp_overlap, self.sp_det_conf,
                  self.sp_batch, self.sp_padding, self.sahi_out_edit,
                  self.sp_ios, self.sp_min_area, self.sp_max_aspect,
                  self.chk_edge, self.sp_edge_margin):
            w.setEnabled(not locked)
        self.btn_sahi_save.setEnabled(not locked)

    def _browse_file(self, edit: QLineEdit, title: str, flt: str):
        if self._admin_locked:
            return
        path, _ = QFileDialog.getOpenFileName(self, title, "", flt)
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit, title: str):
        if self._admin_locked:
            return
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            edit.setText(path)

    def _save_and_load(self):
        """保存分类配置到根目录 app_config.json，并热加载引擎。"""
        if self._admin_locked:
            QMessageBox.warning(self, "只读模式", "请先输入密码解除只读后再保存设置。")
            return
        # 以内存配置为底，更新本 Tab 字段；_save_cfg 会补全全部默认键并重建文件
        cfg = dict(self.state.config)
        cfg.update({
            "pt_path":         self.pt_edit.text().strip(),
            "onnx_path":       self.onnx_edit.text().strip(),
            "data_dir":        self.data_edit.text().strip(),
            "corrections_dir": self.corr_edit.text().strip(),
            "use_gpu":         self.chk_gpu.isChecked(),
        })
        cfg.pop("conf_threshold", None)  # 旧配置残留键，推理未使用
        _save_cfg(cfg)
        self.state.config = _load_cfg()  # 与落盘一致，并补全缺键

        if not self.state.engine:
            self.settings_saved.emit("推理引擎不可用（未安装 inference_engine）")
            return
        try:
            pt = _resolve_cfg_path(cfg["pt_path"])
            onnx = _resolve_cfg_path(cfg["onnx_path"])
            msg = self.state.engine.load(
                pt or None,
                onnx or None,
                cfg["use_gpu"],
            )
            self.settings_saved.emit(msg)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            self.settings_saved.emit(f"加载失败: {exc}")

    def _save_sahi(self):
        """保存 SAHI 切片推理配置到根目录 app_config.json（不涉及模型热加载）。"""
        if self._admin_locked:
            QMessageBox.warning(self, "只读模式", "请先输入密码解除只读后再保存设置。")
            return
        cfg = dict(self.state.config)
        cfg.update({
            "yolo_path":         self.yolo_edit.text().strip(),
            "sahi_device":       self.cmb_sahi_device.currentText(),
            "sahi_slice_size":   self.sp_slice.value(),
            "sahi_overlap":      self.sp_overlap.value(),
            "sahi_det_conf":     self.sp_det_conf.value(),
            "sahi_batch_size":   self.sp_batch.value(),
            "sahi_crop_padding": self.sp_padding.value(),
            "sahi_output_dir":   self.sahi_out_edit.text().strip(),
            "sahi_ios_thresh":       self.sp_ios.value(),
            "sahi_min_area_ratio":   self.sp_min_area.value(),
            "sahi_max_aspect_ratio": self.sp_max_aspect.value(),
            "sahi_edge_filter":      self.chk_edge.isChecked(),
            "sahi_edge_margin_px":   self.sp_edge_margin.value(),
        })
        _save_cfg(cfg)
        self.state.config = _load_cfg()
        self.settings_saved.emit(f"切片推理配置已保存 → {CFG_FILE.name}")


# ═══════════════════════════════════════════════════════════
# 页面 0 ── 钻石检测分类（SAHI 切片检测 + 缺陷分类，导航栏第一位）
# ═══════════════════════════════════════════════════════════
class DiamondDetectPage(QWidget):
    """
    钻石检测分类页 — 导航栏第一位。

    用户侧操作:
      · 选择输入图像（多选文件 或 选择文件夹批量推理）
      · 选择结果保存目录
      · 点击「开始处理」→ SahiPipelineWorker 后台执行
      · 结果表格展示每张图的钻石数、缺陷类别分布、耗时
      · 完成后可打开结果保存文件夹

    每张图的输出目录含：
      · visualization_detection.jpg — 全分辨率，统一绿色检测框
      · visualization_classified.jpg — 全分辨率，按缺陷类别着色 + 中文标签
      · crops/、result.json、statistics.json

    SAHI 参数（YOLO 模型、切片大小、重叠率等）在「设置 → 切片推理配置」Tab 中
    配置并持久化到 app_config.json，本页面从 AppState.config 读取，不再暴露给用户。
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.worker: Optional[SahiPipelineWorker] = None
        self._img_paths: List[str] = []
        self._input_folder: str = ""   # 文件夹模式下的根目录
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("钻石检测分类")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        root.addWidget(title)

        # ── 依赖检查提示 ──────────────────────────────────────────
        if not _HAS_SAHI_DEPS:
            warn = QFrame()
            warn.setStyleSheet(
                "QFrame{background:#FFF3E0;border:1px solid #FFB74D;border-radius:6px;}"
            )
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(12, 8, 12, 8)
            lbl = QLabel(_SAHI_DEPS_MSG.strip() or "切片推理依赖缺失")
            lbl.setStyleSheet("color:#E65100;font-size:12px;")
            lbl.setWordWrap(True)
            wl.addWidget(lbl)
            root.addWidget(warn)

        # ── 输入图像（左 8 : 右 2 — 路径展示 / 选择按钮）────────────
        input_grp = QGroupBox("输入图像")
        il = QVBoxLayout(input_grp)
        il.setSpacing(6)

        input_split = QHBoxLayout()
        input_split.setSpacing(12)

        # 左侧 80%：已选文件夹或文件来源路径
        self.lbl_input_path = QLabel("未选择图像")
        self.lbl_input_path.setWordWrap(True)
        self.lbl_input_path.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_input_path.setMinimumHeight(72)
        self.lbl_input_path.setStyleSheet(
            "color:#37474F;font-size:16px;font-weight:500;padding:4px 2px;"
        )
        input_split.addWidget(self.lbl_input_path, 8)

        # 右侧 20%：选择按钮（纵向排列）
        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)
        btn_files = _mk_btn("选择文件", "flat")
        btn_files.setMinimumHeight(34)
        btn_files.clicked.connect(self._browse_files)
        btn_folder = _mk_btn("选择文件夹", "flat")
        btn_folder.setMinimumHeight(34)
        btn_folder.clicked.connect(self._browse_folder)
        btn_col.addWidget(btn_files)
        btn_col.addWidget(btn_folder)
        btn_col.addStretch()
        input_split.addLayout(btn_col, 2)
        il.addLayout(input_split)

        self.lbl_img_count = QLabel("")
        self.lbl_img_count.setStyleSheet("color:#546E7A;font-size:12px;")
        il.addWidget(self.lbl_img_count)
        root.addWidget(input_grp)

        # ── 结果保存目录（路径与按钮同一行）─────────────────────────
        out_grp = QGroupBox("结果保存目录")
        ol = QHBoxLayout(out_grp)
        ol.setSpacing(10)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("选择保存目录...")
        self.out_edit.setText(
            str(resolve_path(self.state.config.get("sahi_output_dir", "sahi_output")))
        )
        ol.addWidget(self.out_edit, 1)
        btn_out = _mk_btn("选择目录", "flat")
        btn_out.setMinimumWidth(96)
        btn_out.setFixedHeight(34)
        btn_out.clicked.connect(self._browse_output)
        ol.addWidget(btn_out)
        root.addWidget(out_grp)

        # ── 控制按钮 + 进度条 ─────────────────────────────────────
        ctrl = QHBoxLayout()
        self.btn_run = _mk_btn("▶  开始处理", "primary")
        self.btn_run.setFixedHeight(36)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = _mk_btn("■  停止", "danger", 80)
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        ctrl.addWidget(self.btn_run)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch()
        root.addLayout(ctrl)

        self.pbar = QProgressBar()
        self.pbar.setObjectName("sahi_pbar")
        self.pbar.setVisible(False)
        self.pbar.setFixedHeight(28)
        self.pbar.setTextVisible(True)
        self.pbar.setFormat("%v / %m")
        root.addWidget(self.pbar)

        self.lbl_stage = QLabel("")
        self.lbl_stage.setVisible(False)
        self.lbl_stage.setStyleSheet("color:#546E7A;font-size:12px;")
        self.lbl_stage.setWordWrap(True)
        root.addWidget(self.lbl_stage)

        # ── 结果统计表格 ──────────────────────────────────────────
        result_grp = QGroupBox("结果统计")
        rl = QVBoxLayout(result_grp)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["图像", "钻石数", "缺陷类别分布", "总耗时(s)", "保存目录"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setStyleSheet(
            "QTableWidget{font-size:14px;}"
            "QHeaderView::section{font-size:14px;padding:6px;}"
        )
        rl.addWidget(self.table)

        # 合计行
        self.lbl_total = QLabel("合计：0 张图 · 0 颗钻石")
        self.lbl_total.setStyleSheet("color:#1565C0;font-size:16px;font-weight:bold;")
        rl.addWidget(self.lbl_total)
        root.addWidget(result_grp, 1)

        # 保存控件引用以便依赖缺失时禁用
        self._btn_files = btn_files
        self._btn_folder = btn_folder
        self._btn_out = btn_out

        if not _HAS_SAHI_DEPS:
            for w in (btn_files, btn_folder, btn_out, self.btn_run, self.out_edit):
                w.setEnabled(False)

    def _update_file_display(self):
        """刷新左侧路径展示与图像计数。"""
        n = len(self._img_paths)
        if not n:
            self.lbl_input_path.setText("未选择图像")
            self.lbl_input_path.setToolTip("")
            self.lbl_img_count.setText("")
            return

        if self._input_folder:
            folder = str(Path(self._input_folder).resolve())
            self.lbl_input_path.setText(folder)
            self.lbl_input_path.setToolTip(folder)
        else:
            parents = {str(Path(p).resolve().parent) for p in self._img_paths}
            if len(parents) == 1:
                folder = next(iter(parents))
                self.lbl_input_path.setText(folder)
                self.lbl_input_path.setToolTip(
                    "\n".join(str(Path(p).resolve()) for p in self._img_paths[:20])
                    + (f"\n... 共 {n} 个文件" if n > 20 else "")
                )
            else:
                self.lbl_input_path.setText(f"已选 {n} 个文件（多个来源文件夹）")
                self.lbl_input_path.setToolTip(
                    "\n".join(str(Path(p).resolve()) for p in self._img_paths[:30])
                )
        self.lbl_img_count.setText(f"共 {n} 张图像")

    # ── 浏览按钮 ──────────────────────────────────────────────────

    def _browse_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择大图", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if paths:
            self._img_paths = [str(p) for p in paths]
            self._input_folder = ""
            self._update_file_display()

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            imgs = sorted(
                str(f) for f in Path(folder).rglob("*")
                if f.suffix.lower() in IMG_EXTS
            )
            self._img_paths = imgs
            self._input_folder = folder
            self._update_file_display()

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择结果保存目录")
        if folder:
            self.out_edit.setText(folder)

    # ── 运行 / 停止 ───────────────────────────────────────────────

    def _run(self):
        if not _HAS_SAHI_DEPS:
            QMessageBox.warning(self, "依赖缺失", _SAHI_DEPS_MSG)
            return

        cfg = self.state.config
        yolo_path = cfg.get("yolo_path", "").strip()
        if not yolo_path or not Path(yolo_path).exists():
            QMessageBox.warning(
                self, "提示",
                "未配置有效的 YOLO 模型，请先在「设置 → 切片推理配置」中选择。"
            )
            return
        if not self._img_paths:
            QMessageBox.warning(self, "提示", "请先选择输入图像。")
            return
        if not self.state.engine or not self.state.engine.loaded:
            QMessageBox.warning(self, "提示", "分类引擎未加载，请先在「设置 → 分类配置」中加载分类模型。")
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "提示", "请指定结果保存目录。")
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.table.setRowCount(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        n_img = len(self._img_paths)
        self.pbar.setVisible(True)
        self.pbar.setMaximum(max(1, n_img * 4))
        self.pbar.setValue(0)
        self.lbl_stage.setVisible(True)
        self.lbl_stage.setText("准备中…")

        self.worker = SahiPipelineWorker(
            yolo_path=yolo_path,
            img_paths=self._img_paths,
            output_dir=out_dir,
            classifier=self.state.engine,
            device=cfg.get("sahi_device", "auto"),
            slice_size=cfg.get("sahi_slice_size", 1280),
            overlap=cfg.get("sahi_overlap", 0.20),
            det_conf=cfg.get("sahi_det_conf", 0.35),
            batch_size=cfg.get("sahi_batch_size", 8),
            crop_padding=cfg.get("sahi_crop_padding", 15),
            ios_thresh=cfg.get("sahi_ios_thresh", 0.60),
            min_area_ratio=cfg.get("sahi_min_area_ratio", 0.45),
            max_aspect_ratio=cfg.get("sahi_max_aspect_ratio", 1.5),
            edge_filter=cfg.get("sahi_edge_filter", True),
            edge_margin_px=cfg.get("sahi_edge_margin_px", 20),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.stage_msg.connect(self._on_stage_msg)
        self.worker.image_done.connect(self._on_image_done)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag = True
            self.btn_stop.setEnabled(False)

    # ── 信号回调 ──────────────────────────────────────────────────

    def _on_progress(self, current: int, total: int):
        if self.pbar.maximum() != total and total > 0:
            self.pbar.setMaximum(total)
        self.pbar.setValue(min(current, total))

    def _on_stage_msg(self, msg: str):
        self.lbl_stage.setText(msg)

    def _on_image_done(self, idx: int, stats: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        dist = " | ".join(
            f"{k}:{v}" for k, v in stats.get("defect_counts", {}).items()
        )
        if not dist and stats.get("error"):
            dist = f"错误: {stats['error'][:40]}"
        # 后处理剔除摘要（便于确认 IoS 等是否生效）
        skipped = (
            int(stats.get("contained_skipped", 0))
            + int(stats.get("small_skipped", 0))
            + int(stats.get("aspect_skipped", 0))
            + int(stats.get("edge_skipped", 0))
        )
        if skipped and not stats.get("error"):
            dist = (dist + " · " if dist else "") + f"后处理剔除 {skipped}"
        items = [
            stats.get("image", ""),
            str(stats.get("total_diamonds", 0)),
            dist,
            f"{stats.get('total_time_s', 0):.2f}",
            stats.get("output_dir", ""),
        ]
        for c, text in enumerate(items):
            it = QTableWidgetItem(text)
            it.setTextAlignment(
                Qt.AlignCenter if c in (1, 3) else (Qt.AlignLeft | Qt.AlignVCenter)
            )
            self.table.setItem(row, c, it)
        self._update_total()

    def _on_finished(self, all_stats: list):
        self.pbar.setVisible(False)
        self.lbl_stage.setVisible(False)
        self.lbl_stage.setText("")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._update_total()

        if not all_stats:
            return

        out_dir = self.out_edit.text().strip()
        reply = QMessageBox.question(
            self,
            "处理完成",
            f"共处理 {len(all_stats)} 张图像。\n是否打开结果保存文件夹？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes and out_dir and Path(out_dir).is_dir():
            _open_folder(out_dir)

    def _on_error(self, msg: str):
        self.pbar.setVisible(False)
        self.lbl_stage.setVisible(False)
        self.lbl_stage.setText("")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "处理出错", msg)

    def _update_total(self):
        total_imgs = self.table.rowCount()
        total_diamonds = 0
        for r in range(total_imgs):
            item = self.table.item(r, 1)
            if item:
                try:
                    total_diamonds += int(item.text())
                except ValueError:
                    pass
        self.lbl_total.setText(f"合计：{total_imgs} 张图 · {total_diamonds} 颗钻石")


# ═══════════════════════════════════════════════════════════
# 主窗口 — QMainWindow + 侧栏 + QStackedWidget 多页容器
# ═══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    """
    主窗口 — 侧栏导航 + QStackedWidget 多页容器。

    职责:
      · 创建并持有唯一 AppState
      · 集中 connect 各 Page 的 pyqtSignal（跨页联动）
      · 启动时 _auto_load_model 尝试加载 checkpoints
      · 管理员密码验证（再训练 / 设置页共用一把锁）

    页面（开发 / 机台均为 6 页；机台分类走 ONNX，检测仍为 SAHI/YOLO）:
      0 钻石检测分类(SAHI) / 1 缺陷检测 / 2 结果管理
      3 误分类修正 / 4 模型再训练 / 5 设置(分类配置 + 切片推理配置)

    侧栏 nav 选中态: setProperty("sel", "true") + style().polish() 刷新 QSS。
    """

    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.setWindowTitle(APP_NAME)
        icon = _load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)
        self.setMinimumSize(1200, 800)
        self.resize(1440, 920)
        self._build_ui()
        self._auto_load_model()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 侧边栏 ────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(186)
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(0, 0, 0, 0)
        sb_lay.setSpacing(0)

        logo = QLabel(APP_NAME)
        logo.setObjectName("logo")
        logo.setWordWrap(True)
        ver_lbl = QLabel(f"  {APP_VER}")
        ver_lbl.setObjectName("logo_sub")
        sb_lay.addWidget(logo)
        sb_lay.addWidget(ver_lbl)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#CFD8DC;")
        sb_lay.addWidget(sep)

        # 导航项顺序与 NAV_* 常量一致；钻石检测分类位于第一位
        nav_items = [
            ("  💎  钻石检测分类", NAV_DIAMOND),
            ("  🔍  缺陷检测",     NAV_DETECT),
            ("  📋  结果管理",     NAV_RESULTS),
            ("  ✏️   误分类修正",   NAV_CORRECT),
            ("  🔄  模型再训练",   NAV_RETRAIN),
            ("  ⚙️   设 置",       NAV_SETTINGS),
        ]
        self._nav_btns = [None] * 6  # 按 NAV 索引存放按钮，便于按索引访问
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            sb_lay.addWidget(btn)
            self._nav_btns[idx] = btn

        sb_lay.addStretch()
        layout.addWidget(sidebar)

        # ── 内容区域 ──────────────────────────────
        self.stack = QStackedWidget()

        # 按 NAV_* 索引顺序添加页面，保证 setCurrentIndex(NAV_xxx) 一一对应
        self.p_diamond  = DiamondDetectPage(self.state)
        self.p_detect   = DetectionPage(self.state)
        self.p_results  = ResultsPage(self.state)
        self.p_correct  = CorrectionPage(self.state)
        self.p_retrain  = RetrainPage(self.state)
        self.p_settings = SettingsPage(self.state)

        # 顺序必须与 NAV_DIAMOND, NAV_DETECT, ... NAV_SETTINGS 一致
        for p in (self.p_diamond, self.p_detect, self.p_results,
                  self.p_correct, self.p_retrain, self.p_settings):
            self.stack.addWidget(p)

        layout.addWidget(self.stack)

        # ── 跨页信号连接 ────────────────────────────────────────
        # DetectionPage  → 刷新结果表 / 跳转页
        # ResultsPage    → 送修正后更新侧栏角标
        # CorrectionPage → 归档后刷新结果表
        # RetrainPage    → 热加载后更新状态栏；管理员锁
        # SettingsPage   → 保存后更新状态栏；管理员锁
        self.p_detect.results_ready.connect(self._on_new_results)
        self.p_detect.navigate_to.connect(self._switch_page)
        self.p_results.correction_updated.connect(self._on_correction_updated)
        self.p_correct.saved_signal.connect(self._on_correction_saved)
        self.p_retrain.model_updated.connect(self._on_model_updated)
        self.p_settings.settings_saved.connect(self._update_status)
        self.p_retrain.admin_unlock_requested.connect(self._try_unlock_admin_pages)
        self.p_settings.admin_unlock_requested.connect(self._try_unlock_admin_pages)
        self.p_retrain.admin_lock_requested.connect(self._lock_admin_pages)
        self.p_settings.admin_lock_requested.connect(self._lock_admin_pages)

        # ── 状态栏 ────────────────────────────────
        self._lbl_model  = QLabel("模型：未加载")
        self._lbl_device = QLabel("")
        self._lbl_badge  = QLabel("")
        self.statusBar().addWidget(self._lbl_model, 2)
        self.statusBar().addWidget(self._lbl_device, 1)
        self.statusBar().addPermanentWidget(self._lbl_badge)

        # 初始页：钻石检测分类
        self._switch_page(NAV_DIAMOND)
        self._update_correction_badge()

    def _switch_page(self, idx: int):
        """切换 stack 当前页，并更新侧栏 nav 按钮的 QSS 选中态。"""
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            if btn is None:
                continue
            btn.setProperty("sel", str(i == idx).lower())
            # 修改 dynamic property 后需 unpolish/polish 才会重新应用 QSS
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        # 进入结果页 / 修正页时主动 refresh（避免后台数据已变但表格未更新）
        if idx == NAV_CORRECT:
            self.p_correct.refresh()
        if idx == NAV_RESULTS:
            self.p_results.refresh()
        self._update_correction_badge()

    def _update_correction_badge(self):
        """侧栏「误分类修正」显示待处理数量角标。"""
        n = count_flagged_pending(self.state.results)
        if n:
            self._nav_btns[NAV_CORRECT].setText(f"  ✏️   误分类修正 ({n})")
        else:
            self._nav_btns[NAV_CORRECT].setText("  ✏️   误分类修正")

    def _try_unlock_admin_pages(self):
        if self.state.admin_unlocked:
            return
        if _verify_admin_password(self):
            self.state.admin_unlocked = True
            self.p_retrain.set_admin_locked(False)
            self.p_settings.set_admin_locked(False)
            QMessageBox.information(
                self, "已解锁",
                "模型再训练与设置页面已解除只读，可进行修改。"
            )

    def _lock_admin_pages(self):
        self.state.admin_unlocked = False
        self.p_retrain.set_admin_locked(True)
        self.p_settings.set_admin_locked(True)

    def _on_new_results(self, results: List[Dict]):
        n = len(results)
        self._lbl_badge.setText(f"本次检测：{n} 张")
        self.p_results.refresh()
        self._update_correction_badge()

    def _on_correction_updated(self, msg: str):
        self._lbl_badge.setText(msg)
        self._update_correction_badge()
        if self.stack.currentIndex() == NAV_CORRECT:
            self.p_correct.refresh()

    def _on_correction_saved(self, msg: str = ""):
        self._lbl_badge.setText(msg or "修正已归档")
        self._update_correction_badge()
        self.p_results.refresh()

    def _on_model_updated(self):
        if self.state.engine:
            dev = "GPU" if self.state.engine.device == "cuda" else "CPU"
            cls_n = len(self.state.engine.classes)
            self._lbl_model.setText(f"模型：已加载  {cls_n} 类  [{dev}]")
            self._lbl_device.setText(self.state.engine.backend.upper())

    def _update_status(self, msg: str):
        self._lbl_model.setText(msg)
        if self.state.engine and self.state.engine.loaded:
            dev = "GPU" if self.state.engine.device == "cuda" else "CPU"
            self._lbl_device.setText(f"[{self.state.engine.backend.upper()} / {dev}]")

    def _auto_load_model(self):
        """
        启动时自动加载模型。

        优先级: pt_path（开发版且文件存在）或 onnx_path；
        路径经 resolve_path 解析，支持 exe 同级 checkpoints/ 热替换。
        """
        if not self.state.engine:
            self._lbl_model.setText("推理引擎不可用（请检查 inference_engine）")
            return
        cfg = self.state.config
        pt   = _resolve_cfg_path(cfg.get("pt_path", ""))
        onnx = _resolve_cfg_path(cfg.get("onnx_path", ""))
        use_gpu = cfg.get("use_gpu", True)
        has_pt = pt and Path(pt).exists()
        has_onnx = onnx and Path(onnx).exists()
        if not (has_pt or has_onnx):
            self._lbl_model.setText(
                f"模型未找到（{app_dir() / 'checkpoints'}），请放置 model.onnx 后重启"
            )
            return
        try:
            msg = self.state.engine.load(
                pt if has_pt else None,
                onnx if has_onnx else None,
                use_gpu=use_gpu,
            )
            self._update_status(msg)
        except Exception as e:
            self._lbl_model.setText(f"自动加载失败: {e}")


# ═══════════════════════════════════════════════════════════
# 入口 — 创建 QApplication，进入 Qt 事件循环 exec_()
# ═══════════════════════════════════════════════════════════
def main():
    """
    应用入口。

    1. chdir_app_root — 打包后工作目录设为 exe 同级，保证相对路径读写正确
    2. freeze_support  — Windows/PyInstaller 多进程兼容
    3. QApplication    — 全局事件循环；Fusion 风格 + 自定义 QSS
    4. MainWindow.show → app.exec_() 阻塞直到退出
    """
    chdir_app_root()

    # Windows 打包/多进程：PyInstaller 下 spawn 子进程需 freeze_support
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()

    # 每个 Qt 程序有且仅有一个 QApplication，负责事件分发与全局样式
    app = QApplication(sys.argv)
    app.setStyle("Fusion")       # 跨平台统一基础风格，再叠加自定义 QSS
    app.setStyleSheet(STYLE)
    icon = _load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)

    # 中文字体回退链，避免 Linux/macOS 缺字
    font = QFont()
    for name in ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"):
        font.setFamily(name)
        if QFontInfo(font).family() == name:
            break
    font.setPointSize(10)
    app.setFont(font)

    win = MainWindow()
    win.show()                   # 非模态显示；模态对话框用 exec_()
    sys.exit(app.exec_())        # 阻塞直到最后一个窗口关闭


if __name__ == "__main__":
    main()
