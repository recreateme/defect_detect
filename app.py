#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像缺陷分类 PyQt5 部署应用
功能: 单图/批量检测 · 结果管理 · 误分类修正 · 模型再训练（热更新）
依赖: pip install PyQt5 torch torchvision Pillow onnxruntime
"""

import os
import sys
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

try:
    from inference_engine import InferenceEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    InferenceEngine = None  # type: ignore

# ═══════════════════════════════════════════════════════════
# 常量 & 配置
# ═══════════════════════════════════════════════════════════
APP_NAME = "钻石缺陷图像分类系统"
APP_VER  = "v1.0"
CFG_FILE = Path("app_config.json")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_SCRIPT = Path(__file__).parent / "train.py"

DEFAULT_CFG: Dict = {
    "pt_path":         "checkpoints/best_model.pt",
    "onnx_path":       "checkpoints/model.onnx",
    "data_dir":        "data",
    "corrections_dir": "corrections",
    "use_gpu":         True,
    "conf_threshold":  0.5,
}

# 模型再训练 / 设置页解除只读所需密码
ADMIN_PAGE_PASSWORD = "20250508"

# ── QSS ──────────────────────────────────────────────────
STYLE = """
* { font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif; }
QMainWindow, QDialog { background: #F0F2F5; }
/* ── Sidebar ── */
QWidget#sidebar  { background: #1E2D40; }
QPushButton#nav  {
    color:#7F98AE; background:transparent; text-align:left;
    padding:14px 14px 14px 18px; border:none; font-size:15px; border-radius:0;
}
QPushButton#nav:hover    { background:#253449; color:#B0C4D8; }
QPushButton#nav[sel=true]{
    background:#2E4B70; color:#FFFFFF;
    border-left:3px solid #42A5F5; padding-left:17px;
}
QLabel#logo    { color:#ECEFF1; font-size:14px; font-weight:bold; padding:20px 16px 4px 20px; }
QLabel#logo_sub{ color:#4A6070; font-size:11px; padding:0 0 18px 22px; }
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
# 共享状态
# ═══════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.engine: Optional[InferenceEngine] = (
            InferenceEngine() if HAS_ENGINE else None
        )
        self.results: List[Dict] = []
        self.config:  Dict       = _load_cfg()
        self.admin_unlocked: bool = False


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
    """只读模式提示条 + 解锁 / 重新锁定。"""

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


def _load_cfg() -> Dict:
    if CFG_FILE.exists():
        with open(CFG_FILE, "r", encoding="utf-8") as f:
            return {**DEFAULT_CFG, **json.load(f)}
    return dict(DEFAULT_CFG)


def _save_cfg(cfg: Dict):
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 后台工作线程
# ═══════════════════════════════════════════════════════════
class InferenceWorker(QThread):
    result_item = pyqtSignal(int, dict)
    progress    = pyqtSignal(int, int)
    done        = pyqtSignal(list)

    def __init__(self, engine, paths: List[str]):
        super().__init__()
        self.engine    = engine
        self.paths     = paths
        self.stop_flag = False

    def run(self):
        results = []
        for i, path in enumerate(self.paths):
            if self.stop_flag:
                break
            try:
                r = self.engine.predict(path)
            except Exception as exc:
                r = {
                    "path": str(path), "class": "ERROR", "class_idx": -1,
                    "confidence": 0.0, "all_scores": {}, "elapsed_ms": 0.0,
                    "true_class": "", "flagged": False,
                    "correction_saved": False, "error": str(exc),
                }
            results.append(r)
            self.result_item.emit(i, r)
            self.progress.emit(i + 1, len(self.paths))
        self.done.emit(results)


class TrainWorker(QThread):
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
        self.stop_flag = True
        if self._proc:
            self._proc.terminate()


# ═══════════════════════════════════════════════════════════
# 自定义小部件
# ═══════════════════════════════════════════════════════════
class ImageDropZone(QLabel):
    """支持点击选择与拖拽的图像显示区域。"""
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
    右侧预览控制器：悬停临时显示；鼠标离开后若已固定则恢复固定图，否则清空。
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
    btn = QPushButton(text)
    btn.setProperty("class", cls)
    if width:
        btn.setFixedWidth(width)
    return btn


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color:#E0E0E0;")
    return f


# ═══════════════════════════════════════════════════════════
# 页面 1 ── 缺陷检测
# ═══════════════════════════════════════════════════════════
class DetectionPage(QWidget):
    results_ready  = pyqtSignal(list)
    navigate_to    = pyqtSignal(int)

    def __init__(self, state: AppState):
        super().__init__()
        self.state  = state
        self.worker: Optional[InferenceWorker] = None
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
        nav_row.addStretch()
        btn_go = _mk_btn("查看全部结果 →", "primary")
        btn_go.clicked.connect(lambda: self.navigate_to.emit(1))
        nav_row.addWidget(btn_go)
        bl.addLayout(nav_row)

        tabs.addTab(batch, "批量检测")

    # ── 单图 ──────────────────────────────────────────────
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
            self._show_single(r)
        except Exception as e:
            QMessageBox.critical(self, "检测出错", str(e))
        finally:
            self.btn_detect.setEnabled(True)
            self.btn_detect.setText("▶  开始检测")

    def _show_single(self, r: Dict):
        conf  = r["confidence"]
        color = "#1B5E20" if conf >= 0.8 else ("#E65100" if conf >= 0.5 else "#B71C1C")
        self.lbl_cls.setText(r["class"])
        self.lbl_cls.setStyleSheet(f"font-size:28px;font-weight:bold;color:{color};")
        self.lbl_conf.setText(f"置信度：{conf*100:.1f}%")
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

    # ── 批量 ──────────────────────────────────────────────
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
        imgs = sorted(str(f) for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXTS)
        if not imgs:
            QMessageBox.warning(self, "提示", "所选文件夹中没有支持的图像文件。")
            return

        self.btable.setRowCount(0)
        self.state.results.clear()
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
        self.state.results.append(r)
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
        self.results_ready.emit(results)

    def _stop_batch(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop_flag = True
            self.btn_stop.setEnabled(False)


# ═══════════════════════════════════════════════════════════
# 页面 2 ── 结果管理
# ═══════════════════════════════════════════════════════════
class ResultsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
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

        QLabel_c = QLabel("置信度 ≥")
        QLabel_c.setStyleSheet("color:#546E7A;")
        toolbar.addWidget(QLabel_c)
        self.filter_conf = QDoubleSpinBox()
        self.filter_conf.setRange(0.0, 1.0)
        self.filter_conf.setSingleStep(0.05)
        self.filter_conf.setValue(0.0)
        self.filter_conf.setFixedWidth(150)
        self.filter_conf.valueChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_conf)

        toolbar.addStretch()

        self._all_checked = True
        self.btn_toggle_sel = _mk_btn("全不选", "flat", 88)
        self.btn_toggle_sel.clicked.connect(self._toggle_all_checks)
        toolbar.addWidget(self.btn_toggle_sel)

        btn_flag_sel = _mk_btn("标记选中项", "warning")
        btn_flag_sel.clicked.connect(self._flag_selected)
        toolbar.addWidget(btn_flag_sel)

        btn_flag_all = _mk_btn("全部标记为待修正", "warning")
        btn_flag_all.clicked.connect(self._flag_visible)
        toolbar.addWidget(btn_flag_all)

        btn_export = _mk_btn("导出 CSV", "flat")
        btn_export.clicked.connect(self._export_csv)
        toolbar.addWidget(btn_export)

        root.addLayout(toolbar)

        content = QHBoxLayout()
        content.setSpacing(12)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["", "缩略图", "文件名", "预测类别", "置信度", "修正类别", "操作"]
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(1, 72)
        self.table.setColumnWidth(3, 120)
        self.table.setColumnWidth(4, 80)
        self.table.setColumnWidth(5, 130)
        self.table.setColumnWidth(6, 90)
        self.table.setRowHeight(0, 72)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_check_changed)
        content.addWidget(self.table, 1)

        self.preview = ImagePreviewSidePanel(min_size=300)
        self._preview_ctrl = SidePreviewController(self.preview)
        self.preview.bind_controller(self._preview_ctrl)
        content.addWidget(self.preview)

        root.addLayout(content, 1)

        # 底部状态
        self.lbl_bottom = QLabel("")
        self.lbl_bottom.setStyleSheet("color:#546E7A;font-size:12px;")
        root.addWidget(self.lbl_bottom)

    def refresh(self):
        """结果更新后重新填充表格。"""
        results = self.state.results

        # 更新类别筛选下拉
        classes = sorted({r["class"] for r in results if r.get("class") and r["class"] != "ERROR"})
        self.filter_cls.blockSignals(True)
        self.filter_cls.clear()
        self.filter_cls.addItem("全部")
        for c in classes:
            self.filter_cls.addItem(c)
        self.filter_cls.blockSignals(False)

        self._fill_table(results)

    def _meta_for_result(self, r: Dict) -> str:
        return f"预测：{r['class']}  ·  {r['confidence']*100:.1f}%"

    def _fill_table(self, results: List[Dict]):
        cls_f  = self.filter_cls.currentText()
        conf_f = self.filter_conf.value()

        filtered = [
            r for r in results
            if (cls_f == "全部" or r["class"] == cls_f)
            and r["confidence"] >= conf_f
        ]

        self.table.setRowCount(0)
        self.table.setRowCount(len(filtered))

        for row, r in enumerate(filtered):
            self.table.setRowHeight(row, 72)
            idx = self.state.results.index(r)   # 原始索引

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

            # 置信度
            conf_item = QTableWidgetItem(f"{r['confidence']*100:.1f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 4, conf_item)

            # 修正类别（可编辑下拉）
            combo = QComboBox()
            combo.addItem("未修正")
            for c in self.state.engine.classes if self.state.engine else []:
                combo.addItem(c)
            true_cls = r.get("true_class", "")
            if true_cls and true_cls in (self.state.engine.classes if self.state.engine else []):
                combo.setCurrentText(true_cls)
            combo.currentTextChanged.connect(lambda txt, i=idx: self._set_true_class(i, txt))
            self.table.setCellWidget(row, 5, combo)

            # 操作按钮
            btn = QPushButton("🚩 标记" if not r.get("flagged") else "✓ 已标记")
            btn.setStyleSheet(
                "background:#F57C00;color:white;border:none;border-radius:4px;padding:4px 8px;"
                if not r.get("flagged") else
                "background:#388E3C;color:white;border:none;border-radius:4px;padding:4px 8px;"
            )
            btn.clicked.connect(lambda _, i=idx: self._toggle_flag(i))
            self.table.setCellWidget(row, 6, btn)

        n_flagged = sum(1 for r in results if r.get("flagged"))
        self.lbl_stat.setText(f"共 {len(results)} 条")
        self.lbl_bottom.setText(
            f"显示 {len(filtered)} 条  |  已标记待修正 {n_flagged} 条"
        )
        if self._preview_ctrl.pinned_path:
            p = self._preview_ctrl.pinned_path
            for r in results:
                if r["path"] == p:
                    self.preview.show_image(p, self._meta_for_result(r))
                    self.preview.set_pinned(True)
                    break

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

    def _flag_selected(self):
        self._sync_checks_from_table()
        for r in self.state.results:
            if r.get("_checked", False):
                r["flagged"] = True
        self._fill_table(self.state.results)

    def _set_true_class(self, idx: int, text: str):
        if 0 <= idx < len(self.state.results):
            val = "" if text == "—（未修正）" else text
            self.state.results[idx]["true_class"] = val

    def _toggle_flag(self, idx: int):
        if 0 <= idx < len(self.state.results):
            self._sync_checks_from_table()
            self.state.results[idx]["flagged"] = not self.state.results[idx]["flagged"]
            self._fill_table(self.state.results)

    def _flag_visible(self):
        self._sync_checks_from_table()
        cls_f  = self.filter_cls.currentText()
        conf_f = self.filter_conf.value()
        for r in self.state.results:
            if (cls_f == "全部" or r["class"] == cls_f) and r["confidence"] >= conf_f:
                r["flagged"] = True
        self._fill_table(self.state.results)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["文件路径", "文件名", "预测类别", "置信度", "修正类别", "已标记"])
            for r in self.state.results:
                writer.writerow([
                    r["path"], Path(r["path"]).name,
                    r["class"], f"{r['confidence']:.4f}",
                    r.get("true_class", ""), r.get("flagged", False),
                ])
        QMessageBox.information(self, "导出成功", f"已保存至：{path}")


# ═══════════════════════════════════════════════════════════
# 页面 3 ── 误分类修正
# ═══════════════════════════════════════════════════════════
class CorrectionPage(QWidget):
    saved_signal = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)

        # 标题行
        hdr = QHBoxLayout()
        title = QLabel("误分类修正")
        title.setStyleSheet("font-size:18px;font-weight:bold;color:#263238;")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_count = QLabel("待修正：0 项")
        self.lbl_count.setStyleSheet("color:#E65100;font-weight:bold;")
        hdr.addWidget(self.lbl_count)

        btn_sel_all = _mk_btn("全不选", "flat", 88)
        btn_sel_all.clicked.connect(self._toggle_all_checks)
        hdr.addWidget(btn_sel_all)
        self.btn_toggle_sel = btn_sel_all
        self._all_checked = True

        btn_save_all = _mk_btn("保存选中修正", "success")
        btn_save_all.clicked.connect(self._save_all)
        hdr.addWidget(btn_save_all)
        root.addLayout(hdr)

        lbl_hint = QLabel(
            '勾选需要归档的条目，选择正确类别后点击「保存选中修正」。'
            '悬停缩略图或文件名在右侧预览；单击文件名固定预览，单击缩略图打开原图。'
        )
        lbl_hint.setStyleSheet("color:#546E7A;font-size:12px;")
        lbl_hint.setWordWrap(True)
        root.addWidget(lbl_hint)

        body = QHBoxLayout()
        body.setSpacing(12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.card_container = QWidget()
        self.card_lay = QVBoxLayout(self.card_container)
        self.card_lay.setSpacing(8)
        self.card_lay.setContentsMargins(0, 0, 0, 0)
        self.card_lay.addStretch()
        scroll.setWidget(self.card_container)
        body.addWidget(scroll, 1)

        self.preview = ImagePreviewSidePanel(min_size=280)
        self._preview_ctrl = SidePreviewController(self.preview)
        self.preview.bind_controller(self._preview_ctrl)
        body.addWidget(self.preview)

        root.addLayout(body, 1)

        self._card_checks: List[QCheckBox] = []

    def _toggle_all_checks(self):
        self._all_checked = not self._all_checked
        self._set_all_checks(self._all_checked)

    def _set_all_checks(self, checked: bool):
        self._all_checked = checked
        self.btn_toggle_sel.setText("全不选" if checked else "全选")
        for cb in self._card_checks:
            cb.setChecked(checked)

    def refresh(self):
        """刷新已标记待修正的条目。"""
        self._card_checks.clear()
        self.preview.clear_preview()
        # 清空旧卡片
        while self.card_lay.count() > 1:
            it = self.card_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        flagged = [(i, r) for i, r in enumerate(self.state.results) if r.get("flagged")]
        self.lbl_count.setText(f"待修正：{len(flagged)} 项")

        classes = self.state.engine.classes if self.state.engine else []

        for orig_idx, r in flagged:
            card = self._make_card(orig_idx, r, classes)
            self.card_lay.insertWidget(self.card_lay.count() - 1, card)

    def _make_card(self, orig_idx: int, r: Dict, classes: List[str]) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setFixedHeight(100)
        cl = QHBoxLayout(card)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(12)

        chk = QCheckBox()
        chk.setChecked(True)
        chk.setProperty("orig_idx", orig_idx)
        chk.setToolTip("勾选后参与「保存选中修正」")
        self._card_checks.append(chk)
        cl.addWidget(chk)

        # 缩略图：悬停预览，单击打开原图
        meta = f"预测：{r['class']}  ·  {r['confidence']*100:.1f}%"
        thumb_lbl = ThumbnailLabel(r["path"], 72) if Path(r["path"]).exists() else QLabel()
        if isinstance(thumb_lbl, ThumbnailLabel):
            _connect_thumb_preview(thumb_lbl, self._preview_ctrl, meta)
        else:
            thumb_lbl.setFixedSize(72, 72)
        cl.addWidget(thumb_lbl)

        # 文件信息
        info = QVBoxLayout()
        info.setSpacing(4)
        fn_lbl = HoverFilenameLabel(r["path"], Path(r["path"]).name)
        fn_lbl.setStyleSheet(
            "font-weight:bold;color:#263238;font-size:15px;padding:2px 0;"
        )
        fn_lbl.setMinimumWidth(220)
        fn_lbl.setWordWrap(True)
        _connect_filename_preview(fn_lbl, self._preview_ctrl, meta)
        pred_lbl = QLabel(f"预测：{r['class']}  ({r['confidence']*100:.1f}%)")
        pred_lbl.setStyleSheet("color:#C62828;font-size:13px;")
        info.addWidget(fn_lbl)
        info.addWidget(pred_lbl)
        cl.addLayout(info, 1)

        # 正确类别下拉
        combo = QComboBox()
        for c in classes:
            combo.addItem(c)
        current_true = r.get("true_class", "")
        if current_true in classes:
            combo.setCurrentText(current_true)
        combo.setFixedWidth(140)
        combo.currentTextChanged.connect(
            lambda txt, i=orig_idx: self.state.results.__setitem__(
                i, {**self.state.results[i], "true_class": txt}
            )
        )
        cl.addWidget(QLabel("→ 正确类别："))
        cl.addWidget(combo)

        # 状态指示
        saved = r.get("correction_saved", False)
        status = QLabel("✓ 已归档" if saved else "")
        status.setStyleSheet("color:#2E7D32;font-weight:bold;")
        status.setFixedWidth(60)
        cl.addWidget(status)

        card.setProperty("orig_idx", orig_idx)
        card.setProperty("result_ref", r)
        return card

    def _save_all(self):
        """将勾选且已设置正确类别的条目保存到 corrections/ 目录。"""
        corrections_dir = Path(self.state.config.get("corrections_dir", "corrections"))
        saved, skipped  = 0, 0

        checked_indices = {
            cb.property("orig_idx")
            for cb in self._card_checks
            if cb.isChecked() and cb.property("orig_idx") is not None
        }
        if not checked_indices:
            QMessageBox.warning(self, "提示", "请先勾选需要保存的条目。")
            return

        for i, r in enumerate(self.state.results):
            if not r.get("flagged"):
                continue
            if i not in checked_indices:
                continue
            true_cls = r.get("true_class", "").strip()
            if not true_cls:
                skipped += 1
                continue
            src = Path(r["path"])
            if not src.exists():
                skipped += 1
                continue
            dst_dir = corrections_dir / true_cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            # 若重名则加序号
            if dst.exists():
                stem, suf = src.stem, src.suffix
                n = 1
                while dst.exists():
                    dst = dst_dir / f"{stem}_{n}{suf}"
                    n += 1
            shutil.copy2(src, dst)
            r["correction_saved"] = True
            saved += 1

        msg = f"已归档 {saved} 张图像到 corrections/ 目录。"
        if skipped:
            msg += f"\n{skipped} 项未设置正确类别或文件不存在，已跳过。"
        QMessageBox.information(self, "保存完成", msg)
        self.refresh()
        self.saved_signal.emit()


# ═══════════════════════════════════════════════════════════
# 页面 4 ── 模型再训练
# ═══════════════════════════════════════════════════════════
class RetrainPage(QWidget):
    model_updated = pyqtSignal()
    admin_unlock_requested = pyqtSignal()
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
# 页面 5 ── 设置
# ═══════════════════════════════════════════════════════════
class SettingsPage(QWidget):
    settings_saved = pyqtSignal(str)   # emits status message
    admin_unlock_requested = pyqtSignal()
    admin_lock_requested   = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._admin_locked = True
        self._browse_btns: List[QPushButton] = []
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

        # ── 模型文件 ──────────────────────────────
        model_grp = QGroupBox("模型文件")
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

        root.addWidget(model_grp)

        # ── 数据路径 ──────────────────────────────
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

        root.addWidget(data_grp)

        # ── 推理设置 ──────────────────────────────
        infer_grp = QGroupBox("推理设置")
        ifl = QFormLayout(infer_grp)
        ifl.setLabelAlignment(Qt.AlignRight)

        self.chk_gpu = QCheckBox("使用 GPU（需 CUDA）")
        self.chk_gpu.setChecked(self.state.config.get("use_gpu", True))
        ifl.addRow("加速设备:", self.chk_gpu)

        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.0, 1.0)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(self.state.config.get("conf_threshold", 0.5))
        ifl.addRow("低置信度阈值:", self.sp_conf)

        root.addWidget(infer_grp)
        root.addStretch()

        # ── 保存按钮 ──────────────────────────────
        btn_save = _mk_btn("保存并加载模型", "primary")
        btn_save.setFixedHeight(40)
        btn_save.clicked.connect(self._save_and_load)
        self.btn_save = btn_save
        root.addWidget(btn_save)

    def set_admin_locked(self, locked: bool) -> None:
        """切换只读 / 可编辑（默认只读）。"""
        self._admin_locked = locked
        self.lock_bar.set_locked(locked)
        for edit in (self.pt_edit, self.onnx_edit, self.data_edit, self.corr_edit):
            edit.setReadOnly(locked)
        for btn in self._browse_btns:
            btn.setEnabled(not locked)
        self.chk_gpu.setEnabled(not locked)
        self.sp_conf.setEnabled(not locked)
        self.btn_save.setEnabled(not locked)

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
        if self._admin_locked:
            QMessageBox.warning(self, "只读模式", "请先输入密码解除只读后再保存设置。")
            return
        cfg = {
            "pt_path":         self.pt_edit.text().strip(),
            "onnx_path":       self.onnx_edit.text().strip(),
            "data_dir":        self.data_edit.text().strip(),
            "corrections_dir": self.corr_edit.text().strip(),
            "use_gpu":         self.chk_gpu.isChecked(),
            "conf_threshold":  self.sp_conf.value(),
        }
        _save_cfg(cfg)
        self.state.config.update(cfg)

        if not self.state.engine:
            self.settings_saved.emit("推理引擎不可用（未安装 inference_engine）")
            return
        try:
            msg = self.state.engine.load(
                cfg["pt_path"] or None,
                cfg["onnx_path"] or None,
                cfg["use_gpu"],
            )
            self.settings_saved.emit(msg)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            self.settings_saved.emit(f"加载失败: {exc}")


# ═══════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.setWindowTitle(APP_NAME)
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
        sep.setStyleSheet("background:#2C3E50;")
        sb_lay.addWidget(sep)

        nav_items = [
            ("  🔍  缺陷检测",   0),
            ("  📋  结果管理",   1),
            ("  ✏️   误分类修正", 2),
            ("  🔄  模型再训练", 3),
            ("  ⚙️   设 置",     4),
        ]
        self._nav_btns = []
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setObjectName("nav")
            btn.clicked.connect(lambda _, i=idx: self._switch_page(i))
            sb_lay.addWidget(btn)
            self._nav_btns.append(btn)

        sb_lay.addStretch()
        layout.addWidget(sidebar)

        # ── 内容区域 ──────────────────────────────
        self.stack = QStackedWidget()

        self.p_detect  = DetectionPage(self.state)
        self.p_results = ResultsPage(self.state)
        self.p_correct = CorrectionPage(self.state)
        self.p_retrain = RetrainPage(self.state)
        self.p_settings = SettingsPage(self.state)

        for p in (self.p_detect, self.p_results, self.p_correct,
                  self.p_retrain, self.p_settings):
            self.stack.addWidget(p)

        layout.addWidget(self.stack)

        # ── 连接信号 ──────────────────────────────
        self.p_detect.results_ready.connect(self._on_new_results)
        self.p_detect.navigate_to.connect(self._switch_page)
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

        self._switch_page(0)

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.setProperty("sel", str(i == idx).lower())
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        # 进入修正页时刷新
        if idx == 2:
            self.p_correct.refresh()
        # 进入结果页时刷新
        if idx == 1:
            self.p_results.refresh()

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
        # 徽标：待修正计数
        flagged = sum(1 for r in results if r.get("confidence", 1) < self.state.config.get("conf_threshold", 0.5))
        if flagged:
            self._nav_btns[2].setText(f"  ✏️   误分类修正 ({flagged})")

    def _on_correction_saved(self):
        self._lbl_badge.setText("修正已归档")
        self._nav_btns[2].setText("  ✏️   误分类修正")

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
        """启动时尝试自动加载模型。"""
        if not self.state.engine:
            self._lbl_model.setText("推理引擎不可用（请检查 inference_engine.py）")
            return
        cfg = self.state.config
        pt   = cfg.get("pt_path", "")
        onnx = cfg.get("onnx_path", "")
        if not (Path(pt).exists() or (onnx and Path(onnx).exists())):
            self._lbl_model.setText('模型未找到，请前往「设置」配置路径')
            return
        try:
            msg = self.state.engine.load(
                pt or None, onnx or None, cfg.get("use_gpu", True)
            )
            self._update_status(msg)
        except Exception as e:
            self._lbl_model.setText(f"自动加载失败: {e}")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════
def main():
    # Windows 多进程安全
    if sys.platform == "win32":
        import multiprocessing
        multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    font = QFont()
    for name in ("Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC"):
        font.setFamily(name)
        if QFontInfo(font).family() == name:
            break
    font.setPointSize(10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
