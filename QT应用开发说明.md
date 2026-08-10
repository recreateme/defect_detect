# PyQt5 应用开发说明 — 以「钻石缺陷图像分类系统」为例

本文档结合本项目 `app.py` 的实际代码，介绍 PyQt5 的核心概念与常用 API，便于你阅读源码、修改界面或开发新的 Qt 桌面程序。

---

## 1. 本项目 UI 结构一览

```
main()
  └─ QApplication          # 全局应用对象，管理事件循环
       └─ MainWindow       # QMainWindow 主窗口
            ├─ sidebar     # QWidget + QVBoxLayout，固定宽度导航
            │    └─ QPushButton × 5~6（objectName="nav"）
            └─ QStackedWidget（stack）
                 ├─ DiamondDetectPage  钻石检测分类（SAHI，仅开发版显示入口）
                 ├─ DetectionPage    缺陷检测
                 ├─ ResultsPage      结果管理
                 ├─ CorrectionPage   误分类修正
                 ├─ RetrainPage      模型再训练
                 └─ SettingsPage     设置（分类配置 + 切片推理配置 Tab）
```

**设计要点**

| 模式 | 本项目中的用法 |
|------|----------------|
| 多页切换 | `QStackedWidget.setCurrentIndex(i)`，索引与 `NAV_*` 常量一致 |
| 共享数据 | 所有 Page 传入同一个 `AppState` 实例 |
| 跨页通信 | 子页面定义 `pyqtSignal`，在 `MainWindow._build_ui` 里 `connect` |
| 长任务 | `QThread` 子类 + 信号回传，不在主线程做推理/训练/SAHI |

对应源码：`MainWindow._build_ui()`（约 **3560** 行起）。

**双入口说明**：`app.py` 与 `app_deploy.py` 共用同一套 UI。机台入口在 import 前设置 `DEFECTS_DEPLOY=1`，从而选用 `inference_engine_onnx`、隐藏「钻石检测分类」导航与 SAHI 相关配置，并在窗口标题追加「· 机台版」。

---

## 2. PyQt5 三个核心模块

```python
from PyQt5.QtWidgets import *   # 窗口、按钮、表格、布局、对话框
from PyQt5.QtCore import *        # 信号槽、线程、定时器、Qt 枚举
from PyQt5.QtGui import *         # 字体、颜色、QPixmap、QCursor
```

| 模块 | 典型类 | 作用 |
|------|--------|------|
| QtWidgets | `QMainWindow`, `QWidget`, `QPushButton`, `QTableWidget`, `QLayout` | 可见控件与布局 |
| QtCore | `QObject`, `QThread`, `pyqtSignal`, `QTimer`, `Qt` | 对象模型与异步 |
| QtGui | `QFont`, `QPixmap`, `QColor`, `QDesktopServices` | 绘制与系统服务 |

---

## 3. 程序入口：QApplication 与事件循环

每个 Qt 程序必须先创建 `QApplication`，再进入事件循环：

```python
app = QApplication(sys.argv)
app.setStyle("Fusion")           # 基础主题
app.setStyleSheet(STYLE)         # 全局 QSS 样式表

win = MainWindow()
win.show()
sys.exit(app.exec_())            # 阻塞，处理鼠标/键盘/定时器等事件
```

**概念**

- **事件循环** `exec_()`：GUI 线程不断取事件并分发给控件；你的槽函数、重写的 `*Event` 都在此线程执行。
- **规则**：耗时计算（推理、训练、大文件 IO）不要放在槽函数里直接跑，应放 `QThread.run()`，用信号把结果传回界面。

本项目入口：`app.py` 末尾 `main()`（`app_deploy.py` 在设置环境变量后 `from app import main` 调用同一函数）。

`main()` 内会先 `chdir_app_root()`（打包后工作目录设为 exe 同级），Windows 下调用 `multiprocessing.freeze_support()` 以兼容 PyInstaller 子进程。

---

## 4. 布局（Layout）

Qt 不推荐用绝对坐标摆控件，而是用布局管理器自动伸缩。

| 布局类 | 常用场景 | 本项目示例 |
|--------|----------|------------|
| `QVBoxLayout` | 纵向排列 | 侧栏、单页根布局 |
| `QHBoxLayout` | 横向排列 | 表格 + 右侧预览、按钮行 |
| `QFormLayout` | 标签-控件成对 | 设置页、训练参数表单 |
| `QGridLayout` | 网格 | （本项目较少使用） |

**常用 API**

```python
layout = QVBoxLayout(self)           # 设为 widget 的主布局
layout.setContentsMargins(24, 20, 24, 20)  # 外边距
layout.setSpacing(14)                # 控件间距
layout.addWidget(title)              # 加控件
layout.addLayout(sub_row)            # 嵌套子布局
layout.addStretch()                  # 弹性空白，把前面控件顶上去
layout.addWidget(panel, 1)           # stretch=1 表示占据剩余空间
```

**示例**：`DetectionPage._build()` 用 `QTabWidget` 内嵌 `QHBoxLayout`（拖放区 | 结果卡片）。

---

## 5. 信号与槽（Signals & Slots）

Qt 的「观察者模式」：对象状态变化时 `emit` 信号，已 `connect` 的槽函数被调用。

### 5.1 内置信号

```python
btn.clicked.connect(self._run_batch)           # QPushButton
self.folder_edit.textChanged.connect(...)      # QLineEdit
self.sp_conf.valueChanged.connect(...)         # QSpinBox / QDoubleSpinBox
```

### 5.2 自定义信号（pyqtSignal）

在 `QObject` 子类（含 `QWidget`、`QThread`）中定义：

```python
class DetectionPage(QWidget):
    results_ready = pyqtSignal(list)    # 参数类型：list
    navigate_to   = pyqtSignal(int)

# 主窗口连接
self.p_detect.results_ready.connect(self._on_new_results)
self.p_detect.navigate_to.connect(self._switch_page)
```

**本项目自定义信号一览**

| 类 | 信号 | 含义 |
|----|------|------|
| `InferenceWorker` | `result_item`, `progress`, `done` | 批量推理进度与完成 |
| `SahiPipelineWorker` | `progress`, `image_done`, `finished_all`, `error` | SAHI 大图流水线 |
| `TrainWorker` | `log_line`, `finished` | 训练日志与结束 |
| `ImageDropZone` | `image_dropped(str)` | 选图/拖图完成 |
| `ThumbnailLabel` | `preview_hover`, `preview_leave`, `clicked_path` | 悬停预览 / 打开原图 |
| `HoverFilenameLabel` | `pin_requested` | 单击固定右侧预览 |
| `AdminLockBar` | `unlock_requested`, `lock_requested` | 管理员解锁 |
| `DetectionPage` | `results_ready`, `navigate_to` | 结果同步、跳转页 |
| `CorrectionPage` | `saved_signal` | 修正已保存 |
| `RetrainPage` | `admin_unlock_requested`, `model_updated` | 请求解锁 / 新模型已加载 |
| `SettingsPage` | `admin_unlock_requested` | 请求解锁设置页 |

### 5.3 Lambda 与默认参数陷阱

连接信号时若用 lambda 捕获循环变量，需用默认参数固定值：

```python
# 正确：i=idx 在定义时绑定
btn.clicked.connect(lambda _, i=idx: self._switch_page(i))

# 错误：所有按钮都会指向最后一次循环的 idx
btn.clicked.connect(lambda: self._switch_page(idx))
```

---

## 6. QThread：后台线程

### 6.1 为什么需要

主线程负责绘制界面。若在按钮槽里循环推理 1000 张图，窗口会「未响应」。  
做法：把循环放进 `QThread.run()`，每完成一张 `emit` 信号，主线程槽函数更新表格。

### 6.2 本项目 InferenceWorker

批量检测走 `engine.predict_batch`（GPU 下按批一次 `session.run` / forward，比逐张 `predict` 更快），
通过回调把每条结果 `emit` 回主线程更新表格。

```python
class InferenceWorker(QThread):
    result_item = pyqtSignal(int, dict)
    progress    = pyqtSignal(int, int)
    done        = pyqtSignal(list)

    def run(self):
        results = self.engine.predict_batch(
            self.paths,
            progress_cb=lambda c, t: self.progress.emit(c, t),
            result_cb=lambda i, r: self.result_item.emit(i, r),
            should_stop=lambda: self.stop_flag,
        )
        self.done.emit(results)
```

**使用步骤**

```python
self.worker = InferenceWorker(engine, paths)
self.worker.result_item.connect(self._on_item)
self.worker.progress.connect(lambda c, _: self.pbar.setValue(c))
self.worker.done.connect(self._on_done)
self.worker.start()    # 启动线程，自动调用 run()
```

**注意**

- 只在 `run()` 里访问引擎推理；**不要**在子线程直接改 `QTableWidget`。
- 停止任务：设 `stop_flag`，`predict_batch` 的 `should_stop` 会在下一批前退出。
- 开发版引擎为 `inference_engine.py`（PyTorch + ONNX 回退）；机台为 `inference_engine_onnx.py`，二者均调用 `inference_common.run_batch_predict`，接口一致。
- 单张检测完成后会 `_upsert_result` 写入 `AppState.results` 并刷新结果管理页，避免同路径重复记录。
- 线程对象建议挂到 `self.worker`，避免被 Python GC 回收导致崩溃。

### 6.3 TrainWorker

通过 `subprocess.Popen` 跑 `train.py`，逐行读 stdout 并 `log_line.emit`，适合实时日志 `QTextEdit.append`。

---

## 7. 常用控件与本项目用法

### 7.1 QMainWindow

- `setCentralWidget(widget)`：中央区域只能有一个根 widget。
- `statusBar()`：底部状态栏，可 `addWidget` / `addPermanentWidget`。

### 7.2 QStackedWidget

多页容器，同一时间只显示一页：

```python
self.stack = QStackedWidget()
self.stack.addWidget(self.p_detect)
self.stack.setCurrentIndex(0)
```

### 7.3 QTabWidget

标签页，如「单张检测 / 批量检测」：

```python
tabs = QTabWidget()
tabs.addTab(single_widget, "单张检测")
tabs.addTab(batch_widget, "批量检测")
```

### 7.4 QTableWidget

二维表格，比 `QTableView` + Model 更简单，适合中等规模数据。

```python
table = QTableWidget(0, 4)   # 0 行 4 列
table.setHorizontalHeaderLabels(["文件名", "预测类别", ...])
table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
table.setEditTriggers(QAbstractItemView.NoEditTriggers)
table.setSelectionBehavior(QAbstractItemView.SelectRows)

# 插入一行
row = table.rowCount()
table.insertRow(row)
table.setItem(row, 0, QTableWidgetItem("example.jpg"))

# 某一格放自定义 widget（如缩略图、复选框）
table.setCellWidget(row, 0, ThumbnailLabel(path))
```

**本项目**：`ResultsPage` 表格含缩略图列、文件名、`QComboBox` 真值类别、`QCheckBox` 等。

### 7.5 输入类

| 控件 | 用途 | 读取值 |
|------|------|--------|
| `QLineEdit` | 单行文本 | `.text()` |
| `QSpinBox` / `QDoubleSpinBox` | 整数/浮点 | `.value()` |
| `QComboBox` | 下拉 | `.currentText()` |
| `QCheckBox` | 勾选 | `.isChecked()` |
| `QTextEdit` | 多行日志 | `.append()` / `.toPlainText()` |

只读路径框：`line_edit.setReadOnly(True)`，配合「浏览」按钮 + `QFileDialog`。

### 7.6 QProgressBar

```python
pbar.setMaximum(total)
pbar.setValue(current)
pbar.setVisible(True)
```

### 7.7 对话框

| API | 用途 | 本项目 |
|-----|------|--------|
| `QMessageBox.warning/information/critical` | 提示 | 未选图、密码错误 |
| `QInputDialog.getText` | 单行输入 | 管理员密码 |
| `QFileDialog.getOpenFileName` | 选文件 | 选图、选模型 |
| `QFileDialog.getExistingDirectory` | 选文件夹 | 批量检测目录 |
| `QFileDialog.getSaveFileName` | 保存路径 | 导出 CSV |

模态对话框会阻塞直到用户关闭；`QMessageBox` 静态方法即属此类。

---

## 8. 样式：QSS（Qt Style Sheets）

全局样式定义在 `app.py` 的 `STYLE` 字符串，启动时：

```python
app.setStyleSheet(STYLE)
```

语法类似 CSS：

```css
QPushButton.primary {
    background: #1976D2;
    color: white;
    border-radius: 5px;
}
QPushButton#nav[sel=true] {
    border-left: 3px solid #42A5F5;
}
```

**匹配方式**

1. **类型选择器**：`QTableWidget { ... }`
2. **类属性**：`QPushButton.primary` ← `btn.setProperty("class", "primary")`
3. **objectName**：`QWidget#sidebar` ← `widget.setObjectName("sidebar")`
4. **伪状态**：`:hover`, `:disabled`, `:focus`

**动态切换选中导航**

```python
btn.setProperty("sel", str(i == idx).lower())
btn.style().unpolish(btn)   # 刷新样式
btn.style().polish(btn)
```

单控件也可 `widget.setStyleSheet("color:red;")` 覆盖局部（如 `AdminLockBar` 锁定/解锁配色）。

---

## 9. 自定义控件与事件重写

继承现有控件，重写 Qt 事件方法，或用 `pyqtSignal` 向外通知。

### 9.1 ImageDropZone（拖放 + 点击选图）

| 方法 | 触发时机 |
|------|----------|
| `mousePressEvent` | 单击 → `QFileDialog.getOpenFileName` |
| `dragEnterEvent` | 拖入时判断是否含 URL |
| `dropEvent` | 放下文件 → `_load(path)` |
| `setAcceptDrops(True)` | 必须开启才接收拖放 |

显示图片：`QPixmap(path).scaled(..., Qt.KeepAspectRatio, Qt.SmoothTransformation)`。

### 9.2 ThumbnailLabel / HoverFilenameLabel

| 方法 | 行为 |
|------|------|
| `enterEvent` | 鼠标进入 → `preview_hover.emit(path)` |
| `leaveEvent` | 离开 → `preview_leave.emit()` |
| `mousePressEvent` | 左键 → 打开原图或 `pin_requested` |

`setCursor(Qt.PointingHandCursor)` 显示手型光标。

### 9.3 SidePreviewController + QTimer

非可视 `QObject`，协调「悬停临时预览」与「单击固定」：

- `QTimer.setSingleShot(True)` + 300ms：鼠标离开后延迟隐藏，避免闪烁。
- 若已 `pin`，超时后恢复固定图而非清空。

这是 **控制器模式**：视图 `ImagePreviewSidePanel` 只负责显示，逻辑在 `SidePreviewController`。

---

## 10. 图像与系统服务

```python
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices

px = QPixmap(path)
scaled = px.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
label.setPixmap(scaled)

# 用系统默认程序打开文件
QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
```

---

## 11. 只读与权限：AdminLockBar

「模型再训练」「设置」默认只读：

1. `RetrainPage` / `SettingsPage` 顶部 `AdminLockBar`，点击「输入密码解锁」→ `unlock_requested`。
2. 页面转发为 `admin_unlock_requested`，由 `MainWindow._try_unlock_admin_pages` 接收。
3. 主窗口调用 `QInputDialog` 验证 `ADMIN_PAGE_PASSWORD`；通过后 `AppState.admin_unlocked = True`，启用表单；可「重新锁定」。

这种模式适合**同一套 UI、分角色操作**，无需做完整登录系统。

---

## 12. 开发调试与扩展建议

### 12.1 新增一个功能页

1. 新建 `class MyPage(QWidget)`，在 `__init__` 里 `_build()` 布局。
2. `self.stack.addWidget(self.p_my)`。
3. 侧栏增加 `QPushButton`，`connect` 到 `_switch_page` 新索引。
4. 若需与检测联动，定义 `pyqtSignal` 并在 `MainWindow` 里连接。

### 12.2 修改样式

优先改全局 `STYLE`；仅某一状态改色时用局部 `setStyleSheet`。

### 12.3 常见问题

| 现象 | 可能原因 |
|------|----------|
| 界面卡死 | 主线程做了推理/训练，改 QThread |
| 线程里改 UI 崩溃 | 应用 signal 回主线程更新 |
| 样式不生效 | 未 `setProperty` / `objectName`，或未 `polish` |
| 中文乱码 | 检查 `QFont`、文件 UTF-8、CSV 编码 |
| 打包后无界面 | 确认 `main()` 已调用 `multiprocessing.freeze_support()`（本项目已实现） |

### 12.4 学习资源

- [Qt for Python 文档](https://doc.qt.io/qtforpython/)（PySide6，API 与 PyQt5 高度相似）
- [Qt Widgets 模块索引](https://doc.qt.io/qt-5/qtwidgets-index.html)
- 本项目对照阅读顺序建议：  
  `main()` → `MainWindow` → `DiamondDetectPage` + `SahiPipelineWorker` → `DetectionPage` → `InferenceWorker` → `ResultsPage` → `SettingsPage`（Tab）

---

## 13. 源码索引（`app.py`，按符号）

| 符号 / 区域 | 内容 |
|-------------|------|
| 模块头、`DEPLOY_ONNX_ONLY`、`NAV_*` | 机台开关、导航、SAHI 依赖检查 |
| `_DEFAULT_CFG_FULL` / `_load_cfg` / `_save_cfg` | 默认配置与 `app_config.json` |
| `STYLE` | 全局 QSS |
| `AppState`、`AdminLockBar` | 共享状态、管理员解锁条 |
| `InferenceWorker`、`TrainWorker`、`SahiPipelineWorker` | 后台线程 |
| `ImageDropZone`、缩略图、侧栏预览 | 通用 UI 组件 |
| `DetectionPage` | 单张 / 批量缺陷检测 |
| `ResultsPage` | 结果管理 |
| `CorrectionPage` | 误分类修正 |
| `RetrainPage` | 模型再训练（机台只读） |
| `SettingsPage` | 分类配置 + 切片推理配置 Tab（阈值见 `class_thresholds.json`） |
| `DiamondDetectPage` | 钻石检测分类（仅开发版） |
| `MainWindow`、`main()` | 窗口组装与入口（含 `freeze_support`） |

在 IDE 中用「转到定义」定位上述类/函数即可，勿依赖易漂移的行号。

---

## 14. 与本项目业务的关系

Qt 层只负责：**展示、交互、线程调度、配置持久化**。  
分类与训练逻辑在独立模块中，与 UI 解耦：

| 模块 | 用途 |
|------|------|
| `inference_engine.py` | 开发版推理（PyTorch GPU 优先，ONNX 回退；`predict` / `predict_batch`） |
| `inference_engine_onnx.py` | 机台版推理（ONNX Runtime；打包 exe 使用） |
| `inference_common.py` | 元数据；**`run_batch_predict`** 批量循环；**`logits_row_to_result`**；**`build_result_dict`** 阈值决策 |
| `sahi_detector.py` | SAHI 切片检测 + 裁剪 + 调用 `predict_batch` 分类（仅开发版 `app.py`） |
| `train.py` | 训练、微调、ONNX 导出、阈值校准 |
| `app_deploy.py` | 机台版 GUI 入口（无 PyTorch / SAHI） |

### 14.1 检测结果 dict（UI 与引擎之间）

`InferenceWorker` / 单张检测写入 `AppState.results` 的每条记录主要字段：

| 字段 | 说明 |
|------|------|
| `path` | 图像路径（写入前规范化，同路径 upsert） |
| `class` / `confidence` | 预测类别及该类别概率（经逐类阈值） |
| `max_class` / `max_confidence` | argmax 类别（可与预测不同） |
| `all_scores` | 各类别概率，供单张页得分条 |
| `true_class` / `flagged` | 结果管理、误分类修正用 |

修改识别效果 → 动 `inference_common.py` 与训练；修改操作流程与界面 → 动 `app.py` 各 Page。  
机台打包与部署见 [打包部署说明.md](打包部署说明.md)。

如有新需求（例如增加统计图表页、接入摄像头），可先在本说明中对照「布局 + 信号 + 线程」三要素设计，再在 `app.py` 中按现有 Page 模式实现。
