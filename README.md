# 图像缺陷分类系统 — 使用说明

基于 EfficientNet-B0 的工业缺陷五分类系统，包含数据分析、模型训练、PyQt5 部署应用与主动学习闭环。

---

## 项目文件结构

```
defects_classify/
├── analyze_image_sizes.py   # 阶段一：图像尺寸统计分析，推荐 img_size
├── train.py                 # 阶段二：训练管线（四层不均衡策略 + Pipeline）
├── inference_engine.py      # 推理引擎（PyTorch GPU / ONNX CPU 双后端）
├── app.py                   # PyQt5 部署应用（主入口）
├── requirements.txt         # Python 依赖
├── README.md
├── scripts/
│   └── build_win.bat        # Windows PyInstaller 一键打包
│
├── data/                    # 训练数据集（用户提供，不纳入 Git）
│   ├── 局部破损/
│   ├── 断钻/
│   └── ...
│
├── checkpoints/             # 训练产物（建议纳入 Git 或单独分发）
│   ├── best_model.pt        # PyTorch 权重（GPU 推理 / 再训练）
│   ├── model.onnx           # ONNX 模型（CPU 加速推理）
│   ├── class_map.json       # 类别名 ↔ index
│   ├── class_thresholds.json# 逐类最优阈值（提升少数类召回）
│   ├── train_config.json    # 训练超参与指标
│   ├── dataset_split_seed42.json  # train/val 划分缓存（增量训练复用）
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── outputs/                 # analyze_image_sizes.py 输出
├── corrections/             # 应用误分类修正归档（主动学习）
└── app_config.json          # 应用运行时配置（首次启动后生成）
```

---

## 环境安装

推荐使用 Conda 独立环境（示例：`cv-yolo`）。

```bash
# GPU（CUDA 11.8 示例，按本机 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 机台
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 其余依赖
pip install -r requirements.txt
```

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 完整工作流

### Step 1  分析图像尺寸

```bash
python analyze_image_sizes.py --data_dir data
```

查看控制台「推荐统一输入分辨率」。本项目实测 **128×128** 即可（小图、已离线增强）。

### Step 2  训练模型

```bash
# 推荐：GPU + 默认参数（128 分辨率、Focal Loss、类别权重）
python train.py --data_dir data --img_size 128

# 或使用绝对路径（任意工作目录均可）
"D:\Software\MiniAnaconda\envs\cv-yolo\python.exe" "D:\...\defects_classify\train.py"
```

训练完成后检查 `checkpoints/`：

| 文件 | 用途 |
|------|------|
| `best_model.pt` | GPU 推理、增量训练 |
| `model.onnx` | 无 GPU 机台 CPU 加速 |
| `class_thresholds.json` | 推理时逐类阈值（自动被 inference_engine 读取） |
| `confusion_matrix.png` | 查看哪两类易混淆 |

### Step 3  启动应用

```bash
python app.py
```

首次启动会尝试加载 `checkpoints/best_model.pt`；可在「设置」页修改模型路径、置信度阈值、GPU 开关。

---

## 训练参数建议

### 四层不均衡学习（train.py 内置）

| 层级 | 机制 | 默认 | 说明 |
|------|------|------|------|
| 第一层 | WeightedRandomSampler | 离线增强时关闭 | 原始未增强数据用 `--no_pre_augmented` 开启 |
| 第二层 | 逆频类别权重 | 开启 | `--no_class_weight` 可关 |
| 第三层 | Focal Loss (γ=2) | 开启 | `--no_focal_loss` 改用加权 CE |
| 第四层 | Macro-F1 选模 + 阈值校准 | 开启 | 少数类 F1 低会拉低整体，比 Accuracy 更可靠 |

### 按场景选择参数

#### 场景 A：首次训练（当前 data/ 已离线增强）

```bash
python train.py --data_dir data --img_size 128 --batch_size 32
```

- 保持默认 `--pre_augmented`（轻量翻转，避免二次强增强）
- 关注日志中 **Macro-F1** 与 `★` 标记的最优 epoch
- 少数类「局部破损」样本 <10% 时，默认四层策略即可

#### 场景 B：原始图、未做离线增强

```bash
python train.py --no_pre_augmented --img_size 128 --batch_size 16
```

- 启用 WeightedRandomSampler + 完整在线增强
- 可适当增加 `--epochs_phase2 30`

#### 场景 C：误分类修正后再训练（主动学习）

```bash
python train.py --finetune --extra_data_dirs corrections --lr_phase2 5e-5
```

- `--finetune`：自动加载 `best_model.pt`，**跳过阶段一**
- 合并 `data/` + `corrections/`，划分缓存自动复用（除非 `--no_reuse_split`）
- 学习率宜低于首次训练，防止灾难性遗忘

#### 场景 D：仅补跑阈值校准 / ONNX（训练已完成）

```bash
python train.py --postprocess_only
```

#### 场景 E：CPU 训练（无 GPU）

```bash
python train.py --batch_size 8 --num_workers 0
```

### 关键参数速查

| 参数 | 默认 | 建议 |
|------|------|------|
| `--img_size` | 128 | 与 analyze 推荐一致；改尺寸需重训 |
| `--batch_size` | 32 | GPU 显存不足时 16/8 |
| `--epochs_phase1` | 5 | 增量微调时设 0 或用 `--finetune` |
| `--epochs_phase2` | 20 | 数据增多可 25~30 |
| `--lr_phase2` | 1e-4 | 微调 corrections 时用 5e-5 ~ 1e-5 |
| `--patience` | 8 | 验证 F1 连续 8 轮不升则早停 |
| `--val_ratio` | 0.15 | 样本极少时可 0.1 |
| `--seed` | 42 | 固定可复现；改 seed 需 `--no_reuse_split` |

---

## 识别与部署参数建议

### 推理后端选择

| 环境 | 推荐后端 | 配置 |
|------|----------|------|
| 有 NVIDIA GPU | PyTorch + `.pt` | 设置页「使用 GPU」开启 |
| 纯 CPU 产线机 | ONNX + `onnxruntime` | 关闭 GPU，确保 `model.onnx` 存在 |
| 两者都有 | 自动：GPU 优先，否则 ONNX | 默认行为 |

### 置信度阈值（应用「设置」页 `conf_threshold`）

- **0.5（默认）**：平衡模式，适合整体准确率优先
- **0.6 ~ 0.7**：减少误报，适合「宁可漏检、不可错判」的质检场景
- **0.3 ~ 0.4**：提高召回，配合 `class_thresholds.json` 对少数类更友好

### 关注混淆矩阵

训练后查看 `checkpoints/confusion_matrix.png`：

- 若「局部破损」常被误判为「断钻」→ 增加局部破损样本或 corrections 再 `--finetune`
- 若某类 Precision 高、Recall 低 → 降低该类阈值（见 `class_thresholds.json` 中对应值）

### 产线识别流程建议

```
采集图像 → app 批量检测 → 结果管理筛选低置信度
        → 误分类修正归档 corrections/
        → --finetune 再训练 → 应用新模型热更新
```

---

## 应用功能

| 页面 | 功能 |
|------|------|
| 缺陷检测 | 单张/文件夹批量检测，进度条 |
| 结果管理 | 缩略图、置信度、筛选、导出 CSV |
| 误分类修正 | 修正后保存至 `corrections/类别名/` |
| 模型再训练 | 调用 `train.py`，日志实时显示，完成后热更新 |
| 设置 | 模型路径、数据目录、GPU、置信度阈值 |

---

## 应用打包教程（Windows）

将 PyQt5 应用打包为独立 `.exe`，便于在无 Python 环境的产线机台部署。

### 方案选择

| 方案 | 体积 | 适用 | 说明 |
|------|------|------|------|
| **A. CPU 轻量包** | ~200–400 MB | 无 GPU 检测工位 | 打包 ONNX + onnxruntime，不含 CUDA |
| **B. GPU 完整包** | ~2 GB+ | 带 NVIDIA GPU 工位 | 含 torch+cuda，体积大，一般不推荐 PyInstaller |

**推荐方案 A**：检测端用 ONNX CPU；训练仍在开发机完成。

### 前置准备

```bash
conda activate cv-yolo
pip install pyinstaller

# 确保已有训练产物
python train.py --postprocess_only
```

确认以下文件存在：

```
checkpoints/best_model.pt
checkpoints/model.onnx
checkpoints/class_map.json
checkpoints/class_thresholds.json
```

### Step 1  创建打包入口（可选）

默认直接打包 `app.py` 即可。若需自定义图标，准备 `icon.ico` 放在项目根。

### Step 2  执行 PyInstaller

**方式一（推荐）**：使用项目自带脚本

```powershell
cd "D:\工作记录\3 人工智能相关课题\张旭\defects_classify"
conda activate cv-yolo

# 默认：窗口模式 + 内置 checkpoints/
scripts\build_win.bat

# 不打包模型（exe 更小，checkpoints 单独拷贝）
scripts\build_win.bat slim

# 保留控制台窗口（排查闪退）
scripts\build_win.bat console
```

**方式二**：手动命令

在项目根目录执行（PowerShell）：

```powershell
cd "D:\工作记录\3 人工智能相关课题\张旭\defects_classify"

pyinstaller --noconfirm --clean `
  --name "缺陷分类系统" `
  --windowed `
  --add-data "checkpoints;checkpoints" `
  --hidden-import "PyQt5.sip" `
  --hidden-import "onnxruntime" `
  --hidden-import "PIL" `
  --collect-all onnxruntime `
  app.py
```

说明：

- `--windowed`：不显示黑色控制台窗口
- `--add-data "checkpoints;checkpoints"`：Windows 用分号；将模型与配置打入包内
- `--collect-all onnxruntime`：收集 ONNX Runtime 原生 DLL

### Step 3  获取产物

```
dist/缺陷分类系统/
  缺陷分类系统.exe
  checkpoints/          ← 内置模型（可整体拷贝到其他机器）
  ... 依赖 dll ...
```

将整个 `dist/缺陷分类系统/` 文件夹复制到目标机器，双击 `缺陷分类系统.exe` 运行。

### Step 4  打包后配置

首次运行会在 exe 同目录生成 `app_config.json`。若需更新模型：

1. 在开发机重新训练，覆盖 `checkpoints/` 下文件
2. 将新 `checkpoints/` 复制到 exe 目录（或仅替换 `best_model.pt`、`model.onnx`、`class_thresholds.json`）
3. 重启应用或在设置页重新加载

### Step 5  常见问题排查

| 现象 | 处理 |
|------|------|
| 双击 exe 闪退 | 去掉 `--windowed` 重新打包，在 cmd 中运行 exe 查看报错 |
| 找不到模型 | 确认 `checkpoints` 与 exe 同级；设置页检查路径 |
| 中文乱码 | 确保系统已安装「微软雅黑」等字体 |
| 体积过大 | 使用 CPU 版 torch 或方案 A 仅 onnxruntime |
| 杀毒软件拦截 | 添加白名单；或对 `dist` 目录签名 |

### 进阶：分离模型包（减小 exe 更新体积）

打包时不内置 `checkpoints`，改为产线机手动放置：

```powershell
pyinstaller --noconfirm --windowed --name "缺陷分类系统" app.py
```

部署结构：

```
产线部署/
  缺陷分类系统.exe
  checkpoints/          ← 单独拷贝，更新模型时只换此目录
  app_config.json
```

---

## 主动学习闭环

```
运行检测 → 结果管理标记误分类 → 误分类修正保存到 corrections/
        → 再训练页 / 命令行 --finetune → 应用新模型 → 继续检测
```

应用内再训练会自动附加 `--extra_data_dirs corrections --num_workers 0`。

---

## 常见问题

**Q: 命令行报 `No such file or directory`（data）？**  
A: 使用 `train.py` 的绝对路径启动，或先 `cd` 到项目根。路径均相对 `train.py` 所在目录解析。

**Q: 训练结束报 CUDA/CPU tensor 不一致？**  
A: 已修复：阈值校准在 ONNX 导出之前执行。若仍遇到，运行 `python train.py --postprocess_only`。

**Q: Accuracy 很高但少数类仍漏检？**  
A: 看 Macro-F1 与混淆矩阵；使用 `class_thresholds.json`；增加少数类 corrections 后 `--finetune`。

**Q: ONNX 导出警告 opset 版本？**  
A: 可忽略；当前使用 opset 18，ORT 验证通过即可。

**Q: Windows 多进程 DataLoader 报错？**  
A: 训练默认 `--num_workers 0`，应用内再训练已强制该参数。

**Q: 如何只更新模型不重装 exe？**  
A: 仅替换 `checkpoints/` 下 `best_model.pt`、`model.onnx`、`class_thresholds.json`，应用内点击重新加载。

---

## 命令速查

```bash
# 首次训练
python train.py --data_dir data --img_size 128

# 增量微调（含 corrections）
python train.py --finetune --extra_data_dirs corrections

# 后处理补跑
python train.py --postprocess_only

# 启动 GUI
python app.py

# Windows 打包 exe（需先 conda activate cv-yolo）
scripts\build_win.bat
```
