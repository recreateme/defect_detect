# 钻石缺陷图像分类系统

基于 **EfficientNet-B0** 的工业钻石缺陷 **五分类** 系统：数据分析、训练、PyQt5 检测应用与主动学习闭环。

| 类别 | 说明 |
|------|------|
| 局部破损 | 钻石表面局部损伤 |
| 断钻 | 钻具断裂相关缺陷 |
| 棱边朝上 | 棱边朝向检测 |
| 点朝上 | 顶点朝向检测 |
| 面朝上 | 底面/面朝上检测 |

当前模型（见 `checkpoints/train_config.json`）：输入 **128×128**，验证 Macro-F1 ≈ **0.90**，Accuracy ≈ **95%**。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| **本文 (README.md)** | 环境、训练、应用功能、工作流 |
| [打包部署说明.md](打包部署说明.md) | 机台 exe 打包、GPU 部署、故障排查 |
| [QT应用开发说明.md](QT应用开发说明.md) | PyQt5 界面结构、信号槽、线程（改 UI 时阅读） |

---

## 项目结构

```
defects_classify/
├── app.py                    # 开发版 GUI 入口（PyTorch + ONNX + SAHI）
├── app_deploy.py             # 机台版入口（仅 ONNX，PyInstaller 打包用）
├── train.py                  # 训练 / 微调 / ONNX 导出
├── inference_engine.py       # 开发版推理（PyTorch GPU 优先，支持批量）
├── inference_engine_onnx.py  # 机台版推理（ONNX Runtime GPU/CPU）
├── inference_common.py       # 推理公共逻辑（批量循环、阈值、softmax，开发/机台共用）
├── sahi_detector.py          # SAHI 大图切片检测 + 缺陷分类流水线（仅开发版）
├── app_paths.py              # 路径解析（开发 / 打包 exe 通用）
├── analyze_image_sizes.py    # 图像尺寸统计，推荐 img_size
├── requirements.txt          # 开发 / 训练依赖（含 ultralytics、opencv-python）
├── requirements-deploy.txt   # 机台打包专用依赖（无 PyTorch / SAHI）
├── 缺陷分类系统.spec           # PyInstaller 规格参考（打包请以 build_deploy.py 为准）
├── scripts/
│   ├── build_deploy.py       # 机台打包主脚本（推荐）
│   ├── build_deploy.bat      # 上述脚本 Windows 快捷方式
│   ├── setup_deploy_env.ps1  # 创建 defects-deploy Conda 环境
│   ├── verify_deploy.py      # 打包前 ORT + 模型验收
│   └── verify_frozen_sim.py  # 打包后 exe 验收（DEFECTS_VERIFY=1）
├── pyinstaller_hooks/
│   └── rthook_ort_dll.py     # PyInstaller runtime hook（ORT / CUDA DLL）
├── checkpoints/              # 模型与配置（训练产出，纳入版本库）
├── data/                     # 训练数据（按类别分子文件夹，git 忽略）
├── corrections/              # 误分类修正归档（git 忽略）
├── outputs/                  # analyze_image_sizes 等工具输出（git 忽略）
├── sahi_output/              # 钻石检测分类默认输出（git 忽略）
├── build_staging/            # 打包临时目录（checkpoints / cuda_deps，git 忽略）
└── dist/缺陷分类系统/        # 打包输出（git 忽略）
```

---

## 环境

### 开发 / 训练（`cv-yolo` 或自建环境）

```powershell
conda activate cv-yolo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

验证 GPU：`python -c "import torch; print(torch.cuda.is_available())"`

> **RTX 50 系列（sm_120）**：若 PyTorch 版本不支持当前 GPU 算力，SAHI 切片推理会在 `auto` 模式下自动回退 CPU，或安装支持 sm_120 的 PyTorch（CUDA 12.8+ / nightly）。

### 机台打包（`defects-deploy`，与训练环境隔离）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_deploy_env.ps1
conda activate defects-deploy
```

详见 [打包部署说明.md](打包部署说明.md)。

---

## 工作流

### 1. 分析图像尺寸

```powershell
python analyze_image_sizes.py --data_dir data
```

默认输出到 `outputs/`（`image_size_stats.csv`、`image_size_distribution.png`）。  
本项目实测 **128×128** 即可。

### 2. 训练

```powershell
python train.py --data_dir data --img_size 128
```

| checkpoints 文件 | 用途 |
|------------------|------|
| `best_model.pt` | 开发版 GPU 推理、微调 |
| `model.onnx` / `model.onnx.data` | 机台 ONNX 推理 |
| `class_map.json` | 类别名与索引 |
| `train_config.json` | **img_size**、Macro-F1 等（推理必读） |
| `class_thresholds.json` | 逐类置信度阈值 |
| `dataset_split_seed42.json` | 数据集划分记录 |
| `training_curves.png` | 训练曲线 |

仅补跑 ONNX / 阈值：`python train.py --postprocess_only`

### 3. 启动应用

```powershell
# 开发机（完整功能，窗口标题无「机台版」）
python app.py

# 机台模式本地调试（仅 ONNX，标题显示「· 机台版」）
python app_deploy.py
```

首次运行会在 exe/项目根生成 `app_config.json`（模型路径、GPU 开关等）。

### 4. 机台打包

```powershell
conda activate defects-deploy
python scripts/build_deploy.py
```

产物：`dist/缺陷分类系统/`，**整包**复制到机台。说明见 [打包部署说明.md](打包部署说明.md)。

---

## 推理架构

开发与机台**共用** `inference_common.py`，各自引擎只负责后端差异，避免单张/批量结果不一致：

```
app.py / app_deploy.py
    └─ InferenceEngine（按环境二选一）
         ├─ inference_engine.py      开发：PyTorch GPU 优先，无 GPU 时 ONNX CPU 回退
         └─ inference_engine_onnx.py  机台：ONNX Runtime GPU/CPU（打包 exe 专用）
              └─ inference_common.py   元数据、run_batch_predict、logits→结果、阈值决策
```

| 模块 | 开发机 | 机台 exe |
|------|--------|----------|
| `inference_engine.py` | ✅（含 PyTorch） | ❌ 不打包 |
| `inference_engine_onnx.py` | 仅 `app_deploy.py` 调试 | ✅ |
| `inference_common.py` | ✅ | ✅（纯 NumPy，体积极小） |

修改批量推理、置信度、阈值逻辑时，**优先改 `inference_common.py`**；仅 PyTorch 加载或 ORT Provider 差异才改对应 engine。

---

## 推理说明

| 场景 | 引擎 | 批量检测 |
|------|------|----------|
| 开发机 + NVIDIA GPU | `inference_engine.py` → PyTorch | `predict_batch`，默认 batch=32 |
| 开发机无 GPU | ONNX CPU 回退 | 同上 |
| 机台 exe | `inference_engine_onnx.py` | 同上，GPU 时 batch=32 |

批量检测由 `InferenceWorker` 调用 `predict_batch`（内部走 `run_batch_predict`），比逐张 `predict` 更快（尤其 GPU）。  
单张与批量经同一套 logits → softmax → 阈值决策，**结果管理**与**单张检测**显示的类别、置信度一致。

**img_size** 从 `train_config.json` 读取，须与训练一致（当前为 **128**）。

### 置信度含义

| 字段 | 含义 |
|------|------|
| `confidence` | **预测类别**的概率（经 `class_thresholds.json` 逐类阈值决策后） |
| `max_confidence` / `max_class` | argmax 最高分类别（可能与预测类别不同） |

启用逐类阈值时，单张页若预测类别非最高分，会提示「阈值决策；最高得分：…」。结果管理表格中的置信度始终为预测类别的概率。

### 结果去重

同一张图重复检测时，按规范化路径 **更新** 已有记录（保留修正类别、待修正标记），不会追加重复行。

---

## 训练参数（摘要）

四层不均衡策略：类别权重 → Focal Loss → Macro-F1 选模 → 阈值校准。完整参数见 `train.py --help`。

| 场景 | 命令 |
|------|------|
| 首次训练 | `python train.py --data_dir data --img_size 128` |
| 误分类再训练 | `python train.py --finetune --extra_data_dirs corrections` |
| 后处理 / ONNX | `python train.py --postprocess_only` |

---

## 应用功能

| 页面 | 功能 | 开发版 | 机台版 |
|------|------|--------|--------|
| 钻石检测分类 | 5120×5120 大图 SAHI 切片检测 + 缺陷分类、批量推理 | ✅ | ❌ |
| 缺陷检测 | 单张 / 文件夹批量检测、按类别导出 | ✅ | ✅ |
| 结果管理 | 筛选、导出 CSV；与单张检测共用同一结果结构 | ✅ | ✅ |
| 误分类修正 | 归档到 `corrections/`（按类别分子文件夹） | ✅ | ✅ |
| 模型再训练 | 子进程训练；机台版默认只读 | ✅ | 只读 |
| 设置 | **分类配置** Tab：模型路径、GPU；**切片推理配置** Tab：YOLO、SAHI 参数 | ✅ | 仅分类配置 |

「模型再训练」「设置」默认只读，输入管理员密码后可编辑（密码定义于 `app.py` 中 `ADMIN_PAGE_PASSWORD`）。

### 钻石检测分类（开发版）

1. 在「设置 → 切片推理配置」配置 YOLO 权重（`.pt`）、切片大小、重叠率、设备等
2. 在「设置 → 分类配置」加载缺陷分类模型
3. 在「钻石检测分类」选择输入图像（多选或文件夹）与结果保存目录
4. 点击「开始处理」→ 每张图输出：裁剪图、两张全分辨率可视化图（检测框 / 分类着色）、JSON 统计；完成后可打开保存目录

默认保存目录：`sahi_output/`（可在设置或页面中修改，写入 `app_config.json` 的 `sahi_output_dir`）。

---

## 主动学习闭环

```
检测 → 误分类修正 → corrections/ → 开发机 --finetune → 更新 checkpoints → 机台替换模型
```

---

## 常见问题

**Q: 单张检测与结果管理置信度不一致？**  
A: 请使用最新代码（`inference_common.run_batch_predict` 统一单张/批量）。若预测类别因阈值与最高分不同，属正常；看 `confidence`（预测类）而非得分条第一名。

**Q: 开发机批量检测报 `requires grad`？**  
A: 确保 `inference_engine.py` 的 `predict_batch` 带 `@torch.inference_mode()`（机台 exe 不受影响，不含 PyTorch）。

**Q: 机台检测报输入尺寸 224 vs 128？**  
A: 确保 `checkpoints/train_config.json` 含正确 `img_size`，并重新加载模型或使用最新 exe。

**Q: 只更新模型不重装 exe？**  
A: 覆盖机台 `checkpoints/` 下 `model.onnx`、`model.onnx.data`、`class_thresholds.json`、`train_config.json`、`class_map.json` 等，重启或在设置页重新加载。

**Q: 训练路径找不到 data？**  
A: `train.py` 会自动解析项目根目录，也可显式指定 `--data_dir`；确保 `data/` 下按类别分子文件夹。

**Q: 直接用 `缺陷分类系统.spec` 打包？**  
A: 不推荐。请使用 `python scripts/build_deploy.py`，它会动态收集 CUDA DLL、复制 checkpoints 并执行打包后补丁。

**Q: 机台 exe 能用钻石检测分类吗？**  
A: 不能。SAHI / YOLO 依赖 ultralytics 与 PyTorch，仅 `python app.py` 开发版提供；机台 exe 仅含 ONNX 缺陷分类。

**Q: SAHI 报 CUDA error 或 GPU 不兼容？**  
A: 在「设置 → 切片推理配置」将设备设为 `auto` 或 `cpu`；新型号 GPU 需升级 PyTorch 后再用 `cuda:0`。

---

## 命令速查

```powershell
python analyze_image_sizes.py --data_dir data
python train.py --data_dir data --img_size 128
python train.py --finetune --extra_data_dirs corrections
python train.py --postprocess_only
python app.py
python app_deploy.py
conda activate defects-deploy
python scripts/build_deploy.py
python scripts/verify_deploy.py
python scripts/verify_frozen_sim.py
```
