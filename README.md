# 图像缺陷分类系统  使用说明

## 项目文件结构

```
project/
├── analyze_image_sizes.py   # 阶段一：图像尺寸统计分析
├── train.py                 # 阶段二：模型训练管线
├── inference_engine.py      # 阶段三：推理引擎（被 app.py 调用）
├── app.py                   # 阶段三：PyQt5 部署应用（主入口）
├── requirements.txt         # 依赖清单
│
├── data/                    # 原始缺陷图像数据集（需用户提供）
│   ├── class_1/
│   ├── class_2/
│   ├── class_3/
│   ├── class_4/
│   └── class_5/
│
├── outputs/                 # 分析结果自动生成
│   ├── image_size_distribution.png
│   └── image_size_stats.csv
│
├── checkpoints/             # 训练结果自动生成
│   ├── best_model.pt        # PyTorch 模型（GPU 推理）
│   ├── model.onnx           # ONNX 模型（CPU 推理）
│   ├── class_map.json       # 类别映射
│   ├── train_config.json    # 训练配置
│   ├── training_curves.png
│   └── confusion_matrix.png
│
├── corrections/             # 误分类修正数据（应用自动创建）
│   ├── class_1/
│   └── ...
│
└── app_config.json          # 应用配置（首次运行后生成）
```

---

## 快速开始

### Step 1  安装依赖

```bash
# CPU 机台
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install Pillow numpy pandas matplotlib scikit-learn PyQt5 onnx onnxruntime

# GPU 机台（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install Pillow numpy pandas matplotlib scikit-learn PyQt5 onnx onnxruntime
```

### Step 2  分析图像尺寸，确定统一分辨率

```bash
python analyze_image_sizes.py --data_dir data
```

查看控制台输出的"推荐统一输入分辨率"，记录该值（如 224 × 224）。

### Step 3  训练模型

```bash
# 使用推荐分辨率训练（GPU 机台）
python train.py --data_dir data --img_size 224 --batch_size 16

# CPU 机台（减小 batch_size）
python train.py --data_dir data --img_size 224 --batch_size 8 --num_workers 0
```

训练结束后，`checkpoints/` 下会生成 `.pt` 和 `.onnx` 两个模型文件。

### Step 4  启动部署应用

```bash
python app.py
```

首次启动会自动尝试加载 `checkpoints/best_model.pt`；若未找到，请点击左侧「设置」页面手动配置路径。

---

## 应用功能说明

| 页面 | 功能 |
|------|------|
| 🔍 缺陷检测 | 单张图像拖入/点击检测，或整文件夹批量检测 + 进度显示 |
| 📋 结果管理 | 查看全部检测结果（缩略图、置信度），按类别/置信度筛选，标记待修正，导出 CSV |
| ✏️ 误分类修正 | 对标记项逐一指定正确类别，点击「保存全部修正」归档到 corrections/ |
| 🔄 模型再训练 | 数据统计 + 参数配置 + 实时训练日志，训练完成后一键热更新模型 |
| ⚙️ 设置 | 配置模型路径、数据目录、GPU/CPU 选择 |

---

## 主动学习闭环流程

```
运行检测  →  结果管理页标记误分类
         →  误分类修正页保存到 corrections/
         →  再训练页刷新统计 → 开始训练（自动合并 data/ + corrections/）
         →  训练完成 → 点击「应用新模型」热更新 → 继续检测
```

---

## 常见问题

**Q: 无 GPU 时推理速度慢？**  
A: 安装 `onnxruntime` 后，应用会自动切换到 ONNX 后端，CPU 推理速度比 PyTorch CPU 快约 2–4 倍。

**Q: 再训练时 Windows 报多进程错误？**  
A: 应用已自动设置 `--num_workers 0`，无需额外操作。

**Q: 如何在下一轮训练中只用修正数据而不覆盖原始数据？**  
A: `corrections/` 目录是独立的，每次再训练时两个目录合并计算；原始 `data/` 中的文件不会被修改。

**Q: 模型导出 ONNX 失败？**  
A: 确认已安装 `pip install onnx onnxruntime`。ONNX 导出失败时仅影响 CPU 加速路径，`.pt` 模型仍可正常使用。
