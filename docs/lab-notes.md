# HumanCounter3 实验日志

> **目标:** 在一台配备 RTX 4070 笔记本电脑 GPU 的 Windows 11 机器上，使用 PyTorch (CUDA 11.8) 和 Ultralytics YOLOv8-seg 对单个或批量图像执行对象检测和分割。该过程应包括多类别计数，并输出带标注的图像、轮廓掩码和结构化的 JSON 摘要，为未来的 GUI 和 API 集成做准备。

---

## 目录
- [阶段 1: 环境设置与验证](#阶段-1-环境设置与验证)
  - [1.1 项目信息与约定](#11-项目信息与约定)
  - [1.2 硬件与驱动检查](#12-硬件与驱动检查)
  - [1.3 Python 深度学习环境验证](#13-python-深度学习环境验证)
  - [1.4 Ultralytics YOLO 环境验证](#14-ultralytics-yolo-环境验证)
- [阶段 2: 推理流程开发与迭代](#阶段-2-推理流程开发与迭代)
  - [2.1 问题排查: `runs` 目录位置不符](#21-问题排查-runs-目录位置不符)
  - [2.2 文件组织策略](#22-文件组织策略)
  - [2.3 推理流程 v0.1: 建立自定义输出协议](#23-推理流程-v01-建立自定义输出协议)
  - [2.4 推理流程 v0.2: 增加计数、耗时与批量处理](#24-推理流程-v02-增加计数耗时与批量处理)
- [阶段 3: 总结与未来规划](#阶段-3-总结与未来规划)
  - [3.1 项目结论](#31-项目结论)
  - [3.2 未来计划](#32-未来计划)

---

## 阶段 1: 环境设置与验证

### 1.1 项目信息与约定

- **项目根目录:** `D:\Coding\PycharmCode\Counter`
- **Conda 环境:**
  - **名称:** `HumanCounter3`
  - **Python:** 3.11.14
- **关键目录结构:**
  - `artifacts/weights/`: 存储模型权重 (*.pt)。
  - `assets/examples/`: 包含用于测试的示例图像。
  - `result/YYYY-MM-DD_HHMMSS/`: 存放每次推理运行的输出。
    - `annotated/`: 带标注的图像。
    - `silhouette/`: 轮廓掩码图像。
    - `summary.json`: 结构化结果。
  - `artifacts/runs/`: Ultralytics CLI 默认输出目录。

### 1.2 硬件与驱动检查

- **命令:** `nvidia-smi`
- **输出 (关键部分):**
  ```yaml
  NVIDIA-SMI 591.74                 Driver Version: 591.74         CUDA Version: 13.1
  GPU  0  NVIDIA GeForce RTX 4070 ...  WDDM  | 8188MiB
  ```
- **解读:** 驱动正常，`CUDA 13.1` 为驱动支持的最高版本。

### 1.3 Python 深度学习环境验证

- **脚本:** `check_gpu.py`
- **输出 (关键部分):**
  ```yaml
  torch: 2.7.1+cu118
  torch.version.cuda: 11.8
  cuda_available: True
  gpu: NVIDIA GeForce RTX 4070 Laptop GPU
  ```
- **结论:** PyTorch `cu118` 版本与 CUDA `11.8` 匹配，GPU 可用。

### 1.4 Ultralytics YOLO 环境验证

- **自检命令:** `yolo checks`
- **输出 (关键部分):**
  ```yaml
  Ultralytics 8.4.9  Python-3.11.14 torch-2.7.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
  CUDA 11.8
  ```
- **结论:** Ultralytics 成功识别 GPU 和 CUDA 环境，准备就绪。

---

## 阶段 2: 推理流程开发与迭代

### 2.1 问题排查: `runs` 目录位置不符

- **问题:** CLI 推理后，`runs/segment/predict` 目录未在项目内生成。
- **调查:** 使用 `yolo settings` 命令检查配置。
- **分析:** `runs_dir` 设置为相对路径 `.\\artifacts\\runs`，导致其基于 PyCharm 的父级工作目录创建。
- **解决方案:** 调整 PyCharm 终端的启动目录为项目根目录 (`$ProjectFileDir$`)。

### 2.2 文件组织策略

- **权重:** `artifacts/weights/yolov8n-seg.pt`
- **示例图像:** `assets/examples/bus.jpg`
- **目的:** 分离不同类型的文件，保持项目结构清晰，便于未来扩展。

### 2.3 推理流程 v0.1: 建立自定义输出协议

- **脚本:** `scripts/10_infer_to_result.py`
- **目标:** 实现统一的输出结构 `result/YYYY-MM-DD_HHMMSS/{annotated, silhouette, summary.json}`。
- **验证:**
  - **命令:** `python scripts/10_infer_to_result.py --source assets/examples/bus.jpg`
  - **结果:** 成功在 `result/` 目录下生成符合协议的输出。
- **`summary.json` 结构:**
  - `counts_by_class`: 各类别计数。
  - `detections[]`: 每个检测实例的详细信息 (类别、置信度、边界框)。

### 2.4 推理流程 v0.2: 增加计数、耗时与批量处理

- **新功能:**
  - 在标注图上叠加总数和分类计数文本。
  - 在 `summary.json` 中增加 `inference_ms` 字段记录推理耗时。
  - 支持对整个目录的图像进行批量处理。
  - 统一使用 Ultralytics 的官方调色板。
- **验证:**
  - **单图:** `python scripts/10_infer_to_result.py --source assets/examples/bus.jpg`
  - **批量:** `python scripts/10_infer_to_result.py --source assets/examples`
- **效果:**
  - 标注图左上角显示 `total:6  person:4  bus:1  skateboard:1`。
  - `summary.json` 中新增 `"inference_ms": 441.44` 等字段。

---

## 阶段 3: 总结与未来规划

### 3.1 项目结论

- **✅ GPU 环境就绪:** PyTorch 与 CUDA 11.8 正常工作。
- **✅ 推理流程可用:** YOLOv8-seg 可在 GPU 上执行推理。
- **✅ 输出协议确立:** 建立了标准化的 `result/` 输出结构。
- **✅ 功能迭代完成:** 脚本现已支持计数、耗时、批量处理等核心功能。

### 3.2 未来计划

- **阶段 4 (代码重构):**
  - **目标:** 将 `scripts/10_infer_to_result.py` 拆分为模块化的核心库，如 `pipeline`, `predictor`, `result_writer` 等。
- **阶段 5 (GUI 开发):**
  - **目标:** 使用 PyQt 开发图形用户界面，提供文件选择、参数调整和进度显示功能。
- **阶段 6 (模型训练):**
  - **目标:** 在 COCO 2017 数据集上微调模型，并保存最佳权重。
