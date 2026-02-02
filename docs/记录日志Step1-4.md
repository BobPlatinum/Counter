# HumanCounter3 实验记录日志（到 Step 4 为止）
> 目的：把“我做过什么、为什么这么做、遇到什么坑、怎么解决、学到什么”记录下来，方便以后回看项目时快速恢复上下文。  
> 项目根目录：`D:\Coding\PycharmCode\Counter`  
> Conda 环境：`HumanCounter3`（Python 3.11.14）  
> GPU：NVIDIA GeForce RTX 4070 Laptop GPU（8GB）

---

## Step 1｜GPU 环境与深度学习基础环境验收（Phase 1）

### 1.1 本步目标
1) **系统层确认**：显卡驱动可用、GPU 可被识别。  
2) **Python 层确认**：PyTorch 能在 GPU 上跑（CUDA 可用）。  
3) **框架层确认**：Ultralytics YOLO 能识别 GPU，并具备推理条件。

> 为什么必须做 Step 1：  
> 这个项目后续训练/推理都依赖 GPU。如果第一步不严格验收，后面任何“慢/报错/不稳定”都很难定位。

### 1.2 关键操作与命令记录

**（A）系统层 GPU/驱动检查**
- 命令：`nvidia-smi`
- 关键输出（摘录）：  
  - Driver Version：`591.74`  
  - CUDA Version：`13.1`  
  - GPU：`RTX 4070 Laptop GPU`（8188MiB）

**（B）PyTorch GPU 检查**
- 写脚本 `check_gpu.py` 做三件事：
  1. 打印 torch 版本与 `torch.version.cuda`
  2. 检查 `torch.cuda.is_available()`
  3. 在 GPU 上做一次矩阵乘法（确认不是“伪可用”）
- 关键输出（摘录）：
  - `torch: 2.7.1+cu118`
  - `torch.version.cuda: 11.8`
  - `cuda_available: True`
  - `gpu: NVIDIA GeForce RTX 4070 Laptop GPU`
  - `gpu_matmul_mean: ...`（说明 GPU 计算确实发生）

**（C）Ultralytics 自检（CLI）**
- 命令：
  - `yolo version`
  - `yolo checks`
- 关键输出（摘录）：
  - Ultralytics：`8.4.9`
  - Python：`3.11.14`
  - torch：`2.7.1+cu118`
  - CUDA：`11.8`
  - GPU：`CUDA:0 (RTX 4070 Laptop GPU)`

### 1.3 本步重点解析（知识点 + 我问过的“中断问题”整理）

#### Q1：为什么 `nvidia-smi` 显示 CUDA 13.1，但 PyTorch 显示 CUDA 11.8？冲突吗？
- 不冲突。  
- `nvidia-smi` 的 CUDA Version 通常表示“**驱动支持的最高 CUDA 运行时能力**”；  
- PyTorch 的 `torch.version.cuda=11.8` 表示“**这个 PyTorch 轮子编译/绑定的 CUDA 运行时版本（cu118）**”。  
- 只要 **驱动支持版本 ≥ PyTorch 所需版本**，就能正常工作。

#### 我应该学到什么？
- 机器学习项目里，“环境验收”不是可选项，而是第一阶段交付物。
- 以后看到“CUDA 版本不一致”，先区分：驱动能力 vs 程序运行时绑定版本。

### 1.4 本步成果（可复现状态）
- ✅ GPU 驱动正常（`nvidia-smi` OK）  
- ✅ PyTorch GPU 可用（`cuda_available=True`）  
- ✅ Ultralytics 能识别 GPU（`yolo checks` OK）  
- ✅ 具备进入推理流水线与后续训练的前置条件

---

## Step 2｜跑通 Ultralytics 推理 + 搞清楚输出目录（Phase 2.1）

### 2.1 本步目标
1) 用 Ultralytics CLI 做一次最小推理（冒烟测试）。  
2) 理解 Ultralytics 的 `runs_dir` / `settings.json` 机制，避免“结果文件找不到”。

### 2.2 关键行为与记录

**（A）冒烟测试命令（segmentation 预测）**
- 命令示例（当时用于验证推理链路）：
  - `yolo segment predict model=yolov8n-seg.pt source="https://ultralytics.com/images/bus.jpg" device=0`

**（B）现象：我没看到项目里出现 `runs/segment/predict*`**
- 这不是推理没跑，而是“输出目录不在我以为的位置”。

**（C）定位：查看 Ultralytics settings**
- 命令：`yolo settings`
- 关键输出（摘录）：
  - `datasets_dir`: `.\data\datasets`
  - `weights_dir`: `.\artifacts\weights`
  - `runs_dir`: `.\artifacts\runs`
  - settings.json 位于：`C:\Users\...\AppData\Roaming\Ultralytics\settings.json`

**（D）定位：检查我当时的工作目录（cwd）**
- 命令：`pwd`
- 输出：`D:\Coding\PycharmCode`
- 结论：我并不是在 `Counter` 项目根目录跑命令，而是在父目录跑的。

**（E）验证：去 `runs_dir` 对应位置找结果**
- 命令：`ls .\artifacts\runs\segment`
- 找到：`predict` 目录

### 2.3 本步重点解析（知识点 + 中断问题整理）

#### Q2：为什么我项目里没有 `runs/segment`，但推理又成功了？
- 因为 `runs_dir` 被设置为相对路径：`.\artifacts\runs`  
- 相对路径会跟随“当前工作目录（cwd）”解析。  
- 当 cwd 是 `D:\Coding\PycharmCode` 时，输出自然落到：  
  `D:\Coding\PycharmCode\artifacts\runs\...`  
  而不是：  
  `D:\Coding\PycharmCode\Counter\...`

#### Q3：PyCharm 终端为什么突然跑到父目录了？会影响前面的步骤吗？
- 会影响“输出文件落在哪里”，但不会影响 GPU 是否可用。  
- 前面 Step 1 的环境验收依然有效。  
- 解决思路：在 PyCharm 里把项目根正确打开到 `Counter`，并把 Terminal 的 Start directory 固定为项目根。

#### 我应该学到什么？
- **cwd（工作目录）是工程调试的核心变量之一**：相对路径、输出目录、缓存目录都会受它影响。
- **工具的默认输出（runs）不是产品的输出**：产品要自定义输出协议（我们下一步做 result/）。

### 2.4 本步成果
- ✅ 明确 Ultralytics 输出受 `runs_dir` 控制  
- ✅ 明确 cwd 改变会导致结果散落到父目录  
- ✅ 能通过 `yolo settings + pwd` 快速定位“结果去哪了”

---

## Step 3｜建立“可交付”的推理结果流水线：输出到 result/（Phase 2.2 + 工具化增强）

### 3.1 本步目标
把“能推理”升级为“可交付小工具”的关键点：  
1) **稳定输出协议**：每次运行都输出到 `result/YYYY-MM-DD_HHMMSS/`  
2) **支持单图与批量**  
3) **输出三件套**：
   - `annotated/`：框图（类别、置信度、颜色区分）
   - `silhouette/`：剪影图（实例 mask 实心填充）
   - `summary.json`：结构化输出（未来 GUI/API 直接读）

### 3.2 关键实现与记录

**（A）脚本：`scripts/10_infer_to_result.py`**
- 使用 Ultralytics Python API：`YOLO(weights).predict(...)`
- 核心参数：`--source --conf --iou --device --model`

**（B）第一次验收（v0.1 输出协议跑通）**
- 命令：
  - `python scripts/10_infer_to_result.py --source assets/examples/bus.jpg --device 0 --conf 0.25`
- 输出（摘录）：
  - `[OK] bus.jpg  total=6`
  - `Done. Results saved to: ...\result\2026-01-31_054901`
- 目录结构：
  - `result/2026-01-31_054901/annotated/bus.jpg`
  - `result/2026-01-31_054901/silhouette/bus.jpg`
  - `result/2026-01-31_054901/summary.json`

**（C）我阅读并理解了 summary.json**
- 示例（关键字段）：
  - `counts_by_class`: `person:4, bus:1, skateboard:1`
  - `total`: `6`
  - `detections[]`: 每个目标包含 `class_id/class_name/confidence/bbox_xyxy`

**（D）工具化增强（v0.2）**
1) annotated 图左上角叠加汇总：`total + 各类计数`
2) `summary.json` 增加 `inference_ms`
3) 支持批量：`--source assets/examples`

- 单图命令：
  - `python scripts/10_infer_to_result.py --source assets/examples/bus.jpg --device 0 --conf 0.25`
- 批量命令：
  - `python scripts/10_infer_to_result.py --source assets/examples --device 0 --conf 0.25`
- 推理耗时记录（示例）：
  - `"inference_ms": 441.44`

### 3.3 本步重点解析（知识点 + 中断问题整理）

#### Q4：为什么我点 PyCharm “运行脚本”会报：缺少 `--source`？
- 因为脚本用 `argparse` 把 `--source` 设为必填参数；  
- 如果从 IDE 直接运行没设置参数，就会出现：
  - `error: the following arguments are required: --source`
- 解决方案：
  - 在 PyCharm 的 Run/Debug Configuration 中填写 Script parameters；  
  - 或者在终端里按命令行方式运行（更可控）。

#### Q5：`yolov8n-seg.pt` 和 `bus.jpg` 放哪里？
- 权重属于产物 → 放 `artifacts/weights/`
- 示例图片属于测试资源 → 放 `assets/examples/`
- 原因：避免根目录混乱，后续训练数据、结果输出、权重版本管理都更清晰。

#### Q6：我看不懂 Results/Boxes/Masks，是怎么“变成 summary.json”的？
- 一张图推理返回一个 `Results` 对象。  
- `r.boxes` 里含有：
  - `xyxy`（bbox坐标）
  - `conf`（置信度）
  - `cls`（类别ID）
- `r.masks.data` 里含有每个实例的 mask（用于 silhouette）。  
- `r.plot()` 直接生成 annotated 图。

#### Q7：为什么 annotated 上 mask 遮罩很“厚”？
- 因为 plot 叠加了 masks。  
- 这是可视化风格问题，不是错误。  
- 产品角度可以选择：
  - annotated 只画框（更清爽）
  - silhouette 专门展示 mask（你已经做到了）

#### 我应该学到什么？
- **输出协议**是“工具化”的核心：GUI、API、批量处理都靠它对齐。
- `summary.json` 是未来前后端/客户端最重要的“接口契约”雏形。
- `conf` 阈值、`iou`（NMS）会影响检测结果数量与稳定性：这是模型“可调参数”。

### 3.4 本步成果
- ✅ 成功建立 result/ 输出协议（可复现、可批量）  
- ✅ annotated + silhouette + summary.json 三件套产出稳定  
- ✅ summary.json 包含：计数、明细、耗时（inference_ms）  
- ✅ annotated 叠加了“total+各类计数”，更像可交付工具

---

## Step 4｜工程化拆分（进行中/计划）：从“单脚本能跑”到“可维护可扩展”（Phase 4）

### 4.1 本步目标
把 `scripts/10_infer_to_result.py` 拆成模块化结构，以便后续：
- 接 PyQt GUI（不把 GUI 代码和推理代码混在一起）
- 预埋 API（未来做前后端/客户端直接调用）
- 更容易加入：日志、异常处理、模型切换、多线程、进度条

### 4.2 计划的模块边界（我将要做到什么）
建议拆成：
1) `core/predictor.py`：只负责加载模型 + predict  
2) `core/counting.py`：从 Results 提取 counts/detections  
3) `core/render.py`：渲染 annotated/silhouette（统一颜色、叠字）  
4) `io/result_writer.py`：只负责输出目录与写文件（图片/JSON）  
5) `core/pipeline.py`：组装上述模块，形成“推理流水线”  
6) `api/schemas.py`（预埋）：定义请求/响应数据结构（给 GUI/后端用）

> 核心思想：**把“变化快的地方”隔离出去**。GUI 是变化快的，模型/输出协议也会变化；Pipeline/Writer/Schema 的分层能降低改动成本。

### 4.3 本步重点解析（我应该学习什么）
- Python 项目结构：包（package）/模块（module）/脚本（script）的边界
- 单一职责：一个文件/类只做一件事
- 把“可复用的逻辑”从脚本迁移到包中，脚本只做参数解析与调用
- 为 GUI 预留：线程安全调用点（pipeline.run）和结构化返回（summary.json / Response DTO）

### 4.4 本步阶段性成果（当前状态）
- 已明确工程化拆分目标与目录结构方向
- 下一次继续开发时：优先完成拆分，并保证功能不变（回归测试用 bus.jpg）

---

## 总结｜截至 Step 4，我已经具备的能力与产物

### 已具备的能力
- 能正确配置 GPU 深度学习环境并完成验收
- 能使用 Ultralytics YOLOv8-seg 做推理并理解输出结构
- 能设计并实现稳定的输出协议（result/ 目录）
- 能将推理结果结构化为 JSON，面向未来 GUI/API

### 已形成的产物
- 推理脚本：`scripts/10_infer_to_result.py`
- 输出规范：`result/YYYY-MM-DD_HHMMSS/{annotated,silhouette,summary.json}`
- 示例权重与图片：`artifacts/weights/yolov8n-seg.pt`、`assets/examples/bus.jpg`
- 带汇总叠字的 annotated 图、带实心实例的 silhouette 图、含 inference_ms 的 summary.json

### 下一步（Step 4 的下半段）
- 完成模块拆分并保持功能不变（回归测试）
- 预埋 API schema（请求/响应数据结构）
- 再进入 GUI（COUNTER.py）与“结果流水线面板”实现（进度条、日志、阈值滑条、批量处理）
