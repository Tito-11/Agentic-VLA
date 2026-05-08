# LAMA-VLM (Lifelong Autonomous Multi-Agent VLM) 仿真实验设计规划

为全面验证升级后的 LAMA-VLM 在此前架构改进（ChromaDB 记忆演进 Evo-KAM、LangGraph 状态机 DAG-EAM、V-PCC 压缩、Topo-Graph RAG）下的有效性，现设计并实现以下仿真实验测试环境。注意，虽然任务不仅局限于抓取（Grasping），而是可以泛化至通用机器人操作（Robotic Manipulation），但本套基准实验为了对齐之前的性能评测，将以复杂堆叠环境下的抓取与放置（Pick & Place）作为 Manipulation 的首个落地测试子集：

## 1. 实验环境设定 (Simulation Environment Setup)

测试平台：SAPIEN / ManiSkill3 (`PickClutterYCB-v1` 及其变体)
交互模式：`pd_ee_delta_pose` 带有可监控的反向物理反馈和失败报错 (Timeout/Collision/Grasp Error)
对比基线 (Baselines)：
- **VLM-Only (Zero-shot)**: 无外部知识库，单纯依赖当前视觉输入进行规划计算坐标。
- **Agentic RAG-VLM (Vanilla)**: IROS 2026 版本，具备单物体特征的 HAA-RAG 和硬式重试重试与历史硬记录，没有子图空间推理。
- **Hybrid VLM-Action Model (New Phase)**: 将 VLM 规划与基于端到端的模仿学习 ACT/SmolVLA 模块结合的实验版本，运行在 RTX 5090 硬件闭环环境。

## 2. 核心大挑战设计 (Core Extreme Stress Tests)

### Scenario 0a: VLM Grounding + Kinematics Verification (视觉定位与硬编码运动学闭环验证)
*   **目标：** 在不引入新不确定性 (Action Model) 的前提下，验证 Qwen3-VL-8B 的端到端多模态推理与空间坐标定位能力。
*   **设定：** RTX 5090 加载 `models/Qwen3-VL-8B`。移除所有预设的场景坐标，ManiSkill 环境全靠 VLM 直接生成目标物品与放置区域的 Bounding Box 或 [X,Y,Z] 坐标。后续依靠现有的 `pose_utils.py` 完成 IK 抓握。
*   **验证指标：**
    *   **VLM Grounding Accuracy**: VLM 输出的 BBox 与空间中心点相对于真实物体的 RMSE 误差。
    *   **Inference Latency**: VLM 推理单张 RGB 图片与 prompt 的总用时。

### Scenario 0b: VLM Task Planner + Action Model Integration (大脑+小脑集成)
*   **目标：** 在 0a 成功验证 VLM 在本地完美解析出物体方位与操作意图后，进一步剥离运动学部分。
*   **设定：** 将 VLM 生成的高语义特征（例如 target ROI 或 instruction embedding）持续传递给由 PyTorch 驱动的 50Hz 纯模仿小脑模型 (如 ACT) 输出连续电机指令。
*   **验证指标：** 
    *   **VRAM Peak**: 确保双模型在 32GB 显存中稳定运行 (< 26-28GB)。
    *   **Collision Rate Improvement**: 对比硬编码 IK，小脑生成动作引起的意外环境冲撞率的降低幅度。

### Scenario 1: Dense Clutter & Occlusion Track (密集堆叠与遮挡挑战)
*   **目标：** 验证 Topo-Graph RAG 的空间推理能力。
*   **设定：** 目标物体（如番茄汤罐）被盘子压住或紧密遮挡，左侧放有易碎玻璃杯。
*   **验证指标：** 成功提取并匹配 `supports/occluded_by` 拓扑图结构。能主动绕开遮挡，调整更高的 Z-offset 和避障策略，不碰翻玻璃杯。这是纯视觉或单物体匹配无法做到的。

### Scenario 2: Lifelong Friction Drift Track (动态物理扰动与终身自适应挑战)
*   **目标：** 验证 Evo-KAM (PlugMem) 记忆演进与 LangGraph (DAG) 的状态路由。
*   **设定：** 中途悄悄改变仿真环境（如将该物体的滑动摩擦系数大幅降低）。
*   **验证指标：** 在发生连续的“Slipping”报错后，触发冲突阈值 (Conflict Threshold)。能在运行中主动修正 ChromaDB 中记录的 `force` 值 (如从 `soft_grip` 到 `high_grip`)，并利用 conditional edges 路由使得智能体在第 2~3次中调整力度成功抓取。

### Scenario 3: Long-Horizon Marathon Track (大规模桌面清理马拉松)
*   **目标：** 验证 V-PCC (渐进式上下文压缩) 的效率开销抑制能力。
*   **设定：** 桌面上有 15 个杂乱物品需要连续搬运分配。
*   **验证指标：** 
    - 记录 Tokens 随步骤的增长曲线 (Number of items processed vs Prompt Tokens)。
    - Vanilla 版本会在第 5-6 个物品时因上下文溢出崩溃或产生严重计算延迟。而采用状态梳理（Micro Compact 与 SnapKV-like 帧保留）的系统，Token 开销在长程任务中趋于平缓常量。

## 3. Evaluation Metrics
- **Success Rate (SR)**: 总抓取并搬运成功的概率。
- **Action Steps / Trial**: 单物体完成所需步数 (评估 LangGraph 的高效性)。
- **Token Cost Growth Rate**: 验证上下文压缩 V-PCC 的影响。
- **Average Recovery Time**: 从报错（碰撞/滑脱）到自适应抓取成功的时长。

以上 Benchmark 场景为新版系统的论文宣发核心亮点（Contributions）。
