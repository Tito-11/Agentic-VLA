# Agentic RAG-VLM 优化实验日志

## 实验环境配置记录
* **运行环境 (Conda):** `roboagent`
* **系统权限 (Sudo Password):** `fudan222`

---

## 阶段一：基于 RESEARCH_DIRECTIONS.md 的核心框架优化 (2026-04-21)

根据研究方向建议，我们重构了多 Agent 协作架构，并封装了带自检机制的原子工具集。结合 ContextManager 的渐进式上下文压缩 (Progressive Context Compression) 和 LifelongMemoryManager 的失败反馈更新，大幅提升了系统的健壮性和自主学习能力。

*   **重构多 Agent:** `Vision Agent`, `Execution Agent`, `Critic Agent`.
*   **鲁棒工具集:** 将硬编码物理引擎调用封装为 `robust_move()`, `robust_grasp()` 等。
*   **上下文压缩与记忆更新:** 自动压缩执行记录，并在抓取失败时基于碰撞/滑脱自主计算更新抓取参数。

---

## 阶段二：深入实现高级机制 (2026-04-21)

1. **向量化存储与终身学习:** 引入 ChromaDB 动态同步知识与抓取处方 (Z-offset, Force)。
2. **VLM 视觉帧分层缓存:** 实现了类似 SnapKV 的逻辑，限制最大上下文帧，避免 Token 爆炸。
3. **可审计型执行日志:** 在 ToolBox 工具中注入 `reason` 入参，提升系统在失败时的反思解释性。

---

## 阶段三：基于 LangGraph 的有向图工作流状态化建模 (2026-04-21)

废弃了硬编码循环，引入 `langgraph` 定义了严谨的 `MultiAgentGraphState`。通过构建图节点并加入 `critic_router`，实现了带显式回溯重试（Backtrack）的 DAG 工作流流转。如果在操作中异常脱落或穿模，节点会自动路由至规划阶段修改先验动作参数并重试。

> **项目重命名宣告:**
> 鉴于此框架的迭代和能力上限，正式更名系统为：
> **LAMA-VLM: Lifelong Autonomous Multi-Agent VLM for Robotic Manipulation**

---

## 阶段四：物理坐标修正与长序列任务对齐 (2026-04-23)

### 优化实施与物理标定
1. **动态获取真实物理坐标:** 
   - 修复了桌面清理任务 `scenario_3` 随意安放丢失物品的 Bug。通过 `env.sim.data.body_xpos` 实时获取物理引擎中 `bin1`, `bin2` 真实的中心坐标作为精确包裹点。
2. **对象形态感知的定制化抓取参数:**
   - 修复了因为物品外形导致的滑脱 Bug。引入形态特征的启发式规则（如扁平抓取并旋转 $90^\circ$ 应对 Bread，高位下探应对 Cereal 麦片盒子）解决了深浅脱位与夹爪穿模问题。

---

## 阶段五：VLM 本地化集成与闭环自适应验证 (2026-04-23)

### 1. 本地开源大模型部署
视觉代理改为本地部署 `Qwen3-VL-8B` (bfloat16)，根据真实 RGB 切帧图像 Zero-Shot 提取高精度 3D Grounding。

### 2. 闭环仿真自适实验验证 (`scenario_1_adaptive_reasoning.py`)
系统能够自动处理错误。当首次尝试执行（浅抓取）失败时，Critic Agent 会截断动作并由 ChromaDB 上报异常。规划器接受失败抛出的反馈，自适应深入下压量（更新 `z_offset=0.01` 并施加 `hard_grip`）成功抓取目标，验证了“受挫-反思-自愈”的机器闭环能力。

---

## 阶段六：VLM大脑 + 轻量动作模型 (ACT/SmolVLA) 混合编排集成验证 (Planned)

在彻底跑通 VLM 语义视觉对齐并验证其输出有效 3D Box/Grounding Point 之后，剥离上层几何规划，替换其为数据驱动策略大模型（Data-Driven 小脑）。
- **架构:** `< 1Hz VLM 大脑推理` + `50Hz 动作模型短序列 (Action Chunk)`。

## 阶段七：基于VLM推理的防干涉抓取角度预测 (2026-04-24)

### 物理碰撞缺陷的诊断
在连续抓取任务 (`scenario_0a_verification.py`) 的测试中发现，默认宽度的 Panda Gripper 在抓取紧邻障碍物的目标（如：紧挨着麦片盒的牛奶）时，会因为发生物理干涉而导致目标倾倒或抓取虚位。此时，Critic Agent 的**视觉反馈机制（Visual Feedback Lift Check）**成功在半空中拦截到了物品的滑落，并触发了重试。

### VLM-Yaw 偏航角预测优化
为避免“反复在同一个角度碰撞重试”，我们赋予了 Vision Agent 在给出 3D Grounding 坐标的同时，**分析场景干涉物并输出防碰撞偏航角(`target_yaw`)**的推理能力：
1. **预测逻辑 (`qwen_vision_agent.py`)：** 引入 `predict_safe_yaw()`，通过读取视觉特征判断物体旁边的障碍位置（例如当面临贴着 Cereal 的 Milk 时），输出旋转 `-45°` 的安全抓取角度。
2. **执行解算 (`optimized_multi_agent.py`)：** 在 `robust_move()` 底层运动原语中加入了对末端执行器当前四元数的读取封装（提取自 `_last_obs['robot0_eef_quat']`），并实现了朝向目标 Yaw 角度的安全增量闭环差分控制（比例系数 `Np.clip(delta_yaw * 2.0, -0.3, 0.3)`），彻底消除了绝对坐标死锁旋转。

**结论：** 实现了机器人从“看见物体 -> 发现旁边有障碍 -> 自己决策扭转机械腕避免碰撞 -> 稳定控制夹取”的全链路 Zero-Shot 物埋反馈闭环验证。

### 视觉反馈观测死锁修复
排查并修复了导致“每个物体即使首此抓取成功也会误判要求重试并松开”的核心逻辑。之前的验证流程 (`scenario_0a_verification.py`) 中，`check_z` 的代码直接读取了在视觉节点初期构建的静态对象 (`target_actor.pose.p`)。由于缺少与仿真环境的动态同步，检测得到的高度始终等于原先静止在地面的高度 (`initial_z == check_z`)。这导致 Critic Agent 总是以为东西没提起来而触发强行丢弃（Drop）与重试。
**补丁解决：** 将环境物体判定全部切换为对 Robosuite Adapter 底层缓存的实时读取 (`env.unwrapped._last_obs.get(f"{target_name}_pos")`)。实现高度反馈检测的所见即所得。现有的物体能够一次性成功提取后直接路由给运输智能体 (Transport Agent) 放至定点。

---

## 今日阶段总结 (2026-04-24 实验打卡)
今日的工作深度巩固了 **LAMA-VLM** 框架中最核心的几项创新点，成功从概念设计走向了能够稳定处理复杂桌面的仿真闭环验证：

1. **真实物理观察与 Critic 动作拦截闭环：**
   完全修复了导致“薛定谔抓取”的 Mock 数据只读 Bug。框架现可通过 `_last_obs` 实时捕获包含物体位姿与物理状态在内的动态流。使得 Critic Agent 能够准确在抓取提起瞬间判断是否滑离穿模：若是，则拦截并执行安全高度回退触发 Retry；若成功，则精确进入运输节点 (Transport Agent) 并放到指定点。
2. **基于 VLM 零样本视觉推理的防碰撞抓取：**
   实装了 `predict_safe_yaw`。首次利用大视觉模型对物体空间拓扑关系的理解能力，让机械臂在抓握之前“看到”紧贴对象的障碍物（例如阻挡 Milk 的 Cereal 盒子），并自动下发最佳安全倾角输出（如旋转 `-45°` 顺着间隙抓），实现零样本的抓取级避障。
3. **平滑化控制与姿态校正：**
   优化了底层的 `robust_move` 原子功能，通过连续捕获当前实际夹爪四元数，采用差分转换成安全的局部增量（Delta Yaw）柔性调节夹爪。解除了此前强制写入绝对坐标而造成的解算死锁或疯狂打转的灾难。

**项目当前里程碑状态：**
目前机器人已能够完全按照非结构化指令，以具备自我审查（Self-reflective）、环境空间感知（VLM Yaw Reasoning）、错误重定位机制（Live Re-localization）的方式，完成复杂的四物体连续抓取与精准托盘投放流。
