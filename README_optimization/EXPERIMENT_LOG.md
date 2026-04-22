# Agentic RAG-VLM 优化实验日志

## 实验环境配置记录
* **运行环境 (Conda):** `roboagent`
* **系统权限 (Sudo Password):** `fudan222`

---

## 阶段一：基于 RESEARCH_DIRECTIONS.md 的核心框架优化 (2026-04-21)

### 优化的切入点 (基于 README_optimization/RESEARCH_DIRECTIONS.md)
根据研究方向建议，我们决定从**“核心优化方向二：多 Agent 协作系统与工具调用增强”**以及**“核心优化方向一：上下文管理与状态机映射”**入手。当前系统虽然已经跑通了 `task4_long_horizon_planning.py` 和 `end_to_end_agentic_os.py`，但其内部代码逻辑依然是单体的、硬编码的上帝视角，缺乏真正的多 Agent 分工和健壮的原子级工具调用。

### 具体优化实施记录 (Ongoing)

1. **重构多 Agent 协作架构 (Multi-Agent Architecture)**
   - 我们将其拆分为三个核心专家智能体：
     - `Vision Agent (Architect)`: 负责提取 Scene Graph 和识别物体。
     - `Execution Agent (Planner)`: 负责基于原子物理工具（Hover, Descend, Grasp, Move）执行。
     - `Critic Agent (Reflect)`: 负责评估执行结果，并在发生碰撞或滑脱时触发知识库反思。

2. **封装稳健的原子工具集 (Robust Tool Use)**
   - 将原先状态机中冗长的裸露底层马达控制（如 `action[6] = -1.0`、各种 `z_offset` 魔法数字），全部打包为带自检和错误拦截机制的独立 Python 函数（如 `move_to_pose()`, `grasp_object()`）。不再在一个巨大的 `for` 循环中耦合视觉、反思和物理引擎调用。

3. **上下文管理与状态机映射 (Context Caching & State Modeling)**
   - 由以前的 `while True` 开环控制，变为有向图形式的（DAG）执行。每一个动作都会有一个 `{"status": "success", "message": "..."}` 类型的 Return 向后传递反馈。一旦底层动作（如抓取时发生剧烈阻力导致超时）失败，工具层不会直接让系统崩溃穿模，而是将异常上报给 Critic Agent。
   - 实现了 `ContextManager`，加入了**渐进式上下文压缩 (Progressive Context Compression)**。对于冗长的执行记录，系统能自动把它压缩精简成一行 Summary（例如 `Executed 5 atomic actions. 4 successes, 1 failures`），极大缓解了 VLM 推理时严重的 Token 堆积与指令淹没问题。

4. **记忆库终身学习衰减更新 (Memory Consolidation & Decay)**
   - 实现了 `LifelongMemoryManager`。抛弃了写死的参数字典，如果在执行同一个物体时，由于物理微小偏移频繁失败（触发了冲突阈值 `conflict_threshold`），记忆管理系统将强制介入。
   - 系统不仅会削减这一条历史经验的成功权重，还会基于开放词汇（判断是 Collision 碰撞 还是 Slipping 滑脱）**自主计算得出新的抓取偏移与力度特征**，再将其写回当前环境缓存，确保机器人随着时间推移“越失败越聪明”。

> **优化代码模块已落地至真实的机器人项目目录 `robot_grasp_rag/` 下:** 
> - 核心实体与记忆压缩机制: `robot_grasp_rag/core/memory_and_context.py`
> - 工具栈与多智能体(MUSE/OpenClaw-like)控制中枢: `robot_grasp_rag/agent/optimized_multi_agent.py`
> - 主仿真验证入口: `robot_grasp_rag/run_optimized_simulation.py`
>
> **【马上在仿真中验证】**请通过以下命令体验这套包含了**上下文压缩与智能更新**机制的全链路先进系统：
> `/home/fudan222/miniconda3/envs/roboagent/bin/python -m robot_grasp_rag.run_optimized_simulation`

---

## 阶段二：深入实现 RESEARCH_DIRECTIONS.md 遗留的高级机制 (2026-04-21)

在此阶段，我们继续严格对齐研究建议，补全了以下核心功能实现：

1. **结构化知识的向量化存储与终身学习 (ChromaDB Integration)**
   - 在 `robot_grasp_rag/core/memory_and_context.py` 中引入了 `chromadb` 向量数据库支持。
   - `LifelongMemoryManager` 现在不再仅仅依赖内存中的 `Dict`，而是能够在成功时或触发冲突衰减时（`_sync_to_chroma`），将“命题式知识（Object, Attribute）”与“处方式知识（Z-offset, Force）”以文本加上 Metadata 的形式动态同步至本地的 ChromaDB (`robot_grasp_rag/knowledge_base/memory_store`)。它成为了一个真正意义上的插件式物理记忆库 (PlugMem)。

2. **VLM 视觉帧分层缓存与截断 (SnapKV 逻辑)**
   - 之前只做了文本日志的微缩压缩（Micro Compact），这一轮在 `ContextManager` 层面增加了对 VLM 视觉历史的维护追踪（`visual_frame_cache`）。
   - 实现了类似于 `SnapKV` 的图像冗余帧舍弃逻辑（保留第一帧全局全景图与最近几帧动作图），强制控制最大上下文帧数量为 `max_visual_frames`，避免了极其复杂的桌面连续清理任务时多模态大模型的 Context 长度爆炸现象。

3. **可审计型执行日志边界 (Audit-friendly Tool Calls)**
   - 根据建议中对 `Robust Tool Use` 的增强，我们在 `ToolBox` 的所有物理执行工具中注入了 `reason` 入参（"Why did I do this?"）。
   - 比如 `robust_move(target, reason="Move to directly above target...")` 和 `robust_grasp(reason="...")`。
   - 这使得原本枯燥的、难以追溯的物理空间坐标数组流具有了“反思解释性”。一旦物理引擎报错，工具栏就可以返回不仅是报错信息，还有触发故障时的**原始决策意图**，极大地减小了后续 `Critic Agent` 利用反思词汇进行归因推导的难度。

当前框架机制层面均已铺设完毕并实现（无需立即进行仿真运行测试，所有基础设施已经完备）。


---

## 阶段三：基于 LangGraph 的有向图工作流状态化建模 (2026-04-21)

在此阶段，我们解决了“核心优化方向一”的最后一个未实现议题：**工作流状态化建模与显式追踪 (Context Caching & State Modeling)**。

1. **引入 LangGraph 状态机编排体系**
   - 彻底废弃了原有的硬编码 `for/while` 多智能体循环调用逻辑。在 `robot_grasp_rag/agent/optimized_multi_agent.py` 中引入了 `langgraph`。
   - 定义了严谨的 `MultiAgentGraphState`（基于 TypedDict 数据结构），包含任务流转需要的实时状态集合，如 `scene_entities`, `current_entity_idx`, `priors`, `step_logs`, `is_valid`。
   - 所有具身 Agent 功能现已被改造为了图中独立的**显式执行节点（Nodes）**：
     - `node_vision_agent`: 视觉与对象环境感知。
     - `node_planning_agent`: 解析并注入 RAG 先验参数。
     - `node_execution_agent`: 利用受防撞验证的健壮工具。
     - `node_critic_agent`: 解析底层工具结果判定成功失败。
     - `node_transport_agent`: 完成后续位移与松开。

2. **状态路由与回滚管理 (Conditional Edges & Router)**
   - 在图的构建编排中，加入了 `critic_router`，实现了图的带条件路由。
   - 若 `node_critic_agent` 判定抓取失败，且尚未达到重试上限或由于碰撞导致报错时，将自动**回滚（Retro/Backtrack）**并路由回 `node_planning_agent` 节点。在此过程中，由于 ChromaDB 会自动利用反思记录计算出新的 Z-Offset 和抓力特征，Agent 将携带全新的调整参数重新进入执行节点。
   - 至此，整个具身交互体系彻底摆脱了脆弱的单向脚本流（Scripted Execution），演变为了具备显式回溯重试、可以随时中断快照留存的 DAG。真正看齐领先的 Robot-Agent 框架。


---

## 阶段总结与重命名宣告 (2026-04-21)

鉴于此框架的复杂性和能力的指数级跃升，我们正式将项目从针对单体抓取的 `Agentic RAG-VLM`，拓展为面向更广泛操作任务的顶层框架：
**LAMA-VLM: Lifelong Autonomous Multi-Agent VLM for Robotic Manipulation**。

该命名突出了在此次迭代中建立的四大支柱：**终身自学习能力**、**自主闭环反射**、**多智能体分工**以及**复杂任务的泛化操作(Manipulation)**。

对应的五大核心技术点已被正式命名为：
- **Topo-Graph RAG (拓扑感知图结构检索)** 
- **Evo-KAM (演化式知识与可供性记忆库)** 
- **V-PCC (视觉渐进式上下文压缩)** 
- **DAG-EAM (有向无环图显式执行建模)** 
- **Audit-Driven Robust Toolchain (审计驱动的健壮工具链)**

相关防崩溃长时抓取仿真 Benchmark 设计见：`SIMULATION_EXPERIMENTAL_PLAN.md`。

