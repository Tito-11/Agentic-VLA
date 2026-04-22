from typing import Dict, List, Any, TypedDict
import numpy as np
import logging
from ..core.memory_and_context import LifelongMemoryManager, ContextManager
from ..core.graph_rag import GraphRAGManager

try:
    from langgraph.graph import StateGraph, START, END
    import operator
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

logger = logging.getLogger("MultiAgentWorkflow")

class ToolExecutionException(Exception):
    pass

class ToolBox:
    def __init__(self, env):
        self.env = env
        
    def robust_move(self, target_pos, reason: str = "Unknown", precision=0.02, timeout=50, z_offset=0.0) -> Dict[str, Any]:
        """
        基于 RESEARCH_DIRECTIONS.md 优化的工具：可审计型执行拦截、错误回退 (Error+Fallback)
        """
        target = target_pos + np.array([0, 0, z_offset])
        logger.info(f"    [Tool Use] Attempting atomic move to {target} (Precision: {precision}) - Reason: {reason}")
        steps = 0
        try:
            while steps < timeout:
                ee_pose = self.env.unwrapped.agent.tcp.pose.p
                if hasattr(ee_pose, 'cpu'):
                    ee_pos = ee_pose[0].cpu().numpy()
                else:
                    ee_pos = ee_pose[0] if len(ee_pose.shape) > 1 else ee_pose
                    
                delta = target - ee_pos
                action = np.zeros(7)
                action[6] = 1.0 # 夹爪张开
                action[0:3] = np.clip(delta * 12.0, -0.15, 0.15)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                steps += 1
                
                if np.linalg.norm(delta) < precision:
                    return {"status": "success", "message": f"Reached geometric target."}
            
            # 返回给环境的异常信息，替代原本死板的错误字典
            return {"status": "timeout", "message": f"Movement timeout. Collision might have occurred. Delta remaining: {delta}"}
        except Exception as e:
            return {"status": "error", "message": f"Engine execution crashed: {str(e)}"}

    def robust_grasp(self, reason: str = "Unknown", force_profile="medium_grip") -> Dict[str, Any]:
        logger.info(f"    [Tool Use] Applying atomic grasp with {force_profile} - Reason: {reason}")
        steps = 0
        try:
            while steps < 30: # 闭合锁死逻辑
                action = np.zeros(7)
                action[6] = -1.0 # 夹合参数
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                steps += 1
            return {"status": "success", "message": "Grasp closure complete. Tactile engaged."}
        except Exception as e:
            return {"status": "error", "message": f"Grasping crashed: {str(e)}"}

    def robust_release(self, reason: str = "Unknown") -> Dict[str, Any]:
         logger.info(f"    [Tool Use] Releasing payload. - Reason: {reason}")
         steps = 0
         while steps < 20:
             action = np.zeros(7)
             action[6] = 1.0 # 张开
             obs, reward, terminated, truncated, info = self.env.step(action)
             self.env.render()
             steps += 1
         return {"status": "success", "message": "Payload released."}


class MultiAgentGraphState(TypedDict):
    """LangGraph 状态结构定义"""
    scene_entities: List[Dict]
    storage_bin: Any
    current_entity_idx: int
    current_entity_name: str
    target_obj_pos: Any
    priors: Dict[str, Any]
    scene_graph: List[Dict]
    retries: int
    step_logs: List[Dict]
    is_valid: bool
    status: str

class MultiAgentOrchestrator:
    """
    基于 RESEARCH_DIRECTIONS.md 优化的: 多 Agent 协作系统架构 (MUSE/OpenClaw-like)
    并使用 LangGraph 实现了工作流状态化显式追踪。
    """
    def __init__(self, env):
        self.env = env
        self.tools = ToolBox(env)
        self.memory = LifelongMemoryManager()
        self.context = ContextManager()
        self.graph_rag = GraphRAGManager()
        
        if HAS_LANGGRAPH:
            self.graph = self._build_langgraph()
        else:
            logger.warning("LangGraph not installed. Run 'pip install langgraph'. Using fallback.")

    def _build_langgraph(self):
        """构建显式状态追踪的有向图 (DAG) 编排器"""
        workflow = StateGraph(MultiAgentGraphState)
        
        # 注册所有 Agent 与工具节点
        workflow.add_node("vision_agent", self.node_vision_agent)
        workflow.add_node("planning_agent", self.node_planning_agent)
        workflow.add_node("execution_agent", self.node_execution_agent)
        workflow.add_node("critic_agent", self.node_critic_agent)
        workflow.add_node("transport_agent", self.node_transport_agent)

        # 定义边和流转条件
        workflow.add_edge(START, "vision_agent")
        workflow.add_edge("vision_agent", "planning_agent")
        workflow.add_edge("planning_agent", "execution_agent")
        workflow.add_edge("execution_agent", "critic_agent")
        
        # 条件分支: 判断是否抓取成功
        def critic_router(state: MultiAgentGraphState):
            if state["is_valid"]:
                return "transport_agent"
            elif state["retries"] > 0:
                return "planning_agent" # 重试 (进入规划层拿最新 Memory)
            else:
                return "vision_agent" # 失败且无重试次数，跳过该物体，看下一个
        
        workflow.add_conditional_edges(
            "critic_agent",
            critic_router,
            {"transport_agent": "transport_agent", "planning_agent": "planning_agent", "vision_agent": "vision_agent"}
        )
        
        # 存放成功后，找下一个物体
        workflow.add_edge("transport_agent", "vision_agent")

        return workflow.compile()

    def node_vision_agent(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        """负责复杂场景的第一步解析，构建 Scene Graph，如果所有实体已经处理完则返回结束状态"""
        logger.info("👁️ [Arch/Vision Agent] Parsing Environment... (LangGraph Node)")
        # Initialize or index
        if "current_entity_idx" not in state or state["current_entity_idx"] is None:
            state["current_entity_idx"] = 0
            
            actors = self.env.unwrapped.scene.get_all_actors()
            scene_entities = []
            for a in actors:
                name = a.name.lower()
                if any(ignore in name for ignore in ["table", "ground", "goal", "agent"]): 
                    continue
                scene_entities.append({"node_id": name, "actor_ref": a, "relationships": ["supported_by_table"]})
            state["scene_entities"] = scene_entities
        
        if state["current_entity_idx"] >= len(state["scene_entities"]):
            state["status"] = "FINISHED"
            return state

        current_entity = state["scene_entities"][state["current_entity_idx"]]
        clean_name = current_entity["node_id"].split('_')[-1]
        state["current_entity_name"] = clean_name
        
        # 物理坐标系映射
        target_obj_pose = current_entity["actor_ref"].pose.p
        if hasattr(target_obj_pose, 'cpu'):
            pos = target_obj_pose[0].cpu().numpy().copy()
        else:
            pos = target_obj_pose[0].copy() if len(target_obj_pose.shape) > 1 else target_obj_pose.copy()
        
        state["target_obj_pos"] = pos
        if "retries" not in state or state["current_entity_name"] != state.get("last_processed_entity", ""):
            state["retries"] = 2
            state["last_processed_entity"] = state["current_entity_name"]
            
        return state

    def node_planning_agent(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        """检索 RAG 与 终身记忆的先验"""
        if state.get("status") == "FINISHED": return state
        clean_name = state["current_entity_name"]
        logger.info(f"🧠 [Planner Agent] Extracting RAG & Memory for [{clean_name}]")
        
        priors = self.memory.extract_prior(clean_name)
        
        # --- Graph RAG 空间推理 ---
        actors = self.env.unwrapped.scene.get_all_actors()
        current_entity = state["scene_entities"][state["current_entity_idx"]]
        scene_graph = self.graph_rag.build_scene_graph(actors, current_entity["actor_ref"])
        graph_strategy = self.graph_rag.retrieve_strategy(scene_graph)
        if graph_strategy: priors.update(graph_strategy)
        
        logger.info(f"    -> RAG Prior injected: Z-Offset={priors.get('z_offset', 0.015)}, Force={priors.get('force', 'medium')}")
        
        state["priors"] = priors
        state["scene_graph"] = scene_graph
        state["step_logs"] = [] # 重置当次操作日志
        return state

    def node_execution_agent(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        """执行低维控制集合。原子动作节点"""
        if state.get("status") == "FINISHED": return state
        
        target_pos = state["target_obj_pos"]
        priors = state["priors"]
        clean_name = state["current_entity_name"]
        
        logger.info("⚙️  [Execution Agent] Executing action sequence...")
        
        # Hover
        res_1 = self.tools.robust_move(target_pos, precision=0.03, z_offset=0.15, reason="Move to above target")
        state["step_logs"].append(res_1)
        self.context.log_action(f"Hover ({clean_name})", res_1["status"], res_1["message"])
        
        # Descend
        if res_1["status"] != "error":
            res_2 = self.tools.robust_move(target_pos, precision=0.02, timeout=60, z_offset=priors["z_offset"], reason="Descend to precise point")
            state["step_logs"].append(res_2)
            self.context.log_action(f"Descend ({clean_name})", res_2["status"], res_2["message"])
        
        # Grasp
        if state["step_logs"][-1]["status"] != "error":
            res_3 = self.tools.robust_grasp(force_profile=priors["force"], reason="Close gripper")
            state["step_logs"].append(res_3)
            self.context.log_action(f"Grasp ({clean_name})", res_3["status"], res_3["message"])
            
        return state

    def node_critic_agent(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        """审查并进行终身学习归因"""
        if state.get("status") == "FINISHED": return state
        clean_name = state["current_entity_name"]
        step_logs = state["step_logs"]
        
        logger.info("⚖️ [Reflect/Critic Agent] Evaluating execution results...")
        is_success = True
        for res in step_logs:
            if res["status"] != "success":
                is_success = False
                logger.warning(f"    -> Failure encountered: {res.get('message')}")
                 
        if is_success:
            self.memory.consolidate_memory(clean_name, {"status": "success"})
            self.graph_rag.store_graph_experience(state["scene_graph"], {"z_offset": state["priors"].get("z_offset"), "force": state["priors"].get("force")})
            state["is_valid"] = True
        else:
            self.memory.consolidate_memory(clean_name, {"status": "failure", "message": "collision"})
            state["is_valid"] = False
            state["retries"] -= 1
            
            # 返回安全位
            logger.info("    -> [Critic] Rejecting. Returning to safe height.")
            self.tools.robust_release(reason="Emergency release")
            self.tools.robust_move(state["target_obj_pos"], precision=0.03, z_offset=0.20, reason="Emergency retract")
            
            if state["retries"] <= 0:
                logger.error(f"❌ Exhausted retries for {clean_name}. Aborting this specific target.")
                state["current_entity_idx"] += 1 # 失败且无重试，指向下一个
                
        return state

    def node_transport_agent(self, state: MultiAgentGraphState) -> MultiAgentGraphState:
        """整理与搬运，更新全局状态"""
        if state.get("status") == "FINISHED": return state
        clean_name = state["current_entity_name"]
        storage_bin = state["storage_bin"]
        
        # Navigate and Release
        move_res = self.tools.robust_move(storage_bin, precision=0.04, timeout=80, z_offset=0.0, reason="Transfer payload to storage")
        self.context.log_action(f"MoveToBin ({clean_name})", move_res["status"], move_res["message"])
        
        release_res = self.tools.robust_release(reason="Release payload")
        self.context.log_action(f"Release ({clean_name})", release_res["status"], release_res["message"])
        
        # 渐进式上下文压缩
        comp_log = self.context.progressive_compression()
        logger.info(f"🗜️ Context Compressed: {comp_log}")
        
        # 预备下一个目标
        state["storage_bin"][0] += 0.03
        state["storage_bin"][1] += 0.03
        state["current_entity_idx"] += 1
        
        return state

    def vision_agent(self):
        """兼容老版本接口，无实际行为"""
        pass
    def execution_agent(self, scene_entities, storage_bin):
        """启动 LangGraph"""
        if not HAS_LANGGRAPH:
            logger.error("LangGraph absent! Cannot run state machine workflow.")
            return
            
        initial_state = {
            "scene_entities": scene_entities,
            "storage_bin": storage_bin.copy(),
            "current_entity_idx": None,
            "is_valid": False,
            "status": "RUNNING"
        }
        
        logger.info("🚀 Starting explicit State Machine via LangGraph Orchestrator...")
        
        # 流式执行 DAG
        for s in self.graph.stream(initial_state, config={"recursion_limit": 100}):
            current_node = list(s.keys())[0]
            state_data = s[current_node]
            if state_data.get("status") == "FINISHED":
                break
        
        logger.info("\n🎉 All LangGraph processes finished. Desk layout optimized via explicitly tracked states!")

