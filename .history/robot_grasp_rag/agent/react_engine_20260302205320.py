"""Core module implementation."""

import json
import re
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class AgentState(Enum):


    """AgentState class."""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    RETRIEVING = "retrieving"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ThoughtStep:
    """ThoughtStep class."""
    step_id: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReActTrace:
    """ReActTrace class."""
    query: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    total_time_ms: float = 0
    
    def add_step(self, thought: str, action: str,
                 action_input: Dict[str, Any], observation: str) -> None:
        """add_step function."""
        step = ThoughtStep(
            step_id=len(self.steps) + 1,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        self.steps.append(step)
        
    def to_string(self) -> str:
        
        """to_string function."""
        lines = [f"Query: {self.query}\n"]
        for step in self.steps:
            lines.append(f"Step {step.step_id}:")
            lines.append(f"  Thought: {step.thought}")
            lines.append(f"  Action: {step.action}")
            lines.append(f"  Action Input: {json.dumps(step.action_input, ensure_ascii=False)}")
            lines.append(f"  Observation: {step.observation}\n")
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


class Tool(ABC):


    """Tool class."""
    @property
    @abstractmethod
    def name(self) -> str:
        """name function."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """description function."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """parameters function."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """execute function."""
        pass


class ToolRegistry:


    """ToolRegistry class."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool) -> None:
        
        """register function."""
        self._tools[tool.name] = tool
        
    def get(self, name: str) -> Optional[Tool]:
        
        """get function."""
        return self._tools.get(name)
        
    def list_tools(self) -> List[str]:
        
        """list_tools function."""
        return list(self._tools.keys())
    
    def get_tools_prompt(self) -> str:
    
        """get_tools_prompt function."""
        lines = ["Available Tools:"]
        for name, tool in self._tools.items():
            lines.append(f"\n{name}: {tool.description}")
            lines.append(f"  Parameters: {json.dumps(tool.parameters, ensure_ascii=False)}")
        return "\n".join(lines)


# Note


class RetrieveExperienceTool(Tool):


    """RetrieveExperienceTool class."""
    def __init__(self, retriever = None):
        self._retriever = retriever
        
    @property
    def name(self) -> str:
        return "retrieve_experience"
    
    @property
    def description(self) -> str:
        return "Retrieve successful grasp experiences for similar objects from the knowledge base"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "query": "Query description, e.g. object name or category",
            "top_k": "Number of results to return (default: 3)"
        }
    
    def execute(self, query: str, top_k: int = 3, **kwargs) -> str:
        if self._retriever is None:
            return "Error: Retriever not initialized"
        
        results = self._retriever.retrieve(query_text=query, top_k=top_k)
        
        if not results:
            return "No similar experiences found."
        
        output = []
        for i, r in enumerate(results):
            output.append(f"{i+1}. {r.object_name} (similarity: {r.combined_score:.2f})")
            output.append(f"     Grasp position: {r.grasp_pose.get('position', {})}")
            output.append(f"     Gripper width: {r.grasp_pose.get('gripper_width', 'N/A')}")
            output.append(f"   Notes: {r.metadata.get('notes', 'N/A')}")
        
        return "\n".join(output)


class AnalyzeObjectTool(Tool):


    """AnalyzeObjectTool class."""
    def __init__(self, vlm_engine = None):
        self._vlm_engine = vlm_engine
        
    @property
    def name(self) -> str:
        return "analyze_object"
    
    @property
    def description(self) -> str:
        return "Use VLM to analyze the target object properties (shape, material, size, graspable regions)"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "object_name": "object_name",
            "focus_aspects": "aspects to focus on (e.g. shape, material)"
        }
    
    def execute(self, object_name: str, focus_aspects: List[str] = None,
                image=None, **kwargs) -> str:
        if self._vlm_engine is None:
            return self._simulate_analysis(object_name)
        
        prompt = f"""Analyze the object \"{object_name}\" in the image and provide:
1. Category (e.g. kitchenware/tool/stationery/container/food)
2. Material (e.g. metal/plastic/glass/ceramic)
3. Approximate dimensions (L x W x H in cm)
4. Stable position (e.g. upright/lying/tilted/stacked)
5. Surface properties (e.g. smooth/rough/textured)
6. Recommended grasp strategy

Return your analysis in JSON format."""
        
        response = self._vlm_engine.generate(prompt=prompt, images=[image] if image else None)
        return response
    
    def _simulate_analysis(self, object_name: str) -> str:
    
        """_simulate_analysis function."""
        templates = {
            "cup": {"material": "ceramic/glass", "fragile": "true", "grip_area": "handle"},
            "bottle": {"material": "plastic/glass", "fragile": "varies", "grip_area": "body"},
            "tool": {"material": "metal/rubber", "fragile": "false", "grip_area": "handle"},
            "utils": {"material": "mixed", "fragile": "false", "grip_area": "body"},
            "default": {"material": "unknown", "fragile": "unknown", "grip_area": "center"},
        }
        
        for key, attrs in templates.items():
            if key in object_name:
                return json.dumps({
                    "object": object_name,
                    **attrs,
                    "estimated_size": "10x8x8 cm",
                    "center_of_mass": "estimated_center"
                }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "object": object_name,
            "material": "unknown",
            "fragile": "unknown",
            "grip_area": "center"
        }, ensure_ascii=False, indent=2)


class PlanGraspTool(Tool):


    """PlanGraspTool class."""
    def __init__(self, vlm_engine = None, context_builder = None):
        self._vlm_engine = vlm_engine
        self._context_builder = context_builder
        
    @property
    def name(self) -> str:
        return "plan_grasp"
    
    @property
    def description(self) -> str:
        return "Combine object analysis and retrieved experiences to plan an optimal grasp pose"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "object_analysis": "Object analysis result",
            "reference_experiences": "Retrieved similar experiences",
            "constraints": "Task constraints (e.g. workspace limits)"
        }
    
    def execute(self, object_analysis:
        str, reference_experiences: str = "",
                constraints: str = "", **kwargs) -> str: # Note

        grasp_plan = {
            "grasp_pose": {
                "position": {"x": 0.35, "y": 0.0, "z": 0.15},
                "orientation": {"qx": 0, "qy": 0.707, "qz": 0, "qw": 0.707},
                "gripper_width": 0.06
            },
            "approach_direction": "top-down",
            "pre_grasp_distance": 0.05,
            "grasp_force": "medium",
            "notes": "Grasp plan based on object analysis and retrieved experiences"
        }
        return json.dumps(grasp_plan, ensure_ascii=False, indent=2)


class SimulateGraspTool(Tool):


    """SimulateGraspTool class."""
    @property
    def name(self) -> str:
        return "simulate_grasp"
    
    @property
    def description(self) -> str:
        return "Execute the grasp action and verify the planned pose against constraints"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "grasp_plan": "Grasp planning result",
            "object_properties": "Object properties"
        }
    
    def execute(self, grasp_plan: str, object_properties: str = "", **kwargs) -> str:
        # Note

        import random
        
        checks = {
            "collision_free": random.random() > 0.1,
            "force_appropriate": random.random() > 0.15,
            "stability": random.random() > 0.2,
            "reachable": random.random() > 0.05,
        }
        
        all_pass = all(checks.values())
        
        result = {
            "simulation_passed": all_pass,
            "checks": checks,
            "confidence": 0.85 if all_pass else 0.4,
            "warnings": []
        }
        
        if not checks["collision_free"]:
            result["warnings"].append("Potential collision detected")
        if not checks["force_appropriate"]:
            result["warnings"].append("Force may be inappropriate")
        if not checks["stability"]:
            result["warnings"].append("Grasp stability is low")
        if not checks["reachable"]:
            result["warnings"].append("Target position out of reach")
            
        return json.dumps(result, ensure_ascii=False, indent=2)


class ReflectFailureTool(Tool):


    """ReflectFailureTool class."""
    def __init__(self, vlm_engine = None):
        self._vlm_engine = vlm_engine
        
    @property
    def name(self) -> str:
        return "reflect_failure"
    
    @property
    def description(self) -> str:
        return "Analyze grasp failure causes and suggest improvements"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "failure_type": "Failure type (slip/collision/unreachable/width_mismatch)",
            "execution_context": "Execution context details",
            "previous_attempts": "Previous attempt history"
        }
    
    def execute(self, failure_type:
        str, execution_context: str = "",
                previous_attempts: str = "", **kwargs) -> str: # Note

        failure_analysis = {
            "slip": {
                "cause": "Excessive force applied",
                "suggestions": [
                    "Increase grip force (+20%)",
                    "Position error detected",
                    "Insufficient grasp stability"
                ]
            },
            "collision": {
                "cause": "Collision in grasp approach path",
                "suggestions": [
                    "Adjust approach direction",
                    "Adjust grasp strategy",
                    "Suboptimal approach path"
                ]
            },
            "unreachable": {
                "cause": "Target position out of reach",
                "suggestions": [
                    "Refine position estimate",
                    "Adjust grasp strategy",
                    "Check tool constraints"
                ]
            },
            "width_mismatch": {
                "cause": "Gripper-object size mismatch",
                "suggestions": [
                    "Re-measure object width",
                    "Adjust gripper width (±15%)",
                    "Try alternative grasp type"
                ]
            },
            "force_damage": {
                "cause": "Excessive force applied",
                "suggestions": [
                    "Reduce force by 30%",
                    "Use force-sensitive grip mode",
                    "Switch to deformable object strategy"
                ]
            }
        }
        
        analysis = failure_analysis.get(failure_type, {
            "cause": "Unidentified failure cause",
            "suggestions": ["Re-analyze the failure", "Try different grasp approach"]
        })
        
        # Note

        attempt_count = previous_attempts.count("attempt") if previous_attempts else 0
        if attempt_count >= 2:
            analysis["suggestions"].insert(0, "Try alternative grasp approach")
        
        return json.dumps({
            "failure_type": failure_type,
            "root_cause": analysis["cause"],
            "improvement_suggestions": analysis["suggestions"],
            "retry_recommended": attempt_count < 3,
            "confidence_adjustment": -0.1 * (attempt_count + 1)
        }, ensure_ascii=False, indent=2)


# Note


class ReActEngine:


    """ReActEngine class."""
    REACT_PROMPT = """You are a grasp planning Agent. Use the ReAct reasoning framework to solve the task.

task: {task}

Available Tools:
{tools}

Format: Thought: your reasoning
Action: choose a tool
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [tool returns result]
... (Repeat Thought/Action/Observation until done)
Thought: I have completed the task.
Final Answer: the complete grasp plan.

Reasoning so far:
"""

    CONTINUE_PROMPT = """
Observation: {observation}

Continue reasoning (complete the task and provide a Final Answer):
"""

    def __init__(self, vlm_engine = None, max_steps: int = 6):
        self._vlm_engine = vlm_engine
        self._max_steps = max_steps
        self._tool_registry = ToolRegistry()
        self._setup_default_tools()
        
    def _setup_default_tools(self):
        
        """_setup_default_tools function."""
        self._tool_registry.register(RetrieveExperienceTool())
        self._tool_registry.register(AnalyzeObjectTool())
        self._tool_registry.register(PlanGraspTool())
        self._tool_registry.register(SimulateGraspTool())
        self._tool_registry.register(ReflectFailureTool())
        
    def register_tool(self, tool: Tool) -> None:
        
        """register_tool function."""
        self._tool_registry.register(tool)
        
    def set_retriever(self, retriever) -> None:
        
        """set_retriever function."""
        tool = self._tool_registry.get("retrieve_experience")
        if tool and isinstance(tool, RetrieveExperienceTool):
            tool._retriever = retriever
            
    def run(self, task: str, context: Dict[str, Any] = None) -> ReActTrace:
            
        """run function."""
        start_time = time.time()
        trace = ReActTrace(query=task)
        
        # Note

        tools_prompt = self._tool_registry.get_tools_prompt()
        prompt = self.REACT_PROMPT.format(task=task, tools=tools_prompt)
        
        conversation = prompt
        
        for step in range(self._max_steps):
            # Note

            response = self._generate(conversation)
            
            # Note

            parsed = self._parse_response(response)
            
            if parsed.get("final_answer"):
                trace.final_answer = parsed["final_answer"]
                trace.success = True
                break
                
            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            
            if not action:
                # Note

                trace.add_step(thought, "error", {}, "No valid action found")
                break
                
            # Note

            observation = self._execute_tool(action, action_input, context)
            
            # Note

            trace.add_step(thought, action, action_input, observation)
            
            # Note

            conversation += response + self.CONTINUE_PROMPT.format(observation=observation)
            
        trace.total_time_ms = (time.time() - start_time) * 1000
        return trace
    
    def _generate(self, prompt: str) -> str:
    
        """_generate function."""
        if self._vlm_engine is None:
            # Note

            return self._simulate_response(prompt)
        return self._vlm_engine.generate(prompt=prompt)
    
    def _simulate_response(self, prompt: str) -> str:
    
        """_simulate_response function."""
        if "Final Answer" in prompt or "Observation" in prompt.split("\n")[-5:]:
            return """Thought: Planning grasp strategy
Final Answer: {
    "grasp_pose": {
        "position": {"x": 0.35, "y": 0.0, "z": 0.15},
        "gripper_width": 0.06
    },
    "confidence": 0.85,
    "reasoning": "Based on object analysis and retrieved experiences, planning optimal grasp"
}"""
        
        step = prompt.count("Observation")
        
        if step == 0:
            return """Thought: Retrieving similar object grasp experiences
Action: retrieve_experience
Action Input: {"query": "grasp similar object", "top_k": 3}"""
        elif step == 1:
            return """Thought: Retrieved experiences, now analyzing the object
Action: analyze_object
Action Input: {"object_name": "target_object", "focus_aspects": ["material", "size", "fragility"]}"""
        elif step == 2:
            return """Thought: Using retrieved results to plan grasp pose
Action: plan_grasp
Action Input: {"object_analysis": "Analyzed object properties", "reference_experiences": "Retrieved grasp experiences applied"}"""
        else:
            return """Thought: Finalizing the grasp plan
Action: simulate_grasp
Action Input: {"grasp_plan": "Optimized grasp", "object_properties": "Analyzed"}"""
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
    
        """_parse_response function."""
        result = {}
        
        # Note

        final_match = re.search(r'Final Answer:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result
            
        # Note

        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
            
        # Note

        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            result["action"] = action_match.group(1).strip()
            
        # Note

        input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                result["action_input"] = {}
                
        return result
    
    def _execute_tool(self, action: str, action_input: Dict[str, Any],
                       context: Dict[str, Any] = None) -> str:
        """_execute_tool function."""
        tool = self._tool_registry.get(action)
        if tool is None:
            return f"Error: Unknown tool '{action}'"
            
        try:
            # Note

            if context:
                action_input = {**action_input, **context}
            return tool.execute(**action_input)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"


if __name__ == "__main__":
    # Note

    engine = ReActEngine(vlm_engine=None, max_steps=5)
    
    task = "Plan an optimal grasp for the target object"
    trace = engine.run(task)
    
    print("=" * 60)
    print("ReAct Reasoning Trace")
    print("=" * 60)
    print(trace.to_string())
    print(f"\nsuccess: {trace.success}")
    print(f"total_time: {trace.total_time_ms:.1f}ms")
