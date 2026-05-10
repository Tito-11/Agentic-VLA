"""Core module implementation."""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
import traceback


class GraspStatus(Enum):


    """GraspStatus class."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    REJECTED = "rejected"  # Note


class FailureType(Enum):


    """FailureType class."""
    COLLISION = "collision"           # Note
    SLIP = "slip"                     # Note
    MISS = "miss"                     # Note
    DAMAGE = "damage"                 # Note
    POSE_ERROR = "pose_error"         # Note
    FORCE_ERROR = "force_error"       # Note
    UNKNOWN = "unknown"


@dataclass
class ReflectionResult:
    """ReflectionResult class."""
    is_safe: bool                     # Note
    confidence: float                 # Note
    issues: List[str]                 # Note
    suggestions: List[str]            # Note
    should_retry: bool                # Note
    retry_strategy: Optional[str]     # Note


@dataclass
class GraspPlan:
    """GraspPlan class."""
    object_name: str
    position: Dict[str, float]        # x, y, z
    orientation: Dict[str, float]     # qx, qy, qz, qw
    gripper_width: float
    approach_direction: str           # top, side, front
    force_level: str                  # gentle, normal, firm
    key_considerations: List[str]     # Note
    confidence: float


@dataclass
class GraspResult:
    """GraspResult class."""
    status: GraspStatus
    plan: GraspPlan
    reflection: Optional[ReflectionResult]
    execution_result: Optional[Dict[str, Any]]
    failure_analysis: Optional[Dict[str, Any]]
    retry_count: int
    total_time: float


@dataclass
class AgenticConfig:
    """AgenticConfig class."""
    # Note

    max_retries: int = 3
    retry_delay: float = 0.5
    
    # Note

    enable_reflection: bool = True
    reflection_threshold: float = 0.7  # Note
    
    # Note

    enable_learning: bool = True
    failure_penalty: float = 0.3      # Note
    success_bonus: float = 0.1        # Note
    
    # Note

    use_haa_rag: bool = True
    top_k: int = 5


class SelfReflectionModule:


    """SelfReflectionModule class."""
    REFLECTION_PROMPT = """Review the grasp execution result. Analyze the grasp plan and suggest improvements.

## Scene Context
- target_object: {object_name}
- - Description: {object_description}
- Affordance type: {affordance_type}
- Material: {material}
- Fragile: {is_fragile}

## Grasp Plan
- Position: ({px:.3f}, {py:.3f}, {pz:.3f})
- Approach direction: {approach_direction}
- Gripper width: {gripper_width:.3f}m
- Force level: {force_level}
- Considerations: {considerations}

## Similar Experiences
{similar_experiences}

## Review Points:
1. **Grasp stability**: Is the gripper properly matched to the object?
2. **Force appropriateness**: Is the force level suitable for this material?
3. **Experience alignment**: Are similar experiences properly utilized?

JSON file pathoutput:
```json
{{
    "is_safe": true/false,
    "confidence": 0.0-1.0,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "should_retry": true/false,
    "retry_strategy": "description of retry strategy"
}}
```
"""

    def __init__(self, vlm_client):
        self.vlm_client = vlm_client
        self.reflection_history = []
    
    def reflect(
        self,
        image: Image.Image,
        plan: GraspPlan,
        context: Dict[str, Any],
    ) -> ReflectionResult:
        """reflect function."""
        # Note

        prompt = self.REFLECTION_PROMPT.format(
            object_name=plan.object_name,
            object_description=context.get("description", ""),
            affordance_type=context.get("affordance_type", "unknown"),
            material=context.get("material", "unknown"),
            is_fragile=context.get("is_fragile", False),
            px=plan.position.get("x", 0),
            py=plan.position.get("y", 0),
            pz=plan.position.get("z", 0),
            approach_direction=plan.approach_direction,
            gripper_width=plan.gripper_width,
            force_level=plan.force_level,
            considerations=", ".join(plan.key_considerations),
            similar_experiences=self._format_experiences(context.get("experiences", [])),
        )
        
        # Note

        try:
            response = self.vlm_client.generate(
                image=image,
                prompt=prompt,
                max_tokens=512,
            )
            
            # Note

            result = self._parse_reflection_response(response)
            
            # Note

            self.reflection_history.append({
                "plan": plan.__dict__,
                "result": result.__dict__,
                "timestamp": time.time(),
            })
            
            return result
            
        except Exception as e:
            print(f"[Reflection] Analysis failed: {e}")
            return ReflectionResult(
                is_safe=True,  # Note
                confidence=0.5,
                issues=[f"Reflection module error: {str(e)}"],
                suggestions=[],
                should_retry=False,
                retry_strategy=None,
            )
    
    def _format_experiences(self, experiences: List[Dict]) -> str:
    
        """_format_experiences function."""
        if not experiences:
            return "experiences"
        
        lines = []
        for i, exp in enumerate(experiences[:3], 1):
            lines.append(f"{i}. {exp.get('object_name', 'N/A')}: "
                        f"position=({exp.get('grasp_pose', {}).get('position', {}).get('x', 0):.2f}, "
                        f"{exp.get('grasp_pose', {}).get('position', {}).get('y', 0):.2f}, "
                        f"{exp.get('grasp_pose', {}).get('position', {}).get('z', 0):.2f}), "
                        f"force={exp.get('force_level', 'normal')}")
        
        return "\n".join(lines)
    
    def _parse_reflection_response(self, response: str) -> ReflectionResult:
    
        """_parse_reflection_response function."""
        try:
            # Note

            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # Note

                data = json.loads(response)
            
            return ReflectionResult(
                is_safe=data.get("is_safe", True),
                confidence=data.get("confidence", 0.5),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                should_retry=data.get("should_retry", False),
                retry_strategy=data.get("retry_strategy"),
            )
        except:
            return ReflectionResult(
                is_safe=True,
                confidence=0.5,
                issues=["Unable to parse result"],
                suggestions=[],
                should_retry=False,
                retry_strategy=None,
            )


class FailureAnalyzer:


    """FailureAnalyzer class."""
    ANALYSIS_PROMPT = """Analyze the grasp failure and identify root causes.

## Grasp Parameters
- Object: {object_name}
- Position: ({px:.3f}, {py:.3f}, {pz:.3f})
- Force level: {force_level}
- Gripper width: {gripper_width:.3f}m

## Failure Description
{failure_description}

## Sensor Feedback
{sensor_feedback}

Analyze the failure and output JSON:
```json
{{
    "failure_type": "collision/slip/miss/damage/pose_error/force_error/unknown",
    "root_cause": "Detailed root cause analysis",
    "adjustments": {{
        "position_delta": {{"x": 0.0, "y": 0.0, "z": 0.0}},
        "force_adjustment": "increase/decrease/maintain",
        "approach_change": "top/side/front/none",
        "gripper_adjustment": 0.0
    }},
    "retry_confidence": 0.0-1.0,
    "should_abort": false
}}
```
"""
    
    def __init__(self, vlm_client):
        self.vlm_client = vlm_client
        self.failure_history = []
    
    def analyze(
        self,
        image: Image.Image,
        plan: GraspPlan,
        failure_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """analyze function."""
        prompt = self.ANALYSIS_PROMPT.format(
            object_name=plan.object_name,
            px=plan.position.get("x", 0),
            py=plan.position.get("y", 0),
            pz=plan.position.get("z", 0),
            force_level=plan.force_level,
            gripper_width=plan.gripper_width,
            failure_description=failure_info.get("description", "No description available"),
            sensor_feedback=json.dumps(failure_info.get("sensors", {}), indent=2),
        )
        
        try:
            response = self.vlm_client.generate(
                image=image,
                prompt=prompt,
                max_tokens=512,
            )
            
            analysis = self._parse_analysis_response(response)
            
            # Note

            self.failure_history.append({
                "plan": plan.__dict__,
                "failure_info": failure_info,
                "analysis": analysis,
                "timestamp": time.time(),
            })
            
            return analysis
            
        except Exception as e:
            return {
                "failure_type": FailureType.UNKNOWN.value,
                "root_cause": f"Analysis error: {str(e)}",
                "adjustments": {},
                "retry_confidence": 0.3,
                "should_abort": False,
            }
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
    
        """_parse_analysis_response function."""
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return json.loads(response)
        except:
            return {
                "failure_type": FailureType.UNKNOWN.value,
                "root_cause": "Unable to determine root cause",
                "adjustments": {},
                "retry_confidence": 0.3,
                "should_abort": False,
            }
    
    def apply_adjustments(
        self,
        plan: GraspPlan,
        adjustments: Dict[str, Any],
    ) -> GraspPlan:
        """apply_adjustments function."""
        new_position = plan.position.copy()
        delta = adjustments.get("position_delta", {})
        new_position["x"] = new_position.get("x", 0) + delta.get("x", 0)
        new_position["y"] = new_position.get("y", 0) + delta.get("y", 0)
        new_position["z"] = new_position.get("z", 0) + delta.get("z", 0)
        
        # Note

        force_adj = adjustments.get("force_adjustment", "maintain")
        force_map = {
            "gentle": {"increase": "normal", "decrease": "gentle", "maintain": "gentle"},
            "normal": {"increase": "firm", "decrease": "gentle", "maintain": "normal"},
            "firm": {"increase": "firm", "decrease": "normal", "maintain": "firm"},
        }
        new_force = force_map.get(plan.force_level, {}).get(force_adj, plan.force_level)
        
        # Note

        new_approach = adjustments.get("approach_change", "none")
        if new_approach == "none":
            new_approach = plan.approach_direction
        
        # Note

        new_gripper = plan.gripper_width + adjustments.get("gripper_adjustment", 0)
        new_gripper = max(0.01, min(0.15, new_gripper))  # Note
        
        return GraspPlan(
            object_name=plan.object_name,
            position=new_position,
            orientation=plan.orientation,
            gripper_width=new_gripper,
            approach_direction=new_approach,
            force_level=new_force,
            key_considerations=plan.key_considerations + ["Adjusted based on failure analysis"],
            confidence=plan.confidence * adjustments.get("retry_confidence", 0.8),
        )


class ExperienceEvolver:


    """ExperienceEvolver class."""
    def __init__(self, vector_store, config: AgenticConfig):
        self.vector_store = vector_store
        self.config = config
        self.experience_scores = {}  # id -> quality_score
    
    def on_success(self, experience_ids: List[str], plan: GraspPlan):
    
        """on_success function."""
        for exp_id in experience_ids:
            current_score = self.experience_scores.get(exp_id, 1.0)
            new_score = min(2.0, current_score + self.config.success_bonus)
            self.experience_scores[exp_id] = new_score
            
            # Note

            if hasattr(self.vector_store, 'update_metadata'):
                self.vector_store.update_metadata(exp_id, {
                    "quality_score": new_score,
                    "last_success": time.time(),
                })
        
        print(f"[ExperienceEvolver] Updated {len(experience_ids)} experiences")
    
    def on_failure(
        self,
        experience_ids: List[str],
        plan: GraspPlan,
        failure_analysis: Dict[str, Any],
    ):
        """on_failure function."""
        failure_type = failure_analysis.get("failure_type", "unknown")
        
        for exp_id in experience_ids:
            current_score = self.experience_scores.get(exp_id, 1.0)
            # Note

            penalty = self.config.failure_penalty
            if failure_type in ["damage", "collision"]:
                penalty *= 1.5  # Note
            
            new_score = max(0.1, current_score - penalty)
            self.experience_scores[exp_id] = new_score
            
            if hasattr(self.vector_store, 'update_metadata'):
                self.vector_store.update_metadata(exp_id, {
                    "quality_score": new_score,
                    "last_failure": time.time(),
                    "failure_type": failure_type,
                })
        
        print(f"[ExperienceEvolver] Penalized {len(experience_ids)} experiences (failure: {failure_type})")
    
    def get_adjusted_scores(
        self,
        experience_ids: List[str],
        original_scores: List[float],
    ) -> List[float]:
        """get_adjusted_scores function."""
        adjusted = []
        for exp_id, score in zip(experience_ids, original_scores):
            quality = self.experience_scores.get(exp_id, 1.0)
            adjusted.append(score * quality)
        return adjusted


class AgenticGraspPipeline:


    """AgenticGraspPipeline class."""
    PLANNING_PROMPT = """Plan the grasp action. Use the scene image and retrieved experiences to generate an optimal grasp pose.

## task
{task_instruction}

## target_object
- sample: {object_name}
- sample: {affordance_type}
- sample: {material}
- sample: {is_fragile}

## retrieved_experiences
{experiences}

Generate a grasp plan and output JSON:
```json
{{
    "position": {{"x": 0.0, "y": 0.0, "z": 0.0}},
    "orientation": {{"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}},
    "gripper_width": 0.08,
    "approach_direction": "top/side/front",
    "force_level": "gentle/normal/firm",
    "key_considerations": ["consideration 1", "consideration 2"],
    "confidence": 0.85
}}
```
"""
    
    def __init__(
        self,
        vlm_client,
        retriever,  # Note
        executor=None,  # Note
        config: Optional[AgenticConfig] = None,
    ):
        self.vlm_client = vlm_client
        self.retriever = retriever
        self.executor = executor
        self.config = config or AgenticConfig()
        
        # Note

        self.reflection = SelfReflectionModule(vlm_client)
        self.failure_analyzer = FailureAnalyzer(vlm_client)
        self.experience_evolver = ExperienceEvolver(
            retriever.vector_store if hasattr(retriever, 'vector_store') else None,
            config,
        )
        
        # Note

        self.stats = {
            "total_attempts": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "reflection_rejections": 0,
        }
    
    def grasp(
        self,
        image: Image.Image,
        instruction: str,
        target_object: Optional[str] = None,
    ) -> GraspResult:
        """grasp function."""
        start_time = time.time()
        self.stats["total_attempts"] += 1
        
        retry_count = 0
        current_plan = None
        reflection_result = None
        failure_analysis = None
        retrieved_experiences = []
        
        while retry_count <= self.config.max_retries:
            try:
                # Note

                if retry_count == 0:
                    context = self._perceive_and_retrieve(image, instruction, target_object)
                    retrieved_experiences = context.get("experiences", [])
                
                # Note

                if current_plan is None or (failure_analysis and not failure_analysis.get("should_abort")):
                    if failure_analysis:
                        # Note

                        current_plan = self.failure_analyzer.apply_adjustments(
                            current_plan,
                            failure_analysis.get("adjustments", {}),
                        )
                    else:
                        current_plan = self._generate_plan(image, instruction, context)
                
                # Note

                if self.config.enable_reflection:
                    reflection_result = self.reflection.reflect(image, current_plan, context)
                    
                    if not reflection_result.is_safe or reflection_result.confidence < self.config.reflection_threshold:
                        self.stats["reflection_rejections"] += 1
                        
                        if reflection_result.should_retry and retry_count < self.config.max_retries:
                            print(f"[Reflection] Issues found: {reflection_result.issues}")
                            retry_count += 1
                            self.stats["retries"] += 1
                            
                            # Note

                            failure_analysis = {
                                "failure_type": "reflection_rejection",
                                "adjustments": self._suggestions_to_adjustments(reflection_result.suggestions),
                                "retry_confidence": 0.7,
                            }
                            continue
                        else:
                            # Note

                            return GraspResult(
                                status=GraspStatus.REJECTED,
                                plan=current_plan,
                                reflection=reflection_result,
                                execution_result=None,
                                failure_analysis=None,
                                retry_count=retry_count,
                                total_time=time.time() - start_time,
                            )
                
                # Note

                execution_result = self._execute(current_plan)
                
                if execution_result.get("success"):
                    # Note

                    self.stats["successes"] += 1
                    if self.config.enable_learning:
                        self.experience_evolver.on_success(
                            [e.get("id") for e in retrieved_experiences],
                            current_plan,
                        )
                    
                    return GraspResult(
                        status=GraspStatus.SUCCESS,
                        plan=current_plan,
                        reflection=reflection_result,
                        execution_result=execution_result,
                        failure_analysis=None,
                        retry_count=retry_count,
                        total_time=time.time() - start_time,
                    )
                else:
                    # Note

                    failure_analysis = self.failure_analyzer.analyze(
                        image, current_plan, execution_result
                    )
                    
                    if failure_analysis.get("should_abort") or retry_count >= self.config.max_retries:
                        # Note

                        self.stats["failures"] += 1
                        if self.config.enable_learning:
                            self.experience_evolver.on_failure(
                                [e.get("id") for e in retrieved_experiences],
                                current_plan,
                                failure_analysis,
                            )
                        
                        return GraspResult(
                            status=GraspStatus.FAILED,
                            plan=current_plan,
                            reflection=reflection_result,
                            execution_result=execution_result,
                            failure_analysis=failure_analysis,
                            retry_count=retry_count,
                            total_time=time.time() - start_time,
                        )
                    
                    # Note

                    retry_count += 1
                    self.stats["retries"] += 1
                    print(f"[Pipeline] Retry {retry_count}/{self.config.max_retries}, "
                          f"Cause: {failure_analysis.get('root_cause', 'unknown')}")
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                print(f"[Pipeline] Error: {e}")
                traceback.print_exc()
                retry_count += 1
                if retry_count > self.config.max_retries:
                    return GraspResult(
                        status=GraspStatus.FAILED,
                        plan=current_plan,
                        reflection=reflection_result,
                        execution_result={"error": str(e)},
                        failure_analysis={"failure_type": "exception", "root_cause": str(e)},
                        retry_count=retry_count,
                        total_time=time.time() - start_time,
                    )
        
        # Note

        return GraspResult(
            status=GraspStatus.FAILED,
            plan=current_plan,
            reflection=reflection_result,
            execution_result=None,
            failure_analysis=None,
            retry_count=retry_count,
            total_time=time.time() - start_time,
        )
    
    def _perceive_and_retrieve(
        self,
        image: Image.Image,
        instruction: str,
        target_object: Optional[str],
    ) -> Dict[str, Any]:
        """_perceive_and_retrieve function."""
        # Note

        if hasattr(self.retriever, 'retrieve_and_build_context'):
            # HAA-RAG
            context = self.retriever.retrieve_and_build_context(
                query_image=image,
                query_text=instruction,
                top_k=self.config.top_k,
            )
        else:
            # Note

            results = self.retriever.retrieve(image, instruction, self.config.top_k)
            context = {
                "experiences": [r.__dict__ for r in results] if results else [],
            }
        
        # Note

        if target_object:
            context["target_object"] = target_object
        
        return context
    
    def _generate_plan(
        self,
        image: Image.Image,
        instruction: str,
        context: Dict[str, Any],
    ) -> GraspPlan:
        """_generate_plan function."""
        affordance = context.get("target_affordance", {})
        experiences = context.get("experiences", [])
        
        prompt = self.PLANNING_PROMPT.format(
            task_instruction=instruction,
            object_name=context.get("target_object", "target_object"),
            affordance_type=affordance.get("primary_affordance", "unknown"),
            material=affordance.get("material", "unknown"),
            is_fragile=affordance.get("fragility", 0) > 0.5,
            experiences=self._format_experiences_for_planning(experiences),
        )
        
        try:
            response = self.vlm_client.generate(
                image=image,
                prompt=prompt,
                max_tokens=512,
            )
            
            return self._parse_plan_response(response, context)
            
        except Exception as e:
            print(f"[Planning] Planning failed: {e}")
            # Note

            return GraspPlan(
                object_name=context.get("target_object", "unknown"),
                position={"x": 0.0, "y": 0.0, "z": 0.1},
                orientation={"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
                gripper_width=0.08,
                approach_direction="top",
                force_level="normal",
                key_considerations=["Default fallback plan"],
                confidence=0.3,
            )
    
    def _format_experiences_for_planning(self, experiences: List[Dict]) -> str:
    
        """_format_experiences_for_planning function."""
        if not experiences:
            return "No relevant experiences found"
        
        lines = []
        for i, exp in enumerate(experiences[:5], 1):
            pose = exp.get("grasp_pose", {})
            pos = pose.get("position", {})
            lines.append(
                f"{i}. {exp.get('object_name', 'N/A')}: "
                f"position=({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f}), "
                f"gripper_width={pose.get('gripper_width', 0.08):.3f}m, "
                f"experience: {exp.get('key_insight', 'N/A')}"
            )
        
        return "\n".join(lines)
    
    def _parse_plan_response(self, response: str, context: Dict) -> GraspPlan:
    
        """_parse_plan_response function."""
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)
            
            return GraspPlan(
                object_name=context.get("target_object", "unknown"),
                position=data.get("position", {"x": 0, "y": 0, "z": 0.1}),
                orientation=data.get("orientation", {"qx": 0, "qy": 0, "qz": 0, "qw": 1}),
                gripper_width=data.get("gripper_width", 0.08),
                approach_direction=data.get("approach_direction", "top"),
                force_level=data.get("force_level", "normal"),
                key_considerations=data.get("key_considerations", []),
                confidence=data.get("confidence", 0.5),
            )
        except:
            return GraspPlan(
                object_name=context.get("target_object", "unknown"),
                position={"x": 0.0, "y": 0.0, "z": 0.1},
                orientation={"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
                gripper_width=0.08,
                approach_direction="top",
                force_level="normal",
                key_considerations=["Fallback plan, limited information available"],
                confidence=0.3,
            )
    
    def _execute(self, plan: GraspPlan) -> Dict[str, Any]:
    
        """_execute function."""
        if self.executor is not None:
            return self.executor.execute(plan)
        else:
            # Note

            import random
            success = random.random() < plan.confidence
            return {
                "success": success,
                "description": "Execution succeeded" if success else "Execution failed",
                "sensors": {
                    "force": random.uniform(1, 10),
                    "contact": success,
                },
            }
    
    def _suggestions_to_adjustments(self, suggestions: List[str]) -> Dict[str, Any]:
    
        """_suggestions_to_adjustments function."""
        adjustments = {
            "position_delta": {"x": 0, "y": 0, "z": 0},
            "force_adjustment": "maintain",
            "approach_change": "none",
            "gripper_adjustment": 0,
            "retry_confidence": 0.7,
        }
        
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if "position" in suggestion or "position" in suggestion_lower:
                adjustments["position_delta"]["z"] += 0.01
            if "force" in suggestion_lower or "strength" in suggestion_lower:
                if "increase" in suggestion_lower or "firm" in suggestion_lower:
                    adjustments["force_adjustment"] = "increase"
                elif "decrease" in suggestion_lower or "gentle" in suggestion_lower:
                    adjustments["force_adjustment"] = "decrease"
            if "approach" in suggestion_lower or "side" in suggestion_lower:
                adjustments["approach_change"] = "side"
            if "gripper" in suggestion or "gripper" in suggestion_lower:
                if "wider" in suggestion_lower or "open" in suggestion_lower:
                    adjustments["gripper_adjustment"] = 0.01
                elif "narrow" in suggestion_lower or "close" in suggestion_lower:
                    adjustments["gripper_adjustment"] = -0.01
        
        return adjustments
    
    def get_stats(self) -> Dict[str, Any]:
    
        """get_stats function."""
        total = self.stats["total_attempts"]
        return {
            **self.stats,
            "success_rate": self.stats["successes"] / total if total > 0 else 0,
            "retry_rate": self.stats["retries"] / total if total > 0 else 0,
            "reflection_rejection_rate": self.stats["reflection_rejections"] / total if total > 0 else 0,
        }


# Note

def create_agentic_pipeline(
    vlm_client,
    retriever,
    executor=None,
    max_retries: int = 3,
    enable_reflection: bool = True,
    enable_learning: bool = True,
) -> AgenticGraspPipeline:
    """create_agentic_pipeline function."""
    config = AgenticConfig(
        max_retries=max_retries,
        enable_reflection=enable_reflection,
        enable_learning=enable_learning,
    )
    
    return AgenticGraspPipeline(
        vlm_client=vlm_client,
        retriever=retriever,
        executor=executor,
        config=config,
    )
