"""Core module implementation."""

import json
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from .react_engine import ReActEngine, ReActTrace, ToolRegistry
from .reflection import ReflectionModule, FailureType, FailureContext, FailureAnalysis
from .retry_strategy import RetryStrategy, RetryConfig, AdaptiveRetryResult
from .experience_learner import ExperienceLearner, LearningConfig, Experience, ExperienceType


class AgenticGraspState(Enum):


    """AgenticGraspState class."""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    RETRIEVING = "retrieving"
    PLANNING = "planning"
    SIMULATING = "simulating"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RETRYING = "retrying"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GraspAttempt:
    """GraspAttempt class."""
    attempt_number: int
    state_sequence: List[AgenticGraspState]
    grasp_params: Dict[str, Any]
    success: bool
    failure_type: Optional[FailureType] = None
    reflection: Optional[FailureAnalysis] = None
    execution_time_ms: float = 0
    

@dataclass
class AgenticGraspResult:
    """AgenticGraspResult class."""
    task: str
    object_name: str
    object_category: str
    
    # Note

    total_attempts: int
    final_success: bool
    final_params: Dict[str, Any]
    
    # Note

    attempts: List[GraspAttempt]
    react_trace: Optional[ReActTrace] = None
    retry_result: Optional[AdaptiveRetryResult] = None
    
    # Note

    total_time_ms: float = 0
    planning_time_ms: float = 0
    execution_time_ms: float = 0
    reflection_time_ms: float = 0
    
    # Note

    lessons_learned: List[str] = field(default_factory=list)
    knowledge_updated: bool = False


@dataclass 
class AgenticGraspConfig:
    """AgenticGraspConfig class."""
    # Note

    max_react_steps: int = 6
    
    # Note

    max_retry_attempts: int = 3
    min_success_probability: float = 0.2
    
    # Note

    auto_learn: bool = True
    learn_from_failures: bool = True
    
    # Note

    simulate_before_execute: bool = True
    simulation_confidence_threshold: float = 0.7
    
    # Note

    verbose: bool = True


class AgenticGraspAgent:


    """AgenticGraspAgent class."""
    def __init__(
        self,
        vlm_engine=None,
        retriever=None,
        config: Optional[AgenticGraspConfig] = None,
    ):
        self._config = config or AgenticGraspConfig()
        self._vlm_engine = vlm_engine
        self._retriever = retriever
        
        # Note

        self._react_engine = ReActEngine(
            vlm_engine=vlm_engine,
            max_steps=self._config.max_react_steps,
        )
        if retriever:
            self._react_engine.set_retriever(retriever)
            
        self._reflection = ReflectionModule(
            vlm_engine=vlm_engine,
            max_retries=self._config.max_retry_attempts,
        )
        
        self._retry_strategy = RetryStrategy(
            reflection_module=self._reflection,
            config=RetryConfig(
                max_attempts=self._config.max_retry_attempts,
                min_success_probability=self._config.min_success_probability,
            ),
        )
        
        self._learner = ExperienceLearner()
        
        # Note

        self._current_state = AgenticGraspState.IDLE
        self._state_history: List[Tuple[AgenticGraspState, float]] = []
        
    @property
    def state(self) -> AgenticGraspState:
        return self._current_state
    
    def _set_state(self, state: AgenticGraspState) -> None:
    
        """_set_state function."""
        self._state_history.append((state, time.time()))
        self._current_state = state
        if self._config.verbose:
            print(f"[Agent] State: {state.value}")
    
    def plan_and_execute(
        self,
        task: str,
        object_name: str,
        object_category: str,
        object_properties: Dict[str, Any] = None,
        execute_fn: Callable[[Dict], Tuple[bool, Optional[FailureType]]] = None,
        scene_image=None,
    ) -> AgenticGraspResult:
        """plan_and_execute function."""
        start_time = time.time()
        object_properties = object_properties or {}
        attempts = []
        
        self._set_state(AgenticGraspState.IDLE)
        
        if self._config.verbose:
            print("=" * 60)
            print(f"[Agent] task: {task}")
            print(f"[Agent] target: {object_name} ({object_category})")
            print("=" * 60)
        
        # Note

        self._set_state(AgenticGraspState.PLANNING)
        planning_start = time.time()
        
        react_task = f"Plan a grasp for {object_name} ({object_category}). Task: {task}"
        react_trace = self._react_engine.run(
            task=react_task,
            context={"object_properties": object_properties},
        )
        
        planning_time = (time.time() - planning_start) * 1000
        
        # Note

        initial_params = self._parse_react_result(react_trace)
        
        if self._config.verbose:
            print(f"\n[Agent] ReAct planningdone ({planning_time:.0f}ms)")
            print(f"[Agent] Initial params: {json.dumps(initial_params, ensure_ascii=False, indent=2)}")
        
        # Note

        if self._config.simulate_before_execute:
            self._set_state(AgenticGraspState.SIMULATING)
            simulation_ok = self._simulate_grasp(initial_params, object_properties)
            
            if not simulation_ok:
                if self._config.verbose:
                    print("[Agent] Retrying with adjusted params...")
                initial_params = self._adjust_params_after_simulation(initial_params)
        
        # Note

        final_success = False
        final_params = initial_params
        current_params = initial_params
        execution_time = 0
        reflection_time = 0
        lessons = []
        
        for attempt_num in range(1, self._config.max_retry_attempts + 1):
            self._set_state(AgenticGraspState.EXECUTING)
            exec_start = time.time()
            
            if self._config.verbose:
                print(f"\n[Agent] Attempt {attempt_num} executing...")
            
            # Note

            if execute_fn:
                success, failure_type = execute_fn(current_params)
            else:
                # Note

                success, failure_type = self._mock_execute(current_params, attempt_num)
            
            exec_time = (time.time() - exec_start) * 1000
            execution_time += exec_time
            
            # Note

            attempt = GraspAttempt(
                attempt_number=attempt_num,
                state_sequence=self._get_state_sequence_for_attempt(),
                grasp_params=current_params.copy(),
                success=success,
                failure_type=failure_type if not success else None,
                execution_time_ms=exec_time,
            )
            
            if success:
                final_success = True
                final_params = current_params
                attempts.append(attempt)
                
                if self._config.verbose:
                    print(f"[Agent] ✓ graspsuccess!")
                    
                # Note

                if self._config.auto_learn:
                    self._set_state(AgenticGraspState.LEARNING)
                    learn_result = self._learner.learn_from_success(
                        object_name=object_name,
                        object_category=object_category,
                        object_properties=object_properties,
                        grasp_pose=current_params.get("grasp_pose", {}),
                        gripper_width=current_params.get("gripper_width", 0.06),
                        grasp_force=current_params.get("grasp_force", 10.0),
                        retrieval_helped=len(react_trace.steps) > 0,
                    )
                    lessons.extend(learn_result.lessons_extracted)
                    
                break
            else:
                if self._config.verbose:
                    print(f"[Agent] ✗ graspfailure: {failure_type.value if failure_type else 'unknown'}")
                
                # Note

                self._set_state(AgenticGraspState.REFLECTING)
                reflect_start = time.time()
                
                # Note

                failure_context = FailureContext(
                    failure_type=failure_type or FailureType.UNKNOWN,
                    timestamp=time.time(),
                    object_name=object_name,
                    object_category=object_category,
                    object_properties=object_properties,
                    grasp_pose=current_params.get("grasp_pose", {}),
                    gripper_width=current_params.get("gripper_width", 0.06),
                    grasp_force=current_params.get("grasp_force", 10.0),
                    attempt_number=attempt_num,
                )
                
                # Note

                reflection_result = self._reflection.reflect_and_decide(failure_context)
                
                reflect_time = (time.time() - reflect_start) * 1000
                reflection_time += reflect_time
                
                attempt.reflection = reflection_result.analysis
                attempts.append(attempt)
                
                if self._config.verbose:
                    print(f"[Agent] Reflection ({reflect_time:.0f}ms):")
                    print(f"        Root cause: {reflection_result.analysis.root_cause}")
                    print(f"        Should retry: {reflection_result.should_retry}")
                    print(f"        Success probability: {reflection_result.success_probability:.1%}")
                
                # Note

                if self._config.auto_learn and self._config.learn_from_failures:
                    learn_result = self._learner.learn_from_failure(
                        object_name=object_name,
                        object_category=object_category,
                        object_properties=object_properties,
                        grasp_pose=current_params.get("grasp_pose", {}),
                        gripper_width=current_params.get("gripper_width", 0.06),
                        grasp_force=current_params.get("grasp_force", 10.0),
                        failure_type=failure_type.value if failure_type else "unknown",
                        failure_cause=reflection_result.analysis.root_cause,
                        suggested_adjustments=reflection_result.adjusted_parameters,
                    )
                    lessons.extend(learn_result.lessons_extracted)
                
                # Note

                if not reflection_result.should_retry:
                    if self._config.verbose:
                        print("[Agent] Maximum retries reached")
                    break
                    
                # Note

                self._set_state(AgenticGraspState.RETRYING)
                current_params = self._apply_adjustments(
                    current_params,
                    reflection_result.adjusted_parameters,
                )
                
                if self._config.verbose:
                    print(f"[Agent] Adjusted params: {json.dumps(current_params, ensure_ascii=False)}")
        
        # Note

        if final_success:
            self._set_state(AgenticGraspState.COMPLETED)
        else:
            self._set_state(AgenticGraspState.FAILED)
            
        total_time = (time.time() - start_time) * 1000
        
        result = AgenticGraspResult(
            task=task,
            object_name=object_name,
            object_category=object_category,
            total_attempts=len(attempts),
            final_success=final_success,
            final_params=final_params,
            attempts=attempts,
            react_trace=react_trace,
            total_time_ms=total_time,
            planning_time_ms=planning_time,
            execution_time_ms=execution_time,
            reflection_time_ms=reflection_time,
            lessons_learned=lessons,
            knowledge_updated=self._config.auto_learn,
        )
        
        if self._config.verbose:
            print("\n" + "=" * 60)
            print("[Agent] taskdone")
            print(f"[Agent] success: {final_success}")
            print(f"[Agent] Total attempts: {len(attempts)}")
            print(f"[Agent] total_time: {total_time:.0f}ms")
            print("=" * 60)
            
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
    
        """get_statistics function."""
        learner_stats = self._learner.get_statistics()
        reflection_stats = self._reflection.get_failure_statistics()
        
        return {
            **learner_stats,
            **reflection_stats,
            "state_history_length": len(self._state_history),
        }
    
    # Note

    
    def _parse_react_result(self, trace: ReActTrace) -> Dict[str, Any]:

    
        """_parse_react_result function."""
        params = {
            "grasp_pose": {"position": {"x": 0.3, "y": 0, "z": 0.1}},
            "gripper_width": 0.06,
            "grasp_force": 10.0,
            "approach_direction": "top_down",
        }
        
        if trace.final_answer:
            try:
                # Note

                answer = trace.final_answer
                if "{" in answer:
                    json_str = answer[answer.find("{"):answer.rfind("}")+1]
                    parsed = json.loads(json_str)
                    
                    if "grasp_pose" in parsed:
                        params["grasp_pose"] = parsed["grasp_pose"]
                    if "gripper_width" in parsed:
                        params["gripper_width"] = parsed["gripper_width"]
                    if "confidence" in parsed:
                        params["confidence"] = parsed["confidence"]
            except json.JSONDecodeError:
                pass
                
        return params
    
    def _simulate_grasp(
        self,
        params: Dict[str, Any],
        object_properties: Dict[str, Any],
    ) -> bool:
        """_simulate_grasp function."""
        # Note

        confidence = params.get("confidence", 0.5)
        return confidence >= self._config.simulation_confidence_threshold
    
    def _adjust_params_after_simulation(
        self,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """_adjust_params_after_simulation function."""
        adjusted = params.copy()
        
        # Note

        if "grasp_pose" in adjusted and "position" in adjusted["grasp_pose"]:
            adjusted["grasp_pose"]["position"]["z"] = \
                adjusted["grasp_pose"]["position"].get("z", 0.1) + 0.02
                
        # Note

        adjusted["grasp_force"] = adjusted.get("grasp_force", 10.0) * 1.2
        
        return adjusted
    
    def _mock_execute(
        self,
        params: Dict[str, Any],
        attempt: int,
    ) -> Tuple[bool, Optional[FailureType]]:
        """_mock_execute function."""
        import random
        
        # Note

        base_success = 0.4
        
        # Note

        if params.get("force_control"):
            base_success += 0.15
            
        # Note

        if params.get("verify_before_lift"):
            base_success += 0.1
            
        # Note

        if params.get("two_stage_grasp"):
            base_success += 0.15
            
        # Note

        base_success += 0.1 * (attempt - 1)
        
        success = random.random() < base_success
        
        if success:
            return True, None
        else:
            failure_types = [
                FailureType.SLIP,
                FailureType.WIDTH_MISMATCH,
                FailureType.COLLISION,
                FailureType.DROP,
            ]
            weights = [0.4, 0.3, 0.2, 0.1]
            failure = random.choices(failure_types, weights=weights)[0]
            return False, failure
    
    def _apply_adjustments(
        self,
        current_params: Dict[str, Any],
        adjustments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """_apply_adjustments function."""
        new_params = current_params.copy()
        
        for key, value in adjustments.items():
            if key in new_params:
                if isinstance(value, (int, float)) and isinstance(new_params[key], (int, float)):
                    new_params[key] = value
                elif isinstance(value, dict) and isinstance(new_params[key], dict):
                    new_params[key] = {**new_params[key], **value}
                else:
                    new_params[key] = value
            else:
                new_params[key] = value
                
        return new_params
    
    def _get_state_sequence_for_attempt(self) -> List[AgenticGraspState]:
    
        """_get_state_sequence_for_attempt function."""
        # Note

        states = []
        for state, _ in reversed(self._state_history):
            states.insert(0, state)
            if state in (AgenticGraspState.IDLE, AgenticGraspState.RETRYING):
                break
        return states


def run_agentic_grasp_demo():


    """run_agentic_grasp_demo function."""
    print("\n" + "=" * 70)
    print("              Agentic Grasp Planning System Demo")
    print("=" * 70 + "\n")
    
    # Note

    config = AgenticGraspConfig(
        max_retry_attempts=3,
        verbose=True,
        auto_learn=True,
    )
    
    agent = AgenticGraspAgent(config=config)
    
    # Note

    tasks = [
        {
            "task": "Grasp the target object",
            "object_name": "ceramic_mug",
            "object_category": "cup",
            "object_properties": {"material": "glass", "has_handle": True, "fragile": True},
        },
        {
            "task": "Grasp the tool",
            "object_name": "screwdriver",
            "object_category": "tool",
            "object_properties": {"material": "metal", "shape": "elongated"},
        },
    ]
    
    results = []
    
    for i, task_info in enumerate(tasks):
        print(f"\n{'='*70}")
        print(f"                      testtask {i+1}/{len(tasks)}")
        print(f"{'='*70}\n")
        
        result = agent.plan_and_execute(**task_info)
        results.append(result)
        
        print(f"\ntask {i+1} done:")
        print(f"  - success: {result.final_success}")
        print(f"  - Total attempts: {result.total_attempts}")
        print(f"  - Lessons: {result.lessons_learned[:2]}")
    
    # Note

    print("\n" + "=" * 70)
    print("                          Agent Demo Complete")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r.final_success)
    total_attempts = sum(r.total_attempts for r in results)
    avg_time = sum(r.total_time_ms for r in results) / len(results)
    
    print(f"Success rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"Average attempts: {total_attempts/len(results):.1f}")
    print(f"Average time: {avg_time:.0f}ms")
    
    stats = agent.get_statistics()
    print(f"\nExperience stats:")
    print(f"  - Total experiences: {stats.get('total_experiences', 0)}")
    print(f"  - success: {stats.get('success_count', 0)}")
    print(f"  - failure: {stats.get('failure_count', 0)}")
    
    return results


if __name__ == "__main__":
    run_agentic_grasp_demo()
