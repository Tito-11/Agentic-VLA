"""Core module implementation."""

import json
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .reflection import FailureType, FailureContext, FailureAnalysis, ReflectionResult


class RetryLevel(Enum):


    """RetryLevel class."""
    PARAMETER_TUNING = 1      # Note
    METHOD_SWITCHING = 2       # Note
    STRATEGY_RECONSTRUCTION = 3  # Note
    HUMAN_INTERVENTION = 4     # Note


class GraspType(Enum):


    """GraspType class."""
    PINCH = "pinch"           # Note
    WRAP = "wrap"             # Note
    POWER = "power"           # Note
    PRECISION = "precision"   # Note
    SIDE = "side"             # Note
    TOP_DOWN = "top_down"     # Note


@dataclass
class RetryAttempt:
    """RetryAttempt class."""
    attempt_number: int
    retry_level: RetryLevel
    strategy_name: str
    parameter_adjustments: Dict[str, Any]
    success: bool = False
    failure_type: Optional[FailureType] = None
    execution_time_ms: float = 0
    notes: str = ""


@dataclass
class RetryConfig:
    """RetryConfig class."""
    max_attempts: int = 3
    parameter_adjustment_ratio: float = 0.2  # Note
    force_increment: float = 2.0  # N
    width_adjustment: float = 0.005  # m
    height_adjustment: float = 0.02  # m
    speed_reduction_factor: float = 0.7
    min_success_probability: float = 0.2  # Note


@dataclass
class AdaptiveRetryResult:
    """AdaptiveRetryResult class."""
    total_attempts: int
    final_success: bool
    attempts: List[RetryAttempt]
    final_parameters: Dict[str, Any]
    total_time_ms: float
    lessons_learned: List[str]


class AdaptiveRetryPolicy:


    """AdaptiveRetryPolicy class."""
    # Note

    FAILURE_TO_STRATEGY = {
        FailureType.SLIP: [
            ("increase_force", RetryLevel.PARAMETER_TUNING),
            ("wrap_grasp", RetryLevel.METHOD_SWITCHING),
            ("surface_treatment", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
        FailureType.COLLISION: [
            ("raise_approach", RetryLevel.PARAMETER_TUNING),
            ("side_approach", RetryLevel.METHOD_SWITCHING),
            ("replan_path", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
        FailureType.WIDTH_MISMATCH: [
            ("adjust_width", RetryLevel.PARAMETER_TUNING),
            ("find_narrow_point", RetryLevel.METHOD_SWITCHING),
            ("rotate_object", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
        FailureType.FORCE_DAMAGE: [
            ("reduce_force", RetryLevel.PARAMETER_TUNING),
            ("soft_grasp", RetryLevel.METHOD_SWITCHING),
            ("alternative_region", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
        FailureType.DROP: [
            ("increase_force", RetryLevel.PARAMETER_TUNING),
            ("two_stage_grasp", RetryLevel.METHOD_SWITCHING),
            ("support_grasp", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
        FailureType.UNREACHABLE: [
            ("adjust_position", RetryLevel.PARAMETER_TUNING),
            ("alternative_point", RetryLevel.METHOD_SWITCHING),
            ("move_base", RetryLevel.STRATEGY_RECONSTRUCTION),
        ],
    }
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._attempt_history: List[RetryAttempt] = []
        
    def get_next_strategy(
        self,
        failure_type: FailureType,
        attempt_number: int,
        previous_attempts: List[RetryAttempt] = None,
    ) -> Tuple[str, RetryLevel, Dict[str, Any]]:
        """get_next_strategy function."""
        strategies = self.FAILURE_TO_STRATEGY.get(failure_type, [])
        
        if not strategies:
            return ("generic_retry", RetryLevel.PARAMETER_TUNING, {})
            
        # Note

        if attempt_number <= 1:
            # Note

            strategy_index = 0
        elif attempt_number == 2:
            # Note

            strategy_index = min(1, len(strategies) - 1)
        else:
            # Note

            strategy_index = min(2, len(strategies) - 1)
            
        strategy_name, level = strategies[strategy_index]
        
        # Note

        adjustments = self._compute_adjustments(
            strategy_name, failure_type, attempt_number, previous_attempts
        )
        
        return strategy_name, level, adjustments
    
    def _compute_adjustments(
        self,
        strategy_name: str,
        failure_type: FailureType,
        attempt_number: int,
        previous_attempts: List[RetryAttempt] = None,
    ) -> Dict[str, Any]:
        """_compute_adjustments function."""
        adjustments = {}
        ratio = self.config.parameter_adjustment_ratio * attempt_number
        
        if strategy_name == "increase_force":
            adjustments = {
                "force_delta": self.config.force_increment * attempt_number,
                "force_ratio": 1 + ratio,
            }
            
        elif strategy_name == "reduce_force":
            adjustments = {
                "force_delta": -self.config.force_increment * attempt_number,
                "force_ratio": 1 - ratio * 0.5,  # Note
            }
            
        elif strategy_name == "adjust_width":
            # Note

            adjustments = {
                "width_delta": self.config.width_adjustment,
                "remeasure": True,
            }
            
        elif strategy_name == "raise_approach":
            adjustments = {
                "height_delta": self.config.height_adjustment * attempt_number,
                "approach_angle_offset": 10 * attempt_number,  # Note
            }
            
        elif strategy_name == "wrap_grasp":
            adjustments = {
                "grasp_type": GraspType.WRAP.value,
                "width_ratio": 1.3,  # Note
                "approach_direction": "side",
            }
            
        elif strategy_name == "side_approach":
            adjustments = {
                "approach_direction": "side",
                "orientation_offset": 90,  # Note
                "height_delta": self.config.height_adjustment,
            }
            
        elif strategy_name == "soft_grasp":
            adjustments = {
                "force_control": True,
                "max_force": 5.0,  # N
                "compliance": 0.8,
            }
            
        elif strategy_name == "two_stage_grasp":
            adjustments = {
                "two_stage": True,
                "stage1_height": 0.05,  # Note
                "verify_grasp": True,
            }
            
        elif strategy_name == "find_narrow_point":
            adjustments = {
                "search_narrow": True,
                "search_radius": 0.03,  # Note
            }
            
        elif strategy_name == "alternative_point":
            adjustments = {
                "alternative_grasp_point": True,
                "priority": ["edge", "corner", "center"],
            }
            
        # Note

        adjustments["speed_factor"] = self.config.speed_reduction_factor ** attempt_number
        adjustments["verify_before_lift"] = attempt_number >= 2
        
        return adjustments


class RetryStrategy:


    """RetryStrategy class."""
    def __init__(
        self,
        reflection_module=None,
        config: Optional[RetryConfig] = None,
    ):
        from .reflection import ReflectionModule
        
        self._reflection = reflection_module or ReflectionModule()
        self._policy = AdaptiveRetryPolicy(config)
        self._config = config or RetryConfig()
        
    def should_retry(
        self,
        context: FailureContext,
        analysis: Optional[FailureAnalysis] = None,
    ) -> Tuple[bool, str]:
        """should_retry function."""
        # Note

        if context.attempt_number >= self._config.max_attempts:
            return False, f"Maximum attempts reached ({self._config.max_attempts})"
            
        # Note

        non_retryable = {
            FailureType.SENSOR_ERROR,
            FailureType.COMMUNICATION_ERROR,
        }
        if context.failure_type in non_retryable:
            return False, f"Failure type ({context.failure_type.value}) is not retryable"
            
        # Note

        if analysis:
            # Note

            success_prob = self._estimate_retry_success(context, analysis)
            if success_prob < self._config.min_success_probability:
                return False, f"Low success probability ({success_prob:.1%}), skipping retry"
                
        return True, "Retry conditions met"
    
    def plan_retry(
        self,
        context: FailureContext,
    ) -> Dict[str, Any]:
        """plan_retry function."""
        # Note

        reflection_result = self._reflection.reflect_and_decide(context)
        
        # Note

        strategy_name, level, adjustments = self._policy.get_next_strategy(
            context.failure_type,
            context.attempt_number,
        )
        
        # Note

        final_adjustments = self._merge_adjustments(
            reflection_result.adjusted_parameters,
            adjustments,
        )
        
        # Note

        new_parameters = self._apply_adjustments(
            context.grasp_pose,
            context.gripper_width,
            context.grasp_force,
            final_adjustments,
        )
        
        # Note

        success_probability = reflection_result.success_probability
        
        return {
            "should_retry": reflection_result.should_retry,
            "strategy_name": strategy_name,
            "retry_level": level.name,
            "adjustments": final_adjustments,
            "new_parameters": new_parameters,
            "success_probability": success_probability,
            "reasoning": reflection_result.analysis.root_cause,
            "lesson": reflection_result.lesson_learned,
        }
    
    def execute_retry_loop(
        self,
        initial_context: FailureContext,
        execute_grasp_fn,  # Callable[[Dict], Tuple[bool, FailureType]]
        verbose: bool = True,
    ) -> AdaptiveRetryResult:
        """execute_retry_loop function."""
        start_time = time.time()
        attempts = []
        current_context = initial_context
        lessons = []
        
        final_success = False
        final_params = {}
        
        for attempt in range(1, self._config.max_attempts + 1):
            if verbose:
                print(f"\n[Retry] Attempt {attempt} starting...")
                
            # Note

            plan = self.plan_retry(current_context)
            
            if not plan["should_retry"]:
                if verbose:
                    print(f"[Retry] Reasoning: {plan.get('reasoning', '')}")
                break
                
            if verbose:
                print(f"[Retry] Strategy: {plan['strategy_name']} (Level: {plan['retry_level']})")
                print(f"[Retry] Estimated success probability: {plan['success_probability']:.1%}")
                
            # Note

            exec_start = time.time()
            try:
                success, failure_type = execute_grasp_fn(plan["new_parameters"])
            except Exception as e:
                success = False
                failure_type = FailureType.UNKNOWN
                if verbose:
                    print(f"[Retry] Execution error: {e}")
                    
            exec_time = (time.time() - exec_start) * 1000
            
            # Note

            retry_attempt = RetryAttempt(
                attempt_number=attempt,
                retry_level=RetryLevel[plan["retry_level"]],
                strategy_name=plan["strategy_name"],
                parameter_adjustments=plan["adjustments"],
                success=success,
                failure_type=failure_type if not success else None,
                execution_time_ms=exec_time,
            )
            attempts.append(retry_attempt)
            
            if success:
                final_success = True
                final_params = plan["new_parameters"]
                lessons.append(f"Success with strategy: {plan['strategy_name']}")
                if verbose:
                    print(f"[Retry] ✓ Attempt {attempt} succeeded!")
                break
            else:
                lessons.append(plan.get("lesson", ""))
                if verbose:
                    print(f"[Retry] ✗ Failed: {failure_type.value if failure_type else 'unknown'}")
                    
                # Note

                current_context = FailureContext(
                    failure_type=failure_type or FailureType.UNKNOWN,
                    timestamp=time.time(),
                    object_name=initial_context.object_name,
                    object_category=initial_context.object_category,
                    object_properties=initial_context.object_properties,
                    grasp_pose=plan["new_parameters"].get("grasp_pose", initial_context.grasp_pose),
                    gripper_width=plan["new_parameters"].get("gripper_width", initial_context.gripper_width),
                    grasp_force=plan["new_parameters"].get("grasp_force", initial_context.grasp_force),
                    attempt_number=attempt + 1,
                    previous_failures=current_context.previous_failures + [self._reflection._failure_history[-1]],
                )
                
        total_time = (time.time() - start_time) * 1000
        
        return AdaptiveRetryResult(
            total_attempts=len(attempts),
            final_success=final_success,
            attempts=attempts,
            final_parameters=final_params,
            total_time_ms=total_time,
            lessons_learned=lessons,
        )
    
    def _estimate_retry_success(
        self,
        context: FailureContext,
        analysis: FailureAnalysis,
    ) -> float:
        """_estimate_retry_success function."""
        base_prob = 0.5
        
        # Note

        base_prob += (analysis.confidence - 0.5) * 0.3
        
        # Note

        easy_to_fix = {FailureType.SLIP, FailureType.WIDTH_MISMATCH}
        hard_to_fix = {FailureType.UNREACHABLE, FailureType.COLLISION}
        
        if context.failure_type in easy_to_fix:
            base_prob += 0.15
        elif context.failure_type in hard_to_fix:
            base_prob -= 0.1
            
        # Note

        base_prob *= (0.85 ** context.attempt_number)
        
        # Note

        if analysis.immediate_adjustments:
            base_prob += 0.1
            
        return min(max(base_prob, 0.05), 0.95)
    
    def _merge_adjustments(
        self,
        reflection_adjustments: Dict[str, Any],
        policy_adjustments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """_merge_adjustments function."""
        merged = dict(reflection_adjustments)
        
        for key, value in policy_adjustments.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, (int, float)) and isinstance(merged[key], (int, float)):
                # Note

                if "delta" in key or "offset" in key:
                    merged[key] = max(abs(merged[key]), abs(value)) * (1 if value >= 0 else -1)
                elif "ratio" in key or "factor" in key:
                    # Note

                    merged[key] = merged[key] * value
                    
        return merged
    
    def _apply_adjustments(
        self,
        original_pose: Dict[str, Any],
        original_width: float,
        original_force: float,
        adjustments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """_apply_adjustments function."""
        new_params = {
            "grasp_pose": dict(original_pose),
            "gripper_width": original_width,
            "grasp_force": original_force,
        }
        
        # Note

        if "force_delta" in adjustments:
            new_params["grasp_force"] += adjustments["force_delta"]
        if "force_ratio" in adjustments:
            new_params["grasp_force"] *= adjustments["force_ratio"]
        new_params["grasp_force"] = max(1.0, min(50.0, new_params["grasp_force"]))
        
        # Note

        if "width_delta" in adjustments:
            new_params["gripper_width"] += adjustments["width_delta"]
        if "width_ratio" in adjustments:
            new_params["gripper_width"] *= adjustments["width_ratio"]
        new_params["gripper_width"] = max(0.01, min(0.15, new_params["gripper_width"]))
        
        # Note

        if "height_delta" in adjustments:
            if "position" not in new_params["grasp_pose"]:
                new_params["grasp_pose"]["position"] = {}
            z = new_params["grasp_pose"]["position"].get("z", 0.1)
            new_params["grasp_pose"]["position"]["z"] = z + adjustments["height_delta"]
            
        # Note

        if "grasp_type" in adjustments:
            new_params["grasp_type"] = adjustments["grasp_type"]
            
        # Note

        if "approach_direction" in adjustments:
            new_params["approach_direction"] = adjustments["approach_direction"]
            
        # Note

        if "speed_factor" in adjustments:
            new_params["speed_factor"] = adjustments["speed_factor"]
            
        # Note

        if "force_control" in adjustments:
            new_params["force_control"] = adjustments["force_control"]
            
        # Note

        if "two_stage" in adjustments:
            new_params["two_stage_grasp"] = adjustments["two_stage"]
            
        return new_params


if __name__ == "__main__":
    import random
    from .reflection import ReflectionModule
    
    # Note

    config = RetryConfig(max_attempts=3)
    strategy = RetryStrategy(config=config)
    
    # Note

    def mock_execute_grasp(params: Dict[str, Any]) -> Tuple[bool, Optional[FailureType]]:

        """mock_execute_grasp function."""
        # Note

        base_success = 0.4
        
        if params.get("force_control"):
            base_success += 0.15
        if params.get("grasp_type") == "wrap":
            base_success += 0.2
        if params.get("speed_factor", 1.0) < 0.7:
            base_success += 0.1
        if params.get("verify_before_lift"):
            base_success += 0.1
            
        success = random.random() < base_success
        
        if success:
            return True, None
        else:
            # Note

            failure_types = [FailureType.SLIP, FailureType.WIDTH_MISMATCH, FailureType.DROP]
            return False, random.choice(failure_types)
    
    # Note

    initial_context = FailureContext(
        failure_type=FailureType.SLIP,
        timestamp=time.time(),
        object_name="test_cup",
        object_category="cup",
        object_properties={"material": "glass", "surface": "smooth"},
        grasp_pose={"position": {"x": 0.3, "y": 0, "z": 0.1}},
        gripper_width=0.06,
        grasp_force=10.0,
        slip_detected=True,
        attempt_number=1,
    )
    
    # Note

    print("=" * 60)
    print("benchmark")
    print("=" * 60)
    
    result = strategy.execute_retry_loop(
        initial_context=initial_context,
        execute_grasp_fn=mock_execute_grasp,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Total attempts: {result.total_attempts}")
    print(f"Success: {result.final_success}")
    print(f"total_time: {result.total_time_ms:.1f}ms")
    print(f"\nRetry results:")
    for lesson in result.lessons_learned:
        if lesson:
            print(f"  - {lesson}")
