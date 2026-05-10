"""Core module implementation."""

import json
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExperienceType(Enum):


    """ExperienceType class."""
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTED = "corrected"  # Note


class UpdateStrategy(Enum):


    """UpdateStrategy class."""
    INCREMENTAL = "incremental"  # Note
    MERGE = "merge"              # Note
    REPLACE = "replace"          # Note
    SKIP = "skip"                # Note


@dataclass
class Experience:
    """Experience class."""
    id: str
    experience_type: ExperienceType
    
    # Note

    object_name: str
    object_category: str
    object_properties: Dict[str, Any]
    
    # Note

    grasp_pose: Dict[str, Any]
    gripper_width: float
    grasp_force: float
    
    # Note

    success: bool
    failure_type: Optional[str] = None
    failure_cause: Optional[str] = None
    
    # Note

    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)
    corrected_params: Optional[Dict[str, Any]] = None
    
    # Note

    timestamp: float = field(default_factory=time.time)
    weight: float = 1.0
    usage_count: int = 0
    last_used: Optional[float] = None
    
    # Note

    lessons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
    
        """to_dict function."""
        return {
            "id": self.id,
            "experience_type": self.experience_type.value,
            "object_name": self.object_name,
            "object_category": self.object_category,
            "object_properties": self.object_properties,
            "grasp_pose": self.grasp_pose,
            "gripper_width": self.gripper_width,
            "grasp_force": self.grasp_force,
            "success": self.success,
            "failure_type": self.failure_type,
            "failure_cause": self.failure_cause,
            "suggested_adjustments": self.suggested_adjustments,
            "corrected_params": self.corrected_params,
            "timestamp": self.timestamp,
            "weight": self.weight,
            "usage_count": self.usage_count,
            "lessons": self.lessons,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """from_dict function."""
        return cls(
            id=data["id"],
            experience_type=ExperienceType(data.get("experience_type", "success")),
            object_name=data["object_name"],
            object_category=data["object_category"],
            object_properties=data.get("object_properties", {}),
            grasp_pose=data.get("grasp_pose", {}),
            gripper_width=data.get("gripper_width", 0.06),
            grasp_force=data.get("grasp_force", 10.0),
            success=data.get("success", True),
            failure_type=data.get("failure_type"),
            failure_cause=data.get("failure_cause"),
            suggested_adjustments=data.get("suggested_adjustments", {}),
            corrected_params=data.get("corrected_params"),
            timestamp=data.get("timestamp", time.time()),
            weight=data.get("weight", 1.0),
            usage_count=data.get("usage_count", 0),
            lessons=data.get("lessons", []),
        )


@dataclass
class LearningConfig:
    """LearningConfig class."""
    # Note

    similarity_threshold: float = 0.85
    
    # Note

    success_weight_boost: float = 0.2   # Note
    failure_weight_penalty: float = 0.1  # Note
    correction_weight_boost: float = 0.3  # Note
    
    # Note

    time_decay_factor: float = 0.99     # Note
    time_decay_days: int = 30           # Note
    
    # Note

    max_similar_experiences: int = 5     # Note
    
    # Note

    min_weight_to_keep: float = 0.1     # Note


@dataclass
class LearningResult:
    """LearningResult class."""
    experience_id: str
    update_strategy: UpdateStrategy
    weight_change: float
    merged_with: Optional[str] = None
    lessons_extracted: List[str] = field(default_factory=list)
    knowledge_updated: bool = False


class ExperienceLearner:


    """ExperienceLearner class."""
    def __init__(
        self,
        vector_store=None,
        embedding_model=None,
        config: Optional[LearningConfig] = None,
    ):
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._config = config or LearningConfig()
        
        # Note

        self._experience_cache: Dict[str, Experience] = {}
        self._category_rules: Dict[str, List[Dict[str, Any]]] = {}
        
    def learn_from_success(
        self,
        object_name: str,
        object_category: str,
        object_properties: Dict[str, Any],
        grasp_pose: Dict[str, Any],
        gripper_width: float,
        grasp_force: float,
        retrieval_helped: bool = True,
        scene_image=None,
    ) -> LearningResult:
        """learn_from_success function."""
        # Note

        exp_id = self._generate_experience_id(object_name, object_category)
        
        experience = Experience(
            id=exp_id,
            experience_type=ExperienceType.SUCCESS,
            object_name=object_name,
            object_category=object_category,
            object_properties=object_properties,
            grasp_pose=grasp_pose,
            gripper_width=gripper_width,
            grasp_force=grasp_force,
            success=True,
            weight=1.0 + self._config.success_weight_boost,
        )
        
        # Note

        similar_exp = self._find_similar_experience(experience)
        
        # Note

        if similar_exp:
            if self._should_merge(experience, similar_exp):
                # Note

                merged_exp = self._merge_experiences(similar_exp, experience)
                update_strategy = UpdateStrategy.MERGE
                result_id = similar_exp.id
            else:
                # Note

                update_strategy = UpdateStrategy.INCREMENTAL
                result_id = exp_id
        else:
            update_strategy = UpdateStrategy.INCREMENTAL
            result_id = exp_id
            
        # Note

        if update_strategy == UpdateStrategy.MERGE:
            self._update_experience(merged_exp)
        else:
            self._add_experience(experience)
            
        # Note

        lessons = self._extract_lessons_from_success(experience, retrieval_helped)
        
        # Note

        self._update_category_rules(object_category, experience)
        
        return LearningResult(
            experience_id=result_id,
            update_strategy=update_strategy,
            weight_change=self._config.success_weight_boost,
            merged_with=similar_exp.id if update_strategy == UpdateStrategy.MERGE else None,
            lessons_extracted=lessons,
            knowledge_updated=True,
        )
    
    def learn_from_failure(
        self,
        object_name: str,
        object_category: str,
        object_properties: Dict[str, Any],
        grasp_pose: Dict[str, Any],
        gripper_width: float,
        grasp_force: float,
        failure_type: str,
        failure_cause: str,
        suggested_adjustments: Dict[str, Any],
        scene_image=None,
    ) -> LearningResult:
        """learn_from_failure function."""
        # Note

        exp_id = self._generate_experience_id(
            object_name, object_category, suffix="fail"
        )
        
        experience = Experience(
            id=exp_id,
            experience_type=ExperienceType.FAILURE,
            object_name=object_name,
            object_category=object_category,
            object_properties=object_properties,
            grasp_pose=grasp_pose,
            gripper_width=gripper_width,
            grasp_force=grasp_force,
            success=False,
            failure_type=failure_type,
            failure_cause=failure_cause,
            suggested_adjustments=suggested_adjustments,
            weight=0.5,  # Note
        )
        
        # Note

        lessons = self._extract_lessons_from_failure(experience)
        experience.lessons = lessons
        
        # Note

        similar_success = self._find_similar_success_experience(experience)
        
        if similar_success:
            # Note

            self._add_caution_to_experience(similar_success, failure_cause, suggested_adjustments)
            update_strategy = UpdateStrategy.MERGE
        else:
            # Note

            self._add_experience(experience)
            update_strategy = UpdateStrategy.INCREMENTAL
            
        return LearningResult(
            experience_id=exp_id,
            update_strategy=update_strategy,
            weight_change=-self._config.failure_weight_penalty,
            lessons_extracted=lessons,
            knowledge_updated=True,
        )
    
    def learn_from_correction(
        self,
        original_failure: Experience,
        corrected_params: Dict[str, Any],
        final_success: bool,
    ) -> LearningResult:
        """learn_from_correction function."""
        if not final_success:
            return LearningResult(
                experience_id=original_failure.id,
                update_strategy=UpdateStrategy.SKIP,
                weight_change=0,
                knowledge_updated=False,
            )
            
        # Note

        exp_id = self._generate_experience_id(
            original_failure.object_name,
            original_failure.object_category,
            suffix="corrected"
        )
        
        # Note

        param_changes = self._compute_param_changes(
            original_failure.grasp_pose,
            original_failure.gripper_width,
            original_failure.grasp_force,
            corrected_params,
        )
        
        experience = Experience(
            id=exp_id,
            experience_type=ExperienceType.CORRECTED,
            object_name=original_failure.object_name,
            object_category=original_failure.object_category,
            object_properties=original_failure.object_properties,
            grasp_pose=corrected_params.get("grasp_pose", original_failure.grasp_pose),
            gripper_width=corrected_params.get("gripper_width", original_failure.gripper_width),
            grasp_force=corrected_params.get("grasp_force", original_failure.grasp_force),
            success=True,
            failure_type=original_failure.failure_type,
            failure_cause=original_failure.failure_cause,
            suggested_adjustments=original_failure.suggested_adjustments,
            corrected_params=corrected_params,
            weight=1.0 + self._config.correction_weight_boost,  # Note
        )
        
        # Note

        lessons = [
            f"Object '{original_failure.object_name}' failure type: '{original_failure.failure_type}'",
            f"Parameter changes: {json.dumps(param_changes, ensure_ascii=False)}",
            f"Original failure: {original_failure.failure_cause}",
        ]
        experience.lessons = lessons
        
        # Note

        self._add_experience(experience)
        
        # Note

        self._update_category_rules(
            original_failure.object_category,
            experience,
            priority=2,  # Note
        )
        
        return LearningResult(
            experience_id=exp_id,
            update_strategy=UpdateStrategy.INCREMENTAL,
            weight_change=self._config.correction_weight_boost,
            lessons_extracted=lessons,
            knowledge_updated=True,
        )
    
    def get_category_rules(self, category: str) -> List[Dict[str, Any]]:
    
        """get_category_rules function."""
        return self._category_rules.get(category, [])
    
    def decay_old_experiences(self) -> int:
    
        """decay_old_experiences function."""
        updated_count = 0
        current_time = time.time()
        decay_threshold = self._config.time_decay_days * 24 * 3600
        
        for exp_id, exp in self._experience_cache.items():
            age = current_time - exp.timestamp
            if age > decay_threshold:
                # Note

                decay_days = (age - decay_threshold) / (24 * 3600)
                decay_factor = self._config.time_decay_factor ** decay_days
                new_weight = exp.weight * decay_factor
                
                if new_weight < self._config.min_weight_to_keep:
                    # Note

                    exp.weight = 0
                else:
                    exp.weight = new_weight
                    
                updated_count += 1
                
        return updated_count
    
    def cleanup_low_weight_experiences(self) -> int:
    
        """cleanup_low_weight_experiences function."""
        to_remove = [
            exp_id for exp_id, exp in self._experience_cache.items()
            if exp.weight < self._config.min_weight_to_keep
        ]
        
        for exp_id in to_remove:
            del self._experience_cache[exp_id]
            # Note

            if self._vector_store:
                self._vector_store.delete(exp_id)
                
        return len(to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
    
        """get_statistics function."""
        if not self._experience_cache:
            return {
                "total_experiences": 0,
                "success_count": 0,
                "failure_count": 0,
                "corrected_count": 0,
            }
            
        success_count = sum(
            1 for exp in self._experience_cache.values()
            if exp.experience_type == ExperienceType.SUCCESS
        )
        failure_count = sum(
            1 for exp in self._experience_cache.values()
            if exp.experience_type == ExperienceType.FAILURE
        )
        corrected_count = sum(
            1 for exp in self._experience_cache.values()
            if exp.experience_type == ExperienceType.CORRECTED
        )
        
        avg_weight = sum(exp.weight for exp in self._experience_cache.values()) / len(self._experience_cache)
        
        categories = set(exp.object_category for exp in self._experience_cache.values())
        
        return {
            "total_experiences": len(self._experience_cache),
            "success_count": success_count,
            "failure_count": failure_count,
            "corrected_count": corrected_count,
            "average_weight": avg_weight,
            "categories": list(categories),
            "category_rules_count": sum(len(rules) for rules in self._category_rules.values()),
        }
    
    # Note

    
    def _generate_experience_id(
        self,
        object_name: str,
        category: str,
        suffix: str = "",
    ) -> str:
        """_generate_experience_id function."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        content = f"{object_name}_{category}_{timestamp}_{suffix}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"exp_{hash_suffix}"
    
    def _find_similar_experience(
        self,
        experience: Experience,
    ) -> Optional[Experience]:
        """_find_similar_experience function."""
        if not self._experience_cache:
            return None
            
        for exp in self._experience_cache.values():
            if exp.object_category != experience.object_category:
                continue
            if not exp.success:
                continue
                
            # Note

            similarity = self._compute_experience_similarity(exp, experience)
            if similarity >= self._config.similarity_threshold:
                return exp
                
        return None
    
    def _find_similar_success_experience(
        self,
        failure_exp: Experience,
    ) -> Optional[Experience]:
        """_find_similar_success_experience function."""
        for exp in self._experience_cache.values():
            if exp.object_category != failure_exp.object_category:
                continue
            if not exp.success:
                continue
            if exp.object_name == failure_exp.object_name:
                return exp
                
        return None
    
    def _compute_experience_similarity(
        self,
        exp1: Experience,
        exp2: Experience,
    ) -> float:
        """_compute_experience_similarity function."""
        # Note

        category_match = 1.0 if exp1.object_category == exp2.object_category else 0.0
        
        # Note

        name_match = 1.0 if exp1.object_name == exp2.object_name else (
            0.5 if any(w in exp2.object_name for w in exp1.object_name.split()) else 0.0
        )
        
        # Note

        props1 = set(exp1.object_properties.keys())
        props2 = set(exp2.object_properties.keys())
        common_props = props1 & props2
        prop_match = len(common_props) / max(len(props1 | props2), 1)
        
        # Note

        similarity = 0.3 * category_match + 0.4 * name_match + 0.3 * prop_match
        
        return similarity
    
    def _should_merge(
        self,
        new_exp: Experience,
        existing_exp: Experience,
    ) -> bool:
        """_should_merge function."""
        # Note

        if (new_exp.object_name == existing_exp.object_name and
            new_exp.success and existing_exp.success): return True
            
        return False
    
    def _merge_experiences(
        self,
        existing: Experience,
        new_exp: Experience,
    ) -> Experience:
        """_merge_experiences function."""
        # Note

        new_weight = max(existing.weight, new_exp.weight) + self._config.success_weight_boost / 2
        
        # Note

        new_usage = existing.usage_count + 1
        
        # Note

        combined_lessons = list(set(existing.lessons + new_exp.lessons))
        
        # Note

        merged = Experience(
            id=existing.id,
            experience_type=ExperienceType.SUCCESS,
            object_name=existing.object_name,
            object_category=existing.object_category,
            object_properties={**existing.object_properties, **new_exp.object_properties},
            grasp_pose=new_exp.grasp_pose,  # Note
            gripper_width=new_exp.gripper_width,
            grasp_force=new_exp.grasp_force,
            success=True,
            weight=new_weight,
            usage_count=new_usage,
            last_used=time.time(),
            lessons=combined_lessons,
        )
        
        return merged
    
    def _add_experience(self, experience: Experience) -> None:
    
        """_add_experience function."""
        self._experience_cache[experience.id] = experience
        
        if self._vector_store:
            # Note

            text = f"{experience.object_name} {experience.object_category} " + \
                   " ".join(str(v) for v in experience.object_properties.values())
            
            metadata = experience.to_dict()
            metadata["text"] = text
            
            # Note

            # self._vector_store.add(experience.id, text, metadata)
    
    def _update_experience(self, experience: Experience) -> None:
    
        """_update_experience function."""
        self._experience_cache[experience.id] = experience
        
    def _add_caution_to_experience(
        self,
        experience: Experience,
        failure_cause: str,
        adjustments: Dict[str, Any],
    ) -> None:
        """_add_caution_to_experience function."""
        caution = f"Failure cause: {failure_cause}, Adjustments: {json.dumps(adjustments, ensure_ascii=False)}"
        if caution not in experience.lessons:
            experience.lessons.append(caution)
    
    def _extract_lessons_from_success(
        self,
        experience: Experience,
        retrieval_helped: bool,
    ) -> List[str]:
        """_extract_lessons_from_success function."""
        lessons = []
        
        lessons.append(
            f"object'{experience.object_name}'({experience.object_category}): "
            f"Width: {experience.gripper_width:.3f}m, Force: {experience.grasp_force:.1f}N"
        )
        
        if retrieval_helped:
            lessons.append("Leveraged RAG retrieval for experience-based learning")
            
        # Note

        if "material" in experience.object_properties:
            material = experience.object_properties["material"]
            lessons.append(f"{material}objectsENargsEN")
            
        return lessons
    
    def _extract_lessons_from_failure(self, experience: Experience) -> List[str]:
    
        """_extract_lessons_from_failure function."""
        lessons = []
        
        lessons.append(
            f"Failure cause: {experience.failure_cause}"
        )
        
        if experience.suggested_adjustments:
            adj_str = json.dumps(experience.suggested_adjustments, ensure_ascii=False)
            lessons.append(f"Adjustments: {adj_str}")
            
        # Note

        failure_rules = {
            "slip": "objectsENgrasp",
            "collision": "Adjust planning path to avoid collision",
            "width_mismatch": "Re-estimate object width for grasp",
            "force_damage": "Reduce force to avoid object damage",
        }
        
        if experience.failure_type in failure_rules:
            lessons.append(failure_rules[experience.failure_type])
            
        return lessons
    
    def _compute_param_changes(
        self,
        original_pose: Dict[str, Any],
        original_width: float,
        original_force: float,
        corrected_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """_compute_param_changes function."""
        changes = {}
        
        new_width = corrected_params.get("gripper_width", original_width)
        if new_width != original_width:
            changes["width_change"] = new_width - original_width
            changes["width_ratio"] = new_width / original_width
            
        new_force = corrected_params.get("grasp_force", original_force)
        if new_force != original_force:
            changes["force_change"] = new_force - original_force
            changes["force_ratio"] = new_force / original_force
            
        new_pose = corrected_params.get("grasp_pose", original_pose)
        if new_pose != original_pose:
            changes["pose_changed"] = True
            
        return changes
    
    def _update_category_rules(
        self,
        category: str,
        experience: Experience,
        priority: int = 1,
    ) -> None:
        """_update_category_rules function."""
        if category not in self._category_rules:
            self._category_rules[category] = []
            
        # Note

        rule = {
            "category": category,
            "source_experience": experience.id,
            "recommended_width_range": (
                experience.gripper_width * 0.8,
                experience.gripper_width * 1.2
            ),
            "recommended_force_range": (
                experience.grasp_force * 0.8,
                experience.grasp_force * 1.2
            ),
            "priority": priority,
            "timestamp": time.time(),
        }
        
        # Note

        existing = None
        for i, r in enumerate(self._category_rules[category]):
            if r["source_experience"] == experience.id:
                existing = i
                break
                
        if existing is not None:
            self._category_rules[category][existing] = rule
        else:
            self._category_rules[category].append(rule)
            
        # Note

        self._category_rules[category].sort(key=lambda x: -x["priority"])
        
        # Note

        if len(self._category_rules[category]) > self._config.max_similar_experiences:
            self._category_rules[category] = self._category_rules[category][:self._config.max_similar_experiences]


if __name__ == "__main__":
    # Note

    learner = ExperienceLearner()
    
    print("=" * 60)
    print("Experience Learning Benchmark")
    print("=" * 60)
    
    # Note

    print("\n1. Storing successful experience...")
    result1 = learner.learn_from_success(
        object_name="ceramic_cup",
        object_category="cup",
        object_properties={"material": "glass", "has_handle": True},
        grasp_pose={"position": {"x": 0.3, "y": 0, "z": 0.1}},
        gripper_width=0.06,
        grasp_force=12.0,
        retrieval_helped=True,
    )
    print(f"   result: {result1.update_strategy.value}")
    print(f"   Weight change: +{result1.weight_change}")
    print(f"   Lessons: {result1.lessons_extracted}")
    
    # Note

    print("\n2. Storing failure experience...")
    result2 = learner.learn_from_failure(
        object_name="plastic_bottle",
        object_category="bottle",
        object_properties={"material": "plastic", "surface": "smooth"},
        grasp_pose={"position": {"x": 0.35, "y": 0, "z": 0.12}},
        gripper_width=0.05,
        grasp_force=8.0,
        failure_type="slip",
        failure_cause="Object slipped during test",
        suggested_adjustments={"force_ratio": 1.3, "grasp_type": "wrap"},
    )
    print(f"   result: {result2.update_strategy.value}")
    print(f"   Weight change: {result2.weight_change}")
    print(f"   Lessons: {result2.lessons_extracted}")
    
    # Note

    print("\n3. Storing another successful experience...")
    from dataclasses import replace
    failure_exp = Experience(
        id="exp_test",
        experience_type=ExperienceType.FAILURE,
        object_name="plastic_bottle",
        object_category="bottle",
        object_properties={"material": "plastic"},
        grasp_pose={"position": {"x": 0.35, "y": 0, "z": 0.12}},
        gripper_width=0.05,
        grasp_force=8.0,
        success=False,
        failure_type="slip",
        failure_cause="Object contact lost",
    )
    
    result3 = learner.learn_from_correction(
        original_failure=failure_exp,
        corrected_params={
            "gripper_width": 0.065,
            "grasp_force": 12.0,
            "grasp_type": "wrap",
        },
        final_success=True,
    )
    print(f"   result: {result3.update_strategy.value}")
    print(f"   Weight change: +{result3.weight_change}")
    print(f"   Lessons: {result3.lessons_extracted}")
    
    # Note

    print("\n4. Statistics:")
    stats = learner.get_statistics()
    print(f"   Total experiences: {stats['total_experiences']}")
    print(f"   Success: {stats['success_count']}, Failure: {stats['failure_count']}, Corrected: {stats['corrected_count']}")
    print(f"   Average weight: {stats['average_weight']:.2f}")
    
    # Note

    print("\n5. Category rules:")
    for category in ["cup", "bottle"]:
        rules = learner.get_category_rules(category)
        print(f"   {category}: {len(rules)} rules")
        for rule in rules[:2]:
            print(f"      - Width range: {rule['recommended_width_range']}")
