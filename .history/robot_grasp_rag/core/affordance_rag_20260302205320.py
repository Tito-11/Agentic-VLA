"""Core module implementation."""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image


# ============================================================
# Note

# ============================================================

class AffordanceType(str, Enum):

    """AffordanceType class."""
    # Note

    GRASPABLE_HANDLE = "graspable_handle"      # Note
    GRASPABLE_BODY = "graspable_body"          # Note
    GRASPABLE_EDGE = "graspable_edge"          # Note
    
    # Note

    PINCHABLE = "pinchable"                    # Note
    CLAMPABLE = "clampable"                    # Note
    
    # Note

    WRAPPABLE = "wrappable"                    # Note
    SCOOPABLE = "scoopable"                    # Note
    
    # Note

    FRAGILE = "fragile"                        # Note
    DEFORMABLE = "deformable"                  # Note
    HEAVY = "heavy"                            # Note
    LIQUID_CONTAINER = "liquid_container"      # Note


class MaterialType(str, Enum):


    """MaterialType class."""
    CERAMIC = "ceramic"          # Note
    GLASS = "glass"              # Note
    METAL = "metal"              # Note
    PLASTIC = "plastic"          # Note
    WOOD = "wood"                # Note
    FABRIC = "fabric"            # Note
    RUBBER = "rubber"            # Note
    PAPER = "paper"              # Note
    FOOD = "food"                # Note
    UNKNOWN = "unknown"


@dataclass
class AffordanceInfo:
    """AffordanceInfo class."""
    primary_affordance: AffordanceType
    secondary_affordances: List[AffordanceType] = field(default_factory=list)
    material: MaterialType = MaterialType.UNKNOWN
    fragility: float = 0.5  # Note
    deformability: float = 0.0  # Note
    weight_estimate: str = "medium"  # light, medium, heavy
    graspable_regions: List[str] = field(default_factory=list)  # ["handle", "body", "rim"]
    avoid_regions: List[str] = field(default_factory=list)  # ["sharp_edge", "hot_surface"]
    
    def to_embedding_text(self) -> str:
    
        """to_embedding_text function."""
        parts = [
            f"affordance:{self.primary_affordance.value}",
            f"material:{self.material.value}",
            f"fragility:{self.fragility:.1f}",
            f"weight:{self.weight_estimate}",
        ]
        if self.graspable_regions:
            parts.append(f"grasp_at:{','.join(self.graspable_regions)}")
        if self.avoid_regions:
            parts.append(f"avoid:{','.join(self.avoid_regions)}")
        return " ".join(parts)
        
    def similarity_to(self, other: "AffordanceInfo") -> float:
        
        """similarity_to function."""
        score = 0.0
        
        # Note

        if self.primary_affordance == other.primary_affordance:
            score += 0.5
        elif self.primary_affordance in other.secondary_affordances:
            score += 0.3
            
        # Note

        if self.material == other.material:
            score += 0.2
        elif self._material_compatible(self.material, other.material):
            score += 0.1
            
        # Note

        fragility_diff = abs(self.fragility - other.fragility)
        score += 0.15 * (1 - fragility_diff)
        
        # Note

        if self.graspable_regions and other.graspable_regions:
            overlap = len(set(self.graspable_regions) & set(other.graspable_regions))
            total = len(set(self.graspable_regions) | set(other.graspable_regions))
            score += 0.15 * (overlap / total if total > 0 else 0)
            
        return min(score, 1.0)
        
    @staticmethod
    def _material_compatible(m1: MaterialType, m2: MaterialType) -> bool:
        """_material_compatible function."""
        compatible_groups = [
            {MaterialType.CERAMIC, MaterialType.GLASS},  # Note
            {MaterialType.METAL, MaterialType.PLASTIC},  # Note
            {MaterialType.FABRIC, MaterialType.PAPER},   # Note
        ]
        for group in compatible_groups:
            if m1 in group and m2 in group:
                return True
        return False


# ============================================================
# Note

# ============================================================

class AffordancePredictor:

    """AffordancePredictor class."""
    PREDICTION_PROMPT = """Analyze the target object and predict its affordance properties for grasp planning.

Output in JSON format: ```json
{
    "object_name": "target_object",
    "primary_affordance": "affordance category",
    "secondary_affordances": ["liftable"],
    "material": "unknown",
    "fragility": 0.0-1.0,
    "weight_estimate": "light/medium/heavy",
    "graspable_regions": ["list of graspable regions"],
    "avoid_regions": ["fragile_area"],
    "reasoning": "your reasoning"
}
```

Affordance categories:
- graspable_handle: objects with handles
- graspable_body: Can be grasped around the main body
- graspable_edge: Can be grasped at the edge/rim
- pinchable: Can be pinched with fingertips
- clampable: Can be clamped firmly
- wrappable: objects that can be wrapped/enveloped
- fragile: sample
- deformable: sample
- liquid_container: sample: ceramic, glass, metal, plastic, wood, fabric, rubber, paper, food, unknown

Output valid JSON only."""

    def __init__(self, vlm_engine = None):
        self.vlm_engine = vlm_engine
        
    def predict(
        self,
        image: Image.Image,
        object_description: Optional[str] = None,
    ) -> AffordanceInfo:
        """predict function."""
        if self.vlm_engine is None:
            # Note

            return self._default_affordance(object_description)
            
        # Note

        prompt = self.PREDICTION_PROMPT
        if object_description:
            prompt = f"target_objectdescription: {object_description}\n\n" + prompt
            
        # Note

        try:
            response = self.vlm_engine.generate(
                images=[image],
                prompt=prompt,
                max_tokens=500,
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"[AffordancePredictor] VLM prediction failed: {e}")
            return self._default_affordance(object_description)
            
    def _parse_response(self, response: str) -> AffordanceInfo:
            
        """_parse_response function."""
        try:
            # Note

            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("Invalid JSON response from VLM")
                
            data = json.loads(json_match.group())
            
            # Note

            primary = AffordanceType(data.get("primary_affordance", "graspable_body"))
            secondary = [
                AffordanceType(a) for a in data.get("secondary_affordances", [])
                if a in [e.value for e in AffordanceType]
            ]
            material = MaterialType(data.get("material", "unknown"))
            
            return AffordanceInfo(
                primary_affordance=primary,
                secondary_affordances=secondary,
                material=material,
                fragility=float(data.get("fragility", 0.5)),
                weight_estimate=data.get("weight_estimate", "medium"),
                graspable_regions=data.get("graspable_regions", []),
                avoid_regions=data.get("avoid_regions", []),
            )
        except Exception as e:
            print(f"[AffordancePredictor] Prediction failed: {e}")
            return self._default_affordance()
            
    def _default_affordance(self, description: Optional[str] = None) -> AffordanceInfo:
            
        """_default_affordance function."""
        if description:
            desc_lower = description.lower()
            
            # Note

            if any(w in desc_lower for w in ["cup", "mug", "goblet"]):
                return AffordanceInfo(
                    primary_affordance=AffordanceType.GRASPABLE_HANDLE,
                    material=MaterialType.CERAMIC,
                    fragility=0.6,
                    graspable_regions=["handle", "body"],
                )
            elif any(w in desc_lower for w in ["bottle", "flask", "jar"]):
                return AffordanceInfo(
                    primary_affordance=AffordanceType.GRASPABLE_BODY,
                    material=MaterialType.PLASTIC,
                    fragility=0.3,
                    graspable_regions=["body", "neck"],
                )
            elif any(w in desc_lower for w in ["tablet", "phone", "laptop"]):
                return AffordanceInfo(
                    primary_affordance=AffordanceType.CLAMPABLE,
                    material=MaterialType.GLASS,
                    fragility=0.7,
                    graspable_regions=["body"],
                    avoid_regions=["screen"],
                )
                
        # Note

        return AffordanceInfo(
            primary_affordance=AffordanceType.GRASPABLE_BODY,
            material=MaterialType.UNKNOWN,
        )


# ============================================================
# Note

# ============================================================

@dataclass
class HierarchicalRetrievalConfig:
    """HierarchicalRetrievalConfig class."""
    # Note

    category_top_k: int = 30
    category_weight: float = 0.2
    
    # Note

    affordance_top_k: int = 15
    affordance_weight: float = 0.4
    
    # Note

    visual_top_k: int = 5
    visual_weight: float = 0.4
    
    # Note

    final_top_k: int = 5
    min_score: float = 0.3
    
    # Note

    use_contrastive: bool = True
    negative_penalty: float = 0.3


@dataclass
class HierarchicalRetrievalResult:
    """HierarchicalRetrievalResult class."""
    experience_id: str
    object_name: str
    
    # Note

    category_score: float
    affordance_score: float
    visual_score: float
    
    # Note

    final_score: float
    
    # Note

    affordance_info: AffordanceInfo
    
    # Note

    grasp_pose: Dict[str, Any]
    key_insight: str
    
    # Note

    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalAffordanceRetriever:


    """HierarchicalAffordanceRetriever class."""
    def __init__(
        self,
        vector_store,
        embedding_model,
        affordance_predictor: Optional[AffordancePredictor] = None,
        config: Optional[HierarchicalRetrievalConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.affordance_predictor = affordance_predictor or AffordancePredictor()
        self.config = config or HierarchicalRetrievalConfig()
        
        # Note

        self._failure_cache: Dict[str, List[str]] = {}  # category -> [failed_exp_ids]
        
    def retrieve(
        self,
        query_image: Image.Image,
        query_text: str,
        object_description: Optional[str] = None,
    ) -> List[HierarchicalRetrievalResult]:
        """retrieve function."""
        start_time = time.time()
        
        # Note

        query_affordance = self.affordance_predictor.predict(
            query_image,
            object_description or query_text,
        )
        print(f"[HAA-RAG] Query object affordance: {query_affordance.primary_affordance.value}")
        
        # Note

        category_results = self._category_search(query_text)
        if not category_results:
            print("[HAA-RAG] No retrieval results found")
            return []
        print(f"[HAA-RAG] Category retrieval: {len(category_results)} results")
        
        # Note

        affordance_results = self._affordance_filter(
            category_results,
            query_affordance,
        )
        if not affordance_results:
            print("[HAA-RAG] Retrieval complete, processing results")
            affordance_results = category_results[:self.config.affordance_top_k]
        print(f"[HAA-RAG] Affordance filtering: {len(affordance_results)} results")
        
        # Note

        visual_results = self._visual_rerank(
            query_image,
            affordance_results,
        )
        print(f"[HAA-RAG] Visual reranking: {len(visual_results)} results")
        
        # Note

        if self.config.use_contrastive:
            visual_results = self._apply_contrastive_penalty(
                visual_results,
                query_affordance,
            )
            
        # Note

        final_results = sorted(
            visual_results,
            key=lambda x: x.final_score,
            reverse=True,
        )
        final_results = [
            r for r in final_results
            if r.final_score >= self.config.min_score
        ][:self.config.final_top_k]
        
        elapsed = time.time() - start_time
        print(f"[HAA-RAG] Retrieval complete in {elapsed*1000:.1f}ms, returning {len(final_results)} results")
        
        return final_results
        
    def _category_search(self, query_text: str) -> List[Dict[str, Any]]:
        
        """_category_search function."""
        query_embedding = self.embedding_model.encode_text(query_text)
        
        results = self.vector_store.query_by_text(
            query_embedding=query_embedding[0],
            n_results=self.config.category_top_k,
        )
        
        return results
        
    def _affordance_filter(
        self,
        candidates: List[Dict[str, Any]],
        query_affordance: AffordanceInfo,
    ) -> List[Dict[str, Any]]:
        """_affordance_filter function."""
        scored_candidates = []
        
        for candidate in candidates:
            # Note

            candidate_affordance = self._extract_affordance(candidate)
            
            # Note

            affordance_score = query_affordance.similarity_to(candidate_affordance)
            
            # Note

            candidate["affordance_score"] = affordance_score
            candidate["affordance_info"] = candidate_affordance
            
            if affordance_score >= 0.3:
                # Note
                scored_candidates.append(candidate)
                
        # Note

        scored_candidates.sort(key=lambda x: x["affordance_score"], reverse=True)
        
        return scored_candidates[:self.config.affordance_top_k]
        
    def _extract_affordance(self, candidate: Dict[str, Any]) -> AffordanceInfo:
        
        """_extract_affordance function."""
        metadata = candidate.get("metadata", {})
        
        # Note

        if "affordance" in metadata:
            aff_data = metadata["affordance"]
            try:
                return AffordanceInfo(
                    primary_affordance=AffordanceType(aff_data.get("primary", "graspable_body")),
                    material=MaterialType(aff_data.get("material", "unknown")),
                    fragility=aff_data.get("fragility", 0.5),
                    graspable_regions=aff_data.get("graspable_regions", []),
                )
            except:
                pass
                
        # Note

        category = metadata.get("category", "")
        return self._infer_affordance_from_category(category)
        
    def _infer_affordance_from_category(self, category: str) -> AffordanceInfo:
        
        """_infer_affordance_from_category function."""
        category_affordance_map = {
            "cup": AffordanceInfo(
                primary_affordance=AffordanceType.GRASPABLE_HANDLE,
                material=MaterialType.CERAMIC,
                fragility=0.6,
                graspable_regions=["handle", "body"],
            ),
            "bottle": AffordanceInfo(
                primary_affordance=AffordanceType.GRASPABLE_BODY,
                material=MaterialType.PLASTIC,
                fragility=0.3,
                graspable_regions=["body", "neck"],
            ),
            "tool": AffordanceInfo(
                primary_affordance=AffordanceType.PINCHABLE,
                material=MaterialType.METAL,
                fragility=0.1,
                graspable_regions=["handle"],
            ),
            "phone": AffordanceInfo(
                primary_affordance=AffordanceType.CLAMPABLE,
                material=MaterialType.GLASS,
                fragility=0.7,
                graspable_regions=["body"],
                avoid_regions=["screen"],
            ),
            "fruit": AffordanceInfo(
                primary_affordance=AffordanceType.WRAPPABLE,
                material=MaterialType.FOOD,
                fragility=0.5,
                deformability=0.3,
            ),
        }
        
        for key, affordance in category_affordance_map.items():
            if key in category.lower():
                return affordance
                
        # Note

        return AffordanceInfo(
            primary_affordance=AffordanceType.GRASPABLE_BODY,
        )
        
    def _visual_rerank(
        self,
        query_image: Image.Image,
        candidates: List[Dict[str, Any]],
    ) -> List[HierarchicalRetrievalResult]:
        """_visual_rerank function."""
        query_visual = self.embedding_model.encode_image(query_image)
        
        results = []
        for candidate in candidates:
            # Note

            visual_embedding = self.vector_store.get_visual_embedding(candidate["id"])
            if visual_embedding is not None:
                visual_score = float(np.dot(query_visual[0], visual_embedding))
            else:
                visual_score = 0.5  # Note
                
            # Note

            category_score = candidate.get("distance", 0)
            if isinstance(category_score, float) and category_score > 0:
                category_score = 1 - min(category_score, 1)  # Note
            else:
                category_score = candidate.get("similarity", 0.5)
                
            affordance_score = candidate.get("affordance_score", 0.5)
            
            final_score = (
                self.config.category_weight * category_score +
                self.config.affordance_weight * affordance_score +
                self.config.visual_weight * visual_score
            )
            
            # Note

            result = HierarchicalRetrievalResult(
                experience_id=candidate["id"],
                object_name=candidate.get("object_name", ""),
                category_score=category_score,
                affordance_score=affordance_score,
                visual_score=visual_score,
                final_score=final_score,
                affordance_info=candidate.get("affordance_info", AffordanceInfo(
                    primary_affordance=AffordanceType.GRASPABLE_BODY
                )),
                grasp_pose=candidate.get("grasp_pose", {}),
                key_insight=candidate.get("key_insight", ""),
                metadata=candidate.get("metadata", {}),
            )
            results.append(result)
            
        return results
        
    def _apply_contrastive_penalty(
        self,
        results: List[HierarchicalRetrievalResult],
        query_affordance: AffordanceInfo,
    ) -> List[HierarchicalRetrievalResult]:
        """_apply_contrastive_penalty function."""
        category = query_affordance.primary_affordance.value
        failed_ids = self._failure_cache.get(category, [])
        
        for result in results:
            if result.experience_id in failed_ids:
                # Note

                result.final_score *= (1 - self.config.negative_penalty)
                result.metadata["contrastive_penalty"] = True
                
        return results
        
    def record_failure(self, experience_id: str, category: str) -> None:
        
        """record_failure function."""
        if category not in self._failure_cache:
            self._failure_cache[category] = []
        if experience_id not in self._failure_cache[category]:
            self._failure_cache[category].append(experience_id)
            # Note

            if len(self._failure_cache[category]) > 100:
                self._failure_cache[category] = self._failure_cache[category][-100:]


# ============================================================
# Note

# ============================================================

@dataclass 
class DynamicContextConfig:
    """DynamicContextConfig class."""
    max_examples: int = 5
    min_examples: int = 1
    max_tokens: int = 2000
    
    # Note

    weight_by_score: bool = True  # Note
    detail_threshold: float = 0.8  # Note
    summary_threshold: float = 0.5  # Note


class DynamicContextCompressor:


    """DynamicContextCompressor class."""
    def __init__(self, config: Optional[DynamicContextConfig] = None):
        self.config = config or DynamicContextConfig()
        
    def compress(
        self,
        results: List[HierarchicalRetrievalResult],
    ) -> List[Dict[str, Any]]:
        """compress function."""
        if not results:
            return []
            
        examples = []
        current_tokens = 0
        
        for i, result in enumerate(results[:self.config.max_examples]):
            # Note

            if result.final_score >= self.config.detail_threshold:
                example = self._full_detail(result, i + 1)
            elif result.final_score >= self.config.summary_threshold:
                example = self._medium_detail(result, i + 1)
            else:
                example = self._summary_only(result, i + 1)
                
            # Note

            example_tokens = self._estimate_tokens(example)
            
            if current_tokens + example_tokens <= self.config.max_tokens:
                examples.append(example)
                current_tokens += example_tokens
            elif len(examples) < self.config.min_examples:
                # Note

                examples.append(self._summary_only(result, i + 1))
                break
            else:
                break
                
        return examples
        
    def _full_detail(self, result: HierarchicalRetrievalResult, index: int) -> Dict[str, Any]:
        
        """_full_detail function."""
        return {
            "index": index,
            "detail_level": "full",
            "object_name": result.object_name,
            "affordance": result.affordance_info.primary_affordance.value,
            "material": result.affordance_info.material.value,
            "fragility": result.affordance_info.fragility,
            "grasp_regions": result.affordance_info.graspable_regions,
            "avoid_regions": result.affordance_info.avoid_regions,
            "grasp_pose": result.grasp_pose,
            "key_insight": result.key_insight,
            "similarity": result.final_score,
        }
        
    def _medium_detail(self, result: HierarchicalRetrievalResult, index: int) -> Dict[str, Any]:
        
        """_medium_detail function."""
        return {
            "index": index,
            "detail_level": "medium",
            "object_name": result.object_name,
            "affordance": result.affordance_info.primary_affordance.value,
            "grasp_summary": self._summarize_pose(result.grasp_pose),
            "key_insight": result.key_insight,
            "similarity": result.final_score,
        }
        
    def _summary_only(self, result: HierarchicalRetrievalResult, index: int) -> Dict[str, Any]:
        
        """_summary_only function."""
        return {
            "index": index,
            "detail_level": "summary",
            "description": f"{result.object_name}: {result.affordance_info.primary_affordance.value}, score={result.final_score:.2f}",
        }
        
    def _summarize_pose(self, grasp_pose: Dict[str, Any]) -> str:
        
        """_summarize_pose function."""
        if not grasp_pose:
            return "Default grasp pose"
            
        pos = grasp_pose.get("position", {})
        width = grasp_pose.get("gripper_width", 0)
        
        return f"position({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f}), gripper={width:.3f}m"
        
    def _estimate_tokens(self, example: Dict[str, Any]) -> int:
        
        """_estimate_tokens function."""
        text = json.dumps(example, ensure_ascii=False)
        # Note

        return int(len(text) * 0.5)


# ============================================================
# Note

# ============================================================

class HierarchicalAffordanceAwareRAG:

    """HierarchicalAffordanceAwareRAG class."""
    def __init__(
        self,
        vector_store,
        embedding_model,
        vlm_engine=None,
        retrieval_config: Optional[HierarchicalRetrievalConfig] = None,
        context_config: Optional[DynamicContextConfig] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.vlm_engine = vlm_engine
        
        # Note

        self.affordance_predictor = AffordancePredictor(vlm_engine)
        self.retriever = HierarchicalAffordanceRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            affordance_predictor=self.affordance_predictor,
            config=retrieval_config,
        )
        self.compressor = DynamicContextCompressor(context_config)
        
    def retrieve_and_build_context(
        self,
        query_image: Image.Image,
        query_text: str,
        object_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """retrieve_and_build_context function."""
        # Note

        results = self.retriever.retrieve(
            query_image=query_image,
            query_text=query_text,
            object_description=object_description,
        )
        
        # Note

        compressed_examples = self.compressor.compress(results)
        
        # Note

        system_prompt = self._build_affordance_aware_prompt(
            self.retriever.affordance_predictor.predict(
                query_image, object_description or query_text
            )
        )
        
        return {
            "query_affordance": self.retriever.affordance_predictor.predict(
                query_image, object_description or query_text
            ),
            "retrieved_results": results,
            "compressed_examples": compressed_examples,
            "system_prompt": system_prompt,
        }
        
    def _build_affordance_aware_prompt(self, affordance: AffordanceInfo) -> str:
        
        """_build_affordance_aware_prompt function."""
        return f"""Plan the grasp based on the object's affordance properties for optimal grasping.

## Target Object Properties
- Affordance type: {affordance.primary_affordance.value}
- Material: {affordance.material.value}
- Fragility: {affordance.fragility:.1f} (0=robust, 1=fragile)
- Graspable regions: {', '.join(affordance.graspable_regions) if affordance.graspable_regions else 'not specified'}
- Avoid regions: {', '.join(affordance.avoid_regions) if affordance.avoid_regions else 'none'}

## Grasp Strategy
{self._get_affordance_strategy(affordance)}

## Output Format
Output the grasp pose in JSON: ```json
{{
    "grasp_pose": {{
        "position": {{"x": float, "y": float, "z": float}},
        "orientation": {{"qx": float, "qy": float, "qz": float, "qw": float}},
        "gripper_width": float,
        "approach_vector": {{"x": float, "y": float, "z": float}}
    }},
    "grasp_type": "pinch/wrap/edge/handle",
    "force_level": "light/medium/firm",
    "confidence": float,
    "reasoning": string
}}
```"""
        
    def _get_affordance_strategy(self, affordance: AffordanceInfo) -> str:
        
        """_get_affordance_strategy function."""
        strategies = {
            AffordanceType.GRASPABLE_HANDLE: "Grasp the handle with a pinch grip for optimal control",
            AffordanceType.GRASPABLE_BODY: "Grasp the body with a wrapping grip for stability",
            AffordanceType.GRASPABLE_EDGE: "Use edge grasp with firm contact",
            AffordanceType.PINCHABLE: "Use precision pinch grip for small objects",
            AffordanceType.CLAMPABLE: "Use clamping grip on flat surfaces",
            AffordanceType.WRAPPABLE: "Use an enveloping grasp for maximum contact area",
            AffordanceType.FRAGILE: "Use edge grasp with firm contact",
            AffordanceType.DEFORMABLE: "Apply gentle force with adaptive grip",
            AffordanceType.LIQUID_CONTAINER: "Keep upright, grasp from the side with stable grip",
        }
        
        base_strategy = strategies.get(
            affordance.primary_affordance,
            "Apply standard grasp based on object properties"
        )
        
        # Note

        if affordance.fragility > 0.7:
            base_strategy += "\n⚠️ Fragile object detected, reduce force and adjust gripper width"
        elif affordance.fragility < 0.3:
            base_strategy += "\n💪 Heavy object detected, use firm grasp"
            
        return base_strategy
        
    def record_feedback(
        self,
        experience_id: str,
        success: bool,
        affordance_category: str,
    ) -> None:
        """record_feedback function."""
        if not success:
            self.retriever.record_failure(experience_id, affordance_category)


# ============================================================
# Note

# ============================================================

def create_haa_rag(
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "grasp_experiences",
    vlm_engine=None,
) -> HierarchicalAffordanceAwareRAG:
    """create_haa_rag function."""
    from .embedding import get_embedding_model
    from ..knowledge_base.vector_store import ChromaVectorStore, VectorStoreConfig
    
    vector_store = ChromaVectorStore(VectorStoreConfig(
        persist_directory=persist_dir,
        collection_name=collection_name,
    ))
    
    embedding_model = get_embedding_model()
    
    return HierarchicalAffordanceAwareRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        vlm_engine=vlm_engine,
    )
