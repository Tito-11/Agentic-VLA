"""Core module implementation."""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RetrievalMode(Enum):


    """RetrievalMode class."""
    HAA_ONLY = "haa_only"               # Note
    MULTIMODAL_ONLY = "multimodal_only" # Note
    SCENE_GRAPH_ONLY = "scene_graph"    # Note
    FULL_FUSION = "full_fusion"         # Note


class ExecutionMode(Enum):


    """ExecutionMode class."""
    SINGLE_SHOT = "single_shot"     # Note
    WITH_REFLECTION = "reflection"  # Note
    FULL_AGENTIC = "full_agentic"   # Note


@dataclass
class UnifiedQuery:
    """UnifiedQuery class."""
    text: str
    image: Optional[Any] = None
    depth_map: Optional[Any] = None
    point_cloud: Optional[Any] = None
    
    # Note

    retrieval_mode: RetrievalMode = RetrievalMode.FULL_FUSION
    execution_mode: ExecutionMode = ExecutionMode.FULL_AGENTIC
    top_k: int = 5
    max_retries: int = 3


@dataclass
class UnifiedResult:
    """UnifiedResult class."""
    success: bool
    grasp_parameters: Dict[str, Any]
    
    # Note

    retrieval_results: Dict[str, Any] = field(default_factory=dict)
    
    # Note

    scene_context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Note

    attempts: int = 1
    reflections: List[str] = field(default_factory=list)
    
    # Note

    metrics: Dict[str, float] = field(default_factory=dict)


class UnifiedAgenticRAG:


    """UnifiedAgenticRAG class."""
    def __init__(
        self,
        vlm_client=None,
        robot_controller=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.vlm_client = vlm_client
        self.robot_controller = robot_controller
        self.config = config or self._default_config()
        
        # Note

        self._haa_rag = None
        self._multimodal_rag = None
        self._scene_graph_rag = None
        self._agentic_pipeline = None
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            # Note

            "haa_weight": 0.35,
            "multimodal_weight": 0.40,
            "scene_weight": 0.25,
            
            # Note

            "max_retries": 3,
            "reflection_threshold": 0.7,
            
            # Note

            "text_weight": 0.35,
            "image_weight": 0.40,
            "geometry_weight": 0.25,
        }
    
    @property
    def haa_rag(self):
        if self._haa_rag is None:
            try:
                from .affordance_rag import HierarchicalAffordanceRetriever
                self._haa_rag = HierarchicalAffordanceRetriever()
            except ImportError:
                pass
        return self._haa_rag
    
    @property
    def multimodal_rag(self):
        if self._multimodal_rag is None:
            try: from .multimodal_rag import MultiModalRAG
                self._multimodal_rag = MultiModalRAG()
            except ImportError:
                pass
        return self._multimodal_rag
    
    @property
    def scene_graph_rag(self):
        if self._scene_graph_rag is None:
            try: from .scene_graph_rag import SceneGraphRAG
                self._scene_graph_rag = SceneGraphRAG(self.vlm_client)
            except ImportError:
                pass
        return self._scene_graph_rag
    
    @property
    def agentic_pipeline(self):
        if self._agentic_pipeline is None:
            try: from .agentic_grasp_pipeline import AgenticGraspPipeline
                self._agentic_pipeline = AgenticGraspPipeline(
                    vlm_client=self.vlm_client,
                    robot_controller=self.robot_controller,
                )
            except ImportError:
                pass
        return self._agentic_pipeline
    
    def retrieve(
        self,
        query: UnifiedQuery,
    ) -> Dict[str, Any]:
        """retrieve function."""
        results = {
            "haa": None,
            "multimodal": None,
            "scene_graph": None,
            "fused": None,
        }
        
        mode = query.retrieval_mode
        
        # Note

        if mode in [RetrievalMode.HAA_ONLY, RetrievalMode.FULL_FUSION]:
            if self.haa_rag:
                results["haa"] = self._retrieve_haa(query)
        
        # Note

        if mode in [RetrievalMode.MULTIMODAL_ONLY, RetrievalMode.FULL_FUSION]:
            if self.multimodal_rag:
                results["multimodal"] = self._retrieve_multimodal(query)
        
        # Note

        if mode in [RetrievalMode.SCENE_GRAPH_ONLY, RetrievalMode.FULL_FUSION]:
            if self.scene_graph_rag:
                results["scene_graph"] = self._retrieve_scene_graph(query)
        
        # Note

        if mode == RetrievalMode.FULL_FUSION:
            results["fused"] = self._fuse_results(results)
        
        return results
    
    def _retrieve_haa(self, query: UnifiedQuery) -> Dict[str, Any]:
    
        """_retrieve_haa function."""
        try:
            # Note

            results = self.haa_rag.retrieve(
                query_image=query.image,
                query_text=query.text,
                top_k=query.top_k,
            )
            return {
                "experiences": results,
                "affordance_matched": True,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _retrieve_multimodal(self, query: UnifiedQuery) -> Dict[str, Any]:
    
        """_retrieve_multimodal function."""
        try:
            context = self.multimodal_rag.retrieve_and_build_context(
                query_image=query.image,
                query_text=query.text,
                depth_map=query.depth_map,
                point_cloud=query.point_cloud,
                top_k=query.top_k,
            )
            return context
        except Exception as e:
            return {"error": str(e)}
    
    def _retrieve_scene_graph(self, query: UnifiedQuery) -> Dict[str, Any]:
    
        """_retrieve_scene_graph function."""
        try:
            context = self.scene_graph_rag.retrieve_and_build_context(
                query_image=query.image,
                query_text=query.text,
                top_k=query.top_k,
            )
            return context
        except Exception as e:
            return {"error": str(e)}
    
    def _fuse_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
    
        """_fuse_results function."""
        fused = {
            "experiences": [],
            "constraints": [],
            "scene_context": None,
            "prompt": "",
        }
        
        # Note

        if results.get("haa") and "experiences" in results["haa"]:
            for exp in results["haa"]["experiences"][:3]:
                fused["experiences"].append({
                    "source": "haa",
                    "data": exp,
                    "weight": self.config["haa_weight"],
                })
        
        if results.get("multimodal") and "experiences" in results["multimodal"]:
            for exp in results["multimodal"]["experiences"][:3]:
                fused["experiences"].append({
                    "source": "multimodal",
                    "data": exp,
                    "weight": self.config["multimodal_weight"],
                })
        
        # Note

        if results.get("scene_graph") and "constraints" in results["scene_graph"]:
            fused["constraints"] = results["scene_graph"]["constraints"]
            fused["scene_context"] = results["scene_graph"].get("scene_graph")
        
        # Note

        fused["prompt"] = self._build_fused_prompt(fused)
        
        return fused
    
    def _build_fused_prompt(self, fused: Dict[str, Any]) -> str:
    
        """_build_fused_prompt function."""
        lines = ["## retrievalexperiences\n"]
        
        for i, exp in enumerate(fused["experiences"][:5]):
            lines.append(f"### experience {i+1} (sample: {exp['source']}, sample: {exp['weight']:.2f})")
            if isinstance(exp["data"], dict):
                lines.append(f"- object: {exp['data'].get('object_name', 'unknown')}")
                lines.append(f"- graspclassEN: {exp['data'].get('grasp_type', 'unknown')}")
        
        if fused["constraints"]:
            lines.append("\n## ⚠️ sceneEN")
            for c in fused["constraints"]:
                lines.append(f"- [{c.get('type')}] {c.get('description')}")
        
        return "\n".join(lines)
    
    def execute(
        self,
        query: UnifiedQuery,
    ) -> UnifiedResult:
        """execute function."""
        start_time = time.time()
        
        # Note

        retrieval_results = self.retrieve(query)
        
        # Note

        mode = query.execution_mode
        
        if mode == ExecutionMode.SINGLE_SHOT:
            result = self._execute_single_shot(query, retrieval_results)
        elif mode == ExecutionMode.WITH_REFLECTION:
            result = self._execute_with_reflection(query, retrieval_results)
        else:
            # FULL_AGENTIC
            result = self._execute_full_agentic(query, retrieval_results)
        
        # Note

        result.metrics["total_time"] = time.time() - start_time
        result.retrieval_results = retrieval_results
        
        return result
    
    def _execute_single_shot(
        self,
        query: UnifiedQuery,
        retrieval_results: Dict[str, Any],
    ) -> UnifiedResult:
        """_execute_single_shot function."""
        # Note

        fused = retrieval_results.get("fused") or {}
        
        # Note

        success = self._simulate_grasp(fused)
        
        return UnifiedResult(
            success=success,
            grasp_parameters={"type": "precision", "force": 0.5},
            attempts=1,
        )
    
    def _execute_with_reflection(
        self,
        query: UnifiedQuery,
        retrieval_results: Dict[str, Any],
    ) -> UnifiedResult:
        """_execute_with_reflection function."""
        reflections = []
        attempts = 0
        
        fused = retrieval_results.get("fused") or {}
        
        for attempt in range(query.max_retries):
            attempts += 1
            success = self._simulate_grasp(fused)
            
            if success:
                break
            
            # Note

            reflection = self._reflect_on_failure(attempt, fused)
            reflections.append(reflection)
            
            # Note

            fused = self._adjust_based_on_reflection(fused, reflection)
        
        return UnifiedResult(
            success=success,
            grasp_parameters={"type": "adjusted", "force": 0.6},
            attempts=attempts,
            reflections=reflections,
        )
    
    def _execute_full_agentic(
        self,
        query: UnifiedQuery,
        retrieval_results: Dict[str, Any],
    ) -> UnifiedResult:
        """_execute_full_agentic function."""
        if self.agentic_pipeline:
            # Note

            try:
                result = self.agentic_pipeline.execute(
                    query_image=query.image,
                    query_text=query.text,
                )
                return UnifiedResult(
                    success=result.get("success", False),
                    grasp_parameters=result.get("grasp_params", {}),
                    attempts=result.get("attempts", 1),
                    reflections=result.get("reflections", []),
                )
            except Exception as e:
                pass
        
        # Note

        return self._execute_with_reflection(query, retrieval_results)
    
    def _simulate_grasp(self, context: Dict[str, Any]) -> bool:
    
        """_simulate_grasp function."""
        import random
        
        # Note

        base_rate = 0.5
        
        # Note

        if context.get("experiences"):
            base_rate += 0.2
        
        # Note

        if context.get("constraints"):
            base_rate += 0.1  # Note
        
        return random.random() < base_rate
    
    def _reflect_on_failure(
        self,
        attempt: int,
        context: Dict[str, Any],
    ) -> str:
        """_reflect_on_failure function."""
        reflections = [
            "grasp sample, sample",
            "sample, sample",
            "targetpositionEN, sample",
            "objectEN, sample",
        ]
        
        if attempt < len(reflections):
            return reflections[attempt]
        return "ENgrasp sample"
    
    def _adjust_based_on_reflection(
        self,
        context: Dict[str, Any],
        reflection: str,
    ) -> Dict[str, Any]:
        """_adjust_based_on_reflection function."""
        context["adjustment"] = reflection
        return context
    
    def add_experience(
        self,
        result: UnifiedResult,
        query: UnifiedQuery,
    ):
        """add_experience function."""
        experience = {
            "query_text": query.text,
            "success": result.success,
            "attempts": result.attempts,
            "grasp_parameters": result.grasp_parameters,
            "reflections": result.reflections,
            "timestamp": time.time(),
        }
        
        # Note

        if self.haa_rag and hasattr(self.haa_rag, 'add_experience'):
            self.haa_rag.add_experience(experience)
        
        if self.multimodal_rag and hasattr(self.multimodal_rag, 'add_experience'):
            self.multimodal_rag.add_experience(experience)


def create_unified_system(
    vlm_client=None,
    robot_controller=None,
    config: Optional[Dict[str, Any]] = None,
) -> UnifiedAgenticRAG:
    """create_unified_system function."""
    return UnifiedAgenticRAG(vlm_client, robot_controller, config)


# Note

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Agentic RAG System Demo")
    print("=" * 60)
    
    # Note

    system = create_unified_system()
    
    # Note

    query = UnifiedQuery(
        text="grasp sample",
        retrieval_mode=RetrievalMode.FULL_FUSION,
        execution_mode=ExecutionMode.FULL_AGENTIC,
    )
    
    # Note

    result = system.execute(query)
    
    print(f"\n✅ executeresult:")
    print(f"   success: {result.success}")
    print(f"   sample: {result.attempts}")
    print(f"   total_time: {result.metrics.get('total_time', 0):.3f}s")
    
    if result.reflections:
        print(f"\n📝 sample:")
        for i, r in enumerate(result.reflections):
            print(f"   {i+1}. {r}")
