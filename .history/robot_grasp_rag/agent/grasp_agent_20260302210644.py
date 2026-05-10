"""Core module implementation."""

import json
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image

from ..core.vlm_engine import VLMEngine, VLMConfig, GenerationConfig
from ..core.embedding import EmbeddingModel
from ..core.retriever import DualPathRetriever, RetrievalConfig, RetrievalResult
from ..core.context_builder import ContextBuilder, ContextConfig
from ..knowledge_base.vector_store import ChromaVectorStore, VectorStoreConfig
from ..knowledge_base.schema import GraspPose, GraspPrediction, Position3D, Quaternion
from ..knowledge_base.grasp_memory import GraspMemory
from .prompts import PromptTemplates, SafetyPrompts


@dataclass
class GraspAgentConfig:
    """GraspAgentConfig class."""
    # Note

    model_path: str = ""
    quantization: str = "bitsandbytes"
    kv_cache_dtype: str = "fp8"
    
    # Note

    top_k: int = 3
    semantic_weight: float = 0.4
    visual_weight: float = 0.6
    min_similarity: float = 0.3
    
    # Note

    num_examples: int = 3
    max_context_tokens: int = 2048
    
    # Note

    knowledge_base_dir: str = "./data/chroma_db"
    collection_name: str = "grasp_experiences"
    
    # Note

    max_tokens: int = 512
    temperature: float = 0.1


class GraspAgent:


    """GraspAgent class."""
    def __init__(self, config: Optional[GraspAgentConfig] = None):
        self.config = config or GraspAgentConfig()
        
        # Note

        self._vlm_engine: Optional[VLMEngine] = None
        self._embedding_model: Optional[EmbeddingModel] = None
        self._retriever: Optional[DualPathRetriever] = None
        self._context_builder: Optional[ContextBuilder] = None
        self._grasp_memory: Optional[GraspMemory] = None
        
        self._initialized = False
        
    def initialize(self) -> None:
        
        """initialize function."""
        if self._initialized:
            return
            
        print("[GraspAgent] initialize...")
        start_time = time.time()
        
        # Note

        print("[GraspAgent] Loading VLM model...")
        vlm_config = VLMConfig(
            model_path=self.config.model_path,
            quantization=self.config.quantization,
            kv_cache_dtype=self.config.kv_cache_dtype,
        )
        self._vlm_engine = VLMEngine(vlm_config)
        self._vlm_engine.initialize()
        
        # Note

        print("[GraspAgent] load...")
        self._embedding_model = EmbeddingModel()
        
        # Note

        vector_store = ChromaVectorStore(VectorStoreConfig(
            persist_directory=self.config.knowledge_base_dir,
            collection_name=self.config.collection_name,
        ))
        
        # Note

        retrieval_config = RetrievalConfig(
            top_k=self.config.top_k,
            semantic_weight=self.config.semantic_weight,
            visual_weight=self.config.visual_weight,
            min_similarity=self.config.min_similarity,
        )
        self._retriever = DualPathRetriever(
            vector_store=vector_store,
            embedding_model=self._embedding_model,
            config=retrieval_config,
        )
        
        # Note

        context_config = ContextConfig(
            num_examples=self.config.num_examples,
            max_context_tokens=self.config.max_context_tokens,
        )
        self._context_builder = ContextBuilder(context_config)
        
        # Note

        self._grasp_memory = GraspMemory(
            persist_dir=self.config.knowledge_base_dir,
            collection_name=self.config.collection_name,
        )
        
        elapsed = time.time() - start_time
        print(f"[GraspAgent] initializedone, time: {elapsed:.2f}s")
        self._initialized = True
        
    def plan_grasp(
        self,
        scene_image: Image.Image,
        target_object: str,
        task_description: str = "grasptarget_object",
        category: Optional[str] = None,
        use_rag: bool = True,
        verbose: bool = True,
    ) -> GraspPrediction:
        """plan_grasp function."""
        if not self._initialized:
            self.initialize()
            
        total_start = time.time()
        retrieval_time = 0
        inference_time = 0
        
        # Note

        retrieved_results: List[RetrievalResult] = []
        if use_rag:
            retrieval_start = time.time()
            query_text = f"{category or ''} {target_object} {task_description}"
            retrieved_results = self._retriever.retrieve(
                query_image=scene_image,
                query_text=query_text,
                top_k=self.config.top_k,
            )
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            if verbose:
                print(f"[GraspAgent] Retrieved {len(retrieved_results)} experiences")
                for r in retrieved_results:
                    print(f"  - {r.object_name}: similarity {r.combined_score:.3f}")
                    
        # Note

        context = self._context_builder.build_context(
            current_image=scene_image,
            task_instruction=task_description,
            retrieved_results=retrieved_results,
            object_description=target_object,
        )
        
        # Note

        examples_data = [
            {
                "object_name": r.object_name,
                "description": r.object_description,
                "category": r.metadata.get("category", ""),
                "grasp_pose": r.grasp_pose,
                "score": r.combined_score,
            }
            for r in retrieved_results
        ]
        
        system_prompt, user_prompt = PromptTemplates.build_grasp_prompt(
            target_object=target_object,
            task_description=task_description,
            examples=examples_data if use_rag else None,
            use_memory=use_rag,
        )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        if verbose:
            print(f"[GraspAgent] Prompt length: {len(full_prompt)} chars")
            
        # Note

        inference_start = time.time()
        
        gen_config = GenerationConfig(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        
        response = self._vlm_engine.generate(
            prompt=full_prompt,
            images=[scene_image],
            generation_config=gen_config,
        )
        
        inference_time = (time.time() - inference_start) * 1000
        
        if verbose:
            print(f"[GraspAgent] VLM response: {len(response)} chars")
            
        # Note

        grasp_pose, confidence, reasoning = self._parse_response(response)
        
        # Note

        total_time = (time.time() - total_start) * 1000
        
        prediction = GraspPrediction(
            grasp_pose=grasp_pose,
            confidence=confidence,
            reasoning=reasoning,
            retrieved_experiences=[r.experience_id for r in retrieved_results],
            inference_time_ms=inference_time,
            retrieval_time_ms=retrieval_time,
        )
        
        if verbose:
            print(f"[GraspAgent] total_time: {total_time:.1f}ms "
                  f"(retrieval: {retrieval_time:.1f}ms, inference: {inference_time:.1f}ms)")
            print(f"[GraspAgent] confidence: {confidence:.2f}")
            
        return prediction
        
    def _parse_response(
        self,
        response: str,
    ) -> Tuple[GraspPose, float, str]:
        """_parse_response function."""
        # Note

        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Note

            json_match = re.search(r'\{[^{}]*"grasp_pose"[^{}]*\{.*?\}.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Note

                return self._default_grasp_pose(), 0.5, response
                
        try:
            data = json.loads(json_str)
            
            # Note

            pose_data = data.get("grasp_pose", {})
            pos = pose_data.get("position", {})
            ori = pose_data.get("orientation", {})
            
            grasp_pose = GraspPose(
                position=Position3D(
                    x=float(pos.get("x", 0)),
                    y=float(pos.get("y", 0)),
                    z=float(pos.get("z", 0)),
                ),
                orientation=Quaternion(
                    qx=float(ori.get("qx", 0)),
                    qy=float(ori.get("qy", 0)),
                    qz=float(ori.get("qz", 0)),
                    qw=float(ori.get("qw", 1)),
                ),
                gripper_width=float(pose_data.get("gripper_width", 0.08)),
            )
            
            confidence = float(data.get("confidence", 0.8))
            reasoning = data.get("reasoning", "")
            
            return grasp_pose, confidence, reasoning
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[GraspAgent] Execution failed: {e}")
            return self._default_grasp_pose(), 0.5, response
            
    def _default_grasp_pose(self) -> GraspPose:
            
        """_default_grasp_pose function."""
        return GraspPose(
            position=Position3D(x=0.3, y=0.0, z=0.1),
            orientation=Quaternion.identity(),
            gripper_width=0.08,
        )
        
    def record_experience(
        self,
        scene_image: Image.Image,
        object_name: str,
        category: str,
        grasp_pose: GraspPose,
        success: bool,
        confidence: float = 1.0,
        description: str = "",
    ) -> str:
        """record_experience function."""
        if not self._initialized:
            self.initialize()
            
        from ..knowledge_base.schema import ObjectInfo
        
        object_info = ObjectInfo(
            name=object_name,
            category=category,
            description=description,
        )
        
        experience = self._grasp_memory.add_experience(
            object_info=object_info,
            scene_image=scene_image,
            grasp_pose=grasp_pose,
            success=success,
            confidence=confidence,
            save_image=True,
            to_long_term=success,  # Note
        )
        
        print(f"[GraspAgent] experiences: {experience.id}")
        return experience.id
        
    def get_memory_stats(self) -> Dict[str, Any]:
        
        """get_memory_stats function."""
        if not self._initialized:
            self.initialize()
            
        stats = self._grasp_memory.get_statistics()
        memory_usage = self._vlm_engine.get_memory_usage()
        
        return {
            **stats,
            **memory_usage,
        }
        
    def batch_plan(
        self,
        queries: List[Tuple[Image.Image, str, str]],
        use_rag: bool = True,
    ) -> List[GraspPrediction]:
        """batch_plan function."""
        results = []
        for scene_image, target_object, task_description in queries:
            prediction = self.plan_grasp(
                scene_image=scene_image,
                target_object=target_object,
                task_description=task_description,
                use_rag=use_rag,
                verbose=False,
            )
            results.append(prediction)
            
        return results


def create_demo_agent() -> GraspAgent:


    """create_demo_agent function."""
    config = GraspAgentConfig(
        model_path="",
        quantization="bitsandbytes",
        kv_cache_dtype="fp8",
        top_k=3,
        num_examples=3,
        max_tokens=512,
        temperature=0.1,
    )
    return GraspAgent(config)
