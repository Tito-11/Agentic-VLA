"""Core module implementation."""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from PIL import Image

from .retriever import RetrievalResult


@dataclass
class ContextConfig:
    """ContextConfig class."""
    num_examples: int = 3
    max_context_tokens: int = 2048
    include_images: bool = True
    include_pose_details: bool = True
    compression_enabled: bool = True


class ContextBuilder:


    """ContextBuilder class."""
    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()
        
    def build_context(
        self,
        current_image: Image.Image,
        task_instruction: str,
        retrieved_results: List[RetrievalResult],
        object_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """build_context function."""
        # Note

        system_prompt = self._build_system_prompt()
        
        # Note

        examples = self._build_examples(
            retrieved_results[:self.config.num_examples]
        )
        
        # Note

        query = self._build_query(
            task_instruction,
            object_description,
        )
        
        # Note

        images = [current_image]
        if self.config.include_images:
            for result in retrieved_results[:self.config.num_examples]:
                if result.image_path:
                    try:
                        images.append(Image.open(result.image_path))
                    except Exception as e:
                        print(f"[ContextBuilder] Failed to load image: {e}")
                        
        return {
            "system_prompt": system_prompt,
            "examples": examples,
            "query": query,
            "images": images,
        }
        
    def _build_system_prompt(self) -> str:
        
        """_build_system_prompt function."""
        return """You are a robotic grasp planning expert. Your task is to analyze the scene and past successful experiences to generate a 6D grasp pose for the target object.

## outputEN
JSON file pathoutputgrasp_pose: ```json
{
    "grasp_pose": {
        "position": {"x": float, "y": float, "z": float},  // grasp sampleposition
        "orientation": {"qx": float, "qy": float, "qz": float, "qw": float},  // orientation quaternion
        "gripper_width": float,  // gripper_width
        "approach_vector": {"x": float, "y": float, "z": float}  // sample
    },
    "confidence": float,  // confidence 0-1
    "reasoning": string   // Brief explanation of the grasp strategy
}
```

## sample
1. Study successful experiences, especially for similar objects
2. Evaluate grasp feasibility (shape, material, object properties)
3. sample
4. Generate a complete and precise grasp pose."""
        
    def _build_examples(
        self,
        results: List[RetrievalResult],
    ) -> List[Dict[str, Any]]:
        """_build_examples function."""
        examples = []
        
        for i, result in enumerate(results):
            example = {
                "index": i + 1,
                "object_name": result.object_name,
                "description": result.object_description,
                "image_ref": f"[images {i + 1}]",
                "grasp_pose": result.grasp_pose,
                "similarity_score": result.combined_score,
            }
            
            if not self.config.include_pose_details:
                # Note

                if "position" in result.grasp_pose:
                    example["grasp_summary"] = self._summarize_pose(result.grasp_pose)
                    
            examples.append(example)
            
        return examples
        
    def _build_query(
        self,
        task_instruction: str,
        object_description: Optional[str],
    ) -> Dict[str, str]:
        """_build_query function."""
        query = {
            "task": task_instruction,
            "image_ref": "[Scene image]",
        }
        
        if object_description:
            query["target_object"] = object_description
            
        return query
        
    def _summarize_pose(self, pose: Dict[str, Any]) -> str:
        
        """_summarize_pose function."""
        pos = pose.get("position", {})
        return f"Position ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})m"
        
    def format_as_prompt(
        self,
        context: Dict[str, Any],
        include_system: bool = True,
    ) -> str:
        """format_as_prompt function."""
        parts = []
        
        # Note

        if include_system:
            parts.append(context["system_prompt"])
            parts.append("")
            
        # Note

        if context["examples"]:
            parts.append("## Successful Grasp Experiences")
            parts.append("")
            
            for example in context["examples"]:
                parts.append(f"### experience {example['index']}: {example['object_name']}")
                parts.append(f"- description: {example['description']}")
                parts.append(f"- image: {example['image_ref']}")
                
                if self.config.include_pose_details and example.get("grasp_pose"):
                    parts.append(f"- Successful grasp pose:")
                    parts.append(f"```json")
                    parts.append(json.dumps(example["grasp_pose"], indent=2, ensure_ascii=False))
                    parts.append(f"```")
                elif example.get("grasp_summary"):
                    parts.append(f"- Grasp summary: {example['grasp_summary']}")
                    
                parts.append(f"- similarity: {example['similarity_score']:.2f}")
                parts.append("")
                
        # Note

        parts.append("## Current Task")
        parts.append("")
        parts.append(f"- Task: {context['query']['task']}")
        
        if context['query'].get('target_object'):
            parts.append(f"- target_object: {context['query']['target_object']}")
            
        parts.append(f"- Scene image: {context['query']['image_ref']}")
        parts.append("")
        parts.append("Based on the above context, analyze the scene and target object, then output an optimal grasp pose.")
        
        return "\n".join(parts)
        
    def estimate_tokens(self, text: str) -> int:
        
        """estimate_tokens function."""
        # Note

        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
        
    def compress_context(
        self,
        context: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """compress_context function."""
        max_tokens = max_tokens or self.config.max_context_tokens
        
        # Note

        prompt = self.format_as_prompt(context)
        estimated_tokens = self.estimate_tokens(prompt)
        
        if estimated_tokens <= max_tokens:
            return context
            
        # Note

        compressed = context.copy()
        
        # Note

        while (
            len(compressed["examples"]) > 1 and
            self.estimate_tokens(self.format_as_prompt(compressed)) > max_tokens
        ):
            compressed["examples"] = compressed["examples"][:-1]
            compressed["images"] = compressed["images"][:len(compressed["examples"]) + 1]
            
        # Note

        if self.estimate_tokens(self.format_as_prompt(compressed)) > max_tokens:
            for example in compressed["examples"]:
                if example.get("grasp_pose"):
                    example["grasp_summary"] = self._summarize_pose(example["grasp_pose"])
                    del example["grasp_pose"]
                    
        return compressed
