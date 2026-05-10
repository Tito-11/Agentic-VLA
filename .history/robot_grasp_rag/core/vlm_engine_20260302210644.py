"""Core module implementation."""

import os
import time
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class VLMConfig:
    """VLMConfig class."""
    model_path: str = ""
    quantization: str = "bitsandbytes"
    load_format: str = "bitsandbytes"
    dtype: str = "half"
    kv_cache_dtype: str = "fp8"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    max_images_per_prompt: int = 4
    trust_remote_code: bool = True


@dataclass
class GenerationConfig:
    """GenerationConfig class."""
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    stop_tokens: Optional[List[str]] = None


class VLMEngine:


    """VLMEngine class."""
    def __init__(self, config: Optional[VLMConfig] = None):
        self.config = config or VLMConfig()
        self.llm = None
        self._initialized = False
        
    def initialize(self) -> None:
        
        """initialize function."""
        if self._initialized:
            return
            
        # Note

        from vllm import LLM
        
        print(f"[VLMEngine] load: {self.config.model_path}")
        print(f"[VLMEngine] Quantization: {self.config.quantization}")
        print(f"[VLMEngine] KV Cache: {self.config.kv_cache_dtype}")
        
        start_time = time.time()
        
        self.llm = LLM(
            model=self.config.model_path,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            load_format=self.config.load_format,
            kv_cache_dtype=self.config.kv_cache_dtype,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            limit_mm_per_prompt={"image": self.config.max_images_per_prompt},
            trust_remote_code=self.config.trust_remote_code,
        )
        
        load_time = time.time() - start_time
        print(f"[VLMEngine] Model loaded, time: {load_time:.2f}s")
        
        # Note

        self._warmup()
        self._initialized = True
        
    def _warmup(self) -> None:
        
        """_warmup function."""
        from vllm import SamplingParams
        
        print("[VLMEngine] Running inference...")
        dummy_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"}
                ]
            }
        ]
        params = SamplingParams(temperature=0, max_tokens=5)
        _ = self.llm.chat(messages=dummy_messages, sampling_params=params)
        print("[VLMEngine] Inference complete")
        
    def _encode_image(self, image: Union[str, Image.Image]) -> str:
        
        """_encode_image function."""
        if isinstance(image, str):
            # Note

            if os.path.exists(image):
                with open(image, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            # Note

            elif image.startswith("data:image"):
                return image.split(",")[1]
            else:
                return image
        elif isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Image.Image]]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """generate function."""
        if not self._initialized:
            self.initialize()
            
        from vllm import SamplingParams
        
        config = generation_config or GenerationConfig()
        
        # Note

        content = []
        
        # Note

        if images:
            for img in images:
                img_base64 = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
                
        # Note

        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Note

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop_tokens,
        )
        
        # Note

        start_time = time.time()
        outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
        elapsed = time.time() - start_time
        
        result = outputs[0].outputs[0].text
        num_tokens = len(outputs[0].outputs[0].token_ids)
        
        # Note

        speed = num_tokens / elapsed if elapsed > 0 else 0
        print(f"[VLMEngine] Generated {num_tokens} tokens, speed: {speed:.1f} t/s")
        
        return result
        
    def chat(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """chat function."""
        if not self._initialized:
            self.initialize()
            
        from vllm import SamplingParams
        
        config = generation_config or GenerationConfig()
        
        # Note

        processed_messages = []
        for msg in messages:
            if msg["role"] == "user":
                content = []
                # Note

                if msg == messages[-1] and images:
                    for img in images:
                        img_base64 = self._encode_image(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        })
                content.append({"type": "text", "text": msg["content"]})
                processed_messages.append({"role": "user", "content": content})
            else:
                processed_messages.append(msg)
                
        # Note

        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            stop=config.stop_tokens,
        )
        
        outputs = self.llm.chat(messages=processed_messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text
        
    def get_memory_usage(self) -> Dict[str, float]:
        
        """get_memory_usage function."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
            }
        return {"allocated_gb": 0, "reserved_gb": 0}
        
    def __del__(self):
        
        """__del__ function."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Note

_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine(config: Optional[VLMConfig] = None) -> VLMEngine:


    """get_vlm_engine function."""
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = VLMEngine(config)
    return _vlm_engine
