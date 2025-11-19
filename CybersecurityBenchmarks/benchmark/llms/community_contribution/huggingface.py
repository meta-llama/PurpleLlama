# pyre-strict

from __future__ import annotations

import logging
from typing import List, Optional, Any
import torch
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)
from typing_extensions import override

# Assuming these exist in your project structure based on imports
from ...benchmark_utils import audio_to_b64, image_to_b64
from ..llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
)

class HUGGINGFACE(LLM):
    """Accessing Local HuggingFace Models via Transformers"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        
        logging.info(f"Loading HuggingFace model: {self.model}")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine if this is a VLM (Vision Language Model) or standard LLM for loading
        # This is a heuristic; specific architectures might need explicit checking
        self.is_multimodal = "llava" in self.model.lower() or "vision" in self.model.lower() or "vl" in self.model.lower()

        if self.is_multimodal:
             # Helper for VLMs (handles image + text processing)
            self.processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map="auto",
                torch_dtype=torch.float16, # Usually preferred for inference
                trust_remote_code=True
            )
        else:
            self.processor = None
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True
            )

    def _generate_text(
        self, 
        messages: List[dict[str, str]], 
        max_new_tokens: int, 
        temperature: float, 
        top_p: float
    ) -> str:
        """Helper to handle text-only chat template application and generation"""
        
        # Apply chat template
        text_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.hf_model.device)

        # Generate
        with torch.no_grad():
            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (skipping the input tokens to return only the new content)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = []
        
        # If JSON schema is requested, inject it into a system message or the first user message
        # Native strictly constrained decoding in HF requires external libs (outlines/guidance)
        if guided_decode_json_schema:
             system_instruction = f"You are a helpful assistant. Always format your response in json format following this schema: {guided_decode_json_schema}"
             messages.append({"role": "system", "content": system_instruction})

        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append({"role": "assistant", "content": prompt_with_history[i]})

        return self._generate_text(
            messages, 
            max_new_tokens=DEFAULT_MAX_TOKENS, 
            temperature=temperature, 
            top_p=top_p
        )

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        
        full_system_prompt = system_prompt
        if guided_decode_json_schema:
            full_system_prompt += f"\nAlways format your response in json format following this schema: {guided_decode_json_schema}"

        messages = [{"role": "system", "content": full_system_prompt}]
        
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append({"role": "assistant", "content": prompt_with_history[i]})

        return self._generate_text(
            messages, 
            max_new_tokens=DEFAULT_MAX_TOKENS, 
            temperature=temperature, 
            top_p=top_p
        )

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = []
        if guided_decode_json_schema:
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Always format your response in json format following this schema: {guided_decode_json_schema}"
            })
        
        messages.append({"role": "user", "content": prompt})

        return self._generate_text(
            messages, 
            max_new_tokens=DEFAULT_MAX_TOKENS, 
            temperature=temperature, 
            top_p=top_p
        )

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        
        full_system_prompt = system_prompt
        if guided_decode_json_schema:
             full_system_prompt += f"\nAlways format your response in json format following this schema: {guided_decode_json_schema}"

        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": prompt},
        ]

        return self._generate_text(
            messages, 
            max_new_tokens=DEFAULT_MAX_TOKENS, 
            temperature=temperature, 
            top_p=top_p
        )

    @override
    def valid_models(self) -> list[str]:
        # Since this is local, technically any HF string is valid, but we can return the configured one
        return [self.model]

    @override
    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return self.query_with_system_prompt(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p
        )

    @override
    def query_multimodal(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return self.query_with_system_prompt(
            system_prompt=system_prompt,
            prompt=text_prompt,
            temperature=temperature,
            top_p=top_p
        )