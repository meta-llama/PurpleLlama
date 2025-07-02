# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import List, Optional

import openai

from typing_extensions import override

from ..benchmark_utils import image_to_b64
from .llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
    UnretriableQueryException,
)


class TOGETHER(LLM):
    """Accessing TOGETHER"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        messages = []
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for i in range(len(prompt_with_history)):
            if i % 2 == 0:
                messages.append({"role": "user", "content": prompt_with_history[i]})
            else:
                messages.append(
                    {"role": "assistant", "content": prompt_with_history[i]}
                )

        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 1,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        return response.choices[0].message.content

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
        # Together AI supports multiple images but not audio
        if audio_paths and len(audio_paths) > 0:
            raise UnretriableQueryException(
                "TOGETHER chat-completions does not support audio inputs yet."
            )

        if text_prompt is None and (image_paths is None or len(image_paths) == 0):
            raise ValueError(
                "At least one of text_prompt or image_paths must be given."
            )

        # Build OpenAI‑compatible message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Compose the user content as a list for multimodal prompts
        user_content: list[dict[str, str | dict[str, str]]] = []
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if image_paths and len(image_paths) > 0:
            for image_path in image_paths:
                image_data = image_to_b64(image_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )

        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "mistralai/Mistral-7B-v0.1",
            "lmsys/vicuna-7b-v1.5",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
            "togethercomputer/falcon-40b",
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        ]
