# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import List, Optional

import openai
from openai._types import NOT_GIVEN

from typing_extensions import override

from ..benchmark_utils import audio_to_b64, image_to_b64
from .llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
)


class OPENAI(LLM):
    """Accessing OPENAI"""

    def __init__(self, config: LLMConfig) -> None:
        self._reasoning_models = [
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5-mini",
        ]
        super().__init__(config)
        self.client = openai.OpenAI(api_key=self.api_key)  # noqa

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        messages = []
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
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
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
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
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
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        # Best-level effort to suppress openai log-spew.
        # Likely not work well in multi-threaded environment.
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        if guided_decode_json_schema is not None:
            message_list = [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. Always format your response in json format following this schema: {guided_decode_json_schema} """,
                }
            ]
        else:
            message_list = []
        message_list.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_list,
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
            response_format=(
                {"type": "json_object"}
                if guided_decode_json_schema is not None
                else None
            ),
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4-turbo",
        ] + self._reasoning_models

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
        image = image_to_b64(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64, {image}"},
                        },
                    ],
                },
            ],
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_message_content = []
        if text_prompt:
            user_message_content.append({"type": "text", "text": text_prompt})

        if image_paths:
            for image_path in image_paths:
                image_data = image_to_b64(image_path)
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64, {image_data}"},
                    }
                )

        if audio_paths:
            for audio_path in audio_paths:
                user_message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_to_b64(str(audio_path)),
                            "format": "wav",
                        },
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": user_message_content,
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=(
                DEFAULT_MAX_TOKENS
                if (self.model in self._reasoning_models)
                else NOT_GIVEN
            ),
            temperature=(
                temperature if (self.model not in self._reasoning_models) else NOT_GIVEN
            ),
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
        )
        return response.choices[0].message.content
