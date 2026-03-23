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
    """Accessing OPENAI via Chat Completions or Responses API.

    The Responses API is selected by appending ',responses' to the model name
    in the specification string, e.g. 'OPENAI::gpt-4o,responses::API_KEY'.
    When the Responses API is active, conversation state is managed locally
    (full history is passed in each request; remote session storage is not used).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._reasoning_models = [
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-5-mini",
            "gpt-5-codex",
            "gpt-5.1-codex",
            "gpt-5.1-codex-mini",
            "gpt-5.1-codex-max",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.4-mini",
        ]
        self._use_responses_api: bool = config.model.endswith(",responses")
        if self._use_responses_api:
            config = LLMConfig(
                model=config.model[: -len(",responses")],
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                cot_end_tokens=config.cot_end_tokens,
                base_url=config.base_url,
            )
        super().__init__(config)
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    # -- Responses API helpers ------------------------------------------------

    def _responses_common_kwargs(
        self,
        temperature: float,
        top_p: float,
        guided_decode_json_schema: Optional[str] = None,
    ) -> dict:
        """Return the keyword arguments shared across all Responses API calls."""
        is_reasoning = self.model in self._reasoning_models
        kwargs: dict = {
            "model": self.model,
            "store": False,
        }
        if is_reasoning:
            kwargs["max_output_tokens"] = DEFAULT_MAX_TOKENS
        else:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if guided_decode_json_schema is not None:
            kwargs["text"] = {"format": {"type": "json_object"}}
        return kwargs

    @staticmethod
    def _build_input_items(prompt_with_history: List[str]) -> List[dict]:
        """Convert the alternating user/assistant string list into Responses
        API input items, preserving the full conversation locally."""
        items: List[dict] = []
        for i, text in enumerate(prompt_with_history):
            role = "user" if i % 2 == 0 else "assistant"
            items.append({"role": role, "content": text})
        return items

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self._use_responses_api:
            kwargs = self._responses_common_kwargs(
                temperature, top_p, guided_decode_json_schema
            )
            response = self.client.responses.create(
                input=self._build_input_items(prompt_with_history),
                **kwargs,
            )
            return response.output_text

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
        if self._use_responses_api:
            kwargs = self._responses_common_kwargs(
                temperature, top_p, guided_decode_json_schema
            )
            response = self.client.responses.create(
                input=self._build_input_items(prompt_with_history),
                instructions=system_prompt,
                **kwargs,
            )
            return response.output_text

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
        if self._use_responses_api:
            kwargs = self._responses_common_kwargs(
                temperature, top_p, guided_decode_json_schema
            )
            instructions = NOT_GIVEN
            if guided_decode_json_schema is not None:
                instructions = (
                    "You are a helpful assistant. Always format your response"
                    f" in json format following this schema: {guided_decode_json_schema} "
                )
            level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)
            response = self.client.responses.create(
                input=prompt,
                instructions=instructions,
                **kwargs,
            )
            logging.getLogger().setLevel(level)
            return response.output_text

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
        if self._use_responses_api:
            kwargs = self._responses_common_kwargs(
                temperature, top_p, guided_decode_json_schema
            )
            level = logging.getLogger().level
            logging.getLogger().setLevel(logging.WARNING)
            response = self.client.responses.create(
                input=prompt,
                instructions=system_prompt,
                **kwargs,
            )
            logging.getLogger().setLevel(level)
            return response.output_text

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

        if self._use_responses_api:
            kwargs = self._responses_common_kwargs(
                temperature, top_p, guided_decode_json_schema
            )
            response = self.client.responses.create(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64, {image}",
                            },
                        ],
                    }
                ],
                instructions=system_prompt,
                **kwargs,
            )
            return response.output_text

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
        if self._use_responses_api:
            user_content: List[dict] = []
            if text_prompt:
                user_content.append({"type": "input_text", "text": text_prompt})
            if image_paths:
                for image_path in image_paths:
                    image_data = image_to_b64(image_path)
                    user_content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64, {image_data}",
                        }
                    )
            if audio_paths:
                for audio_path in audio_paths:
                    user_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_to_b64(str(audio_path)),
                                "format": "wav",
                            },
                        }
                    )
            kwargs = self._responses_common_kwargs(temperature, top_p)
            response = self.client.responses.create(
                input=[{"role": "user", "content": user_content}],
                instructions=system_prompt if system_prompt else NOT_GIVEN,
                **kwargs,
            )
            return response.output_text

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
