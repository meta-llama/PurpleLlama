# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

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

LOG: logging.Logger = logging.getLogger(__name__)


class LLAMA(LLM):
    """Accessing Meta Llama API"""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.llama.com/v1",
        )

    def _extract_response(self, response: Any) -> str:
        """
        Helper method to extract content from Meta Llama API response.

        Meta's API returns a different structure:
        response.completion_message = {
            (...)
            'content': {'type': 'text', 'text': 'actual response text'}
        }
        """
        text = ""

        try:
            if hasattr(response, "completion_message") and response.completion_message:
                completion_msg = response.completion_message

                # Extract the text content
                content = completion_msg.get("content", {})
                text = content.get("text")

        except AttributeError as e:
            raise UnretriableQueryException(
                f"Unexpected response structure from Meta Llama API: {e}. Response: {response}"
            )

        if text is None or text == "":
            raise ValueError("Extracted response is empty.")

        return text

    def _build_response_format(
        self, guided_decode_json_schema: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build response_format for Meta Llama API.

        Meta uses a different format than OpenAI:
        {
            "type": "json_schema",
            "json_schema": {
                "name": "ResponseSchema",
                "schema": { ... actual JSON schema ... }
            }
        }
        """

        try:
            # Parse the JSON schema if it is a string
            if isinstance(guided_decode_json_schema, str):
                schema = json.loads(guided_decode_json_schema)
            else:
                schema = guided_decode_json_schema

            # Build Meta's required format
            return {
                "type": "json_schema",
                "json_schema": {"name": "ResponseSchema", "schema": schema},
            }
        except json.JSONDecodeError as e:
            LOG.warning(
                f"Failed to parse JSON schema: {e}. Proceeding without response_format."
            )
            return None

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

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "top_p": top_p,
        }

        if guided_decode_json_schema is not None:
            response_format = self._build_response_format(guided_decode_json_schema)
            if response_format is not None:
                params["response_format"] = response_format
        response = self.client.chat.completions.create(**params)

        return self._extract_response(response)

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

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "top_p": top_p,
        }

        if guided_decode_json_schema is not None:
            response_format = self._build_response_format(guided_decode_json_schema)
            if response_format is not None:
                params["response_format"] = response_format
        response = self.client.chat.completions.create(**params)

        logging.getLogger().setLevel(level)

        return self._extract_response(response)

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "top_p": top_p,
        }

        if guided_decode_json_schema is not None:
            response_format = self._build_response_format(guided_decode_json_schema)
            if response_format is not None:
                params["response_format"] = response_format
        response = self.client.chat.completions.create(**params)

        return self._extract_response(response)

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": temperature,
            "top_p": top_p,
        }

        if guided_decode_json_schema is not None:
            response_format = self._build_response_format(guided_decode_json_schema)
            if response_format is not None:
                params["response_format"] = response_format
        response = self.client.chat.completions.create(**params)

        return self._extract_response(response)

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
        if audio_paths and len(audio_paths) > 0:
            raise UnretriableQueryException("Audio inputs are not supported yet.")

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
            # Llama models do not support more than 9 attachements
            if len(image_paths) > 9:
                LOG.warning(
                    f"Found {len(image_paths)} image_paths, but only using the first 9."
                )
                image_paths = image_paths[:9]

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

        return self._extract_response(response)

    @override
    def valid_models(self) -> list[str]:
        return [
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "Llama-4-Scout-17B-16E-Instruct-FP8",
        ]
