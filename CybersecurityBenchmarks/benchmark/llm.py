# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging

import os
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import openai
import requests
from openai._types import NOT_GIVEN

from typing_extensions import override

from .benchmark_utils import audio_to_b64, get_mime_type, image_to_b64
from .cache_handler import CacheHandler


NUM_LLM_RETRIES = 100
MAX_RETRY_TIMEOUT = 600

LOG: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


class RetriesExceededException(Exception):
    def __init__(
        self, message: str = "Maximum number of retries has been exceeded"
    ) -> None:
        self.message = message
        super().__init__(self.message)


class UnretriableQueryException(Exception):
    pass


@dataclass
class LLMConfig:
    model: str
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    cot_end_tokens: list[str] | None = None


class LLM(ABC):
    def __init__(
        self,
        config: LLMConfig,
    ) -> None:
        self.model: str = config.model
        self.api_key: str | None = config.api_key
        self.cache_handler: CacheHandler | None = None
        self.max_tokens: int = config.max_tokens
        self.cot_end_tokens: list[str] | None = config.cot_end_tokens
        if self.model not in self.valid_models():
            LOG.warning(
                f"{self.model} is not in the valid model list for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )

    @abstractmethod
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts and return the response.

        Args:
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    @abstractmethod
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to chat with an LLM with given prompts as well as system prompt and return the response.

        Args:
            system_prompt (str): The system prompt to send to the LLM.
            prompt_with_history ([str]): The prompt to send to the LLM along with chat history. The last string in the list is the prompt.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            system prompt (str): The system prompt to send to the LLM.
            prompt (str): The prompt to send to the LLM.
            guided_decode_json_schema (Optional[str]): JSON schema for guided decoding.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            str: The response from the LLM.
        """
        return self.query(
            system_prompt + "\n" + prompt, guided_decode_json_schema, temperature, top_p
        )

    def _query_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[str, bool, Optional[str], float, float],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error {type(exception)}: {textwrap.shorten(str(exception), width=250)}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def _chat_with_retries(
        self,
        func: Callable[..., str],
        *args: Union[List[str], bool, Optional[str], float, float],
        retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        backoff_factor: float = 0.5,
    ) -> str:
        last_exception = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Chat query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        raise RetriesExceededException(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def query_with_retries(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(self.model, None, prompt)
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query, prompt, guided_decode_json_schema, temperature, top_p
        )

        if self.cache_handler:
            self.cache_handler.save_response(self.model, None, prompt, response)

        return response

    def chat_with_retries(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, None, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, None, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, prompt
            )
            if cached_response:
                return cached_response
        response = self._query_with_retries(
            self.query_with_system_prompt,
            system_prompt,
            prompt,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

        if self.cache_handler:
            self.cache_handler.save_response(
                self.model, system_prompt, prompt, response
            )

        return response

    def chat_with_system_prompt_with_retries(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if self.cache_handler:
            cached_response = self.cache_handler.get_response_chat(
                self.model, system_prompt, prompt_with_history
            )
            if cached_response:
                return cached_response
        response = self._chat_with_retries(
            self.chat_with_system_prompt,
            system_prompt,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )
        if self.cache_handler:
            self.cache_handler.save_response_chat(
                self.model, system_prompt, prompt_with_history, response
            )
        return response

    def query_with_system_prompt_and_image(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Abstract method to query an LLM with a given prompt, system prompt, and image and return the response.
        """
        LOG.error(
            "[ERROR] Please override this method in a subclass to use this method."
        )
        return "Function not implemented."

    def query_with_system_prompt_and_image_with_retries(
        self,
        system_prompt: str,
        prompt: str,
        image_path: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return self._query_with_retries(
            self.query_with_system_prompt_and_image,
            system_prompt,
            prompt,
            image_path,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

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
        """
        Abstract method to query an LLM and return the response. This method
        supports multimodal queries with text, images and audio, and usage of a system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_paths (Optional[List[str]]): The paths to the images to send to the LLM.
            audio_paths (Optional[List[str]]): The paths to the audio files to send to the LLM.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        raise NotImplementedError(
            "Please override this method in a subclass to use this method."
        )

    def query_multimodal_with_retries(
        self,
        system_prompt: Optional[str] = None,
        text_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        num_retries: int = NUM_LLM_RETRIES,
        max_retry_timeout: int = MAX_RETRY_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        backoff_factor: float = 0.5,
    ) -> Optional[str]:
        """
        Abstract method to query an LLM and return the response, wrapped in retries.
        This method supports multimodal queries and system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to send to the LLM.
            text_prompt (Optional[str]): The prompt to send to the LLM.
            image_paths (Optional[List[str]]): The paths to the images to send to the LLM.
            audio_paths (Optional[List[str]]): The paths to the audio files to send to the LLM.
            num_retries (int): Number of retries to attempt.
            max_retry_timeout (int): Maximum timeout between retries.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Controls randomness in the output.
            top_p (float): Controls diversity via nucleus sampling.
            backoff_factor (float): Factor to increase backoff time between retries.

        Returns:
            Optional[str]: The response from the LLM, or None if it failed to query.
        """
        cache_key = json.dumps(
            {
                "text_prompt": text_prompt,
                "image_paths": (
                    sorted(image_paths) if image_paths is not None else None
                ),
                "audio_paths": (
                    sorted(audio_paths) if audio_paths is not None else None
                ),
            }
        )
        if self.cache_handler:
            cached_response = self.cache_handler.get_response(
                self.model, system_prompt, cache_key
            )
            if cached_response:
                return cached_response

        last_exception = None
        for retry in range(num_retries):
            try:
                response = self.query_multimodal(
                    system_prompt=system_prompt,
                    text_prompt=text_prompt,
                    image_paths=image_paths,
                    audio_paths=audio_paths,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if self.cache_handler:
                    self.cache_handler.save_response(
                        self.model, system_prompt, cache_key, response
                    )
                return response
            except NotImplementedError as exception:
                LOG.error(
                    f"LLM Query failed as the method was not implemented: {exception}"
                )
                raise exception
            except UnretriableQueryException as exception:
                LOG.error(f"LLM Query failed due to unretriable error: {exception}")
                return None
            except Exception as exception:
                last_exception = exception
                sleep_time = (
                    backoff_factor * (2**retry)
                    if backoff_factor * (2**retry) <= max_retry_timeout
                    else max_retry_timeout
                )
                time.sleep(sleep_time)
                LOG.error(
                    f"LLM Query failed with error: {exception}. Sleeping for {sleep_time} seconds..."
                )
        LOG.error(f"Unable to query LLM after {num_retries} retries: {last_exception}")
        return None

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []


SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"


def create(
    identifier: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cot_end_tokens: List[str] | None = None,
) -> LLM:
    """
    Takes a string of the format <SPECIFICATION_FORMAT> and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.
    """
    api_key: str | None = None
    split = identifier.split("::")
    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    if api_key is None:
        raise ValueError(f"No API key provided for {provider}: {name}")

    config = LLMConfig(
        model=name,
        api_key=api_key,
        max_tokens=max_tokens,
        cot_end_tokens=cot_end_tokens,
    )

    if provider == "OPENAI":
        return OPENAI(config)
    if provider == "TOGETHER":
        return TOGETHER(config)
    if provider == "ANTHROPIC":
        return ANTHROPIC(config)
    if provider == "GOOGLEGENAI":
        return GOOGLEGENAI(config)

    raise ValueError(f"Unknown provider: {provider}")


class OPENAI(LLM):
    """Accessing OPENAI"""

    def __init__(self, config: LLMConfig) -> None:
        self._reasoning_models = ["o1", "o1-mini", "o3", "o3-mini", "o4-mini"]
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
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
            max_tokens=DEFAULT_MAX_TOKENS
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            max_completion_tokens=DEFAULT_MAX_TOKENS
            if (self.model in self._reasoning_models)
            else NOT_GIVEN,
            temperature=temperature
            if (self.model not in self._reasoning_models)
            else NOT_GIVEN,
            top_p=top_p if (self.model not in self._reasoning_models) else NOT_GIVEN,
        )
        return response.choices[0].message.content


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


class ANTHROPIC(LLM):
    """Accessing Anthropic models"""

    BASE_API_URL: str = "https://api.anthropic.com/v1"

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        url = f"{self.BASE_API_URL}/messages"
        # Define the headers
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Define the data payload
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            logging.info(
                f"Failed to extract content from Anthropic API response.  Response: {response}"
            )
            return "FAIL TO QUERY: no content in response"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            logging.info(
                f"Failed to extract text from Anthropic API response content.  Content: {content}"
            )
            return "FAIL TO QUERY: no text in response"  # Anthropic didn't return a response
        return text_value

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for ANTHROPIC."
            )
        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "system": system_prompt,
        }
        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            return "FAILED TO QUERY"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY"  # Anthropic didn't return a response
        return text_value

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
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for ANTHROPIC."
            )
        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_to_b64(image_path),
                            },
                        },
                    ],
                }
            ],
            "system": system_prompt,
        }
        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            return "FAILED TO QUERY"  # Anthropic didn't return a response
        text_value = content[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY"  # Anthropic didn't return a response
        return text_value

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
        # Anthropic doesn't support audio inputs yet
        if audio_paths:
            raise ValueError("Anthropic does not support audio inputs yet")

        url = f"{self.BASE_API_URL}/messages"
        headers = {
            "x-api-key": f"{self.api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        content = []
        if text_prompt:
            content.append({"type": "text", "text": text_prompt})

        if image_paths:
            for image_path in image_paths:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_b64(image_path),
                        },
                    }
                )

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

        if system_prompt:
            data["system"] = system_prompt

        response = requests.post(url, headers=headers, json=data)
        json_response = response.json()
        content = json_response.get("content", [])
        if not content:
            raise UnretriableQueryException(
                f"Failed to obtain a response for {self.model}, content was null"
            )
        text_value = content[0].get("text", None)
        if not text_value:
            raise UnretriableQueryException(
                f"Failed to obtain a response for {self.model}, content text was null"
            )
        return text_value

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Chat API not implemented yet.")

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("System prompt not implemented yet.")

    @override
    def valid_models(self) -> list[str]:
        return [
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
        ]


class GOOGLEGENAI(LLM):
    """Accessing Google Generative AI"""

    BASE_API_URL: str = "https://generativelanguage.googleapis.com"

    @override
    def query(
        self,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": prompt}], "role": "user"}]}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        json_response = response.json()
        response_str = json.dumps(json_response)
        if '[{"finishReason": "OTHER", "index": 0}]' in response_str:
            return "FAILED TO QUERY"  # Gemini didn't return a response
        text_value = (
            json_response.get("candidates", [])[0]
            .get("content", {})
            .get("parts", [])[0]
            .get("text", None)
        )
        return text_value

    @override
    def query_with_system_prompt(
        self,
        system_prompt: str,
        prompt: str,
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for GOOGLEGENAI."
            )

        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}], "role": "user"}],
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
                "role": "model",
            },
            "generationConfig": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        json_response = response.json()
        response_str = json.dumps(json_response)
        if '[{"finishReason": "OTHER", "index": 0}]' in response_str:
            return "FAILED TO QUERY"  # Gemini didn't return a response
        text_value = (
            json_response.get("candidates", [])[0]
            .get("content", {})
            .get("parts", [])[0]
            .get("text", None)
        )
        return text_value

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
        if guided_decode_json_schema is not None:
            raise NotImplementedError(
                "guided_decode_json_schema not implemented for GOOGLEGENAI."
            )

        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_to_b64(image_path),
                            },
                        },
                    ],
                    "role": "user",
                }
            ],
            "systemInstruction": {
                "parts": [
                    {
                        "text": system_prompt,
                    }
                ],
                "role": "model",
            },
            "generationConfig": {
                "maxOutputTokens": 2048,
            },
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        json_response = response.json()
        candidates = json_response.get("candidates", [])
        if not candidates:
            return "FAILED TO QUERY"  # Gemini didn't return a response
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "FAILED TO QUERY"  # Gemini didn't return a response
        text_value = parts[0].get("text", None)
        if not text_value:
            return "FAILED TO QUERY"  # Gemini didn't return a response
        return text_value

    def _upload_file(
        self,
        file_path: str,
    ) -> str:
        mime_type = get_mime_type(file_path)
        num_bytes = os.path.getsize(file_path)
        display_name = "MEDIA_FILE"
        # Initial resumable request defining metadata
        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(num_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        data = json.dumps({"file": {"display_name": display_name}})

        response = requests.post(
            f"{self.BASE_API_URL}/upload/v1beta/files?key={self.api_key}",
            headers=headers,
            data=data,
        )
        response.raise_for_status()

        upload_url = response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise Exception(f"Failed to upload file ({file_path})")

        # Upload the actual bytes
        with open(file_path, "rb") as f:
            file_data = f.read()
        headers = {
            "Content-Length": str(num_bytes),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        response = requests.post(upload_url, headers=headers, data=file_data)
        file_info = response.json()
        file_uri = file_info.get("file", {}).get("uri")

        if not file_uri:
            raise Exception("Failed to upload file.")

        return file_uri

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
        url = f"{self.BASE_API_URL}/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        headers = {"Content-Type": "application/json"}

        content_parts = []
        if text_prompt:
            content_parts.append({"text": text_prompt})

        file_uris = []
        if image_paths:
            for image_path in image_paths:
                file_uri = self._upload_file(image_path)
                file_uris.append(file_uri)
                image_mimetype = get_mime_type(image_path)

                content_parts.append(
                    {"file_data": {"mime_type": image_mimetype, "file_uri": file_uri}}
                )

        if audio_paths:
            for audio_path in audio_paths:
                file_uri = self._upload_file(audio_path)
                file_uris.append(file_uri)
                audio_mimetype = get_mime_type(audio_path)
                content_parts.append(
                    {"file_data": {"mime_type": audio_mimetype, "file_uri": file_uri}}
                )

        data: Dict[str, Any] = {
            "contents": [{"parts": content_parts, "role": "user"}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            data["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
                "role": "model",
            }

        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Delete files after query
        for file_uri in file_uris:
            delete_url = f"{file_uri}?key={self.api_key}"
            delete_response = requests.delete(delete_url)
            if not delete_response.ok:
                print(f"[WARNING] Failed to delete file: {delete_response.text}")

        if not response.ok:
            raise Exception(
                f"Got non-200 status code when querying model: {response.status_code}"
            )

        json_response = response.json()

        candidates = json_response.get("candidates", [])
        if not candidates:
            raise UnretriableQueryException(
                f"Failed to query model, got no candidates: {json_response}"
            )
        content = candidates[0].get("content", {})
        if not content:
            raise UnretriableQueryException(
                f"Failed to query model, got no content from first candidate: {candidates[0]}"
            )

        finish_reason = candidates[0].get("finishReason")
        if finish_reason != "STOP":
            raise UnretriableQueryException(
                f"Failed to query model, likely due to safety violation. Got unexpected finish reason: {finish_reason}"
            )

        parts = content.get("parts", [])
        if not parts:
            raise UnretriableQueryException(
                f"Failed to query model, got no content parts in first candidate content: {content}"
            )
        text_value = parts[0].get("text", None)
        if not text_value:
            raise UnretriableQueryException(
                f"Failed to query model, candidate content parts did not return text value: {parts[0]}"
            )

        return text_value

    @override
    def chat(
        self,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        raise NotImplementedError("Chat API not implemented yet.")

    @override
    def chat_with_system_prompt(
        self,
        system_prompt: str,
        prompt_with_history: List[str],
        guided_decode_json_schema: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        return super().chat_with_system_prompt(
            system_prompt,
            prompt_with_history,
            guided_decode_json_schema,
            temperature,
            top_p,
        )

    @override
    def valid_models(self) -> list[str]:
        return [
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ]
