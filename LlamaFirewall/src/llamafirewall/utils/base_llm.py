# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import Dict, Generic, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

logger: logging.Logger = logging.getLogger(__name__)

OutputSchemaT = TypeVar("OutputSchemaT", bound=BaseModel)


class LLMClient(Generic[OutputSchemaT]):
    """
    A client for interacting with LLM APIs.
    This provides a simple interface for making calls to LLM APIs with structured output.
    """

    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
    ) -> None:
        """
        Initialize a new LLMClient.

        Args:
            model_name: Name of the LLM model to use
            api_base_url: Base URL for the API
            api_key: API key for authentication (optional if api_key_env_var is provided)
            api_key_env_var: Environment variable name containing the API key (optional if api_key is provided)
        """
        self.model_name = model_name

        # Get API key either directly or from environment variable
        if api_key is None and api_key_env_var is not None:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"[LlamaFirewall] {api_key_env_var} is not set in the environment. Please set it as `export {api_key_env_var}=<your_api_key>` because your configured scanner requires it."
                )
        elif api_key is None and api_key_env_var is None:
            raise ValueError("Either api_key or api_key_env_var must be provided")

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url,
        )

    async def call(
        self,
        prompt: str,
        system_prompt: str,
        output_schema: Type[OutputSchemaT],
        temperature: float = 0.0,
        additional_messages: Optional[List[Dict[str, str]]] = None,
    ) -> OutputSchemaT:
        """
        Make a call to the LLM API with structured output.

        Args:
            prompt: The user prompt to send to the LLM
            system_prompt: The system prompt to use for the LLM
            output_schema: Pydantic model defining the expected output schema
            temperature: Temperature setting for the LLM (default: 0.0)
            additional_messages: Additional messages to include in the conversation (default: None)

        Returns:
            The LLM response parsed according to the output schema
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        # Add any additional messages if provided
        if additional_messages:
            messages.extend(additional_messages)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=output_schema,
                temperature=temperature,
            )
            result = response.choices[0].message.parsed
            return result
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            raise
