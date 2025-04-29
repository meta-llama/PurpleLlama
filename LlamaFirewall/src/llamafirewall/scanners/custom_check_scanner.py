# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import abstractmethod
from typing import Generic, Type, TypeVar

from llamafirewall.utils.base_llm import LLMClient

from pydantic import BaseModel

from ..llamafirewall_data_types import Message, ScanResult, Trace

from .base_scanner import Scanner

LOG: logging.Logger = logging.getLogger(__name__)

OutputSchemaT = TypeVar("OutputSchemaT", bound=BaseModel)


class CustomCheckScanner(Scanner, Generic[OutputSchemaT]):
    """
    [EXPERIMENTAL] A generic scanner that uses LLM prompts to evaluate content.
    This scanner can be configured with different prompts and output schemas
    to support various scanning use cases.
    """

    def __init__(
        self,
        scanner_name: str,
        system_prompt: str,
        output_schema: Type[OutputSchemaT],
        model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        api_base_url: str = "https://api.together.xyz/v1",
        api_key_env_var: str = "TOGETHER_API_KEY",
        temperature: float = 0.0,
        block_threshold: float = 0.0,
    ) -> None:
        """
        Initialize a new CustomCheckScanner.

        Args:
            scanner_name: Name of the scanner
            system_prompt: The system prompt to use for the LLM
            output_schema: Pydantic model defining the expected output schema
            model_name: Name of the LLM model to use
            api_base_url: Base URL for the API
            api_key_env_var: Environment variable name containing the API key
            temperature: Temperature setting for the LLM
            block_threshold: Threshold for blocking content
        """
        super().__init__(scanner_name=scanner_name, block_threshold=block_threshold)
        self.system_prompt = system_prompt
        self.output_schema = output_schema
        self.temperature = temperature

        # Initialize the LLM client
        self.llm: LLMClient[OutputSchemaT] = LLMClient(
            model_name=model_name,
            api_base_url=api_base_url,
            api_key_env_var=api_key_env_var,
        )

    async def _evaluate_with_llm(self, text: str) -> OutputSchemaT:
        try:
            response = await self.llm.call(
                prompt=text,
                system_prompt=self.system_prompt,
                output_schema=self.output_schema,
                temperature=self.temperature,
            )
            LOG.info(f"[LlamaFirewall] LLM-based scanner response: {response}")
            return response
        except Exception as e:
            LOG.error(f"Error in LLM evaluation: {e}")
            return self._get_default_error_response()

    @abstractmethod
    def _get_default_error_response(self) -> OutputSchemaT:
        """
        Returns a default response in case of error.
        """
        raise NotImplementedError("[LlamaFirewall] Default error response not defined")

    @abstractmethod
    async def _convert_llm_response_to_score(self, response: OutputSchemaT) -> float:
        """
        Returns a score that is between 0 and 1, from the LLM structured response.
        """
        raise NotImplementedError("[LlamaFirewall] LLM response conversion not defined")

    @abstractmethod
    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        """
        Scans the message and returns a ScanResult.

        This method must be implemented by concrete scanner implementations.
        """
        raise NotImplementedError("[LlamaFirewall] Scan method not implemented")
