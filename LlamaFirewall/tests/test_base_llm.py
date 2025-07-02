# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llamafirewall.utils.base_llm import LLMClient

from pydantic import BaseModel


class TestOutputSchema(BaseModel):
    """Test schema for LLM output."""

    result: str
    observation: str


class TestLLMClient(unittest.TestCase):
    """Test cases for the LLMClient class."""

    def test_init_with_direct_api_key(self) -> None:
        """Test initialization with a direct API key."""
        client = LLMClient(
            model_name="test-model",
            api_base_url="https://api.example.com",
            api_key="test-api-key",
        )

        self.assertEqual(client.model_name, "test-model")
        self.assertIsNotNone(client.client)

    @patch.dict(os.environ, {"TEST_API_KEY": "env-api-key"})
    def test_init_with_env_var(self) -> None:
        """Test initialization with an API key from an environment variable."""
        client = LLMClient(
            model_name="test-model",
            api_base_url="https://api.example.com",
            api_key_env_var="TEST_API_KEY",
        )

        self.assertEqual(client.model_name, "test-model")
        self.assertIsNotNone(client.client)

    def test_init_missing_api_key(self) -> None:
        """Test initialization with missing API key raises ValueError."""
        with self.assertRaises(ValueError):
            LLMClient(
                model_name="test-model",
                api_base_url="https://api.example.com",
            )

    @patch.dict(os.environ, {})
    def test_init_missing_env_var(self) -> None:
        """Test initialization with missing environment variable raises ValueError."""
        with self.assertRaises(ValueError):
            LLMClient(
                model_name="test-model",
                api_base_url="https://api.example.com",
                api_key_env_var="NONEXISTENT_API_KEY",
            )

    @patch("openai.OpenAI")
    async def test_call_success(self, mock_openai: MagicMock) -> None:
        """Test successful LLM API call."""
        # Set up mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_parse = AsyncMock()
        mock_client.beta.chat.completions.parse = mock_parse

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.parsed = TestOutputSchema(
            result="test result", observation="test observation"
        )
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_parse.return_value = mock_response

        # Create client and make call
        client = LLMClient(
            model_name="test-model",
            api_base_url="https://api.example.com",
            api_key="test-api-key",
        )

        result = await client.call(
            prompt="Test prompt",
            system_prompt="Test system prompt",
            output_schema=TestOutputSchema,
            temperature=0.5,
        )

        # Verify result
        self.assertEqual(result.result, "test result")
        self.assertEqual(result.observation, "test observation")

        # Verify mock was called correctly
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["response_format"], TestOutputSchema)
        self.assertEqual(len(call_args["messages"]), 2)
        self.assertEqual(call_args["messages"][0]["role"], "system")
        self.assertEqual(call_args["messages"][0]["content"], "Test system prompt")
        self.assertEqual(call_args["messages"][1]["role"], "user")
        self.assertEqual(call_args["messages"][1]["content"], "Test prompt")


if __name__ == "__main__":
    unittest.main()
