# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from unittest.mock import MagicMock, patch

from .. import llm
from ..llms.llm_base import LLMConfig
from ..llms.openai import OPENAI


class TestLLM(unittest.TestCase):
    def test_create_without_base_url(self) -> None:
        llama = llm.create("TOGETHER::togethercomputer/llama-2-7b::<my API key>")
        self.assertTrue(isinstance(llama, llm.TOGETHER))
        self.assertEqual(llama.model, "togethercomputer/llama-2-7b")
        self.assertEqual(llama.api_key, "<my API key>")

        self.assertRaises(ValueError, lambda: llm.create("GPT::unknown::<my API key"))

    def test_create_with_base_url(self) -> None:
        """Test that create() function works with base_url parameter."""
        # Test 4-part format: PROVIDER::MODEL::API_KEY::BASE_URL
        openai_llm = llm.create(
            "OPENAI::gpt-4::test-key::https://custom-api.example.com/v1"
        )

        self.assertTrue(isinstance(openai_llm, llm.OPENAI))
        self.assertEqual(openai_llm.model, "gpt-4")
        self.assertEqual(openai_llm.api_key, "test-key")
        self.assertEqual(openai_llm.base_url, "https://custom-api.example.com/v1")

    def test_llm_config_with_base_url(self) -> None:
        """Test that LLMConfig properly handles base_url parameter."""
        config = LLMConfig(
            model="gpt-4",
            api_key="test-api-key",
            base_url="https://custom-openai-endpoint.com/v1/",
        )

        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertEqual(config.base_url, "https://custom-openai-endpoint.com/v1/")

    def test_llm_config_without_base_url(self) -> None:
        """Test that LLMConfig defaults base_url to None when not provided."""
        config = LLMConfig(model="gpt-4", api_key="test-api-key")

        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.api_key, "test-api-key")
        self.assertIsNone(config.base_url)

    def test_openai_llm_with_base_url(self) -> None:
        """Test that OPENAI LLM properly initializes with base_url."""
        config = LLMConfig(
            model="gpt-4",
            api_key="test-api-key",
            base_url="https://custom-openai-endpoint.com/v1/",
        )

        openai_llm = OPENAI(config)

        self.assertEqual(openai_llm.model, "gpt-4")
        self.assertEqual(openai_llm.api_key, "test-api-key")
        self.assertEqual(openai_llm.base_url, "https://custom-openai-endpoint.com/v1/")

        # Verify that the OpenAI client was initialized with the custom base_url
        self.assertEqual(
            str(openai_llm.client.base_url), "https://custom-openai-endpoint.com/v1/"
        )

    def test_openai_llm_without_base_url(self) -> None:
        """Test that OPENAI LLM works without base_url (uses default)."""
        config = LLMConfig(model="gpt-4", api_key="test-api-key")

        openai_llm = OPENAI(config)

        self.assertEqual(openai_llm.model, "gpt-4")
        self.assertEqual(openai_llm.api_key, "test-api-key")
        self.assertIsNone(openai_llm.base_url)

        # When base_url is None, OpenAI client should use default endpoint
        # The default OpenAI base_url is "https://api.openai.com/v1"
        self.assertEqual(str(openai_llm.client.base_url), "https://api.openai.com/v1/")


class TestOpenAIResponsesAPI(unittest.TestCase):
    """Tests for the ',responses' suffix that switches to the Responses API."""

    def test_responses_suffix_detection(self) -> None:
        """Model name with ',responses' suffix enables Responses API mode."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        self.assertTrue(openai_llm._use_responses_api)
        self.assertEqual(openai_llm.model, "gpt-4o")

    def test_no_responses_suffix(self) -> None:
        """Model name without ',responses' keeps Chat Completions mode."""
        config = LLMConfig(model="gpt-4o", api_key="test-key")
        openai_llm = OPENAI(config)

        self.assertFalse(openai_llm._use_responses_api)
        self.assertEqual(openai_llm.model, "gpt-4o")

    def test_responses_suffix_with_reasoning_model(self) -> None:
        """Reasoning model with ',responses' suffix is handled correctly."""
        config = LLMConfig(model="o3-mini,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        self.assertTrue(openai_llm._use_responses_api)
        self.assertEqual(openai_llm.model, "o3-mini")
        self.assertIn(openai_llm.model, openai_llm._reasoning_models)

    def test_create_with_responses_suffix(self) -> None:
        """The llm.create() factory propagates ',responses' correctly."""
        openai_llm = llm.create("OPENAI::gpt-4o,responses::test-key")

        self.assertIsInstance(openai_llm, OPENAI)
        self.assertTrue(openai_llm._use_responses_api)
        self.assertEqual(openai_llm.model, "gpt-4o")

    def test_create_with_responses_and_base_url(self) -> None:
        """The llm.create() factory works with ',responses' and a custom base URL."""
        openai_llm = llm.create(
            "OPENAI::gpt-4o,responses::test-key::https://custom.example.com/v1"
        )

        self.assertIsInstance(openai_llm, OPENAI)
        self.assertTrue(openai_llm._use_responses_api)
        self.assertEqual(openai_llm.model, "gpt-4o")
        self.assertEqual(openai_llm.base_url, "https://custom.example.com/v1")

    def test_query_uses_responses_api(self) -> None:
        """query() calls client.responses.create when Responses API is active."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "test response"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        result = openai_llm.query("Hello")

        self.assertEqual(result, "test response")
        openai_llm.client.responses.create.assert_called_once()
        call_kwargs = openai_llm.client.responses.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("model"), "gpt-4o")
        self.assertEqual(call_kwargs.kwargs.get("store"), False)
        self.assertEqual(call_kwargs.kwargs.get("input"), "Hello")

    def test_query_uses_chat_completions_without_suffix(self) -> None:
        """query() calls client.chat.completions.create without the suffix."""
        config = LLMConfig(model="gpt-4o", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        openai_llm.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        result = openai_llm.query("Hello")

        self.assertEqual(result, "chat response")
        openai_llm.client.chat.completions.create.assert_called_once()

    def test_chat_uses_responses_api(self) -> None:
        """chat() passes full history as input items to Responses API."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "response text"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        result = openai_llm.chat(["Hello", "Hi there!", "How are you?"])

        self.assertEqual(result, "response text")
        call_kwargs = openai_llm.client.responses.create.call_args
        expected_input = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        self.assertEqual(call_kwargs.kwargs.get("input"), expected_input)
        self.assertEqual(call_kwargs.kwargs.get("store"), False)

    def test_chat_with_system_prompt_uses_instructions(self) -> None:
        """chat_with_system_prompt() passes system prompt as 'instructions'."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "response text"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        result = openai_llm.chat_with_system_prompt(
            "You are helpful.", ["Hello", "Hi!", "Bye"]
        )

        self.assertEqual(result, "response text")
        call_kwargs = openai_llm.client.responses.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("instructions"), "You are helpful.")

    def test_query_with_system_prompt_uses_instructions(self) -> None:
        """query_with_system_prompt() passes system prompt as 'instructions'."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "response text"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        result = openai_llm.query_with_system_prompt("Be concise.", "Summarize AI.")

        self.assertEqual(result, "response text")
        call_kwargs = openai_llm.client.responses.create.call_args
        self.assertEqual(call_kwargs.kwargs.get("instructions"), "Be concise.")
        self.assertEqual(call_kwargs.kwargs.get("input"), "Summarize AI.")

    def test_responses_store_is_false(self) -> None:
        """All Responses API calls set store=False (no remote session storage)."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "ok"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        openai_llm.query("test")
        self.assertFalse(
            openai_llm.client.responses.create.call_args.kwargs.get("store")
        )

    def test_responses_json_schema_sets_text_format(self) -> None:
        """When guided JSON schema is provided, text format is set to json_object."""
        config = LLMConfig(model="gpt-4o,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = '{"key": "value"}'
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        openai_llm.query("test", guided_decode_json_schema='{"type": "object"}')

        call_kwargs = openai_llm.client.responses.create.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("text"), {"format": {"type": "json_object"}}
        )

    def test_responses_reasoning_model_uses_max_output_tokens(self) -> None:
        """Reasoning models set max_output_tokens instead of temperature/top_p."""
        config = LLMConfig(model="o3-mini,responses", api_key="test-key")
        openai_llm = OPENAI(config)

        mock_response = MagicMock()
        mock_response.output_text = "ok"
        openai_llm.client.responses.create = MagicMock(return_value=mock_response)

        openai_llm.query("test")

        call_kwargs = openai_llm.client.responses.create.call_args.kwargs
        self.assertIn("max_output_tokens", call_kwargs)
        self.assertNotIn("temperature", call_kwargs)
        self.assertNotIn("top_p", call_kwargs)

    def test_build_input_items(self) -> None:
        """_build_input_items correctly alternates user/assistant roles."""
        items = OPENAI._build_input_items(["q1", "a1", "q2", "a2", "q3"])
        self.assertEqual(
            items,
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ],
        )
