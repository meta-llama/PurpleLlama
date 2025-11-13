# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

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
