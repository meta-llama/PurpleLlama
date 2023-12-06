# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from .. import llm


class TestLLM(unittest.TestCase):
    def test_create(self) -> None:
        llama = llm.create("ANYSCALE::meta-llama/Llama-2-13b-chat-hf::<my API key>")
        self.assertTrue(isinstance(llama, llm.ANYSCALE))
        self.assertEquals(llama.model, "meta-llama/Llama-2-13b-chat-hf")
        self.assertEquals(llama.api_key, "<my API key>")

        self.assertRaises(ValueError, lambda: llm.create("GPT::unknown::<my API key"))
