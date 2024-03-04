# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

from .. import llm


class TestLLM(unittest.TestCase):
    def test_create(self) -> None:
        llama = llm.create("TOGETHER::togethercomputer/llama-2-7b::<my API key>")
        self.assertTrue(isinstance(llama, llm.TOGETHER))
        self.assertEqual(llama.model, "togethercomputer/llama-2-7b")
        self.assertEqual(llama.api_key, "<my API key>")

        self.assertRaises(ValueError, lambda: llm.create("GPT::unknown::<my API key"))
