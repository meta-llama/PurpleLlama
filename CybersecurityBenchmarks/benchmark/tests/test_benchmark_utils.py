# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

from ..benchmark_utils import extract_cot, extract_final_response


class TestBenchmarkUtils(unittest.TestCase):
    def test_reasoning_response_utils(self) -> None:
        llm_response = "Okay, the user wants me to write a Python3 script that writes 'Hello, World!'\n</think>\n\n```python\nprint('Hello, World!\\n')\n```"
        assert (
            extract_cot(llm_response, cot_end_tokens=["</think>"])
            == "Okay, the user wants me to write a Python3 script that writes 'Hello, World!'\n</think>\n\n"
        )
        assert (
            extract_final_response(llm_response, cot_end_tokens=["</think>"])
            == "```python\nprint('Hello, World!\\n')\n```"
        )
