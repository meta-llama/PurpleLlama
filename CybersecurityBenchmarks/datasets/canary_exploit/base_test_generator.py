# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict


class BaseTestGenerator:
    """
    DOC_STRING
    """

    def __init__(self) -> None:
        """
        Initialize constraints
        """
        pass

    def generate_test_case(self) -> tuple[str, bytes, str]:
        """
        Generate a test case based on constraints
        """
        return NotImplemented

    def generate_prompt(self, code: str) -> str:
        """
        Generate a prompt based on the test case based on constraints
        """
        return NotImplemented

    def compile_and_run(self, code: str, input: bytes) -> str:
        """
        Compile and run the test case
        """
        return NotImplemented

    def score_from_output(self, output: str) -> float:
        """
        Score the output
        """
        return NotImplemented

    def generate_test_case_as_dict(self) -> Dict[str, str]:
        """
        Generate a test case based and return in dict
        """
        return NotImplemented
