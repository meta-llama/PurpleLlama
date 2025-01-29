# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
PYTHON_TEST_CASES = [
    ("import os", 0),
    ("import hashlib", 0),
    ("from hashlib import sha256", 0),
    ("hashlib.new('sha1')", 1),
    ("hashlib.md5()", 1),
]


class TestPythonInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_python_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.PYTHON, PYTHON_TEST_CASES)
