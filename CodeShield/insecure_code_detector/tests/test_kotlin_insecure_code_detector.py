# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
KOTLIN_TEST_CASES = []  # type: list[tuple[str,int]]


class TestKotlinInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_kotlin_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.KOTLIN, KOTLIN_TEST_CASES)
