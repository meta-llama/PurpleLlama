# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
CSHARP_REGEX_TEST_CASES = [
    ('<system.web><pages viewStateEncryptionMode="Auto" /></system.web>', 1),
    ('<system.web><pages enableViewStateMac="false" /></system.web>', 1),
]


class TestCSharpInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_csharp_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.CSHARP, CSHARP_REGEX_TEST_CASES)
