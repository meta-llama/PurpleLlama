# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
RUBY_TEST_CASES = [
    ("hello", 0),
    ("Open3.popen3('ls -la')", 1),
]


class TestRubyInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_ruby_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.RUBY, RUBY_TEST_CASES)
