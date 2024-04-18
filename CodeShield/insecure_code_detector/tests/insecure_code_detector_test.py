# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest
from abc import ABC

from .. import insecure_code_detector
from ..languages import Language

try:
    from ..internal import oss
except ImportError:
    from .. import oss


class InsecureCodeDetectorTest(unittest.IsolatedAsyncioTestCase, ABC):
    @unittest.skipUnless(oss.ENABLE_REGEX_ANALYZER, "Regex analyzer not enabled")
    async def run_regex_tests(
        self, language: Language, tests: list[tuple[str, int]]
    ) -> None:
        for test_case in tests:
            test_input, expected_count = test_case
            result = await insecure_code_detector.analyze(language, test_input)
            print(test_case)
            print(result)
            self.assertEqual(len(result), expected_count)

    @unittest.skipUnless(oss.ENABLE_SEMGREP, "Semgrep analyzer not enabled")
    async def run_semgrep_tests(
        self, language: Language, tests: list[tuple[str, int]]
    ) -> None:
        for test_case in tests:
            test_input, expected_count = test_case
            result = await insecure_code_detector.analyze(language, test_input)
            print(test_case)
            print(result)
            self.assertEqual(
                len(result), expected_count, "Test failed: {}".format(test_input)
            )
