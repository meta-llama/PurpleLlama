# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

from CodeShield.insecure_code_detector import insecure_code_detector
from CodeShield.insecure_code_detector.languages import Language

try:
    from CodeShield.insecure_code_detector.internal import oss
except ImportError:
    from CodeShield.insecure_code_detector import oss

# the following test cases contain an input string, the insecure pattern match line number, and the expected rule
C_REGEX_TEST_CASES = [
    ("hello\nstrcpy(dest, src)", 2, "\\bstrcpy\\s*\\("),
    ("strcpy(dest, src)\nhello", 1, "\\bstrcpy\\s*\\("),
    ("strcpy(dest, src)", 1, "\\bstrcpy\\s*\\("),
]

CSHARP_SEMGREP_TEST_CASES = [
    (
        "//hello\nusing (var cryptoServiceProvider = System.Security.Cryptography.MD5.Create())",
        2,
    ),
    ("float randomRestPerc = UnityEngine.Random.Range(0, 101) / 100f;", 1),
]


class TestInsecureCodeDetectorFunctional(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(oss.ENABLE_REGEX_ANALYZER, "Regex analyzer not enabled")
    async def test_regex(self) -> None:
        for test_case in C_REGEX_TEST_CASES:
            test_input, expected_line_number, expected_rule = test_case
            result = await insecure_code_detector.analyze(Language.C, test_input)
            self.assertGreaterEqual(
                len(result), 1, "Test failed: {}".format(test_input)
            )
            self.assertEqual(
                result[0].line,
                expected_line_number,
                "Mismatch in the expected issue line number {}".format(test_input),
            )
            self.assertEqual(
                result[0].rule,
                expected_rule,
                "Mismatch in the expected rule name {}".format(test_input),
            )

    @unittest.skipUnless(oss.ENABLE_SEMGREP, "Semgrep analyzer not enabled")
    async def test_semgrep(self) -> None:
        for test_case in CSHARP_SEMGREP_TEST_CASES:
            test_input, expected_line_number = test_case
            result = await insecure_code_detector.analyze(
                Language.CSHARP,
                test_input,
            )
            print(result)
            self.assertGreater(len(result), 0, "Test failed: {}".format(test_input))
            self.assertEqual(
                result[0].line,
                expected_line_number,
                "Mismatch in the expected issue line number {}".format(test_input),
            )
