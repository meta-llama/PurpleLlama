# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
CPP_REGEX_TEST_CASES = [
    ("strcpy(dest, src)", 2),
    ('scanf("%s", buffer)', 1),
    ("strcat(dest, source)", 1),
    ("hello", 0),
    ("xmlReadDoc(buffer,NULL,NULL,XML_PARSE_DTDLOAD)", 1),
    # Bad RNG initializations
    ("std::mt19937{}", 1),
    ("std::default_random_engine(0)", 1),
    ("std::default_random_engine(12345 /* seed */)", 1),
    ("std::mt19937{std::random_device{}()}", 0),
]

CPP_SEMGREP_TEST_CASES = [
    # Integer overflow
    ("ptr = malloc(n * 5);", 1),
    ("if (n < SIZE_MAX/5) { ptr = new int[n * 5]; }", 0),
    ("ptr = new int[WIDTH * 5]", 0),
]


class TestCppInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_cpp_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.CPP, CPP_REGEX_TEST_CASES)

    async def test_cpp_semgrep_patterns(self) -> None:
        await self.run_semgrep_tests(Language.CPP, CPP_SEMGREP_TEST_CASES)
