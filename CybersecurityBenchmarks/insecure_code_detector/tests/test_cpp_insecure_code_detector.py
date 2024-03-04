# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ..languages import Language
from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
CPP_REGEX_TEST_CASES = [
    ("strcpy(dest, src)", 1),
    ('scanf("%s", buffer)', 1),
    ("strcpy(dest, source)", 1),
    ("strcat(dest, source)", 1),
    ("hello", 0),
    ("xmlReadDoc(buffer,NULL,NULL,XML_PARSE_DTDLOAD)", 1),
    # Bad RNG initializations
    ("std::mt19937{}", 1),
    ("std::default_random_engine(0)", 1),
    ("std::default_random_engine(12345 /* seed */)", 1),
    ("std::mt19937{std::random_device{}()}", 0),
]

CPP_WEGGLI_TEST_CASES = [
    # Dummy C style rule to ensure we run them on C++ code
    ("ptr = malloc(n * 5);", 1),
    # Integer overflow
    ("ptr = new int[n * 5];", 1),
    ("if (n < SIZE_MAX/5) { ptr = new int[n * 5]; }", 0),
    ("ptr = new int[WIDTH * 5]", 0),
    # Invalid free
    ("char buf[10]; delete buf;", 1),
    ("char ptr; delete &ptr;", 1),
    # Use-After-Free due to iterator invalidation
    ("for (it = v.begin(); it !+ v.end(); ++it) { v.push_back(4); } }", 1),
    ("for ( el : v ) { v.push_back(4); }", 1),
    # Stack buffer overflow
    ("char buf[10]; cin >> buf;", 1),
    ("char buf[10]; cin.width(9); cin >> buf;", 0),
]


class TestCppInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_cpp_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.CPP, CPP_REGEX_TEST_CASES)

    async def test_cpp_weggli_patterns(self) -> None:
        await self.run_weggli_tests(Language.CPP, CPP_WEGGLI_TEST_CASES)
