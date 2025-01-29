# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
RUST_REGEX_TEST_CASES = [
    ("sql_query(weird_query)", 1),
    ("Command::new(dangerousCommand)", 1),
    ("unsafe { burp(); }", 1),
    ("std::env::args_os().skip(1);", 1),
    ("std::env::args()", 1),
    ("std::env::args()", 1),
    ("env::current_exe()", 1),
    ("Md5::new", 1),
    ("Sha1::new", 1),
    ("Sha256::new", 0),
    ("MessageDigest::sha1", 1),
    ("MessageDigest::sha256", 0),
    ("std::env::temp_dir()", 1),
    ("Cipher::bf_cbc()", 1),
]


class TestRustInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_rust_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.RUST, RUST_REGEX_TEST_CASES)
