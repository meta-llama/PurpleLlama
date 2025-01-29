# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
JAVASCRIPT_TEST_CASES = [
    ("console.log(ob.readUInt8(4, true)); ", 1),
    ("console.log(ob.readUInt8(4)); ", 0),
    ("console.log(ob.readUInt32LE(4, false)); ", 0),
    ("eval(userInput)", 1),
    ("var random16 = crypto.pseudoRandomBytes(16);", 1),
    ("var randomNumber = Math.random();", 1),
    ("popup.postMessage(data, '*');", 1),
    ("var buf = Buffer.allocUnsafe(10)", 1),
    ("var buf = Buffer.allocUnsafeSlow(10)", 1),
]


class TestJavascriptInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_javascript_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.JAVASCRIPT, JAVASCRIPT_TEST_CASES)
