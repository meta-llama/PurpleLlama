# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
OBJECTIVE_C_TEST_CASES = [
    ("[keychainAttributes setObject:(id)kSecAttrAccessibleAlways", 1),
    ("@{NSFileProtectionKey : NSFileProtectionNone}", 1),
    ("CC_SHA1(data.bytes, (CC_LONG)data.length, digest);", 1),
    ("CCAlgorithm(kCCAlgorithm3DES),", 1),
    ("[[UIWebView alloc] initWithFrame:CGRectMake(0.0, 0.0, 1.0, 1.0)];", 1),
    ("@interface MyObject : NSObject <NSCoding>,", 1),
    ("CCOptions(kCCOptionECBMode,", 1),
    ("@YES, kCFStreamSSLAllowsAnyRoot,", 1),
    ("(NSString *)kCFStreamSocketSecurityLevelSSLv3, kCFStreamSSLLevel,,", 1),
    ("hello", 0),
]


class TestObjectiveCInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_objective_c_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.OBJECTIVE_C, OBJECTIVE_C_TEST_CASES)
