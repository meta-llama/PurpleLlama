# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
SWIFT_TEST_CASES = [
    ("kSecAttrAccessible as String: kSecAttrAccessibleAlways]", 1),
    (
        "let attributes: [FileAttributeKey: Any] = [.protectionKey: FileProtectionType.none]",
        1,
    ),
    ("CC_SHA1(data.bytes, (CC_LONG)data.length, digest);", 1),
    ("CCAlgorithm(kCCAlgorithm3DES),", 1),
    ("var webView: UIWebView!", 1),
    ("class Person: NSObject, NSCoding {", 1),
    ("CCOptions(kCCOptionECBMode,", 1),
    ("settings[kCFStreamSSLAllowsAnyRoot] = NSNumber(value: true)", 1),
    ("settings[kCFStreamSSLValidatesCertificateChain] = NSNumber(value: false)", 1),
    (
        "CFReadStreamSetProperty(readStream, CFStreamPropertyKey(rawValue: kCFStreamSocketSecurityLevelSSLv3)",
        1,
    ),
    ("hello", 0),
]


class TestSwiftInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_swift_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.SWIFT, SWIFT_TEST_CASES)
