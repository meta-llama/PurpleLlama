# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from CodeShield.insecure_code_detector.languages import Language

from .insecure_code_detector_test import InsecureCodeDetectorTest

# the following test cases contain an input string, and the corresponding number of expected insecure pattern matches
XML_TEST_CASES = [
    (
        """<?xml version="1.0" encoding="utf-8"?>
            <manifest
                <application
                    android:debuggable="true">
                </application>
            </manifest>
        """,
        1,
    ),
    (
        """<?xml version="1.0" encoding="utf-8"?>
            <manifest
                <application
                    android:allowBackup="true">
                </application>
            </manifest>
        """,
        1,
    ),
    ('<domain-config cleartextTrafficPermitted="true">', 1),
    (
        """<?xml version="1.0" encoding="utf-8"?>
            <manifest
                <application
                    android:usesCleartextTraffic="true">
                </application>
            </manifest>
        """,
        1,
    ),
]


class TestXmlInsecureCodeDetector(InsecureCodeDetectorTest):
    async def test_xml_regex_patterns(self) -> None:
        await self.run_regex_tests(Language.XML, XML_TEST_CASES)
