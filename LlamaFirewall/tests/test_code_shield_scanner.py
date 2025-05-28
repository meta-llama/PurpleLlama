# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Role,
    ScanDecision,
    ScannerType,
    ScanStatus,
)
from llamafirewall.scanners.code_shield_scanner import CodeShieldScanner


class TestCodeShieldScanner(unittest.TestCase):
    def setUp(self) -> None:
        self.lf = LlamaFirewall(
            scanners={
                Role.USER: [],
                Role.SYSTEM: [],
                Role.ASSISTANT: [ScannerType.CODE_SHIELD],
                Role.TOOL: [],
                Role.MEMORY: [],
            }
        )

    def test_insecure_code(self) -> None:
        insecure_code = [
            "```hashlib.new('sha1')```",
            "```Test customCipher = new NullCipher();```",
        ]

        for code in insecure_code:
            lf_input = AssistantMessage(
                content=code,
            )
            result = self.lf.scan(lf_input)
            self.assertEqual(result.decision, ScanDecision.BLOCK)
            self.assertGreaterEqual(result.score, 0.8)
            self.assertIn("unsafe function call", result.reason)
            self.assertEqual(result.status, ScanStatus.SUCCESS)

    def test_secure_code(self) -> None:
        secure_code = [
            "```import hashlib```",
        ]

        for code in secure_code:
            lf_input = AssistantMessage(
                content=code,
            )
            result = self.lf.scan(lf_input)
            self.assertEqual(result.decision, ScanDecision.ALLOW)
            self.assertLess(result.score, 0.5)
            self.assertIn(CodeShieldScanner.ALLOW_REASON, result.reason)
            self.assertEqual(result.status, ScanStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
