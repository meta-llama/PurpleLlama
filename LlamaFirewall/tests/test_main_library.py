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
    UseCase,
    UserMessage,
)


class TestMainLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.lf = LlamaFirewall(
            scanners={
                Role.USER: [],
                Role.SYSTEM: [],
                Role.ASSISTANT: [ScannerType.CODE_SHIELD],
                Role.TOOL: [ScannerType.CODE_SHIELD],
                Role.MEMORY: [],
            }
        )

    def test_default_enabled_scanners(self) -> None:
        lf_input = AssistantMessage(
            content="hashlib.new('sha1')",
        )
        result = self.lf.scan(lf_input)
        self.assertEqual(result.decision, ScanDecision.BLOCK)
        self.assertGreaterEqual(result.score, 0.8)
        self.assertIn("unsafe function call", result.reason)

    def test_default_disabled_scanners(self) -> None:
        lf_input = UserMessage(
            content="hashlib.new('sha1')",
        )
        result = self.lf.scan(lf_input)
        self.assertEqual(result.decision, ScanDecision.ALLOW)
        self.assertLess(result.score, 0.5)
        self.assertIn("default", result.reason)

    def test_overwrite_enabled_scanners(self) -> None:
        overwritten_lf = LlamaFirewall(
            {
                Role.USER: [ScannerType.CODE_SHIELD],
                Role.SYSTEM: [],
                Role.ASSISTANT: [],
                Role.TOOL: [],
                Role.MEMORY: [],
            }
        )

        lf_input = UserMessage(
            content="hashlib.new('sha1')",
        )
        result = overwritten_lf.scan(lf_input)
        self.assertEqual(result.decision, ScanDecision.BLOCK)
        self.assertGreaterEqual(result.score, 0.8)
        self.assertIn("unsafe function call", result.reason)

        lf_input = AssistantMessage(
            content="hashlib.new('sha1')",
        )
        result = overwritten_lf.scan(lf_input)
        self.assertEqual(result.decision, ScanDecision.ALLOW)
        self.assertLess(result.score, 0.5)
        self.assertIn("default", result.reason)

    def test_construction_via_usecases(self) -> None:
        # Test for chatbot usecase, only Role.user and Role.system are enabled with prompt injection scanner
        lf = LlamaFirewall.from_usecase(UseCase.CHAT_BOT)
        self.assertEqual(lf.scanners[Role.USER], [ScannerType.PROMPT_GUARD])
        self.assertEqual(lf.scanners[Role.SYSTEM], [ScannerType.PROMPT_GUARD])
        self.assertTrue(Role.ASSISTANT not in lf.scanners)
        self.assertTrue(Role.TOOL not in lf.scanners)
        self.assertTrue(Role.MEMORY not in lf.scanners)

        # Test for code security usecase, only Role.assistant and Role.tool are enabled with code shield scanner
        lf = LlamaFirewall.from_usecase(UseCase.CODING_ASSISTANT)
        self.assertEqual(lf.scanners[Role.ASSISTANT], [ScannerType.CODE_SHIELD])
        self.assertEqual(lf.scanners[Role.TOOL], [ScannerType.CODE_SHIELD])
        self.assertTrue(Role.MEMORY not in lf.scanners)
        self.assertTrue(Role.USER not in lf.scanners)
        self.assertTrue(Role.SYSTEM not in lf.scanners)


if __name__ == "__main__":
    unittest.main()
