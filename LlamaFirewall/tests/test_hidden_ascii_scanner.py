# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from llamafirewall import (
    LlamaFirewall,
    Role,
    ScanDecision,
    ScannerType,
    ScanStatus,
    ToolMessage,
)
from llamafirewall.scanners.hidden_ascii_scanner import HiddenASCIIScanner


class TestHiddenASCIIScanner(unittest.TestCase):
    def setUp(self) -> None:
        self.lf = LlamaFirewall(
            scanners={
                Role.TOOL: [ScannerType.HIDDEN_ASCII],
            }
        )

    def test_hidden_input(self) -> None:
        inputs_with_hidden_content = [
            "󠁈󠁥󠁬󠁬󠁯󠀬󠀠󠁨󠁯󠁷󠀠󠁡󠁲󠁥󠀠󠁹󠁯󠁵󠀠󠁤󠁯󠁩󠁮󠁧󠀿",
        ]

        for input in inputs_with_hidden_content:
            lf_input = ToolMessage(
                content=input,
            )
            result = self.lf.scan(lf_input)
            self.assertEqual(result.decision, ScanDecision.BLOCK)
            self.assertGreaterEqual(result.score, 0.8)
            self.assertIn("Hidden ASCII", result.reason)
            self.assertEqual(result.status, ScanStatus.SUCCESS)

    def test_benign_input(self) -> None:
        benign_input = [
            "some_random_string",
        ]

        for input in benign_input:
            lf_input = ToolMessage(
                content=input,
            )
            result = self.lf.scan(lf_input)
            self.assertEqual(result.decision, ScanDecision.ALLOW)
            self.assertLess(result.score, 0.5)
            self.assertIn(HiddenASCIIScanner.ALLOW_REASON, result.reason)
            self.assertEqual(result.status, ScanStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
