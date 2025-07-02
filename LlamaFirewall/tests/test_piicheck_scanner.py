# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
from unittest.mock import AsyncMock, patch

from llamafirewall import (
    LlamaFirewall,
    Role,
    ScanDecision,
    ScannerType,
    ScanStatus,
    UserMessage,
)
from llamafirewall.scanners.experimental.piicheck_scanner import PIICheckOutputSchema


class TestPIICheckScanner(unittest.TestCase):
    def setUp(self) -> None:
        self.lf = LlamaFirewall(
            scanners={
                Role.USER: [ScannerType.PII_DETECTION],
            }
        )
        os.environ["TOGETHER_API_KEY"] = "dummy_key_for_tests"

    @patch(
        "llamafirewall.scanners.experimental.piicheck_scanner.PIICheckScanner._convert_llm_response_to_score",
        new_callable=AsyncMock,
        return_value=1.0,
    )
    @patch(
        "llamafirewall.scanners.custom_check_scanner.CustomCheckScanner._evaluate_with_llm",
        new_callable=AsyncMock,
        return_value=PIICheckOutputSchema(
            detected_pii_types=[
                "Full names",
                "Email addresses",
                "Phone numbers",
            ],
        ),
    )
    def test_high_risk_inputs(
        self, mock_call_llm: AsyncMock, mock_convert_score: AsyncMock
    ) -> None:
        # These mocks are injected by the patch decorator and are required for the test to work
        # even though they're not explicitly used in the method body
        _ = mock_call_llm  # Silence unused variable warning
        _ = mock_convert_score  # Silence unused variable warning

        message = UserMessage(
            content="Hi, my name is John Smith. You can reach me at john.smith@example.com or call me at (555) 123-4567.",
        )
        result = self.lf.scan(message)
        self.assertEqual(result.decision, ScanDecision.BLOCK)
        self.assertAlmostEqual(result.score, 1.0)
        self.assertIn("PII detected", result.reason)
        self.assertEqual(result.status, ScanStatus.SUCCESS)

    @patch(
        "llamafirewall.scanners.experimental.piicheck_scanner.PIICheckScanner._convert_llm_response_to_score",
        new_callable=AsyncMock,
        return_value=0.0,
    )
    @patch(
        "llamafirewall.scanners.custom_check_scanner.CustomCheckScanner._evaluate_with_llm",
        new_callable=AsyncMock,
        return_value=PIICheckOutputSchema(
            detected_pii_types=[],
        ),
    )
    def test_low_risk_inputs(
        self, _mock_call_llm: AsyncMock, _mock_convert_score: AsyncMock
    ) -> None:
        message = UserMessage(
            content="The weather forecast predicts rain tomorrow with temperatures around 65°F. Don't forget to bring an umbrella if you're going out!",
        )
        result = self.lf.scan(message)
        self.assertEqual(result.decision, ScanDecision.ALLOW)
        self.assertAlmostEqual(result.score, 0.0)
        self.assertEqual(result.status, ScanStatus.SUCCESS)

    @patch(
        "llamafirewall.scanners.experimental.piicheck_scanner.PIICheckScanner._convert_llm_response_to_score",
        new_callable=AsyncMock,
        return_value=1.0,
    )
    @patch(
        "llamafirewall.scanners.custom_check_scanner.CustomCheckScanner._evaluate_with_llm",
        new_callable=AsyncMock,
        return_value=PIICheckOutputSchema(
            detected_pii_types=["Email addresses"],
        ),
    )
    def test_medium_risk_inputs(
        self, _mock_call_llm: AsyncMock, _mock_convert_score: AsyncMock
    ) -> None:
        message = UserMessage(
            content="Please contact the administrator at admin@company.com to reset your password. The office is located in Building A.",
        )
        result = self.lf.scan(message)
        self.assertEqual(
            result.decision, ScanDecision.BLOCK
        )  # Now above threshold since score is 1.0
        self.assertAlmostEqual(result.score, 1.0)
        self.assertIn("PII detected", result.reason)
        self.assertEqual(result.status, ScanStatus.SUCCESS)
