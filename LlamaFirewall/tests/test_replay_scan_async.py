# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List
from unittest.mock import AsyncMock, patch

from llamafirewall import (
    AssistantMessage,
    LlamaFirewall,
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    SystemMessage,
    ToolMessage,
    Trace,
    UserMessage,
)


class TestAgentActionsAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.lf = LlamaFirewall()

    async def test_replay_scan_async_human_in_the_loop(self) -> None:
        # Create test messages
        trace: List[Message] = [
            UserMessage(content="Normal message"),
            AssistantMessage(content="Message requiring human review"),
        ]

        # Add type annotation for expected_results
        expected_results: List[ScanResult] = [
            ScanResult(
                decision=ScanDecision.ALLOW,
                reason="default",
                score=0.0,
                status=ScanStatus.SUCCESS,
            ),
            ScanResult(
                decision=ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED,
                reason="needs review",
                score=0.6,
                status=ScanStatus.SUCCESS,
            ),
        ]

        with patch.object(
            self.lf, "scan_async", new_callable=AsyncMock
        ) as mock_scan_async:

            async def side_effect(message: Message, _: Trace) -> ScanResult:
                if "Normal message" in message.content:
                    return expected_results[0]
                else:
                    return expected_results[1]

            mock_scan_async.side_effect = side_effect

            # Execute
            result = await self.lf.scan_replay_async(trace)

            # Verify
            self.assertEqual(result.decision, ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED)
            self.assertEqual(result.reason, "needs review")
            self.assertEqual(result.score, 0.6)
            self.assertEqual(mock_scan_async.call_count, 2)

    async def test_replay_scan_async_allows_all_messages(self) -> None:
        # Create test messages with different roles
        trace: List[Message] = [
            SystemMessage(content="Safe system message"),
            UserMessage(content="Safe user message"),
            AssistantMessage(content="Safe assistant response"),
            ToolMessage(content="Safe tool output"),
        ]

        # All messages are allowed
        allow_result = ScanResult(
            decision=ScanDecision.ALLOW,
            reason="default",
            score=0.0,
            status=ScanStatus.SUCCESS,
        )

        with patch.object(
            self.lf, "scan_async", new_callable=AsyncMock
        ) as mock_scan_async:
            mock_scan_async.return_value = allow_result

            # Execute
            result = await self.lf.scan_replay_async(trace)

            # Verify
            self.assertEqual(result.decision, ScanDecision.ALLOW)
            self.assertEqual(result.reason, "default")
            self.assertEqual(result.score, 0.0)
            self.assertEqual(mock_scan_async.call_count, 4)  # Should check all messages

    async def test_replay_scan_async_blocks_on_first_violation(self) -> None:
        # Create test messages
        trace: List[Message] = [
            UserMessage(content="Safe user message"),
            AssistantMessage(content="Potentially unsafe code"),
            UserMessage(content="User response to unsafe code"),
        ]

        # Mock the scan method to return ALLOW for first message and BLOCK for second
        expected_results: List[ScanResult] = [
            ScanResult(
                decision=ScanDecision.ALLOW,
                reason="default",
                score=0.0,
                status=ScanStatus.SUCCESS,
            ),
            ScanResult(
                decision=ScanDecision.BLOCK,
                reason="unsafe code detected",
                score=0.8,
                status=ScanStatus.SUCCESS,
            ),
        ]

        with patch.object(
            self.lf, "scan_async", new_callable=AsyncMock
        ) as mock_scan_async:
            # Configure mock to return different values based on input
            async def side_effect(message: Message, _: Trace) -> ScanResult:
                if "Safe user message" in message.content:
                    return expected_results[0]
                else:
                    return expected_results[1]

            mock_scan_async.side_effect = side_effect

            # Execute
            result = await self.lf.scan_replay_async(trace)

            # Verify
            self.assertEqual(result.decision, ScanDecision.BLOCK)
            self.assertEqual(result.reason, "unsafe code detected")
            self.assertEqual(result.score, 0.8)
            self.assertEqual(
                mock_scan_async.call_count, 2
            )  # Should stop after second message

    async def test_replay_scan_and_build_trace_async(self) -> None:
        """Test the scan_replay_build_trace_async function."""
        # Create test messages
        message1 = UserMessage(content="First message")
        message2 = AssistantMessage(content="Second message")
        message3 = UserMessage(content="Unsafe message")

        # Mock the scan method to return ALLOW for first two messages and BLOCK for third
        expected_results: List[ScanResult] = [
            ScanResult(
                decision=ScanDecision.ALLOW,
                reason="default",
                score=0.0,
                status=ScanStatus.SUCCESS,
            ),
            ScanResult(
                decision=ScanDecision.ALLOW,
                reason="default",
                score=0.0,
                status=ScanStatus.SUCCESS,
            ),
            ScanResult(
                decision=ScanDecision.BLOCK,
                reason="unsafe content detected",
                score=0.8,
                status=ScanStatus.SUCCESS,
            ),
        ]

        with patch.object(
            self.lf, "scan_async", new_callable=AsyncMock
        ) as mock_scan_async:
            # Configure mock to return different values based on input
            async def side_effect(message: Message, _: Trace | None) -> ScanResult:
                if "Unsafe" in str(message.content):
                    return expected_results[2]
                else:
                    return expected_results[0]

            mock_scan_async.side_effect = side_effect

            # Test with empty initial trace
            result1, trace1 = await self.lf.scan_replay_build_trace_async(message1)

            # Verify first message
            self.assertEqual(result1.decision, ScanDecision.ALLOW)
            self.assertEqual(len(trace1), 1)
            self.assertEqual(trace1[0].content, "First message")

            # Add second message to trace
            result2, trace2 = await self.lf.scan_replay_build_trace_async(
                message2, trace1
            )

            # Verify second message
            self.assertEqual(result2.decision, ScanDecision.ALLOW)
            self.assertEqual(len(trace2), 2)
            self.assertEqual(trace2[0].content, "First message")
            self.assertEqual(trace2[1].content, "Second message")

            # Try to add unsafe message
            result3, trace3 = await self.lf.scan_replay_build_trace_async(
                message3, trace2
            )

            # Verify unsafe message is blocked and trace remains unchanged
            self.assertEqual(result3.decision, ScanDecision.BLOCK)
            self.assertEqual(result3.reason, "unsafe content detected")
            self.assertEqual(
                len(trace3), 2
            )  # Trace should not include the blocked message
            self.assertEqual(trace3, trace2)  # Trace should be unchanged


if __name__ == "__main__":
    unittest.main()
