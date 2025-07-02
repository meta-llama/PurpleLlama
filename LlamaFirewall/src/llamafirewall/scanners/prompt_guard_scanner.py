# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ..llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from .base_scanner import Scanner
from .promptguard_utils import PromptGuard


class PromptGuardScanner(Scanner):
    ALLOW_REASON = "No prompt injection detected"

    def __init__(
        self,
        block_threshold: float = 0.9,
    ) -> None:
        super().__init__(
            scanner_name="Prompt Guard Scanner", block_threshold=block_threshold
        )
        self.pg = PromptGuard()

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content

        score = self.pg.get_jailbreak_score(text=text)

        decision = (
            ScanDecision.BLOCK if score >= self.block_threshold else ScanDecision.ALLOW
        )

        reason = (
            f'Received text is likely to be a prompt injection attack, with a probability of {score}.\n Full text: "{text}"'
            if score >= self.block_threshold
            else self.ALLOW_REASON
        )

        # Return the scan result
        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )
