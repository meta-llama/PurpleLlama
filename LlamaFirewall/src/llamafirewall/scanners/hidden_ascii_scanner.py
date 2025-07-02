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


class HiddenASCIIScanner(Scanner):
    ALLOW_REASON = "No hidden ASCII detected"

    def __init__(
        self,
        scanner_name: str = "Hidden ASCII Scanner",
        block_threshold: float = 1.0,
    ) -> None:
        super().__init__(scanner_name, block_threshold)

    @staticmethod
    def _is_hidden_ascii_present(text: str) -> float:
        if any(0xE0000 <= ord(char) <= 0xE007F for char in text):
            return 1.0
        return 0.0

    @staticmethod
    def _decode_hidden_ascii(text: str) -> str:
        base_tag = 0xE0000
        tag_range = range(0xE0000, 0xE007F + 1)
        decoded_text = ""
        for char in text:
            if ord(char) in tag_range:
                decoded_char = chr(ord(char) - base_tag)
                decoded_text += decoded_char
            else:
                decoded_text += char
        return decoded_text

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content

        score = self._is_hidden_ascii_present(text)

        decision = (
            ScanDecision.BLOCK if score >= self.block_threshold else ScanDecision.ALLOW
        )

        reason = (
            "Hidden ASCII: " + self._decode_hidden_ascii(text)
            if score >= self.block_threshold
            else self.ALLOW_REASON
        )

        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )
