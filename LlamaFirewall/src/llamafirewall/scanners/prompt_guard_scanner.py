# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from llamafirewall.utils.memory_optimizer import ValidationBufferPool

from ..llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from .base_scanner import Scanner
from .promptguard_utils import PromptGuard

guard_mem_pool = ValidationBufferPool()


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
        text: str = message.content

        pool_idx, active_buffer = guard_mem_pool.acquire()
        
        if pool_idx == -1:
            active_buffer = bytearray(len(text) * 4)

        try:
            text_bytes: bytes = text.encode("utf-8")
            buffer_len = len(text_bytes)
            active_buffer[:buffer_len] = text_bytes

            score: float = self.pg.get_jailbreak_score(
                text=active_buffer[:buffer_len].decode("utf-8")
            )
            
        finally:
            guard_mem_pool.release(pool_idx)

        decision: ScanDecision = (
            ScanDecision.BLOCK if score >= self.block_threshold else ScanDecision.ALLOW
        )

        reason: str = (
            f'Received text is likely to be a prompt injection attack, with a probability of {score}.\n Full text: "{text}"'
            if score >= self.block_threshold
            else self.ALLOW_REASON
        )

        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )
