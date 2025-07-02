# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod

from ..llamafirewall_data_types import Message, ScanResult, Trace


class Scanner(ABC):
    def __init__(
        self,
        scanner_name: str,
        block_threshold: float = 1.0,
    ) -> None:
        self.name: str = scanner_name
        self.block_threshold: float = block_threshold
        self.require_full_trace: bool = False

    @abstractmethod
    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        """
        Scans the message and returns a ScanResult.

        This is the main method that concrete scanner implementations should override.
        Each scanner should implement its full scanning logic within this method,
        including any preprocessing, evaluation, and decision making.

        Args:
            message: The message to scan
            past_trace: Optional trace of previous messages

        Returns:
            ScanResult: The result of the scan, containing a decision, reason, and score
        """
        raise NotImplementedError("[LlamaFirewall] Scan method not implemented")

    def __str__(self) -> str:
        return self.name
