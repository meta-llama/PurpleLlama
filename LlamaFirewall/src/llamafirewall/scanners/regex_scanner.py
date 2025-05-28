# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
from typing import Dict, Pattern

from ..llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from .base_scanner import Scanner


# Default regex patterns for common security concerns
DEFAULT_REGEX_PATTERNS: Dict[str, str] = {
    # Prompt injection patterns
    "Prompt injection": r"ignore previous instructions|ignore all instructions",
    # PII detection
    "Email address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Phone number": r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "Credit card": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    "Social security number": r"\b\d{3}-\d{2}-\d{4}\b",
}


class RegexScanner(Scanner):
    """
    A scanner that checks messages against a list of regex patterns.

    It uses a predefined set of regex patterns for common security concerns.
    """

    ALLOW_REASON = "No regex patterns matched"

    def __init__(
        self,
        scanner_name: str = "Regex Scanner",
        block_threshold: float = 1.0,
    ) -> None:
        """
        Initialize the RegexScanner.

        Args:
            scanner_name: Name of the scanner
            block_threshold: Threshold for blocking messages (default: 1.0)
        """
        super().__init__(scanner_name, block_threshold)
        self.patterns: Dict[str, Pattern] = {}

        # Compile all default patterns
        for name, pattern in DEFAULT_REGEX_PATTERNS.items():
            try:
                self.patterns[name] = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            except re.error:
                print(f"[RegexScanner] Invalid regex pattern: {pattern}")

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        """
        Scan a message against the default regex patterns.

        Args:
            message: The message to scan
            past_trace: Optional trace of previous messages (not used in this scanner)

        Returns:
            ScanResult: The result of the scan
        """
        text = message.content

        # Check each pattern
        for name, pattern in self.patterns.items():
            if pattern.search(text):
                # Pattern matched, block the message
                return ScanResult(
                    decision=ScanDecision.BLOCK,
                    reason=f"Regex match: {name}",
                    score=1.0,
                    status=ScanStatus.SUCCESS,
                )

        # No patterns matched, allow the message
        return ScanResult(
            decision=ScanDecision.ALLOW,
            reason=self.ALLOW_REASON,
            score=0.0,
            status=ScanStatus.SUCCESS,
        )
