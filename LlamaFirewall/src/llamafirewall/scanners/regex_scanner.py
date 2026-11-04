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
# Inspired by ATR (Agent Threat Rules) community patterns — MIT licensed
# See: https://github.com/Agent-Threat-Rule/agent-threat-rules
DEFAULT_REGEX_PATTERNS: Dict[str, str] = {
    # Prompt injection — direct instruction override
    "Prompt injection — direct": r"(?i)\b(ignore|disregard|forget|override|bypass|skip|abandon|drop|cancel|suppress|nullify|void|revoke|dismiss|discard|reject)\s+(all\s+)?(the\s+|your\s+|my\s+|any\s+|these\s+|those\s+)?(previous|prior|above|earlier|preceding|original|existing|current|initial|default|given|assigned|preset|established)?\s*(instructions?|prompts?|rules?|guidelines?|directives?|constraints?|context|programming|training|protocols?|policies|configuration|parameters|boundaries|guardrails?|safeguards?|restrictions?|tasks?|assignments?|information|directions?|orders?|commands?)",
    # Prompt injection — persona/s role-play override
    "Prompt injection — persona override": r"(?i)\b(you\s+are\s+now|you\s+have\s+become|you\s+will\s+act\s+as|you\s+shall\s+be|switch\s+to\s+being|become|morph\s+into|adopt\s+the\s+persona)\s+(a|an|the|my|ein[ea]?)?\s*[a-zA-Z\u00C0-\u024F]+",
    # Prompt injection — new task injection
    "Prompt injection — new task": r"(?i)\b(new|updated|revised|replacement|corrected|actual|real|true|overriding)\s+(instructions?|rules?|directives?|guidelines?|orders?|commands?)\s*[:=\-]",
    # Prompt injection — system prompt exfiltration
    "Prompt injection — system prompt exfil": r"(?i)(reveal|show|display|output|print|repeat|provide|give|share|tell|what\s+are|what\s+were|what\s+is)\s+(me\s+)?(your|the|all)\s+(secret|hidden|internal|system|original|previous|initial|complete|full|given|guard|entire)?\s*(instructions?|prompts?|passwords?|secrets?|embeddings?|configuration|directives?|guidelines?|rules)",
    # Prompt injection — multilingual (Spanish)
    "Prompt injection — Spanish": r"(?i)(ignora|ignorar|olvida|olvidar|descarta|descartar|omite|omitir|anula|anular)\s+(todas?\s+)?(las?\s+)?(instrucciones|reglas|directrices|restricciones|indicaciones|limitaciones)\s+(previas|anteriores|originales|existentes|actuales)",
    # Prompt injection — multilingual (German)
    "Prompt injection — German": r"(?i)(ignoriere|vergiss|missachte|umgehe|verwirf)\s+(alle[sn]?|die\s+)?(vorherigen?|bisherigen?|aktuellen?|bestehenden?)?\s*(Anweisungen|Regeln|Richtlinien|Anordnungen|Vorgaben|Instruktionen|Beschraenkungen|Aufgaben|Informationen|alles)",
    # Tool/credential spoofing — typo variants
    "Tool spoofing — typo domains": r"(?i)(?:gtihub|githbu|gihtub|gthub|g1thub|slakc|slck|slaack|s1ack|opeanai|openla|open_al|openaii|anthroplc|anthrop1c|_anthropic|databse|databaes|dtabase)[-_](?:api|search|commit|pr|issues?|repos|send|post|read|message|channel|chat|complete|embed|generate|query|read|write|exec|connect)",
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
