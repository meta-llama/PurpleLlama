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
    # -------------------------------------------------------------------------
    # ATR (Agent Threat Rules) patterns — community-maintained agent security
    # detection rules. Source: https://github.com/anthropics/agent-threat-rules
    # Validated on 53,577 real-world MCP skills with 0% false positive rate.
    # -------------------------------------------------------------------------
    # -- Prompt injection & jailbreak --
    "ATR: Prompt injection - instruction override": (
        r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?"
        r"(?:previous|prior|above|earlier|original|system)\s+"
        r"(?:instructions|rules|guidelines|prompts|directives)"
    ),
    "ATR: Jailbreak - mode switching": (
        r"(?:enable|activate|enter|switch\s+to)\s+"
        r"(?:developer|debug|god|admin|unrestricted|jailbreak|DAN)\s+mode"
    ),
    "ATR: System prompt override": (
        r"(?:new|updated|revised|replacement)\s+system\s+"
        r"(?:prompt|instruction|message|directive)"
    ),
    "ATR: System prompt extraction": (
        r"(?:reveal|show|display|print|output|repeat|echo)\s+"
        r"(?:your|the)\s+(?:system|original|initial|hidden)\s+"
        r"(?:prompt|instructions|message|rules)"
    ),
    "ATR: Role hijacking": (
        r"\[(?:SYSTEM|ADMIN|ROOT|OVERRIDE|MASTER)\]"
        r"|\bsystem:\s*you\s+are\b"
    ),
    "ATR: Social engineering pressure": (
        r"(?:I've\s+already\s+asked\s+you|stop\s+refusing\s+(?:to|my)"
        r"|you\s+must\s+comply\s+(?:with|now)"
        r"|as\s+(?:your|the)\s+(?:creator|developer|owner),?\s+I\s+(?:order|command|demand))"
    ),
    # -- Tool poisoning & data exfiltration --
    "ATR: Tool poisoning - hidden data actions": (
        r"(?:also|additionally|secretly|silently|quietly)\s+"
        r"(?:send|upload|post|transmit|exfiltrate|forward|copy)\s+"
        r"(?:the|all|any|user)?\s*"
        r"(?:data|file|content|information|credential|token|key|secret)"
    ),
    "ATR: Data exfiltration URL": (
        r"(?:curl|wget|fetch|requests\.(?:get|post))\s*\(?\s*['\"]?[^\s'\"]*"
        r"(?:\$\{|\$\(|%7B)[^\s'\"]*"
        r"(?:password|token|key|secret|auth|cookie|session)"
        r"|(?:https?://[^\s'\"]*\.(?:ngrok|requestbin|pipedream|webhook\.site|burpcollaborator)"
        r"\.[a-z]+[^\s'\"]*(?:\$\{|\$\(|%7B))"
    ),
    # -- Infrastructure attacks --
    "ATR: Reverse shell": (
        r"(?:bash|sh|nc|ncat|netcat|python|perl|ruby|php)\s+"
        r"[^\n]{0,200}"
        r"(?:/dev/tcp/|(?:RHOST|LHOST|RPORT)|\b(?:nc|ncat)\s+-[a-z]*[el]\b)"
    ),
    "ATR: Path traversal": (
        r"(?:\.\./|\.\.\\){2,}"
        r"|/etc/(?:passwd|shadow|hosts)"
        r"|~/.(?:ssh|aws|kube|gnupg)"
    ),
    "ATR: Code injection - eval/exec": (
        r"(?:eval|exec)\s*\(\s*(?:input|request|user|data|payload|args|params|cmd|command)"
        r"|(?:os\.system|subprocess\.(?:call|run|Popen))\s*\(\s*(?:f['\"]|input|request|user|data|cmd|command|\+)"
        r"|child_process\.exec\s*\("
    ),
    "ATR: Persistence mechanisms": (
        r"(?:crontab|schtasks|launchctl|systemctl\s+enable)\s+.*?"
        r"(?:curl|wget|python|bash|sh|nc)"
    ),
    "ATR: Privilege escalation": (
        r"(?:sudo\s+(?:su\b|-[is]\b|bash\b)"
        r"|chmod\s+[2467]\d{3}\b"
        r"|chown\s+root\b|\bsetuid\b)"
    ),
    # -- Credential & secret exposure --
    "ATR: Credential patterns": (
        r"(?:sk-[a-zA-Z0-9]{20,}"
        r"|ghp_[a-zA-Z0-9]{36}"
        r"|AKIA[0-9A-Z]{16}"
        r"|xox[bpsar]-[0-9a-zA-Z-]+"
        r"|glpat-[0-9a-zA-Z_-]{20})"
    ),
    "ATR: Environment variable harvesting": (
        r"(?:echo|print|cat|type)\s+.*?"
        r"(?:\$(?:ENV|HOME|PATH|USER|AWS_|OPENAI_|ANTHROPIC_)"
        r"|%(?:APPDATA|USERPROFILE)%)"
    ),
    "ATR: Encoded payload delivery": (
        r"(?:base64\s+(?:-d|--decode)|atob|Buffer\.from)\s*\(?\s*['\"]?"
        r"[A-Za-z0-9+/]{40,}={0,2}"
    ),
    # -- Network exfiltration --
    "ATR: DNS/network exfiltration": (
        r"(?:nslookup|dig|host)\s+.*?\$[\({]"
        r"|\.(?:dnslog|ceye|oast)\.(?:cn|fun|me|io)"
    ),
    # -- Cross-agent & injection --
    "ATR: Agent-to-agent manipulation": (
        r"(?:tell|instruct|command|direct|ask)\s+(?:the\s+)?"
        r"(?:other|next|downstream|target)\s+"
        r"(?:agent|assistant|model|AI)\s+to"
    ),
    "ATR: Markdown/HTML injection": (
        r"!\[.*?\]\(https?://[^)]*\?.*?"
        r"(?:password|token|secret|key|auth|session|cookie)="
        r"|<(?:script|object|embed)\s"
        r"|<(?:img|iframe|svg)\s[^>]*(?:onerror|onload|javascript:)"
    ),
    "ATR: Hidden instruction markers": (
        r"(?:<!--\s*(?:hidden|secret|system|admin|override)[^>]*-->"
        r"|\\u200[bB]|\\u200[cC]|\\uFEFF)"
    ),
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
