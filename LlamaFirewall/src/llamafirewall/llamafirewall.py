# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
from typing import Callable, Type

from llamafirewall.config import Configuration, PREDEFINED_USE_CASES, UseCase
from llamafirewall.llamafirewall_data_types import (
    Message,
    Role,
    ScanDecision,
    ScannerType,
    ScanResult,
    ScanStatus,
    Trace,
)

from typing_extensions import Self

# Only import the base Scanner class at module level
from .scanners import Scanner

LOG: logging.Logger = logging.getLogger(__name__)

custom_scanner_registry: dict[str, type[Scanner]] = {}


def register_llamafirewall_scanner(
    scanner_name: str,
) -> Callable[[Type[Scanner]], Type[Scanner]]:
    def decorator(scanner_class: Type[Scanner]) -> Type[Scanner]:
        if not issubclass(scanner_class, Scanner):
            raise ValueError(
                "[LlamaFirewall] Registered class must be a subclass of Scanner"
            )
        custom_scanner_registry[scanner_name] = scanner_class
        return scanner_class

    return decorator


def create_scanner(scanner_type: ScannerType | str) -> Scanner:
    if isinstance(scanner_type, str) and scanner_type in custom_scanner_registry:
        scanner_class = custom_scanner_registry[scanner_type]
        # pyre-ignore[20]: Registered class must be a subclass of Scanner, so it has a constructor with scanner_name as a parameter
        return scanner_class()
    elif isinstance(scanner_type, ScannerType):
        if scanner_type == ScannerType.CODE_SHIELD:
            from .scanners.code_shield_scanner import CodeShieldScanner

            return CodeShieldScanner()
        elif scanner_type == ScannerType.HIDDEN_ASCII:
            from .scanners.hidden_ascii_scanner import HiddenASCIIScanner

            return HiddenASCIIScanner()
        elif scanner_type == ScannerType.PROMPT_GUARD:
            from .scanners.prompt_guard_scanner import PromptGuardScanner

            return PromptGuardScanner()
        elif scanner_type == ScannerType.AGENT_ALIGNMENT:
            from .scanners.experimental.alignmentcheck_scanner import (
                AlignmentCheckScanner,
            )

            return AlignmentCheckScanner()
        elif scanner_type == ScannerType.PII_DETECTION:
            from .scanners.experimental.piicheck_scanner import PIICheckScanner

            return PIICheckScanner()
        elif scanner_type == ScannerType.REGEX:
            from .scanners.regex_scanner import RegexScanner

            # Create a RegexScanner with default patterns
            return RegexScanner()
        else:
            raise ValueError(
                f"[LlamaFirewall] Unsupported scanner type: {scanner_type}"
            )
    else:
        raise ValueError(f"[LlamaFirewall] Unknown scanner type: {scanner_type}")


class LlamaFirewall:
    def __init__(self, scanners: Configuration | None = None) -> None:
        # Default scanners for each Role
        default_scanners = {
            Role.TOOL: [ScannerType.CODE_SHIELD, ScannerType.PROMPT_GUARD],
            Role.USER: [ScannerType.PROMPT_GUARD],
            Role.SYSTEM: [],
            Role.ASSISTANT: [ScannerType.CODE_SHIELD],
            Role.MEMORY: [],
        }
        # Use provided scanners or default to all scanners
        self.scanners: Configuration = scanners or default_scanners

    @classmethod
    def from_usecase(cls, usecase: UseCase) -> Self:
        if usecase not in PREDEFINED_USE_CASES:
            raise Warning(f"Usecase: {usecase} is not predefined with any scanners.")
            return cls()  # or some other default behavior
        return cls(scanners=PREDEFINED_USE_CASES[usecase])

    def scan(
        self,
        input: Message,
        trace: Trace | None = None,
    ) -> ScanResult:
        scanners = self.scanners.get(input.role, [])
        reasons = []
        decisions = {}
        last_reason = ""
        for scanner_type in scanners:
            scanner_instance = create_scanner(scanner_type)
            LOG.debug(
                f"[LlamaFirewall] Scanning with {scanner_instance.name}, for the input {str(input.content)[:20]}"
            )
            scanner_result = asyncio.run(scanner_instance.scan(input, trace))
            reasons.append(
                f"{scanner_type}: {scanner_result.reason.strip()} - score: {scanner_result.score}"
            )
            last_reason = scanner_result.reason.strip()

            # Record the highest score for each found ScanDecision
            decisions[scanner_result.decision] = max(
                scanner_result.score,
                decisions.get(scanner_result.decision, scanner_result.score),
            )

        if len(scanners) == 1:
            return ScanResult(
                decision=list(decisions.keys())[0],
                reason=last_reason,
                score=list(decisions.values())[0],
                status=ScanStatus.SUCCESS,
            )

        formatted_reasons = "; ".join(set(reasons))
        # Select BLOCK as the final decision if present in the list,
        # otherwise select the decision with the highest score
        final_decision = (
            ScanDecision.BLOCK
            if ScanDecision.BLOCK in decisions.keys()
            else (
                max(decisions, key=decisions.get) if decisions else ScanDecision.ALLOW
            )
        )
        final_score = (
            decisions[ScanDecision.BLOCK]
            if ScanDecision.BLOCK in decisions.keys()
            else max(list(decisions.values()) + [0.0])
        )

        return ScanResult(
            decision=final_decision,
            reason=formatted_reasons if formatted_reasons else "default",
            score=final_score,
            status=ScanStatus.SUCCESS,
        )

    async def scan_async(
        self,
        input: Message,
        trace: Trace | None = None,
    ) -> ScanResult:
        for scanner_type in self.scanners.get(input.role, []):
            scanner_instance = create_scanner(scanner_type)
            scanner_result = await scanner_instance.scan(input, trace)
            if (
                scanner_result.decision == ScanDecision.BLOCK
                or scanner_result.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED
            ):
                return scanner_result
        return ScanResult(
            decision=ScanDecision.ALLOW,
            reason="default",
            score=0.0,
            status=ScanStatus.SUCCESS,
        )

    def scan_replay(
        self,
        trace: Trace,
    ) -> ScanResult:
        """
        Process the full trace of messages. Return the final scan result.
        """
        scan_result = ScanResult(
            decision=ScanDecision.ALLOW,
            reason="default",
            score=0.0,
            status=ScanStatus.SUCCESS,
        )
        for current_ix, message in enumerate(trace):
            # Create a new trace up to the current message (excluding it)
            past_trace = [msg for ix, msg in enumerate(trace) if ix < current_ix]
            scan_result = self.scan(message, past_trace if past_trace else None)
            # Single block or human prevents the rest of the conversation from being scanned
            if scan_result.decision == ScanDecision.BLOCK:
                break
            elif scan_result.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED:
                break
        return scan_result

    def scan_replay_build_trace(
        self,
        message: Message,
        stored_trace: Trace | None = None,
    ) -> tuple[ScanResult, Trace]:
        """
        Process a single message, build and maintain a trace over time.

        Args:
            message: The current message to process
            stored_trace: The existing trace of messages (if any)

        Returns:
            A tuple containing:
            - The scan result
            - The updated trace including the current message (if it passed the scan)
        """
        # Initialize trace if None
        if stored_trace is None:
            stored_trace = []

        # Scan the message with the existing trace
        scan_result = self.scan(message, stored_trace)

        # If the message passes the scan, add it to the trace
        if scan_result.decision == ScanDecision.ALLOW:
            # Create a new trace with the current message added
            updated_trace = stored_trace + [message]
            return scan_result, updated_trace

        # If the message is blocked or requires human review, don't add it to the trace
        return scan_result, stored_trace

    async def scan_replay_async(
        self,
        trace: Trace,
    ) -> ScanResult:
        """
        Process the full trace of messages asynchronously. Return the final scan result.

        This is the async version of scan_replay, suitable for use in async contexts
        like Jupyter notebooks.
        """
        scan_result = ScanResult(
            decision=ScanDecision.ALLOW,
            reason="default",
            score=0.0,
            status=ScanStatus.SUCCESS,
        )
        for current_ix, message in enumerate(trace):
            # Create a new trace up to the current message (excluding it)
            past_trace = [msg for ix, msg in enumerate(trace) if ix < current_ix]
            scan_result = await self.scan_async(
                message, past_trace if past_trace else None
            )
            # Single block or human prevents the rest of the conversation from being scanned
            if scan_result.decision == ScanDecision.BLOCK:
                break
            elif scan_result.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED:
                break
        return scan_result

    async def scan_replay_build_trace_async(
        self,
        message: Message,
        stored_trace: Trace | None = None,
    ) -> tuple[ScanResult, Trace]:
        """
        Process a single message, build and maintain a trace over time asynchronously.

        This is the async version of scan_replay_build_trace, suitable for use in async contexts
        like Jupyter notebooks.

        Args:
            message: The current message to process
            stored_trace: The existing trace of messages (if any)

        Returns:
            A tuple containing:
            - The scan result
            - The updated trace including the current message (if it passed the scan)
        """
        # Initialize trace if None
        if stored_trace is None:
            stored_trace = []

        # Scan the message with the existing trace
        scan_result = await self.scan_async(message, stored_trace)

        # If the message passes the scan, add it to the trace
        if scan_result.decision == ScanDecision.ALLOW:
            # Create a new trace with the current message added
            updated_trace = stored_trace + [message]
            return scan_result, updated_trace

        # If the message is blocked or requires human review, don't add it to the trace
        return scan_result, stored_trace
