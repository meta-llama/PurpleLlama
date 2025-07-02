# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import asyncio

try:
    from codeshield.insecure_code_detector import (  # pyre-ignore[21]
        insecure_code_detector,
        languages,
    )
    from codeshield.insecure_code_detector.issues import Issue  # pyre-ignore[21]
    from codeshield.insecure_code_detector.usecases import UseCase  # pyre-ignore[21]
except ImportError:
    from CodeShield.insecure_code_detector import insecure_code_detector, languages
    from CodeShield.insecure_code_detector.issues import Issue
    from CodeShield.insecure_code_detector.usecases import UseCase


from ..llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from .base_scanner import Scanner


class CodeShieldScanner(Scanner):
    ALLOW_REASON = "No unsafe function call detected"

    def __init__(
        self,
    ) -> None:
        super().__init__("Code Shield Scanner", 1.0)

    async def _execute_icd(
        self, language: languages.Language, text: str
    ) -> list[Issue]:
        issues = await insecure_code_detector.analyze(
            language=language,
            generated_code=text,
            code_before=None,
            code_after=None,
            path=None,
            usecase=UseCase.CODESHIELD,
        )
        return issues or []

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content
        languages_to_scan = languages.get_supported_languages()
        tasks = [self._execute_icd(language, text) for language in languages_to_scan]
        issues_all_languages = await asyncio.gather(*tasks)

        all_issues = [
            issue
            for issue_per_scanning_language in issues_all_languages
            for issue in issue_per_scanning_language
        ]

        if len(all_issues) == 0:
            return ScanResult(
                decision=ScanDecision.ALLOW,
                reason=self.ALLOW_REASON,
                score=0.0,
                status=ScanStatus.SUCCESS,
            )

        issues_details = []
        for issue in all_issues:
            issue_detail = f"- {issue.description}"
            if hasattr(issue, "cwe_id") and issue.cwe_id:
                issue_detail += f" (CWE-{issue.cwe_id})"
            if hasattr(issue, "line") and issue.line:
                issue_detail += f" at line {issue.line}"
            if hasattr(issue, "severity") and issue.severity:
                severity_str = str(issue.severity)
                issue_detail += f" [Severity: {severity_str}]"
            issues_details.append(issue_detail)

        return ScanResult(
            decision=ScanDecision.BLOCK,
            reason=f"{len(issues_details)} unsafe function call detected:\n"
            + "\n".join(issues_details),
            score=1.0,
            status=ScanStatus.SUCCESS,
        )
