#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio

import enum

import logging
from dataclasses import dataclass

from CodeShield.insecure_code_detector import insecure_code_detector, languages

from CodeShield.insecure_code_detector.issues import Issue, Severity
from CodeShield.insecure_code_detector.languages import Language
from CodeShield.insecure_code_detector.usecases import UseCase

LOG: logging.Logger = logging.getLogger(__name__)


class Treatment(enum.Enum):
    BLOCK = "block"
    WARN = "warn"
    IGNORE = "ignore"


@dataclass
class CodeShieldScanResult:
    is_insecure: bool
    issues_found: list[Issue] | None
    recommended_treatment: Treatment


class CodeShield:
    @classmethod
    async def _execute_icd(
        cls, language: languages.Language, text: str
    ) -> list[Issue] | None:
        issues = await insecure_code_detector.analyze(
            language, text, usecase=UseCase.CODESHIELD
        )
        return issues

    @classmethod
    async def scan_code(
        cls, code: str, language: Language | None = None
    ) -> CodeShieldScanResult:
        """
        This function calls insecure code detector to scan the code for insecure code by running the rules of the programing language provided (or
        all available languages if a programing language is not provided).

        Args:
            code (str): The code to scan.
            language (Language, optional): The programming language of the code.
        Returns:
            A CodeShieldScanResult object with the following fields:
                is_insecure: True if the code is insecure, False otherwise.
                issues_found: A list of issues found in the code.
                recommended_treatment: The recommended treatment for the issues found.
        """
        result = CodeShieldScanResult(
            is_insecure=False,
            issues_found=None,
            recommended_treatment=Treatment.IGNORE,
        )
        issues = []
        try:
            if language is None:
                languages_to_scan = languages.get_supported_languages()
                tasks = [
                    cls._execute_icd(language, code) for language in languages_to_scan
                ]
                issues_list = await asyncio.gather(*tasks)
                for issue in issues_list:
                    if issue is not None:
                        issues.extend(issue)
            else:
                issues = await cls._execute_icd(language, code)
        except Exception as e:
            LOG.error(f"Exception while scanning for code security: {e}")
            return result
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Optional[List[Any]]`.
        if len(issues) > 0:
            result.is_insecure = True
            result.issues_found = issues
            # pyre-ignore
            if any(issue.severity == Severity.ERROR for issue in issues):
                result.recommended_treatment = Treatment.BLOCK
            else:
                result.recommended_treatment = Treatment.WARN
        return result
