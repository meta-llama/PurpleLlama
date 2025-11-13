# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio
import concurrent
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from typing import List, Optional

from . import insecure_patterns, languages
from .analyzers import Analyzer

from .insecure_patterns import InsecurePattern
from .issues import Issue, Severity
from .languages import Language
from .usecases import UseCase

LOG: logging.Logger = logging.getLogger(__name__)

try:
    from .internal import oss
except ImportError:
    from . import oss

LANGUAGE_ANALYZER_MAP: dict[Language, list[Analyzer]] = {
    Language.C: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.CPP: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.CSHARP: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.HACK: [Analyzer.REGEX],
    Language.JAVA: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.JAVASCRIPT: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.KOTLIN: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.OBJECTIVE_C: [Analyzer.REGEX],
    Language.PHP: [Analyzer.REGEX],
    Language.PYTHON: [
        Analyzer.REGEX,
        Analyzer.SEMGREP,
    ],
    Language.RUBY: [Analyzer.REGEX],
    Language.RUST: [Analyzer.REGEX],
    Language.SWIFT: [Analyzer.REGEX],
    Language.XML: [Analyzer.REGEX],
}


@dataclass
class CodeContext:
    language: Language
    before: str | None
    after: str | None
    path: Path
    is_existing_path: bool = False


async def analyze(
    language: Language,
    generated_code: str,
    code_before: Optional[str] = None,
    code_after: Optional[str] = None,
    path: Optional[Path] = None,
    usecase: UseCase = UseCase.CYBERSECEVAL,
) -> List[Issue]:
    """
    Checks if the string contains insecure code.
    :return: True if the string is insecure, False otherwise.
    """
    result = []
    is_path_preexisting = True if path else False
    file_path = (
        path
        if is_path_preexisting
        else _write_code_to_file(
            (code_before if code_before else "")
            + generated_code
            + (code_after if code_after else ""),
            languages.get_file_extension(language),
        )
    )

    code_context = CodeContext(
        language,
        code_before,
        code_after,
        file_path or Path(),
        is_path_preexisting,
    )

    # Create tasks for each analyzer
    tasks = []

    # regex analyze
    if oss.ENABLE_REGEX_ANALYZER and (
        Analyzer.REGEX in LANGUAGE_ANALYZER_MAP.get(language, [])
    ):
        result += _regex_analyze(generated_code, code_context, usecase)

    if usecase == UseCase.CODESHIELD:
        # fast mode is greedy. returns for any match and doesn't include all matches
        if len(result) > 0:
            return result

    # SEMGREP analyze
    if oss.ENABLE_SEMGREP and (
        Analyzer.SEMGREP in LANGUAGE_ANALYZER_MAP.get(language, [])
    ):
        if usecase == UseCase.CODESHIELD:
            if not _is_semgrep_scan_recommended(generated_code, code_context, usecase):
                return result
        tasks.append(
            asyncio.create_task(_semgrep_analyze(generated_code, code_context, usecase))
        )

    # Gather results from all tasks
    results_from_tasks = await asyncio.gather(*tasks)

    # Flatten the list of results
    result += [issue for sublist in results_from_tasks for issue in sublist]

    # end of static analysis: clean up
    try:
        if not is_path_preexisting:
            os.remove(code_context.path)
    except OSError:
        LOG.warn("File created for static analysis could not be deleted.")

    return result


def _is_comment_or_directive(code: str) -> bool:
    if (
        code.strip().startswith("#")
        or code.strip().startswith("//")
        or code.strip().startswith("/*")
        or code.strip().endswith("*/")
    ):
        return True
    return False


def _search_pattern(pattern: InsecurePattern, generated_code: str) -> List[Issue]:
    search_result = re.search(pattern.rule, generated_code)
    if search_result:
        match_start = search_result.start()
        list_of_lines_until_match = generated_code[:match_start].split("\n")
        line_no = len(list_of_lines_until_match)
        line_content = (
            list_of_lines_until_match[-1] if list_of_lines_until_match else ""
        )
        # remove any match for comments
        if line_content and _is_comment_or_directive(line_content):
            return []
        return [
            Issue(
                description=pattern.description,
                cwe_id=pattern.cwe_id,
                severity=Severity.from_string(pattern.severity),
                line=line_no,
                rule=pattern.rule,
                analyzer=Analyzer.REGEX,
                pattern_id=pattern.pattern_id,
            )
        ]
    return []


def _run_regex(generated_code: str, patterns: list[InsecurePattern]) -> List[Issue]:
    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(_search_pattern, pattern, generated_code)
            for pattern in patterns
        ]
    # Flatten the list of results and return the list of issues
    return [
        issue
        for future in concurrent.futures.as_completed(tasks)
        for issue in future.result()
    ]


def _regex_analyze(
    generated_code: str, code_context: CodeContext, usecase: UseCase
) -> List[Issue]:
    patterns = insecure_patterns.load(code_context.language, Analyzer.REGEX, usecase)
    return _run_regex(generated_code, patterns)


async def _run_command(command: list[str]) -> str:
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        LOG.fatal(
            f"Process {' '.join(command)} failed with exit code {process.returncode}\nStdout: {stdout}\nStderr:{stderr}"
        )
    return stdout.decode("utf-8", errors="replace")


def _write_code_to_file(file_content: str, extension: str) -> Path:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension, mode="w", delete=False
        ) as tmp:
            tmp.write(file_content)
            return Path(tmp.name)
    except Exception as e:
        LOG.error(e)
        raise Exception(f"Could not create file: {e}")


def _filter_issues_in_generated_code(
    issues: List[Issue], generated_code: str, code_before: Optional[str] = None
) -> List[Issue]:
    generated_code_start_line = len(code_before.splitlines()) + 1 if code_before else 1
    generated_code_end_line = (
        len(generated_code.splitlines()) + generated_code_start_line - 1
    )
    result = [
        x
        for x in issues
        if (
            x.line is None
            or (
                isinstance(x.line, int)
                and (generated_code_start_line <= x.line <= generated_code_end_line)
            )
        )
    ]
    return result


def _parse_semgrep_issues(output: str) -> list[Issue]:
    """
    Parse the output of the run into a list of issues
    """
    result = []
    # Skip if json is not present in output
    json_start_index = output.find("{")
    if json_start_index < 0:
        return result
    try:
        json_output = json.loads(output[json_start_index:])
    except json.JSONDecodeError:
        LOG.warning(f"Could not parse JSON from semgrep output: {output}")
        return result
    if "results" in json_output:
        for issue in json_output["results"]:
            issue_details = issue.get("extra", {})
            result.append(
                Issue(
                    description=issue_details.get("message", ""),
                    cwe_id=issue_details.get("metadata", {}).get("cwe_id", ""),
                    severity=issue_details.get("severity", ""),
                    line=issue.get("start", {}).get("line", None),
                    char=issue.get("start", {}).get("col", None),
                    rule=issue.get("rule", None),
                    analyzer=Analyzer.SEMGREP,
                    pattern_id=issue.get("check_id", None),
                    path=issue.get("path", None),
                    name=issue.get("check_id", None),
                )
            )
    return result


def _is_semgrep_scan_recommended(
    generated_code: str, code_context: CodeContext, usecase: UseCase
) -> bool:
    prescan_patterns = insecure_patterns.load_prescan_regexes(
        code_context.language, usecase
    )
    if len(prescan_patterns) == 0:
        return True
    issues = _run_regex(generated_code, prescan_patterns)
    if len(issues) > 0:
        return True
    return False


async def _semgrep_analyze(
    generated_code: str,
    code_context: CodeContext,
    usecase: UseCase,
) -> list[Issue]:
    config_file_path = (
        oss.SEMGREP_GENERATED_RULES_PATH
        / f"{code_context.language.value}_{usecase.value}.json"
    )

    root_folder = None
    if oss.INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER:
        root_folder = "/" if os.path.isabs(code_context.path) else None

    command = (
        oss.SEMGREP_COMMAND
        + [
            (
                str(config_file_path)
                if config_file_path.exists()
                else str(oss.SEMGREP_RULE_REPO_PATH / str(code_context.language))
            )
        ]
        + (["--project-root", root_folder] if root_folder else [])
        + [str(code_context.path)]
    )
    output = await _run_command(command)
    result = _parse_semgrep_issues(output)
    if code_context.is_existing_path:
        return result
    filtered_result = _filter_issues_in_generated_code(
        result, generated_code, code_context.before
    )
    return filtered_result
