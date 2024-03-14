# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzers import Analyzer
from .languages import Language

try:
    from .internal import oss
except ImportError:
    from . import oss

LOG: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsecurePattern:
    description: str
    cwe_id: str
    rule: str
    severity: str
    regexes: list[str] | None = None
    pattern_id: str | None = None


@functools.lru_cache(maxsize=None)
def load(language: Language, analyzer: Analyzer) -> list[InsecurePattern]:
    yaml_file = oss.RULES_ROOT / f"{analyzer}/{language}.yaml"

    result = _load_patterns(yaml_file)

    if language == Language.CPP:
        result += load(Language.C, analyzer)
    elif language == Language.OBJECTIVE_C:
        result += load(Language.C, analyzer)
    elif language == Language.OBJECTIVE_CPP:
        result += load(Language.OBJECTIVE_C, analyzer)
        result += load(Language.CPP, analyzer)
    elif analyzer == Analyzer.REGEX:
        result += _load_patterns(oss.RULES_ROOT / "regex/language_agnostic.yaml")
    return result


@functools.lru_cache(maxsize=None)
def load_prescan_regexes(language: Language) -> list[InsecurePattern]:
    rule_directory = oss.RULES_ROOT / "semgrep" / language
    patterns = []
    for root, _dirs, files in os.walk(rule_directory):
        for file in files:
            if not file.endswith(".yaml"):
                continue
            with open(os.path.join(root, file), "r") as file:
                data = yaml.safe_load(file)
                if "rules" in data:
                    rule = data["rules"][0]
                    if ("metadata" in rule) and ("prescan_regex" in rule["metadata"]):
                        patterns.append(
                            InsecurePattern(
                                description=rule["message"],
                                cwe_id=rule["metadata"]["cwe_id"],
                                rule=rule["metadata"]["prescan_regex"],
                                severity=rule["severity"],
                            )
                        )
                    # even if one rule doesn't have prescan_regex,
                    # the list is ineffective
                    else:
                        return []
    return patterns


@functools.lru_cache(maxsize=None)
def _load_patterns(filename: Path) -> list[InsecurePattern]:
    result = []
    try:
        with open(filename, "r") as file:
            patterns = yaml.safe_load(file)
            for pattern in patterns:
                result.append(
                    InsecurePattern(
                        description=pattern["description"],
                        cwe_id=pattern["cwe_id"],
                        rule=pattern["rule"],
                        severity=pattern["severity"],
                        regexes=pattern["regexes"] if "regexes" in pattern else None,
                        pattern_id=(
                            pattern["pattern_id"] if "pattern_id" in pattern else None
                        ),
                    )
                )
    except FileNotFoundError:
        LOG.fatal(f"No such file or directory: {filename}")
    except yaml.YAMLError as exc:
        LOG.fatal(f"Error in YAML syntax: {exc}")
    except Exception as exc:
        LOG.fatal(f"Unexpected error while loading patterns: {exc}")
    return result
