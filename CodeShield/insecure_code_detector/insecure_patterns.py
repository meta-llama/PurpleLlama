# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
import json

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .analyzers import Analyzer
from .languages import Language
from .usecases import UseCase

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
def load(
    language: Language, analyzer: Analyzer, usecase: UseCase
) -> list[InsecurePattern]:
    yaml_file = oss.RULES_ROOT / f"{analyzer}/{language}.yaml"
    result = _load_patterns(yaml_file, language, analyzer, usecase)

    if language == Language.CPP:
        result += load(Language.C, analyzer, usecase)
    elif language == Language.OBJECTIVE_C:
        result += load(Language.C, analyzer, usecase)
    elif language == Language.OBJECTIVE_CPP:
        result += load(Language.OBJECTIVE_C, analyzer, usecase)
        result += load(Language.CPP, analyzer, usecase)
    elif analyzer == Analyzer.REGEX:
        result += _load_patterns(
            oss.RULES_ROOT / "regex/language_agnostic.yaml",
            Language.LANGUAGE_AGNOSTIC,
            analyzer,
            usecase,
        )
    return result


@functools.lru_cache(maxsize=None)
def load_prescan_regexes(language: Language, usecase: UseCase) -> list[InsecurePattern]:
    rule_file_path = oss.SEMGREP_GENERATED_RULES_PATH / f"{language}_{usecase}.json"
    patterns = []
    if not os.path.exists(rule_file_path):
        return []
    with open(rule_file_path, "r") as file:
        data = json.load(file)
        rules = data["rules"] if "rules" in data else []
        for rule in rules:
            if ("metadata" in rule) and ("prescan_regex" in rule["metadata"]):
                patterns.append(
                    InsecurePattern(
                        description=rule["message"],
                        cwe_id=rule["metadata"]["cwe_id"],
                        rule=rule["metadata"]["prescan_regex"],
                        severity=rule["severity"],
                    )
                )
    return patterns


@functools.lru_cache(maxsize=None)
def _load_patterns(
    filename: Path, language: Language, analyzer: Analyzer, usecase: UseCase
) -> list[InsecurePattern]:
    result = []
    enabled_rules = get_enabled_rules(language, analyzer, usecase)
    try:
        with open(filename, "r") as file:
            patterns = yaml.safe_load(file)
            for pattern in patterns:
                if (
                    "pattern_id" in pattern
                    and pattern["pattern_id"] not in enabled_rules
                ):
                    continue
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


@functools.lru_cache(maxsize=None)
def get_enabled_rules(
    language: Language, analyzer: Analyzer, usecase: UseCase
) -> list[str]:
    enabled_rules = []
    config_data = _read_rule_config(usecase)
    if (
        (config_data)
        and (language.value in config_data)
        and (analyzer.value in config_data[language.value])
    ):
        enabled_rules.extend(config_data[language.value][analyzer.value]["rules"])
    return enabled_rules


@functools.lru_cache(maxsize=None)
def _read_rule_config(usecase: UseCase) -> dict[str, Any]:
    with open(oss.RULES_CONFIG_FILE_PATH, "r") as file:
        data = yaml.safe_load(file)
        if "config" in data and usecase.value in data["config"]:
            return data["config"][usecase.value]
    return {}
