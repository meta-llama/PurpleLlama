# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import yaml

from ....analyzers import Analyzer
from ....insecure_patterns import get_enabled_rules
from ....languages import Language
from ....usecases import get_supported_usecases, UseCase

LOG: logging.Logger = logging.getLogger(__name__)

SEMGREP_RULE_REPO_PATH: Path = Path(__file__).parent.parent


def _get_fbcode_path() -> Path:
    return Path(
        os.path.join(
            subprocess.check_output(["hg", "root"]).decode("utf-8").strip(), "fbcode"
        )
    )


def _get_enabled_semgrep_rules(language: Language, usecase: UseCase) -> list[str]:
    return get_enabled_rules(language, Analyzer.SEMGREP, usecase)


def _get_all_rules(lang: Language, usecase: UseCase) -> list[str]:
    directory = SEMGREP_RULE_REPO_PATH / lang.value
    enabled_rules = _get_enabled_semgrep_rules(lang, usecase)
    rules = []
    try:
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml"):
                    with open(os.path.join(root, file), "r") as f:
                        yaml_data = yaml.safe_load(f)
                        if "rules" in yaml_data and len(yaml_data["rules"]) > 0:
                            if not yaml_data["rules"][0]["id"] in enabled_rules:
                                continue
                            rules.append(yaml_data["rules"][0])
    except Exception as e:
        LOG.fatal(e)
    return rules


def gen_raw_rule(lang: Language, usecase: UseCase) -> None:
    rules = _get_all_rules(lang, usecase)
    if lang == Language.CPP:
        rules.extend(_get_all_rules(Language.C, usecase))
    rule_file_contents = json.dumps({"rules": list(rules)}, indent=2, sort_keys=True)
    json_file_path = (
        _get_fbcode_path()
        / "security"
        / "insecure_code_detector"
        / "insecure_code_detector"
        / "rules"
        / "semgrep"
        / "_generated_"
        / (lang + "_" + usecase + ".json")
    )

    with open(json_file_path, "w") as f:
        f.write(rule_file_contents)
        f.flush()
        f.close()


def main() -> None:
    SUPPORTED_LANGUAGES = [
        Language.C,
        Language.CPP,
        Language.CSHARP,
        Language.JAVA,
        Language.JAVASCRIPT,
        Language.PHP,
        Language.PYTHON,
    ]
    supported_usecases = get_supported_usecases()

    for lang in SUPPORTED_LANGUAGES:
        for usecase in supported_usecases:
            gen_raw_rule(lang, usecase)


if __name__ == "__main__":
    main()
