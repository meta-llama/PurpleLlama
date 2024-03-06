# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

LOG: logging.Logger = logging.getLogger(__name__)

SEMGREP_RULE_REPO_PATH: Path = Path(__file__).parent.parent


def _get_all_rules(directory: Path) -> list[str]:
    rules = []
    try:
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml"):
                    with open(os.path.join(root, file), "r") as f:
                        yaml_data = yaml.safe_load(f)
                        if "rules" in yaml_data and len(yaml_data["rules"]) > 0:
                            rules.append(yaml_data["rules"][0])
    except Exception as e:
        LOG.fatal(e)
    return rules


def gen_raw_rule(lang: str, rule_directory: Path) -> None:
    rules = _get_all_rules(rule_directory / lang)
    rule_file_contents = json.dumps({"rules": list(rules)}, indent=2, sort_keys=True)
    json_file_path = rule_directory / "raw_rules" / (lang + ".json")

    with open(json_file_path, "w") as f:
        f.write(rule_file_contents)
        f.flush()
        f.close()


def main() -> Optional[str]:
    SUPPORTED_LANGUAGES = [
        "csharp",
        "java",
        "javascript",
        "python",
    ]

    for lang in SUPPORTED_LANGUAGES:
        rule_directory = SEMGREP_RULE_REPO_PATH
        gen_raw_rule(lang, rule_directory)
        return str(rule_directory)


if __name__ == "__main__":
    main()
