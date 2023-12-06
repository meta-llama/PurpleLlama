# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from pathlib import Path


# We use buck internally to bundle the code into a .par file which
# can't access the rules directly. Hence the special casing here.
RULES_ROOT: Path = Path(__file__).parent / "rules"
ENABLE_REGEX_ANALYZER = True

ENABLE_WEGGLI = True
WEGGLI_COMMAND: list[str] = [
    os.environ.get("WEGGLI_PATH", "weggli"),
    "-n",
    "-C",
    "-A 0",
    "-B 0",
]

ENABLE_SEMGREP: bool = True
SEMGREP_RULE_REPO_PATH: Path = Path(__file__).parent / "rules" / "semgrep"

SEMGREP_COMMAND: list[str] = [
    os.environ.get("SEMGREP_PATH", "semgrep"),
    "--metrics",
    "off",
    "--quiet",
    "--json",
    "--config",
]
