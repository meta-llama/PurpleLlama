# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import functools
import importlib.resources

import os
import shutil
from pathlib import Path

# We use buck internally to bundle the code into a .par file which
# can't access the rules directly. Hence the special casing here.
RULES_ROOT: Path = Path(__file__).parent / "rules"
RULES_CONFIG_FILE_PATH: Path = RULES_ROOT / "config.yaml"
ENABLE_REGEX_ANALYZER = True
INCLUDE_SEMGREP_PROJECT_ROOT_PARAMETER = True

ENABLE_SEMGREP: bool = True
SEMGREP_RULE_REPO_PATH: Path = Path(__file__).parent / "rules" / "semgrep"
SEMGREP_GENERATED_RULES_PATH: Path = SEMGREP_RULE_REPO_PATH / "_generated_"


def _get_semgrep_core_path() -> Path:
    semgrep_core_name = "semgrep-core"
    try:
        with importlib.resources.path("semgrep.bin", semgrep_core_name) as path:
            if path.is_file():
                return path
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    path = shutil.which(semgrep_core_name)
    if path is not None:
        return Path(path)

    raise Exception(
        f"Failed to find {semgrep_core_name} in PATH or in the semgrep package."
    )


@functools.lru_cache(maxsize=None)
def _make_semgrep_binary_path() -> Path:
    # create symlink to semgrep-core
    source = _get_semgrep_core_path()
    destination = Path(str(source).replace("semgrep-core", "osemgrep"))
    if not os.path.islink(destination):
        os.symlink(source, destination)
    return Path(destination)


SEMGREP_BINARY_PATH: Path = _make_semgrep_binary_path()
SEMGREP_COMMAND: list[str] = [
    str(SEMGREP_BINARY_PATH),
    "--experimental",
    "--optimizations",
    "all",
    "--metrics",
    "off",
    "--quiet",
    "--json",
    "--config",
]
