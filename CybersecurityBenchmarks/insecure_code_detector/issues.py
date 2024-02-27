# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json

from dataclasses import dataclass

from .analyzers import Analyzer


@dataclass(frozen=True)
class Issue:
    description: str
    cwe_id: str
    severity: str
    rule: str | None = None
    line: int | None = None
    path: str | None = None
    char: int | None = None
    name: str | None = None
    original: str | None = None
    replacement: str | None = None
    analyzer: Analyzer | None = None
    pattern_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
