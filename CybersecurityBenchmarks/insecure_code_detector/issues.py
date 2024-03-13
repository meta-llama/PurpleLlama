# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import enum

import json

from dataclasses import dataclass

from .analyzers import Analyzer


class Severity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_string(severity: str) -> Severity:
        severity = severity.lower()
        if severity == "error":
            return Severity.ERROR
        elif severity == "warning":
            return Severity.WARNING
        elif severity == "advice":
            return Severity.ADVICE
        elif severity == "disabled":
            return Severity.DISABLED
        else:
            raise ValueError(f"Unknown severity: {severity}")


@dataclass(frozen=True)
class Issue:
    description: str
    cwe_id: str
    severity: Severity
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
