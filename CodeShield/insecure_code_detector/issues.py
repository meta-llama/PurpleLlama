# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import enum

import json

from dataclasses import dataclass
from typing import Optional

from .analyzers import Analyzer


class Severity(enum.Enum):
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


@dataclass
class Issue:
    description: str
    cwe_id: str
    severity: Severity
    rule: Optional[str] = None
    line: Optional[int] = None
    path: Optional[str] = None
    char: Optional[int] = None
    name: Optional[str] = None
    original: Optional[str] = None
    replacement: Optional[str] = None
    analyzer: Optional[Analyzer] = None
    pattern_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
