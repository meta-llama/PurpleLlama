# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Python root

# Re-export for convenience
__all__ = [
    "AlignmentCheckScanner",
    "PIICheckScanner",
]


# Implement lazy loading
from typing import Type


def __getattr__(name: str) -> Type[object]:
    if name == "AlignmentCheckScanner":
        from .alignmentcheck_scanner import AlignmentCheckScanner

        return AlignmentCheckScanner
    elif name == "PIICheckScanner":
        from .piicheck_scanner import PIICheckScanner

        return PIICheckScanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
