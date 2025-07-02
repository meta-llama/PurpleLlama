# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Python root

# Only import the base Scanner class at module level
from .base_scanner import Scanner

# Re-export for convenience
__all__ = [
    "Scanner",
    "CodeShieldScanner",
    "HiddenASCIIScanner",
    "PromptGuardScanner",
    "CustomCheckScanner",
    "AlignmentCheckScanner",
    "PIICheckScanner",
]


# Implement lazy loading for scanner classes
from typing import Type


def __getattr__(name: str) -> Type[object]:
    if name == "CodeShieldScanner":
        from .code_shield_scanner import CodeShieldScanner

        return CodeShieldScanner
    elif name == "HiddenASCIIScanner":
        from .hidden_ascii_scanner import HiddenASCIIScanner

        return HiddenASCIIScanner
    elif name == "PromptGuardScanner":
        from .prompt_guard_scanner import PromptGuardScanner

        return PromptGuardScanner
    elif name == "CustomCheckScanner":
        from .custom_check_scanner import CustomCheckScanner

        return CustomCheckScanner
    elif name == "AlignmentCheckScanner":
        from .experimental.alignmentcheck_scanner import AlignmentCheckScanner

        return AlignmentCheckScanner
    elif name == "PIICheckScanner":
        from .experimental.piicheck_scanner import PIICheckScanner

        return PIICheckScanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
