# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Python root

# Import the most commonly used classes directly to ensure they're available for import statements
from .config import UseCase
from .llamafirewall import LlamaFirewall, register_llamafirewall_scanner
from .llamafirewall_data_types import (
    AssistantMessage,
    MemoryMessage,
    Message,
    Role,
    ScanDecision,
    ScannerType,
    ScanResult,
    ScanStatus,
    SystemMessage,
    ToolMessage,
    Trace,
    UserMessage,
)
from .scanners import Scanner

# Define __all__ to control what's exported
__all__ = [
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "MemoryMessage",
    "Role",
    "ScannerType",
    "ScanResult",
    "ScanDecision",
    "ScanStatus",
    "LlamaFirewall",
    "UseCase",
    "Scanner",
    "register_llamafirewall_scanner",
    "LLMClient",
    "Trace",
]


# Implement lazy loading for less commonly used classes
from typing import Type


def __getattr__(name: str) -> Type[object]:
    if name == "LLMClient":
        from .utils.base_llm import LLMClient

        return LLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
