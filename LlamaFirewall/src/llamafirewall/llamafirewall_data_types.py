# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypeAlias


class ScannerType(Enum):
    CODE_SHIELD = "code_shield"
    PROMPT_GUARD = "prompt_guard"
    AGENT_ALIGNMENT = "agent_alignment"
    HIDDEN_ASCII = "hidden_ascii"
    PII_DETECTION = "pii_detection"
    REGEX = "regex"


class ScanDecision(Enum):
    ALLOW = "allow"
    HUMAN_IN_THE_LOOP_REQUIRED = "human_in_the_loop_required"
    BLOCK = "block"


class ScanStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class Role(Enum):
    TOOL = "tool"  # tool output
    USER = "user"  # user input
    ASSISTANT = "assistant"  # LLM output
    MEMORY = "memory"  # memory
    SYSTEM = "system"  # system input


@dataclass
class ScanResult:
    decision: ScanDecision
    reason: str
    score: float
    status: ScanStatus = ScanStatus.SUCCESS  # Default to success


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class SystemMessage(Message):
    def __init__(
        self,
        content: str,
    ) -> None:
        super().__init__(role=Role.SYSTEM, content=content)


class UserMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role=Role.USER, content=content)


class AssistantMessage(Message):
    def __init__(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


class ToolMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(
            role=Role.TOOL,
            content=content,
        )


class MemoryMessage(Message):
    def __init__(self, content: str) -> None:
        super().__init__(role=Role.MEMORY, content=content)


Trace: TypeAlias = List[Message]
