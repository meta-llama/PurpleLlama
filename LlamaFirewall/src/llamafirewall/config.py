# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import Enum
from typing import Mapping, Sequence, TypeAlias

from .llamafirewall_data_types import Role, ScannerType


Configuration: TypeAlias = Mapping[Role, Sequence[ScannerType | str]]


class UseCase(Enum):
    CHAT_BOT = "chatbot"
    CODING_ASSISTANT = "coding_assistant"


PREDEFINED_USE_CASES: Mapping[UseCase, Configuration] = {
    UseCase.CHAT_BOT: {
        Role.USER: [ScannerType.PROMPT_GUARD],
        Role.SYSTEM: [ScannerType.PROMPT_GUARD],
    },
    UseCase.CODING_ASSISTANT: {
        Role.ASSISTANT: [ScannerType.CODE_SHIELD],
        Role.TOOL: [ScannerType.CODE_SHIELD],
    },
}
