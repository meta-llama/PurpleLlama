# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import enum


class UseCase(enum.Enum):
    CODESHIELD = "codeshield"
    CYBERSECEVAL = "cyberseceval"

    def __str__(self) -> str:
        return self.name.lower()


def get_supported_usecases() -> list[UseCase]:
    return [UseCase.CODESHIELD, UseCase.CYBERSECEVAL]
