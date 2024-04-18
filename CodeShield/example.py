#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio

import logging

from .codeshield import CodeShield

LOG: logging.Logger = logging.getLogger(__name__)


async def main() -> None:
    sample_llm_generated_code = """
    eval(input)
    """
    result = await CodeShield.scan_code(sample_llm_generated_code)
    if result.is_insecure:
        print(
            "Security issues found in code. Action needed to block or warn about insecure code."
        )
    else:
        print("No issues found while scanning for code security. No action needed")


if __name__ == "__main__":
    asyncio.run(main())
