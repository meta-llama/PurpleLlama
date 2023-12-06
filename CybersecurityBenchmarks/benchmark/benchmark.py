# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import Optional

from .llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)

from pathlib import Path


class Benchmark:
    def __init__(
        self,
        *,
        llms_under_test: list[LLM],
        response_path: Path,
        judge_response_path: Optional[Path] = None,
        stat_path: Optional[Path] = None,
        is_test: bool = False,
        expansion_llm: Optional[LLM] = None,
        judge_llm: Optional[LLM] = None,
        pass_k: int = 1,
    ) -> None:
        self.response_path: Path = response_path
        self.is_test: bool = is_test
        self.llms_under_test: list[LLM] = llms_under_test
        self.judge_response_path = judge_response_path
        self.stat_path = stat_path
        self.expansion_llm = expansion_llm
        self.judge_llm = judge_llm
        self.pass_k = pass_k

    def extract_content_in_code_blocks(self, input: str) -> list[str]:
        # Using regular expression to find content between code blocks ```
        return re.findall(r"```(.*?)```", input, re.DOTALL)

    @abstractmethod
    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: bool = False
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the responses in a file.
        """
        pass

    @abstractmethod
    async def run(
        self,
    ) -> None:
        """
        Processes a LLM-response.
        """
        pass
