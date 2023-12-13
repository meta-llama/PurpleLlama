# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Union

from tqdm import tqdm

from .llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)


def process_prompt(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
) -> Dict[str, Union[int, str]]:
    prompt = cast(
        str,
        test_case.get(
            "test_case_prompt",
            cast(
                str,
                test_case.get(
                    "mutated_prompt",
                ),
            ),
        ),
    )
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")
    query_result = llm.query_with_retries(prompt)
    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "response": query_result,
    }

    # fields to (if available) add to response from the test case
    test_case_fields = [
        "cwe_identifier",
        "mitre_category",
        "language",
        "line_text",
        "origin_code",
        "variant",
        "rule",
        "repo",
    ]
    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


def query_llm_to_generate_responses(
    llms: Dict[str, LLM],
    prompt_path: Path,
    is_parallel: bool = False,
    num_threads: int = 20,
    to_process_prompt: Callable[
        [
            int,
            int,
            Dict[str, LLM],
            Dict[str, Union[int, str]],
        ],
        Dict[str, Union[int, str]],
    ] = process_prompt,
    is_test: bool = False,
    pass_k: int = 1,
    enumerate_prompt_and_pass_id: bool = True,
) -> List[Dict[str, Optional[str]]]:
    if not prompt_path.exists():
        raise ValueError("Please provide valid path for prompts.")
    prompt_dataset: List[Dict[str, Union[int, str]]] = json.loads(
        prompt_path.read_text()
    )
    if is_test:
        prompt_dataset = prompt_dataset[:5]
    response_result = []

    if is_parallel:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(to_process_prompt, idx, k, llms, test_case)
                if enumerate_prompt_and_pass_id
                else executor.submit(
                    to_process_prompt,
                    cast(int, test_case["prompt_id"]),
                    cast(int, test_case["pass_id"]),
                    llms,
                    test_case,
                )
                for idx, test_case in enumerate(prompt_dataset)
                for k in range(pass_k)
            ]

            for future in tqdm(
                futures,
                desc="Processing prompts",
                unit="prompt",
            ):
                try:
                    result = future.result()
                    response_result.append(result)
                except Exception as e:
                    LOG.error(f"Exception occurred in thread {future}: {e}")
    else:
        for idx, test_case in enumerate(
            tqdm(prompt_dataset * pass_k, desc="Processing prompts", unit="prompt")
        ):
            response_result.append(
                to_process_prompt(
                    idx % len(prompt_dataset),
                    idx // len(prompt_dataset),
                    llms,
                    test_case,
                )
            )
    return response_result
