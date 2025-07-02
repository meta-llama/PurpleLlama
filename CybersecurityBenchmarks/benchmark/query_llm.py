# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Union

from tqdm import tqdm

from .llm import LLM

LOG: logging.Logger = logging.getLogger(__name__)

# fields to (if available) add to response from the test case
test_case_fields = [
    "cwe_identifier",
    "mitre_category",
    "attack_type",
    "language",
    "line_text",
    "origin_code",
    "variant",
    "rule",
    "repo",
    "code",
    "answer",
    "challenge_type",
    "language",
    "speaking_language",
    "false_refusal_type",
    "caught_by_promptguard",
    "categories",
    "subcategories",
    "voice_tone",
    "label",
    "sample_id",
    "correct_answer",
    "explanation",
    "category",
    "question_text",
    "url_source",
    "scenario",
    "question_text",
    "options",
]


def get_another_prompt_with_response(prompt: str) -> str:
    return prompt


def process_prompts_multi_turn(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
    get_another_prompt_with_response: Callable[
        [str], str
    ] = get_another_prompt_with_response,
    num_tries: int = 10,
) -> Dict[str, Union[int, str, List[str]]]:
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
    prompt_with_history = [prompt]
    for i in range(num_tries):
        try:
            query_result = llm.chat_with_retries(
                prompt_with_history,
                guided_decode_json_schema=guided_decode_json_schema,
            )
            prompt_with_history.append(query_result)
            if i < num_tries - 1:
                new_prompt = get_another_prompt_with_response(query_result)
                prompt_with_history.append(new_prompt)
        except Exception as e:
            LOG.error(f"Query failed with error: {e}. Failed prompt: {prompt}")
            prompt = "FAIL TO QUERY"
            break

    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "response": prompt_with_history[-1],
        "chat_history": prompt_with_history,
    }

    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


def process_prompt_and_user_input(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
) -> Dict[str, Union[int, str]]:
    prompt = cast(
        str,
        test_case.get("test_case_prompt"),
    )
    user_input = cast(
        str,
        test_case.get("user_input"),
    )

    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    llm = llms["initial_query_llm"]
    if not llm:
        raise ValueError("Initial query LLM must be specified.")
    query_result = llm.query_with_system_prompt_with_retries(
        system_prompt=prompt,
        prompt=user_input,
        guided_decode_json_schema=guided_decode_json_schema,
    )
    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "user_input": user_input,
        "response": query_result,
    }

    test_case_fields = [
        "injection_type",
        "injection_variant",
        "judge_question",
        "risk_category",
        "speaking_language",
        "false_refusal_type",
        "caught_by_promptguard",
    ]
    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


def process_prompt(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
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
    try:
        query_result = llm.query_with_retries(
            prompt,
            guided_decode_json_schema=guided_decode_json_schema,
        )
    except Exception as e:
        LOG.error(f"Query failed with error: {e}. Failed prompt: {prompt}")
        query_result = "FAIL TO QUERY"
    response = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "test_case_prompt": prompt,
        "response": query_result,
    }

    for field in test_case_fields:
        if field in test_case:
            response[field] = test_case[field]

    if llm.model is not None:
        response["model"] = llm.model
    return response


def query_llm_to_generate_responses(
    llms: Dict[str, LLM],
    prompt_path: Path,
    run_llm_in_parallel: int = 1,
    to_process_prompt: Callable[
        [
            int,
            int,
            Dict[str, LLM],
            Dict[str, Union[int, str]],
            Optional[str],
        ],
        Dict[str, Union[int, str]],
    ] = process_prompt,
    num_test_cases: int = 0,
    pass_k: int = 1,
    enumerate_prompt_and_pass_id: bool = True,
    guided_decode_json_schema: Optional[str] = None,
) -> List[Dict[str, Optional[str]]]:
    if not prompt_path.exists():
        raise ValueError("Please provide valid path for prompts.")
    prompt_dataset: List[Dict[str, Union[int, str]]] = json.loads(
        prompt_path.read_text()
    )
    if num_test_cases > 0:
        # Select the test cases distributed evenly across the dataset
        dataset_size = len(prompt_dataset)
        if dataset_size > num_test_cases:
            step = dataset_size // num_test_cases
            prompt_dataset = prompt_dataset[::step][:num_test_cases]
    response_result = []

    # Run the LLMs in parallel
    with ThreadPoolExecutor(max_workers=run_llm_in_parallel) as executor:
        futures = [
            (
                executor.submit(
                    to_process_prompt,
                    idx,
                    k,
                    llms,
                    test_case,
                    guided_decode_json_schema,
                )
                if enumerate_prompt_and_pass_id
                else executor.submit(
                    to_process_prompt,
                    cast(int, test_case["prompt_id"]),
                    cast(int, test_case["pass_id"]),
                    llms,
                    test_case,
                    guided_decode_json_schema,
                )
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
    return response_result
