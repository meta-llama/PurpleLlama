# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import logging

import os

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Union

import openai
from langchain.llms import Anyscale, Together

from tqdm import tqdm

from typing_extensions import override


NUM_LLM_RETRIES = 10

MAX_TOKENS = 1000

LOG: logging.Logger = logging.getLogger(__name__)


class LLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in self.valid_models():
            raise ValueError(
                f"{model} is not a valid model for {type(self).__name__}. Valid models are: {', '.join(self.valid_models())}."
            )
        self.model: str = model
        self.api_key: str | None = api_key

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Abstract method to query an LLM with a given prompt and return the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def query_with_retries(
        self, prompt: str, retries: int = NUM_LLM_RETRIES, backoff_factor: float = 0.5
    ) -> str:
        """
        Wrapper around query that retries the query if it fails.
        Implements exponential backoff.
        """
        last_exception = None
        for retry in range(retries):
            try:
                return self.query(prompt)
            except Exception as exception:
                last_exception = exception
                sleep_time = backoff_factor * (2**retry)
                time.sleep(sleep_time)
        raise RuntimeError(
            f"Unable to query LLM after {retries} retries: {last_exception}"
        )

    def valid_models(self) -> list[str]:
        """List of valid model parameters, e.g. 'gpt-3.5-turbo' for GPT"""
        return []


SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"


def create(identifier: str) -> LLM:
    """
    Takes a string of the format <SPECIFICATION_FORMAT> and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.
    """
    api_key: str | None = None
    split = identifier.split("::")
    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    if api_key is None:
        raise ValueError(f"No API key provided for {provider}: {name}")

    if provider == "OPENAI":
        return OPENAI(name, api_key)
    if provider == "ANYSCALE":
        return ANYSCALE(name, api_key)
    if provider == "TOGETHER":
        return TOGETHER(name, api_key)

    raise ValueError(f"Unknown provider: {provider}")


class OPENAI(LLM):
    """Accessing OPENAI"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        # pyre-ignore[4, 16]: we're using slightly older version of openai internally
        self.client = openai.OpenAI(api_key=api_key)  # noqa

    @override
    def query(self, prompt: str) -> str:
        # Best-level effort to suppress openai log-spew.
        # Likely not work well in multi-threaded environment.
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4"]


class ANYSCALE(LLM):
    """Accessing ANYSCALE"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        os.environ["ANYSCALE_API_BASE"] = "https://api.endpoints.anyscale.com/v1"
        os.environ["ANYSCALE_API_KEY"] = api_key

    @override
    def query(self, prompt: str) -> str:
        llm = Anyscale(
            model_name=self.model,
            temperature=0.75,
            top_p=1,
            model_kwargs={"max_new_tokens": MAX_TOKENS},
        )
        response = llm(prompt)
        return "".join(response)

    @override
    def valid_models(self) -> list[str]:
        return [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
        ]


class TOGETHER(LLM):
    """Accessing TOGETHER"""

    @override
    def query(self, prompt: str) -> str:
        llm = Together(
            model_name=self.model,
            temperature=0.75,
            top_p=1,
            max_tokens=MAX_TOKENS,
            together_api_key=self.api_key,
        )
        response = llm(prompt)
        return "".join(response)

    @override
    def valid_models(self) -> list[str]:
        return [
            "togethercomputer/llama-2-7b",
            "togethercomputer/llama-2-7b-chat",
            "togethercomputer/llama-2-13b",
            "togethercomputer/llama-2-13b-chat",
            "togethercomputer/llama-2-70b",
            "togethercomputer/llama-2-70b-chat",
            "togethercomputer/CodeLlama-7b",
            "togethercomputer/CodeLlama-7b-Python",
            "togethercomputer/CodeLlama-7b-Instruct",
            "togethercomputer/CodeLlama-13b",
            "togethercomputer/CodeLlama-13b-Python",
            "togethercomputer/CodeLlama-13b-Instruct",
        ]


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
            tqdm(prompt_dataset, desc="Processing prompts", unit="prompt")
        ):
            response_result.append(to_process_prompt(1, idx, llms, test_case))
    return response_result
