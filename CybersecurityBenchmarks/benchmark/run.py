# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import logging
from enum import Enum

from pathlib import Path

from . import llm

from .benchmark import Benchmark

from .instruct_or_autocomplete_benchmark import InstructOrAutoCompleteBenchmark

from .mitre_benchmark import MitreBenchmark

LOG: logging.Logger = logging.getLogger(__name__)


class BenchmarkKind(Enum):
    autocomplete = "autocomplete"
    instruct = "instruct"
    mitre = "mitre"

    def __str__(self) -> str:
        return self.value


def main(
    *,
    default_benchmark: BenchmarkKind | None = None,
    llms_under_test: list[llm.LLM] | None = None,
    expansion_llm: llm.LLM | None = None,
    judge_llm: llm.LLM | None = None,
    default_prompt_path: str | None = None,
    default_response_path: str | None = None,
    default_judge_response_path: str | None = None,
    default_stat_path: str | None = None,
    default_use_precomputed_responses: bool = False,
    default_only_generate_llm_responses: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description="Run the Cybersecurity Benchmarks.")

    parser.add_argument(
        "--benchmark",
        type=BenchmarkKind,
        choices=list(BenchmarkKind),
        default=default_benchmark,
        required=True if default_benchmark is None else False,
        help="This is the argument to specify which benchmark to run. Currently supported benchmarks are: autocomplete, instruct, mitre.",
    )

    if llms_under_test is None:
        parser.add_argument(
            "--llm-under-test",
            help=f"LLM to benchmark provided as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
            action="append",
            required=True,
        )

    parser.add_argument(
        "--prompt-path",
        help="The under-test LLM(s) would be queried with the prompts in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_prompt_path,
        required=default_prompt_path is None and not default_use_precomputed_responses,
    )

    parser.add_argument(
        "--response-path",
        help="The under-test LLM(s) would store their responses in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_response_path,
        required=default_response_path is None,
    )

    parser.add_argument(
        "--stat-path",
        help="After processing the under-test LLM's responses, the final status would be stored in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_stat_path,
        required=False,
    )

    if judge_llm is None:
        parser.add_argument(
            "--judge-llm",
            help=f"Used for MITRE benchmark. We need to specify the llm model to judge the final expanded response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
        )
    if expansion_llm is None:
        parser.add_argument(
            "--expansion-llm",
            help=f"Used for MITRE benchmark. We need to specify the llm model to expand the context from the initial query response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
        )

    parser.add_argument(
        "--judge-response-path",
        help="Used for MITRE benchmark. The path to save the judge responses.",
        default=default_judge_response_path,
        required=False,
    )

    parser.add_argument(
        "--run-llm-in-parallel",
        action="store_true",
        help="If this flag is set, we will make parallel calls to the LLM to speed up execution.",
    )
    parser.add_argument(
        "--num-queries-per-prompt",
        help="Set the number of queries per prompt. Default value is 1.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use-precomputed-responses",
        action="store_true",
        default=default_use_precomputed_responses,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--only-generate-llm-responses",
        action="store_true",
        default=default_only_generate_llm_responses,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--num-test-cases",
        type=int,
        default=0,
        help="Only set this argument for testing purpose: If set to a positive value, we test the given number of cases instead of all cases.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )

    materialized_llms_under_test = llms_under_test or [
        llm.create(specification) for specification in args.llm_under_test
    ]
    if len(materialized_llms_under_test) == 0:
        raise ValueError("No LLMs were specified.")

    benchmark: Benchmark
    stat_response_path = Path(args.stat_path) if args.stat_path else None
    if (
        args.benchmark == BenchmarkKind.autocomplete
        or args.benchmark == BenchmarkKind.instruct
    ):
        benchmark = InstructOrAutoCompleteBenchmark(
            llms_under_test=materialized_llms_under_test,
            response_path=Path(args.response_path),
            num_test_cases=args.num_test_cases,
            pass_k=args.num_queries_per_prompt,
            stat_path=stat_response_path,
        )
    elif args.benchmark == BenchmarkKind.mitre:
        if (judge_llm is None or expansion_llm is None) and (
            not args.judge_llm or not args.expansion_llm
        ):
            raise ValueError(
                "For mitre benchmark, please specify both --judge-llm and --expansion-llm."
            )
        judge_response_path = (
            Path(args.judge_response_path) if args.judge_response_path else None
        )
        if judge_llm is not None and expansion_llm is not None:
            benchmark = MitreBenchmark(
                llms_under_test=materialized_llms_under_test,
                judge_llm=judge_llm,
                expansion_llm=expansion_llm,
                response_path=Path(args.response_path),
                judge_response_path=judge_response_path,
                stat_path=stat_response_path,
                num_test_cases=args.num_test_cases,
                pass_k=args.num_queries_per_prompt,
            )
        else:
            benchmark = MitreBenchmark(
                llms_under_test=materialized_llms_under_test,
                judge_llm=llm.create(args.judge_llm),
                expansion_llm=llm.create(args.expansion_llm),
                response_path=Path(args.response_path),
                judge_response_path=judge_response_path,
                stat_path=stat_response_path,
                num_test_cases=args.num_test_cases,
                pass_k=args.num_queries_per_prompt,
            )
    else:
        raise AssertionError("Invalid benchmark option")

    if args.use_precomputed_responses:
        # Using precomputed responses require a valid response path.
        response_path = Path(args.response_path)
        if not response_path.exists():
            raise ValueError(
                f"Please provide valid path for responses. This path does not exist: {response_path}"
            )
        LOG.info("Skipping LLM query and using precomputed response.")
    else:
        # If we are not using precomputed responses, then we require a valid prompt path.
        prompt_path = Path(args.prompt_path)
        if not prompt_path.exists():
            raise ValueError(
                f"Please provide valid path for prompts. This path does not exist: {prompt_path}"
            )
        LOG.info("Querying LLMs for responses...")
        benchmark.query_llm_to_generate_responses(
            prompt_path=Path(args.prompt_path),
            run_llm_in_parallel=args.run_llm_in_parallel,
        )

    if not args.only_generate_llm_responses:
        LOG.info("Evaluating...")
        asyncio.run(benchmark.run())


if __name__ == "__main__":
    main()
