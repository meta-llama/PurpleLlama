# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import argparse
import asyncio
import logging

from pathlib import Path

from typing import Callable, Optional

from . import llm
from .autonomous_uplift_benchmark import AutonomousUpliftBenchmark
from .autopatching_benchmark import AutoPatchingBenchmark

from .benchmark import Benchmark, BenchmarkConfig
from .cache_handler import CacheHandler
from .canary_exploit_benchmark import CanaryExploitBenchmark

from .instruct_or_autocomplete_benchmark import InstructOrAutoCompleteBenchmark

from .interpreter_benchmark import InterpreterBenchmark
from .mitre_benchmark import MitreBenchmark

from .mitre_frr_benchmark import MitreFRRBenchmark
from .multiturn_phishing_benchmark import MultiturnPhishingBenchmark

from .prompt_injection_benchmark import PromptInjectionBenchmark

from .visual_prompt_injection_benchmark import VisualPromptInjectionBenchmark

LOG: logging.Logger = logging.getLogger(__name__)

Benchmark.register_benchmark(InstructOrAutoCompleteBenchmark)
Benchmark.register_benchmark(MitreBenchmark)
Benchmark.register_benchmark(MitreFRRBenchmark)
Benchmark.register_benchmark(PromptInjectionBenchmark)
Benchmark.register_benchmark(CanaryExploitBenchmark)
Benchmark.register_benchmark(InterpreterBenchmark)
Benchmark.register_benchmark(AutonomousUpliftBenchmark)
Benchmark.register_benchmark(VisualPromptInjectionBenchmark)
Benchmark.register_benchmark(MultiturnPhishingBenchmark)
Benchmark.register_benchmark(AutoPatchingBenchmark)


def validate_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if not args.use_precomputed_responses:
        if args.prompt_path is None:
            parser.error(
                "The following argument is required: --prompt-path, unless argument --use-precomputed-responses is specified."
            )
        if "llm_under_test" in args and args.llm_under_test is None:
            parser.error(
                "The following argument is required: --llm-under-test, unless argument --use-precomputed-responses is specified."
            )

    return args


def main(
    *,
    default_benchmark: str | None = None,
    llms_under_test: list[llm.LLM] | None = None,
    expansion_llm: llm.LLM | None = None,
    judge_llm: llm.LLM | None = None,
    default_prompt_path: str | None = None,
    default_response_path: str | None = None,
    default_judge_response_path: str | None = None,
    default_stat_path: str | None = None,
    default_use_precomputed_responses: bool = False,
    default_only_generate_llm_responses: bool = False,
    default_llm_creator: Callable[
        [str, int, Optional[list[str]]], llm.LLM
    ] = llm.create,
    default_run_llm_in_parallel: int = 1,
    default_should_cleanup_after_eval: bool = True,
    default_container_repository: Optional[str] = None,
) -> None:
    parser = argparse.ArgumentParser(description="Run the Cybersecurity Benchmarks.")

    parser.add_argument(
        "--benchmark",
        default=default_benchmark,
        required=True if default_benchmark is None else False,
        help="This is the argument to specify which benchmark to run. Currently supported benchmarks are: autocomplete, instruct, mitre.",
    )

    if llms_under_test is None:
        parser.add_argument(
            "--llm-under-test",
            help=f"LLM to benchmark provided as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
            action="append",
        )

    parser.add_argument(
        "--prompt-path",
        help="The under-test LLM(s) would be queried with the prompts in this file. This argument works for benchmarks: autocomplete, instruct, mitre.",
        default=default_prompt_path,
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

    parser.add_argument(
        "--judge-llm",
        help=f"Used for MITRE benchmark. We need to specify the llm model to judge the final expanded response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
    )

    parser.add_argument(
        "--expansion-llm",
        help=f"Used for MITRE benchmark. We need to specify the llm model to expand the context from the initial query response as {llm.SPECIFICATION_FORMAT}, e.g., {llm.EXAMPLE_SPECIFICATION}",
    )

    parser.add_argument(
        "--judge-prompt-path",
        help="Used for Autonomous Uplift benchmark. The judge LLM is queried with the prompts in this file.",
        required=False,
    )

    parser.add_argument(
        "--judge-response-path",
        help="Used for MITRE benchmark. The path to save the judge responses.",
        default=default_judge_response_path,
        required=False,
    )

    parser.add_argument(
        "--run-llm-in-parallel",
        nargs="?",
        type=int,
        const=16,
        default=default_run_llm_in_parallel,
        help=(
            "Specify the number of parallel calls to the LLM to speed up execution. "
            "When not provided, it defaults to 1 thread. "
            "When set as a flag (--run-llm-in-parallel), it defaults to 16 threads. "
            "Users can also specify the number of threads, e.g., --run-llm-in-parallel=8."
        ),
    )

    parser.add_argument(
        "--truncate-input",
        action="store_true",
        default=False,
        help="Set this argument to truncate the input provided to the LLM.",
    )

    parser.add_argument(
        "--should-cleanup-after-eval",
        action="store_true",
        default=default_should_cleanup_after_eval,
        help="Set this argument to configure post-evaluation cleanup.",
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
        help="Only set this argument for testing purpose: If set to a positive value, we test the given number of prompts instead of all prompts.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--enable-cache",
        action="store_true",
        help="Enable caching of LLM responses to a local file.",
    )

    parser.add_argument(
        "--cache-file",
        default="llm_cache.json",
        help="Specify the cache file path. Default is 'llm_cache.json'.",
    )

    parser.add_argument(
        "--agent",
        type=str,
        help="For agentic benchmarks - which agent implementation to use.",
    )

    parser.add_argument(
        "--enable-lf",
        type=str,
        default=None,
        help="Optional keyword to enable LlamaFirewall on all input and output layers.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Set the max token size for querying LLMs.",
    )

    parser.add_argument(
        "--cot-end-tokens",
        nargs="*",
        default=[],
        help="A list of strings that should be used to locate the end of the CoT.",
    )

    # NOTE: this argument will be saved in `BenchmarkConfig.additional_args`
    # which can be used by any individual benchmark to pass additional arguments
    # specific to that benchmark. For example, the `AutoPatchingBenchmark` uses
    # this argument to pass the `--test-mode` flag. The command line argument
    # looks like: `--additional-args \'--test-mode\'`. The synatx for quote
    # escaping may be shell-specific.
    parser.add_argument(
        "--additional-args",
        type=str,
        help="Additional arguments to pass to the benchmark.",
    )

    parser.add_argument(
        "--container-repository",
        type=str,
        help="Specify a reference to a container repository containing images used in this benchmark to avoid rebuilding images locally. Ensure this does not point to ARVO images, but rather images produced by our benchmark.",
        default=default_container_repository,
    )

    parser.add_argument(
        "--input-modality",
        type=str,
        help="Optional keyword to specify the input modality/modialities for the benchmark. Currently supported modalities are: text, image, text_and_image for ThreatIntelReasoning benchmark.",
        default=None,
    )

    args = validate_arguments(parser)

    logging.basicConfig(
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )
    AutoPatchingBenchmark.set_log_level(logging.DEBUG if args.debug else logging.INFO)

    materialized_llms_under_test = []
    if not args.use_precomputed_responses:
        if llms_under_test is None and not args.llm_under_test:
            raise ValueError("No LLMs were specified.")
        materialized_llms_under_test = llms_under_test or [
            default_llm_creator(specification, args.max_tokens, args.cot_end_tokens)
            for specification in args.llm_under_test
        ]

    stat_response_path = Path(args.stat_path) if args.stat_path else None
    judge_response_path = (
        Path(args.judge_response_path) if args.judge_response_path else None
    )
    if args.judge_llm:
        judge_llm = default_llm_creator(args.judge_llm, 2048, None)
    if args.expansion_llm:
        expansion_llm = default_llm_creator(args.expansion_llm, 2048, None)

    judge_prompt_path = None
    if args.judge_prompt_path:
        if Path(args.judge_prompt_path).exists():
            judge_prompt_path = Path(args.judge_prompt_path)
        else:
            raise ValueError(
                f"Please provide valid path for judge prompts. This path does not exist: {args.judge_prompt_path}"
            )

    if args.enable_cache:
        cache_handler = CacheHandler(args.cache_file)
        for materialized_llm in materialized_llms_under_test:
            materialized_llm.cache_handler = cache_handler
    else:
        for materialized_llm in materialized_llms_under_test:
            materialized_llm.cache_handler = None

    config: BenchmarkConfig = BenchmarkConfig(
        llms_under_test=materialized_llms_under_test,
        judge_llm=judge_llm,
        expansion_llm=expansion_llm,
        response_path=Path(args.response_path),
        judge_prompt_path=judge_prompt_path,
        judge_response_path=judge_response_path,
        stat_path=stat_response_path,
        num_test_cases=args.num_test_cases,
        pass_k=args.num_queries_per_prompt,
        agent=args.agent,
        enable_lf=args.enable_lf,
        cot_end_tokens=args.cot_end_tokens,
        additional_args=(args.additional_args or "").strip("'\""),
        container_repository=args.container_repository,
        prompt_path=args.prompt_path,
        max_concurrency=args.run_llm_in_parallel,
        truncate_input=args.truncate_input,
        input_modality=args.input_modality,
    )

    benchmark: Benchmark = Benchmark.create_instance(args.benchmark, config)

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
        # The variable num_test_cases is defined to specify the number of prompts that need to be evaluated. However, if we are testing multiple LLMs or if we are conducting multiple rounds of testing queries per prompt, these factors will need below steps to reflect in `num_test_cases`` during the expansion/judge step.
        if not args.use_precomputed_responses:
            if len(materialized_llms_under_test) > 0:
                args.num_test_cases = (
                    len(materialized_llms_under_test) * args.num_test_cases
                )
            if args.num_queries_per_prompt:
                args.num_test_cases = args.num_test_cases * args.num_queries_per_prompt
        asyncio.run(
            benchmark.run(
                args.num_test_cases,
                args.run_llm_in_parallel,
                args.should_cleanup_after_eval,
            )
        )


if __name__ == "__main__":
    main()
