# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List

from .base_test_generator import BaseTestGenerator
from .constraint_satisfaction.c.c_generator import CGenerator
from .constraint_satisfaction.javascript.javascript_generator import JavascriptGenerator
from .constraint_satisfaction.python.python_generator import PythonGenerator
from .constraint_satisfaction.sqlite.sqlite_generator import SQLiteGenerator
from .memory_corruption.c_buffer_overflow_generator import CBufferOverflowGenerator
from .memory_corruption.examples import build_with_prompt, EXAMPLES

from .memory_corruption.generator import generate_advanced_memory_corruption_test_cases


def generate_test_cases(
    generators: List[BaseTestGenerator], num_test_case: int
) -> List[Dict[str, str]]:
    prompts = []

    for generator in generators:
        for _ in range(0, num_test_case):
            prompts.append(generator.generate_test_case_as_dict())

    return prompts


def generate_constraint_satisfaction_test_cases(
    num_test_case: int,
) -> List[Dict[str, str]]:
    generators: List[BaseTestGenerator] = [
        CGenerator(),
        JavascriptGenerator(),
        SQLiteGenerator(),
        PythonGenerator(),
    ]
    prompts = generate_test_cases(generators, num_test_case)
    return prompts


def generate_memory_corruption_test_cases(
    num_test_case: int, memory_corruption_path: str
) -> List[Dict[str, str]]:
    generators: List[BaseTestGenerator] = [
        CBufferOverflowGenerator(),
    ]
    # Simple buffer overflow tests generation
    prompts = generate_test_cases(generators, num_test_case)

    # Advanced buffer overflow static tests
    prompts += build_with_prompt(
        Path(memory_corruption_path),
        Path(memory_corruption_path + "build"),
        EXAMPLES,
    )

    # Advanced buffer overflow randomized tests generation
    prompts += generate_advanced_memory_corruption_test_cases()

    return prompts


def supported_cpu() -> bool:
    if platform.machine() == "x86_64":
        return True
    return False


def invoke_main() -> None:
    if supported_cpu() is False:
        print("Unsupported CPU architecture. Only x86_64 is supported for now.")
        return

    parser = argparse.ArgumentParser(
        description="Run the Canary Exploit Test Generation."
    )

    parser.add_argument(
        "--language",
        default="all",
        help="This is the argument to specify the language of the tests we generate.",
    )

    parser.add_argument(
        "--num_test_case",
        type=int,
        default=100,
        help="This is the argument to specify the number of tests to generate.",
    )
    parser.add_argument(
        "--prompt_path",
        default="CybersecurityBenchmarks/datasets/canary_exploit/canary_exploit.json",
        help="This is the path to which prompts will be written for the canary exploit tests.",
    )
    parser.add_argument(
        "--memory_corruption_path",
        default="CybersecurityBenchmarks/datasets/canary_exploit/memory_corruption/",
        help="This is the argument to specify the path containing the memory corruption tests.",
    )

    args = parser.parse_args()
    generators: List[BaseTestGenerator]
    prompts: List[Dict[str, str]] = []

    if args.language == "all":
        # Add Constraint Satisfaction Tests
        prompts = generate_constraint_satisfaction_test_cases(args.num_test_case)

        # Add Memory Corruption Tests (static)
        prompts += generate_memory_corruption_test_cases(
            args.num_test_case, args.memory_corruption_path
        )
    elif args.language == "c":
        generators = [
            CGenerator(),
        ]
        prompts = generate_test_cases(generators, args.num_test_case)
    elif args.language == "javascript":
        generators = [
            JavascriptGenerator(),
        ]
        prompts = generate_test_cases(generators, args.num_test_case)
    elif args.language == "sqlite":
        generators = [
            SQLiteGenerator(),
        ]
        prompts = generate_test_cases(generators, args.num_test_case)
    elif args.language == "python":
        generators = [
            PythonGenerator(),
        ]
        prompts = generate_test_cases(generators, args.num_test_case)
    elif args.language == "memory_corruption":
        prompts = generate_memory_corruption_test_cases(
            args.num_test_case, args.memory_corruption_path
        )

    # Dump the prompts to a json file
    with open(args.prompt_path, "w") as f:
        json.dump(prompts, f, indent=4)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
