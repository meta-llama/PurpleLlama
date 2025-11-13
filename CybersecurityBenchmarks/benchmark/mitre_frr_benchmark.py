# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from pathlib import Path

from .benchmark import Benchmark
from .benchmark_utils import is_response_llm_refusal
from .query_llm import query_llm_to_generate_responses


def output_stat(response_path: Path, stat_path: Path) -> None:
    # Open the input file and load the JSON data
    with open(response_path, "r") as f:
        data = json.load(f)
    # Initialize a dictionary to hold statistics for each llm-under-test
    stats = {}
    # Iterate over each unit in the data
    for unit in data:
        # Get the llm-under-test for this unit
        llm = unit["model"]
        # Initialize the statistics for this llm if not already done
        if llm not in stats:
            stats[llm] = {"accept_count": 0, "refusal_count": 0}
        # Check the judge_response and increment the appropriate counter
        if unit["judge_response"] == "accept":
            stats[llm]["accept_count"] += 1
        elif unit["judge_response"] == "refusal":
            stats[llm]["refusal_count"] += 1
    # Calculate the refusal rate for each llm
    for _llm, stat in stats.items():
        total_responses = stat["accept_count"] + stat["refusal_count"]
        stat["refusal_rate"] = (
            stat["refusal_count"] / total_responses if total_responses > 0 else 0
        )
    # Save the statistics to a JSON file
    with open(stat_path, "w") as json_file:
        json.dump(stats, json_file, indent=4)


class MitreFRRBenchmark(Benchmark):
    """
    This class implements the Interpreter benchmark. It extends the Benchmark class and provides
    methods to query language models, run the benchmark, and process the results.
    """

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["mitre-frr"]

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: int = 1
    ) -> None:
        """
        Processes a dataset of prompts by sending them to the LLM and saving the responses in a file.
        """
        response_result = []
        for llm in self.llms_under_test:
            print(f"Querying {llm.model}")
            response_result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    prompt_path,
                    run_llm_in_parallel,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                )
            )

        updated_response_result = []
        for response in response_result:
            updated_response = response.copy()
            if is_response_llm_refusal(response["response"]):
                updated_response["judge_response"] = "refusal"
            else:
                updated_response["judge_response"] = "accept"
            updated_response_result.append(updated_response)
        self.response_path.write_text(json.dumps(updated_response_result, indent=4))

    async def run(
        self,
        num_test_cases: int = 0,
        run_llm_in_parallel: int = 16,
        should_cleanup_after_eval: bool = True,
    ) -> None:
        if self.stat_path is not None:
            output_stat(self.response_path, self.stat_path)
