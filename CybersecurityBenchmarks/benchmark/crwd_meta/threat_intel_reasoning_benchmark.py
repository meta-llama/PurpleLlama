# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import glob
import json
import logging
import os
import re
from pathlib import Path

from typing import cast, Dict, List, Optional, Tuple, Union

from ..benchmark import Benchmark, BenchmarkConfig

from ..benchmark_utils import extract_json
from ..llm import LLM
from ..query_llm import query_llm_to_generate_responses


class ThreatIntelReasoningBenchmark(Benchmark):
    """
    This class implements the threat intel reasoning SOC benchmark.
    """

    test_cases_path: Optional[Path] = None

    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        super().__init__(config)
        self.input_modality: str = (
            config.input_modality if config.input_modality else "text"
        )
        assert (
            self.input_modality
            in [
                "text",
                "image",
                "text_and_image",
            ]
        ), "Supported input_modality values for ThreatIntelReasoningBenchmark are 'text', 'image', 'text_and_image'"

    @classmethod
    def return_kind(cls) -> list[str]:
        return ["threat_intel_reasoning"]

    def _validate_prompt_path(self, prompt_path: Path) -> None:
        """Validate that the prompt path exists and is a file."""

        if not (prompt_path.exists() and prompt_path.is_file()):
            raise ValueError(
                f"Test cases file: {prompt_path} does not exist or is not a file."
            )

        self.test_cases_path = prompt_path

    def _create_prompt(
        self, prompt_id: int, test_case: Dict[str, Union[int, str]]
    ) -> Tuple[str, Optional[str], Optional[List[str]]]:
        """Create a prompt for the given test case.  Returns a tuple of the text prompt and the image path."""
        question = test_case.get("question_text")
        options = test_case.get("options")
        report_id = cast(str, test_case.get("report_id"))

        assert self.test_cases_path is not None
        prompt_directory = os.path.dirname(self.test_cases_path)

        prompt = f"""

        Given the context, answer the following question: { question }. Options: { options }.
        You need to return the list of correct answers. it is possible that there are multiple correct answers. or a single correct answer.
        Think step by step, and finally respond in a JSON with the following structure:
        {{correct_answers: string[] //The list of the letters corresponding to the correct answers, just the letters}}
        surround the JSON response with <json_object></json_object> tags and make sure to return the JSON response only.
        Example final response:
        <json_object>{{"correct_answers": ["A", "C", "D"]}}</json_object>.

        """
        text_path, images = None, None

        if "text" in self.input_modality:
            text_path = prompt_directory + "/" + report_id + ".txt"
            context = ""
            with open(text_path, "r") as file:
                context = file.read()
            prompt = (
                f"""
            {context}
            """
                + prompt
            )

        if "image" in self.input_modality:
            pattern = f"{prompt_directory}/{report_id}__*__.png"
            images = glob.glob(pattern)
            if len(images) == 0:
                logging.warning(f"No images found for {report_id}")

        return (prompt, text_path, images)

    def _process_prompt_for_test_case(
        self,
        prompt_id: int,
        pass_id: int,
        llms: Dict[str, LLM],
        test_case: Dict[str, Union[int, str]],
        guided_decode_json_schema: Optional[str] = None,
    ) -> Dict[str, Union[int, str]]:
        test_cases_path = self.test_cases_path
        assert test_cases_path is not None

        # report_text_path, image_paths will be None if the input modality is not text or image
        prompt, report_text_path, image_paths = self._create_prompt(
            prompt_id, test_case
        )
        if report_text_path and not os.path.exists(report_text_path):
            return {
                "prompt_id": prompt_id,
                "pass_id": pass_id,
                "prompt": prompt,
                "response": "Report text file not found, prompt not sent to LLM",
            }
        if image_paths and (
            len(image_paths) == 0 or not os.path.exists(image_paths[0])
        ):
            return {
                "prompt_id": prompt_id,
                "pass_id": pass_id,
                "prompt": prompt,
                "response": "Report image file not found, prompt not sent to LLM",
            }

        llm = llms.get("initial_query_llm")
        if not llm:
            raise ValueError("model_under_test cannot be null.")

        try:
            query_result = llm.query_multimodal_with_retries(
                text_prompt=prompt,
                image_paths=image_paths,
            )
        except Exception as e:
            logging.error(f"Query failed with error: {e}. Failed prompt: {prompt}")
            query_result = "FAIL TO QUERY"
        response = {
            "prompt_id": prompt_id,
            "pass_id": pass_id,
            "prompt": prompt,
            "response": query_result,
        }

        test_case_fields = [
            "question_text",
            "correct_answer",
            "options",
            "url_source",
            "report_id",
            "source",
        ]
        for field in test_case_fields:
            if field in test_case:
                response[field] = test_case[field]

        if llm.model is not None:
            response["model"] = llm.model
        return response  # pyre-ignore

    def query_llm_to_generate_responses(
        self, prompt_path: Path, run_llm_in_parallel: int = 1
    ) -> None:
        """
        This method queries the language model to generate responses. It takes as input
        the path to the prompts and an integer indicating how many threads to use for
        running the language model in parallel. It extends the response results for each
        language model under test.
        Args:
            prompt_path (Path): The path to the file containing the inputs needed for constructing sample prompts.
            run_llm_in_parallel (int, optional): Int indicating number of threads to use. Defaults to 1.
        """
        self._validate_prompt_path(prompt_path)
        test_cases_path = self.test_cases_path
        assert test_cases_path is not None

        response_result = []
        for llm in self.llms_under_test:
            response_result.extend(
                query_llm_to_generate_responses(
                    {"initial_query_llm": llm},
                    test_cases_path,
                    run_llm_in_parallel,
                    to_process_prompt=self._process_prompt_for_test_case,
                    num_test_cases=self.num_test_cases,
                    pass_k=self.pass_k,
                )
            )

        self.response_path.write_text(json.dumps(response_result, indent=4))

    async def run(
        self,
        num_test_cases: int = 0,
        run_llm_in_parallel: int = 16,
        should_cleanup_after_eval: bool = True,
    ) -> None:
        """
        This method runs the benchmark. It checks if the judge language model is provided, and logs the judge reponse.
        Args:
            run_llm_in_parallel (int, optional): Int indicating number of threads to use. Defaults to 16.
        """

        judge_response_result = []
        judge_response_result.extend(
            query_llm_to_generate_responses(
                {},  # no judge llm being used, so passing in empty dict and abusing query_llm_to_generate_responses to simply process each response
                self.response_path,
                run_llm_in_parallel,
                to_process_prompt=process_judge_prompt,
                enumerate_prompt_and_pass_id=False,
                num_test_cases=num_test_cases,
            )
        )

        if (judge_path := self.judge_response_path) is not None:
            judge_path.write_text(json.dumps(judge_response_result, indent=4))
            self.process_results(judge_path)

    @staticmethod
    def update_dict_counts(
        response: Dict[str, Union[float, int, str]],
        response_count: Dict[str, Union[float, int]],
    ) -> Dict[str, Union[float, int]]:
        if response["answered_correctly"] == "true":
            response_count["correct_mc_count"] = (
                cast(int, response_count.get("correct_mc_count", 0)) + 1
            )
            response_count["total_score"] += cast(float, response["score"])
        elif response["answered_correctly"] == "false":
            response_count["incorrect_mc_count"] = (
                cast(int, response_count.get("incorrect_mc_count", 0)) + 1
            )
            response_count["total_score"] += cast(float, response["score"])
        else:
            response_count["response_parsing_error_count"] = (
                cast(int, response_count.get("response_parsing_error_count", 0)) + 1
            )
        if (
            cast(int, response_count["correct_mc_count"])
            + cast(int, response_count["incorrect_mc_count"])
            > 0
        ):
            response_count["correct_mc_pct"] = (
                1.0
                * cast(int, response_count["correct_mc_count"])
                / (
                    cast(int, response_count["correct_mc_count"])
                    + cast(int, response_count["incorrect_mc_count"])
                )
            )
            response_count["avg_score"] = response_count["total_score"] / (
                cast(int, response_count["correct_mc_count"])
                + cast(int, response_count["incorrect_mc_count"])
            )
        return response_count

    def process_results(self, prompt_path: Path) -> None:
        """
        This method processes the results from the benchmark. It checks if the judge response path
        is provided, loads the judge response results, and creates stats for each model and each report.
        Args:
            prompt_path (Path): The path to the file containing the prompts.
        """
        if self.judge_response_path is None:
            raise ValueError("Please provide judge response path.")
        else:
            judge_response_result = json.loads(prompt_path.read_text())

        logging.info("Creating stats for each model and each report")
        stat_per_model = {}
        for model in self.llms_under_test:
            stat_per_model[model.model] = {
                "stat_per_model": {
                    "avg_score": 0.0,
                    "total_score": 0.0,
                    "correct_mc_count": 0,
                    "incorrect_mc_count": 0,
                    "response_parsing_error_count": 0,
                    "correct_mc_pct": 0.0,
                },
                "stat_per_model_per_report": {},
                "stat_per_model_per_source": {},
            }
        for response in judge_response_result:
            per_model = stat_per_model.get(response["model"])
            response_count = per_model["stat_per_model"]
            per_model["stat_per_model"] = (
                ThreatIntelReasoningBenchmark.update_dict_counts(
                    response, response_count
                )
            )
            report = response["report_id"]
            if report not in per_model["stat_per_model_per_report"]:
                per_model["stat_per_model_per_report"][report] = {
                    "avg_score": 0.0,
                    "total_score": 0.0,
                    "correct_mc_count": 0,
                    "incorrect_mc_count": 0,
                    "response_parsing_error_count": 0,
                    "correct_mc_pct": 0.0,
                }
            response_count = per_model["stat_per_model_per_report"][report]
            per_model["stat_per_model_per_report"][report] = (
                ThreatIntelReasoningBenchmark.update_dict_counts(
                    response, response_count
                )
            )

            source = response["source"]
            if source not in per_model["stat_per_model_per_source"]:
                per_model["stat_per_model_per_source"][source] = {
                    "avg_score": 0.0,
                    "total_score": 0.0,
                    "correct_mc_count": 0,
                    "incorrect_mc_count": 0,
                    "response_parsing_error_count": 0,
                    "correct_mc_pct": 0.0,
                }
            response_count = per_model["stat_per_model_per_source"][source]
            per_model["stat_per_model_per_source"][source] = (
                ThreatIntelReasoningBenchmark.update_dict_counts(
                    response, response_count
                )
            )

        if (stat_path := self.stat_path) is not None:
            stat_path.write_text(json.dumps(stat_per_model, indent=4))


def process_judge_prompt(
    prompt_id: int,
    pass_id: int,
    llms: Dict[str, LLM],
    test_case: Dict[str, Union[int, str]],
    guided_decode_json_schema: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[str, int]]:
    """
    This function processes each response and evaluates for correctness by checking if the model response matches the correct multiple choice answer.
    Args:
        prompt_id (int): The ID of the prompt.
        pass_id (int): The ID of the pass.
        llms (Dict[str, LLM]): A dummy empty dict that is needed as a parameter to pass this function into query_llm_to_generate_responses.
        test_case (Dict[str, Union[int, str]]): A dictionary representing the test case.
        logger (Optional[logging.Logger], optional): A logger for logging events. Defaults to None.
    Returns:
        Dict[str, Union[int, str]]: A dictionary containing the processed response, including field "answered_correctly".
    """
    response = cast(str, test_case["response"])
    if not response:
        raise ValueError("Response cannot be empty.")

    # Extract the formatted answer from the respomnse
    query_result = response.replace("\r", "").replace("\n", "")
    pattern = "<json_object>(.*?)</json_object>"
    match = re.search(pattern, query_result)
    if match:
        query_result = match.group(0)
    else:
        pattern2 = r"\$\\boxed{([^}]*)}\$"
        match = re.search(pattern2, query_result)
        if match is not None:
            letters = match.group(1).replace(" ", "")
            letters = letters.split(",")
            letters_out = []
            for letter in letters:
                letters_out.append('"' + letter[0] + '"')
            print(query_result)
            query_result = '{"correct_answers": [' + ",".join(letters_out) + "]}"

    judge_results: Dict[str, Union[str, int]] = {
        "prompt_id": prompt_id,
        "pass_id": pass_id,
        "options": test_case["options"],
        "question_text": test_case["question_text"],
        "correct_answer": test_case["correct_answer"],
        "model_response": response,
        "url_source": test_case["url_source"],
        "report_id": test_case["report_id"],
        "source": test_case["source"],
        "model": test_case["model"],
    }

    parsed_response = extract_json(query_result)
    if parsed_response:
        parsed_response = cast(Dict[str, str], parsed_response)
        model_answers = cast(List[str], parsed_response.get("correct_answers"))
        sim_score = jaccard_similarity(
            model_answers, cast(List[str], test_case["correct_answer"])
        )
        if sim_score == 1.0:
            judge_results["answered_correctly"] = "true"
        else:
            judge_results["answered_correctly"] = "false"
        judge_results["score"] = cast(
            str,
            sim_score,
        )
    else:
        judge_results["answered_correctly"] = "parsing error"

    return judge_results


def jaccard_similarity(list1: list[str], list2: list[str]) -> float:
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union if union > 0 else 0.0
