# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, Union

import pkg_resources

from .arvo_utils import ArvoContainer

from .autopatch.autopatch_tools import AutoPatchAgent
from .autopatch.report import ChatHistory, write_chat_history_markdown
from .autopatch.types import PatchGenerationReport

from .benchmark import Benchmark, BenchmarkConfig
from .llm import LLM


@dataclass
class AutoPatchConfig:
    """Configuration for the autopatching benchmark."""

    rerun_failed: bool = False
    run_new_cases_only: bool = False
    test_mode: bool = False

    @staticmethod
    def from_args(args: Optional[argparse.Namespace]) -> "AutoPatchConfig":
        """Constructs an AutoPatchConfig from the given arguments."""
        if args:
            return AutoPatchConfig(
                rerun_failed=args.rerun_failed,
                run_new_cases_only=args.run_new_cases_only,
                test_mode=args.test_mode,
            )
        else:
            return AutoPatchConfig()

    @staticmethod
    def from_additional_args(args: Optional[str]) -> "AutoPatchConfig":
        """Constructs an AutoPatchConfig from the given additional arguments."""
        parser = get_parser()
        parsed_args = parser.parse_args((args or "").split())
        return AutoPatchConfig.from_args(parsed_args)


def get_parser() -> argparse.ArgumentParser:
    """Returns the argument parser for the autopatching benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the autopatching benchmark.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun only the QA failed cases.",
    )
    parser.add_argument(
        "--run-new-cases-only",
        action="store_true",
        help="Run only the new cases that response json is missing.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode which skips QA checks and patch generation. For testing the eval script only.",
    )
    return parser


class AutoPatchingBenchmark(Benchmark):
    """
    This class implements the autopatching benchmark.
    """

    PER_SAMPLE_PATCH_GENERATION_TIMEOUT: int = 60 * 60 * 8  # 8 hours

    """Lock to ensure thread-safe access to response file during concurrent operations."""
    RESPONSE_FILE_LOCK: LockType = Lock()
    """Lock to ensure thread-safe access to patch gen report file during concurrent operations."""
    PATCH_REPORT_FILE_LOCK: LockType = Lock()

    """Name identifier for this benchmark type."""
    BENCHMARK_NAME: str = "autopatch"

    # Response JSON keys - used when storing benchmark results
    """Key for the model name in response JSON."""
    RESPONSES_MODEL_KEY: str = "model"
    """Key for the ARVO challenge number in response JSON."""
    RESPONSES_ARVO_CHALLENGE_NUMBER_KEY: str = "arvo_challenge_number"
    """Key indicating whether containers passed quality assurance checks."""
    RESPONSES_CONTAINERS_PASS_QA_CHECKS_KEY: str = "containers_pass_qa_checks"
    """Key for the file path to the generated patch."""
    RESPONSES_GENERATED_PATCH_KEY: str = "generated_patch"
    """Key for the file path to the rebuilt binary."""
    RESPONSES_REBUILT_BINARY_KEY: str = "rebuilt_binary"
    """Key for the name of the function that was patched."""
    RESPONSES_PATCHED_FUNCTION_KEY: str = "patched_function"
    """Patch generator reported success."""
    RESPONSES_PATCH_SUCCESS_KEY: str = "patch_success"
    """Value used when a field is not applicable."""
    RESPONSES_NOT_APPLICABLE: str = "n/a"
    """Key for the exception message, if an exception occurred."""
    RESPONSES_EXCEPTION_MESSAGE_KEY: str = "exception_message"

    # Evaluation result keys - used when analyzing benchmark performance
    """Key indicating whether a patch passed quality assurance checks."""
    EVALUATION_PASSED_QA_CHECKS_KEY: str = "passed_qa_checks"
    """Key indicating whether a patched function name was generated successfully."""
    EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY: str = "patched_function_name_exists"
    """Key indicating whether a patch passed all fuzzing tests."""
    EVALUATION_FULL_FUZZING_PASSED_KEY: str = "full_fuzzing_passed"
    """Key indicating whether original functionality was preserved after patching."""
    EVALUATION_FUNCTIONALITY_PRESERVED_KEY: str = "functionality_preserved"
    """Key indicating whether a fuzzing decode error occurred."""
    EVALUATION_FUZZING_DECODE_ERROR_KEY: str = "fuzzing_decode_error"
    """Key indicating whether a differential debugging error occurred."""
    EVALUATION_DIFFERENTIAL_DEBUGGING_ERROR_KEY: str = "differential_debugging_error"
    """Key indicating whether an error occurred while running debugging."""
    EVALUATION_ERROR_RUNNING_DEBUGGING_KEY: str = "error_running_debugging"
    """Key for the count of patches that passed QA checks."""
    EVALUATION_NUM_PASSED_QA_CHECKS_KEY: str = "num_passed_qa_checks"
    """Key for the count of patch generation that generated a patched function name."""
    EVALUATION_NUM_GENERATED_PATCHED_FUNCTION_NAME_KEY: str = (
        "num_generated_patched_function_name"
    )
    """Key for the count of patches that passed fuzzing tests."""
    EVALUATION_NUM_PASSED_FUZZING_KEY: str = "num_passed_fuzzing"
    """Key for the count of fuzzing decode errors encountered."""
    EVALUATION_NUM_FUZZING_DECODE_ERRORS_KEY: str = "num_fuzzing_decode_errors"
    """Key for the count of differential debugging errors encountered."""
    EVALUATION_NUM_DIFFERENTIAL_DEBUGGING_ERRORS_KEY: str = (
        "num_differential_debugging_errors"
    )
    """Key for the count of patches that passed both fuzzing and differential debugging."""
    EVALUATION_NUM_PASSED_FUZZING_AND_DEBUGGING_KEY: str = (
        "num_passed_fuzzing_and_differential_debugging"
    )
    """Key for the rate of patches generated"""
    EVALUATION_SHR_PATCHES_GENERATED_KEY: str = "shr_patches_generated"
    """Key for the success rate of patches passing fuzzing (ratio of successful to total)."""
    EVALUATION_SHR_PASSING_FUZZING_KEY: str = "shr_passing_fuzzing"
    """Key for the success rate of correct patches (ratio of successful to total)."""
    EVALUATION_SHR_CORRECT_PATCHES_KEY: str = "shr_correct_patches"

    """File path for the rebuilt binary used in evaluation."""
    EVALUATION_REBUILT_BINARY_FILEPATH: str = "/tmp/rebuilt_binary"

    _log_level: int = logging.INFO

    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        super().__init__(config)
        ArvoContainer.set_container_repository(config.container_repository)
        prompt_path = config.prompt_path
        assert prompt_path is not None, "Prompt path must be provided."
        self.images_available: list[int] = ArvoContainer.build_container_images(
            self._get_test_cases(config.num_test_cases, Path(prompt_path)),
            config.max_concurrency,
            AutoPatchingBenchmark.get_log_level(),
        )

    @lru_cache(maxsize=1)
    def get_config(self) -> AutoPatchConfig:
        return AutoPatchConfig.from_additional_args(self.additional_args)

    @classmethod
    def get_log_level(cls) -> int:
        return cls._log_level

    @classmethod
    def set_log_level(cls, level: int) -> None:
        cls._log_level = level

    @classmethod
    def return_kind(cls) -> list[str]:
        return [AutoPatchingBenchmark.BENCHMARK_NAME]

    @staticmethod
    def _get_test_cases(num_test_cases: int, dataset_path: Path) -> list[int]:
        """
        Canonical getter within this benchmark for getting the list of test cases to be evaluated.
        """
        numbers = json.loads(dataset_path.read_text())
        return numbers[:num_test_cases] if num_test_cases > 0 else numbers

    def _get_output_dir(self, case_id: int, model_name: str) -> Path:
        out_dir = self.response_path.parent / "files" / f"case_{case_id}" / model_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _get_log_filepath(self, case_id: int, prefix: str, model: str) -> Path:
        return self._get_output_dir(case_id, model) / f"log_{prefix}.txt"

    async def _gen_patch_and_binary(
        self,
        case_id: int,
        llm_under_test: LLM,
        overwrite: bool = False,
    ) -> None:
        test_mode = self.get_config().test_mode

        # Create a directory for storing patch and binary files, if it doesn't exist already
        out_dir = self._get_output_dir(case_id, llm_under_test.model)
        patch_filepath = out_dir / "patch.patch"
        binary_filepath = out_dir / "binary.bin"
        report_filepath = out_dir / "report.json"
        chat_filepath = out_dir / "chat.md"
        if overwrite:
            for file in out_dir.glob("*"):
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()

        # Spin up the vul container for the autopatch agent
        vul_container = ArvoContainer(
            case_id,
            container_type=ArvoContainer.CONTAINER_TYPE_VUL,
            model_under_test=llm_under_test.model,
            log_level=self.get_log_level(),
            log_filepath=self._get_log_filepath(case_id, "vul", llm_under_test.model),
            artifacts_dir=out_dir,
        )
        await vul_container.start_container()

        containers_pass_qa_checks = True

        # Spin up the fix container for the autopatch agent
        fix_container = ArvoContainer(
            case_id,
            container_type=ArvoContainer.CONTAINER_TYPE_FIX,
            model_under_test=llm_under_test.model,
            log_level=self.get_log_level(),
            log_filepath=self._get_log_filepath(case_id, "fix", llm_under_test.model),
            artifacts_dir=out_dir,
        )
        await fix_container.start_container()

        # QA checks of vul and fix containers
        containers_pass_qa_checks = (await vul_container.qa_checks()) and (
            await fix_container.qa_checks()
        )

        patch_success = False
        patched_function_name = None
        exception_message = None
        if containers_pass_qa_checks:
            # Hand off execution to the agent, instructing it to generate a patch and binary at the given paths
            report = PatchGenerationReport()
            try:
                agent = AutoPatchAgent(
                    vul_container,
                    llm_under_test,
                    patch_filepath,
                    binary_filepath,
                )
                vul_container.logger.info(
                    f"Starting patch generation agent for case #{case_id}"
                )
                if not test_mode:
                    report = await asyncio.wait_for(
                        agent.generate_and_persist_patch_and_binary(),
                        timeout=AutoPatchingBenchmark.PER_SAMPLE_PATCH_GENERATION_TIMEOUT,
                    )
                    patch_success = report.max_patch_generation_status.is_success()
                    patched_function_name = report.patched_function_name
                else:
                    vul_container.logger.warning(
                        "Running in test mode. Skipping patch generation."
                    )

            except asyncio.TimeoutError:
                exception_message = f"Patch generation timed out after {AutoPatchingBenchmark.PER_SAMPLE_PATCH_GENERATION_TIMEOUT} seconds"
                vul_container.logger.error(exception_message)
                report.exception += "\n" + exception_message
            except Exception as e:
                vul_container.logger.error(f"Error during patch generation: {e}")
                exception_message = str(e)
                report.exception += "\n" + exception_message

            # Write reports
            with open(report_filepath, "wt") as f:
                f.write(report.to_json())
            write_chat_history_markdown(
                [ChatHistory.from_report(report)], chat_filepath.as_posix()
            )
            self._write_to_patch_gen_reports_json(
                llm_under_test=llm_under_test, report=report, overwrite=overwrite
            )

        # insert entry containing patch & binary filepaths into response json
        self._write_to_response_json(
            binary_filepath=binary_filepath,
            case_id=case_id,
            containers_pass_qa_checks=containers_pass_qa_checks,
            exception_message=exception_message,
            llm_under_test=llm_under_test,
            overwrite=overwrite,
            patch_filepath=patch_filepath,
            patch_success=patch_success,
            patched_function_name=patched_function_name,
        )

    @staticmethod
    def __extract_diff_output(diff_output: str) -> Dict[str, str]:
        """
        Extracts file paths from git diff output.
        Args:
            diff_output (str): The output of `git diff`.
        Returns:
            tuple: A tuple containing the original and modified file paths.
        """
        # Regular expression pattern to match file paths in git diff output
        pattern = r"diff --git a/(.*) b/(.*)"

        # Search for the pattern in the diff output
        match = re.search(pattern, diff_output)

        if match:
            # Return the original and modified file paths
            file_path = match.group(2)
        else:
            raise RuntimeError(
                f"Could not extract file paths from git diff output {diff_output}"
            )

        # Find the first occurrence of '@@' and get the string slice after it
        diff_start_index = diff_output.find("@@")
        if diff_start_index == -1:
            raise RuntimeError("Could not find '@@' in git diff output")

        diff_content = diff_output[diff_start_index + 2 :].strip()
        return {
            "filename": file_path,
            "patch": diff_content,
        }

    async def _identify_correct_fix(
        self,
        case_id: int,
        llm_under_test: LLM,
        overwrite: bool = False,
    ) -> None:
        RED = "\033[1m\033[91m"
        YELLOW = "\033[1m\033[93m"
        END = "" "\033[0m"
        BLUE = "\033[1m\033[94m"
        metadata_path = pkg_resources.resource_filename(
            __name__, f"arvo_meta/{case_id}-meta.json"
        )
        if not metadata_path.exists():
            logging.warning(
                f"Metadata file not found for case {case_id}: {metadata_path}"
            )
            return

        fix_commits = []
        try:
            # Extract relevant information from metadata
            with open(metadata_path, "r") as f:
                metadata_json = json.load(f)
            metadata = metadata_json.get("arvo_metadata", {})
            if "fix_commit" in metadata:
                fix_commits = metadata["fix_commit"]
                if not isinstance(fix_commits, list):
                    fix_commits = [fix_commits]
            else:
                logging.warning(f"No fix commit found in metadata for case {case_id}")
                return
        except Exception as e:
            logging.error(f"Error processing metadata for case {case_id}: {e}")
            return

        # start fix container
        out_dir = self._get_output_dir(case_id, llm_under_test.model)
        fix_container = ArvoContainer(
            case_id,
            container_type=ArvoContainer.CONTAINER_TYPE_FIX,
            model_under_test=llm_under_test.model,
            log_level=self.get_log_level(),
            log_filepath=self._get_log_filepath(case_id, "fix", llm_under_test.model),
            artifacts_dir=out_dir,
        )
        await fix_container.start_container()
        fix_container.logger.info(f"Starting fix container for case {case_id}")

        # get current hash
        head_commit = await fix_container.get_current_commit_hash()
        # if head_commit not in fix_commits:
        head_timestamp = await fix_container.get_commit_timestamp(head_commit)
        commit_ts_list = [
            (commit, await fix_container.get_commit_timestamp(commit))
            for commit in fix_commits
        ]
        timestamps = "\n".join(
            f"\t{commit} ({timestamp})" for commit, timestamp in commit_ts_list
        )
        fix_container.logger.info(
            f"{YELLOW}HEAD commit {head_commit} ({head_timestamp}) is not one of the GT fix commits\n{timestamps} {END}."
        )
        try:
            found_commit = await fix_container.git_bisect_groundtruth_fix(
                fix_commits[0], head_commit
            )
            if found_commit:
                fix_container.logger.info(
                    f"{BLUE}Found correct fix for {case_id} is {found_commit} {END}"
                )
                patch_path = pkg_resources.resource_filename(
                    __name__, f"arvo_meta/{case_id}-patch.json"
                )
                with open(patch_path, "r") as f:
                    patch_dict = json.load(f)[0]
                metadata_json["arvo_metadata"]["fix_commit"] = found_commit
                try:
                    _patch_dict = self.__extract_diff_output(
                        await fix_container.get_git_diff(found_commit)
                    )
                    patch_dict["filename"] = _patch_dict["filename"]
                    if "patched_line_numbers" in _patch_dict:
                        patch_dict["patched_line_numbers"] = _patch_dict["patch"]
                    patch_dict["patch"] = _patch_dict["patch"]
                    patch_dict["summary"] = await fix_container.get_git_summary(
                        found_commit
                    )
                    with open(patch_path, "w") as f:
                        json.dump([patch_dict], f, indent=2)
                    with open(metadata_path, "w") as f:
                        json.dump(metadata_json, f, indent=2)

                except RuntimeError as e:
                    fix_container.logger.warning(
                        f"{RED}Could not find correct fix for {case_id}: {str(e)} {END}"
                    )
        except RuntimeError as e:
            fix_container.logger.warning(
                f"{RED}Could not find correct fix for {case_id}: {str(e)} {END}"
            )

    def _write_to_patch_gen_reports_json(
        self,
        llm_under_test: LLM,
        report: PatchGenerationReport,
        overwrite: bool = False,
    ) -> None:
        """
        Writes patch generation report information to a JSON file.
        This function is used to track and aggregate patch generation reports across multiple cases.

        Args:
            case_id: The ARVO challenge number
            llm_under_test: The LLM model being evaluated
            report_filepath: Path to the individual patch generation report file
        """
        # Use a multiprocessing lock to ensure thread-safe access to the file
        with self.PATCH_REPORT_FILE_LOCK:
            # Determine the path for the aggregated reports file
            reports_path = self.response_path.parent / "patch_gen_reports.json"
            if not reports_path.exists():
                reports_path.touch()

            # Write patch generation report to the aggregated reports file
            with open(reports_path, "r") as f:
                try:
                    reports = json.load(f)
                except json.JSONDecodeError:
                    reports = []
            if overwrite:
                # find the reports for the given case_id and model, and remove
                reports = [
                    r
                    for r in reports
                    if not (
                        r["case_id"] == report.crash_id
                        and r["model"] == llm_under_test.model
                    )
                ]

            # Create a new report entry
            report_entry = {
                "build_iters": report.build_iters,
                "case_id": report.crash_id,
                "chat_history_length": len(report.chat_history),
                "crash_repro_time": report.crash_repro_time,
                "crash_type": report.crash_type,
                "fix_crash_iters": report.fix_crash_iters,
                "llm_query_cnt": report.llm_query_cnt,
                "llms_query_time": report.llms_query_time,
                "max_patch_generation_status": report.max_patch_generation_status.name,
                "model": llm_under_test.model,
                "patch_gen_time": report.patch_gen_time,
                "patched_file_path": report.patched_file_path,
                "patched_function_name": report.patched_function_name,
                "retry_round": report.retry_round,
                "sanitizer_crash_type": report.sanitizer_crash_type,
                "sanity_check_iters": report.sanity_check_iters,
                "sanity_check_time": report.sanity_check_time,
            }
            reports.append(report_entry)

            # Rewrite the file with updated reports
            reports_path.write_text(json.dumps(reports, indent=4))

        logging.info(
            f"Added patch generation report for case #{report.crash_id} using {llm_under_test.model}"
        )

    def _write_to_response_json(
        self,
        case_id: int,
        llm_under_test: LLM,
        patch_filepath: Path,
        binary_filepath: Path,
        containers_pass_qa_checks: bool,
        patch_success: bool,
        patched_function_name: Optional[str],
        exception_message: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        # Use a multiprocessing lock to ensure that only one process accesses the file at a time
        with self.RESPONSE_FILE_LOCK:
            # Read the existing responses from the file
            if self.response_path.exists():
                responses = json.loads(self.response_path.read_text())
            else:
                responses = []
            if overwrite:
                # find the responses for the given case_id and model, and remove
                responses = [
                    r
                    for r in responses
                    if not (
                        r[AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY]
                        == case_id
                        and r[AutoPatchingBenchmark.RESPONSES_MODEL_KEY]
                        == llm_under_test.model
                    )
                ]
            # Append the new response to the list of responses
            response_entry = {
                AutoPatchingBenchmark.RESPONSES_MODEL_KEY: llm_under_test.model,
                AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY: case_id,
                AutoPatchingBenchmark.RESPONSES_CONTAINERS_PASS_QA_CHECKS_KEY: containers_pass_qa_checks,
                AutoPatchingBenchmark.RESPONSES_PATCH_SUCCESS_KEY: patch_success,
                AutoPatchingBenchmark.RESPONSES_GENERATED_PATCH_KEY: str(
                    patch_filepath
                ),
                AutoPatchingBenchmark.RESPONSES_REBUILT_BINARY_KEY: str(
                    binary_filepath
                ),
                AutoPatchingBenchmark.RESPONSES_PATCHED_FUNCTION_KEY: (
                    patched_function_name
                    if patched_function_name
                    else AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE
                ),
            }
            if exception_message:
                response_entry[
                    AutoPatchingBenchmark.RESPONSES_EXCEPTION_MESSAGE_KEY
                ] = exception_message
            responses.append(response_entry)
            self.response_path.write_text(json.dumps(responses, indent=4))
            logging.info(
                f"Appended response from {llm_under_test.model} for case #{case_id} to {self.response_path}."
            )

    def _read_resonpose_json(self) -> list[dict[str, Any]]:
        if self.response_path.exists():
            return json.loads(self.response_path.read_text())
        else:
            return []

    def _is_patch_success(self, case_id: int, llm_under_test: LLM) -> bool:
        responses = self._read_resonpose_json()
        return any(
            response.get(AutoPatchingBenchmark.RESPONSES_PATCH_SUCCESS_KEY, False)
            or response.get(
                AutoPatchingBenchmark.RESPONSES_PATCHED_FUNCTION_KEY,
                AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE,
            )
            != AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE
            for response in responses
            if response[AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY]
            == case_id
            and response[AutoPatchingBenchmark.RESPONSES_MODEL_KEY]
            == llm_under_test.model
        )

    def _is_result_exists_in_response_json(
        self, case_id: int, llm_under_test: LLM
    ) -> bool:
        responses = self._read_resonpose_json()
        return any(
            response[AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY]
            == case_id
            and response[AutoPatchingBenchmark.RESPONSES_MODEL_KEY]
            == llm_under_test.model
            for response in responses
        )

    # Define a default function for processing patch generation
    async def _default_process_patch_generation(
        self, case_id: int, llm: LLM, overwrite: bool = False
    ) -> None:
        logging.info(f"Generating patch for case #{case_id} using {llm.model}")
        # await self._identify_correct_fix(
        await self._gen_patch_and_binary(
            case_id=case_id, llm_under_test=llm, overwrite=overwrite
        )

    async def _run_patch_generation(
        self,
        test_cases: list[int],
        llms_under_test: list[LLM],
        process_patch_generation: Callable[
            [int, LLM, bool], Coroutine[None, None, None]
        ],
        semaphore: asyncio.Semaphore,
        rerun_failure_cases: bool = False,
        run_new_cases_only: bool = False,
    ) -> None:
        async def sem_task(case_id: int, llm: LLM, overwrite: bool = False) -> None:
            async with semaphore:
                await process_patch_generation(case_id, llm, overwrite)

        if rerun_failure_cases and run_new_cases_only:
            raise RuntimeError(
                "Only one of rerun_failure_cases and run_new_cases_only can be True."
            )

        if rerun_failure_cases:
            tasks = [
                sem_task(case_id=case_id, llm=llm, overwrite=True)
                for llm in llms_under_test
                for case_id in test_cases
                if not self._is_patch_success(case_id, llm)
            ]
            logging.info(f"Rerunning patch generation for {len(tasks)} failed cases.")
        elif run_new_cases_only:
            tasks = [
                sem_task(case_id=case_id, llm=llm, overwrite=False)
                for llm in llms_under_test
                for case_id in test_cases
                if not self._is_result_exists_in_response_json(case_id, llm)
            ]
            logging.info(f"Running patch generation for {len(tasks)} new cases.")
        else:
            tasks = [
                sem_task(case_id, llm)
                for llm in llms_under_test
                for case_id in test_cases
            ]
            logging.info(f"Running patch generation for {len(tasks)} cases.")
        await asyncio.gather(*tasks)

    def query_llm_to_generate_responses(
        self,
        prompt_path: Path,
        run_llm_in_parallel: int = 1,
    ) -> None:
        logging.getLogger().setLevel(self.get_log_level())
        test_cases = self.images_available
        llms_under_test = self.llms_under_test
        semaphore = asyncio.Semaphore(run_llm_in_parallel)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self._run_patch_generation(
                test_cases=test_cases,
                llms_under_test=llms_under_test,
                process_patch_generation=self._default_process_patch_generation,
                semaphore=semaphore,
                rerun_failure_cases=self.get_config().rerun_failed,
                run_new_cases_only=self.get_config().run_new_cases_only,
            )
        )

    async def run(
        self,
        num_test_cases: int = 0,
        run_llm_in_parallel: int = 16,
        should_cleanup_after_eval: bool = False,
    ) -> None:
        """
        Processes an LLM-generated patch by checking whether the generated patch is 'correct'.
        Evaluates patches through fuzzing and differential debugging.
        """
        if self.get_config().test_mode:
            logging.info("Running in test mode. Terminating.")
            return
        # Load previously generated responses
        responses = json.loads(self.response_path.read_text())

        evaluation_results = {}
        evaluation_statistics = {}

        # Extract unique models from responses
        models: set[str] = {
            response[AutoPatchingBenchmark.RESPONSES_MODEL_KEY]
            for response in responses
        }

        # Process each model's responses
        for model in models:
            evaluation_results[model] = {}
            evaluation_statistics[model] = {}

            # Filter responses for this model
            model_responses = [
                response
                for response in responses
                if response[AutoPatchingBenchmark.RESPONSES_MODEL_KEY] == model
            ]

            # Limit test cases if specified
            if num_test_cases > 0:
                model_responses = model_responses[:num_test_cases]

            # Initialize counters for metrics
            num_passed_qa_checks: int = 0
            num_patch_generated: int = 0
            num_passed_fuzzing: int = 0
            num_fuzzing_decode_errors: int = 0
            num_differential_debugging_errors: int = 0
            num_passed_fuzzing_and_differential_debugging: int = 0

            # Process responses in parallel with concurrency limit
            semaphore = asyncio.Semaphore(run_llm_in_parallel)
            tasks = [
                self._process_response(response, model, evaluation_results, semaphore)
                for response in model_responses
            ]

            # Wait for all tasks to complete and gather results
            results = await asyncio.gather(*tasks)

            # Aggregate metrics from results
            for result in results:
                if result[AutoPatchingBenchmark.EVALUATION_PASSED_QA_CHECKS_KEY]:
                    num_passed_qa_checks += 1
                    if result.get(
                        AutoPatchingBenchmark.EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY,
                        False,
                    ):
                        num_patch_generated += 1
                    if result.get(
                        AutoPatchingBenchmark.EVALUATION_FULL_FUZZING_PASSED_KEY, False
                    ):
                        num_passed_fuzzing += 1
                        if result.get(
                            AutoPatchingBenchmark.EVALUATION_FUNCTIONALITY_PRESERVED_KEY,
                            False,
                        ):
                            num_passed_fuzzing_and_differential_debugging += 1
                if result.get(
                    AutoPatchingBenchmark.EVALUATION_FUZZING_DECODE_ERROR_KEY
                ):
                    num_fuzzing_decode_errors += 1
                if result.get(
                    AutoPatchingBenchmark.EVALUATION_DIFFERENTIAL_DEBUGGING_ERROR_KEY
                ):
                    num_differential_debugging_errors += 1

            # Store metrics for this model
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_PASSED_QA_CHECKS_KEY
            ] = num_passed_qa_checks
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_GENERATED_PATCHED_FUNCTION_NAME_KEY
            ] = num_patch_generated
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_PASSED_FUZZING_KEY
            ] = num_passed_fuzzing
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_FUZZING_DECODE_ERRORS_KEY
            ] = num_fuzzing_decode_errors
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_PASSED_FUZZING_AND_DEBUGGING_KEY
            ] = num_passed_fuzzing_and_differential_debugging
            evaluation_statistics[model][
                AutoPatchingBenchmark.EVALUATION_NUM_DIFFERENTIAL_DEBUGGING_ERRORS_KEY
            ] = num_differential_debugging_errors

            # Calculate derived metrics
            if num_passed_qa_checks > 0:
                evaluation_statistics[model][
                    AutoPatchingBenchmark.EVALUATION_SHR_PATCHES_GENERATED_KEY
                ] = f"{num_patch_generated/num_passed_qa_checks:.3f}"
                evaluation_statistics[model][
                    AutoPatchingBenchmark.EVALUATION_SHR_PASSING_FUZZING_KEY
                ] = f"{num_passed_fuzzing/num_passed_qa_checks:.3f}"
                evaluation_statistics[model][
                    AutoPatchingBenchmark.EVALUATION_SHR_CORRECT_PATCHES_KEY
                ] = f"{num_passed_fuzzing_and_differential_debugging/num_passed_qa_checks:.3f}"
            else:
                evaluation_statistics[model][
                    AutoPatchingBenchmark.EVALUATION_SHR_CORRECT_PATCHES_KEY
                ] = AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE
                evaluation_statistics[model][
                    AutoPatchingBenchmark.EVALUATION_SHR_PASSING_FUZZING_KEY
                ] = AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE

        # Write results to files if paths are provided
        if (stat_path := self.stat_path) is not None:
            stat_path.write_text(json.dumps(evaluation_statistics, indent=4))

        if (judge_response_path := self.judge_response_path) is not None:
            judge_response_path.write_text(json.dumps(evaluation_results, indent=4))

        if should_cleanup_after_eval:
            await ArvoContainer.cleanup_all_containers()

    async def _process_response(
        self,
        response: Dict[str, Union[str, int, bool]],
        model: str,
        evaluation_results: Dict[str, Dict[int, Dict[str, Union[bool, str]]]],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Union[bool, str]]:
        """Process a single response with semaphore for concurrency control."""
        async with semaphore:
            result = {}
            arvo_challenge_number = response[
                AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY
            ]
            assert isinstance(
                arvo_challenge_number, int
            ), f"{AutoPatchingBenchmark.RESPONSES_ARVO_CHALLENGE_NUMBER_KEY} should be an int"

            rebuilt_binary = response.get(
                AutoPatchingBenchmark.RESPONSES_REBUILT_BINARY_KEY
            )
            assert isinstance(
                rebuilt_binary, str
            ), f"{AutoPatchingBenchmark.RESPONSES_REBUILT_BINARY_KEY} should be a str"

            patched_function_name = response.get(
                AutoPatchingBenchmark.RESPONSES_PATCHED_FUNCTION_KEY
            )
            assert isinstance(
                patched_function_name, str
            ), f"{AutoPatchingBenchmark.RESPONSES_PATCHED_FUNCTION_KEY} should be a str"

            containers_pass_qa_checks = response.get(
                AutoPatchingBenchmark.RESPONSES_CONTAINERS_PASS_QA_CHECKS_KEY, False
            )
            assert isinstance(
                containers_pass_qa_checks, bool
            ), f"{AutoPatchingBenchmark.RESPONSES_CONTAINERS_PASS_QA_CHECKS_KEY} should be a bool"

            # Initialize result dictionary for this challenge
            evaluation_results[model][arvo_challenge_number] = {}
            result_dict = evaluation_results[model][arvo_challenge_number]

            # Check if containers passed QA checks
            if not containers_pass_qa_checks:
                result_dict[AutoPatchingBenchmark.EVALUATION_PASSED_QA_CHECKS_KEY] = (
                    False
                )
                result[AutoPatchingBenchmark.EVALUATION_PASSED_QA_CHECKS_KEY] = False
                logging.info(
                    f"ARVO #{arvo_challenge_number} failed QA checks. Sample being excluded from set."
                )
                return result

            # QA checks passed
            result_dict[AutoPatchingBenchmark.EVALUATION_PASSED_QA_CHECKS_KEY] = True
            result[AutoPatchingBenchmark.EVALUATION_PASSED_QA_CHECKS_KEY] = True

            if (
                not patched_function_name
                or patched_function_name
                == AutoPatchingBenchmark.RESPONSES_NOT_APPLICABLE
                or not Path(rebuilt_binary).exists()
            ):
                result_dict[
                    AutoPatchingBenchmark.EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY
                ] = False
                result[
                    AutoPatchingBenchmark.EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY
                ] = False
                logging.info(
                    f"ARVO #{arvo_challenge_number} did not specify a patched function"
                    " or the rebuilt binary does not exist. This sample is excluded from set."
                )
                return result

            # Patched function name was generated
            result_dict[
                AutoPatchingBenchmark.EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY
            ] = True
            result[
                AutoPatchingBenchmark.EVALUATION_PATCHED_FUNCTION_NAME_EXISTS_KEY
            ] = True

            # Set up container for testing
            arvo_fix = ArvoContainer(
                arvo_challenge_number,
                container_type=ArvoContainer.CONTAINER_TYPE_FIX,
                model_under_test=model,
                log_level=self.get_log_level(),
                log_filepath=self._get_log_filepath(
                    arvo_challenge_number, "fix", model
                ),
                artifacts_dir=self._get_output_dir(arvo_challenge_number, model),
            )
            await arvo_fix.start_container()
            await arvo_fix.copy_to_container(
                rebuilt_binary, AutoPatchingBenchmark.EVALUATION_REBUILT_BINARY_FILEPATH
            )

            # Run fuzzing tests
            try:
                crash_found = await arvo_fix.does_program_crash_in_full_fuzzing()
                full_fuzzing_passed = not crash_found
                result_dict[
                    AutoPatchingBenchmark.EVALUATION_FULL_FUZZING_PASSED_KEY
                ] = full_fuzzing_passed
                result[AutoPatchingBenchmark.EVALUATION_FULL_FUZZING_PASSED_KEY] = (
                    full_fuzzing_passed
                )
            except Exception as e:
                print(f"FAILED FUZZING: {e} {traceback.format_exc()}")
                result_dict[
                    AutoPatchingBenchmark.EVALUATION_FUZZING_DECODE_ERROR_KEY
                ] = str(e)
                result[AutoPatchingBenchmark.EVALUATION_FUZZING_DECODE_ERROR_KEY] = True
                return result

            # Run differential debugging if fuzzing passed
            if result[AutoPatchingBenchmark.EVALUATION_FULL_FUZZING_PASSED_KEY]:
                try:
                    functionality_preserved = await arvo_fix.run_differential_debugging(
                        patched_function_name
                    )
                    result_dict[
                        AutoPatchingBenchmark.EVALUATION_FUNCTIONALITY_PRESERVED_KEY
                    ] = functionality_preserved
                    result[
                        AutoPatchingBenchmark.EVALUATION_FUNCTIONALITY_PRESERVED_KEY
                    ] = functionality_preserved
                except Exception as e:
                    print(f"FAILED FUZZING: {e} {traceback.format_exc()}")
                    result_dict[
                        AutoPatchingBenchmark.EVALUATION_ERROR_RUNNING_DEBUGGING_KEY
                    ] = str(e)
                    result[
                        AutoPatchingBenchmark.EVALUATION_DIFFERENTIAL_DEBUGGING_ERROR_KEY
                    ] = True

            return result
