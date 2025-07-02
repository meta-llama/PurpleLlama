# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import json
import logging
import pathlib
import re
import subprocess
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from pathlib import Path

from typing import Callable, List, Optional, Tuple, Type

from .llm_interface import forget_middle_of_the_convo, LLMInterface
from .prompts import (
    FOLLOWUP_PATCH_PROMPT,
    GET_SUMMARY_PROMPT,
    GET_TITLE_PROMPT,
    RETRY_BUILD_ERROR_PROMPT,
    RETRY_NOT_FIXED_PROMPT,
    RETRY_TEST_FAILURE_PROMPT,
    ROOTCAUSE_PROMPT,
    SYSTEM_PROMPT,
)
from .source_utils import (
    parse_function_name,
    PathAndLine,
    SourceUtilsInterface,
    Stopwatch,
)
from .tools import PatchVerifierInterface

from .types import (
    Commit,
    CrashReproduceStatus,
    LLMOutputFormatError,
    PatchGenCrashInfo,
    PatchGenerationReport,
    PatchGenerationStatus,
    PatchVerificationResult,
)

LOGGER: Logger = logging.getLogger(__name__)


def get_autopatch_resource(filename: str) -> Path:
    return (
        pathlib.Path(__file__).parent.resolve()
        / "fix_examples"
        / "fix_examples"
        / filename
    )


@dataclass
class PatchGenConfig:
    delegate_roles: bool
    increasing_temperature: bool
    max_iters: int
    max_retries: int
    minimize_context: bool
    # If True, the patch will be generated even if the crash does not reproduce.
    override_non_repro: bool
    shorten_crash_output: bool
    show_fix_example: bool
    # We will show `min(stack_ctx_depth, len(path_and_lines) - 1)` functions in the stack trace to the LLM.
    stack_ctx_depth: int
    # simulation mode is used for e2e testing
    is_simulation: bool

    @staticmethod
    def from_args(args: argparse.Namespace) -> "PatchGenConfig":
        return PatchGenConfig(
            delegate_roles=args.pg_delegate_roles,
            increasing_temperature=args.pg_increasing_temperature,
            max_iters=args.pg_max_iters,
            max_retries=args.pg_max_retries,
            minimize_context=args.pg_minimize_context,
            override_non_repro=args.override_non_repro,
            shorten_crash_output=args.pg_shorten_crash_output,
            show_fix_example=args.pg_show_fix_example,
            stack_ctx_depth=args.pg_stack_ctx_depth,
            is_simulation=args.crashid == 0,
        )

    @staticmethod
    def create_example() -> "PatchGenConfig":
        return PatchGenConfig(
            delegate_roles=False,
            increasing_temperature=False,
            max_iters=4,
            max_retries=1,
            minimize_context=False,
            override_non_repro=False,
            shorten_crash_output=True,
            show_fix_example=False,
            stack_ctx_depth=1,
            is_simulation=False,
        )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not (1 <= self.max_iters <= 100):
            raise ValueError("max_iters must be between 1 and 100")
        if not (0 <= self.max_retries <= 100):
            raise ValueError("max_retries must be between 0 and 100")
        if not (1 <= self.stack_ctx_depth <= 15):
            raise ValueError("stack_ctx_depth must be between 1 and 15")


def extract_code(answer: str, marker: str = "```") -> str:
    if answer.count(marker) % 2 != 0:
        raise LLMOutputFormatError(
            f"Unexpected number of {marker} in the answer \n{answer}"
        )
    # Extract all the content wrapped between all two marks:
    re_pattern = re.escape(marker) + r"([\s\S]*?)" + re.escape(marker)
    # A heuristic to remove the extra letters that sometimes LLMs include.
    modified_answer = answer.replace("```cpp\n", "```\n")
    modified_answer = modified_answer.replace("```c\n", "```\n")
    results = re.findall(re_pattern, modified_answer)
    result_list = [each.strip() for each in results]
    if len(result_list) == 0:
        raise LLMOutputFormatError(f"Failed to extract code from response {answer}")
    # With step by step reasoning, LLM may return multiple code blocks. Only
    # consider the last one as a valid answer since the former ones are
    # intermediate steps.
    return result_list[-1]


@dataclass(frozen=True)
class FunctionSourceInfoItem:
    """Describes that `function_src` is inside the file `source_path`."""

    source_path: str
    function_src: str


@dataclass(frozen=True)
class FunctionSourceInfo:
    items: List[FunctionSourceInfoItem]

    def get_function_src_by_path(self, source_path: str) -> Optional[str]:
        for item in self.items:
            if item.source_path == source_path:
                return item.function_src
        return None


@dataclass(frozen=True)
class PatchPlanItem:
    """Describes that we need to patch `original_function_src` in `source_path` with `new_function_src`."""

    source_path: str
    original_function_src: str
    new_function_src: str


@dataclass(frozen=True)
class PatchPlan:
    # Describe what to patch
    items: List[PatchPlanItem]
    # dummy plan simply adds a line of comment to the file
    is_dummy: bool = False


class PatcherInterface(ABC):
    """
    Interface for a patcher.
    It offers better testability as we can use a fake patcher in the tests.
    """

    def __init__(
        self, source_utils: SourceUtilsInterface, logger: Optional[Logger] = None
    ) -> None:
        self._source_utils = source_utils
        self._logger: Logger = logger or LOGGER

    @abstractmethod
    async def patch_file(
        self, filename: str, original_text: str, new_text: str
    ) -> None:
        """
        Patches the file by replacing the original text with the new text.
        """


class Patcher(PatcherInterface):
    def __init__(
        self, source_utils: SourceUtilsInterface, logger: Optional[Logger] = None
    ) -> None:
        super().__init__(source_utils, logger)

    async def patch_file(
        self, filename: str, original_text: str, new_text: str
    ) -> None:
        self._logger.info(f"Patching {filename}...")
        filename_path = self._source_utils.get_absolute_path(filename)
        with open(filename_path, "r") as fp:
            content = fp.read()
        if content.find(original_text) == -1:
            raise RuntimeError(
                f"Failed to patch file `{filename}`. "
                "The original text is not found in the file content."
            )
        content = content.replace(original_text, new_text)
        with open(filename_path, "w") as fp:
            fp.write(content)
            self._logger.info(f"Successfully patched {filename}.")


class PatchGenerator:
    def __init__(
        self,
        llm_class: Type[LLMInterface],
        source_utils: SourceUtilsInterface,
        patch_verifier: PatchVerifierInterface,
        patcher: PatcherInterface,
        config: PatchGenConfig,
        new_report: Callable[
            [Optional[PatchGenCrashInfo], Optional[PatchGenerationReport]],
            PatchGenerationReport,
        ],
        get_test_plan: Callable[[int], str] = lambda x: "Test plan for crash id: "
        + str(x),
        # pyre-ignore[24]
        logger: Optional[LoggerAdapter] = None,
    ) -> None:
        self._llm_class: Type[LLMInterface] = llm_class
        self._source_utils = source_utils
        self._patch_verifier = patch_verifier
        self._patcher = patcher
        self._config = config
        self._get_commit_test_plan = get_test_plan
        self._get_new_report = new_report
        self._report: PatchGenerationReport = PatchGenerationReport()
        # pyre-ignore[24]
        self._logger: LoggerAdapter = logger or LoggerAdapter(LOGGER, {})

    async def _query_llm(
        self,
        llm: LLMInterface,
        prompt: str,
        extract_output: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """Query LLM. May throw LLMOutputFormatError."""
        self._report.chat_history.append(prompt)
        with Stopwatch() as sw:
            answer = await llm.query(prompt, temperature=temperature)
        self._report.llms_query_time += sw.elapsed_seconds()
        self._report.llm_query_cnt += 1
        self._report.chat_history.append(answer)
        if extract_output:
            # may throw
            answer = extract_code(answer)
        return answer

    async def _fetch_function_source_info(
        self, crash_info: PatchGenCrashInfo
    ) -> Optional[FunctionSourceInfo]:
        path_and_lines = PathAndLine.from_crash_output(crash_info.output)
        # For ST3 only: choose the first stack trace and the first function involved in the stack trace.
        if len(path_and_lines) == 0 or len(path_and_lines[0]) == 0:
            # An example crash where len(path_and_lines[0]) == 0: 612321827082112
            self._logger.warning(
                "The stack trace is empty or no function is found in the first stack trace."
            )
            return None

        num_func_to_show = min(self._config.stack_ctx_depth, len(path_and_lines[0]) - 1)
        self._logger.info(f"num_func_to_show: {num_func_to_show}")

        func_src_info_items: List[FunctionSourceInfoItem] = []

        for path_and_line in path_and_lines[0]:
            # Get the source code of the function encompassing the stacktrace item path.
            try:
                func_src = await self._source_utils.retrieve_src_of_func(path_and_line)
                if not func_src:
                    raise RuntimeError()
            except RuntimeError:
                self._logger.warning(
                    f"Unable to retrieve the source code of the function encompassing the stacktrace item path {path_and_line}."
                )
                continue

            func_src_info_items.append(
                FunctionSourceInfoItem(path_and_line.file_path, func_src)
            )
            if len(func_src_info_items) >= num_func_to_show:
                break

        if func_src_info_items:
            self._report.set_new_status(PatchGenerationStatus.FETCH_SOURCE_SUCCESSFUL)

            return FunctionSourceInfo(func_src_info_items)
        else:
            return None

    def get_fix_example(self, crash_type: str) -> str:
        """Returns a fix example for the given crash type. Returns an empty string if no fix example is found."""
        with open(get_autopatch_resource("fix_examples_index.json")) as f:
            fix_examples_index = json.load(f)

        if crash_type in fix_examples_index:
            example_txt_files = fix_examples_index[crash_type]
            # Currently we choose the first example
            example_txt_file = example_txt_files[0]
            self._logger.info(
                f"Using example_txt_file: {example_txt_file} for crash type {crash_type}"
            )
            with open(get_autopatch_resource(example_txt_file)) as f:
                example_txt = f.read()
            return example_txt
        else:
            self._logger.info(f"No fix example found for crash type {crash_type}.")
            return ""

    def match_rewritten_function(
        self, function_source_info: FunctionSourceInfo, new_function_src: str
    ) -> Optional[Tuple[str, str, Optional[str]]]:
        original_function_src = None
        if len(function_source_info.items) == 1:
            source_path = function_source_info.items[0].source_path
            original_function_src = function_source_info.items[0].function_src
        else:
            new_function_name = parse_function_name(new_function_src, self._logger)
            if new_function_name is None:
                self._logger.warning(
                    f"Unable to parse the function name from the rewritten function:\n{new_function_src}"
                )
                return None
            source_path = None
            for each in function_source_info.items:
                func_name = parse_function_name(each.function_src, self._logger)
                if not func_name:
                    continue
                if func_name == new_function_name:
                    source_path = each.source_path
                    original_function_src = each.function_src
                    self._logger.info(
                        f"LLM answer matches the function {new_function_name} in {each.source_path}"
                    )
                    break
            if not source_path:
                self._logger.warning(
                    f"Unable to locate the source path for the function in the LLM answer: {new_function_name}"
                )
                return None
        assert original_function_src is not None
        return (
            source_path,
            original_function_src,
            parse_function_name(original_function_src, self._logger),
        )

    async def _get_patch_plan(
        self,
        llm: LLMInterface,
        crash_info: PatchGenCrashInfo,
        temperature: Optional[float] = None,
    ) -> Optional[PatchPlan]:
        """
        Tries to ask the LLM to generate a patch plan based on the given PatchGenCrashInfo.

        Raises: RuntimeError if the LLM fails to generate a valid patch plan.
        """
        if crash_info.crash_id == 0:
            # For e2e testing
            self._report.set_new_status(
                PatchGenerationStatus.PATCH_FORMAT_CORRECT, check=False
            )
            return PatchPlan(items=[PatchPlanItem("", "", "")], is_dummy=True)

        self._report.set_new_status(PatchGenerationStatus.INIT_STATUS, check=False)

        function_source_info = await self._fetch_function_source_info(crash_info)
        if function_source_info is None:
            # patch generator requires some soure code info to be present in the
            # crash output to work.
            self._report.set_new_status(
                PatchGenerationStatus.NOT_SUPPORTED, check=False
            )
            raise RuntimeError(
                f"NOT_SUPPORTED: Unable to fetch the function source info for the crash output:\n{crash_info.output}"
            )

        stack_trace = crash_info.get_output(
            shorten_depth=10 if self._config.shorten_crash_output else 99
        )
        fix_example = (
            self.get_fix_example(self._report.crash_type)
            if self._config.show_fix_example
            else None
        )
        # We send the source code, and ask the LLM to rewrite the function.
        function_src = "\n\n".join(
            [each.function_src for each in function_source_info.items]
        )
        prompt1 = ROOTCAUSE_PROMPT.format(
            stack_trace=stack_trace,
            function_src=function_src,
        )
        _ = await self._query_llm(llm, prompt1, temperature=temperature)
        prompt2 = FOLLOWUP_PATCH_PROMPT.format(
            fix_example=fix_example,
        )
        answer = await self._query_llm(llm, prompt2, temperature=temperature)
        if not answer:
            self._logger.warning("The LLM returned an empty answer.")
            return None

        # Extract the rewritten function from the answer
        try:
            new_function_src = extract_code(answer)
        except LLMOutputFormatError:
            self._logger.warning("Failed to parse the LLM answer.")
            return None

        # Next, we need to know which function has been rewritten
        res = self.match_rewritten_function(function_source_info, new_function_src)
        if not res:
            self._logger.warn("Failed to match the rewritten function.")
            return None

        source_path, original_function_src, patched_function_name = res

        self._report.set_new_status(
            PatchGenerationStatus.PATCH_FORMAT_CORRECT,
        )
        self._report.patched_function_name = patched_function_name
        self._report.patched_file_path = source_path

        return PatchPlan(
            [
                PatchPlanItem(
                    source_path=source_path,
                    original_function_src=original_function_src,
                    new_function_src=new_function_src,
                )
            ]
        )

    async def _apply_patch(self, patch_plan: PatchPlan) -> None:
        if patch_plan.is_dummy:
            return

        for item in patch_plan.items:
            await self._patcher.patch_file(
                item.source_path, item.original_function_src, item.new_function_src
            )

    async def _verify_patch(self, crash_id: int) -> PatchVerificationResult:
        result = PatchVerificationResult()

        with Stopwatch() as sw:
            result.crash_repro_result = await self._patch_verifier.reproduce_crash(
                crash_id
            )
        self._report.crash_repro_time += sw.elapsed_seconds()

        self._logger.info(f"Crash reproduce result: {result.crash_repro_result}")

        self._report.build_iters += 1
        if result.crash_repro_result.status & CrashReproduceStatus.BUILD_SUCCESSFUL:
            self._report.set_new_status(PatchGenerationStatus.PATCH_BUILD_SUCCESSFUL)

        self._report.fix_crash_iters += 1
        if not (result.crash_repro_result.status & CrashReproduceStatus.NO_REPRODUCE):
            return result

        self._report.set_new_status(PatchGenerationStatus.PATCH_FIXES_CRASH)

        self._report.sanity_check_iters += 1
        with Stopwatch() as sw:
            result.sanity_check_result = self._patch_verifier.run_sanity_checks(
                is_simulation=crash_id == 0
            )
        self._report.sanity_check_time += sw.elapsed_seconds()

        if not result.sanity_check_result.passed:
            self._logger.info("The patch doesn't pass the sanity checks.")
            return result

        self._logger.info("The patch passes the sanity checks.")
        self._report.set_new_status(PatchGenerationStatus.PATCH_PASSES_CHECKS)
        return result

    async def _gen_commit_title(
        self, llm: LLMInterface, fallback_title: str = "Bug fix"
    ) -> str:
        MAX_COMMIT_TITLE_LENGTH = 60
        try:
            return extract_code(
                await self._query_llm(
                    llm,
                    GET_TITLE_PROMPT.format(
                        MAX_COMMIT_TITLE_LENGTH=MAX_COMMIT_TITLE_LENGTH
                    ),
                )
            )
        except LLMOutputFormatError:
            self._logger.warning("The LLM did not provide a valid commit title.")
            return fallback_title

    async def _gen_commit_summary(
        self,
        llm: LLMInterface,
        fallback_summary: str = "No summary available.",
    ) -> str:
        try:
            commit_summary = await self._query_llm(llm, GET_SUMMARY_PROMPT)
        except LLMOutputFormatError:
            self._logger.warning("The LLM did not provide a valid commit summary.")
            return fallback_summary

        return commit_summary

    def get_report(self) -> PatchGenerationReport:
        return self._report

    async def generate_patch(
        self,
        crash_info: PatchGenCrashInfo,
        output_patch: Optional[Path] = None,
        generate_commit: bool = True,
    ) -> PatchGenerationReport:
        """
        Generates a patch for the given crash instance.

        Args:
            - crash_info: Specifies the crash instance to generate a patch for.
            - output_patch: If specified, a successfully generated patch will
            also be saved to this file as a git commit file. Defaults to None.

        Returns:
            A PatchGenerationReport containing the results of the patch
            generation. When max_retries > 0, the report of the last retry will
            be returned.
        """
        self._report = self._get_new_report(crash_info, None)

        repro_result = await self._patch_verifier.reproduce_crash(crash_info.crash_id)
        if not (self._config.override_non_repro or repro_result.is_reproduced()):
            raise RuntimeError(
                f"Aborting: the crash does not reproduce: {repro_result.logs}"
            )

        for retry_round in range(self._config.max_retries):
            # Get a new report for each patch generation
            self._report: PatchGenerationReport = self._get_new_report(
                crash_info, self.get_report()
            )
            self._logger.info(
                f"Retry round: {retry_round+1} / {self._config.max_retries}"
            )

            with Stopwatch() as sw:
                try:
                    await self._generate_patch_impl(
                        crash_info,
                        output_patch,
                        self._config.max_iters,
                        generate_commit,
                    )
                except Exception as e:
                    self._report.exception = str(e)
                    self._logger.warning(
                        f"An exception occurred when generating patch for CrashID {crash_info.crash_id}: {e}\n\n{traceback.format_exc()}"
                    )
                    self._report.set_new_status(
                        PatchGenerationStatus.FAILED, check=False
                    )

                self._report.patch_gen_time += sw.elapsed_seconds()

            if self._report.patch_generation_status in (
                PatchGenerationStatus.PATCH_PASSES_CHECKS,
                PatchGenerationStatus.NOT_SUPPORTED,
            ):
                # PATCH_PASSES_CHECKS: We have a patch that fixes the crash.
                # NOT_SUPPORTED: The tool cannot fix this crash
                break
            else:
                await self._source_utils.revert_current_changes(hard=True)

        return self.get_report()

    async def _gen_another_patch_plan(
        self,
        llm: LLMInterface,
        old_patch_plan: PatchPlan,
        verification_result: PatchVerificationResult,
        temperature: Optional[float] = None,
    ) -> Optional[PatchPlan]:
        """Ask LLM to generate another patch plan when the old_patch_plan did
        not fix the crash.

        Raises: RuntimeError if the LLM fails to generate a valid patch plan.
        """
        # We reset the status before generating another patch plan.
        self._report.set_new_status(
            PatchGenerationStatus.FETCH_SOURCE_SUCCESSFUL, check=False
        )

        if len(old_patch_plan.items) != 1:
            raise RuntimeError(
                "We only support one and only one item in the old patch plan for now."
                f"The unexpected old patch plan is: {old_patch_plan}"
            )

        prompt = self._get_try_again_prompt(old_patch_plan, verification_result)
        response = await self._query_llm(llm, prompt, temperature=temperature)
        try:
            new_function_src = extract_code(response)
        except LLMOutputFormatError:
            return None

        self._report.set_new_status(
            PatchGenerationStatus.PATCH_FORMAT_CORRECT,
        )
        old_patch_plan_item = old_patch_plan.items[0]

        return PatchPlan(
            [
                PatchPlanItem(
                    source_path=old_patch_plan_item.source_path,
                    original_function_src=old_patch_plan_item.original_function_src,
                    new_function_src=new_function_src,
                )
            ]
        )

    async def save_patch(
        self,
        crash_info: PatchGenCrashInfo,
        llm: LLMInterface,
        output_patch: Optional[Path],
        is_simulation: bool = False,
        generate_commit: bool = True,
    ) -> None:
        """Saves the generated patch either as a file or as a commit.

        Args:
            crash_info: Specifies the crash instance to generate a patch for.
            llm: The LLM interface to use for generating the patch.
            output_patch: If specified, a successfully generated patch will also
                be saved to this file as a git commit file.  Defaults to None.
            is_simulation: If True, the patch will be saved as a test commit.
            Defaults to False.
        """

        if output_patch is not None:
            await self._source_utils.save_current_changes_to_file(
                output_patch, self._report.patched_file_path
            )

        if generate_commit:
            commit_summary = await self._gen_commit_summary(llm)
            commit_title = (
                "TEST COMMIT " if is_simulation else ""
            ) + await self._gen_commit_title(llm)

            commit_test_plan = (
                self._get_commit_test_plan(crash_info.crash_id)
                if crash_info.crash_id
                else "Test plan is not available."
            )

            self._report.commit = Commit(
                id="",
                title=f"[llm_patch] {commit_title}",
                summary=commit_summary,
                test_plan=commit_test_plan,
            )

    def _get_try_again_prompt(
        self,
        old_patch_plan: PatchPlan,
        verification_result: PatchVerificationResult,
    ) -> str:
        """Returns a feedback prompt to append to the previous one for the next
        patch generation iteration."""
        if (
            verification_result.crash_repro_result.status
            == CrashReproduceStatus.BUILD_FAILED
        ):
            build_errors = self._patch_verifier.parse_crash_reproduce_build_errors(
                verification_result.crash_repro_result.logs
            )
            build_errors_str = (
                f"Here are the build errors:\n```\n{build_errors.strip()}\n```\n"
                if build_errors
                else ""
            )
            return RETRY_BUILD_ERROR_PROMPT.format(
                build_errors_str=build_errors_str
            ).strip()
        elif (
            verification_result.crash_repro_result.status
            == CrashReproduceStatus.BUILD_SUCCESSFUL
        ):
            # Maybe we can include the reproduced crash stack trace here in the future.
            return RETRY_NOT_FIXED_PROMPT.strip()
        elif not verification_result.sanity_check_result.passed:
            err_log = verification_result.sanity_check_result.logs.strip()
            test_failure_str = (
                f"Test failed with the following error:\n```\n{err_log}\n```\n"
                if err_log
                else ""
            )
            return RETRY_TEST_FAILURE_PROMPT.format(
                test_failure_str=test_failure_str
            ).strip()
        else:
            raise RuntimeError("Unexpected verification result.")

    async def _generate_verified_patch_plan(
        self, llm: LLMInterface, crash_info: PatchGenCrashInfo, max_iterations: int
    ) -> Optional[PatchPlan]:
        temperature = None
        patch_plan: Optional[PatchPlan] = None
        verification_result: Optional[PatchVerificationResult] = None
        for i in range(max_iterations):
            self._logger.info(f"Iteration {i + 1} of {max_iterations}.")
            try:
                if patch_plan is None:
                    patch_plan = await self._get_patch_plan(
                        llm, crash_info, temperature
                    )
                else:
                    assert verification_result is not None

                    if self._config.minimize_context:
                        forget_middle_of_the_convo(llm)

                    if self._config.increasing_temperature:
                        temperature = min(0.05 * (i), 0.8)
                    patch_plan = await self._gen_another_patch_plan(
                        llm, patch_plan, verification_result, temperature
                    )
            except Exception as e:
                self._logger.warning(
                    f"Failed the {i+1}/{max_iterations} patch trajectory: {e}\n\n{traceback.format_exc()}."
                )
                continue
            if not patch_plan:
                continue

            await self._apply_patch(patch_plan)
            verification_result = await self._verify_patch(crash_info.crash_id)
            if verification_result.sanity_check_result.passed:
                self._patch_verifier.run_lint()
                return patch_plan

            # revert and continue
            self._logger.info(
                f"Attempted patch ({i+1}/{max_iterations}) did not fix the crash and pass the tests."
            )
            try:
                await self._source_utils.revert_current_changes()
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to revert changes.")

        self._logger.info(
            f"Failed to generate a successful patch in {max_iterations} iterations."
        )

    async def _generate_patch_impl(
        self,
        crash_info: PatchGenCrashInfo,
        output_patch: Optional[Path],
        max_iterations: int,
        generate_commit: bool = True,
    ) -> None:
        self._logger.info(f"Generating patch for crash {crash_info.crash_id}.")

        llm = self._llm_class(SYSTEM_PROMPT)
        if not await self._generate_verified_patch_plan(
            llm, crash_info, max_iterations
        ):
            self._logger.info("Failed to generate patch plan successfully.")
            return

        # Note: The test checks whether this message exists in the log. Check the test before modifying the message.
        await self.save_patch(
            crash_info=crash_info,
            is_simulation=crash_info.crash_id == 0,
            llm=llm,
            output_patch=output_patch,
            generate_commit=generate_commit,
        )
        self._logger.info("A verified patch plan has been generated successfully.")
