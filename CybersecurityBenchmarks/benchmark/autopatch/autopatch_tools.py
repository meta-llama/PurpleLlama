# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import asyncio
import functools
import logging

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Type

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..arvo_utils import ArvoContainer, BuildStatus

from ..llm import LLM

from .crash_native import NativeCrash
from .llm_interface import LLMInterface

from .patch_generator import PatcherInterface, PatchGenConfig, PatchGenerator
from .source_utils import SourceUtilsInterface, TreeSitterSourceUtils
from .tools import PatchVerifierInterface
from .types import (
    CrashReproduceResult,
    CrashReproduceStatus,
    PatchGenCrashInfo,
    PatchGenerationReport,
    SanityCheckResult,
)


def _remove_ansi_escape_codes(text: str) -> str:
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def _get_new_report(
    crash_info: Optional[PatchGenCrashInfo],
    previous_report: Optional[PatchGenerationReport],
) -> PatchGenerationReport:
    report = PatchGenerationReport()
    if crash_info:
        report.crash_id = crash_info.crash_id
        report.crash_type = crash_info.crash_type
        report.sanitizer_crash_type = crash_info.sanitizer_crash_type
    if previous_report:
        report.max_patch_generation_status = previous_report.max_patch_generation_status
        report.retry_round = previous_report.retry_round + 1
    return report


class AutoPatchAgent:
    """
    This class implements the autopatch agent.
    """

    def __init__(
        self,
        container: ArvoContainer,
        llm_under_test: LLM,
        patch_filepath: Path,
        binary_filepath: Path,
    ) -> None:
        self.container: ArvoContainer = container
        self.llm_under_test: LLM = llm_under_test
        self.patch_filepath: Path = patch_filepath
        self.binary_filepath: Path = binary_filepath
        self._patched_function_name: Optional[str] = None

    async def generate_and_persist_patch_and_binary(self) -> PatchGenerationReport:
        """
        Generates a patch for the given case.
        """
        autopatch_source_utils = get_autopatch_source_utils(self.container)
        pg = PatchGenerator(
            llm_class=get_autopatch_llm(self.llm_under_test),
            source_utils=autopatch_source_utils,
            patch_verifier=AutoPatchVerifier(self.container),
            patcher=AutoPatcher(self.container, autopatch_source_utils),
            # TODO: Make these configurable
            config=PatchGenConfig(
                # Whether to delegate a separate role to the LLM for fixing the build errors
                delegate_roles=False,
                # Whether to apply increasing temperature of the LLM when retrying the patch gen to encourage it to generate more diverse patches
                increasing_temperature=True,
                # The maximum number of iterations on each fix trajectory
                max_iters=5,
                # The maximum number of fix trajectories to attempt before giving up
                max_retries=10,
                # Whether to minimize the context (i.e., conversation history) before sending it to the LLM
                minimize_context=False,
                # Generate patch even when crash does not reproduce
                override_non_repro=False,
                # Whether to shorten the crash output before sending it to the LLM
                shorten_crash_output=True,
                # Whether to show a fixing example in the prompt for specific crash_types
                show_fix_example=False,  # TODO: Create some open-source examples
                # Specifies the number of functions in the stack trace to show to the LLM for patch generation
                stack_ctx_depth=5,
                # Whether to run the patch generation in simulation mode
                is_simulation=False,
            ),
            new_report=_get_new_report,
            get_test_plan=lambda crash_id: "",
            logger=self.container.logger,
        )

        # Run fuzzing to get the crash output
        fuzzing_result = await self.container.exec_command(
            cmd_args=["arvo"], combine_outputs=True
        )
        crash_output = _remove_ansi_escape_codes(fuzzing_result.stdout.decode())
        crash = NativeCrash(crash_output)
        crash_info = PatchGenCrashInfo(
            crash_id=self.container.arvo_id,
            output=crash_output,
            crash_commit="N/A",  # Not relevant
            crash_type=crash.crash_type.value,
            sanitizer_crash_type=PatchGenCrashInfo.get_sanitizer_crash_type(
                crash_output
            ),
        )

        # The actual magic happens here!
        # This executes an agent that generates and applies patches to resolve the crash,
        # and saves the successful patch to self.patch_filepath.
        report = await pg.generate_patch(
            crash_info=crash_info,
            output_patch=self.patch_filepath,
            generate_commit=False,
        )
        if not report.patch_generation_status.is_success():
            logging.info(
                f"Agent failed to generate a patch for ARVO ID {self.container.arvo_id} using {self.llm_under_test.model}"
            )
            return report

        logging.info(
            f"Agent successfully generated a patch for ARVO ID {self.container.arvo_id} using {self.llm_under_test.model}"
        )
        self._patched_function_name = report.patched_function_name

        # Job's done! Copy the patched binary to self.binary_filepath
        patched_binary_filepath = await self.container.get_binary_filepath()
        await self.container.copy_from_container(
            patched_binary_filepath, str(self.binary_filepath)
        )

        return report

    def get_patched_function_name(self) -> Optional[str]:
        """
        Returns the name of the patched function.
        """
        return self._patched_function_name


def get_autopatch_llm(llm_under_test: LLM) -> Type[LLMInterface]:
    class AutoPatchLLM(LLMInterface):
        # With a backoff rate of 2, this allows up to about an hour of retries before escalating the exception.
        MAX_RETRIES: int = 11
        INITIAL_RETRY_DELAY: int = 1  # in seconds
        RETRY_BACKOFF_RATE: int = 2

        def __init__(self, system_prompt: str) -> None:
            self._model: str = llm_under_test.model
            self._messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        async def query(self, message: str, temperature: Optional[float] = None) -> str:
            system_message, history = self._get_message_history_in_str()
            new_message = HumanMessage(content=message)
            delay = AutoPatchLLM.INITIAL_RETRY_DELAY
            for attempt in range(AutoPatchLLM.MAX_RETRIES):
                try:
                    # Run the LLM query in a separate thread to avoid nested event loops
                    response = await asyncio.get_running_loop().run_in_executor(
                        None,
                        llm_under_test.chat_with_system_prompt,
                        system_message,
                        # Defer appending the new message to the context until we have a response from the LLM.
                        history + [new_message.content],
                    )
                    response = self._strip_reasoning(response or "")
                    if not response or not response.strip():
                        # retry if the response is empty
                        raise ValueError("The LLM returned an empty response.")
                    self.set_context(
                        self.get_context() + [new_message, AIMessage(content=response)]
                    )
                    # ensure string type
                    return response or ""
                except Exception as e:
                    logging.warning(f"Exception raised: {e}")
                    if attempt == AutoPatchLLM.MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"LLM query failed after {AutoPatchLLM.MAX_RETRIES} retries."
                        )
                    await asyncio.sleep(delay)
                    delay *= AutoPatchLLM.RETRY_BACKOFF_RATE

        def _get_message_history_in_str(self) -> Tuple[str, List[str]]:
            return self.get_context()[0].content, [
                message.content for message in self.get_context()[1::]
            ]

    return AutoPatchLLM


def get_autopatch_source_utils(container: ArvoContainer) -> SourceUtilsInterface:
    class AutoPatchSourceUtils(TreeSitterSourceUtils):
        async def does_path_exist(self, path: Path) -> bool:
            return (
                await container.exec_command(cmd_args=["test", "-e", str(path)])
            ).returncode == 0

        @staticmethod
        async def get_file_content(path: Path) -> bytes:
            return await container.get_content(str(path))

        @staticmethod
        @functools.lru_cache()
        def _get_repo_root() -> str:
            return (
                container.exec_command_blocking(cmd_args=["pwd"])
                .stdout.decode()
                .strip()
            )

        @staticmethod
        async def save_current_changes_to_file(
            output_patch: Path, patched_src_path: Optional[str] = None
        ) -> None:
            args = ["git", "diff"]
            if patched_src_path:
                args.extend(["--", patched_src_path])
            diff_content = _remove_ansi_escape_codes(
                (await container.exec_command(cmd_args=args)).stdout.decode("utf-8")
            )
            if not diff_content.strip():
                raise RuntimeError("No changes to save")

            output_patch.write_text(diff_content)
            container.logger.debug(f"Saved current changes to {output_patch}")

        @staticmethod
        async def revert_current_changes(hard: bool = False) -> None:
            """
            Reverts the current changes.
            """
            if hard:
                await container.restart_container()
            else:
                await container.exec_command(cmd_args=["git", "checkout", "--", "."])

            container.logger.debug(
                f"Reverted current changes {'(hard)' if hard else ''}"
            )

    return AutoPatchSourceUtils()


class AutoPatchVerifier(PatchVerifierInterface):
    def __init__(self, container: ArvoContainer) -> None:
        self.container: ArvoContainer = container

    async def reproduce_crash(self, crash_id: int) -> CrashReproduceResult:
        """
        Tries to reproduce the crash on the current revision.

        Returns a CrashReproduceResult.
        """
        crash_reproduction_status = CrashReproduceStatus.BUILD_FAILED
        logs = ""

        if self.container.build_status == BuildStatus.UNBUILT:
            # Attempt to recompile the repo
            recompilation_result = await self.container.recompile(combine_outputs=True)
            if recompilation_result.returncode != 0:
                logs = recompilation_result.stdout.decode()
                self.container.logger.debug(f"Build failed with error: {logs}")
                self.container.build_status = BuildStatus.FAILED
                return CrashReproduceResult(crash_reproduction_status, logs)
            else:
                self.container.build_status = BuildStatus.SUCCESSFUL
                crash_reproduction_status |= CrashReproduceStatus.BUILD_SUCCESSFUL

        # Re-run fuzzing with the given POC if build was successful
        if self.container.build_status == BuildStatus.SUCCESSFUL:
            fuzzing_result = await self.container.exec_command(
                cmd_args=["arvo"], combine_outputs=True
            )
            if fuzzing_result.returncode == 0:
                crash_reproduction_status |= CrashReproduceStatus.NO_REPRODUCE
                return CrashReproduceResult(crash_reproduction_status, logs)
            logs = _remove_ansi_escape_codes(fuzzing_result.stdout.decode())

        # TODO: Potentially re-run fuzzing with variants of given POC
        return CrashReproduceResult(crash_reproduction_status, logs)

    def run_sanity_checks(self, is_simulation: bool = False) -> SanityCheckResult:
        return SanityCheckResult(passed=True, logs="")

    @staticmethod
    def parse_crash_reproduce_build_errors(stderr_output: str) -> str:
        """
        Parse build errors from stderr output of crash reproduce command.
        """
        # Remove ANSI escape codes
        uncolored_output = _remove_ansi_escape_codes(stderr_output)
        # Return output starting from the first line that contains "error"
        # TODO: Make sure this works for most, if not all, build systems
        has_error_started = False
        error_lines = []
        for line in uncolored_output.splitlines():
            if has_error_started:
                error_lines.append(line)
                continue
            if "error" in line.lower():
                has_error_started = True
                error_lines.append(line)
        return "\n".join(error_lines)

    @staticmethod
    def run_lint(is_simulation: bool = False) -> None:
        pass


class AutoPatcher(PatcherInterface):
    def __init__(
        self, container: ArvoContainer, source_utils: SourceUtilsInterface
    ) -> None:
        super().__init__(source_utils)
        self.container: ArvoContainer = container

    async def patch_file(
        self, filename: str, original_text: str, new_text: str
    ) -> None:
        """Patch a file in the container."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        # Copy file out of container
        await self.container.copy_from_container(filename, temp_file.name)

        # Overwrite original text with new text
        with open(temp_file.name, "r+") as file:
            original_contents = file.read()
            if original_text not in original_contents:
                original_text_with_unix_line_endings = original_text.replace(
                    "\r\n", "\n"
                )
                if original_text_with_unix_line_endings in original_contents:
                    original_text = original_text_with_unix_line_endings
                    self.container.logger.debug(
                        f"Standardized line endings from \\r\\n to \\n in the function source code extracted from {filename}."
                    )
                else:
                    raise RuntimeError(
                        f"Failed to patch file `{filename}`. "
                        "The original text is not found in the file content."
                    )
            modified_contents = original_contents.replace(original_text, new_text)
            file.seek(0)
            file.write(modified_contents)
            file.truncate()

        # Copy file back into container
        await self.container.ensure_tracked_by_git(filename)
        await self.container.copy_to_container(temp_file.name, filename)
        self.container.build_status = BuildStatus.UNBUILT
        os.remove(temp_file.name)
