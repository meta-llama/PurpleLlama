# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import re
from dataclasses import dataclass, field
from enum import Enum, Flag
from typing import Any, Dict, List, Optional

from .crash_native import minimize_frames, NativeCrash


def shorten_string(s: str, length: int) -> str:
    """Shortens a long string to a given length."""
    return s[:length] + "..." if len(s) > length else s


class CrashReproduceStatus(Flag):
    BUILD_FAILED = 0b00
    BUILD_SUCCESSFUL = 0b01
    NO_REPRODUCE = 0b10


@dataclass
class CrashReproduceResult:
    status: CrashReproduceStatus = CrashReproduceStatus.BUILD_FAILED
    # Logs produced during crash reproduce, including the build errors if the build fails.
    logs: str = ""

    def is_reproduced(self) -> bool:
        # build successful, crash not fixed
        return bool(self.status & CrashReproduceStatus.BUILD_SUCCESSFUL) and not bool(
            self.status & CrashReproduceStatus.NO_REPRODUCE
        )

    def __str__(self) -> str:
        return f"CrashReproduceResult(status={self.status}, logs={shorten_string(self.logs, 20)})"


@dataclass
class SanityCheckResult:
    """
    The result of a sanity check process.
    """

    passed: bool = False
    logs: str = ""

    def __str__(self) -> str:
        return f"SanityCheckResult(passed={self.passed}, logs={shorten_string(self.logs, 20)})"


@dataclass
class PatchVerificationResult:
    """
    The result of a patch verification process.
    """

    crash_repro_result: CrashReproduceResult = field(
        default_factory=CrashReproduceResult
    )
    sanity_check_result: SanityCheckResult = field(default_factory=SanityCheckResult)


# Failed to parse the output
class LLMOutputFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchGenCrashInfo:
    """Information about a crash.

    This class contains a subset of the info stored in the CrashInstance class.
    We use this class to make the code less coupled with the CrashInstance
    class. It also benefits testing, as we use a faked PatchGenCrashInfo.
    """

    # The crash instance ID
    crash_id: int
    # The raw output of the crash
    output: str
    # Crash commit
    crash_commit: str
    # Lionhead crash_type
    crash_type: str
    # Crash type based on the sanitizer message
    sanitizer_crash_type: str

    @staticmethod
    def create_example(crash_output: str) -> "PatchGenCrashInfo":
        return PatchGenCrashInfo(
            crash_id=0,
            output=crash_output,
            crash_commit="",
            crash_type="",
            sanitizer_crash_type="",
        )

    @staticmethod
    def get_sanitizer_crash_type(crash_output: str) -> str:
        sanitizer_summary_pattern = r"SUMMARY: (.*?Sanitizer: [a-zA-Z].*?) "
        sanitizer_summary_match = re.search(sanitizer_summary_pattern, crash_output)

        if sanitizer_summary_match:
            return sanitizer_summary_match.group(1)

        leak_pattern = r"SUMMARY: .*?Sanitizer: [0-9]* byte\(s\) leaked"
        if re.search(leak_pattern, crash_output):
            return "AddressSanitizer: memory leak"

        return "null"

    def get_output(self, shorten_depth: Optional[int] = None) -> str:
        """
        Shorten the stack trace to make it more concise.
        """
        if shorten_depth is None:
            return self.output

        native_crash = NativeCrash(self.output)
        if len(native_crash.stacktraces) == 0:
            return self.output

        frames = minimize_frames(native_crash.stacktraces[0].minimized_frames)
        depth_to_show = min(len(frames), shorten_depth)
        frames_to_show = "\n".join(
            frame.presentable_frame for frame in frames[:depth_to_show]
        )
        if shorten_depth < len(frames):
            frames_to_show += "\n...\n"

        return frames_to_show.strip() + "\n\nSUMMARY: " + native_crash.summary.strip()


@dataclass(frozen=True)
class PathAndLine:
    """
    Path and line of a stacktrace item.
    """

    file_path: str
    line_number: int

    @classmethod
    def from_crash_output(cls, crash_output: str) -> List[List["PathAndLine"]]:
        """
        `crash_output` can contain multiple stacktraces.
        This function returns a list of lists of `PathAndLine`s parsed from `crash_output`.
        """
        native_crash = NativeCrash(crash_output)
        result: List[List[PathAndLine]] = []
        for stacktrace in native_crash.stacktraces:
            tmp_list: List[PathAndLine] = []
            for frame in minimize_frames(stacktrace.minimized_frames):
                if frame.lineno is None:
                    continue
                tmp_list.append(PathAndLine(frame.file, frame.lineno))
            if tmp_list:
                result.append(tmp_list)
        return result


@dataclass(frozen=True)
class SourceLocation:
    """
    Location of a source code region.
    """

    file_path: str
    line_begin: int
    column_begin: int
    line_end: int
    column_end: int

    def __post_init__(self) -> None:
        if self.line_begin > self.line_end:
            raise RuntimeError(f"Invalid SourceLocation: {str(self)}")

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_begin}:{self.column_begin}-{self.line_end}:{self.column_end}"


@dataclass(frozen=True)
class Commit:
    """Commit (diff) information"""

    id: str = ""
    reviewers: List[str] = field(default_factory=list)
    summary: str = ""
    tags: List[str] = field(default_factory=list)
    test_plan: str = ""
    title: str = ""

    def set_id(self, id: str) -> "Commit":
        return Commit(
            id=id, title=self.title, summary=self.summary, test_plan=self.test_plan
        )

    def set_reviewers(self, reviewers: List[str]) -> "Commit":
        return Commit(
            id=self.id,
            reviewers=reviewers,
            summary=self.summary,
            test_plan=self.test_plan,
            title=self.title,
        )

    def set_tags(self, tags: List[str]) -> "Commit":
        return Commit(
            id=self.id,
            reviewers=self.reviewers,
            summary=self.summary,
            tags=tags,
            test_plan=self.test_plan,
            title=self.title,
        )


class PatchGenerationStatus(Enum):
    # Initial value
    INIT_STATUS = 0
    # Unhandled exception
    FAILED = 1
    # Whether the source code info about the crash is successfully fetched
    FETCH_SOURCE_SUCCESSFUL = 2
    # Whether the LLM provided a correctly formatted patch
    PATCH_FORMAT_CORRECT = 3
    # Whether the LLM provided a patch that builds successfully
    PATCH_BUILD_SUCCESSFUL = 4
    # Whether the LLM provided a patch that fixes the crash
    PATCH_FIXES_CRASH = 5
    # Whether the LLM provided a patch that passes the sanity checks
    PATCH_PASSES_CHECKS = 6
    # Patch generator doesn't support this type of crash yet
    NOT_SUPPORTED = 7

    def is_success(self) -> bool:
        return self == PatchGenerationStatus.PATCH_PASSES_CHECKS


@dataclass
class PatchGenerationReport:
    """
    This class maintains the report for a patch generation process.
    """

    crash_id: int = 0
    crash_type: str = ""
    sanitizer_crash_type: str = ""
    commit: Commit = field(default_factory=Commit)

    # --- Patch Generation Status ---
    # The current patch generation status.
    patch_generation_status: PatchGenerationStatus = PatchGenerationStatus.INIT_STATUS
    # The max patch_generation_status that the patch generation process has reached.
    max_patch_generation_status: PatchGenerationStatus = (
        PatchGenerationStatus.INIT_STATUS
    )
    exception: str = ""

    # --- Patch Generation Retry Round ---
    retry_round: int = 0

    # --- Iterations taken ---
    build_iters: int = 0
    fix_crash_iters: int = 0
    sanity_check_iters: int = 0

    crash_repro_time: int = 0
    llm_query_cnt: int = 0
    llms_query_time: int = 0
    sanity_check_time: int = 0
    patch_gen_time: int = 0

    chat_history: List[str] = field(default_factory=list)

    # record all the logging messages.
    logs: List[str] = field(default_factory=list)

    patched_function_name: Optional[str] = None
    patched_file_path: Optional[str] = None

    def _update_max_status(self, new_status: PatchGenerationStatus) -> None:
        if int(new_status.value) > int(self.max_patch_generation_status.value):
            self.max_patch_generation_status = new_status

    def set_new_status(
        self, new_status: PatchGenerationStatus, check: bool = True
    ) -> None:
        """
        Sets the new status of the patch generation process.

        If check is True, then the new status must be a valid transition from the current status.
        """
        if check:
            if int(new_status.value) < int(self.patch_generation_status.value):
                raise RuntimeError(
                    f"Cannot set status from {self.patch_generation_status} to {new_status}."
                )

        self.patch_generation_status = new_status

        self._update_max_status(new_status)

    @classmethod
    def __flatten(
        cls,
        output_dict: Dict[str, str],
        dic: Dict[str, Any],
        parent_name: Optional[str] = None,
    ) -> None:
        for k, v in dic.items():
            new_key = f"{parent_name}_{k}" if parent_name else k
            if isinstance(v, dict):
                cls.__flatten(output_dict, v, parent_name=k)
            elif isinstance(v, Enum):
                output_dict[new_key] = str(v)
            else:
                output_dict[new_key] = v

    def to_json(self) -> str:
        import json
        from dataclasses import asdict

        dic = asdict(self)
        dic.pop("commit")

        flattened = {}
        self.__flatten(flattened, dic)
        return json.dumps(flattened)
