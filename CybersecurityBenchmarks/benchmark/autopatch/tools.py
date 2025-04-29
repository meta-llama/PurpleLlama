# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import os
import shlex
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from io import BufferedReader
from typing import Dict, List, Optional, TextIO, Type

from .types import (
    CrashReproduceResult,
    PatchGenCrashInfo,
    PatchGenerationReport,
    SanityCheckResult,
)


class RedirectOutput:
    def __init__(
        self,
        stdout_target: Optional[TextIO] = None,
        stderr_target: Optional[TextIO] = None,
    ) -> None:
        self.stdout_target: TextIO = stdout_target or tempfile.TemporaryFile(mode="w+")
        self.stderr_target: TextIO = stderr_target or tempfile.TemporaryFile(mode="w+")
        self._stdout_fd: int = sys.stdout.fileno()
        self._stderr_fd: int = sys.stderr.fileno()
        # Save original file descriptors
        self._saved_stdout_fd: int = os.dup(self._stdout_fd)
        self._saved_stderr_fd: int = os.dup(self._stderr_fd)

    def __enter__(self) -> "RedirectOutput":
        # Redirect stdout and stderr to the target files
        os.dup2(self.stdout_target.fileno(), self._stdout_fd)
        os.dup2(self.stderr_target.fileno(), self._stderr_fd)

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        # Restore original stdout and stderr file descriptors
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.dup2(self._saved_stderr_fd, self._stderr_fd)

        # Close the duplicated file descriptors
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)

        # Seek to the beginning of the target files if they are TemporaryFiles
        if hasattr(self.stdout_target, "seek"):
            self.stdout_target.seek(0)
        if hasattr(self.stderr_target, "seek"):
            self.stderr_target.seek(0)

    def get_stdout(self) -> Optional[str]:
        if hasattr(self.stdout_target, "read"):
            return self.stdout_target.read()
        return None

    def get_stderr(self) -> Optional[str]:
        if hasattr(self.stderr_target, "read"):
            return self.stderr_target.read()
        return None


class PatchVerifierInterface(ABC):
    """
    Interface for patch verifying functions.
    It offers better testability as we can use a fake verifier implementation in the tests.
    """

    @abstractmethod
    async def reproduce_crash(self, crash_id: int) -> CrashReproduceResult:
        """
        Tries to reproduce the crash on the current revision.

        Returns a CrashReproduceResult.
        """

    @abstractmethod
    def run_sanity_checks(self, is_simulation: bool = False) -> SanityCheckResult:
        """
        Tries to run sanity checks on the current revision to make sure the patch does not cause regression.

        Returns a SanityCheckResult.
        """

    @staticmethod
    @abstractmethod
    def parse_crash_reproduce_build_errors(stderr_output: str) -> str:
        """
        Parse build errors from stderr output of crash reproduce command.
        """

    @staticmethod
    @abstractmethod
    def run_lint(is_simulation: bool = False) -> None:
        """Run linter"""


class CrashInfoRetrieverInterface(ABC):
    """
    Interface for getting the information about a crash.
    It offers better testability as we can use a fake CrashInfoRetriever in the tests.
    """

    @abstractmethod
    async def get_crash_info_by_id(self, crash_id: int) -> Optional[PatchGenCrashInfo]:
        """
        Returns a CrashInstance from the crash_id, or None if the crash_id is not found.
        """


class CustomFormatter(logging.Formatter):
    """Custom formatter that lets us define the line number and filename
    for a log call.  See log_method()"""

    def format(self, record: logging.LogRecord) -> str:
        r = getattr(record, "myline", None)
        if r is not None:
            record.lineno = r
        r = getattr(record, "myfile", None)
        if r is not None:
            record.filename = os.path.basename(r)
        return logging.Formatter.format(self, record)


class GlogFormatter(CustomFormatter):
    """
    A Formatter that outputs messages in the same format used by the google
    logging library.
    """

    LEVEL_CHARS: Dict[int, str] = {
        logging.CRITICAL: "C",
        logging.ERROR: "E",
        logging.WARNING: "W",
        logging.INFO: "I",
        logging.DEBUG: "V",  # glog defines this as "VERBOSE"
        logging.NOTSET: "N",  # This doesn't actually exist in glog
    }

    DEFAULT_FORMAT = (
        "%(levelchar)s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
    )
    DEFAULT_DATEFORMAT = "%m%d %H%M%S"
    COLON_DATEFORMAT = "%m%d %H:%M:%S"

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        escape_newlines: bool = True,
    ) -> None:
        self.escape_newlines = escape_newlines
        if fmt is None:
            fmt = self.DEFAULT_FORMAT
        if datefmt is None:
            datefmt = self.DEFAULT_DATEFORMAT
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # pyre-fixme[16]: `LogRecord` has no attribute `levelchar`.
        record.levelchar = GlogFormatter.get_level_char(record.levelno)
        result = CustomFormatter.format(self, record)
        if self.escape_newlines:
            result = result.replace("\n", r"\n")
        return result

    @staticmethod
    def get_level_char(level: int) -> str:
        # Just return '?' for numerical levels that don't correspond to one of
        # the pre-defined symbolic level names.
        return GlogFormatter.LEVEL_CHARS.get(level, "?")


class ReportHandler(logging.Handler):
    """Logs the logging messages to a PatchGenerationReport"""

    def __init__(self, report: PatchGenerationReport) -> None:
        super().__init__()
        self._report = report
        self.setFormatter(
            GlogFormatter(datefmt=GlogFormatter.COLON_DATEFORMAT, escape_newlines=False)
        )

    # COLON_DATEFORMAT = "%m%d %H:%M:%S"

    def emit(self, record: logging.LogRecord) -> None:
        """
        This method is called by the logging framework when a logging message is emitted.
        """
        self._report.logs.append(self.format(record))


def get_new_report(
    logger: logging.Logger,
    crash_info: Optional[PatchGenCrashInfo] = None,
    prev_report: Optional[PatchGenerationReport] = None,
) -> PatchGenerationReport:
    """Gets a new report. We should get a new report before each patch generation."""
    report = PatchGenerationReport()
    logger.addHandler(ReportHandler(report))
    if crash_info:
        report.crash_id = crash_info.crash_id
        report.crash_type = crash_info.crash_type
        report.sanitizer_crash_type = crash_info.sanitizer_crash_type
    if prev_report:
        report.max_patch_generation_status = prev_report.max_patch_generation_status
        report.retry_round = prev_report.retry_round + 1
    return report


def subprocess_run(
    cmd: List[str], check: bool = False, stdin: Optional[BufferedReader] = None
) -> subprocess.CompletedProcess[str]:
    logging.info(f"Running: {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, capture_output=True, check=check, text=True, stdin=stdin)
