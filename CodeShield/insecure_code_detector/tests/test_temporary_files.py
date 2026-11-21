# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

from CodeShield.insecure_code_detector import insecure_code_detector
from CodeShield.insecure_code_detector.issues import Issue, Severity
from CodeShield.insecure_code_detector.languages import Language
from CodeShield.insecure_code_detector.usecases import UseCase


class TestTemporaryFiles(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.directory_type = type(directory)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(tempfile, "tempdir", directory.name))
        self.temporary_directory = stack.enter_context(
            patch.object(
                tempfile, "TemporaryDirectory", wraps=tempfile.TemporaryDirectory
            )
        )
        self.temporary_file = stack.enter_context(
            patch.object(
                tempfile, "NamedTemporaryFile", wraps=tempfile.NamedTemporaryFile
            )
        )
        stack.enter_context(
            patch.object(insecure_code_detector.oss, "ENABLE_REGEX_ANALYZER", True)
        )
        stack.enter_context(
            patch.object(insecure_code_detector.oss, "ENABLE_SEMGREP", True)
        )
        self.regex = stack.enter_context(
            patch.object(insecure_code_detector, "_regex_analyze", return_value=[])
        )
        self.prescan = stack.enter_context(
            patch.object(
                insecure_code_detector,
                "_is_semgrep_scan_recommended",
                return_value=True,
            )
        )
        self.semgrep = stack.enter_context(
            patch.object(
                insecure_code_detector, "_semgrep_analyze", new_callable=AsyncMock
            )
        )
        self.semgrep.return_value = []

    async def test_no_file_after_regex_match(self) -> None:
        issue = Issue(
            description="insecure code", cwe_id="CWE-327", severity=Severity.ERROR
        )
        self.regex.return_value = [issue]

        result = await insecure_code_detector.analyze(
            Language.PYTHON, "code", usecase=UseCase.CODESHIELD
        )

        self.assertEqual(result, [issue])
        self.semgrep.assert_not_awaited()
        self.temporary_directory.assert_not_called()
        self.temporary_file.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_no_file_when_prescan_skips_semgrep(self) -> None:
        self.prescan.return_value = False

        result = await insecure_code_detector.analyze(
            Language.PYTHON, "code", usecase=UseCase.CODESHIELD
        )

        self.assertEqual(result, [])
        self.semgrep.assert_not_awaited()
        self.temporary_directory.assert_not_called()
        self.temporary_file.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_no_file_without_semgrep(self) -> None:
        for language, enabled in ((Language.RUST, True), (Language.PYTHON, False)):
            with self.subTest(language=language, enabled=enabled):
                with patch.object(
                    insecure_code_detector.oss, "ENABLE_SEMGREP", enabled
                ):
                    result = await insecure_code_detector.analyze(language, "code")

                self.assertEqual(result, [])
                self.semgrep.assert_not_awaited()
                self.temporary_directory.assert_not_called()
                self.temporary_file.assert_not_called()
                self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cleanup_after_semgrep(self) -> None:
        async def analyze(
            generated_code: str,
            context: insecure_code_detector.CodeContext,
            usecase: UseCase,
        ) -> list[Issue]:
            self.assertEqual(context.path.suffix, ".py")
            self.assertEqual(context.path.read_text(), "before\ncode\nafter")
            self.assertFalse(context.is_existing_path)
            return []

        self.semgrep.side_effect = analyze

        result = await insecure_code_detector.analyze(
            Language.PYTHON, "code", code_before="before\n", code_after="\nafter"
        )

        self.assertEqual(result, [])
        self.semgrep.assert_awaited_once()
        self.temporary_directory.assert_called_once()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cleanup_after_write_failure(self) -> None:
        def write_code(path: Path, text: str, *, encoding: str) -> int:
            path.write_bytes(b"partial code")
            raise OSError("write failed")

        with patch.object(Path, "write_text", write_code):
            with self.assertRaisesRegex(OSError, "write failed"):
                await insecure_code_detector.analyze(Language.PYTHON, "code")

        self.semgrep.assert_not_awaited()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_directory_creation_failure(self) -> None:
        self.temporary_directory.side_effect = PermissionError("directory unavailable")

        with self.assertRaisesRegex(PermissionError, "directory unavailable"):
            await insecure_code_detector.analyze(Language.PYTHON, "code")

        self.semgrep.assert_not_awaited()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cleanup_after_encoding_failure(self) -> None:
        with self.assertRaises(UnicodeEncodeError):
            await insecure_code_detector.analyze(Language.PYTHON, "\ud800")

        self.semgrep.assert_not_awaited()
        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cleanup_failure_does_not_replace_scan_outcome(self) -> None:
        cleanup = self.directory_type.cleanup

        def fail_cleanup(directory: tempfile.TemporaryDirectory[str]) -> None:
            cleanup(directory)
            raise PermissionError("cleanup failed")

        for failure in (
            None,
            RuntimeError("scan failed"),
            asyncio.CancelledError("scan cancelled"),
        ):
            with self.subTest(failure=failure):
                self.semgrep.side_effect = failure
                with patch.object(
                    self.directory_type,
                    "cleanup",
                    fail_cleanup,
                ):
                    with self.assertLogs(
                        insecure_code_detector.LOG.name, level="WARNING"
                    ):
                        if failure is None:
                            result = await insecure_code_detector.analyze(
                                Language.PYTHON, "code"
                            )
                            self.assertEqual(result, [])
                        else:
                            with self.assertRaises(type(failure)) as raised:
                                await insecure_code_detector.analyze(
                                    Language.PYTHON, "code"
                                )
                            if isinstance(failure, RuntimeError):
                                self.assertEqual(str(raised.exception), "scan failed")

    async def test_cleanup_after_analyzer_failure(self) -> None:
        for analyzer in (self.regex, self.prescan, self.semgrep):
            with self.subTest(analyzer=analyzer):
                analyzer.side_effect = RuntimeError("scan failed")
                with self.assertRaisesRegex(RuntimeError, "scan failed"):
                    await insecure_code_detector.analyze(
                        Language.PYTHON, "code", usecase=UseCase.CODESHIELD
                    )
                analyzer.side_effect = None
                self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cleanup_after_cancellation(self) -> None:
        started = asyncio.Event()

        async def analyze(
            generated_code: str,
            context: insecure_code_detector.CodeContext,
            usecase: UseCase,
        ) -> list[Issue]:
            self.assertTrue(context.path.is_file())
            started.set()
            await asyncio.Event().wait()
            return []

        self.semgrep.side_effect = analyze
        task = asyncio.create_task(
            insecure_code_detector.analyze(Language.PYTHON, "code")
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_caller_file_is_preserved(self) -> None:
        path = self.directory / "existing.py"
        path.write_text("original code")
        for outcome in ([], RuntimeError("scan failed"), asyncio.CancelledError()):
            with self.subTest(outcome=outcome):
                self.semgrep.side_effect = (
                    outcome if isinstance(outcome, BaseException) else None
                )
                if isinstance(outcome, BaseException):
                    with self.assertRaises(type(outcome)):
                        await insecure_code_detector.analyze(
                            Language.PYTHON, "code", path=path
                        )
                else:
                    await insecure_code_detector.analyze(
                        Language.PYTHON, "code", path=path
                    )
                self.assertEqual(path.read_text(), "original code")
                self.assertEqual(list(self.directory.iterdir()), [path])

    async def test_concurrent_scan_cancellation_is_isolated(self) -> None:
        ready = asyncio.Event()
        release = asyncio.Event()
        paths: dict[str, Path] = {}

        async def analyze(
            generated_code: str,
            context: insecure_code_detector.CodeContext,
            usecase: UseCase,
        ) -> list[Issue]:
            paths[generated_code] = context.path
            self.assertEqual(context.path.read_text(), generated_code)
            if len(paths) == 3:
                ready.set()
            await release.wait()
            self.assertTrue(context.path.is_file())
            return []

        self.semgrep.side_effect = analyze
        tasks = [
            asyncio.create_task(insecure_code_detector.analyze(Language.PYTHON, code))
            for code in ("one", "two", "three")
        ]
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            self.assertEqual(len(set(paths.values())), 3)
            tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[0]
            self.assertFalse(paths["one"].exists())
            self.assertTrue(paths["two"].is_file())
            self.assertTrue(paths["three"].is_file())
            release.set()
            self.assertEqual(await asyncio.gather(*tasks[1:]), [[], []])
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        self.assertEqual(list(self.directory.iterdir()), [])

    async def test_cancelled_command_stops_child_process(self) -> None:
        started = asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []
        create_subprocess = asyncio.create_subprocess_exec

        async def create_process(
            *command: str, stdout: int, stderr: int
        ) -> asyncio.subprocess.Process:
            process = await create_subprocess(*command, stdout=stdout, stderr=stderr)
            processes.append(process)
            started.set()
            return process

        with patch.object(
            insecure_code_detector.asyncio, "create_subprocess_exec", create_process
        ):
            task = asyncio.create_task(
                insecure_code_detector._run_command(
                    [sys.executable, "-c", "import time; time.sleep(60)"]
                )
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIsNotNone(processes[0].returncode)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                        await process.communicate()
