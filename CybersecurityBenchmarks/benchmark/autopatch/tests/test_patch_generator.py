# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import json
import logging
import os
from logging import Handler
from pathlib import Path
from typing import Iterator, List, Optional, Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from later.unittest import TestCase

from ..llm_interface import forget_middle_of_the_convo, LLMInterface

from ..patch_generator import (
    extract_code,
    LOGGER as patch_generator_logger,
    PatcherInterface,
    PatchGenConfig,
    PatchGenCrashInfo,
    PatchGenerator,
)
from ..source_utils import FakeSourceUtils, TreeSitterSourceUtils

from ..tools import get_new_report, PatchVerifierInterface
from ..types import (
    CrashReproduceResult,
    CrashReproduceStatus,
    LLMOutputFormatError,
    PatchGenerationReport,
    PatchGenerationStatus,
    SanityCheckResult,
)


TEST_ROOT_PATH: Path = Path(__file__).parent
LOGGER: logging.Logger = logging.getLogger(__name__)

# -- Environment variables to control the behavior of the test. --
# If set, we will not use any fake implementations.
USE_REAL_FOR_ALL = "USE_REAL_FOR_ALL"
# If set, we will use the real LLM instead of a fake one that uses cached responses.
USE_REAL_LLM = "USE_REAL_LLM"
# E.g., "LOGGING_FILE=/tmp/debug.log". Redirects logs to the specified file. This is useful for checking logs in real-time, as buck2 test will cache the logs before the test finishes.
LOGGING_FILE = "LOGGING_FILE"


def get_fake_llm(fake_result: str) -> Type[LLMInterface]:
    class FakeLLM(LLMInterface):
        """
        This is a fake LLM that responds Hello World if the input contains Hello World.
        """

        def __init__(self, system_prompt: str) -> None:
            # self._fake_result = fake_result
            self._messages = [SystemMessage(content=system_prompt)]

        async def query(self, message: str, temperature: Optional[float] = None) -> str:
            self._messages.append(HumanMessage(content=message))
            self._messages.append(AIMessage(content=fake_result))
            return fake_result

    return FakeLLM


class FakePatchVerifier(PatchVerifierInterface):
    """
    This is a fake PatchVerifier that returns a fake result.
    """

    def __init__(
        self,
        reproduce_results: List[CrashReproduceResult],
        passes_sanity_checks: bool = True,
    ) -> None:
        self._reproduce_results: Iterator[CrashReproduceResult] = iter(
            reproduce_results
        )
        self._passes_sanity_checks = passes_sanity_checks

    async def reproduce_crash(self, crash_id: int) -> CrashReproduceResult:
        return next(self._reproduce_results)

    def run_sanity_checks(self, is_simulation: bool = False) -> SanityCheckResult:
        return SanityCheckResult(self._passes_sanity_checks)

    @staticmethod
    def parse_crash_reproduce_build_errors(stderr_output: str) -> str:
        return ""

    @staticmethod
    def run_lint(is_simulation: bool = False) -> None:
        pass


class FakePatcher(PatcherInterface):
    """
    A fake Patcher that does nothing.
    """

    async def patch_file(
        self, filename: str, original_text: str, new_text: str
    ) -> None:
        pass


class TestPatchGenerator(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._use_real_for_all: bool = os.getenv(USE_REAL_FOR_ALL) is not None
        self._use_real_llm: bool = (
            self._use_real_for_all or os.getenv(USE_REAL_LLM) is not None
        )

        self._fake_source_utils = FakeSourceUtils()
        self._fake_patcher = FakePatcher(self._fake_source_utils)
        self._fake_patch_verifier = FakePatchVerifier(
            [
                CrashReproduceResult(status=CrashReproduceStatus.BUILD_SUCCESSFUL),
                CrashReproduceResult(
                    status=CrashReproduceStatus.BUILD_SUCCESSFUL
                    | CrashReproduceStatus.NO_REPRODUCE
                ),
            ],
            passes_sanity_checks=True,
        )
        self._logging_file = os.getenv(LOGGING_FILE)
        if self._logging_file is not None:
            file_handler: Handler = logging.FileHandler(self._logging_file)
            LOGGER.addHandler(file_handler)

    def _get_patch_generator(self) -> PatchGenerator:
        """Returns a patch generator that uses fake implementations. The mocked
        patch verifier will return "crash reproducible" for the first call and
        "crash not reproducible" for the second call.
        """
        return PatchGenerator(
            llm_class=get_fake_llm(""),
            source_utils=self._fake_source_utils,
            patch_verifier=self._fake_patch_verifier,
            patcher=self._fake_patcher,
            config=PatchGenConfig.create_example(),
            new_report=lambda crash_info, prev_report: get_new_report(
                patch_generator_logger, crash_info, prev_report
            ),
        )

    async def test_handle_empty_stacktrace(self) -> None:
        pg = self._get_patch_generator()
        with open(
            TEST_ROOT_PATH
            / "cached_inputs"
            / "test_patch_generator_handle_empty_stacktrace.json",
            "r",
        ) as fp:
            cached_inputs_content = json.load(fp)
            crash_output_with_an_empty_stacktrace: str = cached_inputs_content[
                "crash_output"
            ]
        report = await pg.generate_patch(
            PatchGenCrashInfo(1, crash_output_with_an_empty_stacktrace, "", "", "")
        )

        self.assertEqual(
            report.patch_generation_status, PatchGenerationStatus.NOT_SUPPORTED
        )

        # Verify that the generation failed for the correct reason
        self.assertIn(
            "The stack trace is empty or no function is found in the first stack trace",
            "".join(report.logs),
        )

    async def test_crash_id_zero(self) -> None:
        pg = self._get_patch_generator()
        report = await pg.generate_patch(PatchGenCrashInfo(0, "", "", "", ""))
        self.assertEqual(
            report.patch_generation_status,
            PatchGenerationStatus.PATCH_PASSES_CHECKS,
        )
        self.assertEqual(
            report.max_patch_generation_status,
            PatchGenerationStatus.PATCH_PASSES_CHECKS,
        )

    async def test_patch_repro_check(self) -> None:
        repro_fixed = CrashReproduceResult(
            status=CrashReproduceStatus.BUILD_SUCCESSFUL
            | CrashReproduceStatus.NO_REPRODUCE
        )

        self.assertFalse(
            (await FakePatchVerifier([repro_fixed]).reproduce_crash(0)).is_reproduced()
        )

        crash_info = PatchGenCrashInfo(
            crash_id=0,
            output="output",
            crash_commit="crash_commit",
            crash_type="crash_type",
            sanitizer_crash_type="sanitizer_crash_type",
        )
        pg = PatchGenerator(
            llm_class=get_fake_llm(""),
            source_utils=self._fake_source_utils,
            patch_verifier=FakePatchVerifier([repro_fixed]),
            patcher=self._fake_patcher,
            config=PatchGenConfig.create_example(),
            new_report=lambda crash_info, prev_report: get_new_report(
                patch_generator_logger, crash_info, prev_report
            ),
        )
        with self.assertRaises(RuntimeError) as ex:
            await pg.generate_patch(crash_info)
        self.assertIn("crash does not reproduce", str(ex.exception))

        # Report should be populated even when the patch generator fails early
        report = pg.get_report()
        self.assertEqual(report.crash_id, 0)
        self.assertEqual(report.crash_type, crash_info.crash_type)
        self.assertEqual(report.sanitizer_crash_type, crash_info.sanitizer_crash_type)

    async def test_context_minimization(self) -> None:
        # 1 system prompt, 4 user prompts, 4 AI responses
        llm = get_fake_llm("Fake result")("System prompt")
        await llm.query("Q1: Explain")
        await llm.query("Q2: Fix")
        await llm.query("Q3: Try again")
        self.assertEqual(len(llm.get_context()), 7)
        self.assertIsInstance(llm.get_context()[3], HumanMessage)
        self.assertIsInstance(llm.get_context()[4], AIMessage)
        self.assertIsInstance(llm.get_context()[5], HumanMessage)
        self.assertIsInstance(llm.get_context()[6], AIMessage)
        forget_middle_of_the_convo(llm)
        # one system prompt, 2 user prompt, 2 AI response
        self.assertEqual(len(llm.get_context()), 5)
        self.assertIsInstance(llm.get_context()[0], SystemMessage)
        self.assertIsInstance(llm.get_context()[1], HumanMessage)
        self.assertIsInstance(llm.get_context()[2], AIMessage)
        self.assertEqual(llm.get_context()[1].content, "Q1: Explain")
        self.assertEqual(llm.get_context()[3].content, "Q2: Fix")
        self.assertEqual(llm.get_context()[-2].content, "Q2: Fix")
        self.assertIsInstance(llm.get_context()[-1], AIMessage)

    def test_extract_code(self) -> None:
        code = "// code"
        responses = [
            (
                f"""Here's a response with exactly one code block
                ```cpp
                {code}
                ```
                That's it!
                """,
                None,
            ),
            (
                """Here's an invalid response
                ```cpp
                // did not close the code section properly
                """,
                LLMOutputFormatError,
            ),
            (
                "Here's a response without any code block",
                LLMOutputFormatError,
            ),
        ]
        for response, exception in responses:
            if exception:
                with self.assertRaises(exception):
                    extract_code(response)
            else:
                self.assertEqual(extract_code(response).strip(), "// code")

    def test_report_serialization(self) -> None:
        report = PatchGenerationReport()
        serialized = report.to_json()
        expected = """{"crash_id": 0, "crash_type": "", "sanitizer_crash_type": "", "patch_generation_status": "PatchGenerationStatus.INIT_STATUS", "max_patch_generation_status": "PatchGenerationStatus.INIT_STATUS", "exception": "", "retry_round": 0, "build_iters": 0, "fix_crash_iters": 0, "sanity_check_iters": 0, "crash_repro_time": 0, "llm_query_cnt": 0, "llms_query_time": 0, "sanity_check_time": 0, "patch_gen_time": 0, "chat_history": [], "logs": [], "patched_function_name": null, "patched_file_path": null}"""
        self.assertEqual(serialized, expected)

    def test_get_fix_examples(self) -> None:
        crash_types = [
            "direct-leak",
            "write-heap-buffer-overflow",
            "null-deref",
            "allocation-size-too-big",
        ]
        pg = self._get_patch_generator()
        for crash_type in crash_types:
            self.assertNotEqual(pg.get_fix_example(crash_type), "")

    def test_strip_reasoning(self) -> None:
        test_cases = [
            (
                f"""
{LLMInterface.REASONING_START_TAG}
Okay, let's try to figure ...

So the line in question is:

if (iPixel > nPixels)
  ThrowRDE("Buffer overflow");

So the main issue is the buffer overflow check in HighPassBand::decode.
{LLMInterface.REASONING_END_TAG}
The root cause of the heap buffer overflow is in the `VC5Decompressor::Wavelet::HighPassBand::decode()` function. 

**Fix:**  
```cpp
if (iPixel >= nPixels)
```

**Explanation:**  
The ...
""".strip(),
                """
The root cause of the heap buffer overflow is in the `VC5Decompressor::Wavelet::HighPassBand::decode()` function. 

**Fix:**  
```cpp
if (iPixel >= nPixels)
```

**Explanation:**  
The ...
""".strip(),
            )
        ]
        for msg, expected in test_cases:
            self.assertEqual(LLMInterface._strip_reasoning(msg).strip(), expected)

    async def test_code_extraction(self) -> None:
        # NOTE: TreeSitter 0.21 does not parse this case correctly. More recent
        # versions of TreeSitter might fix the issue.
        # TODO: Update the test case to check the correctness of the code
        # extraction when we update the TreeSitter version.
        test_case_path = TEST_ROOT_PATH / "test_files/unparsables.c"
        code = await TreeSitterSourceUtils._retrieve_src_of_func(test_case_path, 216)
        print(code)
        self.assertIn("unparsable_function_1", code or "")
