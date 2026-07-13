# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

from ..multiturn_phishing_benchmark import grade_attribute, grading_function

NEUTRAL_GRADE = 0
NEUTRAL_NOTES = "Error while grading"


class FakeLLM:
    """Minimal LLM stub that returns a canned judge response."""

    def __init__(self, response: str) -> None:
        self._response = response

    # matches the call site in grade_attribute
    def query_with_system_prompt_with_retries(
        self, *args: object, **kwargs: object
    ) -> str:
        return self._response


class TestGradeAttribute(unittest.TestCase):
    def _grade(self, response: str) -> dict:
        return grade_attribute("dialogue", "persuasion", FakeLLM(response))

    def test_valid_schema_is_returned(self) -> None:
        response = (
            '{"numerical_grade": 3, '
            '"step_by_step_grading_logic_description_and_grading_notes": "ok"}'
        )
        result = self._grade(response)
        self.assertEqual(result["numerical_grade"], 3)
        self.assertEqual(
            result["step_by_step_grading_logic_description_and_grading_notes"], "ok"
        )

    def test_invalid_json_falls_back_to_neutral_grade(self) -> None:
        result = self._grade("not json at all")
        self.assertEqual(result["numerical_grade"], NEUTRAL_GRADE)
        self.assertEqual(
            result["step_by_step_grading_logic_description_and_grading_notes"],
            NEUTRAL_NOTES,
        )

    def test_valid_json_missing_grade_key_falls_back(self) -> None:
        # Guided decoding is best-effort; a judge can return valid JSON that
        # omits numerical_grade. Must not raise KeyError downstream.
        result = self._grade(
            '{"step_by_step_grading_logic_description_and_grading_notes": "ok"}'
        )
        self.assertEqual(result["numerical_grade"], NEUTRAL_GRADE)

    def test_valid_json_non_object_falls_back(self) -> None:
        for response in ("5", "[1, 2, 3]", '"just a string"'):
            with self.subTest(response=response):
                result = self._grade(response)
                self.assertEqual(result["numerical_grade"], NEUTRAL_GRADE)

    def test_non_numeric_grade_falls_back(self) -> None:
        # A string grade would break the sum in grading_function.
        result = self._grade(
            '{"numerical_grade": "3", '
            '"step_by_step_grading_logic_description_and_grading_notes": "ok"}'
        )
        self.assertEqual(result["numerical_grade"], NEUTRAL_GRADE)


class TestGradingFunction(unittest.TestCase):
    def test_off_schema_judge_output_does_not_raise(self) -> None:
        # Every attribute grade comes back off-schema; grading must degrade to a
        # neutral overall score instead of raising KeyError/TypeError.
        grades = grading_function("dialogue", FakeLLM('{"unexpected": true}'))
        self.assertEqual(grades["overall_score"], 0)
        self.assertEqual(set(grades["scores"]), {"persuasion", "rapport", "argumentation"})

    def test_valid_judge_output_averages(self) -> None:
        response = (
            '{"numerical_grade": 4, '
            '"step_by_step_grading_logic_description_and_grading_notes": "ok"}'
        )
        grades = grading_function("dialogue", FakeLLM(response))
        self.assertEqual(grades["overall_score"], 4)


if __name__ == "__main__":
    unittest.main()
