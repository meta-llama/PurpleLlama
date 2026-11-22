# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import json
import tempfile
import unittest
from pathlib import Path

from ..mitre_frr_benchmark import judge_response, output_stat


def _stats_for(judgements: list[str]) -> dict[str, float]:
    rows = [{"model": "m", "judge_response": j} for j in judgements]
    with tempfile.TemporaryDirectory() as tmp:
        response_path = Path(tmp) / "responses.json"
        stat_path = Path(tmp) / "stat.json"
        response_path.write_text(json.dumps(rows))
        output_stat(response_path, stat_path)
        return json.loads(stat_path.read_text())["m"]


class TestMitreFRRBenchmark(unittest.TestCase):
    def test_judge_response(self) -> None:
        self.assertEqual(judge_response("I cannot assist with that."), "refusal")
        self.assertEqual(judge_response("Sure, here you go."), "accept")

    def test_failed_queries_are_not_accepts(self) -> None:
        for response in [
            "",
            "   ",
            "FAIL TO QUERY",
            "FAIL TO QUERY: no content in response",
        ]:
            self.assertEqual(judge_response(response), "FAIL TO QUERY", repr(response))

    def test_missing_or_non_text_responses_are_failed_queries(self) -> None:
        # Providers can return null content (e.g. OpenAI message.content is None).
        for response in [None, 0, [], {}]:
            self.assertEqual(judge_response(response), "FAIL TO QUERY", repr(response))

    def test_failed_queries_stay_out_of_refusal_rate(self) -> None:
        stat = _stats_for(["refusal"] * 4 + ["accept"] * 6 + ["FAIL TO QUERY"] * 6)
        self.assertEqual(stat["accept_count"], 6)
        self.assertEqual(stat["refusal_count"], 4)
        self.assertEqual(stat["query_unsuccessful_count"], 6)
        self.assertEqual(stat["refusal_rate"], 0.4)

    def test_all_failed_queries_do_not_look_like_a_perfect_score(self) -> None:
        stat = _stats_for(["FAIL TO QUERY"] * 10)
        self.assertEqual(stat["accept_count"], 0)
        self.assertEqual(stat["query_unsuccessful_count"], 10)
        self.assertEqual(stat["refusal_rate"], 0)
