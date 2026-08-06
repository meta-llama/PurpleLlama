# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List, Set

from ..autopatch.build.microsnapshots import lib_dump


class RecursiveCompareDictsTest(unittest.TestCase):
    def test_reports_non_string_scalar_changes(self) -> None:
        cases = (
            ("integer", 1032, 2032),
            ("float", 1.25, 2.5),
            ("boolean", False, True),
            ("null", None, 0),
        )

        for name, expected, suggested in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    list(
                        lib_dump.recursive_compare_dicts(
                            {"values": {"observed": expected}},
                            {"values": {"observed": suggested}},
                        )
                    ),
                    [(["values", "observed"], expected, suggested)],
                )

    def test_preserves_existing_comparison_policy(self) -> None:
        self.assertEqual(
            list(lib_dump.recursive_compare_dicts({"value": 7}, {"value": 7})),
            [],
        )
        self.assertEqual(
            list(
                lib_dump.recursive_compare_dicts(
                    {"value": "before"}, {"value": "after"}
                )
            ),
            [(["value"], "before", "after")],
        )
        self.assertEqual(
            list(
                lib_dump.recursive_compare_dicts({"name": "ptr_1"}, {"name": "ptr_2"})
            ),
            [],
        )
        self.assertEqual(
            list(
                lib_dump.recursive_compare_dicts(
                    {"items": [1]},
                    {"items": [2]},
                )
            ),
            [],
        )
        self.assertEqual(
            list(lib_dump.recursive_compare_dicts({"value": 7}, {})),
            [(["value"], 7, None)],
        )

    def test_find_deltas_reports_integer_change(self) -> None:
        expected = {
            "after(f)": [
                {"desc": "after(f)", "values": {"observed": 1032}},
            ]
        }
        suggested = {
            "after(f)": [
                {"desc": "after(f)", "values": {"observed": 2032}},
            ]
        }
        deltas: List[str] = []

        lib_dump.find_deltas(expected, suggested, deltas)

        self.assertEqual(
            deltas,
            ["after(f).[0].values.observed got 2032 expected 1032"],
        )

    def test_find_noisy_fields_reports_integer_local(self) -> None:
        first = {
            "after(f)": [
                {"desc": "after(f)", "values": {"observed": 0}},
            ]
        }
        second = {
            "after(f)": [
                {"desc": "after(f)", "values": {"observed": 1}},
            ]
        }
        skipped_struct_fields: Dict[str, Set[str]] = {}
        skipped_locals: Set[str] = set()

        lib_dump.find_noisy_fields(
            first,
            second,
            skipped_struct_fields,
            skipped_locals,
        )

        self.assertEqual(skipped_struct_fields, {})
        self.assertEqual(skipped_locals, {"observed"})


if __name__ == "__main__":
    unittest.main()
