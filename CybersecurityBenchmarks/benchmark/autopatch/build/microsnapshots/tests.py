# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import importlib.resources
import json
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

from security.genai.CybersecurityBenchmarks.benchmark.autopatch.build.microsnapshots.lib_dump import (
    dump_stacktrace,
)

from . import lib_dump
from .lib_dump import Config, dump_values


class MemCpy(unittest.TestCase):
    def val_x(self, ptr_offset: int) -> Dict[str, Any]:
        return {
            "stype": "STRUCT",
            "type": "SomeObject",
            "values": {
                "a": 1337,
                "b": 42,
                "c": 1.23457,
                "other": {
                    "stype": "ALLOC",
                    "name": f"ptr_{ptr_offset+1}",
                    "value": {
                        "stype": "STRUCT",
                        "type": "OtherObject",
                        "values": {"c": 1, "d": -1},
                    },
                },
                "skipped": {
                    "stype": "ALLOC",
                    "name": f"ptr_{ptr_offset+2}",
                    "value": self.main_z,
                },
                "tag": [84, 65, 71, 88],
                "name": "foo",
            },
        }

    def setUp(self) -> None:
        self.main_z = {
            "stype": "STRUCT",
            "type": "SkippedObject",
            "values": {"e": 1},
        }
        self.some_fn_x = self.val_x(ptr_offset=0)
        self.some_template_fn_vec = [
            {"stype": "STRUCT", "type": "OtherObject", "values": {"c": 3, "d": 4}}
        ]
        self.some_more_classes_map = [
            [
                42,
                '"some string"',
            ]
        ]
        self.enum_and_ptrs = {
            "foo": "BAR",
            "other_1": {
                "stype": "ALLOC",
                "name": "ptr_1",
                "value": {
                    "stype": "STRUCT",
                    "type": "OtherObject",
                    "values": {"c": 1, "d": -1},
                },
            },
            "other_2": {"stype": "BACKREF", "name": "ptr_1"},
        }
        self.config = Config()
        self.config.timeout_seconds = 20
        return super().setUp()

    def assertJSONEqual(self, expectedObj: object, actualObj: object) -> None:
        """
        Provides a readable unified diff of two JSON serializable objects
        Source: https://github.com/wizzat/pyutil (MIT)
        """
        import difflib

        expected = json.dumps(expectedObj, indent=4, sort_keys=True)
        actual = json.dumps(actualObj, indent=4, sort_keys=True)

        diff = "\n".join(
            difflib.unified_diff(
                expected.split("\n"), actual.split("\n"), "expected", "actual", n=20
            )
        )

        if diff:
            raise AssertionError("Assert JSON failed:\n{}".format(diff))

    def run_dumper(self):
        with importlib.resources.path(__package__, "test_bin") as target:
            return dump_values(target, self.config)

    def run_stacktrace_dumper(self):
        with importlib.resources.path(__package__, "test_bin") as target:
            return dump_stacktrace(target, self.config)

    @contextmanager
    def fake_fuzz(self) -> Generator[Tuple[Path, Path, Path], None, None]:
        with importlib.resources.path(
            __package__, "fake_fuzz_delta"
        ) as fake_fuzz_delta:
            with importlib.resources.path(
                __package__, "fake_fuzz_nodelta"
            ) as fake_fuzz_nodelta:
                with tempfile.TemporaryDirectory() as input_dir:
                    yield (Path(input_dir), fake_fuzz_delta, fake_fuzz_nodelta)

    def test_skipped(self):
        self.config.skip_struct_field("SomeObject", "skipped")
        self.config.add_breakpoint("some_fn", "(*x)")
        dump = self.run_dumper()
        value = copy.deepcopy(self.some_fn_x)
        del value["values"]["skipped"]
        expected = [{"desc": "some_fn", "values": {"(*x)": value}}]
        self.assertJSONEqual(dump, expected)

    def test_not_skipped(self):
        self.config.add_breakpoint("some_fn", "(*x)")
        dump = self.run_dumper()
        expected = [{"desc": "some_fn", "values": {"(*x)": self.some_fn_x}}]
        self.assertJSONEqual(dump, expected)

    def test_file_line_bp(self):
        self.config.add_breakpoint("some_fn", "(*x)")
        self.config.add_breakpoint("test.cpp:15", "(*z)")
        dump = self.run_dumper()
        expected = [
            {"desc": "test.cpp:15", "values": {"(*z)": self.main_z}},
            {"desc": "some_fn", "values": {"(*x)": self.some_fn_x}},
        ]
        self.assertJSONEqual(dump, expected)

    def test_stl(self):
        self.config.add_breakpoint("some_template_fn", "vec")
        self.config.add_breakpoint("some_more_classes", "map")
        dump = self.run_dumper()
        expected = [
            {"desc": "some_template_fn", "values": {"vec": self.some_template_fn_vec}},
            {
                "desc": "some_more_classes",
                "values": {"map": self.some_more_classes_map},
            },
        ]
        self.assertJSONEqual(dump, expected)

    def test_enum_and_ptrs(self):
        self.config.add_breakpoint("enum_and_ptrs", "foo")
        self.config.add_breakpoint("enum_and_ptrs", "other_1")
        self.config.add_breakpoint("enum_and_ptrs", "other_2")
        self.config.add_breakpoint("test_typedef_enum", "foo")
        dump = self.run_dumper()
        expected = [
            {"desc": "enum_and_ptrs", "values": self.enum_and_ptrs},
            {"desc": "test_typedef_enum", "values": {"foo": "FOO"}},
        ]
        self.assertJSONEqual(dump, expected)

    def test_hook_after(self):
        self.config.dump_after_return("some_fn")
        self.config.skip_local("argc")
        self.config.skip_local("argv")
        self.config.skip_local("std::__ioinit")
        dump = self.run_dumper()
        expected = [
            {
                "desc": "after(some_fn)",
                "values": {
                    "x": {
                        "name": "ptr_1",
                        "stype": "ALLOC",
                        "value": self.val_x(ptr_offset=1),
                    },
                    "y": {"stype": "BACKREF", "name": "ptr_2"},
                    "z": {"stype": "BACKREF", "name": "ptr_3"},
                },
            }
        ]
        self.assertJSONEqual(dump, expected)

    def test_dump_stacktrace(self):
        self.config.add_patched_line("test.h", 44)
        dump = [
            [fn for fn in stack if fn[0] != "_"]
            for stack in self.run_stacktrace_dumper()
        ]
        expected = [
            [
                "some_fn(SomeObject*)",
                "main",
            ]
        ]
        self.assertJSONEqual(dump, expected)

    def test_limit_recursion(self):
        self.config.add_breakpoint("some_fn", "(*x)")
        self.config.max_recursion_depth = 0
        self.config.catch_exceptions = True
        dump = self.run_dumper()
        expected = [
            {
                "desc": "some_fn",
                "values": {
                    "(*x)": {
                        "stype": "ERROR",
                        "desc": "Error while dumping",
                    }
                },
            }
        ]
        dump[0]["values"]["(*x)"]["desc"] = re.sub(
            r"0x[0-9a-f]+", "PTR", dump[0]["values"]["(*x)"]["desc"]
        )
        self.assertJSONEqual(dump, expected)

    def test_get_noisy(self):
        with self.fake_fuzz() as (tmp, fake_fuzz_delta, fake_fuzz_nodelta):
            print(tmp, fake_fuzz_delta, fake_fuzz_nodelta)
            input_p = tmp / "input"
            with open(input_p, "w") as f:
                f.write("bar")
            args = lib_dump.parse_args(
                [
                    "get-noisy-config",
                    "--binary",
                    str(fake_fuzz_nodelta),
                    "--hook-after",
                    "someFunction",
                    "--input",
                    str(input_p),
                ]
            )
            self.assertEqual(
                args,
                lib_dump.Arguments(
                    command="get-noisy-config",
                    hook_after="someFunction",
                    inputs=[str(input_p)],
                    binary_orig=str(fake_fuzz_nodelta),
                    binary_patched=None,
                ),
            )
            skipped_fields, skipped_locals = lib_dump.extract_nosiy_fields(args)
            print("skipped_fields: " + repr(skipped_fields))
            print("skipped_locals: " + repr(skipped_locals))
            self.assertJSONEqual(skipped_fields, {"SomeObject": ["some_noise"]})
            self.assertJSONEqual(skipped_locals, ["noisy_local"])

    def denoise_deltas(self, deltas):
        denoised = []
        for line in deltas:
            line = re.sub(r"got \d{6,}", "got NOISE", line)
            line = re.sub(r"expected \d{6,}", "expected NOISE", line)
            denoised.append(line)
        return sorted(denoised)

    def test_get_delta(self):
        with self.fake_fuzz() as (tmp, fake_fuzz_delta, fake_fuzz_nodelta):
            print(tmp, fake_fuzz_delta, fake_fuzz_nodelta)
            input_p = tmp / "input"
            with open(input_p, "w") as f:
                f.write("foo")
            args = lib_dump.parse_args(
                [
                    "get-delta",
                    "--binary-correct",
                    str(fake_fuzz_nodelta),
                    "--binary-suggested",
                    str(fake_fuzz_delta),
                    "--hook-after",
                    "someFunction",
                    "--input",
                    str(input_p),
                ]
            )
            self.assertEqual(
                args,
                lib_dump.Arguments(
                    command="get-delta",
                    hook_after="someFunction",
                    inputs=[str(input_p)],
                    binary_orig=str(fake_fuzz_nodelta),
                    binary_patched=str(fake_fuzz_delta),
                ),
            )
            deltas = lib_dump.get_delta(args)
            denoised = self.denoise_deltas(deltas)
            self.assertJSONEqual(
                denoised,
                sorted(
                    [
                        "after(someFunction).[0].values.obj.values.some_noise got NOISE expected NOISE",
                        "after(someFunction).[0].values.obj.values.some_value got 4 expected 3",
                        "after(someFunction).[0].values.noisy_local got NOISE expected NOISE",
                        "after(someFunction).[1].values.obj.values.some_noise got NOISE expected NOISE",
                        "after(someFunction).[1].values.obj.values.some_value got 4 expected 3",
                        "after(someFunction).[1].values.noisy_local got NOISE expected NOISE",
                    ]
                ),
            )

            # the input bar shouldn't trigger a delta
            with open(input_p, "w") as f:
                f.write("bar")
            deltas = lib_dump.get_delta(args)
            denoised = self.denoise_deltas(deltas)
            self.assertJSONEqual(
                denoised,
                sorted(
                    [
                        "after(someFunction).[0].values.obj.values.some_noise got NOISE expected NOISE",
                        "after(someFunction).[0].values.noisy_local got NOISE expected NOISE",
                        "after(someFunction).[1].values.obj.values.some_noise got NOISE expected NOISE",
                        "after(someFunction).[1].values.noisy_local got NOISE expected NOISE",
                    ]
                ),
            )

    def test_get_delta_with_noise_file(self):
        with self.fake_fuzz() as (tmp, fake_fuzz_delta, fake_fuzz_nodelta):
            print(tmp, fake_fuzz_delta, fake_fuzz_nodelta)
            input_p = tmp / "input"
            with open(input_p, "w") as f:
                f.write("foo")
            skip_file = tmp / "noise_file"
            with open(skip_file, "w") as f:
                json.dump(
                    {
                        "struct_to_skipped_fields": {"SomeObject": ["some_noise"]},
                        "skipped_locals": ["noisy_local"],
                    },
                    f,
                )
            args = lib_dump.parse_args(
                [
                    "get-delta",
                    "--binary-correct",
                    str(fake_fuzz_nodelta),
                    "--binary-suggested",
                    str(fake_fuzz_delta),
                    "--hook-after",
                    "someFunction",
                    "--input",
                    str(input_p),
                    "--skip-file",
                    str(skip_file),
                ]
            )
            self.assertEqual(
                args,
                lib_dump.Arguments(
                    command="get-delta",
                    hook_after="someFunction",
                    inputs=[str(input_p)],
                    binary_orig=str(fake_fuzz_nodelta),
                    binary_patched=str(fake_fuzz_delta),
                    skip_file=str(skip_file),
                ),
            )
            deltas = lib_dump.get_delta(args)
            denoised = self.denoise_deltas(deltas)
            self.assertJSONEqual(
                denoised,
                sorted(
                    [
                        "after(someFunction).[0].values.obj.values.some_value got 4 expected 3",
                        "after(someFunction).[1].values.obj.values.some_value got 4 expected 3",
                    ]
                ),
            )
