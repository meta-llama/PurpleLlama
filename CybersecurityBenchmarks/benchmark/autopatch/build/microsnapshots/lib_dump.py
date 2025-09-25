# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import importlib.resources
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple


class Config:
    def __init__(self):
        self.struct_to_skipped_fields = {}
        self.skipped_locals = set()
        self.skipped_ptr_types = {}
        self.breakpoint_positions_to_expressions = {}
        self.positions = set()
        self.max_objects = 10000
        self.max_recursion_depth = 20
        self.catch_exceptions = True
        self.args = []
        self.timeout_seconds = 10
        self.patched_lines = set()

    def skip_struct_field(self, struct_name: str, field_name: str):
        if struct_name not in self.struct_to_skipped_fields:
            self.struct_to_skipped_fields[struct_name] = []
        self.struct_to_skipped_fields[struct_name].append(field_name)

    def skip_local(self, name: str):
        self.skipped_locals.add(name)

    def add_breakpoint(self, position: str, expression: str):
        if position not in self.breakpoint_positions_to_expressions:
            self.breakpoint_positions_to_expressions[position] = set()
        self.breakpoint_positions_to_expressions[position].add(expression)

    def add_patched_line(self, file: str, line: int):
        self.patched_lines.add(f"{file}:{line}")

    def dump_after_return(self, position: str):
        self.positions.add(position)

    def write(self, path: Path):
        bps = {
            k: sorted(v) for (k, v) in self.breakpoint_positions_to_expressions.items()
        }
        with open(path, "w") as f:
            jj = {
                "struct_to_skipped_fields": self.struct_to_skipped_fields,
                "skipped_ptr_types": self.skipped_ptr_types,
                "skipped_locals": sorted(self.skipped_locals),
                "breakpoint_to_expressions": bps,
                "dump_after_return_breakpoints": sorted(self.positions),
                "dump_stacktrace_breakpoints": sorted(self.patched_lines),
                "max_objects": self.max_objects,
                "max_recursion_depth": self.max_recursion_depth,
                "catch_exceptions": self.catch_exceptions,
            }
            json_string = json.dumps(jj, indent=4)
            print("Config dump: ", json_string)
            f.write(json_string)


LLDB_TEMPLATE: str = """
command script import {microsnapshot_path}
hook_dump_structs {config_path} {out_path}
run
quit
"""


def dump_values(target: Path, config: Config):
    target = target.resolve()
    target = target.absolute()
    root = Path(__file__).parent / "resources" if __package__ else Path("/")
    MICROSNAPSHOT_PLUGIN = root / "microsnapshots" / "microsnapshot.py"
    if not MICROSNAPSHOT_PLUGIN.exists():
        MICROSNAPSHOT_PLUGIN = root / "microsnapshot.py"

    with TemporaryDirectory() as tmpdirobj:
        tmpdir = Path(tmpdirobj)
        microsnapshot_path = MICROSNAPSHOT_PLUGIN
        config_path = tmpdir / "rules.json"
        out_path = tmpdir / "out.json"
        script_path = tmpdir / "test_lldb.lldb"

        with open(script_path, "w") as f:
            script = LLDB_TEMPLATE.format(
                microsnapshot_path=microsnapshot_path,
                config_path=config_path,
                out_path=out_path,
            )
            f.write(script)

        config.write(config_path)

        try:
            subprocess_result = subprocess.run(
                ["lldb", "-s", script_path, target, "--", *config.args],
                timeout=config.timeout_seconds,
            ).returncode
        except subprocess.TimeoutExpired:
            print(" ENCOUNTERED TIMEOUT")
            return [{"desc": "TIMEOUT", "values": {"TIMEOUT": True}}]
        assert subprocess_result == 0
        res = []
        with open(out_path, "r") as f:
            for line in f.readlines():
                res.append(json.loads(line.strip()))
        # print("dump_values: ", repr(res))
        return res


LLDB_TEMPLATE_STACKTRACES: str = """
command script import {microsnapshot_path}
hook_dump_stacktrace {config_path} {out_path}
breakpoint list
run
quit
"""


def dump_stacktrace(target: Path, config: Config):
    target = target.resolve()
    target = target.absolute()
    root = Path(__file__).parent / "resources" if __package__ else Path("/")
    MICROSNAPSHOT_PLUGIN = root / "microsnapshots" / "microsnapshot.py"
    if not MICROSNAPSHOT_PLUGIN.exists():
        MICROSNAPSHOT_PLUGIN = root / "microsnapshot.py"

    with TemporaryDirectory() as tmpdirobj:
        tmpdir = Path(tmpdirobj)
        microsnapshot_path = MICROSNAPSHOT_PLUGIN
        config_path = tmpdir / "rules.json"
        out_path = tmpdir / "out.json"
        script_path = tmpdir / "test_lldb.lldb"

        with open(script_path, "w") as f:
            script = LLDB_TEMPLATE_STACKTRACES.format(
                microsnapshot_path=microsnapshot_path,
                config_path=config_path,
                out_path=out_path,
            )
            f.write(script)

        config.write(config_path)

        try:
            cmd = ["lldb", "-s", script_path, target, "--", *config.args]
            print(f"Running debugger: {repr(cmd)}")
            subprocess_result = subprocess.run(
                cmd,
                timeout=config.timeout_seconds,
            ).returncode
        except subprocess.TimeoutExpired:
            print(" ENCOUNTERED TIMEOUT")
            return [{"desc": "TIMEOUT", "values": {"TIMEOUT": True}}]
        assert subprocess_result == 0
        res = []
        with open(out_path, "r") as f:
            for line in f.readlines():
                res.append(json.loads(line.strip()))
        # print("dump_values: ", repr(res))
        return res


@dataclass
class Arguments:
    command: str
    inputs: List[str]
    binary_orig: str
    hook_after: str = ""
    binary_patched: Optional[str] = None
    outfile: Optional[str] = None
    skip_file: Optional[str] = None
    patched_lines: Optional[List[str]] = None


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input files or directories containing input files. Directories will be expanded to include all files they contain.",
        nargs="+",
        dest="inputs",
    )

    parser.add_argument(
        "--hook-after",
        required=False,
        type=str,
        dest="hook_after",
    )

    parser.add_argument(
        "--out",
        type=str,
        dest="outfile",
    )
    parser.add_argument(
        "--skip-file",
        type=str,
        dest="skip_file",
    )


def parse_args(args: List[str]) -> Arguments:
    parser = argparse.ArgumentParser(prog="PROG")

    sub_parsers = parser.add_subparsers(help="sub-command help", dest="command")

    # get stacktraces
    parser_stacktrace = sub_parsers.add_parser(
        "get-stacktraces", help="extract stacktraces for the file:line pairs provided"
    )
    parser_stacktrace.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input files or directories containing input files. Directories will be expanded to include all files they contain.",
        nargs="+",
        dest="inputs",
    )

    parser_stacktrace.add_argument(
        "--out",
        type=str,
        dest="outfile",
    )

    parser_stacktrace.add_argument(
        "--binary",
        required=True,
        type=str,
        help="The binary to analyze",
        dest="binary_orig",
    )

    parser_stacktrace.add_argument(
        "--patched-line",
        action="append",
        help="file:line pairs that we want to dump stacktraces for",
        required=True,
        type=str,
        dest="patched_lines",
    )

    # get noisy config
    parser_noisy = sub_parsers.add_parser(
        "get-noisy-config", help="extract noisy fields that we dont want to dump"
    )
    add_common_args(parser_noisy)

    parser_noisy.add_argument(
        "--binary",
        required=True,
        type=str,
        help="The binary to analyze",
        dest="binary_orig",
    )

    # get delta
    parser_delta = sub_parsers.add_parser(
        "get-delta",
        help="list all paths that differ between the values obtained from the local stack frame after returning from the same function in two binaries",
    )
    add_common_args(parser_delta)
    parser_delta.add_argument(
        "--binary-correct",
        required=True,
        type=str,
        help="The binary with the correct behavior",
        dest="binary_orig",
    )
    parser_delta.add_argument(
        "--binary-suggested",
        required=True,
        type=str,
        help="The binary with the proposed fix",
        dest="binary_patched",
    )

    return Arguments(**vars(parser.parse_args(args)))


def recursive_compare_dicts(event1, event2, path=[]):
    for key in set(event1.keys()) | set(event2.keys()):
        key_path = path + [key]
        if key not in event2:
            yield (key_path, event1[key], None)
            continue
        if key not in event1:
            yield (key_path, None, event2[key])
            continue
        v1 = event1[key]
        v2 = event2[key]
        if isinstance(v1, dict) and isinstance(v2, dict):
            for diff in recursive_compare_dicts(v1, v2, key_path):
                yield diff
        else:
            if (
                isinstance(v1, str)
                and ("ptr_" not in v1)
                and isinstance(v2, str)
                and ("ptr_" not in v2)
            ):
                if v1 != v2:
                    yield (key_path, v1, v2)


def find_struct_field_for_path(
    dump1: Dict[str, Any], path: List[str]
) -> Optional[Tuple[str, str]]:
    objects = []
    node = dump1
    for key in path:
        objects.append((key, node))
        if key not in node:
            return None
        node = node[key]
    # iterate over objects in reverse order to find the first parent structure
    # then we can use the last key to find the field name
    last_key = None
    for i in range(len(objects) - 1, -1, -1):
        cur_key, obj = objects[i]
        if (
            obj is not None
            and isinstance(obj, dict)
            and "stype" in obj
            and obj["stype"] == "STRUCT"
        ):
            return (obj["type"], last_key)
        last_key = cur_key


def find_noisy_fields(dump1, dump2, out_struct_to_fields, out_locals):
    assert set(dump1.keys()) == set(dump2.keys())
    for desc_key in dump1.keys():
        dump1 = dump1[desc_key]
        dump2 = dump2[desc_key]
        assert len(dump1) == len(dump2)
        for i in range(len(dump1)):
            for path, v1, _v2 in recursive_compare_dicts(dump1[i], dump2[i]):
                # v1 might be None if the key wasn't present in dump1, but present in dump2
                # we'll try to resolve the field name from the root where it was present
                root = dump1 if v1 else dump2
                tup = find_struct_field_for_path(root[i], path)
                if tup:
                    out_struct_to_fields.setdefault(tup[0], set()).add(tup[1])
                if len(path) == 2 and path[0] == "values":
                    out_locals.add(path[1])
    return False


def transpose_event_list(events):
    res = {}
    for event in events:
        if event["desc"] not in res:
            res[event["desc"]] = []
        res[event["desc"]].append(event)
    return res


def _expand_input_files(inputs: List[str]) -> List[str]:
    args = []
    for input_path in inputs:
        path = Path(input_path)
        if path.is_file():
            args.append(input_path)
        elif path.is_dir():
            args.extend(str(file) for file in path.iterdir() if file.is_file())
    return args


def extract_noisy_fields(args: Arguments):
    assert args.command == "get-noisy-config"
    config = Config()
    config.catch_exceptions = True
    config.dump_after_return(args.hook_after)
    all_inputs = _expand_input_files(args.inputs)
    if Path("/tmp/poc").exists():
        all_inputs = ["/tmp/poc"]  # + all_inputs[:10]
    assert Path(args.binary_orig).exists(), f"Expect {args.binary_orig} to exist"

    out_skipped_struct_to_fields = {}
    out_skipped_locals = set()

    for inp in all_inputs:
        config.args = [inp]
        all_events = []
        for _i in range(3):
            events = dump_values(Path(args.binary_orig), config)
            print(f"full dump: {repr(events)}")
            all_events.append(transpose_event_list(events))
        for i in range(1, len(all_events)):
            find_noisy_fields(
                all_events[0],
                all_events[i],
                out_struct_to_fields=out_skipped_struct_to_fields,
                out_locals=out_skipped_locals,
            )
    return (
        {k: list(v) for k, v in out_skipped_struct_to_fields.items()},
        list(out_skipped_locals),
    )


def extract_stacktraces(args: Arguments):
    assert args.command == "get-stacktraces"
    config = Config()
    config.catch_exceptions = True
    all_inputs = _expand_input_files(args.inputs)
    lines = args.patched_lines
    assert lines is not None
    config.patched_lines = lines
    if Path("/tmp/poc").exists():
        all_inputs = ["/tmp/poc"] + all_inputs[:100]
    assert Path(args.binary_orig).exists(), f"Expect {args.binary_orig} to exist"

    all_stacktraces = set()
    for inp in all_inputs:
        config.args = [inp]
        stacktraces = dump_stacktrace(Path(args.binary_orig), config)
        for st in stacktraces:
            all_stacktraces.add(tuple(st[::-1]))
    return list(all_stacktraces)


def list_all_fields(dump1: Any, cur_path: List[str], out_paths: List[str]):
    if isinstance(dump1, dict):
        for k, v in dump1.items():
            list_all_fields(v, cur_path + [k], out_paths)
    if isinstance(dump1, list):
        for i, v in enumerate(dump1):
            list_all_fields(v, cur_path + [f"[{i}]"], out_paths)
    out_paths.append("missed " + ".".join(cur_path))


# dump has format "addr/name/file:num" -> [{"desc": key, values: TreeOfDumpedValues}]
def find_deltas(dump_expected, dump_patched, out_deltas):
    for desc_key in dump_expected.keys():
        dump1 = dump_expected[desc_key]
        dump2 = dump_patched.get(desc_key, None)
        for i in range(len(dump1)):
            if dump2 is None or i >= len(dump2):
                list_all_fields(dump1[i], cur_path=[desc_key], out_paths=out_deltas)
            else:
                for path, v1, v2 in recursive_compare_dicts(
                    dump1[i], dump2[i], path=[desc_key + f".[{i}]"]
                ):
                    if v2 is None:
                        list_all_fields(v1, path, out_deltas)
                    else:
                        sv1 = repr(v1)[0:100]
                        sv2 = repr(v2)[0:100]
                        out_deltas.append(".".join(path) + f" got {sv2} expected {sv1}")


def get_delta(args: Arguments):
    config = Config()
    config.catch_exceptions = True
    config.dump_after_return(args.hook_after)
    config.args = _expand_input_files(args.inputs)
    binary_patched = args.binary_patched
    if args.skip_file:
        with open(args.skip_file, "r") as f:
            skip_info = json.load(f)
            for typename, field in skip_info["struct_to_skipped_fields"].items():
                for f in field:
                    config.skip_struct_field(typename, f)
            for local in skip_info["skipped_locals"]:
                config.skip_local(local)
    assert binary_patched is not None, "Expect --binary-suggested to be set"
    assert Path(args.binary_orig).exists(), f"Expect {args.binary_orig} to exist"
    assert Path(binary_patched).exists(), f"Expect {args.binary_patched} to exist"

    out_deltas = []

    all_inputs = _expand_input_files(args.inputs)
    if Path("/tmp/poc").exists():
        all_inputs = ["/tmp/poc"] + all_inputs[:100]
    hit_orig_bp = False
    for inp in all_inputs:
        config.args = [inp]
        print("RUN DUMP ORIG", file=sys.stderr)
        events_orig = dump_values(Path(args.binary_orig), config)
        print("RUN DUMP PATCHED", file=sys.stderr)
        events_patched = dump_values(Path(binary_patched), config)
        t_events_orig = transpose_event_list(events_orig)
        t_events_patched = transpose_event_list(events_patched)
        print(f"FIND DELTAS ORIG: {repr(t_events_orig)}")
        print(f"FIND DELTAS PATCH: {repr(t_events_patched)}")
        if len(t_events_orig) >= 1:
            hit_orig_bp = True
        find_deltas(t_events_orig, t_events_patched, out_deltas)
    if not hit_orig_bp:
        return out_deltas + ["BREAKPOINT IN GROUNDTRUTH WAS NEVER EXECUTED"]
    return out_deltas


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        arg = "dump"
    else:
        arg = sys.argv[1]
    if arg not in ["dump", "noise", "delta"]:
        print("USAGE runner (dump|noise|delta)")

    if arg == "noise":
        process = os.popen(
            "buck build //security/genai/CybersecurityBenchmarks/benchmark/autopatch/build/microsnapshots:fake_fuzz_nodelta --show-full-simple-output"
        )
        fake_fuzz_nodelta = process.read().strip()
        assert Path(fake_fuzz_nodelta).exists(), f"Expecte {fake_fuzz_nodelta} to exist"
        args = parse_args(
            [
                "get-noisy-config",
                "--binary",
                fake_fuzz_nodelta,
                "--input",
                "test_input.txt",
                "--hook-after",
                "someFunction",
            ]
        )
        skipped_fields, skipped_locals = extract_noisy_fields(args)
        print("skipped_fields: " + repr(skipped_fields))
        print("skipped_locals: " + repr(skipped_locals))
    elif arg == "dump":
        process = os.popen(
            "buck build @//mode/dbg //security/genai/CybersecurityBenchmarks/benchmark/autopatch/build/microsnapshots:bin --show-full-simple-output"
        )
        target_bin = process.read().strip()
        process.close()

        config = Config()
        config.catch_exceptions = False  # we want to see and debug exceptions

        config.struct_to_skipped_fields["SomeObject"] = ["skipped"]
        # config.add_breakpoint("some_fn", "(*x)")
        config.add_breakpoint("some_template_fn", "vec")
        config.add_breakpoint("test.cpp:15", "(*z)")
        config.add_breakpoint("some_more_classes", "map")
        config.add_breakpoint("enum_and_ptrs", "foo")
        config.add_breakpoint("enum_and_ptrs", "other_1")
        config.add_breakpoint("enum_and_ptrs", "other_2")
        config.add_breakpoint("test_typedef_enum", "foo")
        config.dump_after_return("some_fn")
        assert Path(target_bin).exists(), f"Expecte {target_bin} to exist"
        events = dump_values(Path(target_bin), config)

        for event in events:
            print(event)
        out_deltas = []
        t_events = transpose_event_list(events)
        find_deltas(t_events, t_events, out_deltas)
        print("out_deltas: " + repr(out_deltas))


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
