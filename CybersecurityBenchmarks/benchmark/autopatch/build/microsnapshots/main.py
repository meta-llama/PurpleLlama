# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import sys

try:
    from . import lib_dump  # Relative import for when part of a package
except ImportError:
    # pyre-ignore[21]: Could not find a module corresponding to import `lib_dump`
    import lib_dump  # @manual Absolute import for when running as a standalone script (e.g. in containers)


def main() -> None:
    args = lib_dump.parse_args(sys.argv[1:])
    if args.command == "get-delta":
        delta = lib_dump.get_delta(args)
        if args.outfile:
            with open(args.outfile, "w") as f:
                json.dump(delta, f)
            print("DELTA: " + repr(delta))
        else:
            print(repr(delta))
    elif args.command == "get-noisy-config":
        skipped_fields, skipped_locals = lib_dump.extract_noisy_fields(args)
        skip_info = {
            "struct_to_skipped_fields": skipped_fields,
            "skipped_locals": skipped_locals,
        }
        print("OBTAINED NOISY CONFIG: " + repr(skip_info))
        if args.outfile:
            with open(args.outfile, "w") as f:
                json.dump(
                    skip_info,
                    f,
                )
        else:
            print(repr(skip_info))
    elif args.command == "get-stacktraces":
        print("GET STACK TRACES FOR BPs" + repr(args.patched_lines))
        traces = lib_dump.extract_stacktraces(args)
        print("OBTAINED STACK TRACES: " + repr(traces) + f"storing in {args.outfile}")
        if args.outfile:
            with open(args.outfile, "w") as f:
                json.dump(traces, f)
        else:
            print(repr(traces))
    else:
        raise Exception("Unknown command: " + args.command)


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
