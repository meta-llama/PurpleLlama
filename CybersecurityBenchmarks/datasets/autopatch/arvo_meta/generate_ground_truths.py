# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import glob
import json
import re
from typing import Any, Dict


def extract_and_insert_line_number(each_patch_json: Dict[str, Any]) -> Dict[str, Any]:
    # Extract line number from the patch header
    diff_output = each_patch_json["patch"]
    diff_lines = diff_output.strip().split("\n")

    patched_line_numbers = []
    added_line_numbers = []
    deleted_line_numbers = []
    added_lines = []
    current_line_number = -1
    for line in diff_lines:
        print(f"{current_line_number}: {line}")
        if line.startswith("@@"):
            match = re.search(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
            if match:
                # line_offset = int(match.group(4)) - int(match.group(2))
                # print("line offset", line_offset)
                # current_line_number += line_offset
                current_line_number = int(match.group(3))
                # print("match", line_number)
            else:
                current_line_number += 1
        elif line.startswith("+"):
            stripped = line[1:].lstrip().strip()
            if stripped not in {"", "{", "}"}:
                added_line_numbers.append(current_line_number)
                added_lines.append(stripped)
            current_line_number += 1
        elif line.startswith("-"):
            deleted_line_numbers.append(current_line_number)
            pass
        else:
            current_line_number += 1

    # print("Filtering")
    if added_line_numbers:
        prev_line_nbr = added_line_numbers[0] - 1
        in_comment = False
        last_comment_line = False
        filtered_line_numbers = []
        for line_nbr, line in zip(added_line_numbers, added_lines):
            if last_comment_line:
                last_comment_line = False
                in_comment = False

            # print(f"{line_nbr}: '{line}'")
            if line.startswith("//") or (line.startswith("/*") and line.endswith("*/")):
                prev_line_nbr = line_nbr
                continue

            if line.startswith("/*"):
                # print("begin comment: ", line_nbr)
                in_comment = True
            elif in_comment and line.startswith("*/"):
                # print("end comment: ", line_nbr)
                last_comment_line = True

            if (prev_line_nbr + 1 == line_nbr) and in_comment:
                prev_line_nbr = line_nbr
                continue
            else:
                filtered_line_numbers.append(line_nbr)

            prev_line_nbr = line_nbr

        added_line_numbers = filtered_line_numbers

    patched_line_numbers = added_line_numbers
    if not patched_line_numbers:
        if deleted_line_numbers:
            prev_nbr = -1
            for nbr in deleted_line_numbers:
                if prev_nbr + 1 == nbr or prev_nbr == nbr:
                    prev_nbr = nbr
                    continue
                else:
                    patched_line_numbers.append(nbr)
                    prev_nbr = nbr
        if patched_line_numbers:
            print("No added lines found, using deleted lines instead")
    print("Patched line numbers:", patched_line_numbers)

    # order
    if "line_number" in each_patch_json:
        each_patch_json.pop("line_number")
    if "patched_line_numbers" in each_patch_json:
        each_patch_json.pop("patched_line_numbers")
    each_patch_json["filename"] = each_patch_json.pop("filename")
    each_patch_json["patched_line_numbers"] = patched_line_numbers
    each_patch_json["patch"] = each_patch_json.pop("patch")

    return each_patch_json


# Example usage
patch_text = """[
  {
    "filename": "epan/dissectors/packet-dns.c",
    "patch": "@@ -1504,11 +1504,9 @@ add_rr_to_tree(proto_tree  *rr_tree, tvbuff_t *tvb, int offset,\\n   proto_item *ttl_item;\\n   gchar      **srv_rr_info;\\n \\n-  if (type == T_SRV) {\\n+  if (type == T_SRV && name[0]) {\\n     srv_rr_info = wmem_strsplit(wmem_packet_scope(), name, \".\", 3);\\n \\n-    /* The + 1 on the strings is to skip the leading '_' */\\n-\\n     proto_tree_add_string(rr_tree, hf_dns_srv_service, tvb, offset,\\n                           namelen, srv_rr_info[0]);\\n "
  }
]"""

for patch_path in glob.glob("*patch.json"):
    with open(patch_path, "r") as f:
        patch_json = json.load(f)
    print(f"\n{patch_path}")
    updated_patch = [
        extract_and_insert_line_number(each_patch) for each_patch in patch_json
    ]
    with open(patch_path, "w") as f:
        json.dump(updated_patch, f, indent=2)
