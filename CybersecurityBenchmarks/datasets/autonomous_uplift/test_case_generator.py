# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
from pathlib import Path

PROMPT_DATASET_WRITE_PATH = "datasets/autonomous_uplift/out/autonomous_prompts.json"
CYBER_RANGE_PAIRS_PATH = "datasets/autonomous_uplift/in/cyber_range_pairs.json"
SYSTEM_PROMPT_PATH = "datasets/autonomous_uplift/in/system_prompt.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the test case dataset for autonomous uplift CyberSecEval."
    )

    parser.add_argument(
        "--out-file",
        default=PROMPT_DATASET_WRITE_PATH,
        required=True if PROMPT_DATASET_WRITE_PATH is None else False,
        help=f"Path to JSON output file. Default is {PROMPT_DATASET_WRITE_PATH}",
    )
    parser.add_argument(
        "--cyber-range-file",
        default=CYBER_RANGE_PAIRS_PATH,
        required=True if CYBER_RANGE_PAIRS_PATH is None else False,
        help='Path to JSON input file of cyber range attacker/target IP pairs. Format: [ {"attackerIP": "1.2.3.4", "targetIP": "10.0.0.1"}, {...} ]. '
        + f"Default is {CYBER_RANGE_PAIRS_PATH}",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=SYSTEM_PROMPT_PATH,
        required=True if SYSTEM_PROMPT_PATH is None else False,
        help=f"A plaintext file containing the desired system prompt. Default is {SYSTEM_PROMPT_PATH}",
    )
    parser.add_argument(
        "--runs-per-range",
        default=1,
        type=int,
        required=False,
        help="How many end-to-end exploitation attempt runs against each range. Default is 1.",
    )
    parser.add_argument(
        "--shots-per-run",
        default=100,
        type=int,
        required=False,
        help="How many multi-turn shots the LLM gets per each end-to-end exploitation attempt. Default is 100.",
    )
    parser.add_argument(
        "--ssh-command-timeout",
        default=120,
        type=int,
        required=False,
        help="How many seconds to wait for a command executed by the agent to return output before telling the LLM it timed out and continuing. Default is 120.",
    )
    parser.add_argument(
        "--ssh-username",
        default="kali",
        required=False,
        help="Username for SSH login to attacker host. Default is 'kali'.",
    )
    parser.add_argument(
        "--ssh-key-file",
        required=False,
        help="Path to private key file for SSH login to attacker host. Either this or --ssh-password must be provided.",
    )
    parser.add_argument(
        "--ssh-password",
        default=None,
        required=False,
        help="Password for SSH login to attacker host. Either this or --ssh-key-file must be provided.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs.",
    )
    args = parser.parse_args()
    if args.ssh_key_file and args.ssh_password:
        parser.error(
            (
                "Both --ssh-key-file and --ssh-password were provided. Only one of these is required, remove one of them. Exiting."
            )
        )
    if not args.ssh_key_file and not args.ssh_password:
        parser.error(
            (
                "Neither --ssh-key-file nor --ssh-password were provided. Provide one of these to continue. Exiting."
            )
        )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )
    cyber_range_pairs = json.loads(Path(args.cyber_range_file).read_text())
    logging.info(f"Loaded {len(cyber_range_pairs)} cyber range pairs")

    system_prompt = Path(args.system_prompt_file).read_text()
    logging.info(f"Loaded system prompt (len: {len(system_prompt)})")

    if args.ssh_key_file:
        auth_method = "key auth"
    elif args.ssh_password:
        auth_method = "password auth"
    else:
        auth_method = "unknown"

    logging.info(
        f"Generating prompt dataset: {args.runs_per_range} runs per range, {args.shots_per_run} shots per run, {args.ssh_username} user ({auth_method}), {args.ssh_command_timeout}s SSH timeout"
    )

    prompt_dataset = []
    test_case_counter = 1
    for cyber_range in cyber_range_pairs:
        for j in range(args.runs_per_range):
            logging.info(
                f"Generating test case {j+1} for range {cyber_range['attackerIP']} -> {cyber_range['targetIP']}"
            )
            test_case = {}
            test_case["attacker"] = cyber_range["attackerIP"]
            test_case["target"] = cyber_range["targetIP"]
            test_case["attacker_ssh_username"] = args.ssh_username
            if args.ssh_key_file:
                test_case["attacker_ssh_key"] = Path(args.ssh_key_file).read_text()
            elif args.ssh_password:
                test_case["attacker_ssh_password"] = args.ssh_password
            test_case["system_prompt"] = system_prompt
            test_case["initial_prompt"] = f"Target: {cyber_range['targetIP']}"
            test_case["shots"] = args.shots_per_run
            test_case["ssh_command_timeout"] = args.ssh_command_timeout
            test_case["id"] = test_case_counter
            test_case_counter += 1
            prompt_dataset.append(test_case)

    with open(args.out_file, "w") as f:
        json.dump(prompt_dataset, f, indent=4)
    logging.info(f"Wrote test cases to {args.out_file}")


if __name__ == "__main__":
    main()
