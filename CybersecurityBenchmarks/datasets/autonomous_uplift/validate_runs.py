# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
import os
from pathlib import Path

PROMPT_DATASET_PATH = (
    "CybersecurityBenchmarks/datasets/autonomous_uplift/out/autonomous_prompts.json"
)
RESPONSE_JSON_PATH = (
    "CybersecurityBenchmarks/datasets/autonomous_uplift/out/autonomous_responses.json"
)
CONTEXT_FOLDER_PATH = "CybersecurityBenchmarks/datasets/autonomous_uplift/results/run/"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate how many of the input cyber range pairs in --prompt-file JSON appear in either the --response-file JSON or the --context-folder path, indicating those cyber ranges were used. For autonomous uplift CyberSecEval."
    )

    parser.add_argument(
        "--prompt-file",
        default=PROMPT_DATASET_PATH,
        required=True if PROMPT_DATASET_PATH is None else False,
        help=f"Path to JSON file of prompt inputs. Default is {PROMPT_DATASET_PATH}",
    )
    parser.add_argument(
        "--response-file",
        help=f"Path to JSON file of prompt responses, for example {RESPONSE_JSON_PATH}. ",
    )
    parser.add_argument(
        "--context-folder",
        help=f"Path to folder of context transcript text files, for example {CONTEXT_FOLDER_PATH}. Filename format must be 'context_[target]_[llm]_[run.id].txt'.",
    )
    parser.add_argument(
        "--response-file-out",
        help="Output path for an updated response JSON file with all of the input contexts and prompts. This is useful if the run broke when partially complete and a response JSON was not written. Only works when paired with --context-folder.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(name)s][%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )
    logger = logging.getLogger("validate_runs")

    prompts = json.loads(Path(args.prompt_file).read_text())
    logger.debug(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    attacker_in = [prompt.get("attacker") for prompt in prompts]
    contexts = []

    if args.response_file:
        responses = json.loads(Path(args.response_file).read_text())
        logger.debug(f"Loaded {len(responses)} prompts from {args.response_file}")

        attacker_out = [response.get("attacker") for response in responses]
    elif args.context_folder:
        context_paths = Path(args.context_folder).glob("context_*.txt")
        contexts = [file for file in context_paths if (os.path.getsize(file) > 0)]

        logger.debug(
            f"Loaded {len(contexts)} non-empty 'context_*.txt' files from {args.context_folder}"
        )

        attacker_out = [context.name.split("_")[1] for context in contexts]
    else:
        raise ValueError("Must provide either --response-file or --context-folder")

    if attacker_in_only_not_out := [e for e in attacker_in if e not in attacker_out]:
        print(
            f"{len(attacker_in_only_not_out)} attacker hosts found in {Path(args.prompt_file).name} and not in output:\n{attacker_in_only_not_out}"
        )

        if args.response_file_out:
            if args.response_file:
                logger.error(
                    "Can only write an updated response file (--reponse-file-out) when given a --context-folder. I can gain no new information from an existing --reponse-file."
                )
                return

            logger.debug(
                f"Writing updated response file {Path(args.response_file_out)}"
            )

            updated_responses = []
            for prompt in prompts:
                if prompt["attacker"] in attacker_out:
                    # Find the file in the --context-folder with this attacker IP in its name
                    relevant_file = [
                        context
                        for context in contexts
                        if prompt["attacker"] in str(context)
                    ][0]

                    # Preserve \r and other line endings. read_text() will normalize to \n, we don't want that
                    context = relevant_file.read_bytes().decode("utf-8")

                    # Assuming name format 'context_[target]_[llm]_[run.id].txt', this fixes bug when model name contains _
                    model = "_".join(relevant_file.name.split("_")[2:-1])

                    intermediate = {}
                    intermediate["pass_id"] = 0
                    intermediate["operation_log"] = context
                    intermediate["model"] = model
                    intermediate["attacker"] = prompt["attacker"]
                    intermediate["target"] = prompt["target"]
                    intermediate["system_prompt"] = prompt["system_prompt"]
                    intermediate["initial_prompt"] = prompt["initial_prompt"]
                    intermediate["ssh_command_timeout"] = prompt["ssh_command_timeout"]
                    intermediate["id"] = prompt["id"]
                    updated_responses.append(intermediate)

            with open(args.response_file_out, "w") as f:
                json.dump(updated_responses, f, indent=4)

            logger.debug(
                f"Wrote {len(updated_responses)} updated responses to {Path(args.response_file_out).name}"
            )
    else:
        print(
            f"All {len(prompts)} attacker hosts found in output. No need to update response file."
        )


if __name__ == "__main__":
    main()
