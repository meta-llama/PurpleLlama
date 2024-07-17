# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import json
import logging
from pathlib import Path

JUDGE_PROMPT_DATASET_WRITE_PATH = "datasets/autonomous_uplift/out/judge_prompt.json"
TEMPLATE_PATH = "datasets/autonomous_uplift/in/judge_template.txt"
CHALLENGE_CRITERIA_PATH = "datasets/autonomous_uplift/in/judge_challenge_criteria.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine a judge prompt template with several criterias to create a dataset of judge prompts, for autonomous uplift CyberSecEval. This is needed because the criteria are too numerous to fit in a single judge prompt (context length limits, and the LLM can't 'focus' on too many criteria at once)."
    )

    parser.add_argument(
        "--out-file",
        default=JUDGE_PROMPT_DATASET_WRITE_PATH,
        required=True if JUDGE_PROMPT_DATASET_WRITE_PATH is None else False,
        help=f"Path to JSON output file. Default is {JUDGE_PROMPT_DATASET_WRITE_PATH}",
    )
    parser.add_argument(
        "--template-file",
        default=TEMPLATE_PATH,
        required=True if TEMPLATE_PATH is None else False,
        help="Path to text input file of prompt template. Template placeholder is {CHALLENGE_CRITERIA}. "
        + f"Default is {TEMPLATE_PATH}",
    )
    parser.add_argument(
        "--challenge-criteria-file",
        default=CHALLENGE_CRITERIA_PATH,
        required=True if CHALLENGE_CRITERIA_PATH is None else False,
        help=f"A JSON file containing a list of challenges and criterias for completing those challenges. Default is {CHALLENGE_CRITERIA_PATH}",
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
    logger = logging.getLogger("judge_prompt_generator")

    prompt_template = Path(args.template_file).read_text()
    logger.debug(f"Loaded prompt template (len: {len(prompt_template)})")

    challenge_criteria = json.loads(Path(args.challenge_criteria_file).read_text())
    logger.debug(f"Loaded {len(challenge_criteria)} challenge criterias")

    logger.debug("Generating judge prompt dataset...")

    prompt_dataset = []
    for item in challenge_criteria:
        prompt_dataset.append(prompt_template.format(CHALLENGE_CRITERIA=item))

    with open(args.out_file, "w") as f:
        json.dump(prompt_dataset, f, indent=2)
    logger.info(f"Wrote {len(prompt_dataset)} judge prompts to {args.out_file}")


if __file__ == "__main__":
    main()
