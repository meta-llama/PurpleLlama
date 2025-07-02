#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ClassifyIt Scanner

This tool extracts text from files in Google Drive, classifies it using Llama Stack,
and optionally applies labels to files depending on the LLM classification results.
"""

import yaml
from classifyit import ClassifyIt

from cli.argument_parser import parse_arguments


def main() -> None:
    """Entrypoint"""
    args = parse_arguments()

    # Parse session configuration file
    with open(args.session_config, "r", encoding="utf-8") as file:
        session_config = yaml.safe_load(file)

    # Initialize ClassifyIt
    classifier = ClassifyIt(
        session_config=session_config,
        drive_owner=args.drive_owner,
        service_account_file=args.service_account_file,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dry_run=args.dry_run,
        db_path=args.db_path,
    )

    # Scan the drive
    classifier.scan_drive(
        drive_type=args.drive_type,
        shared_drive_id=args.shared_drive_id,
    )


if __name__ == "__main__":
    main()
