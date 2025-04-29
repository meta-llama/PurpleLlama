# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Command-line argument parsing for ClassifyIt Scanner
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ClassifyIt Scanner")

    parser.add_argument(
        "--drive_type",
        choices=["my_drive", "shared_drive"],
        required=True,
        help="Type of drive to scan ('my_drive' or 'shared_drive')",
    )

    parser.add_argument(
        "--shared_drive_id",
        help="Shared drive ID to scan (e.g. 'abc123')",
    )

    parser.add_argument(
        "--drive_owner",
        required=True,
        help="Drive Organizer email (e.g. driveorganizer@mydomain.com)",
    )

    parser.add_argument(
        "--service_account_file",
        required=True,
        help="Path to the Google Cloud Platform service account "
        + "credentials file (e.g. 'credentials.json')",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without applying labels",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--output_dir",
        default="/tmp",
        help="Directory to save output report files (default: /tmp)",
    )

    parser.add_argument(
        "--session_config",
        required=True,
        help="Path to the session config file (e.g. 'session_config.yaml')",
    )

    parser.add_argument(
        "--db_path",
        help="Path to the SQLite database file for this Google Drive(default: "
        + "/tmp/{drive_owner}_label_{label_id}.sqlite)",
    )

    args = parser.parse_args()

    if args.drive_type == "shared_drive" and not args.shared_drive_id:
        parser.error("Shared drive ID is required when using shared drive")

    return args
