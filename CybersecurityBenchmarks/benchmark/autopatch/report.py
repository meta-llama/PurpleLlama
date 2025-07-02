# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .types import PatchGenerationReport


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple JSON files into a single JSONL file"
    )
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing JSON files"
    )

    parser.add_argument("--json", type=str, help="Output JSONL file")
    parser.add_argument("--markdown", type=str, help="Optional markdown output file")
    parser.add_argument("--chat", type=str, help="Optional chat history output file")
    return parser.parse_args()


def validate_json(data):
    """Validate JSON structure and raise detailed errors"""
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                raise ValueError(f"Dictionary key must be string, got {type(k)}")
            validate_json(v)
    elif isinstance(data, list):
        for item in data:
            validate_json(item)
    elif not isinstance(data, (str, int, float, bool, type(None))):
        raise ValueError(f"Invalid JSON type: {type(data)}")


def parse_patch_generation_reports(input_files: List[str]) -> List[Dict[str, Any]]:
    out_data = []
    for input_file in input_files:
        try:
            with open(input_file, "r") as infile:
                data = json.load(infile)
                validate_json(data)
                out_data.append(data)
        except json.JSONDecodeError as e:
            logger.error(f"Error reading {input_file}: Invalid JSON format - {str(e)}")
        except FileNotFoundError:
            logger.error(f"Error: File {input_file} not found")
        except Exception as e:
            logger.error(f"Error processing {input_file}: {str(e)}")

    return out_data


def write_markdown_table(data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Write data to a markdown file as a table.

    Args:
        data: List of dictionaries containing the data
        output_file: Path to the output markdown file
    """
    if not data:
        logger.warning("No data to write to markdown file")
        return

    logger.info(f"Writing markdown table to {output_file}")

    # Define the specific order for certain keys
    priority_keys = [
        "crash_id",
        "max_patch_generation_status",
        "_last_written_function_name",
        "patch_gen_time",
        "retry_round",
        "build_iters",
        "crash_repro_time",
    ]

    # Create a sorted list of headers with priority keys first, then the rest alphabetically
    headers = []

    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    all_keys.remove("patch_generation_status")

    # Add priority keys if they exist in all_keys
    for key in priority_keys:
        if key in sorted(all_keys):
            headers.append(key)
            all_keys.remove(key)

    # Add remaining keys in alphabetical order
    headers.extend(sorted(all_keys))

    with open(output_file, "w") as md_file:
        # Write table header
        md_file.write("| " + " | ".join(headers) + " |\n")
        md_file.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        # Write table rows
        for item in data:
            row = []
            for key in headers:
                # Format the value, handle missing keys
                value = item.get(key, "")
                # Escape pipe characters and newlines for markdown
                if isinstance(value, str):
                    value = value.replace("|", "\\|").replace("\n", "<br>")
                elif isinstance(value, (dict, list)):
                    value = json.dumps(value).replace("|", "\\|")
                row.append(str(value))

            md_file.write("| " + " | ".join(row) + " |\n")

    logger.info(
        f"Successfully wrote markdown table with {len(data)} rows and {len(headers)} columns"
    )


@dataclass(frozen=True)
class ChatHistory:
    crash_id: int
    history: List[str]
    max_patch_generation_status: str

    @staticmethod
    def from_report(report: PatchGenerationReport) -> "ChatHistory":
        return ChatHistory(
            crash_id=report.crash_id,
            history=report.chat_history,
            max_patch_generation_status=str(report.max_patch_generation_status),
        )

    @staticmethod
    def from_dict(report_dict: Dict[str, Any]) -> "ChatHistory":
        return ChatHistory(
            crash_id=report_dict["crash_id"],
            history=report_dict["chat_history"],
            max_patch_generation_status=report_dict["max_patch_generation_status"],
        )


def write_chat_history_markdown(histories: List[ChatHistory], output_file: str) -> None:
    """
    Extract chat_history from each data item and write them to a markdown file.
    Each chat history is written in a separate section with a header.

    Args:
        data: List of dictionaries containing the data with chat_history
        output_file: Path to the output markdown file
    """
    if not histories:
        logger.warning("No data to extract chat history from")
        return

    logger.info(f"Writing chat histories to {output_file}")

    with open(output_file, "w") as md_file:
        md_file.write("# Chat Histories\n\n")

        for history in histories:
            # Use crash_id as identifier if available, otherwise use index
            section_title = f"## Crash ID {history.crash_id}, Status: {history.max_patch_generation_status}\n\n"
            md_file.write(section_title)

            if not history.history:
                md_file.write("No chat history found for this item.\n")
            else:
                # Write each message in the chat history
                for i, msg in enumerate(history.history):
                    md_file.write(f"### Message {i+1}\n\n")
                    msg = msg.replace("###", "@@@")
                    msg = msg.replace("##", "@@")
                    msg = msg.replace("# ", "#### ")
                    msg = msg.replace("@@", "##### ")
                    md_file.write(msg + "\n\n")

    logger.info(
        f"Successfully wrote chat histories for {len(histories)} items to {output_file}"
    )


def main():
    args = parse_args()
    import glob

    input_files = glob.glob(f"{args.input_dir}/report_*.json")
    logger.info(
        f"Found {len(input_files)} JSON files matching pattern 'report_*.json' in {args.input_dir}"
    )

    logger.info(f"Processing JSON files to output file: {args.output_file}")
    data = parse_patch_generation_reports(input_files)

    if args.json:
        with open(args.output_file, "w") as outfile:
            data_without_chat = copy.deepcopy(data)
            for item in data_without_chat:
                item.pop("chat_history", None)  # Remove chat_history from each item
            json_str = json.dumps(data, indent=2)
            outfile.write(json_str)

    if args.markdown:
        write_markdown_table(data, args.markdown)

    if args.chat:
        write_chat_history_markdown(
            [ChatHistory.from_dict(item) for item in data], args.chat
        )


if __name__ == "__main__":
    main()
