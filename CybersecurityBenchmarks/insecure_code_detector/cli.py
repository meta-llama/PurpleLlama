# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from . import insecure_code_detector, languages

LOG: logging.Logger = logging.getLogger(__name__)


def validate_arguments(
    path: str,
    language: str,
) -> None:
    # TODO: Path validation
    if language not in languages.get_supported_languages():
        raise ValueError(f"Language {language} is not supported.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Insecure Code Detector.")

    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Specify the path on which you want to run the Insecure Code Detector.",
    )

    parser.add_argument(
        "-l",
        "--language",
        required=True,
        choices=languages.get_supported_languages(),
        help="Specify the language of the code you want to run the Insecure Code Detector on.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    validate_arguments(args.path, args.language)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO if not args.debug else logging.DEBUG,
    )
    LOG.info(f"Running Insecure Code Detector on {args.path} in {args.language}...")
    start_time = time.monotonic()
    results = await insecure_code_detector.analyze(
        args.language, "", None, None, args.path
    )
    for result in results:
        print(result.to_json())
    end_time = time.monotonic()
    logging.info("ICD took %dms", (end_time - start_time) * 1000)
    logging.info("Found {} issues".format(len(results)))


if __name__ == "__main__":
    asyncio.run(main())
