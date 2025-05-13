# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import asyncio
import logging
import os

from llamafirewall import UserMessage
from llamafirewall.scanners.experimental.piicheck_scanner import PIICheckScanner

try:
    # Try relative import first
    from .logging_setup import init_logger
except ImportError:
    try:
        # pyre-ignore[21]
        from LlamaFirewall.examples.logging_setup import init_logger  # @manual
    except ImportError:
        # pyre-ignore[21]
        from logging_setup import init_logger  # @manual

logger: logging.Logger = init_logger()


async def run_pii_scan(input_text: str, description: str) -> None:
    """Run PIICheckScanner on the provided input and display the results."""
    logger.info(f"\n\n=== Testing {description} ===")
    logger.info(f"Input: {input_text}")

    try:
        # Initialize the scanner
        pii_scanner = PIICheckScanner()

        # Create a Message object and scan it
        message = UserMessage(content=input_text)
        scan_result = await pii_scanner.scan(message)

        # Display the results
        logger.info(
            (
                f"=== Scan Result ===\n"
                f"Score: {scan_result.score}\n"
                f"Decision: {scan_result.decision}\n"
                f"Reason: {scan_result.reason}\n"
                f"=================="
            )
        )

    except Exception as e:
        logger.error(f"Error scanning {description}: {e}")


async def main() -> None:
    # Example 1: High-risk input (contains multiple PII types)
    high_risk_input = "Hi, my name is John Smith. You can reach me at john.smith@example.com or call me at (555) 123-4567. I live at 123 Main St, Anytown, CA 12345."
    await run_pii_scan(high_risk_input, "High-risk input (multiple PII types)")

    # Example 2: Low-risk input (no PII)
    low_risk_input = "The weather forecast predicts rain tomorrow with temperatures around 65°F. Don't forget to bring an umbrella if you're going out!"
    await run_pii_scan(low_risk_input, "Low-risk input (no PII)")

    # Example 3: Medium-risk input (ambiguous PII)
    medium_risk_input = "Please contact the administrator at admin@company.com to reset your password. The office is located in Building A."
    await run_pii_scan(medium_risk_input, "Medium-risk input (ambiguous PII)")

    # Example 4: High-risk input (financial information)
    financial_input = "My credit card number is 4111-1111-1111-1111 with expiration date 12/25 and CVV 123."
    await run_pii_scan(financial_input, "High-risk input (financial information)")

    # Example 5: High-risk input (government identifiers)
    government_id_input = "My social security number is 123-45-6789 and my driver's license number is D1234567."
    await run_pii_scan(government_id_input, "High-risk input (government identifiers)")


if __name__ == "__main__":
    # Check if TOGETHER_API_KEY is set
    if not os.environ.get("TOGETHER_API_KEY"):
        logger.error(
            "TOGETHER_API_KEY environment variable is not set. Please set it before running this demo."
        )
        logger.info("You can get an API key from https://together.ai")
        exit(1)

    # Run the demo
    asyncio.run(main())
