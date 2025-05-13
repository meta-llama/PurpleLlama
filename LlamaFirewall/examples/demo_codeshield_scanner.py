# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Demo script for the CodeShield scanner in LlamaFirewall.

This script demonstrates how to use the CodeShield scanner to detect
potentially insecure code from a given input text.
"""

import logging
import sys

# pyre-strict

from typing import Dict, Optional

from llamafirewall import AssistantMessage, LlamaFirewall, Role, ScannerType, ScanResult

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


def get_sample_messages() -> Dict[str, AssistantMessage]:
    """
    Create example messages for demonstration purposes.

    Returns:
        Dict[str, AssistantM]: A dictionary mapping scenario descriptions to their corresponding messages.
    """
    # Example 1: Benign code
    # This example shows an agent suggesting a benign code snippet
    benign_message = """Sure, here's a snippet of a math function that calculates the damped sine wave:
    ```python
    import math

    def damped_sine_wave(x):
        return math.sin(x) * math.exp(-x/5) if x >= 0 else math.log(abs(x)+1)

    print(damped_sine_wave(0))
    ```"""

    # Example 2: Insecure code
    # This example shows an agent that is properly following the user's instructions
    insecure_message = """That would be done with the following code:
    ```python
    import hashlib

    def store_password(password):
        # Hashing the password using MD5 (not recommended)
        hash_object = hashlib.md5(password.encode())
        return hash_object.hexdigest())

    print(store_password("securePassword123"))
    ```"""

    return {
        "benign": AssistantMessage(content=benign_message),
        "insecure": AssistantMessage(content=insecure_message),
    }


def run_code_shield_check(message: AssistantMessage) -> Optional[ScanResult]:
    """
    Run CodeShield scanner on the provided message

    Args:
        trace: The message to analyze.

    Returns:
        Optional[ScanResult]: The scan result or None if an error occurred.
    """
    try:
        # Initialize LlamaFirewall with CodeShield scanner
        scanner_config = {
            Role.ASSISTANT: [ScannerType.CODE_SHIELD],
        }
        firewall = LlamaFirewall(scanner_config)
        return firewall.scan(message)
    except Exception as e:
        logger.error(f"Error running code shield: {str(e)}")
        return None


def main() -> int:
    """
    Main function to run the demo.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Get example messages
    example_messages = get_sample_messages()

    # Run CodeShield check on each message
    for description, message in example_messages.items():
        result = run_code_shield_check(message)
        if result is None:
            logger.error(f"Error running code shield for {description}")
            continue

        logger.info(
            f"=== Scan Result ===\n"
            f"Score: {result.score:.4f}\n"
            f"Decision: {result.decision}\n"
            f"Reason: {result.reason}\n"
            f"=================="
        )
    return 0


def invoke_main() -> None:
    sys.exit(main())


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    invoke_main()  # pragma: no cover
