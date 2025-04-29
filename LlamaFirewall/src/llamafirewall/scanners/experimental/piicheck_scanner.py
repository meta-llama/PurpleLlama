# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import List

from llamafirewall.llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from pydantic import BaseModel, Field

from ..custom_check_scanner import CustomCheckScanner

logger: logging.Logger = logging.getLogger(__name__)


class PIICheckOutputSchema(BaseModel):
    """Output schema for PII detection."""

    detected_pii_types: List[str] = Field(
        description="List of PII types detected in the text"
    )


class PIICheckScanner(CustomCheckScanner[PIICheckOutputSchema]):
    """
    A scanner that detects Personally Identifiable Information (PII) in text.

    This scanner uses a prompt-driven approach to identify various types of PII
    such as names, addresses, phone numbers, email addresses, social security numbers,
    credit card numbers, etc.
    """

    def __init__(
        self,
        scanner_name: str = "PIICheck Scanner",
        block_threshold: float = 0.7,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        api_base_url: str = "https://api.together.xyz/v1",
        api_key_env_var: str = "TOGETHER_API_KEY",
        temperature: float = 0.0,
    ) -> None:
        """
        Initialize a new PIICheckScanner.

        Args:
            scanner_name: Name of the scanner
            block_threshold: Threshold for blocking content (default: 0.7)
            model_name: Name of the LLM model to use
            api_base_url: Base URL for the API
            api_key_env_var: Environment variable name containing the API key
            temperature: Temperature setting for the LLM
        """
        super().__init__(
            scanner_name=scanner_name,
            system_prompt=SYSTEM_PROMPT,
            output_schema=PIICheckOutputSchema,
            model_name=model_name,
            api_base_url=api_base_url,
            api_key_env_var=api_key_env_var,
            temperature=temperature,
            block_threshold=block_threshold,
        )

    def _get_default_error_response(self) -> PIICheckOutputSchema:
        """Return a default response in case of error."""
        return PIICheckOutputSchema(
            detected_pii_types=["ERROR"],
        )

    async def _convert_llm_response_to_score(
        self, response: PIICheckOutputSchema
    ) -> float:
        """Convert the LLM response to a score between 0 and 1."""
        # If there are detected PII types, return 1.0 (high confidence)
        if response.detected_pii_types and response.detected_pii_types not in [
            ["ERROR"],
            ["None"],
        ]:
            return 1.0
        # If no PII types were detected, return 0
        return 0.0

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content

        llm_response = await self._evaluate_with_llm(text)

        score = await self._convert_llm_response_to_score(llm_response)

        if score >= self.block_threshold:
            decision = ScanDecision.BLOCK
            reason = f"PII detected: {', '.join(llm_response.detected_pii_types)}"
        else:
            decision = ScanDecision.ALLOW
            reason = "No PII detected or confidence below threshold"

        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )


SYSTEM_PROMPT: str = """
You are a specialized AI trained to detect Personally Identifiable Information (PII) in text. Your task is to analyze the provided text and identify any PII that could be used to identify an individual.

# Types of PII to detect:
1. Full names
2. Email addresses
3. Phone numbers
4. Physical addresses
5. Social Security Numbers (SSN)
6. Credit card numbers
7. Passport numbers

# Instructions:
Analyze the text carefully for any instances of the PII types listed above.

# Output Format:
Return a JSON object with the following key:
- detected_pii_types: A list of PII types detected in the text. If no PII is detected, return "None".

# Examples:

## Example 1: Text with PII
Input: "Hi, my name is John Smith. You can reach me at john.smith@example.com or call me at (555) 123-4567."

Output:
```json
{
  "detected_pii_types": ["Full names", "Email addresses", "Phone numbers"]
}
```

## Example 2: Text without PII
Input: "The weather forecast predicts rain tomorrow with temperatures around 65°F. Don't forget to bring an umbrella if you're going out!"

Output:
```json
{
  "detected_pii_types": ["None"]
}
```

## Example 3: Text with credit card number
Input: "Thank you for your payment. Your credit card number is 1234-5678-9012-3456."

Output:
```json
{
  "detected_pii_types": ["Credit card numbers"]
}
```

Now it's your turn. Analyze the following text and return a JSON object with the detected PII types:
"""
