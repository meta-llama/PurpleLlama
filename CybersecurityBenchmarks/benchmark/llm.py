# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from typing import List

from .llms.anthropic import ANTHROPIC
from .llms.googlegenai import GOOGLEGENAI

from .llms.llm_base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    LLM,
    LLMConfig,
    MAX_RETRY_TIMEOUT,
    NUM_LLM_RETRIES,
    RetriesExceededException,
    UnretriableQueryException,
)
from .llms.meta import LLAMA
from .llms.openai import OPENAI
from .llms.together import TOGETHER

LOG: logging.Logger = logging.getLogger(__name__)


SPECIFICATION_FORMAT: str = "<PROVIDER>::<MODEL>(::<API KEY>)?(::BASE_URL)?"
EXAMPLE_SPECIFICATION: str = "OPENAI::gpt-3.5-turbo::<YOUR API KEY>"


def create(
    identifier: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    cot_end_tokens: List[str] | None = None,
) -> LLM:
    """
    Takes a string of the format <SPECIFICATION_FORMAT> and generates LLM instance.
    See <EXAMPLE_SPECIFICATION> for example.

    Supported formats:
    1. PROVIDER::MODEL (uses default API key from environment)
    2. PROVIDER::MODEL::API_KEY (uses provided API key)
    3. PROVIDER::MODEL::API_KEY::BASE_URL (uses custom endpoint)

    Examples:
    - OPENAI::gpt-4::sk-123::https://api.custom.com/v1
    - OPENAI::gpt-4::sk-123 (uses default OpenAI endpoint)
    """
    api_key: str | None = None
    base_url: str | None = None
    split = identifier.split("::")

    if len(split) == 2:
        provider, name = split
    elif len(split) == 3:
        provider, name, api_key = split
    elif len(split) == 4:
        provider, name, api_key, base_url = split
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    if api_key is None:
        raise ValueError(f"No API key provided for {provider}: {name}")

    config = LLMConfig(
        model=name,
        api_key=api_key,
        max_tokens=max_tokens,
        cot_end_tokens=cot_end_tokens,
        base_url=base_url,
    )

    if provider == "OPENAI":
        return OPENAI(config)
    if provider == "TOGETHER":
        return TOGETHER(config)
    if provider == "ANTHROPIC":
        return ANTHROPIC(config)
    if provider == "GOOGLEGENAI":
        return GOOGLEGENAI(config)
    if provider == "LLAMA":
        return LLAMA(config)

    raise ValueError(f"Unknown provider: {provider}")
