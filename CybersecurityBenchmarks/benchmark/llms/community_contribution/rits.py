import os
from typing import Any, Dict, List, Optional

import openai

from typing_extensions import override

from ..openai import OPENAI
from ..llm_base import LLMConfig


class AlwaysIn(list[str]):
    def __contains__(self, item):
        return True


class RITS(OPENAI):
    def __init__(self, config: LLMConfig):
        config.base_url = config.base_url or "/".join(['https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com', config.model, "v1"])
        config.api_key = config.api_key or os.getenv("RITS_API_KEY")
        config.model = {
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "llama-4-mvk-17b-128e-fp8": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "llama-4-scout-17b-16e-instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        }[config.model]
        super().__init__(config, default_headers={"RITS_API_KEY": config.api_key})
        self._reasoning_models += [
            "gpt-oss-20b",
            "gpt-oss-120b",
        ]
