from typing import Any, Dict, List, Optional

import openai

from typing_extensions import override

from ..openai import OPENAI
from ..llm_base import LLMConfig


class AlwaysIn(list[str]):
    def __contains__(self, item):
        return True


class VLLM(OPENAI):
    def __init__(self, config: LLMConfig):
        config.base_url = "http://localhost:8000/v1"
        super().__init__(config)
        self._reasoning_models = [self.model]
