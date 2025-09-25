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
        super().__init__(config)
        self._reasoning_models = AlwaysIn()
    
    @override
    def valid_models(self) -> list[str]:
        return AlwaysIn()
    
    @override
    def _init_client(self) -> openai.OpenAI:
        return openai.OpenAI(base_url="http://localhost:8000/v1", api_key="-") # noqa