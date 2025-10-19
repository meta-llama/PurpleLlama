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
        config.base_url = config.base_url or "/".join(['https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com', self.model, "v1"])
        config.api_key = config.api_key or os.getenv("RITS_API_KEY")
        super().__init__(config)
        self._reasoning_models = [self.model]
