# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional


class CacheHandler:
    def __init__(self, cache_file: str) -> None:
        self.cache_file = Path(cache_file)
        self.lock = threading.Lock()
        self.cache: Dict[str, str] = self.load_cache()

    def load_cache(self) -> Dict[str, str]:
        with self.lock:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as file:
                    return json.load(file)
            else:
                return {}

    def get_response(
        self, model: str, system_prompt: Optional[str], user_prompt: str
    ) -> Optional[str]:
        key = json.dumps((model, system_prompt, user_prompt))
        with self.lock:
            return self.cache.get(key)

    def save_response(
        self, model: str, system_prompt: Optional[str], user_prompt: str, response: str
    ) -> None:
        key = json.dumps((model, system_prompt, user_prompt))
        with self.lock:
            self.cache[key] = response
            with open(self.cache_file, "w") as file:
                json.dump(self.cache, file, indent=4)

    def get_response_chat(
        self,
        model: str,
        system_prompt: Optional[str],
        prompt_with_history: List[str],
    ) -> Optional[str]:
        prompt_with_history_key = "".join(prompt_with_history)
        key = json.dumps((model, system_prompt, prompt_with_history_key))
        with self.lock:
            return self.cache.get(key)

    def save_response_chat(
        self,
        model: str,
        system_prompt: Optional[str],
        prompt_with_history: List[str],
        response: str,
    ) -> None:
        prompt_with_history_key = "".join(prompt_with_history)
        key = json.dumps((model, system_prompt, prompt_with_history_key))
        with self.lock:
            self.cache[key] = response
            with open(self.cache_file, "w") as file:
                json.dump(self.cache, file, indent=4)

    def clear_cache(self) -> None:
        with self.lock:
            self.cache = {}
            if self.cache_file.exists():
                self.cache_file.unlink()
