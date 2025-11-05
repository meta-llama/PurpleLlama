# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os
from typing import Tuple

LOG: logging.Logger = logging.getLogger(__name__)


def _ensure_transformers_and_torch() -> Tuple[object, object, object, object, object]:
    """Lazily import heavy ML deps and return the symbols we need.

    This avoids requiring torch/transformers at package import time. If the
    optional dependencies are missing, raise an informative ImportError.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Prompt Guard scanner requires 'torch' and 'transformers'.\n"
            "Install with: pip install torch transformers huggingface-hub"
        ) from e

    try:
        from huggingface_hub import HfFolder, login
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Prompt Guard scanner requires 'huggingface_hub'.\n"
            "Install with: pip install huggingface-hub"
        ) from e

    try:
        from torch import Tensor
        from torch._prims_common import DeviceLikeType
        from torch.nn import Module
        from torch.nn.functional import softmax
    except Exception:
        # If the torch package layout differs, still continue but raise later
        Tensor = object
        DeviceLikeType = object
        Module = object
        from torch.nn.functional import softmax  # type: ignore

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            PreTrainedTokenizer,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "Prompt Guard scanner requires 'transformers'.\n"
            "Install with: pip install transformers"
        ) from e

    return (
        torch,
        HfFolder,
        login,
        Tensor,
        DeviceLikeType,
        Module,
        softmax,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedTokenizer,
    )


class PromptGuard:
    def __init__(self) -> None:
        # Defer heavy imports until an instance is created
        (
            self._torch,
            self._HfFolder,
            self._login,
            self._Tensor,
            self._DeviceLikeType,
            self._Module,
            self._softmax,
            self._AutoModelForSequenceClassification,
            self._AutoTokenizer,
            self._PreTrainedTokenizer,
        ) = _ensure_transformers_and_torch()

        self.model: object
        self.tokenizer: object
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        if self.model:
            self.device: object = self._torch.device(
                "cuda" if self._torch.cuda.is_available() else "cpu"
            )
            try:
                self.model.to(self.device)
            except Exception:
                # Some models may not implement to(); ignore in that case
                pass

    def _load_model_and_tokenizer(
        self,
        model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M",
    ) -> tuple[Module, PreTrainedTokenizer]:
        # Set the local linux path for the model if not already set
        if not os.environ.get("HF_HOME"):
            LOG.info(
                "[LlamaFirewall] No pre-defined Hugging Face home path found. Setting it to linux default path ~/.cache/huggingface."
            )
            os.environ["HF_HOME"] = "~/.cache/huggingface"
        model_path = os.path.expanduser(
            os.path.join(os.environ["HF_HOME"], model_name.replace("/", "--"))
        )

        # Check if the model is already saved locally
        if not os.path.exists(model_path):
            LOG.warning(
                f"Prompt Guard Scanner requires {model_name} available locally to run. We are downloading it from Hugging Face Hub. This may take a while..."
            )
            if not HfFolder.get_token():
                LOG.warning(
                    "No Hugging Face token found. You will be prompted to enter it via Hugging Face login function, no data will be stored by LlamaFirewall."
                )
                self._login()
            # Load the model and tokenizer from Hugging Face
            model = self._AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            tokenizer = self._AutoTokenizer.from_pretrained(model_name)
            # Save the model and tokenizer locally
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        else:
            # Load the model and tokenizer from the local path
            model = self._AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            tokenizer = self._AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    def _preprocess_text_for_promptguard(self, text: str) -> str:
        try:
            cleaned_text = ""
            index_map = []
            for i, char in enumerate(text):
                if not char.isspace():
                    cleaned_text += char
                    index_map.append(i)
            tokens = self.tokenizer.tokenize(cleaned_text)
            result = []
            last_end = 0
            for token in tokens:
                token_str = self.tokenizer.convert_tokens_to_string([token])
                start = cleaned_text.index(token_str, last_end)
                end = start + len(token_str)
                original_start = index_map[start]
                if original_start > 0 and text[original_start - 1].isspace():
                    result.append(" ")
                result.append(token_str)
                last_end = end
            return "".join(result)
        except Exception as e:
            LOG.error(f"Error preprocessing text: {e}")
            return text

    def _get_class_probabilities(
        self,
        text: str,
        temperature: float = 1.0,
        preprocess: bool = True,
    ) -> Tensor:
        if preprocess:
            text = self._preprocess_text_for_promptguard(text)
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = inputs.to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        scaled_logits = logits / temperature
        probabilities = self._softmax(scaled_logits, dim=-1)
        return probabilities

    def get_jailbreak_score(
        self,
        text: str,
        temperature: float = 1.0,
        preprocess: bool = True,
    ) -> float:
        probabilities = self._get_class_probabilities(text, temperature, preprocess)
        try:
            return float(probabilities[0, -1].item())
        except Exception:
            # If probabilities isn't a torch tensor, try to coerce
            try:
                return float(probabilities[0][-1])
            except Exception:
                return 0.0
