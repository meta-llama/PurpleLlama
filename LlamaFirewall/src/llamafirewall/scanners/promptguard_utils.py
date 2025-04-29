# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os

import torch
from huggingface_hub import HfFolder, login

from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.nn import Module
from torch.nn.functional import softmax

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)

LOG: logging.Logger = logging.getLogger(__name__)


class PromptGuard:
    def __init__(self) -> None:
        self.model: Module
        self.tokenizer: PreTrainedTokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()

        if self.model:
            self.device: DeviceLikeType = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)

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
                login()
            # Load the model and tokenizer from Hugging Face
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Save the model and tokenizer locally
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        else:
            # Load the model and tokenizer from the local path
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        with torch.no_grad():
            logits: Tensor = self.model(**inputs).logits
        scaled_logits: Tensor = logits / temperature
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities

    def get_jailbreak_score(
        self,
        text: str,
        temperature: float = 1.0,
        preprocess: bool = True,
    ) -> float:
        probabilities = self._get_class_probabilities(text, temperature, preprocess)
        return probabilities[0, -1].item()
