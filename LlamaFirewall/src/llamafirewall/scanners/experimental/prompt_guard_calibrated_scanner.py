# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import numpy as np
import torch

from ...llamafirewall_data_types import (
    Message,
    ScanDecision,
    ScanResult,
    ScanStatus,
    Trace,
)
from ..base_scanner import Scanner
from ..promptguard_utils import PromptGuard

LOG: logging.Logger = logging.getLogger(__name__)

NATIVE: str = "native"
CALIBRATED: str = "calibrated"


class PromptGuardCalibratedScanner(Scanner):
    """Prompt Guard 2 scanner with an optional caller-supplied calibrated head.

    The default (``score_mode="native"``) is identical to ``PromptGuardScanner``:
    it uses Prompt Guard 2's native classification head via
    ``PromptGuard.get_jailbreak_score`` and blocks at ``block_threshold``.

    In ``score_mode="calibrated"`` the base encoder is left frozen and the pooled
    (penultimate) embedding is scored by a caller-supplied linear head:
    ``sigmoid(embedding . w + b) >= tau``. This is useful when a deployment needs a
    more recall-oriented operating point than the precision-first native head. See
    https://github.com/meta-llama/PurpleLlama/issues/257 for background.

    The calibrated head is **caller-provided** and never shipped in this repository.
    Supply a local NumPy ``.npz`` via ``calibrated_head_path`` containing:
      - ``w``   : 1-D float array, length == encoder hidden size (aliases: ``coef``)
      - ``b``   : scalar float bias                              (aliases: ``intercept``)
      - ``tau`` : scalar float decision threshold in [0, 1]      (aliases: ``threshold``)

    The caller is responsible for training ``w``/``b`` on the same embedding this
    scanner computes (the encoder's pooled output) and for the license/provenance of
    those weights. If ``score_mode="calibrated"`` but no valid head is available, the
    scanner raises at construction time, unless ``fallback_to_native=True``.
    """

    ALLOW_REASON: str = "No prompt injection detected"

    def __init__(
        self,
        block_threshold: float = 0.9,
        score_mode: str = NATIVE,
        calibrated_head_path: str | None = None,
        fallback_to_native: bool = False,
    ) -> None:
        super().__init__(
            scanner_name="Prompt Guard Calibrated Scanner",
            block_threshold=block_threshold,
        )
        if score_mode not in (NATIVE, CALIBRATED):
            raise ValueError(
                f"score_mode must be '{NATIVE}' or '{CALIBRATED}', got {score_mode!r}"
            )
        self.pg = PromptGuard()
        self.score_mode: str = score_mode
        self._w: "np.ndarray | None" = None
        self._b: float = 0.0
        self._tau: float = block_threshold

        if self.score_mode == CALIBRATED:
            try:
                if calibrated_head_path is None:
                    raise ValueError(
                        "score_mode='calibrated' requires calibrated_head_path "
                        "(a local .npz with w/b/tau)."
                    )
                self._load_calibrated_head(calibrated_head_path)
            except Exception as e:
                if fallback_to_native:
                    LOG.warning(
                        "[LlamaFirewall] Calibrated head unavailable (%s); "
                        "falling back to native scoring.",
                        e,
                    )
                    self.score_mode = NATIVE
                else:
                    raise

    def _load_calibrated_head(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)

        def pick(*keys: str) -> "np.ndarray":
            for key in keys:
                if key in data:
                    return np.asarray(data[key])
            raise KeyError(
                f"Calibrated head file {path!r} is missing one of {keys!r}."
            )

        w = pick("w", "coef").astype(np.float64).reshape(-1)
        b = float(pick("b", "intercept").astype(np.float64).reshape(-1)[0])
        tau = float(pick("tau", "threshold").astype(np.float64).reshape(-1)[0])
        if w.size == 0:
            raise ValueError(f"Calibrated head weight vector in {path!r} is empty.")
        self._w = w
        self._b = b
        self._tau = tau

    def _embedding(self, text: str) -> "np.ndarray":
        """Pooled (penultimate) embedding from the frozen encoder."""
        inputs = self.pg.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.pg.device)
        with torch.no_grad():
            encoder = getattr(self.pg.model, "deberta", None) or self.pg.model.base_model
            hidden = encoder(**inputs).last_hidden_state
            pooler = getattr(self.pg.model, "pooler", None)
            pooled = pooler(hidden) if pooler is not None else hidden[:, 0]
        return pooled[0].detach().cpu().numpy().astype(np.float64)

    def _calibrated_score(self, text: str) -> float:
        assert self._w is not None
        emb = self._embedding(text)
        if emb.shape[0] != self._w.shape[0]:
            raise ValueError(
                f"Embedding dim {emb.shape[0]} != calibrated head dim "
                f"{self._w.shape[0]}."
            )
        z = float(emb @ self._w) + self._b
        return float(1.0 / (1.0 + np.exp(-z)))

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content
        try:
            if self.score_mode == CALIBRATED:
                score = self._calibrated_score(text)
                threshold = self._tau
            else:
                score = self.pg.get_jailbreak_score(text=text)
                threshold = self.block_threshold
        except Exception as e:  # noqa: BLE001 - one bad input must not crash the caller
            LOG.error("[LlamaFirewall] Calibrated scan failed: %s", e)
            return ScanResult(
                decision=ScanDecision.ALLOW,
                reason=f"Scan error: {e}",
                score=0.0,
                status=ScanStatus.ERROR,
            )

        if score >= threshold:
            decision = ScanDecision.BLOCK
            reason = (
                "Received text is likely to be a prompt injection attack, with a "
                f'probability of {score} (mode={self.score_mode}).\n Full text: "{text}"'
            )
        else:
            decision = ScanDecision.ALLOW
            reason = self.ALLOW_REASON

        return ScanResult(
            decision=decision,
            reason=reason,
            score=score,
            status=ScanStatus.SUCCESS,
        )
