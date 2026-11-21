# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import os

import numpy as np

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

    ``score_mode="native"`` (default) is identical to ``PromptGuardScanner``: it uses
    Prompt Guard 2's native head via ``PromptGuard.get_jailbreak_score`` and blocks at
    ``block_threshold``. Exceptions propagate exactly as they do there (no fail-open).

    ``score_mode="calibrated"`` leaves the base encoder frozen and scores the pooled
    (penultimate) embedding with a caller-supplied linear head:
    ``sigmoid(embedding . w + b) >= tau``. This suits deployments that want a more
    recall-oriented operating point than the precision-first native head. See
    https://github.com/meta-llama/PurpleLlama/issues/257 for background.

    Behavior notes:
      - In calibrated mode the decision uses ``tau`` from the head file; the
        ``block_threshold`` argument is ignored (it only applies to native mode).
      - The calibrated path tokenizes the raw string and does NOT call
        ``_preprocess_text_for_promptguard`` (it uses
        ``PromptGuard.get_pooled_embedding(..., preprocess=False)``). A caller training
        ``w``/``b`` must match that. Native ``get_jailbreak_score`` preprocesses by
        default; the two paths therefore see slightly different tokenizations.

    The calibrated head is **caller-provided** and never shipped in this repository.
    Supply a local NumPy ``.npz`` via ``calibrated_head_path`` containing:
      - ``w``   : 1-D float array, length == encoder hidden size (alias: ``coef``)
      - ``b``   : scalar float bias                              (alias: ``intercept``)
      - ``tau`` : scalar float decision threshold in [0, 1]      (alias: ``threshold``)

    Failure policy: if ``score_mode="calibrated"`` and the file is missing, construction
    raises ``FileNotFoundError`` (or, with ``fallback_to_native=True``, degrades to
    native with a warning). A file that exists but is corrupt/invalid always raises,
    even with ``fallback_to_native=True`` — silently downgrading a caller who asked for
    recall mode is exactly the failure this feature exists to avoid.
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
        self._w: np.ndarray | None = None
        self._b: float = 0.0
        self._tau: float = block_threshold

        if self.score_mode == CALIBRATED:
            # Only a MISSING file is a fallback trigger; a present-but-broken file
            # always fails clear so a caller in recall mode is never silently
            # downgraded to the precision-first native head.
            if calibrated_head_path is None or not os.path.exists(calibrated_head_path):
                if fallback_to_native:
                    LOG.warning(
                        "[LlamaFirewall] Calibrated head not found at %r; "
                        "falling back to native scoring.",
                        calibrated_head_path,
                    )
                    self.score_mode = NATIVE
                else:
                    raise FileNotFoundError(
                        "score_mode='calibrated' requires an existing "
                        f"calibrated_head_path (.npz with w/b/tau); got "
                        f"{calibrated_head_path!r}."
                    )
            else:
                self._load_calibrated_head(calibrated_head_path)

    def _load_calibrated_head(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)

        def pick(*keys: str) -> np.ndarray:
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

    def _calibrated_score(self, text: str) -> float:
        assert self._w is not None
        emb = self.pg.get_pooled_embedding(text, preprocess=False)
        if emb.shape[0] != self._w.shape[0]:
            raise ValueError(
                f"Embedding dim {emb.shape[0]} != calibrated head dim "
                f"{self._w.shape[0]}."
            )
        z = float(emb @ self._w) + self._b
        # Clip to avoid overflow warnings in exp() for large |z|; the sigmoid is
        # saturated well before this range so the decision is unaffected.
        z = max(-30.0, min(30.0, z))
        return float(1.0 / (1.0 + np.exp(-z)))

    async def scan(
        self, message: Message, past_trace: Trace | None = None
    ) -> ScanResult:
        text = message.content

        if self.score_mode == CALIBRATED:
            score = self._calibrated_score(text)
            threshold = self._tau
        else:
            score = self.pg.get_jailbreak_score(text=text)
            threshold = self.block_threshold

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
