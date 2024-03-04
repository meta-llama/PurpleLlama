# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings

import sacrebleu

warnings.filterwarnings("ignore")


def compute_bleu_score(hyp: str, ref: str) -> float:
    """Compute BLEU score between two strings using SacreBleu."""
    # Compute and return the BLEU score using SacreBleu
    return sacrebleu.corpus_bleu(
        [hyp],
        [[ref]],
        smooth_method="exp",
        force=False,
        lowercase=False,
        use_effective_order=False,
    ).score
