r"""
AlphaFold 2 training objectives.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor


def masked_msa_criterion(pred: Any, target: Any) -> Tensor:
    r"""
    Score the released masked-MSA reconstruction head.

    AlphaFold 2 averages the token cross entropy within each example over the
    BERT mask, then averages the examples. The realized labels and mask belong to
    the output because MSA corruption is sampled inside the Model.
    """

    del target
    per_token = F.cross_entropy(
        pred.masked_msa_logits.movedim(-1, 1),
        pred.true_msa,
        reduction="none",
    )
    mask = pred.bert_mask.to(per_token.dtype)
    per_example = (per_token * mask).sum(dim=(-2, -1)) / (1e-8 + mask.sum(dim=(-2, -1)))
    return per_example.mean()
