r"""
Functional tensor helpers for modules.
"""

from __future__ import annotations

from torch import Tensor


def pair_mask_from_token_mask(token_mask: Tensor) -> Tensor:
    r"""
    Build a pair-validity mask by taking the outer conjunction of a token mask.

    A token pair ``(i, j)`` is valid only when both token ``i`` and token ``j`` are valid, so the
    result is ``token_mask[..., :, None] & token_mask[..., None, :]``.

    Args:
        token_mask: `[..., N_token]` token-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask.

    Examples:
        >>> import torch
        >>> token_mask = torch.tensor([[1, 1, 0]])
        >>> pair_mask_from_token_mask(token_mask).int().tolist()
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]
    """

    token_mask = token_mask.bool()
    return token_mask[..., :, None] & token_mask[..., None, :]


def pair_mask_from_msa_mask(msa_mask: Tensor) -> Tensor:
    r"""
    Build a token-pair mask from an MSA mask, keeping tokens with any valid MSA row.

    A token is treated as valid when at least one MSA row covers it (``any`` over the MSA axis);
    the per-token mask is then expanded into a pair mask via
    [pair_mask_from_token_mask][ofoldx.modules.functional.pair_mask_from_token_mask].

    Args:
        msa_mask: `[..., N_msa, N_token]` MSA-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask for tokens with at least one valid
        MSA row.

    Examples:
        >>> import torch
        >>> msa_mask = torch.tensor([[[1, 0, 0], [0, 1, 0]]])
        >>> pair_mask_from_msa_mask(msa_mask).int().tolist()
        [[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]
    """

    token_mask = msa_mask.bool().any(dim=-2)
    return pair_mask_from_token_mask(token_mask)
