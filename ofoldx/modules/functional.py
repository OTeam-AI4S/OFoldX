r"""
Functional tensor helpers for modules.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch import Tensor

from .backend._oops import oops_kernel

_oops_pair_mask_from_token_mask = oops_kernel("pair_mask_from_token_mask")
_oops_pair_mask_from_msa_mask = oops_kernel("pair_mask_from_msa_mask")


@torch.library.custom_op("ofoldx::_packed_pair_mask", mutates_args=())
def _packed_pair_mask(token_values: Tensor, token_offsets: Tensor) -> Tensor:
    r"""Build packed square pair values from packed tokens and device row splits."""

    lengths = token_offsets[1:] - token_offsets[:-1]
    samples = torch.arange(lengths.shape[0], device=token_offsets.device, dtype=torch.long)
    token_samples = torch.repeat_interleave(samples, lengths)
    row_values = torch.repeat_interleave(token_values, lengths.index_select(0, token_samples))
    pair_lengths = lengths.square()
    pair_samples = torch.repeat_interleave(samples, pair_lengths)
    pair_offsets = F.pad(pair_lengths.cumsum(0), (1, 0))
    positions = torch.arange(pair_samples.shape[0], device=token_offsets.device, dtype=torch.long)
    local_positions = positions - pair_offsets.index_select(0, pair_samples)
    columns = torch.remainder(local_positions, lengths.index_select(0, pair_samples))
    column_indices = token_offsets.index_select(0, pair_samples) + columns
    return row_values & token_values.index_select(0, column_indices)


@torch.library.register_fake(_packed_pair_mask)
def _packed_pair_mask_fake(token_values: Tensor, token_offsets: Tensor) -> Tensor:
    del token_offsets
    output_size = torch.library.get_ctx().new_dynamic_size(min=0)
    return token_values.new_empty((output_size,), dtype=torch.bool)


def pair_mask_from_token_mask(token_mask: Tensor | NestedTensor) -> Tensor | NestedTensor:
    r"""
    Build a pair-validity mask by taking the outer conjunction of a token mask.

    A token pair ``(i, j)`` is valid only when both token ``i`` and token ``j`` are valid, so the
    result is ``token_mask[..., :, None] & token_mask[..., None, :]``.

    Args:
        token_mask: `[..., N_token]` token-validity mask; any truthy dtype is accepted.

    Returns:
        `[..., N_token, N_token]` boolean pair-validity mask.
    """

    if isinstance(token_mask, NestedTensor):
        if (
            not token_mask.batch_first
            or token_mask.dim() != 2
            or token_mask.ragged_dims != (0,)
            or token_mask.packed_dim_order != (0,)
        ):
            raise ValueError("nested token_mask elements must use explicit canonical (N_i,) packing")
        token_values = token_mask.concat.bool()
        if token_values.is_cuda:
            return cast(NestedTensor, _oops_pair_mask_from_token_mask(token_mask))
        device_offsets = token_mask.packed_offsets(device=token_values.device, dtype=torch.int64)
        pair_values = _packed_pair_mask(token_values, device_offsets)
        cpu_offsets = token_mask.packed_offsets(dtype=torch.int64)
        return token_mask.packed_with_square_lengths(pair_values, cpu_offsets[1:] - cpu_offsets[:-1])

    token_mask = token_mask.bool()
    return token_mask[..., :, None] & token_mask[..., None, :]


def pair_mask_from_msa_mask(msa_mask: Tensor | NestedTensor) -> Tensor | NestedTensor:
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
    """

    if isinstance(msa_mask, NestedTensor):
        return cast(NestedTensor, _oops_pair_mask_from_msa_mask(msa_mask))
    token_mask = msa_mask.bool().any(dim=-2)
    return pair_mask_from_token_mask(token_mask)
