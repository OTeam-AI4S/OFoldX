r"""
Relative-position embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ChainGate = Literal["same_entity", "not_same_chain"]


@dataclass(eq=False)
class RelativePositionBuckets:
    r"""
    Integer buckets consumed by [`RelativePositionEmbedder`][].

    Attributes:
        residue_bucket: Relative residue-distance bucket ids.
        token_bucket: Relative token-distance bucket ids.
        same_entity: Boolean bucket indicating whether token pairs share an entity.
        chain_bucket: Relative chain-distance bucket ids.
        residue_size: Width of each residue/token one-hot block.
        chain_size: Width of the chain one-hot block.
        dtype: Output dtype used when embedding the buckets.
    """

    residue_bucket: Tensor
    token_bucket: Tensor
    same_entity: Tensor
    chain_bucket: Tensor
    residue_size: int
    chain_size: int
    dtype: torch.dtype


class RelativePositionEmbedder(nn.Module):
    r"""
    Project relative-position buckets into pair representations.

    Args:
        input_size: Total width of the conceptual relative-position one-hot feature
            (``2 * residue_size + 1 + chain_size``; ``139`` for the default bucket ranges).
        pair_size: Output pair representation width.
    """

    def __init__(self, input_size: int = 139, pair_size: int = 128) -> None:
        super().__init__()
        self.output_projection = nn.Linear(input_size, pair_size, bias=False)

    def forward(self, buckets: RelativePositionBuckets) -> Tensor:
        r"""
        Project relative-position buckets into pair features.

        Args:
            buckets: Relative-position bucket container produced by
                ``generate_relative_position_features``.

        Returns:
            ``[..., N_token, N_token, pair_size]`` relative-position features in
            ``buckets.dtype``.
        """

        weight = self.output_projection.weight  # (pair_size, input_size = 2 * residue_size + 1 + chain_size)
        residue_size, chain_size = buckets.residue_size, buckets.chain_size
        residue_table = weight[:, 0:residue_size].t()
        token_table = weight[:, residue_size : 2 * residue_size].t()
        chain_start = 2 * residue_size + 1
        chain_table = weight[:, chain_start : chain_start + chain_size].t()
        encoding = (
            residue_table[buckets.residue_bucket]
            + token_table[buckets.token_bucket]
            + chain_table[buckets.chain_bucket]
            + buckets.same_entity.unsqueeze(-1).to(weight.dtype) * weight[:, 2 * residue_size]
        )
        return encoding.to(buckets.dtype)


def generate_relative_position_features(
    *,
    residue_index: Tensor,
    chain_index: Tensor,
    entity_index: Tensor,
    symmetry_index: Tensor,
    token_index: Tensor,
    cyclic_period: Tensor | None = None,
    max_relative_residue: int = 32,
    max_relative_chain: int = 2,
    chain_gate: ChainGate = "same_entity",
    dtype: torch.dtype,
) -> RelativePositionBuckets:
    r"""
    Build pairwise relative-position buckets from token metadata.

    Residue offsets are retained only within a chain. Positive ``cyclic_period``
    values wrap offsets using the key token's period, broadcast across the query
    axis. ``chain_gate`` controls which pairs retain chain-offset buckets.

    Args:
        residue_index: Residue indices with shape ``(..., num_tokens)``.
        chain_index: Chain indices with shape ``(..., num_tokens)``.
        entity_index: Entity indices with shape ``(..., num_tokens)``.
        symmetry_index: Symmetry-copy indices with shape ``(..., num_tokens)``.
        token_index: Token indices with shape ``(..., num_tokens)``.
        cyclic_period: Optional per-token cyclic periods with shape
            ``(..., num_tokens)``; non-positive values leave offsets linear.
        max_relative_residue: Largest signed residue or token offset represented
            without clipping.
        max_relative_chain: Largest signed chain offset represented without clipping.
        chain_gate: ``"same_entity"`` retains chain offsets within an entity;
            ``"not_same_chain"`` retains offsets between different chains.
        dtype: Floating-point dtype recorded for the resulting embedding.

    Returns:
        Residue, token, chain, and entity buckets for every token pair.

    Raises:
        ValueError: If ``chain_gate`` is unsupported.
    """

    same_chain = chain_index[..., :, None] == chain_index[..., None, :]
    same_residue = residue_index[..., :, None] == residue_index[..., None, :]
    same_entity = entity_index[..., :, None] == entity_index[..., None, :]

    residue_difference = residue_index[..., :, None] - residue_index[..., None, :]
    if cyclic_period is not None:
        period = cyclic_period[..., None, :]
        cyclic = period > 0
        safe_period = torch.where(cyclic, period, torch.ones_like(period))
        wrapped = (residue_difference - safe_period * torch.round(residue_difference / safe_period)).long()
        residue_difference = torch.where(cyclic, wrapped, residue_difference)
    residue_offset = torch.clamp(
        residue_difference + max_relative_residue,
        min=0,
        max=2 * max_relative_residue,
    )
    residue_bucket = torch.where(
        same_chain,
        residue_offset,
        residue_offset.new_full((), 2 * max_relative_residue + 1),
    )

    token_offset = torch.clamp(
        token_index[..., :, None] - token_index[..., None, :] + max_relative_residue,
        min=0,
        max=2 * max_relative_residue,
    )
    token_bucket = torch.where(
        same_chain & same_residue,
        token_offset,
        token_offset.new_full((), 2 * max_relative_residue + 1),
    )

    chain_offset = torch.clamp(
        symmetry_index[..., :, None] - symmetry_index[..., None, :] + max_relative_chain,
        min=0,
        max=2 * max_relative_chain,
    )
    if chain_gate == "same_entity":
        chain_survives = same_entity
    elif chain_gate == "not_same_chain":
        chain_survives = ~same_chain
    else:
        raise ValueError(f"unknown chain_gate: {chain_gate!r}")
    chain_bucket = torch.where(
        chain_survives,
        chain_offset,
        chain_offset.new_full((), 2 * max_relative_chain + 1),
    )

    return RelativePositionBuckets(
        residue_bucket=residue_bucket,
        token_bucket=token_bucket,
        same_entity=same_entity,
        chain_bucket=chain_bucket,
        residue_size=2 * (max_relative_residue + 1),
        chain_size=2 * (max_relative_chain + 1),
        dtype=dtype,
    )
