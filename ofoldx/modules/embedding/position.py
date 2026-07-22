r"""
Relative-position embedding modules.

Encode the pairwise relationship between tokens (relative residue / token / chain offset and
whether two tokens share an entity) as integer buckets, then embed those buckets into the pair
representation. Bucketing is factored out from projection so the embedder can gather rows of its
weight instead of materialising the dense ``[..., N, N, input_size]`` one-hot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ChainGate = Literal["same_entity", "not_same_chain"]


@dataclass
class RelativePositionBuckets:
    r"""
    The integer relative-position buckets that index the relative-position embedding tables.

    Because ``onehot(b) @ Wᵀ == Wᵀ[b]``, the embedder gathers the relevant rows of its single projection weight
    instead of materialising the dense ``(..., N, N, input_size)`` one-hot. This carries the buckets, the one-hot
    block widths needed to slice that weight, and the activation dtype for the gathered output.

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
    residue_size: int  # width of each of the residue & token one-hot blocks: 2 * (max_relative_residue + 1)
    chain_size: int  # width of the chain one-hot block: 2 * (max_relative_chain + 1)
    dtype: torch.dtype  # dtype of the embedder output (the buckets themselves stay integer)


class RelativePositionEmbedder(nn.Module):
    r"""
    Project relative-position buckets into pair representations.

    Gathers rows of the single projection weight (one block per bucket type) rather than projecting a dense
    one-hot — identical result (``onehot(b) @ Wᵀ == Wᵀ[b]``) without the ``(..., N, N, input_size)`` materialisation.

    Args:
        input_size: Total width of the conceptual relative-position one-hot feature
            (``2 * residue_size + 1 + chain_size``; ``139`` for the default bucket ranges).
        pair_size: Output pair representation width.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> index = torch.arange(4).unsqueeze(0)
        >>> zeros = torch.zeros(1, 4, dtype=torch.long)
        >>> buckets = generate_relative_position_features(
        ...     residue_index=index,
        ...     chain_index=zeros,
        ...     entity_index=zeros,
        ...     symmetry_index=zeros,
        ...     token_index=index,
        ...     max_relative_residue=2,
        ...     max_relative_chain=1,
        ...     dtype=torch.float32,
        ... )
        >>> input_size = 2 * buckets.residue_size + 1 + buckets.chain_size
        >>> embedder = RelativePositionEmbedder(input_size=input_size, pair_size=8)
        >>> embedder(buckets).shape
        torch.Size([1, 4, 4, 8])
    """

    def __init__(self, input_size: int = 139, pair_size: int = 128) -> None:
        super().__init__()
        self.output_projection = nn.Linear(input_size, pair_size, bias=False)

    def forward(self, buckets: RelativePositionBuckets) -> Tensor:
        r"""
        Project relative-position buckets into pair features.

        Args:
            buckets: Relative-position bucket container produced by
                [`generate_relative_position_features`][ofoldx.modules.embedding.position.generate_relative_position_features].

        Returns:
            ``[..., N_token, N_token, pair_size]`` relative-position encoding, cast to ``buckets.dtype``.

        Examples:
            >>> import torch
            >>> _ = torch.manual_seed(0)
            >>> index = torch.arange(3).unsqueeze(0)
            >>> zeros = torch.zeros(1, 3, dtype=torch.long)
            >>> buckets = generate_relative_position_features(
            ...     residue_index=index,
            ...     chain_index=zeros,
            ...     entity_index=zeros,
            ...     symmetry_index=zeros,
            ...     token_index=index,
            ...     max_relative_residue=2,
            ...     max_relative_chain=1,
            ...     dtype=torch.float32,
            ... )
            >>> input_size = 2 * buckets.residue_size + 1 + buckets.chain_size
            >>> RelativePositionEmbedder(input_size=input_size, pair_size=8)(buckets).shape
            torch.Size([1, 3, 3, 8])
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
    Build the pairwise relative-position buckets from token metadata.

    When ``cyclic_period`` (shape ``(..., N)``) is supplied, positive entries wrap the raw residue
    offset modulo the period before bucketing. Zero entries leave the linear offset unchanged. Only same-chain
    pairs survive the chain gate, and a cyclic chain has a constant period, so broadcasting the period over the
    key axis matches the per-pair period for every pair that counts.

    ``chain_gate`` selects which pairs keep their real ``chain_offset`` bucket (the rest get the out-of-range
    sentinel): ``"same_entity"`` (default) keeps same-entity pairs, ``"not_same_chain"`` (the Boltz-1 variant)
    keeps cross-chain pairs instead.

    Args:
        residue_index: ``[..., N_token]`` residue indices.
        chain_index: ``[..., N_token]`` chain indices.
        entity_index: ``[..., N_token]`` entity indices.
        symmetry_index: ``[..., N_token]`` symmetry/copy indices.
        token_index: ``[..., N_token]`` token indices.
        cyclic_period: Optional ``[..., N_token]`` cyclic period per token.
        max_relative_residue: Maximum residue offset bucket.
        max_relative_chain: Maximum chain offset bucket.
        chain_gate: Which pairs keep the real chain-offset bucket (see above).
        dtype: Desired dtype of the eventual embedding output.

    Returns:
        Relative-position bucket container holding the residue, token, chain, and same-entity buckets
        plus the one-hot block widths and output dtype.

    Examples:
        >>> import torch
        >>> index = torch.arange(4).unsqueeze(0)
        >>> zeros = torch.zeros(1, 4, dtype=torch.long)
        >>> buckets = generate_relative_position_features(
        ...     residue_index=index,
        ...     chain_index=zeros,
        ...     entity_index=zeros,
        ...     symmetry_index=zeros,
        ...     token_index=index,
        ...     max_relative_residue=2,
        ...     max_relative_chain=1,
        ...     dtype=torch.float32,
        ... )
        >>> buckets.residue_bucket.tolist()
        [[[2, 1, 0, 0], [3, 2, 1, 0], [4, 3, 2, 1], [4, 4, 3, 2]]]
        >>> buckets.residue_size, buckets.chain_size
        (6, 4)
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
