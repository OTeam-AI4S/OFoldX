from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.embedding import RelativePositionEmbedder, generate_relative_position_features


def features(
    residue_index, *, chain_index=None, entity_index=None, symmetry_index=None, cyclic_period=None, dtype=torch.float32
):
    r"""
    Relative-position buckets from a ``(1, N)`` residue index, defaulting metadata to a single chain.
    """
    zeros = torch.zeros_like(residue_index)
    return generate_relative_position_features(
        residue_index=residue_index,
        chain_index=zeros if chain_index is None else chain_index,
        entity_index=zeros if entity_index is None else entity_index,
        symmetry_index=zeros if symmetry_index is None else symmetry_index,
        token_index=residue_index,
        cyclic_period=cyclic_period,
        dtype=dtype,
    )


@pytest.fixture
def embedder():
    torch.manual_seed(0)
    return RelativePositionEmbedder(pair_size=8)


def test_metadata_buckets_and_projection(embedder):
    # Two two-residue chains; off-chain pairs hit the residue sentinel 2*32+1, off-entity pairs the chain
    # sentinel 2*2+1, and same-entity is the within-block gate. Defaults: max_relative_residue=32, _chain=2.
    idx = torch.tensor([[0, 1, 0, 1]])
    chain = torch.tensor([[0, 0, 1, 1]])
    buckets = features(idx, chain_index=chain, entity_index=chain, symmetry_index=chain)

    # residue_bucket = clamp(i-j+32, 0, 64) within a chain, sentinel 65 across chains.
    assert torch.equal(
        buckets.residue_bucket,
        torch.tensor([[[32, 31, 65, 65], [33, 32, 65, 65], [65, 65, 32, 31], [65, 65, 33, 32]]]),
    )
    # token_bucket falls to its sentinel 65 unless same chain AND same residue (here only the diagonal).
    assert torch.equal(
        buckets.token_bucket,
        torch.tensor([[[32, 65, 65, 65], [65, 32, 65, 65], [65, 65, 32, 65], [65, 65, 65, 32]]]),
    )
    # chain_bucket = clamp(sym_i-sym_j+2, 0, 4) for same entity, sentinel 5 otherwise.
    assert torch.equal(
        buckets.chain_bucket,
        torch.tensor([[[2, 2, 5, 5], [2, 2, 5, 5], [5, 5, 2, 2], [5, 5, 2, 2]]]),
    )
    assert torch.equal(buckets.same_entity, chain[..., :, None] == chain[..., None, :])
    assert (buckets.residue_size, buckets.chain_size) == (2 * (32 + 1), 2 * (2 + 1))

    out = embedder(buckets)
    assert out.shape == (1, 4, 4, 8)
    # Validate the gather fast path against the dense projection it stands in for (onehot(b) @ Wᵀ == Wᵀ[b]),
    # rebuilding the one-hot independently in input layout [residue | token | same_entity | chain].
    rs, cs = buckets.residue_size, buckets.chain_size
    dense = torch.cat(
        [
            F.one_hot(buckets.residue_bucket, rs).float(),
            F.one_hot(buckets.token_bucket, rs).float(),
            buckets.same_entity.unsqueeze(-1).float(),
            F.one_hot(buckets.chain_bucket, cs).float(),
        ],
        dim=-1,
    )
    # tol: float32 GEMM accumulation over a 139-wide one-hot, almost all zeros, so a few ulps.
    torch.testing.assert_close(out, embedder.output_projection(dense), atol=1e-5, rtol=1e-5)
    # The embedder carries the bucket dtype through to its output.
    for dtype in (torch.float16, torch.float32):
        assert embedder(features(torch.arange(3).unsqueeze(0), dtype=dtype)).dtype == dtype


@pytest.mark.parametrize("period", [None, 0.0, 5.0])
def test_cyclic_period(period):
    # residue_index [0, 5]: the raw offset 0-5=-5 bucketises to clamp(-5+32)=27 / clamp(5+32)=37 off-diagonal.
    idx = torch.tensor([[0, 5]])
    cyclic = None if period is None else torch.full((1, 2), period)
    buckets = features(idx, cyclic_period=cyclic)
    if period:
        # A period-5 chain wraps the 0<->5 offset to 0, collapsing both off-diagonals onto the diagonal bucket 32.
        expected = torch.tensor([[[32, 32], [32, 32]]])
    else:
        # None or all-zero period leaves the linear offset untouched.
        expected = torch.tensor([[[32, 27], [37, 32]]])
    assert torch.equal(buckets.residue_bucket, expected)
