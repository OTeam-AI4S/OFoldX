from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.trunk.msa import (
    MsaPairWeightedAveraging,
    MsaTrunk,
    OuterProductMean,
    PostProjectionOuterProductMean,
    TrianglePairUpdateStack,
)


@pytest.mark.parametrize(
    ("module_type", "normalize_before_projection", "denominator_epsilon"),
    [
        pytest.param(OuterProductMean, True, 0.0, id="pre-projection"),
        pytest.param(PostProjectionOuterProductMean, False, 1e-3, id="post-projection"),
    ],
)
def test_outer_product_mean_equation(
    module_type: type[OuterProductMean] | type[PostProjectionOuterProductMean],
    normalize_before_projection: bool,
    denominator_epsilon: float,
) -> None:
    torch.manual_seed(7)
    module = module_type(
        msa_size=4,
        hidden_size=2,
        pair_size=5,
        backend="eager",
    )
    with torch.no_grad():  # non-zero bias so the divide/project branches place it differently
        module.output_projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5]))
    batch, seqs, residues, msa_size = 1, 2, 3, 4
    msa = torch.randn(batch, seqs, residues, msa_size)
    # mask off residue 2 of sequence 0 so pairs touching it see count 1 instead of 2
    msa_mask = torch.tensor([[[True, True, False], [True, True, True]]])

    output = module(msa, msa_mask=msa_mask)

    # expected recomputed from the outer-product-mean definition via F.* builtins (independent contraction)
    norm = module.msa_norm
    normalized = F.layer_norm(msa, (msa_size,), norm.weight, norm.bias, norm.eps)
    left = F.linear(normalized, module.left_projection.weight)
    right = F.linear(normalized, module.right_projection.weight)
    keep = msa_mask.to(msa.dtype)[..., None]
    left, right = left * keep, right * keep
    outer = torch.einsum("bsia,bsjc->bijac", left, right).reshape(batch, residues, residues, -1)
    count = torch.einsum("bsi,bsj->bij", keep[..., 0], keep[..., 0])[..., None]
    out_w, out_b = module.output_projection.weight, module.output_projection.bias
    if normalize_before_projection:
        expected = F.linear(outer / count.clamp(min=1), out_w, out_b)
    else:
        expected = F.linear(outer, out_w, out_b) / (count + denominator_epsilon)

    torch.testing.assert_close(output, expected)


def test_msa_trunk_one_block_runs_only_final_pair_update() -> None:
    torch.manual_seed(11)
    trunk = MsaTrunk(
        msa_size=4,
        pair_size=4,
        msa_feature_size=5,
        token_feature_size=3,
        num_blocks=1,
        outer_product_hidden_size=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        msa_dropout=0.0,
        pair_dropout=0.0,
        backend="eager",
    )
    msa_features = torch.randn(1, 2, 3, 5)
    token_embedding = torch.randn(1, 3, 3)
    pair = torch.randn(1, 3, 3, 4)
    msa_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    pair_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    projected_msa = trunk.msa_projection(msa_features) + trunk.token_projection(token_embedding).unsqueeze(1)
    assert trunk.final_pair_update is not None
    expected_pair = trunk.final_pair_update(
        projected_msa,
        pair,
        msa_mask=msa_mask,
        pair_mask=pair_mask,
    )

    msa, updated_pair = trunk(
        msa_features,
        pair,
        token_embedding,
        msa_mask=msa_mask,
        pair_mask=pair_mask,
    )

    torch.testing.assert_close(msa, projected_msa)
    torch.testing.assert_close(updated_pair, expected_pair)


def test_pair_weighted_average() -> None:
    torch.manual_seed(3)
    module = MsaPairWeightedAveraging(msa_size=4, pair_size=5, num_attention_heads=2, backend="eager")
    # zero the pair projection so logits vanish and softmax-over-valid-j becomes a hand-computable uniform average
    with torch.no_grad():
        module.pair_projection.weight.zero_()
    batch, seqs, residues, msa_size, heads = 1, 2, 3, 4, 2
    head_dim = msa_size // heads
    msa = torch.randn(batch, seqs, residues, msa_size)
    pair = torch.randn(batch, residues, residues, 5)
    # distinct per-row valid sets: row 0 -> {0,1}, row 1 -> {1,2}, row 2 -> {0,2}
    pair_mask = torch.tensor([[[True, True, False], [False, True, True], [True, False, True]]])

    output = module(msa, pair, pair_mask=pair_mask)

    # expected from the PWA definition with weights derived by hand (uniform over valid j, no softmax replay)
    normalized = F.layer_norm(msa, (msa_size,), module.msa_norm.weight, module.msa_norm.bias, module.msa_norm.eps)
    values = F.linear(normalized, module.value_projection.weight).reshape(batch, seqs, residues, heads, head_dim)
    weights = pair_mask[0].to(values.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    averaged = torch.einsum("ij,bsjhd->bsihd", weights, values).reshape(batch, seqs, residues, msa_size)
    gated = averaged * torch.sigmoid(F.linear(normalized, module.gate_projection.weight))
    expected = F.linear(gated, module.output_projection.weight)

    torch.testing.assert_close(output, expected)


def test_triangle_pair_update_stack() -> None:
    pair = torch.randn(1, 3, 3, 4)
    stack = TrianglePairUpdateStack(
        num_blocks=2,
        pair_size=4,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        dropout=0.0,
        backend="eager",
    )
    expected = pair
    for block in stack.blocks:
        expected = block(expected)

    output = stack(pair)

    torch.testing.assert_close(output, expected)
