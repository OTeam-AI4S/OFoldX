from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.trunk.msa import MsaPairWeightedAveraging, OuterProductMean


@pytest.mark.parametrize("divide_before_projection", [False, True])
def test_outer_product_mean(divide_before_projection: bool) -> None:
    torch.manual_seed(7)
    module = OuterProductMean(
        msa_size=4,
        hidden_size=2,
        pair_size=5,
        operator_backend="eager",
        divide_before_projection=divide_before_projection,
    )
    with torch.no_grad():  # non-zero bias so the divide/project branches place it differently
        module.output_projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5]))
    batch, seqs, residues, msa_size = 1, 2, 3, 4
    msa = torch.randn(batch, seqs, residues, msa_size)
    # mask off residue 2 of sequence 0 so pairs touching it see count 1 instead of 2
    msa_mask = torch.tensor([[[True, True, False], [True, True, True]]])

    output = module(msa, msa_mask=msa_mask)

    # expected recomputed from the outer-product-mean definition via F.* builtins (independent contraction)
    norm = module.norm_msa
    normalized = F.layer_norm(msa, (msa_size,), norm.weight, norm.bias, norm.eps)
    left = F.linear(normalized, module.left_projection.weight)
    right = F.linear(normalized, module.right_projection.weight)
    keep = msa_mask.to(msa.dtype)[..., None]
    left, right = left * keep, right * keep
    outer = torch.einsum("bsia,bsjc->bijac", left, right).reshape(batch, residues, residues, -1)
    count = torch.einsum("bsi,bsj->bij", keep[..., 0], keep[..., 0])[..., None]
    out_w, out_b = module.output_projection.weight, module.output_projection.bias
    if divide_before_projection:
        expected = F.linear(outer / count.clamp(min=1), out_w, out_b)
    else:
        expected = F.linear(outer, out_w, out_b) / (count + module.eps)

    assert output.shape == (1, 3, 3, 5)
    assert output.std() > 0.1  # non-degenerate, not a constant bias field
    torch.testing.assert_close(output, expected)


def test_pair_weighted_average() -> None:
    torch.manual_seed(3)
    module = MsaPairWeightedAveraging(msa_size=4, pair_size=5, num_attention_heads=2, operator_backend="eager")
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
    normalized = F.layer_norm(msa, (msa_size,), module.norm_msa.weight, module.norm_msa.bias, module.norm_msa.eps)
    values = F.linear(normalized, module.value_projection.weight).reshape(batch, seqs, residues, heads, head_dim)
    weights = pair_mask[0].to(values.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    averaged = torch.einsum("ij,bsjhd->bsihd", weights, values).reshape(batch, seqs, residues, msa_size)
    gated = averaged * torch.sigmoid(F.linear(normalized, module.gate_projection.weight))
    expected = F.linear(gated, module.output_projection.weight)

    assert output.shape == (1, 2, 3, 4)
    assert output.abs().max() > 1e-3  # real signal, not the all-zero degenerate case
    torch.testing.assert_close(output, expected)
