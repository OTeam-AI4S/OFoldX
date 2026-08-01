from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.trunk.msa import (
    MsaPairWeightedAveraging,
    MsaTrunk,
    OuterProductMean,
    PostProjectionOuterProductMean,
    TrianglePairUpdateStack,
)


def _trunk(msa_dropout: float = 0.0) -> MsaTrunk:
    return MsaTrunk(
        msa_size=4,
        pair_size=4,
        msa_vocab_size=3,
        token_feature_size=3,
        num_stages=2,
        outer_product_hidden_size=2,
        num_msa_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        msa_dropout=msa_dropout,
        pair_dropout=0.0,
        backend="eager",
    )


def _stack() -> TrianglePairUpdateStack:
    return TrianglePairUpdateStack(
        num_blocks=2,
        pair_size=4,
        triangle_multiplication_hidden_size=4,
        triangle_attention_hidden_size=2,
        num_triangle_attention_heads=2,
        dropout=0.0,
        backend="eager",
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


class TestMsaTrunk:
    def test_shapes_and_gradients(self) -> None:
        msa_tokens = torch.randint(3, (1, 2, 3))
        msa_continuous = torch.randn(1, 2, 3, 2, requires_grad=True)
        pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        token_embedding = torch.randn(1, 3, 3, requires_grad=True)

        output_msa, output_pair = _trunk()(msa_tokens, msa_continuous, pair, token_embedding)
        (output_msa.sum() + output_pair.sum()).backward()

        assert output_msa.shape == (1, 2, 3, 4)
        assert output_pair.shape == pair.shape
        assert msa_continuous.grad is not None
        assert pair.grad is not None
        assert token_embedding.grad is not None

    def test_gradient_checkpointing(self) -> None:
        eager = _trunk(msa_dropout=0.25)
        checkpointed = _trunk(msa_dropout=0.25)
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        msa_tokens = torch.randint(3, (1, 2, 3))
        eager_msa = torch.randn(1, 2, 3, 2, requires_grad=True)
        eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        eager_token = torch.randn(1, 3, 3, requires_grad=True)
        checkpointed_msa = eager_msa.detach().clone().requires_grad_()
        checkpointed_pair = eager_pair.detach().clone().requires_grad_()
        checkpointed_token = eager_token.detach().clone().requires_grad_()

        torch.manual_seed(13)
        eager_output = eager(msa_tokens, eager_msa, eager_pair, eager_token)
        torch.manual_seed(13)
        checkpointed_output = checkpointed(msa_tokens, checkpointed_msa, checkpointed_pair, checkpointed_token)
        sum(output.sum() for output in eager_output).backward()
        sum(output.sum() for output in checkpointed_output).backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_msa.grad, eager_msa.grad)
        torch.testing.assert_close(checkpointed_pair.grad, eager_pair.grad)
        torch.testing.assert_close(checkpointed_token.grad, eager_token.grad)

    def test_fullgraph(self) -> None:
        reference = _trunk()
        compiled_module = _trunk()
        compiled_module.load_state_dict(reference.state_dict())
        compiled_module.gradient_checkpointing = True
        compiled_module._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        compiled = torch.compile(compiled_module, backend="aot_eager", fullgraph=True, dynamic=True)

        for depth in (3, 5):
            msa_tokens = torch.randint(3, (1, depth, 3))
            eager_msa = torch.randn(1, depth, 3, 2, requires_grad=True)
            eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
            eager_token = torch.randn(1, 3, 3, requires_grad=True)
            compiled_msa = eager_msa.detach().clone().requires_grad_()
            compiled_pair = eager_pair.detach().clone().requires_grad_()
            compiled_token = eager_token.detach().clone().requires_grad_()

            expected = reference(msa_tokens, eager_msa, eager_pair, eager_token)
            actual = compiled(msa_tokens, compiled_msa, compiled_pair, compiled_token)
            sum(output.square().sum() for output in expected).backward()
            sum(output.square().sum() for output in actual).backward()

            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(compiled_msa.grad, eager_msa.grad)
            torch.testing.assert_close(compiled_pair.grad, eager_pair.grad)
            torch.testing.assert_close(compiled_token.grad, eager_token.grad)


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


class TestTrianglePairUpdateStack:
    def test_shape_and_gradient(self) -> None:
        pair = torch.randn(1, 3, 3, 4, requires_grad=True)

        output = _stack()(pair)
        output.sum().backward()

        assert output.shape == pair.shape
        assert pair.grad is not None

    def test_gradient_checkpointing(self) -> None:
        eager = _stack()
        checkpointed = _stack()
        checkpointed.load_state_dict(eager.state_dict())
        checkpointed.gradient_checkpointing = True
        checkpointed._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        eager_pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        checkpointed_pair = eager_pair.detach().clone().requires_grad_()

        eager_output = eager(eager_pair)
        checkpointed_output = checkpointed(checkpointed_pair)
        eager_output.sum().backward()
        checkpointed_output.sum().backward()

        torch.testing.assert_close(checkpointed_output, eager_output)
        torch.testing.assert_close(checkpointed_pair.grad, eager_pair.grad)
