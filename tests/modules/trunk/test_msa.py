from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.functional import pair_mask_from_msa_mask
from ofoldx.modules.trunk.msa import (
    MsaPairWeightedAveraging,
    MsaTrunk,
    OuterProductMean,
    PostProjectionOuterProductMean,
    TrianglePairUpdateStack,
)


def _trunk(msa_dropout: float = 0.0, backend: str = "eager") -> MsaTrunk:
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
        backend=backend,
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
    def test_nested_vertical_matches_independent_dense_and_reuses_one_graph(self) -> None:
        pytest.importorskip("oops")
        torch.manual_seed(20260828)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        module = _trunk(backend="oops").to(device=device, dtype=dtype)
        module.gradient_checkpointing = True
        module._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        reference = _trunk(backend="eager").to(device=device, dtype=dtype)
        reference.load_state_dict(module.state_dict())
        counter = CompileCounter()

        def run(msa_tokens, msa_continuous, pair, token_embedding, msa_mask, pair_mask):
            output_msa, output_pair = module(
                msa_tokens,
                msa_continuous,
                pair,
                token_embedding,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )
            return output_msa.concat, output_pair.concat

        compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)

        def packed(shapes, *, value_dtype=dtype, requires_grad=False):
            ragged_dims = tuple(range(len(shapes[0]) - (0 if value_dtype is torch.long else 1)))
            template = NestedTensor(
                [torch.empty(shape, dtype=value_dtype) for shape in shapes],
                ragged_dims=ragged_dims,
            )
            if value_dtype is torch.long:
                values = torch.randint(0, 3, template.concat.shape, device=device)
            else:
                values = torch.randn(
                    template.concat.shape,
                    device=device,
                    dtype=value_dtype,
                    requires_grad=requires_grad,
                )
            return template.packed_like(values), values

        def masks(layout):
            msa_parts = []
            pair_parts = []
            for msa_length, token_length in layout:
                msa_mask = torch.ones(msa_length, token_length, device=device, dtype=torch.bool)
                msa_mask[0] = False
                if msa_length > 1:
                    msa_mask[1, 1::2] = False
                pair_mask = torch.ones(token_length, token_length, device=device, dtype=torch.bool)
                pair_mask[0] = False
                if token_length > 1:
                    pair_mask[1, ::2] = False
                msa_parts.append(msa_mask)
                pair_parts.append(pair_mask)
            return (
                NestedTensor(msa_parts, ragged_dims=(0, 1)),
                NestedTensor(pair_parts, ragged_dims=(0, 1)),
            )

        def guard():
            return nested_execution_guard(
                forbid_iteration=True,
                forbid_storage_map=True,
                forbid_eager_fallback=True,
                forbid_padded_materialization=True,
                forbid_dense_repack=True,
            )

        for layout in (((2, 3), (4, 5)), ((3, 4), (2, 6))):
            msa_tokens, _ = packed(tuple((msa, tokens) for msa, tokens in layout), value_dtype=torch.long)
            msa_continuous, msa_values = packed(
                tuple((msa, tokens, 2) for msa, tokens in layout),
                requires_grad=True,
            )
            pair, pair_values = packed(
                tuple((tokens, tokens, 4) for _msa, tokens in layout),
                requires_grad=True,
            )
            token_embedding, token_values = packed(
                tuple((tokens, 3) for _msa, tokens in layout),
                requires_grad=True,
            )
            msa_mask, pair_mask = masks(layout)

            expected_msa_parts = []
            expected_pair_parts = []
            for dense_tokens, dense_continuous, dense_pair, dense_token, dense_msa_mask, dense_pair_mask in zip(
                torch.unbind(msa_tokens),
                torch.unbind(msa_continuous),
                torch.unbind(pair),
                torch.unbind(token_embedding),
                torch.unbind(msa_mask),
                torch.unbind(pair_mask),
                strict=True,
            ):
                dense_msa, dense_pair_output = reference(
                    dense_tokens.unsqueeze(0),
                    dense_continuous.unsqueeze(0),
                    dense_pair.unsqueeze(0),
                    dense_token.unsqueeze(0),
                    msa_mask=dense_msa_mask.unsqueeze(0),
                    pair_mask=dense_pair_mask.unsqueeze(0),
                )
                expected_msa_parts.append(dense_msa.squeeze(0).reshape(-1, 4))
                expected_pair_parts.append(dense_pair_output.squeeze(0).reshape(-1, 4))
            expected_msa = torch.cat(expected_msa_parts)
            expected_pair = torch.cat(expected_pair_parts)

            with guard():
                actual_msa, actual_pair = compiled(
                    msa_tokens,
                    msa_continuous,
                    pair,
                    token_embedding,
                    msa_mask,
                    pair_mask,
                )
            assert torch.isfinite(actual_msa).all()
            assert torch.isfinite(actual_pair).all()
            torch.testing.assert_close(actual_msa, expected_msa, rtol=1e-1, atol=1e-1)
            torch.testing.assert_close(actual_pair, expected_pair, rtol=1e-1, atol=1e-1)

            actual_inputs = (msa_values, pair_values, token_values, *module.parameters())
            expected_inputs = (msa_values, pair_values, token_values, *reference.parameters())
            grad_msa = torch.randn_like(actual_msa)
            grad_pair = torch.randn_like(actual_pair)
            actual_gradients = torch.autograd.grad(
                (actual_msa, actual_pair),
                actual_inputs,
                (grad_msa, grad_pair),
                retain_graph=True,
                allow_unused=True,
            )
            expected_gradients = torch.autograd.grad(
                (expected_msa, expected_pair),
                expected_inputs,
                (grad_msa, grad_pair),
                allow_unused=True,
            )
            parameter_difference_square = 0.0
            parameter_reference_square = 0.0
            activation_max_absolute = 0.0
            activation_difference_square = 0.0
            activation_reference_square = 0.0
            for index, (actual_gradient, expected_gradient) in enumerate(
                zip(actual_gradients, expected_gradients, strict=True)
            ):
                assert (actual_gradient is None) == (expected_gradient is None)
                if actual_gradient is None or expected_gradient is None:
                    continue
                if index < 3:
                    difference = actual_gradient.float() - expected_gradient.float()
                    activation_max_absolute = max(activation_max_absolute, difference.abs().max().item())
                    activation_difference_square += difference.square().sum().item()
                    activation_reference_square += expected_gradient.float().square().sum().item()
                else:
                    difference = actual_gradient.float() - expected_gradient.float()
                    parameter_difference_square += difference.square().sum().item()
                    parameter_reference_square += expected_gradient.float().square().sum().item()
            parameter_scaled_l2 = parameter_difference_square**0.5 / max(parameter_reference_square**0.5, 1e-6)
            activation_scaled_l2 = activation_difference_square**0.5 / max(activation_reference_square**0.5, 1e-6)
            print(
                f"layout={layout} activation_max_abs={activation_max_absolute:.6g} "
                f"activation_scaled_l2={activation_scaled_l2:.6g} parameter_scaled_l2={parameter_scaled_l2:.6g}",
                flush=True,
            )
            assert activation_scaled_l2 < 3e-2
            assert parameter_scaled_l2 < 5e-3
        assert counter.frame_count == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
    def test_nested_derived_pair_mask_reuses_one_dynamic_fullgraph(self) -> None:
        pytest.importorskip("oops")
        torch.manual_seed(20260829)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        module = _trunk(backend="oops").to(device=device, dtype=dtype)
        reference = _trunk(backend="eager").to(device=device, dtype=dtype)
        reference.load_state_dict(module.state_dict())
        counter = CompileCounter()

        def run(msa_tokens, msa_continuous, pair, msa_mask):
            output_msa, output_pair = module(
                msa_tokens,
                msa_continuous,
                pair,
                msa_mask=msa_mask,
                pair_mask=None,
            )
            return output_msa.concat, output_pair.concat

        compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)

        def guard():
            return nested_execution_guard(
                forbid_iteration=True,
                forbid_storage_map=True,
                forbid_eager_fallback=True,
                forbid_padded_materialization=True,
                forbid_dense_repack=True,
            )

        for layout in (((2, 3), (4, 5)), ((3, 4), (2, 6))):
            token_template = NestedTensor(
                [torch.empty(msa, tokens, dtype=torch.long) for msa, tokens in layout],
                ragged_dims=(0, 1),
            )
            msa_tokens = token_template.packed_like(torch.randint(0, 3, token_template.concat.shape, device=device))
            continuous_template = NestedTensor(
                [torch.empty(msa, tokens, 2) for msa, tokens in layout],
                ragged_dims=(0, 1),
            )
            continuous_values = torch.randn(
                continuous_template.concat.shape,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            msa_continuous = continuous_template.packed_like(continuous_values)
            pair_template = NestedTensor(
                [torch.empty(tokens, tokens, 4) for _msa, tokens in layout],
                ragged_dims=(0, 1),
            )
            pair_values = torch.randn(
                pair_template.concat.shape,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            pair = pair_template.packed_like(pair_values)
            mask_parts = []
            for msa_length, token_length in layout:
                mask = torch.ones(msa_length, token_length, device=device, dtype=torch.bool)
                mask[0] = False
                mask[:, 0] = False
                if msa_length > 1:
                    mask[1, 1::2] = False
                mask_parts.append(mask)
            msa_mask = NestedTensor(mask_parts, ragged_dims=(0, 1))
            explicit_pair_mask = NestedTensor(
                [token_mask[:, None] & token_mask[None, :] for token_mask in (mask.any(dim=0) for mask in mask_parts)],
                ragged_dims=(0, 1),
            )
            with guard():
                derived_pair_mask = pair_mask_from_msa_mask(msa_mask)
            assert isinstance(derived_pair_mask, NestedTensor)
            assert torch.equal(derived_pair_mask.concat, explicit_pair_mask.concat)
            assert torch.equal(derived_pair_mask.packed_offsets(), explicit_pair_mask.packed_offsets())

            expected_msa_parts = []
            expected_pair_parts = []
            for dense_tokens, dense_continuous, dense_pair, dense_msa_mask in zip(
                torch.unbind(msa_tokens),
                torch.unbind(msa_continuous),
                torch.unbind(pair),
                torch.unbind(msa_mask),
                strict=True,
            ):
                dense_msa, dense_pair_output = reference(
                    dense_tokens.unsqueeze(0),
                    dense_continuous.unsqueeze(0),
                    dense_pair.unsqueeze(0),
                    msa_mask=dense_msa_mask.unsqueeze(0),
                    pair_mask=None,
                )
                expected_msa_parts.append(dense_msa.squeeze(0).reshape(-1, 4))
                expected_pair_parts.append(dense_pair_output.squeeze(0).reshape(-1, 4))
            expected_msa = torch.cat(expected_msa_parts)
            expected_pair = torch.cat(expected_pair_parts)

            with guard():
                actual_msa, actual_pair = compiled(msa_tokens, msa_continuous, pair, msa_mask)
            with torch.no_grad(), guard():
                explicit_msa, explicit_pair = module(
                    msa_tokens,
                    msa_continuous,
                    pair,
                    msa_mask=msa_mask,
                    pair_mask=explicit_pair_mask,
                )
            torch.testing.assert_close(actual_msa, explicit_msa.concat, rtol=0, atol=0)
            torch.testing.assert_close(actual_pair, explicit_pair.concat, rtol=0, atol=0)
            assert torch.isfinite(actual_msa).all()
            assert torch.isfinite(actual_pair).all()
            torch.testing.assert_close(actual_msa, expected_msa, rtol=1e-1, atol=1e-1)
            torch.testing.assert_close(actual_pair, expected_pair, rtol=1e-1, atol=1e-1)

            actual_inputs = (continuous_values, pair_values, *module.parameters())
            expected_inputs = (continuous_values, pair_values, *reference.parameters())
            grad_msa = torch.randn_like(actual_msa)
            grad_pair = torch.randn_like(actual_pair)
            actual_gradients = torch.autograd.grad(
                (actual_msa, actual_pair),
                actual_inputs,
                (grad_msa, grad_pair),
                retain_graph=True,
                allow_unused=True,
            )
            expected_gradients = torch.autograd.grad(
                (expected_msa, expected_pair),
                expected_inputs,
                (grad_msa, grad_pair),
                allow_unused=True,
            )
            activation_difference_square = 0.0
            activation_reference_square = 0.0
            parameter_difference_square = 0.0
            parameter_reference_square = 0.0
            for index, (actual_gradient, expected_gradient) in enumerate(
                zip(actual_gradients, expected_gradients, strict=True)
            ):
                assert (actual_gradient is None) == (expected_gradient is None)
                if actual_gradient is None or expected_gradient is None:
                    continue
                difference = actual_gradient.float() - expected_gradient.float()
                if index < 2:
                    activation_difference_square += difference.square().sum().item()
                    activation_reference_square += expected_gradient.float().square().sum().item()
                else:
                    parameter_difference_square += difference.square().sum().item()
                    parameter_reference_square += expected_gradient.float().square().sum().item()
            activation_scaled_l2 = activation_difference_square**0.5 / max(activation_reference_square**0.5, 1e-6)
            parameter_scaled_l2 = parameter_difference_square**0.5 / max(parameter_reference_square**0.5, 1e-6)
            print(
                f"derived layout={layout} activation_scaled_l2={activation_scaled_l2:.6g} "
                f"parameter_scaled_l2={parameter_scaled_l2:.6g}",
                flush=True,
            )
            assert activation_scaled_l2 < 8e-2
            assert parameter_scaled_l2 < 5e-3
        assert counter.frame_count == 1


def test_pair_weighted_average() -> None:
    torch.manual_seed(3)
    module = MsaPairWeightedAveraging(
        msa_size=4,
        pair_size=5,
        num_attention_heads=2,
        head_dim=3,
        backend="eager",
    )
    # zero the pair projection so logits vanish and softmax-over-valid-j becomes a hand-computable uniform average
    with torch.no_grad():
        module.pair_projection.weight.zero_()
    batch, seqs, residues, msa_size, heads = 1, 2, 3, 4, 2
    head_dim = 3
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
    averaged = torch.einsum("ij,bsjhd->bsihd", weights, values).reshape(
        batch,
        seqs,
        residues,
        heads * head_dim,
    )
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
