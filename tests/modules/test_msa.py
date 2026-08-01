# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from danling.tensors import NestedTensor
from torch._dynamo.testing import CompileCounter
from torch.utils.checkpoint import checkpoint

from ofoldx.modules.functional import pair_mask_from_msa_mask
from ofoldx.modules.msa import (
    MsaEncoder,
    MsaPairWeightedAveraging,
    MsaUpdateBlock,
    OuterProductMean,
    PostProjectionOuterProductMean,
)
from tests.modules import require_oops


def _encoder(
    msa_dropout: float = 0.0,
    backend: str = "eager",
    *,
    pair_dropout: float = 0.0,
    num_stages: int = 2,
) -> MsaEncoder:
    return MsaEncoder(
        msa_size=4,
        pair_size=4,
        msa_vocab_size=3,
        token_feature_size=3,
        num_stages=num_stages,
        outer_product_hidden_size=2,
        num_msa_attention_heads=2,
        triangle_multiplication_hidden_size=4,
        triangle_attention_head_size=2,
        num_triangle_attention_heads=2,
        msa_dropout=msa_dropout,
        pair_dropout=pair_dropout,
        backend=backend,
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


class TestMsaEncoder:
    def test_shapes_and_gradients(self) -> None:
        msa_tokens = torch.randint(3, (1, 2, 3))
        msa_continuous = torch.randn(1, 2, 3, 2, requires_grad=True)
        pair = torch.randn(1, 3, 3, 4, requires_grad=True)
        token_embedding = torch.randn(1, 3, 3, requires_grad=True)

        output_msa, output_pair = _encoder()(msa_tokens, msa_continuous, pair, token_embedding)
        (output_msa.sum() + output_pair.sum()).backward()

        assert output_msa.shape == (1, 2, 3, 4)
        assert output_pair.shape == pair.shape
        assert msa_continuous.grad is not None
        assert pair.grad is not None
        assert token_embedding.grad is not None

    @pytest.mark.parametrize("num_stages", [1, 2])
    def test_gradient_checkpointing(self, num_stages: int) -> None:
        eager = _encoder(msa_dropout=0.25, pair_dropout=0.25, num_stages=num_stages)
        checkpointed = _encoder(msa_dropout=0.25, pair_dropout=0.25, num_stages=num_stages)
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
        for parameter, reference in zip(checkpointed.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, reference.grad)

    def test_fullgraph(self) -> None:
        reference = _encoder()
        compiled_module = _encoder()
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
    @pytest.mark.parametrize(
        ("backend", "compiled"), [("auto", False), ("oops", True)], ids=["auto-eager", "oops-fullgraph"]
    )
    def test_nested_vertical_matches_independent_dense(self, backend: str, compiled: bool) -> None:
        if backend == "oops":
            require_oops("pair_weighted_average", "transition")
        torch.manual_seed(20260828)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        module = _encoder(backend=backend).to(device=device, dtype=dtype)
        module.gradient_checkpointing = True
        module._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        reference = _encoder(backend="eager").to(device=device, dtype=dtype)
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

        execute = torch.compile(run, backend=counter, fullgraph=True, dynamic=True) if compiled else run

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

            actual_msa, actual_pair = execute(
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
        if compiled:
            assert counter.frame_count == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
    @pytest.mark.parametrize(
        ("backend", "compiled"), [("auto", False), ("oops", True)], ids=["auto-eager", "oops-fullgraph"]
    )
    def test_nested_derived_pair_mask(self, backend: str, compiled: bool) -> None:
        if backend == "oops":
            require_oops("pair_weighted_average", "transition")
        torch.manual_seed(20260829)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        module = _encoder(backend=backend).to(device=device, dtype=dtype)
        reference = _encoder(backend="eager").to(device=device, dtype=dtype)
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

        execute = torch.compile(run, backend=counter, fullgraph=True, dynamic=True) if compiled else run

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

            actual_msa, actual_pair = execute(msa_tokens, msa_continuous, pair, msa_mask)
            with torch.no_grad():
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
        if compiled:
            assert counter.frame_count == 1


def test_pair_weighted_average() -> None:
    torch.manual_seed(3)
    module = MsaPairWeightedAveraging(
        msa_size=4,
        pair_size=5,
        num_attention_heads=2,
        head_size=3,
        backend="eager",
    )
    # zero the pair projection so logits vanish and softmax-over-valid-j becomes a hand-computable uniform average
    with torch.no_grad():
        module.pair_projection.weight.zero_()
    batch, seqs, residues, msa_size, heads = 1, 2, 3, 4, 2
    head_size = 3
    msa = torch.randn(batch, seqs, residues, msa_size)
    pair = torch.randn(batch, residues, residues, 5)
    # distinct per-row valid sets: row 0 -> {0,1}, row 1 -> {1,2}, row 2 -> {0,2}
    pair_mask = torch.tensor([[[True, True, False], [False, True, True], [True, False, True]]])

    output = module(msa, pair, pair_mask=pair_mask)

    # expected from the PWA definition with weights derived by hand (uniform over valid j, no softmax replay)
    normalized = F.layer_norm(msa, (msa_size,), module.msa_norm.weight, module.msa_norm.bias, module.msa_norm.eps)
    values = F.linear(normalized, module.value_projection.weight).reshape(batch, seqs, residues, heads, head_size)
    weights = pair_mask[0].to(values.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    averaged = torch.einsum("ij,bsjhd->bsihd", weights, values).reshape(
        batch,
        seqs,
        residues,
        heads * head_size,
    )
    gated = averaged * torch.sigmoid(F.linear(normalized, module.gate_projection.weight))
    expected = F.linear(gated, module.output_projection.weight)

    torch.testing.assert_close(output, expected)


class TestMsaUpdateBlock:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA OOps kernels")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("backend", ["auto", "oops"])
    def test_autocast_residual(self, dtype: torch.dtype, backend: str) -> None:
        if backend == "oops":
            require_oops("pair_weighted_average", "transition")
        torch.manual_seed(73)
        module = MsaUpdateBlock(msa_size=8, pair_size=4, num_attention_heads=2, dropout=0.25, backend=backend).cuda()
        msa = torch.randn(1, 3, 4, 8, device="cuda", dtype=dtype)
        pair = torch.randn(1, 4, 4, 4, device="cuda", dtype=dtype)
        results = []
        for direct in (True, False):
            module.zero_grad(set_to_none=True)
            current_msa = msa.detach().clone().requires_grad_()
            current_pair = pair.detach().clone().requires_grad_()
            torch.manual_seed(91)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if direct:
                    refined = current_msa + module.rowwise_dropout(
                        module.msa_pair_weighted_averaging(current_msa, current_pair)
                    )
                    output = refined + module.transition(refined)
                else:
                    output = module(current_msa, current_pair)
            assert output.dtype == dtype
            torch.testing.assert_close(current_msa, msa)
            torch.testing.assert_close(current_pair, pair)
            output.float().square().sum().backward()
            gradients = []
            for gradient in (current_msa.grad, current_pair.grad, *(p.grad for p in module.parameters())):
                assert gradient is not None
                gradients.append(gradient.detach().clone())
            results.append((output.detach().clone(), *gradients))
        for actual, expected in zip(results[1], results[0], strict=True):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("payload", [-1, 10000])
def test_encoder_categorical_availability(payload):
    encoder = _encoder(num_stages=0).double()
    tokens = torch.tensor([[[1, payload]]])
    mask = torch.tensor([[[True, False]]])
    continuous = torch.tensor([[[[0.0, 0.2], [torch.nan, torch.inf]]]], dtype=torch.float64, requires_grad=True)
    pair = torch.zeros(1, 2, 2, 4, dtype=torch.float64)
    msa, _ = encoder(tokens, continuous, pair, msa_mask=mask)
    reference, _ = encoder(
        tokens.masked_fill(~mask, 0), torch.where(mask[..., None], continuous, 0), pair, msa_mask=mask
    )
    torch.testing.assert_close(msa, reference)
    gradient = torch.autograd.grad(msa.sum(), continuous)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))


@pytest.mark.parametrize("module_type", [OuterProductMean, PostProjectionOuterProductMean])
@pytest.mark.parametrize("numeric", [False, True])
def test_outer_product_unavailable_payload(module_type, numeric):
    module = module_type(msa_size=4, hidden_size=2, pair_size=3, backend="eager").double()
    mask = torch.tensor([[[True, False], [True, True]]])
    coefficients = mask.double().requires_grad_() if numeric else mask
    values = torch.randn(1, 2, 2, 4, dtype=torch.float64)
    values[~mask] = torch.nan
    values.requires_grad_()
    result = module(values, msa_mask=coefficients)
    assert torch.isfinite(result).all()
    gradient = torch.autograd.grad(result.square().sum(), values)[0]
    assert torch.isfinite(gradient).all()
    assert torch.equal(gradient[~mask], torch.zeros_like(gradient[~mask]))


def test_outer_product_coefficient_gradient():
    module = OuterProductMean(msa_size=4, hidden_size=2, pair_size=3, backend="eager").double()
    values = torch.randn(1, 2, 2, 4, dtype=torch.float64)
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    actual = module(values, msa_mask=mask)
    normalized = F.layer_norm(values, (4,), module.msa_norm.weight, module.msa_norm.bias, module.msa_norm.eps)
    left = module.left_projection(normalized) * mask[..., None]
    right = module.right_projection(normalized) * mask[..., None]
    outer = torch.einsum("bsih,bsjk->bijhk", left, right).flatten(-2)
    count = torch.einsum("bsi,bsj->bij", mask, mask)[..., None]
    expected = module.output_projection(outer / count.clamp_min(1))
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), mask, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), mask)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
