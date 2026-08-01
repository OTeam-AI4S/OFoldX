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

"""Shared checkpoint transforms preserve learned equations, dtypes and gradients."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ofoldx.models.convert import cast_floating_state, pack_conditioned_adaptive_norms
from ofoldx.modules.adaptive_norm import AdaptiveLayerNorm


def test_conditioned_norm_gradients() -> None:
    hidden_size, conditioning_size = 4, 3
    source = {
        "block.conditioned_norm.conditioning_norm.weight": torch.randn(conditioning_size, dtype=torch.float64),
        "block.conditioned_norm.gate_projection.weight": torch.randn(
            hidden_size, conditioning_size, dtype=torch.float64
        ),
        "block.conditioned_norm.gate_projection.bias": torch.randn(hidden_size, dtype=torch.float64),
        "block.conditioned_norm.shift_projection.weight": torch.randn(
            hidden_size, conditioning_size, dtype=torch.float64
        ),
    }
    original = {name: value.clone().requires_grad_() for name, value in source.items()}
    pack_conditioned_adaptive_norms(source)
    holder = nn.Module()
    holder.block = nn.Module()
    holder.block.conditioned_norm = AdaptiveLayerNorm(hidden_size, conditioning_size, backend="eager").double()
    holder.load_state_dict(source, strict=True)
    module = holder.block.conditioned_norm
    actual_hidden = torch.randn(2, 5, hidden_size, dtype=torch.float64, requires_grad=True)
    actual_conditioning = torch.randn(2, 5, conditioning_size, dtype=torch.float64, requires_grad=True)
    expected_hidden = actual_hidden.detach().clone().requires_grad_()
    expected_conditioning = actual_conditioning.detach().clone().requires_grad_()
    normalized_conditioning = F.layer_norm(
        expected_conditioning,
        (conditioning_size,),
        original["block.conditioned_norm.conditioning_norm.weight"],
        None,
        1e-5,
    )
    scale = F.linear(
        normalized_conditioning,
        original["block.conditioned_norm.gate_projection.weight"],
        original["block.conditioned_norm.gate_projection.bias"],
    ).sigmoid()
    shift = F.linear(normalized_conditioning, original["block.conditioned_norm.shift_projection.weight"])
    expected = scale * F.layer_norm(expected_hidden, (hidden_size,), eps=1e-5) + shift
    actual = module(actual_hidden, actual_conditioning)
    cotangent = torch.randn_like(actual)
    actual_targets = (
        actual_hidden,
        actual_conditioning,
        module.conditioning_norm.weight,
        module.conditioning_projection.weight,
        module.conditioning_scale_bias,
    )
    expected_targets = (
        expected_hidden,
        expected_conditioning,
        original["block.conditioned_norm.conditioning_norm.weight"],
        original["block.conditioned_norm.gate_projection.weight"],
        original["block.conditioned_norm.gate_projection.bias"],
        original["block.conditioned_norm.shift_projection.weight"],
    )
    actual_gradients = torch.autograd.grad((actual * cotangent).sum(), actual_targets)
    expected_gradients = torch.autograd.grad((expected * cotangent).sum(), expected_targets)

    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    for actual_gradient, expected_gradient in zip(actual_gradients[:3], expected_gradients[:3], strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual_gradients[3][:hidden_size], expected_gradients[3], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual_gradients[4], expected_gradients[4], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual_gradients[3][hidden_size:], expected_gradients[5], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [None, torch.float64, torch.float16, torch.bfloat16])
def test_float_state_cast(dtype) -> None:
    integer = torch.tensor([1, 2, 3], dtype=torch.int64)
    boolean = torch.tensor([True, False])
    complex_value = torch.tensor([1 + 2j], dtype=torch.complex64)
    floating = torch.tensor([1.25, -2.5], dtype=torch.float32)
    source = {"integer": integer, "boolean": boolean, "complex": complex_value, "floating": floating}
    original = {name: value.clone() for name, value in source.items()}
    result = cast_floating_state(source, dtype)

    assert result.keys() == source.keys()
    for name in ("integer", "boolean", "complex"):
        torch.testing.assert_close(result[name], original[name], rtol=0, atol=0)
        assert result[name].dtype == original[name].dtype
    expected_floating = floating if dtype is None else floating.to(dtype)
    torch.testing.assert_close(result["floating"], expected_floating, rtol=0, atol=0)
    assert result["floating"].dtype == expected_floating.dtype
    for name, value in source.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "projection_names", [("value_projection", "gate_projection"), ("gate_projection", "value_projection")]
)
def test_gate_value_projection_gradients(projection_names: tuple[str, str]) -> None:
    from torch.func import functional_call

    from ofoldx.models.convert import unpack_gate_value_projection
    from ofoldx.modules.feed_forward import SwiGluFeedForward

    torch.manual_seed(714)
    values = [torch.randn(shape, dtype=torch.float64) for shape in ((2, 3, 5), (8, 5), (8,), (6, 4), (6,))]

    def evaluate(reference: bool):
        hidden, packed_weight, packed_bias, output_weight, output_bias = (v.clone().requires_grad_() for v in values)
        if reference:
            by_name = dict(zip(projection_names, (slice(0, 4), slice(4, 8))))
            gate = F.linear(hidden, packed_weight[by_name["gate_projection"]], packed_bias[by_name["gate_projection"]])
            value = F.linear(
                hidden, packed_weight[by_name["value_projection"]], packed_bias[by_name["value_projection"]]
            )
            output = F.linear(F.silu(gate) * value, output_weight, output_bias)
        else:
            state = dict(
                unpack_gate_value_projection("unit.gate_value_projection.weight", packed_weight, projection_names)
            )
            state.update(unpack_gate_value_projection("unit.gate_value_projection.bias", packed_bias, projection_names))
            state = {key.removeprefix("unit."): value for key, value in state.items()}
            state["output_projection.weight"] = output_weight
            state["output_projection.bias"] = output_bias
            module = SwiGluFeedForward(5, 4, 6, bias=True).double()
            module.load_state_dict({key: value.detach() for key, value in state.items()}, strict=True)
            output = functional_call(module, state, (hidden,), strict=True)
        cotangent = torch.linspace(-0.3, 0.7, output.numel(), dtype=output.dtype).reshape_as(output)
        targets = (hidden, packed_weight, packed_bias, output_weight, output_bias)
        return output, torch.autograd.grad((output * cotangent).sum(), targets)

    actual, actual_vjp = evaluate(False)
    expected, expected_vjp = evaluate(True)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_vjp, expected_vjp):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_template_projection() -> None:
    from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
    from ofoldx.models.protenix.convert import translate_reference_state
    from ofoldx.models.protenix.model import PROTENIX_SOURCE_TOKENS
    from ofoldx.modules.embedding import FactorizedPairLinear

    size = len(PROTENIX_SOURCE_TOKENS)
    weight = torch.randn(7, 44 + 2 * size, dtype=torch.float64)
    pair = torch.randn(2, 5, 5, 44, dtype=torch.float64, requires_grad=True)
    token = torch.randn(2, 5, size, dtype=torch.float64, requires_grad=True)
    owner = nn.Module()
    owner.template_embedder = nn.Module()
    owner.template_embedder.template_projection = FactorizedPairLinear(
        44, len(TOKENS), 7, pair_prefix_size=40, bias=False, backend="eager"
    ).double()
    source_key = "template_embedder.linear_no_bias_a.weight"
    owner.load_state_dict(translate_reference_state(owner, {source_key: weight}, lambda _: source_key), strict=True)
    rows = torch.tensor([TOKEN_INDEX[t] for t in PROTENIX_SOURCE_TOKENS])
    canonical = token.new_zeros(2, 5, len(TOKENS)).index_copy(-1, rows, token)
    actual = owner.template_embedder.template_projection(pair, canonical)
    packed = torch.cat(
        [pair[..., :40], token[:, None].expand(-1, 5, -1, -1), token[:, :, None].expand(-1, -1, 5, -1), pair[..., 40:]],
        dim=-1,
    )
    expected = F.linear(packed, weight)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    cotangent = torch.randn_like(actual)
    gradients = torch.autograd.grad((actual * cotangent).sum(), (pair, token), retain_graph=True)
    reference = torch.autograd.grad((expected * cotangent).sum(), (pair, token))
    for gradient, wanted in zip(gradients, reference, strict=True):
        torch.testing.assert_close(gradient, wanted, atol=1e-12, rtol=1e-12)
