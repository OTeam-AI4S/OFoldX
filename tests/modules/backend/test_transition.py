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

import pytest
import torch
import torch.nn.functional as F

from ofoldx.modules.backend.transition import transition


def _constant_roundoff_bounds(references, dtype):
    """Propagate norm conditioning through an independent SwiGLU equation.

    At a constant input, the exact normalized activation is the bias. Native
    affine evaluation can cancel two terms of scale |x| / sqrt(eps); this does
    not justify comparing that ill-conditioned path at the random-input bound.
    """

    x, weight, bias, gate, up, down = (value.detach() for value in references)
    rows = x.numel() // x.shape[-1]
    eps = torch.finfo(dtype).eps
    stats_eps = torch.finfo(torch.float32).eps
    gamma = 16 * eps / (1 - 16 * eps)
    stats_gamma = 4 * stats_eps / (1 - 4 * stats_eps)
    rstd = 1e-5**-0.5
    norm_condition = 2 * x.abs().amax() * rstd
    dz = stats_gamma * norm_condition * weight.abs() + eps * bias.abs()
    gate_value, up_value = F.linear(bias, gate), F.linear(bias, up)
    dg = F.linear(dz, gate.abs()) + gamma * F.linear(bias.abs() + dz, gate.abs())
    du = F.linear(dz, up.abs()) + gamma * F.linear(bias.abs() + dz, up.abs())
    sigmoid = gate_value.sigmoid()
    activation = gate_value * sigmoid
    slope = sigmoid + gate_value * sigmoid * (1 - sigmoid)
    # Global SiLU derivative bounds: |s'| <= 1.1 and |s''| <= 0.5.
    da = 1.1 * dg + gamma * (activation.abs() + 1.1 * dg)
    ds = 0.5 * dg + gamma * (slope.abs() + 0.5 * dg)
    product = activation * up_value
    dp = up_value.abs() * da + activation.abs() * du + da * du
    dp = dp + gamma * (activation.abs() + da) * (up_value.abs() + du)
    output_bound = F.linear(dp, down.abs()) + gamma * F.linear(product.abs() + dp, down.abs())

    # Sum-output cotangent coefficients; absolute column sums bound their
    # accumulation even when signed output coefficients cancel.
    coefficient = down.sum(0)
    dc = gamma * down.abs().sum(0)
    gate_term = coefficient * slope * up_value
    up_term = coefficient * activation
    gate_term_bound = (coefficient.abs() + dc) * (slope.abs() + ds) * (
        up_value.abs() + du
    ) - coefficient.abs() * slope.abs() * up_value.abs()
    gate_term_bound += gamma * (gate_term.abs() + gate_term_bound)
    up_term_bound = (coefficient.abs() + dc) * (activation.abs() + da) - coefficient.abs() * activation.abs()
    up_term_bound += gamma * (up_term.abs() + up_term_bound)
    bias_gradient = gate.T @ gate_term + up.T @ up_term
    bias_scale = gate.abs().T @ gate_term.abs() + up.abs().T @ up_term.abs()
    dbias = gate.abs().T @ gate_term_bound + up.abs().T @ up_term_bound
    dbias += gamma * (bias_scale + dbias)
    weighted_gradient = weight * bias_gradient
    dx = rstd * (weight.abs() * dbias + (weight.abs() * dbias).mean())
    dx += gamma * rstd * (weighted_gradient.abs() + weighted_gradient.abs().mean())
    norm_weight_bound = rows * stats_gamma * norm_condition * (bias_gradient.abs() + dbias)
    norm_bias_bound = rows * (dbias + gamma * (bias_gradient.abs() + dbias))
    gate_gradient_bound = rows * (
        gate_term_bound[:, None] * (bias.abs() + dz)[None, :] + gate_term.abs()[:, None] * dz[None, :]
    )
    gate_gradient_bound += gamma * (rows * gate_term.abs()[:, None] * bias.abs()[None, :] + gate_gradient_bound)
    up_gradient_bound = rows * (
        up_term_bound[:, None] * (bias.abs() + dz)[None, :] + up_term.abs()[:, None] * dz[None, :]
    )
    up_gradient_bound += gamma * (rows * up_term.abs()[:, None] * bias.abs()[None, :] + up_gradient_bound)
    down_gradient_bound = rows * (dp + gamma * (product.abs() + dp))
    return output_bound, (
        dx.expand_as(x),
        norm_weight_bound,
        norm_bias_bound,
        gate_gradient_bound,
        up_gradient_bound,
        down_gradient_bound.expand_as(down),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("constant", [False, True])
def test_transition_complete_centered_equation_and_all_vjps(dtype, constant) -> None:
    generator = torch.Generator().manual_seed(649)
    hidden = (
        torch.full((2, 3, 5), 1000.5, dtype=dtype)
        if constant
        else torch.randn(2, 3, 5, generator=generator, dtype=dtype)
    ).requires_grad_()
    norm_weight = torch.randn(5, generator=generator, requires_grad=True)
    norm_bias = torch.randn(5, generator=generator, requires_grad=True)
    gate_weight = torch.randn(9, 5, generator=generator, dtype=dtype, requires_grad=True) * 0.2
    up_weight = torch.randn(9, 5, generator=generator, dtype=dtype, requires_grad=True) * 0.2
    down_weight = torch.randn(5, 9, generator=generator, dtype=dtype, requires_grad=True) * 0.2
    tensors = (hidden, norm_weight, norm_bias, gate_weight, up_weight, down_weight)
    output = transition(
        hidden,
        input_norm_weight=norm_weight,
        input_norm_bias=norm_bias,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        input_norm_eps=1e-5,
        backend="eager",
    )
    references = tuple(value.detach().double().requires_grad_() for value in tensors)
    x, weight, bias, gate, up, down = references
    centered = x - x.mean(-1, keepdim=True)
    normalized = centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    normalized = normalized * weight + bias
    expected = F.linear(F.silu(F.linear(normalized, gate)) * F.linear(normalized, up), down)
    tolerance = {torch.float16: 0.003, torch.bfloat16: 0.025, torch.float32: 2e-5}[dtype]
    assert torch.isfinite(output).all()
    if constant:
        output_bound, gradient_bounds = _constant_roundoff_bounds(references, dtype)
        assert ((output.double() - expected).abs() <= output_bound).all()
    else:
        torch.testing.assert_close(output.double(), expected, rtol=tolerance, atol=tolerance * 0.1)
    actual_vjps = torch.autograd.grad(output.sum(), tensors)
    expected_vjps = torch.autograd.grad(expected.sum(), references)
    for index, (actual, wanted) in enumerate(zip(actual_vjps, expected_vjps, strict=True)):
        assert torch.isfinite(actual).all()
        if constant:
            assert ((actual.double() - wanted).abs() <= gradient_bounds[index]).all(), (dtype, index)
        else:
            error = (actual.double() - wanted).norm() / wanted.norm().clamp_min(1)
            assert error < tolerance * 2, (dtype, constant, error.item())
