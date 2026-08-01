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

from ofoldx.modules.diffusion.conditioning import SingleConditioning


@pytest.mark.parametrize("bias", [False, True])
def test_single_conditioning_prepared_noise_equation_and_gradients(bias: bool) -> None:
    module = SingleConditioning(
        5, 4, fourier_embedding_size=3, num_transitions=0, norm_bias=bias, single_projection_bias=bias, backend="eager"
    ).double()
    trunk = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    inputs = torch.randn(2, 6, 2, dtype=torch.float64, requires_grad=True)
    noise = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    normalized = torch.nn.functional.layer_norm(
        torch.cat((trunk, inputs), -1), (5,), module.single_norm.weight, module.single_norm.bias, module.single_norm.eps
    )
    single = torch.nn.functional.linear(normalized, module.single_projection.weight, module.single_projection.bias)
    features = torch.cos(
        2
        * torch.pi
        * (noise[..., None] * module.noise_embedding.projection.weight.T + module.noise_embedding.projection.bias)
    )
    features = torch.nn.functional.layer_norm(
        features, (3,), module.noise_norm.weight, module.noise_norm.bias, module.noise_norm.eps
    )
    expected = single[:, None] + torch.nn.functional.linear(features, module.noise_projection.weight)[:, :, None]
    actual = module.condition_noise(noise, module.prepare(trunk, inputs))
    torch.testing.assert_close(actual, expected)
    arguments = (trunk, inputs, noise, *(p for p in module.parameters() if p.requires_grad))
    direction = torch.randn_like(expected)
    actual_gradients = torch.autograd.grad(actual, arguments, direction, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, arguments, direction)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
