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

import torch
import torch.nn.functional as F

from ofoldx.modules.adaptive_norm import AdaptiveLayerNorm


class TestAdaptiveLayerNorm:
    def test_forward_and_gradients(self) -> None:
        module = AdaptiveLayerNorm(4, 3, backend="eager")
        hidden = torch.randn(2, 5, 4, requires_grad=True)
        conditioning = torch.randn(2, 5, 3, requires_grad=True)

        actual = module(hidden, conditioning)
        normalized_hidden = F.layer_norm(hidden, (4,))
        normalized_conditioning = F.layer_norm(
            conditioning,
            (3,),
            module.conditioning_norm.weight,
            eps=module.conditioning_norm.eps,
        )
        scale, shift = F.linear(normalized_conditioning, module.conditioning_projection.weight).chunk(2, dim=-1)
        expected = torch.sigmoid(scale + module.conditioning_scale_bias) * normalized_hidden + shift
        torch.testing.assert_close(actual, expected)

        actual.sum().backward()
        assert hidden.grad is not None
        assert conditioning.grad is not None
