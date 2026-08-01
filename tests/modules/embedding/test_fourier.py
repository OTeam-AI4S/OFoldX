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

from ofoldx.modules.embedding.fourier import FrozenFourierEmbedding


class TestFrozenFourierEmbedding:
    def test_equation(self) -> None:
        embedding = FrozenFourierEmbedding(4)

        output = embedding(torch.ones(2))
        projected = F.linear(
            torch.ones(2, 1),
            embedding.projection.weight,
            embedding.projection.bias,
        )

        torch.testing.assert_close(output, torch.cos(2 * torch.pi * projected))
        assert output.shape == (2, 4)

    def test_frozen(self) -> None:
        embedding = FrozenFourierEmbedding(4)

        assert not embedding.projection.weight.requires_grad
        assert not embedding.projection.bias.requires_grad
