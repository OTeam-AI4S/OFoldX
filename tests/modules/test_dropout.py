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

from ofoldx.modules.dropout import StructuredDropout


class TestStructuredDropout:
    @pytest.mark.parametrize(
        ("broadcast_dims", "mask_shape"),
        [
            (-3, (2, 1, 4, 5)),
            ((-2, -1), (2, 3, 1, 1)),
            ((-3, -1), (2, 1, 4, 1)),
        ],
    )
    def test_training(
        self,
        broadcast_dims: int | tuple[int, ...],
        mask_shape: tuple[int, ...],
    ) -> None:
        hidden = torch.arange(1, 2 * 3 * 4 * 5 + 1, dtype=torch.float32).reshape(2, 3, 4, 5)
        dropout = StructuredDropout(0.5, broadcast_dims=broadcast_dims)

        torch.manual_seed(0)
        actual = dropout.train()(hidden)
        torch.manual_seed(0)
        expected = hidden * F.dropout(hidden.new_ones(mask_shape), p=0.5, training=True)

        torch.testing.assert_close(actual, expected)

    def test_eval(self) -> None:
        hidden = torch.randn(2, 3, 4, 5)

        torch.testing.assert_close(StructuredDropout(0.5, broadcast_dims=-3).eval()(hidden), hidden)
