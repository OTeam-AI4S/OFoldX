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

"""Public MSA column availability before normalization."""

import torch

from ofoldx.modules.attention.msa import MsaColumnGlobalAttention


def test_global_column_unavailable_gradients():
    module = MsaColumnGlobalAttention(msa_size=4, num_attention_heads=1, backend="eager").double()
    values = torch.randn(1, 2, 2, 4, dtype=torch.float64)
    mask = torch.tensor([[[True, True], [False, True]]])
    values[~mask] = torch.nan
    values.requires_grad_()
    output = module(values, msa_mask=mask)
    assert torch.isfinite(output).all()
    gradients = torch.autograd.grad(output.square().sum(), (values, *module.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(gradients[0][~mask], torch.zeros_like(gradients[0][~mask]))
