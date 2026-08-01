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

"""Dense atom mappings choose a provider that supports their actual layout."""

import pytest
import torch

from ofoldx.modules.backend import atom_window


@pytest.mark.parametrize("same_values", [False, True])
def test_distinct_mapping_layout(monkeypatch, same_values):
    def forbidden(*args, **kwargs):
        raise AssertionError("distinct dense mapping storage is outside this provider capability")

    monkeypatch.setattr(atom_window, "_oops_token_pair_to_atom_window_gather", forbidden)
    pair = torch.randn(2, 3, 3, 2, dtype=torch.float64, requires_grad=True)
    mapping = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    if same_values:
        mapping[1] = mapping[0]
    actual = atom_window.token_pair_to_atom_window_gather(
        pair, mapping, num_query_atoms=2, num_key_atoms=2, backend="oops"
    )
    expected = torch.stack(
        [
            atom_window.token_pair_to_atom_window_gather(
                sample, indices, num_query_atoms=2, num_key_atoms=2, backend="eager"
            )
            for sample, indices in zip(pair, mapping, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.square().sum(), pair, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.square().sum(), pair)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
