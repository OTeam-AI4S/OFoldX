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

import pytest
import torch

from ofoldx.modules.head import AtomConfidenceProjector, ConfidencePairEmbedder


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_confidence_distance_bin_edges_and_projection_gradient(dtype: torch.dtype) -> None:
    embedder = ConfidencePairEmbedder(token_feature_size=2, pair_size=2).to(dtype=dtype)
    num_bins = 39
    projection = embedder.distance_one_hot_projection
    with torch.no_grad():
        projection.weight.copy_(torch.arange(1, 2 * num_bins + 1, dtype=dtype).reshape(2, num_bins))

    distances = []
    selected_bins: list[int | None] = []
    lower_limit = torch.tensor(3.25, dtype=dtype)
    distances.extend([torch.nextafter(lower_limit, lower_limit.new_tensor(0.0)), lower_limit])
    selected_bins.extend([None, None])
    distances.append(torch.nextafter(lower_limit, lower_limit.new_tensor(torch.inf)))
    selected_bins.append(0)
    for bin_index in range(1, num_bins):
        edge = torch.tensor(3.25 + 1.25 * bin_index, dtype=dtype)
        distances.extend(
            [
                torch.nextafter(edge, edge.new_tensor(0.0)),
                edge,
                torch.nextafter(edge, edge.new_tensor(torch.inf)),
            ]
        )
        selected_bins.extend([bin_index - 1, bin_index - 1, bin_index])
    distances.append(lower_limit.new_tensor(100.0))
    selected_bins.append(num_bins - 1)
    distance = torch.stack(distances)
    pair = torch.arange(2 * len(distances), dtype=dtype).reshape(-1, 2).requires_grad_()
    cotangent = (pair.detach() + 1.0) / len(distances)

    expected = pair.detach().clone()
    expected_weight_gradient = torch.zeros_like(projection.weight)
    for row, bin_index in enumerate(selected_bins):
        if bin_index is not None:
            expected[row] += projection.weight.detach()[:, bin_index]
            expected_weight_gradient[:, bin_index] += cotangent[row]

    actual = embedder.embed_distance(pair, distance)
    pair_gradient, weight_gradient = torch.autograd.grad((actual * cotangent).sum(), (pair, projection.weight))

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pair_gradient, cotangent)
    torch.testing.assert_close(weight_gradient, expected_weight_gradient)


class TestAtomConfidenceProjector:
    def test_atom_limit(self) -> None:
        projector = AtomConfidenceProjector(single_size=4, max_atoms_per_token=2, backend="eager")

        with pytest.raises(
            ValueError,
            match="a token carries 3 atoms but this confidence head was built for max_atoms_per_token=2",
        ):
            projector(
                torch.randn(1, 1, 4),
                torch.zeros(1, 1, dtype=torch.long),
                torch.tensor([[2]]),
            )

    def test_fullgraph(self) -> None:
        projector = AtomConfidenceProjector(single_size=4, max_atoms_per_token=3, backend="eager")
        with torch.no_grad():
            projector.plddt_weight.normal_()
            projector.resolved_weight.normal_()
        compiled = torch.compile(projector, backend="eager", fullgraph=True, dynamic=True)
        single = torch.randn(1, 3, 4)

        for atom_to_token, atom_token_index in (
            (torch.tensor([[0, 0, 1, 2]]), torch.tensor([[0, 1, 0, 0]])),
            (torch.tensor([[1, 1]]), torch.tensor([[0, 1]])),
        ):
            expected = projector(single, atom_to_token, atom_token_index)
            actual = compiled(single, atom_to_token, atom_token_index)
            assert tuple(value.shape for value in actual) == tuple(value.shape for value in expected)
            assert tuple(value.dtype for value in actual) == tuple(value.dtype for value in expected)
            torch.testing.assert_close(actual, expected)
