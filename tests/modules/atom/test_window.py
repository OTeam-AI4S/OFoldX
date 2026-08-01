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
from danling.tensors import NestedTensor
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.atom import ClampedLocalAtomWindowing, LocalAtomWindowing


class TestLocalAtomWindowing:
    @pytest.mark.parametrize("window", [LocalAtomWindowing, ClampedLocalAtomWindowing])
    def test_unavailable_reference_gradients(self, window) -> None:
        positions = torch.tensor([[[1.0, 2.0, 3.0], [float("nan"), float("inf"), 0.0]]], requires_grad=True)
        geometry = window(2, 4)(
            positions,
            torch.zeros(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[True, False]]),
        )
        assert torch.isfinite(geometry.distance_vectors).all()
        gradient = torch.autograd.grad(geometry.distance_vectors.square().sum(), positions)[0]
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient[:, 1], torch.zeros(1, 3))

    def test_centred_padding(self) -> None:
        tensor = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)

        info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(torch.ones(1, 5, dtype=torch.bool))
        query, key = info.gather(tensor)

        assert query.shape == (1, 3, 2, 2)
        assert key.shape == (1, 3, 4, 2)
        assert info.key_index is None
        assert info.valid_extent is None
        assert torch.equal(query[0, 2], torch.tensor([[8.0, 9.0], [0.0, 0.0]]))
        assert torch.equal(key[0, 0], torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
        assert torch.equal(
            info.attention_mask[0],
            torch.tensor(
                [
                    [[0, 1, 1, 1], [0, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.bool,
            ),
        )

    def test_centred_masked_atoms(self) -> None:
        # Masked atoms keep their centred slots but are neither queries nor keys.
        atom_mask = torch.tensor([[True, False, True, True, True]])

        info = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)

        assert info.key_index is None
        assert torch.equal(
            info.attention_mask[0],
            torch.tensor(
                [
                    [[0, 1, 0, 1], [0, 0, 0, 0]],
                    [[0, 1, 1, 1], [0, 1, 1, 1]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.bool,
            ),
        )

    def test_clamped_rows(self) -> None:
        tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
        atom_mask = torch.tensor(
            [
                [True] * 8,
                [True, True, True, True, True, False, False, False],
            ]
        )

        info = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)
        _, key = info.gather(tensor)

        # Each centred start is clamped into [0, max(V - 4, 0)] for the valid extent V of its row.
        assert info.key_index is not None
        assert info.valid_extent is None
        assert torch.equal(info.key_index[0, :, 0], torch.tensor([0, 1, 3, 4]))
        assert torch.equal(info.key_index[1, :, 0], torch.tensor([0, 1, 1, 1]))
        assert torch.equal(key[0, -1, :, 0], tensor[0, 4:8, 0])
        assert torch.equal(key[1, -1, :, 0], tensor[1, 1:5, 0])
        assert torch.equal(
            info.attention_mask[1].any(-1),
            torch.tensor([[True, True], [True, True], [True, False], [False, False]]),
        )

    def test_clamped_short_row_holds_every_valid_atom(self) -> None:
        atom_mask = torch.tensor([[True, False, True, False, False, False]])

        info = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(atom_mask)

        assert info.key_index is not None
        assert torch.equal(info.key_index[0], torch.tensor([[0, -1, 2, -1]]).expand(3, -1))

    @pytest.mark.parametrize(
        ("num_query_atoms", "num_key_atoms"),
        [(4, 2), (3, 4), (2, 5)],
    )
    def test_invalid_sizes(self, num_query_atoms: int, num_key_atoms: int) -> None:
        with pytest.raises(ValueError, match="must be even"):
            LocalAtomWindowing(num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms)

    @pytest.mark.parametrize(
        ("lengths", "masks", "expected_extent"),
        [
            (
                (3, 9),
                (
                    torch.tensor([True, False, True]),
                    torch.tensor([True, True, False, True, True, False, False, False, False]),
                ),
                (3, 5),
            ),
            (
                (1, 6),
                (torch.zeros(1, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)),
                (0, 0),
            ),
        ],
        ids=("holes-and-short-row", "all-false"),
    )
    def test_nested_clamped_matches_independent_rows(
        self,
        lengths: tuple[int, int],
        masks: tuple[torch.Tensor, torch.Tensor],
        expected_extent: tuple[int, int],
    ) -> None:
        windowing = ClampedLocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)
        values = tuple(torch.randn(length, 3) for length in lengths)
        nested_values = _nested_from_parts(values)
        nested_masks = _nested_from_parts(masks)

        info = windowing.window_info(nested_masks)
        query, key = info.gather(nested_values)
        packed_query = info.values(info.pack_like(nested_values, query))
        expected = tuple(windowing.window_info(mask.unsqueeze(0)) for mask in masks)
        expected_windows = tuple(
            row_info.gather(value.unsqueeze(0)) for row_info, value in zip(expected, values, strict=True)
        )

        assert info.key_index is None
        assert info.packed_layout is not None
        assert info.valid_extent is not None
        assert torch.equal(info.valid_extent, torch.tensor(expected_extent))
        assert torch.equal(query, torch.cat(tuple(windows[0].squeeze(0) for windows in expected_windows)))
        assert torch.equal(key, torch.cat(tuple(windows[1].squeeze(0) for windows in expected_windows)))
        assert torch.equal(packed_query, query)
        assert torch.equal(
            info.attention_mask,
            torch.cat(tuple(row_info.attention_mask.squeeze(0) for row_info in expected)),
        )

    def test_nested_centred_matches_dense_rows(self) -> None:
        windowing = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)
        values = (torch.arange(18, dtype=torch.float32).reshape(9, 2), torch.randn(5, 2))
        masks = tuple(torch.ones(value.shape[0], dtype=torch.bool) for value in values)

        info = windowing.window_info(_nested_from_parts(masks))
        query, key = info.gather(_nested_from_parts(values))
        expected = tuple(windowing.window_info(mask.unsqueeze(0)) for mask in masks)
        expected_windows = tuple(
            row_info.gather(value.unsqueeze(0)) for row_info, value in zip(expected, values, strict=True)
        )

        assert info.valid_extent is None
        assert torch.equal(query, torch.cat(tuple(windows[0].squeeze(0) for windows in expected_windows)))
        assert torch.equal(key, torch.cat(tuple(windows[1].squeeze(0) for windows in expected_windows)))
        assert torch.equal(
            info.attention_mask,
            torch.cat(tuple(row_info.attention_mask.squeeze(0) for row_info in expected)),
        )

    def test_nested_rejects_empty_elements_before_device_gather(self) -> None:
        masks = _nested_from_parts((torch.empty(0, dtype=torch.bool), torch.ones(3, dtype=torch.bool)))

        with pytest.raises(ValueError, match="at least one atom"):
            LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).window_info(masks)


class TestLocalAtomGeometry:
    def test_geometry(self) -> None:
        positions = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0], [3.0, 0, 0], [0, 0, 4.0], [5.0, 0, 0]]])
        spaces = torch.tensor([[0, 0, 1, 1, 1]])

        geometry = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(
            positions,
            spaces,
            torch.ones(1, 5, dtype=torch.bool),
        )

        assert geometry.distance_vectors.shape == (1, 3, 2, 4, 3)
        torch.testing.assert_close(geometry.distance_vectors[0, 0, 0, 2], torch.tensor([1.0, -2.0, 0.0]))
        assert geometry.distance_mask[0, 0, :, :, 0].tolist() == [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]

    def test_masked_atoms_are_invalid_pairs(self) -> None:
        positions = torch.randn(1, 5, 3)
        positions[:, 1] = torch.nan
        positions[:, 2] = torch.inf
        geometry = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(
            positions,
            torch.zeros(1, 5, dtype=torch.long),
            torch.tensor([[True, False, True, True, True]]),
            torch.tensor([[True, True, False, True, True]]),
        )

        torch.testing.assert_close(
            geometry.distance_mask.squeeze(-1),
            torch.tensor(
                [[[[0, 1, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 0], [0, 0, 0, 0]]]],
                dtype=positions.dtype,
            ),
        )
        assert geometry.window_info.attention_mask[0].tolist() == [
            [[False, True, False, True], [False, False, False, False]],
            [[False, True, True, True], [False, True, True, True]],
            [[True, True, False, False], [False, False, False, False]],
        ]
        assert torch.isfinite(geometry.distance_vectors).all()
        assert (geometry.distance_vectors[~geometry.distance_mask.squeeze(-1).bool()] == 0).all()

    @pytest.mark.parametrize("windowing_type", [LocalAtomWindowing, ClampedLocalAtomWindowing])
    def test_shared_prefix_broadcasts(self, windowing_type: type[LocalAtomWindowing]) -> None:
        windowing = windowing_type(num_query_atoms=2, num_key_atoms=4)
        positions = torch.randn(2, 3, 5, 3)
        spaces = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        atom_mask = torch.tensor([[True, True, True, False, False], [True] * 5])

        geometry = windowing(positions, spaces, atom_mask)
        expanded = windowing(
            positions,
            spaces.unsqueeze(1).expand(-1, 3, -1),
            atom_mask.unsqueeze(1).expand(-1, 3, -1),
        )

        assert geometry.distance_vectors.shape == (2, 3, 3, 2, 4, 3)
        torch.testing.assert_close(geometry.distance_vectors, expanded.distance_vectors)
        torch.testing.assert_close(geometry.distance_mask, expanded.distance_mask)

    @pytest.mark.parametrize("windowing_type", [LocalAtomWindowing, ClampedLocalAtomWindowing])
    def test_nested_dynamic_fullgraph_reuses_one_frame(self, windowing_type: type[LocalAtomWindowing]) -> None:
        windowing = windowing_type(num_query_atoms=2, num_key_atoms=4)
        counter = CompileCounter()
        compiled = torch.compile(
            lambda positions, spaces, mask: windowing(positions, spaces, mask).distance_vectors,
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        for lengths in ((3, 9), (5, 7)):
            positions = _nested_from_parts(tuple(torch.randn(length, 3) for length in lengths))
            spaces = _nested_from_parts(tuple(torch.arange(length) // 3 for length in lengths))
            masks = _nested_from_parts(tuple(torch.arange(length) % 4 != 1 for length in lengths))
            output = compiled(positions, spaces, masks)
            assert output.shape == (sum((length + 1) // 2 for length in lengths), 2, 4, 3)
        assert counter.frame_count == 1


def _nested_from_parts(parts: tuple[torch.Tensor, ...]) -> NestedTensor:
    reference = NestedTensor([torch.empty(1, *part.shape[1:], dtype=part.dtype) for part in parts], ragged_dims=(0,))
    lengths = torch.tensor(tuple(part.shape[0] for part in parts), dtype=torch.int64)
    return reference.packed_with_lengths(torch.cat(parts), lengths)
