from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard
from torch._dynamo.testing import CompileCounter

from ofoldx.modules.atom import LocalAtomGeometryBuilder, LocalAtomWindowing


class TestLocalAtomWindowing:
    def test_padding(self) -> None:
        tensor = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)

        windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(tensor)

        assert windows.query.shape == (1, 3, 2, 2)
        assert windows.key.shape == (1, 3, 4, 2)
        assert windows.info.key_index is None
        assert windows.info.valid_extent is None
        assert torch.equal(windows.query[0, 2], torch.tensor([[8.0, 9.0], [0.0, 0.0]]))
        assert torch.equal(windows.key[0, 0], torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
        assert torch.equal(
            windows.info.attention_mask[0],
            torch.tensor(
                [
                    [[0, 1, 1, 1], [0, 1, 1, 1]],
                    [[1, 1, 1, 1], [1, 1, 1, 1]],
                    [[1, 1, 0, 0], [0, 0, 0, 0]],
                ],
                dtype=torch.bool,
            ),
        )

    def test_row_masks(self) -> None:
        tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)
        atom_mask = torch.tensor(
            [
                [True] * 8,
                [True, True, True, True, True, False, False, False],
            ]
        )

        windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(tensor, atom_mask)

        assert windows.info.key_index is not None
        assert torch.equal(windows.info.valid_extent, torch.tensor([8, 5]))
        assert torch.equal(windows.key[0, -1, :, 0], tensor[0, 4:8, 0])
        assert torch.equal(windows.key[1, -1, :, 0], tensor[1, 1:5, 0])
        assert torch.equal(
            windows.info.attention_mask[1].any(-1),
            torch.tensor([[True, True], [True, True], [True, False], [False, False]]),
        )

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
    def test_nested_matches_independent_rows(
        self,
        lengths: tuple[int, int],
        masks: tuple[torch.Tensor, torch.Tensor],
        expected_extent: tuple[int, int],
    ) -> None:
        values = tuple(torch.randn(length, 3) for length in lengths)
        nested_values = _nested_from_parts(values)
        nested_masks = _nested_from_parts(masks)

        with _guard():
            actual = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(nested_values, nested_masks)
            reused = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4).gather(nested_values, actual.info)
            packed_query = actual.info.pack_like(nested_values, actual.query)
            packed_query_values = actual.info.values(packed_query)
        expected = tuple(
            LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(value.unsqueeze(0), mask.unsqueeze(0))
            for value, mask in zip(values, masks, strict=True)
        )

        assert actual.info.key_index is None
        assert actual.info.packed_layout is not None
        assert torch.equal(actual.info.valid_extent, torch.tensor(expected_extent))
        assert torch.equal(actual.query, torch.cat(tuple(windows.query.squeeze(0) for windows in expected)))
        assert torch.equal(actual.key, torch.cat(tuple(windows.key.squeeze(0) for windows in expected)))
        assert torch.equal(reused.query, actual.query)
        assert torch.equal(reused.key, actual.key)
        assert torch.equal(packed_query_values, actual.query)
        assert torch.equal(
            actual.info.attention_mask,
            torch.cat(tuple(windows.info.attention_mask.squeeze(0) for windows in expected)),
        )

    def test_nested_unmasked_windows_match_centered_dense_rows(self) -> None:
        values = (torch.arange(18, dtype=torch.float32).reshape(9, 2), torch.randn(5, 2))
        nested_values = _nested_from_parts(values)

        with _guard():
            actual = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(nested_values)
        expected = tuple(LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(value.unsqueeze(0)) for value in values)

        assert torch.equal(actual.query, torch.cat(tuple(windows.query.squeeze(0) for windows in expected)))
        assert torch.equal(actual.key, torch.cat(tuple(windows.key.squeeze(0) for windows in expected)))
        assert torch.equal(
            actual.info.attention_mask,
            torch.cat(tuple(windows.info.attention_mask.squeeze(0) for windows in expected)),
        )

    def test_nested_rejects_empty_elements_before_device_gather(self) -> None:
        values = _nested_from_parts((torch.empty(0, 3), torch.randn(3, 3)))
        masks = _nested_from_parts((torch.empty(0, dtype=torch.bool), torch.ones(3, dtype=torch.bool)))

        with _guard(), pytest.raises(ValueError, match="at least one atom"):
            LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(values, masks)


class TestLocalAtomGeometryBuilder:
    def test_geometry(self) -> None:
        positions = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0], [3.0, 0, 0], [0, 0, 4.0], [5.0, 0, 0]]])
        spaces = torch.tensor([[0, 0, 1, 1, 1]])

        geometry = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)(positions, spaces)

        assert geometry.distance_vectors.shape == (1, 3, 2, 4, 3)
        torch.testing.assert_close(geometry.distance_vectors[0, 0, 0, 2], torch.tensor([1.0, -2.0, 0.0]))
        assert geometry.distance_mask[0, 0, :, :, 0].tolist() == [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]

    def test_nested_dynamic_fullgraph_reuses_one_frame(self) -> None:
        builder = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)
        counter = CompileCounter()
        compiled = torch.compile(
            lambda positions, spaces, mask: builder(positions, spaces, mask).distance_vectors,
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        for lengths in ((3, 9), (5, 7)):
            positions = _nested_from_parts(tuple(torch.randn(length, 3) for length in lengths))
            spaces = _nested_from_parts(tuple(torch.arange(length) // 3 for length in lengths))
            masks = _nested_from_parts(tuple(torch.arange(length) % 4 != 1 for length in lengths))
            with _guard():
                output = compiled(positions, spaces, masks)
            assert output.shape == (sum((length + 1) // 2 for length in lengths), 2, 4, 3)
        assert counter.frame_count == 1


def _nested_from_parts(parts: tuple[torch.Tensor, ...]) -> NestedTensor:
    reference = NestedTensor([torch.empty(1, *part.shape[1:], dtype=part.dtype) for part in parts], ragged_dims=(0,))
    lengths = torch.tensor(tuple(part.shape[0] for part in parts), dtype=torch.int64)
    return reference.packed_with_lengths(torch.cat(parts), lengths)


def _guard():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )
