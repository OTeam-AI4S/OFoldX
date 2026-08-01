from __future__ import annotations

import pytest
import torch

from ofoldx.modules.atom import LocalAtomGeometryBuilder, LocalAtomWindowing


class TestLocalAtomWindowing:
    def test_padding(self) -> None:
        tensor = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)

        windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(tensor)

        assert windows.query.shape == (1, 3, 2, 2)
        assert windows.key.shape == (1, 3, 4, 2)
        assert windows.info.key_index is None
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
