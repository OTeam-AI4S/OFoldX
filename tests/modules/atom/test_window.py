from __future__ import annotations

import pytest
import torch

from ofoldx.modules.atom import LocalAtomGeometryBuilder, LocalAtomWindowing
from ofoldx.modules.atom.window import gather_key_windows


def test_windowing_shapes_padding_and_mask() -> None:
    # 5 atoms -> ceil(5/2)=3 query windows; key windows overlap by (4-2)/2=1 on each side.
    tensor = torch.arange(5 * 2, dtype=torch.float32).reshape(1, 5, 2)

    windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(tensor)

    assert windows.query.shape == (1, 3, 2, 2)
    assert windows.key.shape == (1, 3, 4, 2)
    assert windows.info.attention_mask.shape == (1, 3, 2, 4)
    assert windows.info.key_index is None

    # Last query window is short one atom, so it pads with a zero row.
    atoms = tensor[0]
    assert torch.equal(windows.query[0, 0], atoms[0:2])
    assert torch.equal(windows.query[0, 2, 0], atoms[4])
    assert torch.equal(windows.query[0, 2, 1], torch.zeros(2))

    # Key window i spans atoms [2i-1, 2i+2], zero-padded past the [0, 5) atom range.
    assert torch.equal(windows.key[0, 0], torch.tensor([[0.0, 0.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    assert torch.equal(windows.key[0, 2], torch.tensor([[6.0, 7.0], [8.0, 9.0], [0.0, 0.0], [0.0, 0.0]]))

    # A pair is valid iff the query atom and the (left-shifted) key atom both lie in [0, num_atoms).
    expected_mask = torch.tensor(
        [
            [[0, 1, 1, 1], [0, 1, 1, 1]],
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[1, 1, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(windows.info.attention_mask[0], expected_mask)


def test_windowing_uses_row_specific_atom_masks() -> None:
    tensor = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 1000.0, 2000.0, 3000.0],
        ]
    ).unsqueeze(-1)
    atom_mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False],
        ]
    )

    windows = LocalAtomWindowing(num_query_atoms=2, num_key_atoms=4)(tensor, atom_mask)

    key_index = windows.info.key_index
    assert key_index is not None
    assert torch.equal(
        key_index,
        torch.tensor(
            [
                [[0, 1, 2, 3], [1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7]],
                [[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            ]
        ),
    )
    assert torch.equal(windows.key[0, -1, :, 0], tensor[0, 4:8, 0])
    assert torch.equal(windows.key[1, -1, :, 0], tensor[1, 1:5, 0])
    assert torch.equal(
        windows.info.attention_mask[1].any(-1),
        torch.tensor([[True, True], [True, True], [True, False], [False, False]]),
    )


def test_geometry() -> None:
    # Distinct positions and two reference spaces so neither the displacements nor the same-space mask is true
    # by construction; padded query/key slots are excluded by the window mask.
    builder = LocalAtomGeometryBuilder(num_query_atoms=2, num_key_atoms=4)
    positions = torch.tensor([[[1.0, 0, 0], [0, 2.0, 0], [3.0, 0, 0], [0, 0, 4.0], [5.0, 0, 0]]])
    reference_space_id = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)

    geometry = builder(positions, reference_space_id)

    assert geometry.distance_vectors.shape == (1, 3, 2, 4, 3)
    assert geometry.distance_mask.shape == (1, 3, 2, 4, 1)
    assert geometry.window_info.attention_mask.shape == (1, 3, 2, 4)

    # Hand-derived key-minus-query displacements per window. The builder returns the AF3/Protenix-convention
    # query-minus-key (matching the dense encode_atom_pair path), i.e. the negation of this table.
    key_minus_query = torch.tensor(
        [
            [
                [
                    [[-1.0, 0, 0], [0, 0, 0], [-1.0, 2.0, 0], [2.0, 0, 0]],
                    [[0, -2.0, 0], [1.0, -2.0, 0], [0, 0, 0], [3.0, -2.0, 0]],
                ],
                [
                    [[-3.0, 2.0, 0], [0, 0, 0], [-3.0, 0, 4.0], [2.0, 0, 0]],
                    [[0, 2.0, -4.0], [3.0, 0, -4.0], [0, 0, 0], [5.0, 0, -4.0]],
                ],
                [
                    [[-5.0, 0, 4.0], [0, 0, 0], [-5.0, 0, 0], [-5.0, 0, 0]],
                    [[0, 0, 4.0], [5.0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
            ]
        ]
    )
    torch.testing.assert_close(geometry.distance_vectors, -key_minus_query)

    # A pair is same-space iff space_id[query atom] == space_id[key atom]; padded atoms read as space id 0.
    expected_mask = torch.tensor(
        [
            [
                [[0, 1, 1, 0], [0, 1, 1, 0]],
                [[0, 1, 1, 1], [0, 1, 1, 1]],
                [[1, 1, 0, 0], [0, 0, 0, 0]],
            ]
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)
    torch.testing.assert_close(geometry.distance_mask, expected_mask)


@pytest.mark.parametrize(("num_query_atoms", "num_key_atoms"), [(4, 8), (32, 128)])
@pytest.mark.parametrize("num_atoms", [1, 5, 31, 32, 33, 100])
def test_query_key_windows_match_reference(num_atoms: int, num_query_atoms: int, num_key_atoms: int) -> None:
    torch.manual_seed(0)
    feature_dim = 3
    tensor = torch.randn(1, num_atoms, feature_dim)
    windows = LocalAtomWindowing(num_query_atoms, num_key_atoms).forward(tensor)
    keys = gather_key_windows(tensor, num_query_atoms, num_key_atoms)

    num_windows = (num_atoms + num_query_atoms - 1) // num_query_atoms
    key_offset = (num_key_atoms - num_query_atoms) // 2
    assert tuple(windows.info.attention_mask.shape) == (1, num_windows, num_query_atoms, num_key_atoms)

    assert torch.equal(windows.key, keys)
    for window in range(num_windows):
        for query in range(num_query_atoms):
            query_atom = window * num_query_atoms + query
            if query_atom < num_atoms:
                assert torch.equal(windows.query[0, window, query], tensor[0, query_atom])
            else:
                assert torch.equal(windows.query[0, window, query], torch.zeros(feature_dim))
            for key in range(num_key_atoms):
                key_atom = window * num_query_atoms - key_offset + key
                valid = query_atom < num_atoms and 0 <= key_atom < num_atoms
                assert bool(windows.info.attention_mask[0, window, query, key]) == valid
                if 0 <= key_atom < num_atoms:
                    assert torch.equal(keys[0, window, key], tensor[0, key_atom])
                else:
                    assert torch.equal(keys[0, window, key], torch.zeros(feature_dim))


@pytest.mark.parametrize(
    ("num_query_atoms", "num_key_atoms"),
    [(4, 2), (3, 4), (2, 5)],  # key < query, odd query, odd key
)
def test_invalid_inputs(num_query_atoms: int, num_key_atoms: int) -> None:
    with pytest.raises(ValueError, match="must be even"):
        LocalAtomWindowing(num_query_atoms=num_query_atoms, num_key_atoms=num_key_atoms)
