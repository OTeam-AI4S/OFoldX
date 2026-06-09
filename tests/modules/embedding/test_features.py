from __future__ import annotations

import pytest
import torch

from ofoldx.modules.embedding import pack_atom_features, pack_token_tail


def test_packers_layout() -> None:
    atom_mask = torch.tensor([[True, False]])
    element = torch.arange(6.0).reshape(1, 2, 3)
    atom_name = torch.arange(16.0).reshape(1, 2, 2, 4)
    atom_features = pack_atom_features(
        atom_mask,
        element,
        atom_name,
        element_channels=3,
        atom_name_characters_count=2,
        atom_name_character_channels=4,
    )
    assert atom_features.shape == (1, 2, 12)
    assert torch.equal(atom_features[..., 0], atom_mask.to(dtype=atom_features.dtype))
    assert torch.equal(atom_features[..., 1:4], element)
    assert torch.equal(atom_features[..., 4:], atom_name.reshape(1, 2, 8))

    # distinct per-element values so layout assertions catch block-placement bugs.
    residue_type = torch.arange(6.0).reshape(1, 2, 3)
    profile = torch.arange(10.0, 18.0).reshape(1, 2, 4)
    deletion_mean = torch.tensor([[[20.0], [21.0]]])
    token_features = pack_token_tail(
        residue_type,
        profile,
        deletion_mean,
        residue_type_channels=3,
        profile_channels=4,
        deletion_mean_channels=1,
    )
    assert token_features.shape == (1, 2, 8)
    # rows are residue_type || profile || deletion_mean concatenated on the channel axis.
    expected_token_features = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 20.0],
                [3.0, 4.0, 5.0, 14.0, 15.0, 16.0, 17.0, 21.0],
            ]
        ]
    )
    torch.testing.assert_close(token_features, expected_token_features)


@pytest.mark.parametrize(
    ("pack", "match"),
    [
        (
            lambda: pack_atom_features(
                torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 4), torch.zeros(1, 2, 2, 4)
            ),
            "element_features",
        ),
        (
            lambda: pack_atom_features(
                torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, 128), torch.zeros(1, 2, 2, 64)
            ),
            "atom_name_characters",
        ),
        (
            lambda: pack_token_tail(torch.zeros(1, 2, 8), torch.zeros(1, 2, 32), torch.zeros(1, 2, 1)),
            "channels",
        ),
        (
            lambda: pack_token_tail(torch.zeros(1, 2, 32), torch.zeros(1, 3, 32), torch.zeros(1, 2, 1)),
            "leading dimensions must match",
        ),
    ],
)
def test_invalid_inputs(pack, match) -> None:
    with pytest.raises(ValueError, match=match):
        pack()
