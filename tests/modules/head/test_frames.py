from __future__ import annotations

import math

import pytest
import torch

from ofoldx.modules.head.frames import (
    express_coords_in_frame,
    frame_aligned_error_squared,
    gather_frame_atoms,
)


@pytest.fixture
def coords() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 3)


@pytest.fixture
def frames() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 3, 3)


def _rigid(angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([1.0, -2.0, 3.0])
    return rotation, translation


def test_express_in_frame(coords: torch.Tensor, frames: torch.Tensor) -> None:
    expressed = express_coords_in_frame(coords, frames)
    assert expressed.shape == (frames.shape[0], coords.shape[0], 3)

    # Frame origin (atom b) maps to its own basis origin; orthonormal basis preserves distance from b.
    origins = frames[:, 1, :]
    origins_in_frames = express_coords_in_frame(origins, frames)
    own_origin = origins_in_frames.diagonal(dim1=0, dim2=1).T
    torch.testing.assert_close(own_origin, torch.zeros_like(own_origin), atol=1e-5, rtol=0)

    distance = (coords[None] - origins[:, None]).norm(dim=-1)
    torch.testing.assert_close(expressed.norm(dim=-1), distance, atol=1e-5, rtol=0)


@pytest.mark.parametrize("angle", [0.0, 0.9, -2.3])
def test_express_in_frame_rigid_invariant(coords: torch.Tensor, frames: torch.Tensor, angle: float) -> None:
    # One rigid transform applied to both coordinate and frame leaves the local expression fixed: PAE is pose-invariant.
    rotation, translation = _rigid(angle)
    base = express_coords_in_frame(coords, frames)
    moved = express_coords_in_frame(coords @ rotation.T + translation, frames @ rotation.T + translation)
    torch.testing.assert_close(base, moved, atol=1e-5, rtol=0)


def test_frame_aligned_error_shape() -> None:
    torch.manual_seed(0)
    coords = torch.randn(1, 5, 3)
    frame_index = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4]]])
    frames = gather_frame_atoms(coords, frame_index)
    assert frames.shape == (1, 3, 3, 3)
    centers = coords[:, :3]  # one center per token

    # Identical pred/true poses give exactly zero aligned error everywhere.
    squared = frame_aligned_error_squared(centers.unsqueeze(-3), centers, frames.unsqueeze(-4), frames)
    assert squared.shape == (1, 1, 3, 3)
    torch.testing.assert_close(squared, torch.zeros_like(squared), atol=1e-5, rtol=0)

    # Translating pred and its frames together moves nothing in the local expression: error stays zero.
    shift = torch.tensor([2.0, -1.0, 0.5])
    squared_shift = frame_aligned_error_squared(
        (centers + shift).unsqueeze(-3), centers, (frames + shift).unsqueeze(-4), frames
    )
    torch.testing.assert_close(squared_shift, torch.zeros_like(squared_shift), atol=1e-5, rtol=0)


def test_frame_aligned_error_isometry_value() -> None:
    # Identical pred/true pose + orthonormal basis -> expression is an isometry, so each token's error is
    # its squared pred/true coordinate distance in every frame; expected is hand-derived from that distance.
    frames = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]],  # arbitrary distinct pose
        ]
    )
    true_coords = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    pred_coords = torch.tensor([[4.0, 6.0, 3.0], [0.0, 0.0, 5.0]])  # token diffs [3,4,0]->25, [0,0,5]->25

    squared = frame_aligned_error_squared(pred_coords.unsqueeze(-3), true_coords, frames.unsqueeze(-4), frames)
    assert squared.shape == (1, 2, 2)
    # Squared distance per token (25, 25), identical across both frames by the isometry above.
    expected = torch.tensor([[[25.0, 25.0], [25.0, 25.0]]])
    torch.testing.assert_close(squared, expected, atol=1e-4, rtol=0)  # tol covers fp normalize round-off
