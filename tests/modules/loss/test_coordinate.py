from __future__ import annotations

import math

import pytest
import torch

from ofoldx.modules.loss import CoordinateMseLoss, DiffusionDenoisingLoss


@pytest.fixture
def mse_loss() -> CoordinateMseLoss:
    return CoordinateMseLoss()


@pytest.fixture
def denoise_loss() -> DiffusionDenoisingLoss:
    return DiffusionDenoisingLoss()


@pytest.mark.parametrize("num_samples", [1, 2])
def test_coordinate_mse(mse_loss: CoordinateMseLoss, num_samples: int) -> None:
    # Non-uniform per-atom errors so a wrong normalizer / mis-weighted axis is caught (uniform errors
    # pass any normalizer). pred=0 makes each error equal the target; valid atoms have squared norms
    # 1, 4, 16 (atom 2 masked out), so the masked element-mean is (1+4+16)/(N_valid*3) = 7/3, and the
    # sample axis does not change it.
    pred_coords = torch.zeros(1, num_samples, 4, 3, requires_grad=True)
    target_positions = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0]]])
    target_mask = torch.tensor([[True, True, False, True]])

    value = mse_loss(pred_coords, target_positions, target_mask)

    assert value.shape == ()
    torch.testing.assert_close(value, torch.tensor(7.0 / 3.0))
    value.backward()
    assert pred_coords.grad is not None


def test_invalid_inputs(mse_loss: CoordinateMseLoss) -> None:
    # pred_coords must carry a sample axis: dim == target.dim() + 1.
    with pytest.raises(ValueError, match="S, N_atom"):
        mse_loss(torch.zeros(1, 4, 3), torch.ones(1, 4, 3), torch.ones(1, 4, dtype=torch.bool))


def test_diffusion_denoising_invariants(denoise_loss: DiffusionDenoisingLoss) -> None:
    torch.manual_seed(0)
    noise_level = torch.full((1, 2), 1.5)

    # Masked (unresolved) atoms must not affect the loss or the weighted-Kabsch alignment: an extra
    # atom with garbage prediction and zero target, masked out, leaves the loss unchanged.
    pred = torch.randn(1, 2, 3, 3)
    target = torch.randn(1, 2, 3, 3)
    base = denoise_loss(pred, target, noise_level, torch.ones(1, 3))
    pred_extra = torch.cat([pred, torch.randn(1, 2, 1, 3) * 100.0], dim=-2)
    target_extra = torch.cat([target, torch.zeros(1, 2, 1, 3)], dim=-2)
    resolved = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    with_unresolved = denoise_loss(pred_extra, target_extra, noise_level, resolved)
    # atol loosened for float32 Kabsch (SVD) accumulation noise.
    torch.testing.assert_close(base, with_unresolved, atol=1e-4, rtol=0.0)

    # The Algorithm 28 alignment removes global pose: rotating/translating the ground truth must
    # leave the loss unchanged (an unaligned MSE would change drastically).
    denoised = torch.randn(1, 2, 8, 3, requires_grad=True)
    aligned_target = denoised.detach() + 0.05 * torch.randn(1, 2, 8, 3)
    atom_mask = torch.ones(1, 8)
    angle = 0.7
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved_target = aligned_target @ rotation.T + torch.tensor([3.0, -2.0, 1.0])
    pose_base = denoise_loss(denoised, aligned_target, noise_level, atom_mask)
    moved = denoise_loss(denoised, moved_target, noise_level, atom_mask)
    # atol loosened for float32 Kabsch (SVD) accumulation noise.
    torch.testing.assert_close(pose_base, moved, atol=1e-4, rtol=0.0)

    assert pose_base.shape == ()
    pose_base.backward()
    assert denoised.grad is not None
