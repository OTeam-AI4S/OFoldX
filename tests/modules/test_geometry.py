from __future__ import annotations

import torch

from ofoldx.modules.geometry import (
    rigid_from_three_points,
    uniform_random_rotation,
    virtual_beta_carbon,
    weighted_rigid_align,
)


def test_rigid_from_three_points() -> None:
    rotation, translation = rigid_from_three_points(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.zeros(3),
        torch.tensor([1.0, 0.0, 0.0]),
    )

    torch.testing.assert_close(rotation, torch.eye(3))
    torch.testing.assert_close(translation, torch.zeros(3))


def test_virtual_beta_carbon() -> None:
    nitrogen = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])
    alpha_carbon = nitrogen + torch.tensor([1.0, 0.0, 0.0])
    carbon = alpha_carbon + torch.tensor([0.0, 1.0, 0.0])

    beta_carbon = virtual_beta_carbon(nitrogen, alpha_carbon, carbon)

    expected_offset = torch.tensor([1.56802827, -0.54067466, -0.58273431])
    torch.testing.assert_close(beta_carbon, nitrogen + expected_offset)


def test_weighted_rigid_align() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target = (source.detach() @ rotation.mT + source.new_tensor([2.0, -1.0, 4.0])).requires_grad_()
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)

    aligned = weighted_rigid_align(source=source, target=target, weight=weight)

    torch.testing.assert_close(aligned, target)
    assert not aligned.requires_grad


def test_uniform_random_rotation() -> None:
    device = torch.device("cpu")
    rotations = uniform_random_rotation(
        4,
        device=device,
        dtype=torch.float64,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    repeated = uniform_random_rotation(
        4,
        device=device,
        dtype=torch.float64,
        generator=torch.Generator(device=device).manual_seed(7),
    )

    assert rotations.shape == (4, 3, 3)
    assert rotations.dtype == torch.float64
    assert rotations.device == device
    torch.testing.assert_close(rotations, repeated)
    torch.testing.assert_close(rotations @ rotations.mT, torch.eye(3, dtype=rotations.dtype).expand_as(rotations))
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(4, dtype=rotations.dtype))


def test_random_rotation_compile() -> None:
    def dynamic_rotation(reference: torch.Tensor) -> torch.Tensor:
        return uniform_random_rotation(
            torch.numel(reference[..., 0, 0]),
            device=reference.device,
            dtype=reference.dtype,
        )

    compiled = torch.compile(dynamic_rotation, backend="eager", dynamic=True, fullgraph=True)
    assert compiled(torch.ones(2, 4, 5, 3)).shape == (8, 3, 3)
    assert compiled(torch.ones(3, 2, 4, 5, 3)).shape == (24, 3, 3)
