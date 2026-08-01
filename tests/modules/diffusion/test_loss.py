from __future__ import annotations

from typing import NamedTuple, cast

import torch
from danling.tensors import NestedTensor, nested_execution_guard

from ofoldx.modules.diffusion.loss import edm_denoising_loss


class _Prediction(NamedTuple):
    coordinates: torch.Tensor
    augmented_target_coordinates: torch.Tensor
    noise_level: torch.Tensor


def _forbid_ragged_materialization():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def _nested_prediction(
    atom_lengths: tuple[int, ...],
) -> tuple[_Prediction, NestedTensor]:
    samples = 2
    generator = torch.Generator().manual_seed(7)
    targets = NestedTensor(
        [torch.randn(samples, atoms, 3, generator=generator) for atoms in atom_lengths],
        ragged_dims=(1,),
    )
    coordinates = NestedTensor(
        [target + 0.1 * torch.randn(target.shape, generator=generator) for target in targets],
        ragged_dims=(1,),
    )
    resolved = NestedTensor(
        [torch.arange(atoms).remainder(3) != 2 for atoms in atom_lengths],
        ragged_dims=(0,),
    )
    return (
        _Prediction(
            coordinates=coordinates,
            augmented_target_coordinates=targets,
            noise_level=torch.tensor([[0.5, 1.0], [0.75, 1.25]]),
        ),
        resolved,
    )


def test_edm_denoising_loss_ragged_matches_per_structure_dense() -> None:
    prediction, resolved = _nested_prediction((4, 7))

    with _forbid_ragged_materialization():
        actual = edm_denoising_loss(prediction, resolved, sigma_data=16.0)

    expected = torch.stack(
        [
            edm_denoising_loss(
                _Prediction(
                    coordinates=coordinates.unsqueeze(0),
                    augmented_target_coordinates=target.unsqueeze(0),
                    noise_level=prediction.noise_level[index : index + 1],
                ),
                mask.unsqueeze(0),
                sigma_data=16.0,
            )
            for index, (coordinates, target, mask) in enumerate(
                zip(prediction.coordinates, prediction.augmented_target_coordinates, resolved, strict=True)
            )
        ]
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_edm_denoising_loss_ragged_fullgraph_backward() -> None:
    template, resolved = _nested_prediction((4, 7))
    coordinate_layout = cast(NestedTensor, template.coordinates)
    eager_values = coordinate_layout.concat.detach().requires_grad_()
    compiled_values = eager_values.detach().clone().requires_grad_()

    def loss(
        coordinate_values: torch.Tensor,
        coordinate_layout: NestedTensor,
        targets: NestedTensor,
        noise_level: torch.Tensor,
        mask: NestedTensor,
    ) -> torch.Tensor:
        prediction = _Prediction(
            coordinates=coordinate_layout.packed_like(coordinate_values),
            augmented_target_coordinates=targets,
            noise_level=noise_level,
        )
        return edm_denoising_loss(prediction, mask, sigma_data=16.0)

    arguments = (
        coordinate_layout,
        template.augmented_target_coordinates,
        template.noise_level,
        resolved,
    )
    eager_loss = loss(eager_values, *arguments)
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True, dynamic=True)
    with _forbid_ragged_materialization():
        compiled_loss = compiled(compiled_values, *arguments)
        compiled_loss.backward()
    eager_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_values.grad, eager_values.grad)
