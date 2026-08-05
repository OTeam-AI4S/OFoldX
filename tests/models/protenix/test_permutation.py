from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from ofoldx.models.protenix.permutation import (
    ProtenixPredictionPermutation,
    _atom_to_token_permutation,
    match_protenix_prediction,
    match_protenix_truth,
)


def _target(
    coordinates: Tensor,
    *,
    molecule_id: Tensor | None = None,
    molecule_equivalence_index: Tensor | None = None,
    molecule_atom_index: Tensor | None = None,
    coordinate_mask: Tensor | None = None,
    representative_atom_mask: Tensor | None = None,
    permutation_pool_mask: Tensor | None = None,
    atom_molecule_type: Tensor | None = None,
    symmetry_atom_index: Tensor | None = None,
    symmetry_permutation: Tensor | None = None,
) -> SimpleNamespace:
    atom_count = coordinates.shape[0]
    molecule_id = torch.zeros(atom_count, dtype=torch.long) if molecule_id is None else molecule_id
    molecule_equivalence_index = (
        molecule_id.clone() if molecule_equivalence_index is None else molecule_equivalence_index
    )
    molecule_atom_index = torch.arange(atom_count) if molecule_atom_index is None else molecule_atom_index
    coordinate_mask = torch.ones(atom_count, dtype=torch.bool) if coordinate_mask is None else coordinate_mask
    representative_atom_mask = (
        torch.ones(atom_count, dtype=torch.bool)
        if representative_atom_mask is None
        else representative_atom_mask
    )
    permutation_pool_mask = (
        torch.ones(atom_count, dtype=torch.bool) if permutation_pool_mask is None else permutation_pool_mask
    )
    atom_molecule_type = (
        torch.ones(atom_count, dtype=torch.long) if atom_molecule_type is None else atom_molecule_type
    )
    if symmetry_atom_index is None:
        symmetry_atom_index = torch.empty(0, 0, dtype=torch.long)
    if symmetry_permutation is None:
        symmetry_permutation = torch.empty(0, 0, 0, dtype=torch.long)
    return SimpleNamespace(
        coordinates=coordinates.unsqueeze(0),
        coordinate_mask=coordinate_mask.unsqueeze(0),
        molecule_id=molecule_id.unsqueeze(0),
        molecule_equivalence_index=molecule_equivalence_index.unsqueeze(0),
        molecule_atom_index=molecule_atom_index.unsqueeze(0),
        representative_atom_mask=representative_atom_mask.unsqueeze(0),
        permutation_pool_mask=permutation_pool_mask.unsqueeze(0),
        atom_molecule_type=atom_molecule_type.unsqueeze(0),
        symmetry_atom_index=symmetry_atom_index.unsqueeze(0),
        symmetry_permutation=symmetry_permutation.unsqueeze(0),
    )


def _input(atom_index: Tensor, atom_to_token: Tensor | None = None) -> SimpleNamespace:
    output = {"atom_index": atom_index.unsqueeze(0)}
    if atom_to_token is not None:
        output["atom_to_token"] = atom_to_token.unsqueeze(0)
    return SimpleNamespace(**output)


def _homodimer() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    first = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 1.0],
            [0.7, 0.4, 0.6],
        ]
    )
    angle = torch.tensor(torch.pi / 3)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    second = first @ rotation + torch.tensor([5.0, 2.0, 1.0])
    coordinates = torch.cat((first, second))
    molecule_id = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    molecule_atom_index = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    equivalence = torch.zeros(10, dtype=torch.long)
    return coordinates, molecule_id, molecule_atom_index, equivalence


def test_shared_token_pair_gather_uses_each_batch_sample_row_and_column_permutation() -> None:
    token_permutation = torch.tensor(
        [
            [[0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]],
            [[3, 2, 1, 0], [0, 2, 3, 1], [0, 1, 2, 3]],
        ]
    )
    result = ProtenixPredictionPermutation(
        coordinates=torch.empty(2, 3, 0, 3),
        atom_permutation=torch.empty(2, 3, 0, dtype=torch.long),
        token_permutation=token_permutation,
    )
    values = torch.arange(2 * 4 * 4 * 2, dtype=torch.float64).reshape(2, 4, 4, 2)

    gathered = result.gather_shared_token_pair_axes(values)

    expected = torch.stack(
        [
            torch.stack(
                [
                    values[batch_index]
                    .index_select(0, token_permutation[batch_index, sample_index])
                    .index_select(1, token_permutation[batch_index, sample_index])
                    for sample_index in range(3)
                ]
            )
            for batch_index in range(2)
        ]
    )
    assert gathered.shape == (2, 3, 4, 4, 2)
    assert gathered.dtype == values.dtype
    assert gathered.device == values.device
    assert torch.equal(gathered, expected)
    assert not torch.equal(gathered[0, 0], gathered[0, 1])
    assert not torch.equal(gathered[0, 1, 0, 2], gathered[0, 1, 2, 0])


@pytest.mark.parametrize(
    "values",
    [
        torch.empty(2, 4),
        torch.empty(1, 4, 4),
        torch.empty(2, 3, 4),
        torch.empty(2, 4, 3, 2),
        torch.empty(2, 4, 4, 2, 1).transpose(2, 3),
    ],
)
def test_shared_token_pair_gather_rejects_nonmatching_shapes(values: Tensor) -> None:
    result = ProtenixPredictionPermutation(
        coordinates=torch.empty(2, 3, 0, 3),
        atom_permutation=torch.empty(2, 3, 0, dtype=torch.long),
        token_permutation=torch.arange(4).expand(2, 3, 4),
    )

    with pytest.raises(ValueError, match=r"shape \(B, N, N"):
        result.gather_shared_token_pair_axes(values)


@pytest.mark.parametrize("cropped", [False, True])
def test_swapped_homodimer_matches_full_and_cropped_pools(cropped: bool) -> None:
    coordinates, molecule_id, molecule_atom_index, equivalence = _homodimer()
    if cropped:
        atom_index = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
        pool_mask = torch.tensor([True, True, True, True, False, True, True, True, True, False])
    else:
        atom_index = torch.arange(10)
        pool_mask = torch.ones(10, dtype=torch.bool)
    expected_index = torch.where(atom_index < 5, atom_index + 5, atom_index - 5)
    predicted = coordinates.index_select(0, expected_index).unsqueeze(0).unsqueeze(0)
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
        representative_atom_mask=torch.tensor([True, True, True, True, False] * 2),
        permutation_pool_mask=pool_mask,
    )

    identity = coordinates.index_select(0, atom_index)
    assert not torch.equal(predicted[0, 0], identity)

    matched, mask = match_protenix_truth(predicted, target, _input(atom_index))

    torch.testing.assert_close(matched[0], predicted[0, 0], atol=1e-5, rtol=1e-5)
    assert torch.linalg.vector_norm(predicted[0, 0] - matched[0]) < torch.linalg.vector_norm(
        predicted[0, 0] - identity
    )
    assert bool(mask.all())


def test_local_symmetric_atoms_select_lowest_fixed_frame_rmsd() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 1.0, 0.3],
            [1.0, 3.0, 0.7],
        ]
    )
    predicted = coordinates.clone()
    predicted[[4, 5]] = predicted[[5, 4]]
    predicted.requires_grad_()
    target = _target(
        coordinates,
        symmetry_atom_index=torch.tensor([[0, 1, 2, 3, 4, 5]]),
        symmetry_permutation=torch.tensor([[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4]]]),
    )

    matched, mask = match_protenix_truth(predicted.unsqueeze(0), target, _input(torch.arange(6)))

    torch.testing.assert_close(matched[0], predicted.detach(), atol=1e-5, rtol=1e-5)
    assert bool(mask.all())
    assert not matched.requires_grad
    assert not mask.requires_grad


def test_local_symmetric_atom_mask_follows_selected_coordinates() -> None:
    coordinates = torch.tensor(
        [
            [-5.0, -5.0, -5.0],
            [-5.0, -5.0, 5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, 5.0, 5.0],
            [5.0, -5.0, -5.0],
            [5.0, -5.0, 5.0],
            [5.0, 5.0, -5.0],
            [5.0, 5.0, 5.0],
            [1.5, 0.0, 0.5],
            [-1.5, 0.0, 0.7],
        ]
    )
    predicted = coordinates.clone()
    predicted[[8, 9]] = predicted[[9, 8]]
    target = _target(
        coordinates,
        coordinate_mask=torch.tensor([True, True, True, True, True, True, True, True, False, True]),
        symmetry_atom_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        symmetry_permutation=torch.tensor(
            [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]]
        ),
    )

    matched, mask = match_protenix_truth(predicted.unsqueeze(0), target, _input(torch.arange(10)))

    torch.testing.assert_close(matched[0, 8:], coordinates[[9, 8]])
    assert mask[0, 8:].tolist() == [True, False]


def test_equal_local_rmsd_deterministically_retains_identity() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ]
    )
    original_mask = torch.tensor([True, True, True, True, True, False])
    target = _target(
        coordinates,
        coordinate_mask=original_mask,
        symmetry_atom_index=torch.tensor([[0, 1, 2, 3, 4, 5]]),
        symmetry_permutation=torch.tensor([[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4]]]),
    )

    first = match_protenix_truth(coordinates.unsqueeze(0), target, _input(torch.arange(6)))
    second = match_protenix_truth(coordinates.unsqueeze(0), target, _input(torch.arange(6)))

    assert first[1][0].tolist() == original_mask.tolist()
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_padding_is_invalid_and_padded_symmetry_rows_are_ignored() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    atom_index = torch.tensor([0, 1, 2, -1])
    predicted = torch.cat((coordinates, torch.tensor([[99.0, 99.0, 99.0]])))
    target = _target(
        coordinates,
        symmetry_atom_index=torch.tensor([[-1, -1]]),
        symmetry_permutation=torch.tensor([[[-1, -1]]]),
    )

    matched, mask = match_protenix_truth(predicted.unsqueeze(0), target, _input(atom_index))

    torch.testing.assert_close(matched[0, :3], coordinates)
    assert matched[0, 3].tolist() == [0.0, 0.0, 0.0]
    assert mask.tolist() == [[True, True, True, False]]


def test_fewer_than_three_resolved_atoms_returns_canonical_identity() -> None:
    coordinates, molecule_id, molecule_atom_index, equivalence = _homodimer()
    predicted = torch.cat((coordinates[5:], coordinates[:5])).unsqueeze(0)
    coordinate_mask = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
        coordinate_mask=coordinate_mask,
    )

    matched, mask = match_protenix_truth(predicted, target, _input(torch.arange(10)))

    torch.testing.assert_close(matched[0], coordinates)
    assert mask[0].tolist() == coordinate_mask.tolist()


def test_training_matching_is_unlimited_by_default_with_optional_twenty_molecule_guard() -> None:
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coordinates = torch.cat([base + torch.tensor([3.0 * index, 0.2 * index, 0.0]) for index in range(21)])
    molecule_id = torch.arange(21).repeat_interleave(4)
    molecule_atom_index = torch.arange(4).repeat(21)
    equivalence = torch.zeros(84, dtype=torch.long)
    predicted = coordinates.clone()
    predicted[:4] = coordinates[4:8]
    predicted[4:8] = coordinates[:4]
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
    )

    unlimited, _ = match_protenix_truth(predicted.unsqueeze(0), target, _input(torch.arange(84)))
    guarded, _ = match_protenix_truth(
        predicted.unsqueeze(0),
        target,
        _input(torch.arange(84)),
        max_molecules=20,
    )

    torch.testing.assert_close(unlimited[0], predicted, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(guarded[0], coordinates)


@pytest.mark.parametrize("resolved_tail", [False, True])
def test_equivalent_or_worse_heuristic_assignment_reverts_to_identity(
    resolved_tail: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representative = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    coordinates = torch.cat(
        (
            representative,
            torch.tensor([[10.0, 0.0, 0.0]]),
            representative,
            torch.tensor([[20.0, 0.0, 0.0]]),
        )
    )
    coordinate_mask = torch.tensor([True, True, True, True, resolved_tail] * 2)
    target = _target(
        coordinates,
        molecule_id=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        molecule_equivalence_index=torch.zeros(10, dtype=torch.long),
        molecule_atom_index=torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        coordinate_mask=coordinate_mask,
        representative_atom_mask=torch.tensor([True, True, True, True, False] * 2),
    )
    monkeypatch.setattr("ofoldx.models.protenix.permutation.random.choice", lambda values: values[-1])

    matched, mask = match_protenix_truth(coordinates.unsqueeze(0), target, _input(torch.arange(10)))

    torch.testing.assert_close(matched[0], coordinates)
    assert mask[0].tolist() == coordinate_mask.tolist()


def test_bfloat16_prediction_does_not_quantize_float32_truth() -> None:
    coordinates = torch.tensor(
        [
            [0.0012345, 1.0034567, 2.0056789],
            [3.0078912, 4.0091234, 5.0013579],
            [6.0024681, 7.0035791, 8.0046812],
            [9.0057926, 10.006813, 11.007924],
        ],
        dtype=torch.float32,
    )
    predicted = coordinates.to(dtype=torch.bfloat16).unsqueeze(0)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        matched, mask = match_protenix_truth(predicted, _target(coordinates), _input(torch.arange(4)))

    assert matched.dtype == torch.float32
    assert torch.equal(matched[0], coordinates)
    assert bool(mask.all())


def test_local_symmetry_skips_group_with_fewer_than_three_resolved_atoms() -> None:
    coordinates = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, -4.0, 0.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 12.0],
        ]
    )
    predicted = coordinates.clone()
    predicted[[0, 1]] = predicted[[1, 0]]
    coordinate_mask = torch.tensor([True, True, False, False, True, True, True, True])
    target = _target(
        coordinates,
        coordinate_mask=coordinate_mask,
        symmetry_atom_index=torch.tensor([[0, 1, 2, 3]]),
        symmetry_permutation=torch.tensor([[[0, 1, 2, 3], [1, 0, 2, 3]]]),
    )

    matched, mask = match_protenix_truth(predicted.unsqueeze(0), target, _input(torch.arange(8)))

    torch.testing.assert_close(matched[0], coordinates)
    assert mask[0].tolist() == coordinate_mask.tolist()


def test_prediction_matching_keeps_nonidentity_full_bijection_on_current_axis() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    atom_index = torch.tensor([2, 0, 3, 1])
    current_truth = coordinates.index_select(0, atom_index)
    predicted = current_truth.unsqueeze(0).unsqueeze(0)

    result = match_protenix_prediction(
        predicted,
        _target(coordinates),
        _input(atom_index, torch.arange(4)),
    )

    torch.testing.assert_close(result.coordinates[0, 0], current_truth)
    assert not torch.equal(result.coordinates[0, 0], coordinates)
    assert result.atom_permutation.tolist() == [[[0, 1, 2, 3]]]
    assert result.token_permutation.tolist() == [[[0, 1, 2, 3]]]


def test_prediction_matching_accepts_non_token_major_mapping_in_representative_order() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    result = match_protenix_prediction(
        coordinates.unsqueeze(0).unsqueeze(0),
        _target(coordinates),
        _input(torch.arange(2), torch.tensor([1, 0])),
    )

    assert result.atom_permutation.tolist() == [[[0, 1]]]
    assert result.token_permutation.tolist() == [[[1, 0]]]


def test_atom_to_token_permutation_gathers_representatives_with_atom_order() -> None:
    atom_permutation = torch.tensor([2, 0, 3, 1])
    atom_to_token = torch.tensor([0, 1, 0, 1])
    representative_mask = torch.tensor([True, True, False, False])

    token_permutation = _atom_to_token_permutation(
        atom_permutation,
        atom_to_token,
        representative_mask,
        token_count=2,
    )

    assert representative_mask[atom_permutation].tolist() == [False, True, False, True]
    assert token_permutation.tolist() == [0, 1]


def test_atom_to_token_permutation_preserves_bijection_validation() -> None:
    with pytest.raises(ValueError, match="one-to-one token permutation"):
        _atom_to_token_permutation(
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            token_count=2,
        )


def test_prediction_matching_independently_permutes_samples_and_confidence_heads() -> None:
    coordinates, molecule_id, molecule_atom_index, equivalence = _homodimer()
    atom_count = coordinates.shape[0]
    swapped_atom = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    predicted = torch.stack((coordinates.index_select(0, swapped_atom), coordinates)).unsqueeze(0)
    atom_to_token = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6, 7, 7])
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
        representative_atom_mask=torch.tensor([True, True, True, True, False] * 2),
    )

    result = match_protenix_prediction(
        predicted,
        target,
        _input(torch.arange(atom_count), atom_to_token),
    )

    expected_atom = torch.stack((swapped_atom, torch.arange(atom_count))).unsqueeze(0)
    expected_token = torch.tensor([[[4, 5, 6, 7, 0, 1, 2, 3], list(range(8))]])
    torch.testing.assert_close(result.coordinates, coordinates.expand(1, 2, -1, -1))
    assert torch.equal(result.atom_permutation, expected_atom)
    assert torch.equal(result.token_permutation, expected_token)

    plddt = torch.arange(1 * 2 * atom_count * 2).reshape(1, 2, atom_count, 2)
    resolved = plddt + 1000
    pae = torch.arange(1 * 2 * 8 * 8 * 2).reshape(1, 2, 8, 8, 2)
    pde = pae + 2000
    heads = result.gather_heads(
        plddt_logits=plddt,
        resolved_logits=resolved,
        pae_logits=pae,
        pde_logits=pde,
    )
    expected_plddt = torch.stack(
        [plddt[0, sample].index_select(0, expected_atom[0, sample]) for sample in range(2)]
    ).unsqueeze(0)
    expected_pae = torch.stack(
        [
            pae[0, sample]
            .index_select(0, expected_token[0, sample])
            .index_select(1, expected_token[0, sample])
            for sample in range(2)
        ]
    ).unsqueeze(0)
    assert torch.equal(heads["plddt_logits"], expected_plddt)
    assert torch.equal(heads["resolved_logits"], expected_plddt + 1000)
    assert torch.equal(heads["pae_logits"], expected_pae)
    assert torch.equal(heads["pde_logits"], expected_pae + 2000)


def test_prediction_matching_combines_full_axis_bijection_chain_swap_and_heads() -> None:
    coordinates, molecule_id, molecule_atom_index, equivalence = _homodimer()
    atom_index = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    atom_to_token = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6, 7, 7])
    current_truth = coordinates.index_select(0, atom_index)
    predicted = coordinates.unsqueeze(0).unsqueeze(0)
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
        representative_atom_mask=torch.tensor([True, True, True, True, False] * 2),
    )
    identity_error = torch.linalg.vector_norm(predicted[0, 0] - current_truth)
    assert identity_error > 0

    result = match_protenix_prediction(
        predicted,
        target,
        _input(atom_index, atom_to_token),
    )

    expected_atom = atom_index.view(1, 1, -1)
    expected_token = torch.tensor([[[4, 5, 6, 7, 0, 1, 2, 3]]])
    torch.testing.assert_close(result.coordinates[0, 0], current_truth, atol=1e-5, rtol=1e-5)
    assert torch.linalg.vector_norm(result.coordinates[0, 0] - current_truth) < identity_error
    assert torch.equal(result.atom_permutation, expected_atom)
    assert torch.equal(result.token_permutation, expected_token)
    assert not torch.equal(result.atom_permutation[0, 0], torch.arange(10))
    assert not torch.equal(result.token_permutation[0, 0], torch.arange(8))

    plddt = torch.arange(1 * 1 * 10 * 2).reshape(1, 1, 10, 2)
    resolved = plddt + 100
    pae = torch.arange(1 * 1 * 8 * 8 * 2).reshape(1, 1, 8, 8, 2)
    pde = pae + 1000
    heads = result.gather_heads(
        plddt_logits=plddt,
        resolved_logits=resolved,
        pae_logits=pae,
        pde_logits=pde,
    )
    expected_plddt = plddt.index_select(2, expected_atom[0, 0])
    expected_pae = pae.index_select(2, expected_token[0, 0]).index_select(3, expected_token[0, 0])
    assert torch.equal(heads["plddt_logits"], expected_plddt)
    assert torch.equal(heads["resolved_logits"], expected_plddt + 100)
    assert torch.equal(heads["pae_logits"], expected_pae)
    assert torch.equal(heads["pde_logits"], expected_pae + 1000)


def test_prediction_matching_combines_chain_and_local_atom_permutations() -> None:
    first = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [2.0, 0.3, 0.1],
            [0.2, 2.4, 0.5],
        ]
    )
    second = first + torch.tensor([30.0, 20.0, 15.0])
    coordinates = torch.cat((first, second))
    expected_atom = torch.tensor([6, 7, 8, 9, 11, 10, 0, 1, 2, 3, 5, 4])
    predicted = torch.empty_like(coordinates)
    predicted[expected_atom] = coordinates
    target = _target(
        coordinates,
        molecule_id=torch.arange(2).repeat_interleave(6),
        molecule_equivalence_index=torch.zeros(12, dtype=torch.long),
        molecule_atom_index=torch.arange(6).repeat(2),
        representative_atom_mask=torch.tensor([True, True, True, True, False, False] * 2),
        symmetry_atom_index=torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
        symmetry_permutation=torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4]],
                [[6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 11, 10]],
            ]
        ),
    )

    result = match_protenix_prediction(
        predicted.unsqueeze(0).unsqueeze(0),
        target,
        _input(torch.arange(12), torch.tensor([0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 7, 7])),
    )

    torch.testing.assert_close(result.coordinates[0, 0], coordinates, atol=1e-5, rtol=1e-5)
    assert torch.equal(result.atom_permutation[0, 0], expected_atom)


def test_prediction_matching_is_unlimited_by_default_above_twenty_molecules() -> None:
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coordinates = torch.cat([base + torch.tensor([3.0 * index, 0.2 * index, 0.0]) for index in range(21)])
    expected_atom = torch.arange(84)
    expected_atom[:4] = torch.arange(4, 8)
    expected_atom[4:8] = torch.arange(4)
    predicted = torch.empty_like(coordinates)
    predicted[expected_atom] = coordinates
    target = _target(
        coordinates,
        molecule_id=torch.arange(21).repeat_interleave(4),
        molecule_equivalence_index=torch.zeros(84, dtype=torch.long),
        molecule_atom_index=torch.arange(4).repeat(21),
    )

    result = match_protenix_prediction(
        predicted.unsqueeze(0).unsqueeze(0),
        target,
        _input(torch.arange(84), torch.arange(84)),
    )

    torch.testing.assert_close(result.coordinates[0, 0], coordinates, atol=1e-5, rtol=1e-5)
    assert torch.equal(result.atom_permutation[0, 0], expected_atom)


def test_prediction_matching_rejects_cropped_current_atom_axis() -> None:
    coordinates, molecule_id, molecule_atom_index, equivalence = _homodimer()
    atom_index = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8])
    target = _target(
        coordinates,
        molecule_id=molecule_id,
        molecule_equivalence_index=equivalence,
        molecule_atom_index=molecule_atom_index,
        representative_atom_mask=torch.tensor([True, True, True, True, False] * 2),
    )

    with pytest.raises(ValueError, match="full/no-crop"):
        match_protenix_prediction(
            coordinates.index_select(0, atom_index).unsqueeze(0).unsqueeze(0),
            target,
            _input(atom_index, torch.arange(8)),
        )
