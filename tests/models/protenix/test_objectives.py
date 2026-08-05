from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from ofoldx.data.molecular import ENTITY_DNA, ENTITY_MOLECULE, ENTITY_PROTEIN, ENTITY_RNA
from ofoldx.models.protenix import objectives
from ofoldx.models.protenix.objectives import (
    ProtenixCriterion,
    ProtenixObjectiveConfig,
    criterion_factory,
    protenix_evaluation_objective,
    protenix_training_objective,
)


def _coordinates(batch_size: int) -> Tensor:
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return base.expand(batch_size, -1, -1).clone()


def _case(
    *,
    batch_size: int = 1,
    sample_count: int = 1,
    molecule_type: Tensor | None = None,
    matched_mask: Tensor | None = None,
    resolution: Tensor | None = None,
    resolution_is_known: Tensor | None = None,
    auxiliary_heads: bool = True,
) -> tuple[Any, Any, Any]:
    atom_count = 4
    token_count = 2
    target_coordinates = _coordinates(batch_size)
    if matched_mask is None:
        matched_mask = torch.ones(batch_size, atom_count, dtype=torch.bool)
    if molecule_type is None:
        molecule_type = torch.full((batch_size, atom_count), ENTITY_PROTEIN, dtype=torch.long)
    if resolution is None:
        resolution = torch.full((batch_size,), 2.0)
    if resolution_is_known is None:
        resolution_is_known = torch.ones(batch_size, dtype=torch.bool)

    model_input = SimpleNamespace(
        atom_mask=torch.ones(batch_size, atom_count, dtype=torch.bool),
        atom_index=torch.arange(atom_count).expand(batch_size, -1),
        token_mask=torch.ones(batch_size, token_count, dtype=torch.bool),
        token_to_rep_atom=torch.tensor([0, 3]).expand(batch_size, -1),
    )
    target = SimpleNamespace(
        coordinates=target_coordinates.clone(),
        coordinate_mask=matched_mask.clone(),
        atom_molecule_type=molecule_type,
        plddt_rep_atom_mask=torch.tensor([True, True, True, False]).expand(batch_size, -1),
        ligand_polymer_bond_index=torch.tensor([[[2, 3]]]).expand(batch_size, -1, -1),
        ligand_polymer_bond_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        token_center_atom_index=torch.tensor([0, 3]).expand(batch_size, -1),
        token_frame_atom_index=torch.tensor([[[1, 0, 2], [0, 3, 2]]]).expand(batch_size, -1, -1),
        token_has_frame=torch.ones(batch_size, token_count, dtype=torch.bool),
        resolution=resolution,
        resolution_is_known=resolution_is_known,
    )
    output = SimpleNamespace(
        coordinates=target_coordinates[:, None].expand(-1, sample_count, -1, -1).clone().requires_grad_(),
        mini_rollout_coordinates=target_coordinates[:, None].clone().detach(),
        matched_target_coordinates=target_coordinates,
        matched_target_mask=matched_mask,
        augmented_target_coordinates=target_coordinates[:, None].expand(-1, sample_count, -1, -1).clone(),
        augmented_target_mask=matched_mask[:, None].expand(-1, sample_count, -1),
        noise_level=torch.ones(batch_size, sample_count),
        distogram_logits=(
            torch.zeros(batch_size, token_count, token_count, 64, requires_grad=True) if auxiliary_heads else None
        ),
        plddt_logits=(
            torch.zeros(batch_size, 1, atom_count, 50, requires_grad=True) if auxiliary_heads else None
        ),
        pae_logits=(
            torch.zeros(batch_size, 1, token_count, token_count, 64, requires_grad=True)
            if auxiliary_heads
            else None
        ),
        pde_logits=(
            torch.zeros(batch_size, 1, token_count, token_count, 64, requires_grad=True)
            if auxiliary_heads
            else None
        ),
        resolved_logits=(
            torch.zeros(batch_size, 1, atom_count, 2, requires_grad=True) if auxiliary_heads else None
        ),
    )
    return output, model_input, target


def _only(**weights: Any) -> ProtenixObjectiveConfig:
    values: dict[str, Any] = {
        "mse_weight": 0.0,
        "smooth_lddt_weight": 0.0,
        "bond_weight": 0.0,
        "distogram_weight": 0.0,
        "plddt_weight": 0.0,
        "pde_weight": 0.0,
        "resolved_weight": 0.0,
        "pae_weight": 0.0,
        "compute_zero_weight_terms": False,
    }
    values.update(weights)
    return ProtenixObjectiveConfig(**values)  # type: ignore[arg-type]


def _repeat_confidence_samples(output: Any, sample_count: int) -> None:
    for name in ("plddt_logits", "pde_logits", "resolved_logits", "pae_logits"):
        logits = getattr(output, name)
        repeated = logits.detach().expand(logits.shape[0], sample_count, *logits.shape[2:]).clone()
        setattr(output, name, repeated.requires_grad_())


def test_raw_stage_factors_expand_to_released_final_weights() -> None:
    initial = ProtenixObjectiveConfig()
    assert initial.term_weights == {
        "mse": 4.0,
        "smooth_lddt": 4.0,
        "bond": 0.0,
        "distogram": 0.03,
        "plddt": 1e-4,
        "pde": 1e-4,
        "resolved": 1e-4,
        "pae": 0.0,
    }

    bond_finetuning = ProtenixObjectiveConfig(alpha_bond=1.0, smooth_lddt=0.0)
    assert bond_finetuning.term_weights["bond"] == 4.0
    assert bond_finetuning.term_weights["smooth_lddt"] == 0.0

    confidence_finetuning = ProtenixObjectiveConfig(
        alpha_pae=1.0,
        alpha_diffusion=0.0,
        alpha_distogram=0.0,
        smooth_lddt=0.0,
        train_confidence_only=True,
    )
    assert confidence_finetuning.term_weights == {
        "mse": 0.0,
        "smooth_lddt": 0.0,
        "bond": 0.0,
        "distogram": 0.0,
        "plddt": 1e-4,
        "pde": 1e-4,
        "resolved": 1e-4,
        "pae": 1e-4,
    }
    assert ProtenixObjectiveConfig(alpha_pae=1.0).term_weights["pae"] == 1e-4
    assert ProtenixObjectiveConfig(mse_weight=2.0).term_weights["mse"] == 2.0
    assert ProtenixObjectiveConfig(train_confidence_only=True).term_weights == initial.term_weights


def test_default_component_formulas_aggregate_and_gradients() -> None:
    output, model_input, target = _case()

    result = protenix_training_objective(output, model_input, target, sigma_data=1.0)

    zero_error_smooth_lddt = 1.0 - 0.25 * sum(torch.sigmoid(torch.tensor(value)) for value in (0.5, 1, 2, 4))
    torch.testing.assert_close(result.raw.mse, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.smooth_lddt, zero_error_smooth_lddt)
    torch.testing.assert_close(result.raw.bond, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.distogram, torch.tensor(math.log(64.0)))
    torch.testing.assert_close(result.raw.plddt, torch.tensor(math.log(50.0)))
    torch.testing.assert_close(result.raw.pde, torch.tensor(math.log(64.0)))
    torch.testing.assert_close(result.raw.resolved, torch.tensor(math.log(2.0)))
    torch.testing.assert_close(result.raw.pae, torch.tensor(math.log(64.0)))
    expected = (
        4.0 * zero_error_smooth_lddt
        + 0.03 * math.log(64.0)
        + 1e-4 * (math.log(50.0) + math.log(64.0) + math.log(2.0))
    )
    torch.testing.assert_close(result.loss, torch.as_tensor(expected))
    torch.testing.assert_close(result.weighted.pae, torch.tensor(0.0))
    assert result.loss.dtype == torch.float32
    assert result.confidence_resolution_valid.item() == 1.0
    assert all(not metric.requires_grad for metric in result.metrics.values())

    result.loss.backward()
    assert output.coordinates.grad is not None
    assert torch.isfinite(output.coordinates.grad).all()
    for name in ("distogram_logits", "plddt_logits", "pde_logits", "resolved_logits", "pae_logits"):
        gradient = getattr(output, name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_mse_uses_mixed_chemistry_weights_resolved_denominator_and_sample_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule_type = torch.tensor([[ENTITY_PROTEIN, ENTITY_DNA, ENTITY_RNA, ENTITY_MOLECULE]])
    output, model_input, target = _case(sample_count=2, molecule_type=molecule_type, auxiliary_heads=False)
    output.augmented_target_coordinates.zero_()
    output.matched_target_coordinates.zero_()
    first = torch.tensor([[1.0, 0.0, 0.0]]).expand(4, -1)
    second = torch.tensor([[2.0, 0.0, 0.0]]).expand(4, -1)
    output.coordinates = torch.stack((first, second), dim=0)[None].clone().requires_grad_()
    captured_weight: list[Tensor] = []

    def no_alignment(*, source: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        del target
        captured_weight.append(weight.detach().clone())
        return source.detach()

    monkeypatch.setattr(objectives, "weighted_rigid_align", no_alignment)
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(mse_weight=1.0),
    )

    chemistry_sum = 1.0 + 6.0 + 6.0 + 11.0
    expected = (1.0 / 3.0) * 2.0 * chemistry_sum * (1.0 + 4.0) / (2.0 * (4.0 + 1e-6))
    torch.testing.assert_close(result.raw.mse, torch.tensor(expected))
    torch.testing.assert_close(captured_weight[0], torch.tensor([[[1.0, 6.0, 6.0, 11.0], [1.0, 6.0, 6.0, 11.0]]]))
    result.loss.backward()
    gradient = output.coordinates.grad[..., 0]
    torch.testing.assert_close(gradient[..., 1] / gradient[..., 0], torch.full((1, 2), 6.0))
    torch.testing.assert_close(gradient[..., 2] / gradient[..., 0], torch.full((1, 2), 6.0))
    torch.testing.assert_close(gradient[..., 3] / gradient[..., 0], torch.full((1, 2), 11.0))


def test_matched_mask_and_batch_reduction_are_per_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    matched_mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    output, model_input, target = _case(batch_size=2, matched_mask=matched_mask, auxiliary_heads=False)
    output.augmented_target_coordinates.zero_()
    output.matched_target_coordinates.zero_()
    prediction = torch.tensor(
        [
            [[1.0, 0.0, 0.0]] * 4,
            [[2.0, 0.0, 0.0]] * 4,
        ]
    )
    output.coordinates = prediction[:, None].clone().requires_grad_()
    monkeypatch.setattr(
        objectives,
        "weighted_rigid_align",
        lambda *, source, target, weight: source.detach(),
    )

    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(mse_weight=1.0),
    )

    first_structure = (1.0 / 3.0) * 2.0 / (1.0 + 1e-6)
    second_structure = (1.0 / 3.0) * 2.0 * 16.0 / (4.0 + 1e-6)
    torch.testing.assert_close(result.raw.mse, torch.tensor((first_structure + second_structure) / 2.0))
    result.loss.backward()
    torch.testing.assert_close(output.coordinates.grad[0, 0, 1:], torch.zeros(3, 3))


def test_mse_aligns_augmented_target_onto_prediction_for_a_rigid_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, model_input, target = _case(auxiliary_heads=False)
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([4.0, -3.0, 2.0])
    transformed_target = output.matched_target_coordinates @ rotation.T + translation
    output.augmented_target_coordinates = transformed_target[:, None]
    real_align = objectives.weighted_rigid_align
    alignment_called = False

    def capture_alignment(*, source: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        nonlocal alignment_called
        alignment_called = True
        torch.testing.assert_close(source, transformed_target[:, None])
        torch.testing.assert_close(target, output.coordinates)
        return real_align(source=source, target=target, weight=weight)

    monkeypatch.setattr(objectives, "weighted_rigid_align", capture_alignment)
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(mse_weight=1.0),
    )

    assert alignment_called
    torch.testing.assert_close(result.raw.mse, torch.tensor(0.0), atol=1e-10, rtol=0.0)


def test_smooth_lddt_chunking_matches_unchunked_values_and_gradients() -> None:
    assert ProtenixObjectiveConfig().smooth_lddt_chunk_size == 1

    def evaluate(chunk_size: int | None) -> tuple[Tensor, Tensor]:
        output, model_input, target = _case(sample_count=3, auxiliary_heads=False)
        coordinates = output.coordinates.detach().clone()
        coordinates[0, 0, 3] += torch.tensor([0.4, -0.2, 0.1])
        coordinates[0, 1, 2] += torch.tensor([-0.3, 0.5, 0.2])
        coordinates[0, 2, 1] += torch.tensor([0.1, 0.2, -0.6])
        output.coordinates = coordinates.requires_grad_()
        result = protenix_training_objective(
            output,
            model_input,
            target,
            sigma_data=1.0,
            config=_only(smooth_lddt_weight=1.0, smooth_lddt_chunk_size=chunk_size),
        )
        result.loss.backward()
        assert output.coordinates.grad is not None
        return result.raw.smooth_lddt, output.coordinates.grad.detach().clone()

    chunked_value, chunked_gradient = evaluate(1)
    unchunked_value, unchunked_gradient = evaluate(None)
    torch.testing.assert_close(chunked_value, unchunked_value)
    torch.testing.assert_close(chunked_gradient, unchunked_gradient)


def test_bond_and_distogram_use_released_formulas() -> None:
    output, model_input, target = _case()
    output.coordinates = output.coordinates.detach().clone()
    output.coordinates[0, 0, 3, 0] += 1.0
    output.coordinates.requires_grad_()
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(bond_weight=1.0),
    )
    expected_bond = 2.0 * (math.sqrt(3.0) - math.sqrt(2.0)) ** 2
    torch.testing.assert_close(result.raw.bond, torch.tensor(expected_bond))

    output, model_input, target = _case(auxiliary_heads=False)
    logits = torch.tensor([2.0, 0.0, -1.0]).expand(1, 2, 2, -1).clone().requires_grad_()
    output.distogram_logits = logits
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(distogram_weight=1.0),
    )
    expected_distogram = -torch.log_softmax(torch.tensor([2.0, 0.0, -1.0]), dim=-1)[0]
    torch.testing.assert_close(result.raw.distogram, expected_distogram)


def test_nonuniform_confidence_logits_use_baseline_bin_labels() -> None:
    output, model_input, target = _case()
    plddt_class_logits = torch.linspace(-1.0, 2.0, 50)
    pair_class_logits = torch.linspace(1.5, -2.0, 64)
    resolved_class_logits = torch.tensor([-0.75, 1.25])
    output.plddt_logits = plddt_class_logits.view(1, 1, 1, -1).expand(1, 1, 4, -1).clone().requires_grad_()
    output.pde_logits = pair_class_logits.view(1, 1, 1, 1, -1).expand(1, 1, 2, 2, -1).clone().requires_grad_()
    output.pae_logits = pair_class_logits.view(1, 1, 1, 1, -1).expand(1, 1, 2, 2, -1).clone().requires_grad_()
    output.resolved_logits = resolved_class_logits.view(1, 1, 1, -1).expand(1, 1, 4, -1).clone().requires_grad_()
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(plddt_weight=1.0, pde_weight=1.0, resolved_weight=1.0, pae_weight=1.0),
    )

    pair_reduction = 4.0 / (4.0 + 1e-6)
    expected_plddt = -torch.log_softmax(plddt_class_logits, dim=-1)[49]
    expected_pde = -torch.log_softmax(pair_class_logits, dim=-1)[0] * pair_reduction
    expected_resolved = -torch.log_softmax(resolved_class_logits, dim=-1)[1]
    expected_pae = -torch.log_softmax(pair_class_logits, dim=-1)[0] * pair_reduction
    torch.testing.assert_close(result.raw.plddt, expected_plddt)
    torch.testing.assert_close(result.raw.pde, expected_pde)
    torch.testing.assert_close(result.raw.resolved, expected_resolved)
    torch.testing.assert_close(result.raw.pae, expected_pae)


def test_atom_confidence_exact_means_handle_batchwise_nonuniform_valid_counts() -> None:
    matched_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    output, model_input, target = _case(batch_size=2, matched_mask=matched_mask)
    model_input.atom_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    plddt_class_logits = torch.linspace(-1.0, 2.0, 50)
    resolved_class_logits = torch.tensor([0.8, -0.4])
    output.plddt_logits = plddt_class_logits.view(1, 1, 1, -1).expand(2, 1, 4, -1).clone().requires_grad_()
    output.resolved_logits = (
        resolved_class_logits.view(1, 1, 1, -1).expand(2, 1, 4, -1).clone().requires_grad_()
    )
    output.pde_logits = None
    output.pae_logits = None
    output.distogram_logits = None
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(plddt_weight=1.0, resolved_weight=1.0),
    )

    plddt_error = -torch.log_softmax(plddt_class_logits, dim=-1)
    resolved_error = -torch.log_softmax(resolved_class_logits, dim=-1)
    expected_plddt = 0.5 * (plddt_error[0] + plddt_error[49])
    first_resolved = resolved_error[1]
    second_resolved = 0.5 * (resolved_error[1] + resolved_error[0])
    expected_resolved = 0.5 * (first_resolved + second_resolved)
    torch.testing.assert_close(result.raw.plddt, expected_plddt)
    torch.testing.assert_close(result.raw.resolved, expected_resolved)
    torch.testing.assert_close(result.loss, expected_plddt + expected_resolved)


def test_atom_confidence_empty_masks_return_graph_safe_exact_zero() -> None:
    matched_mask = torch.zeros(1, 4, dtype=torch.bool)
    output, model_input, target = _case(matched_mask=matched_mask)
    model_input.atom_mask.zero_()
    output.pde_logits = None
    output.pae_logits = None
    output.distogram_logits = None
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(plddt_weight=1.0, resolved_weight=1.0),
    )

    torch.testing.assert_close(result.raw.plddt, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.resolved, torch.tensor(0.0))
    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    assert result.loss.requires_grad
    result.loss.backward()
    assert output.plddt_logits.grad is not None
    assert output.resolved_logits.grad is not None
    torch.testing.assert_close(output.plddt_logits.grad, torch.zeros_like(output.plddt_logits.grad))
    torch.testing.assert_close(output.resolved_logits.grad, torch.zeros_like(output.resolved_logits.grad))


def test_zero_pair_frame_and_bond_sets_match_baseline_empty_reductions() -> None:
    matched_mask = torch.tensor([[True, False, False, False]])
    output, model_input, target = _case(matched_mask=matched_mask)
    model_input.token_mask.zero_()
    target.token_has_frame.zero_()
    target.ligand_polymer_bond_index = torch.empty(1, 0, 2, dtype=torch.long)
    target.ligand_polymer_bond_mask = torch.empty(1, 0, dtype=torch.bool)
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(
            smooth_lddt_weight=1.0,
            bond_weight=1.0,
            distogram_weight=1.0,
            pde_weight=1.0,
            pae_weight=1.0,
        ),
    )

    torch.testing.assert_close(result.raw.smooth_lddt, torch.tensor(1.0))
    torch.testing.assert_close(result.raw.bond, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.distogram, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.pde, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.pae, torch.tensor(0.0))
    assert result.loss.requires_grad
    result.loss.backward()
    assert output.coordinates.grad is not None
    torch.testing.assert_close(output.coordinates.grad, torch.zeros_like(output.coordinates.grad))


@pytest.mark.parametrize(
    ("resolution", "known", "valid"),
    [
        (0.1, True, 1.0),
        (4.0, True, 1.0),
        (0.09, True, 0.0),
        (4.01, True, 0.0),
        (-1.0, False, 0.0),
        (float("nan"), True, 0.0),
    ],
)
def test_resolution_gate_never_treats_unknown_or_invalid_values_as_known(
    resolution: float,
    known: bool,
    valid: float,
) -> None:
    output, model_input, target = _case(
        resolution=torch.tensor([resolution]),
        resolution_is_known=torch.tensor([known]),
    )
    result = protenix_training_objective(output, model_input, target, sigma_data=1.0)
    assert result.confidence_resolution_valid.item() == valid
    expected_plddt = valid * math.log(50.0)
    torch.testing.assert_close(result.raw.plddt, torch.tensor(expected_plddt))
    torch.testing.assert_close(result.raw.pde, torch.tensor(valid * math.log(64.0)))
    torch.testing.assert_close(result.raw.resolved, torch.tensor(valid * math.log(2.0)))
    torch.testing.assert_close(result.raw.pae, torch.tensor(valid * math.log(64.0)))
    confidence_metric_names = ("plddt", "pde", "resolved", "pae")
    for name in confidence_metric_names:
        assert (f"objective/{name}" in result.metrics) is bool(valid)
        assert (f"objective/weighted_{name}" in result.metrics) is bool(valid)


def test_mixed_resolution_batch_gates_total_and_reports_only_valid_confidence_metrics() -> None:
    output, model_input, target = _case(
        batch_size=2,
        resolution=torch.tensor([2.0, 5.0]),
        resolution_is_known=torch.tensor([True, True]),
    )
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(plddt_weight=1.0),
    )

    expected = 0.5 * math.log(50.0)
    torch.testing.assert_close(result.raw.plddt, torch.tensor(expected))
    torch.testing.assert_close(result.loss, torch.tensor(expected))
    torch.testing.assert_close(result.confidence_resolution_valid, torch.tensor(0.5))
    torch.testing.assert_close(result.metrics["objective/plddt"], torch.tensor(math.log(50.0)))
    torch.testing.assert_close(result.metrics["objective/weighted_plddt"], torch.tensor(math.log(50.0)))
    assert "objective/confidence_resolution_valid" in result.metrics


def test_invalid_resolution_confidence_total_retains_a_graph_safe_zero() -> None:
    output, model_input, target = _case(
        resolution=torch.tensor([5.0]),
        resolution_is_known=torch.tensor([True]),
    )
    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=_only(plddt_weight=1.0),
    )

    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    assert result.loss.requires_grad
    assert "objective/plddt" not in result.metrics
    assert "objective/weighted_plddt" not in result.metrics
    result.loss.backward()
    assert output.plddt_logits.grad is not None
    torch.testing.assert_close(output.plddt_logits.grad, torch.zeros_like(output.plddt_logits.grad))


def test_missing_zero_weight_bond_and_pae_are_optional_but_nonzero_terms_fail() -> None:
    output, model_input, target = _case()
    target.ligand_polymer_bond_index = None
    target.ligand_polymer_bond_mask = None
    target.token_frame_atom_index = None
    result = protenix_training_objective(output, model_input, target, sigma_data=1.0)
    torch.testing.assert_close(result.raw.bond, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.pae, torch.tensor(0.0))

    with pytest.raises(ValueError, match="nonzero bond_weight"):
        protenix_training_objective(
            output,
            model_input,
            target,
            sigma_data=1.0,
            config=_only(bond_weight=1.0),
        )
    with pytest.raises(ValueError, match="nonzero pae_weight"):
        protenix_training_objective(
            output,
            model_input,
            target,
            sigma_data=1.0,
            config=_only(pae_weight=1.0),
        )


def test_confidence_only_output_requires_zero_coordinate_weights_without_stage_flag() -> None:
    output, model_input, target = _case()
    output.coordinates = None
    output.augmented_target_coordinates = None
    output.augmented_target_mask = None
    output.noise_level = None
    output.distogram_logits = None
    confidence_only = _only(plddt_weight=1e-4, pde_weight=1e-4, resolved_weight=1e-4, pae_weight=1.0)

    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=confidence_only,
    )

    torch.testing.assert_close(result.raw.mse, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.smooth_lddt, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.bond, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.distogram, torch.tensor(0.0))
    assert result.loss.requires_grad
    with pytest.raises(ValueError, match="nonzero coordinate-loss weights"):
        protenix_training_objective(output, model_input, target, sigma_data=1.0)


def test_train_confidence_only_flag_skips_default_coordinate_and_distogram_factors() -> None:
    output, model_input, target = _case()
    output.coordinates = None
    output.augmented_target_coordinates = None
    output.augmented_target_mask = None
    output.noise_level = None
    output.distogram_logits = None

    result = protenix_training_objective(
        output,
        model_input,
        target,
        sigma_data=1.0,
        config=ProtenixObjectiveConfig(train_confidence_only=True),
    )

    assert result.available_terms == ("plddt", "pde", "resolved", "pae")
    torch.testing.assert_close(result.raw.mse, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.smooth_lddt, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.bond, torch.tensor(0.0))
    torch.testing.assert_close(result.raw.distogram, torch.tensor(0.0))
    expected = 1e-4 * (math.log(50.0) + math.log(64.0) + math.log(2.0))
    torch.testing.assert_close(result.loss, torch.tensor(expected))
    assert "objective/mse" not in result.metrics
    assert "objective/distogram" not in result.metrics


def test_unsupported_resolved_molecule_type_is_not_silently_weighted() -> None:
    output, model_input, target = _case(molecule_type=torch.tensor([[5, 1, 1, 1]]))
    with pytest.raises(ValueError, match="no baseline chemistry weight"):
        protenix_training_objective(output, model_input, target, sigma_data=1.0)


def test_evaluation_mse_gathers_nonidentity_truth_mask_and_chemistry_without_edm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule_type = torch.tensor([[ENTITY_PROTEIN, ENTITY_DNA, ENTITY_RNA, ENTITY_MOLECULE]])
    output, model_input, target = _case(sample_count=2, molecule_type=molecule_type, auxiliary_heads=False)
    model_input.atom_index = torch.tensor([[3, 0, 2, 1]])
    target.coordinate_mask = torch.tensor([[True, False, True, True]])
    current_target = target.coordinates[:, model_input.atom_index[0]]
    current_mask = target.coordinate_mask[:, model_input.atom_index[0]]
    masked_current_target = current_target.masked_fill(~current_mask[..., None], 0)
    coordinates = current_target[:, None].expand(-1, 2, -1, -1).clone()
    coordinates[:, 0, :, 0] += 1.0
    coordinates[:, 1, :, 0] += 2.0
    output.coordinates = coordinates.requires_grad_()
    del output.noise_level
    captured_weight: list[Tensor] = []

    def no_alignment(*, source: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        del target
        torch.testing.assert_close(source, masked_current_target[:, None].expand_as(coordinates))
        captured_weight.append(weight.detach().clone())
        return source.detach()

    monkeypatch.setattr(objectives, "weighted_rigid_align", no_alignment)
    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=_only(mse_weight=1.0),
    )

    chemistry_sum = 11.0 + 1.0 + 6.0
    expected = (1.0 / 3.0) * chemistry_sum * (1.0 + 4.0) / (2.0 * (3.0 + 1e-6))
    torch.testing.assert_close(result.raw.mse, torch.tensor(expected))
    torch.testing.assert_close(
        captured_weight[0],
        torch.tensor([[[11.0, 1.0, 6.0, 0.0], [11.0, 1.0, 6.0, 0.0]]]),
    )
    assert result.mode == "eval"
    assert result.available_terms == ("mse", "smooth_lddt", "bond", "plddt", "pde", "resolved", "pae")


def test_evaluation_rejects_a_nonbijective_atom_axis() -> None:
    output, model_input, target = _case(auxiliary_heads=False)
    model_input.atom_index = torch.tensor([[0, 1, 1, 3]])

    with pytest.raises(ValueError, match="canonical bijection"):
        protenix_evaluation_objective(output, model_input, target, config=_only(mse_weight=1.0))


def test_evaluation_bond_has_no_edm_scale_or_noise_requirement() -> None:
    output, model_input, target = _case(auxiliary_heads=False)
    output.coordinates = output.coordinates.detach().clone()
    output.coordinates[0, 0, 3, 0] += 1.0
    output.coordinates.requires_grad_()
    del output.noise_level
    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=_only(bond_weight=1.0),
    )

    expected_bond = (math.sqrt(3.0) - math.sqrt(2.0)) ** 2
    torch.testing.assert_close(result.raw.bond, torch.tensor(expected_bond))
    torch.testing.assert_close(result.loss, torch.tensor(expected_bond))


def test_evaluation_mse_kabsch_removes_rigid_reverse_diffusion_transform() -> None:
    output, model_input, target = _case(auxiliary_heads=False)
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([7.0, -2.0, 4.0])
    transformed = target.coordinates @ rotation.T + translation
    output.coordinates = transformed[:, None].clone().requires_grad_()
    del output.noise_level

    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=_only(mse_weight=1.0),
    )

    torch.testing.assert_close(result.raw.mse, torch.tensor(0.0), atol=1e-10, rtol=0.0)


def test_evaluation_excludes_distogram_even_with_default_training_weight() -> None:
    output, model_input, target = _case()
    output.distogram_logits = None

    result = protenix_evaluation_objective(output, model_input, target)

    assert "distogram" not in result.available_terms
    torch.testing.assert_close(result.raw.distogram, torch.tensor(0.0))
    torch.testing.assert_close(result.weighted.distogram, torch.tensor(0.0))
    assert "objective/distogram" not in result.metrics
    assert "objective/weighted_distogram" not in result.metrics
    zero_error_smooth_lddt = 1.0 - 0.25 * sum(torch.sigmoid(torch.tensor(value)) for value in (0.5, 1, 2, 4))
    torch.testing.assert_close(result.raw.smooth_lddt, zero_error_smooth_lddt)
    expected = sum(
        (getattr(result.weighted, name) for name in result.available_terms),
        torch.tensor(0.0),
    )
    torch.testing.assert_close(result.loss, expected)


def test_evaluation_ignores_training_confidence_only_execution_flag() -> None:
    output, model_input, target = _case()

    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=ProtenixObjectiveConfig(train_confidence_only=True),
    )

    assert "mse" in result.available_terms
    assert "smooth_lddt" in result.available_terms
    assert "bond" in result.available_terms
    assert "distogram" not in result.available_terms
    assert result.weighted.smooth_lddt.item() > 0
    assert "objective/smooth_lddt" in result.metrics


def test_evaluation_confidence_uses_all_reverse_diffusion_samples() -> None:
    output, model_input, target = _case(sample_count=2)
    del output.mini_rollout_coordinates
    plddt_logits = torch.stack((torch.linspace(-1.0, 2.0, 50), torch.linspace(2.0, -1.0, 50)))
    pair_logits = torch.stack((torch.linspace(1.5, -2.0, 64), torch.linspace(-2.0, 1.5, 64)))
    resolved_logits = torch.tensor([[-0.75, 1.25], [1.0, -0.5]])
    output.plddt_logits = plddt_logits.view(1, 2, 1, -1).expand(1, 2, 4, -1).clone().requires_grad_()
    output.pde_logits = pair_logits.view(1, 2, 1, 1, -1).expand(1, 2, 2, 2, -1).clone().requires_grad_()
    output.pae_logits = pair_logits.view(1, 2, 1, 1, -1).expand(1, 2, 2, 2, -1).clone().requires_grad_()
    output.resolved_logits = resolved_logits.view(1, 2, 1, -1).expand(1, 2, 4, -1).clone().requires_grad_()
    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=_only(plddt_weight=1.0, pde_weight=1.0, resolved_weight=1.0, pae_weight=1.0),
    )

    pair_reduction = 4.0 / (4.0 + 1e-6)
    expected_plddt = -torch.log_softmax(plddt_logits, dim=-1)[:, 49].mean()
    expected_pair = -torch.log_softmax(pair_logits, dim=-1)[:, 0].mean() * pair_reduction
    expected_resolved = -torch.log_softmax(resolved_logits, dim=-1)[:, 1].mean()
    torch.testing.assert_close(result.raw.plddt, expected_plddt)
    torch.testing.assert_close(result.raw.pde, expected_pair)
    torch.testing.assert_close(result.raw.resolved, expected_resolved)
    torch.testing.assert_close(result.raw.pae, expected_pair)
    torch.testing.assert_close(
        result.loss,
        expected_plddt + expected_pair + expected_resolved + expected_pair,
    )


def test_evaluation_invalid_resolution_keeps_graph_zero_and_omits_confidence_metrics() -> None:
    output, model_input, target = _case(sample_count=2, resolution=torch.tensor([5.0]))
    _repeat_confidence_samples(output, 2)
    result = protenix_evaluation_objective(
        output,
        model_input,
        target,
        config=_only(plddt_weight=1.0),
    )

    torch.testing.assert_close(result.loss, torch.tensor(0.0))
    assert result.loss.requires_grad
    assert "objective/plddt" not in result.metrics
    result.loss.backward()
    assert output.plddt_logits.grad is not None
    torch.testing.assert_close(output.plddt_logits.grad, torch.zeros_like(output.plddt_logits.grad))


def test_criterion_factory_returns_runner_scalar_and_detached_latest_metrics() -> None:
    model = SimpleNamespace(config=SimpleNamespace(denoiser=SimpleNamespace(sigma_data=1.0)))
    criterion = criterion_factory(model, object())
    output, model_input, target = _case()

    loss = criterion(output, {"input": model_input, "target": target})

    assert isinstance(criterion, ProtenixCriterion)
    assert loss.ndim == 0
    assert criterion.latest_result is not None
    assert criterion.latest_result.loss is loss
    assert criterion.latest_metrics
    assert all(not metric.requires_grad for metric in criterion.latest_metrics.values())

    evaluation_loss = criterion.evaluate(output, {"input": model_input, "target": target})
    assert criterion.latest_result is not None
    assert criterion.latest_result.mode == "eval"
    assert criterion.latest_result.loss is evaluation_loss
    assert "objective/distogram" not in criterion.latest_metrics


def test_config_rejects_invalid_ranges_and_scales() -> None:
    with pytest.raises(ValueError, match="mse_weight"):
        ProtenixObjectiveConfig(mse_weight=-1.0)
    with pytest.raises(ValueError, match="smooth_lddt_chunk_size"):
        ProtenixObjectiveConfig(smooth_lddt_chunk_size=0)
    with pytest.raises(ValueError, match="resolution_min"):
        ProtenixObjectiveConfig(resolution_min=5.0, resolution_max=4.0)
    with pytest.raises(ValueError, match="sigma_data"):
        ProtenixCriterion(sigma_data=0.0)
