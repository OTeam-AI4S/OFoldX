from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from ofoldx.models import StructurePredictionOutput
from ofoldx.modules.loss import CompositeLoss, LossOutput
from ofoldx.modules.loss.composite import _atom_loss_weight


@pytest.fixture
def model_output():
    r"""
    Return a factory for eval-mode ``StructurePredictionOutput`` with optional confidence logits.
    """

    def build(*, confidence: bool = True) -> StructurePredictionOutput:
        return StructurePredictionOutput(
            pred_coords=torch.zeros(1, 2, 6, 3, requires_grad=True),
            distogram_logits=torch.zeros(1, 3, 3, 7, requires_grad=True),
            plddt_logits=torch.zeros(1, 2, 6, 5, requires_grad=True) if confidence else None,
            pae_logits=torch.zeros(1, 2, 3, 3, 7, requires_grad=True) if confidence else None,
            pde_logits=torch.zeros(1, 2, 3, 3, 8, requires_grad=True) if confidence else None,
            resolved_logits=torch.zeros(1, 2, 6, 2, requires_grad=True) if confidence else None,
        )

    return build


def test_atom_loss_weight() -> None:
    molecule_type = torch.tensor([[0, 1, 2, 3]])
    atom_to_token = torch.tensor([[0, 0, 1, 2, 3]])

    weight = _atom_loss_weight(molecule_type, atom_to_token)

    # 1 + 5*is_dna + 5*is_rna + 10*is_ligand per token (protein/rna/dna/ligand), spread to atoms.
    torch.testing.assert_close(weight, torch.tensor([[1.0, 1.0, 6.0, 6.0, 11.0]]))


@pytest.mark.parametrize(
    ("confidence", "components"),
    [
        (True, {"coordinate", "distogram", "confidence"}),
        (False, {"coordinate", "distogram"}),
    ],
)
def test_forward(dummy_model_input, confidence, components) -> None:
    # Non-degenerate coords keep the coordinate term a real masked MSE (not 0-by-construction): five
    # valid atoms with squared norms 1,4,9,16,25 (atom 5 masked), origin target -> 55 / (5 valid * 3 xyz).
    valid_atoms = torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 3], [4, 0, 0], [0, 5, 0], [9, 9, 9]])
    pred_coords = valid_atoms.view(1, 1, 6, 3).expand(1, 2, 6, 3).clone().requires_grad_(True)
    target_mask = torch.ones(1, 6, dtype=torch.bool)
    target_mask[0, 5] = False
    batch = dataclasses.replace(
        dummy_model_input(num_tokens=3, num_atoms=6),
        target_positions=torch.zeros(1, 6, 3),
        target_mask=target_mask,
    )
    output = StructurePredictionOutput(
        pred_coords=pred_coords,
        distogram_logits=torch.zeros(1, 3, 3, 7, requires_grad=True),
        plddt_logits=torch.zeros(1, 2, 6, 5, requires_grad=True) if confidence else None,
        pae_logits=torch.zeros(1, 2, 3, 3, 7, requires_grad=True) if confidence else None,
        pde_logits=torch.zeros(1, 2, 3, 3, 8, requires_grad=True) if confidence else None,
        resolved_logits=torch.zeros(1, 2, 6, 2, requires_grad=True) if confidence else None,
    )
    loss = CompositeLoss(weights={"distogram": 0.5})

    result = loss(output, batch)

    assert isinstance(result, LossOutput)
    assert set(result.components) == components
    assert result.total.shape == ()

    torch.testing.assert_close(result.components["coordinate"], torch.tensor(55.0 / 15.0))
    # All-zero logits make every masked cross entropy collapse to log(num_bins); distogram has 7 bins.
    torch.testing.assert_close(result.components["distogram"], torch.tensor(math.log(7.0)))
    if confidence:
        # Mean of four uniform-logit cross entropies: pLDDT(5), resolved(2), PAE(7), PDE(8) bins.
        expected_confidence = (math.log(5.0) + math.log(2.0) + math.log(7.0) + math.log(8.0)) / 4.0
        torch.testing.assert_close(result.components["confidence"], torch.tensor(expected_confidence))

    # Built from the independently derived components above (not result.components) to check weighting.
    expected_total = 55.0 / 15.0 + 0.5 * math.log(7.0)
    if confidence:
        expected_total += (math.log(5.0) + math.log(2.0) + math.log(7.0) + math.log(8.0)) / 4.0
    torch.testing.assert_close(result.total, torch.tensor(expected_total))

    result.total.backward()
    assert output.pred_coords.grad is not None


@pytest.mark.parametrize(
    "field",
    ["target_positions", "target_mask", "molecule_type", "frame_atom_index", "frame_mask"],
)
def test_missing_targets(dummy_model_input, model_output, field) -> None:
    batch = dataclasses.replace(dummy_model_input(num_tokens=3, num_atoms=6), **{field: None})

    with pytest.raises(ValueError, match=field):
        CompositeLoss()(model_output(), batch)
