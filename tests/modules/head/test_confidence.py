from __future__ import annotations

import math

import pytest
import torch

from ofoldx.modules.head import ConfidenceLoss


@pytest.fixture
def loss_inputs() -> dict:
    # Atom 4 is padding (masked everywhere) so the masked means are not true-by-construction.
    torch.manual_seed(0)
    return {
        "pred_coords": torch.randn(1, 2, 5, 3),
        "target_positions": torch.randn(1, 5, 3),
        "target_mask": torch.tensor([[True, True, True, True, False]]),
        "representative_atom_mask": torch.tensor([True, False, True, True, False]),
        "token_to_rep_atom": torch.tensor([[0, 2, 3]]),
        "frame_atom_index": torch.tensor([[[0, 2, 3], [1, 2, 4], [0, 1, 2]]]),
        "frame_mask": torch.ones(1, 3, dtype=torch.bool),
        "molecule_type": torch.zeros(1, 3, dtype=torch.long),
        "atom_to_token": torch.tensor([[0, 0, 1, 2, 2]]),
        "atom_mask": torch.tensor([[True, True, True, True, False]]),
        "plddt_logits": torch.randn(1, 2, 5, 5, requires_grad=True),
        "resolved_logits": torch.randn(1, 2, 5, 2, requires_grad=True),
        "pae_logits": torch.randn(1, 2, 3, 3, 7, requires_grad=True),
        "pde_logits": torch.randn(1, 2, 3, 3, 8, requires_grad=True),
    }


# Head emits either atom-level (N_atom=5) or token-level (N_token=3) logits.
@pytest.mark.parametrize(("plddt_axis", "resolved_axis"), [(5, 5), (3, 3)])
def test_forward(loss_inputs: dict, plddt_axis: int, resolved_axis: int) -> None:
    loss_inputs["plddt_logits"] = torch.randn(1, 2, plddt_axis, 5, requires_grad=True)
    loss_inputs["resolved_logits"] = torch.randn(1, 2, resolved_axis, 2, requires_grad=True)

    value = ConfidenceLoss()(**loss_inputs)

    assert value.shape == ()
    # No value is hand-derivable here, so assert invariants: a positive, finite CE average.
    assert torch.isfinite(value) and value > 0.0
    # Head is dropout-free, so identical inputs must give the identical scalar.
    rerun = ConfidenceLoss()(**{k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in loss_inputs.items()})
    torch.testing.assert_close(value.detach(), rerun)

    value.backward()
    for name in ("plddt_logits", "resolved_logits", "pae_logits", "pde_logits"):
        assert loss_inputs[name].grad is not None


def test_resolved_loss_value() -> None:
    # Distinct per-atom class-1 logits (class-0 fixed at 0) so the label gather is exercised.
    vals = torch.tensor([2.0, -1.0, 0.5, 9.0, 9.0])
    logits = torch.zeros(1, 2, 5, 2)
    logits[0, :, :, 1] = vals
    target_mask = torch.tensor([[True, True, False, False, False]])  # resolved flags -> labels 1,1,0,0,0
    atom_mask = torch.tensor([[True, True, True, False, False]])  # atoms 3,4 are padding (masked out)
    token_to_rep_atom = torch.tensor([[0, 1, 2]])

    base = ConfidenceLoss._resolved_loss(
        logits, target_mask=target_mask, atom_mask=atom_mask, token_to_rep_atom=token_to_rep_atom
    )

    # Recompute the same masked CE a different way (F.cross_entropy) to derive the expected value independently.
    labels = target_mask.long().unsqueeze(1).expand(1, 2, 5)
    mask = atom_mask.float().unsqueeze(1).expand(1, 2, 5)
    per_element = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction="none")
    expected = (per_element * mask.reshape(-1)).sum() / mask.sum()
    torch.testing.assert_close(base, expected)


def test_resolved_loss_mask() -> None:
    # Zero logits over 2 classes give a uniform-prediction cross entropy of exactly log(2) per element.
    logits = torch.zeros(1, 2, 5, 2)
    target_mask = torch.tensor([[True, True, True, False, False]])
    atom_mask = torch.tensor([[True, True, True, False, False]])  # last two atoms are padding
    token_to_rep_atom = torch.tensor([[0, 1, 2]])

    base = ConfidenceLoss._resolved_loss(
        logits, target_mask=target_mask, atom_mask=atom_mask, token_to_rep_atom=token_to_rep_atom
    )
    torch.testing.assert_close(base, torch.tensor(math.log(2.0)))

    # Perturbing only padded atoms' class-1 logit is a real CE change but must be masked out -> loss unchanged.
    perturbed = logits.clone()
    perturbed[..., 3:, 1] += 100.0
    torch.testing.assert_close(
        base,
        ConfidenceLoss._resolved_loss(
            perturbed, target_mask=target_mask, atom_mask=atom_mask, token_to_rep_atom=token_to_rep_atom
        ),
    )


def test_forward_padding_mask_invariance(loss_inputs: dict) -> None:
    # End-to-end guard on the resolved mask: perturbing padded atom 4's logits must not move the total loss.
    base = ConfidenceLoss()(**loss_inputs)

    perturbed = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in loss_inputs.items()}
    perturbed["resolved_logits"][:, :, 4, 1] += 100.0  # one class only -> a real CE change, were atom 4 unmasked
    other = ConfidenceLoss()(**perturbed)
    torch.testing.assert_close(base.detach(), other)


# A logit axis must match the atom count (5) or token count (3); anything else raises.
@pytest.mark.parametrize("key", ["plddt_logits", "resolved_logits"])
def test_invalid_logit_axis(loss_inputs: dict, key: str) -> None:
    bins = loss_inputs[key].shape[-1]
    loss_inputs[key] = torch.zeros(1, 2, 4, bins, requires_grad=True)  # axis 4 is neither atoms (5) nor tokens (3)

    with pytest.raises(ValueError, match="must target either atoms or tokens"):
        ConfidenceLoss()(**loss_inputs)
