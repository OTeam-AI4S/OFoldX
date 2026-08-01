# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

"""Behavior of ofoldx.models.odesign.objective."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.molecular import AtomSite, ComponentRef
from ofoldx.models.design import PositionSelection
from ofoldx.models.odesign import ODesignProcessor
from ofoldx.models.odesign.model import ODesignOutput
from ofoldx.models.odesign.objective import odesign_loss, weighted_denoising_loss


@pytest.fixture
def task(peptide):
    processor = ODesignProcessor()
    prepared, truth = processor.prepare(
        peptide("ALA", "GLY"),
        target=peptide("ALA", "GLY"),
        design=PositionSelection(components=(ComponentRef("A", 0),)),
        fixed_atoms=(AtomSite("A", 1, "CA"),),
    )
    return prepared.features, truth


def displaced(input, target) -> ODesignOutput:
    r"""One denoising output half an ångström off the truth, with flat pair logits."""
    tokens = input.token_mask.shape[-1]
    return ODesignOutput(
        coordinates=(target.coordinates + 0.5)[:, None].clone().requires_grad_(),
        distogram_logits=torch.zeros(1, tokens, tokens, 4),
        token_bond_logits=torch.zeros(1, tokens, tokens, 2),
        noise_level=torch.ones(1, 1),
        batch_size=[1],
    )


class TestWeightedDenoisingLoss:
    def test_fixed_atoms_unscored(self, task):
        input, target = task
        output = displaced(input, target)
        loss = weighted_denoising_loss(output, input, target)
        (gradient,) = torch.autograd.grad(loss, output.coordinates)
        fixed = input.fixed_coordinate_mask[0]
        assert bool(torch.isfinite(loss))
        assert not bool(gradient[0, 0][fixed].any())
        assert bool(gradient[0, 0][~fixed].any())


class TestOdesignLoss:
    def test_inapplicable_terms_nan(self, task):
        input, target = task
        total, losses = odesign_loss(displaced(input, target), input, target, num_distogram_bins=4)
        assert bool(torch.isfinite(total))
        assert bool(torch.isnan(losses["covalent_bond_distance_loss"]))
        assert bool(torch.isnan(losses["token_bond_loss"]))


@pytest.mark.parametrize("observed", [True, False])
def test_nucleic_linker_truth_scores_each_sample_only_when_available(library, placed, observed):
    import numpy as np

    from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
    from ofoldx.data.structure import CoordinateEvidence, Structure
    from ofoldx.models.odesign import ODesignProcessorConfig

    source_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("N", "N"))}, {"A": "R"}), library
    )
    target_system, _, _ = build_system(
        Recipe({"R": PolymerSequence("polyribonucleotide", ("A", "C"))}, {"A": "R"}), library
    )
    target = placed(target_system)
    atom = int(target_system.atom_index([AtomSite("A", 0, "N9")])[0])
    if not observed:
        arrays = target.build()
        arrays.columns["coordinates"][atom] = 0
        arrays.columns["coordinate_evidence"][atom] = CoordinateEvidence.NONE
        target = Structure.from_arrays(target_system, arrays)
    processor = ODesignProcessor(ODesignProcessorConfig(artifact="nucleic_acid_rigid"))
    prepared, truth = processor.prepare(Structure.unobserved(source_system), target=target)
    slots = prepared.context.slots
    row = int(np.flatnonzero((slots.token == 0) & (slots.atom < 0))[0])
    coordinates = truth.coordinates[:, None].expand(-1, 2, -1, -1).clone()
    coordinates[0, :, row] += torch.tensor([1.0, 2.0, 3.0])
    coordinates.requires_grad_()
    token_count = prepared.features.token_mask.shape[-1]
    output = ODesignOutput(
        coordinates=coordinates,
        distogram_logits=torch.zeros(1, token_count, token_count, 4),
        token_bond_logits=torch.zeros(1, token_count, token_count, 2),
        noise_level=torch.ones(1, 2),
        batch_size=[1],
    )
    loss = weighted_denoising_loss(output, prepared.features, truth)
    (gradient,) = torch.autograd.grad(loss, coordinates)
    assert bool(torch.isfinite(loss))
    assert bool((gradient[0, :, row].abs().sum(dim=-1) > 0).all()) is observed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("case", ["ordinary", "degenerate", "empty"])
def test_weighted_loss_precision_and_zero_weight_semantics(task, dtype, device, case):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    input, truth = task
    input = input.to(device)
    truth = truth.to(device)
    if case == "degenerate":
        truth.coordinates = torch.zeros_like(truth.coordinates)
    if case == "empty":
        truth.coordinate_mask = torch.zeros_like(truth.coordinate_mask)
    output = displaced(input, truth).to(device)
    output.coordinates = output.coordinates.to(dtype).detach().requires_grad_()
    output.noise_level = output.noise_level.to(dtype)
    reference = displaced(input, truth).to(device)
    reference.coordinates = (
        output.coordinates.detach().to(torch.float32 if dtype != torch.float64 else dtype).requires_grad_()
    )
    reference.noise_level = output.noise_level.detach().to(reference.coordinates.dtype)
    with torch.autocast(
        device, dtype=dtype if dtype != torch.float64 else torch.bfloat16, enabled=dtype != torch.float64
    ):
        actual = weighted_denoising_loss(output, input, truth)
    expected = weighted_denoising_loss(reference, input, truth)
    if case == "empty":
        assert torch.isnan(actual) and torch.isnan(expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual)
    (gradient,) = torch.autograd.grad(actual, output.coordinates)
    (expected_gradient,) = torch.autograd.grad(expected, reference.coordinates)
    torch.testing.assert_close(gradient, expected_gradient.to(dtype), rtol=0, atol=0)
    assert bool(torch.isfinite(gradient).all())
