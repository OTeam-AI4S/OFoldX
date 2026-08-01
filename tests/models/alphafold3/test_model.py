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

"""Behavior of ofoldx.models.alphafold3.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.errors import InvalidRequestError
from ofoldx.models.alphafold3.model import (
    AlphaFold3InputEmbeddings,
    AlphaFold3TemplateEmbedder,
    AlphaFold3TemplateFeatures,
    normalize_alphafold3_input,
)
from ofoldx.models.alphafold3.processor import AlphaFold3Processor


def test_unavailable_embedding_gradients():
    model = AlphaFold3InputEmbeddings(single_size=8, atom_feature_size=5)
    positions = torch.tensor([[[1.0, 2.0, 3.0], [float("nan"), float("inf"), 0.0]]], requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    observed = torch.tensor([[True, False]])
    output = model(
        torch.ones(1, 2, 5),
        reference_positions=positions,
        reference_charges=torch.zeros(1, 2, 1),
        reference_mask=observed,
        token_features=torch.ones(1, 1, 8),
        atom_to_token=torch.zeros(1, 2, dtype=torch.long),
        atom_mask=valid,
        token_mask=torch.ones(1, 1, dtype=torch.bool),
        local_geometry=model.windowing(positions, torch.zeros(1, 2, dtype=torch.long), valid, observed),
    )
    assert torch.isfinite(output).all()
    gradient = torch.autograd.grad(output.square().sum(), positions)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[:, 1], torch.zeros(1, 3))


def test_msa_input_availability(peptide):
    value = AlphaFold3Processor().prepare_input(peptide("ALA", "GLY")).features
    value.msa_tokens = torch.tensor([[[1, -10000]]])
    value.msa_mask = torch.tensor([[[True, False]]])
    value.token_mask = torch.tensor([[True, False]])
    value.msa_deletion_counts = torch.tensor([[[100000.0, torch.nan]]], dtype=torch.float64)
    normalized = normalize_alphafold3_input(value, dtype=torch.float16)
    assert normalized.msa_tokens.tolist() == [[[1, 0]]]
    assert value.msa_tokens.tolist() == [[[1, -10000]]]
    assert normalized.msa_deletion_counts.tolist() == [[[100000.0, 0.0]]]
    value.msa_tokens[0, 0, 0] = -1
    with pytest.raises(InvalidRequestError):
        normalize_alphafold3_input(value)


def test_template_slot_categories():
    templates = AlphaFold3TemplateFeatures(
        distogram=torch.full((1, 1, 2, 2, 39), torch.nan),
        pseudo_beta_mask=torch.zeros(1, 1, 2, 2, dtype=torch.bool),
        residue_type_ids=torch.full((1, 1, 2), -10000, dtype=torch.long),
        unit_vector=torch.full((1, 1, 2, 2, 3), torch.nan),
        backbone_frame_mask=torch.zeros(1, 1, 2, 2, dtype=torch.bool),
        template_mask=torch.zeros(1, 1, dtype=torch.bool),
        batch_size=[1],
    )
    embedder = AlphaFold3TemplateEmbedder(
        pair_size=4,
        hidden_size=4,
        num_blocks=1,
        triangle_attention_head_size=2,
        num_triangle_attention_heads=1,
        backend="eager",
    )
    pair = torch.randn(1, 2, 2, 4, requires_grad=True)
    output = embedder(pair, templates, same_chain_mask=torch.ones(1, 2, 2, dtype=torch.bool))
    assert torch.equal(output, torch.zeros_like(output))
    gradient = torch.autograd.grad(output.sum(), pair)[0]
    assert torch.equal(gradient, torch.zeros_like(pair))
    templates.template_mask.fill_(True)
    with pytest.raises(InvalidRequestError):
        embedder(pair, templates, same_chain_mask=torch.ones(1, 2, 2, dtype=torch.bool))
