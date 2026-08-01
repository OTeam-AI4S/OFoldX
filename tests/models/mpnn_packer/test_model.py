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

"""Behavior of ofoldx.models.mpnn_packer.model."""

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError, UnsupportedError


@pytest.mark.parametrize("entry", ["forward", "encode"])
def test_unavailable_coordinates(small_packer, peptide, entry):
    input = small_packer.prepare(peptide("LYS", "LYS")).features
    input.context_mask[:] = False
    input.residue_mask[0, -1] = False
    points = torch.zeros((1, 2, 14, 3))
    available = torch.ones((1, 2, 14), dtype=torch.bool)
    available[0, -1] = False
    model = small_packer.model

    def evaluate():
        result = getattr(model, entry)(input, atom14_coordinates=points, atom14_mask=available)
        return result.mean if entry == "forward" else result[0]

    expected = evaluate()
    input.sequence_tokens[0, -1] = 10**6
    input.context_atomic_numbers.fill_(10**6)
    input.context_coordinates.fill_(float("nan"))
    points[~available] = float("nan")
    points.requires_grad_()
    actual = evaluate()
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.square().sum(), points)[0]
    assert torch.isfinite(gradient).all()
    assert not gradient[~available].any()


def test_inactive_sample_tokens(small_packer, peptide):
    input = small_packer.prepare(peptide("LYS", "LYS")).features
    input.residue_mask[0, -1] = False
    input.pack_mask[0, -1] = False

    def sample():
        return small_packer.model.sample(
            input,
            num_denoising_steps=1,
            num_torsion_samples=1,
            generator=torch.Generator().manual_seed(19),
        )

    expected = sample()
    input.sequence_tokens[0, -1] = -1
    actual = sample()
    for name in ("atom14_coordinates", "atom14_mask", "mean", "concentration", "mix_logits"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    assert input.sequence_tokens[0, -1] == -1


@pytest.mark.parametrize("entry", ["encode", "decode"])
def test_inactive_feature_tokens(small_packer, peptide, entry):
    input = small_packer.prepare(peptide("LYS", "LYS")).features
    input.residue_mask[0, -1] = False
    arguments = {
        "atom14_coordinates": torch.zeros((1, 2, 14, 3)),
        "atom14_mask": torch.ones((1, 2, 14), dtype=torch.bool),
        "sequence_tokens": input.sequence_tokens,
        "residue_mask": input.residue_mask,
        "context_coordinates": input.context_coordinates,
        "context_atomic_numbers": input.context_atomic_numbers,
        "context_mask": input.context_mask,
    }
    if entry == "encode":
        arguments.update(residue_index=input.residue_index, chain_index=input.chain_index)
    else:
        arguments["neighbor_indices"] = torch.zeros((1, 2, 1), dtype=torch.int64)
    operation = getattr(small_packer.model.feature_encoder, entry)
    expected = operation(**arguments)
    input.sequence_tokens[0, -1] = 10**6
    input.context_atomic_numbers[~input.context_mask] = -1
    actual = operation(**arguments)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_active_token_range(small_packer, peptide):
    input = small_packer.prepare(peptide("LYS")).features
    input.sequence_tokens[0, 0] = -1
    with pytest.raises(InvalidRequestError):
        small_packer.model(
            input,
            atom14_coordinates=torch.zeros((1, 1, 14, 3)),
            atom14_mask=torch.ones((1, 1, 14), dtype=torch.bool),
        )


def test_unknown_packing_template(small_packer, peptide):
    input = small_packer.prepare(peptide("LYS")).features
    input.sequence_tokens[0, 0] = TOKEN_INDEX["X"]
    with pytest.raises(UnsupportedError):
        small_packer.model.sample(input, num_denoising_steps=1, num_torsion_samples=1)


@pytest.mark.parametrize("element", [-1, 120, float("nan"), 1.5])
class TestContextElements:
    @pytest.fixture
    def context_input(self, small_packer, peptide, element):
        input = small_packer.prepare(peptide("LYS")).features
        input.context_mask[..., 0] = True
        input.context_atomic_numbers = input.context_atomic_numbers.to(torch.float64)
        input.context_atomic_numbers[..., 0] = element
        return input

    @pytest.fixture
    def atom_state(self):
        return {
            "atom14_coordinates": torch.zeros((1, 1, 14, 3)),
            "atom14_mask": torch.ones((1, 1, 14), dtype=torch.bool),
        }

    @pytest.mark.parametrize("entry", ["forward", "encode"])
    def test_invalid_model_context(self, small_packer, context_input, atom_state, entry):
        with pytest.raises(InvalidRequestError):
            getattr(small_packer.model, entry)(context_input, **atom_state)

    @pytest.mark.parametrize("entry", ["encode", "decode"])
    def test_invalid_feature_context(self, small_packer, context_input, atom_state, entry):
        arguments = dict(
            **atom_state,
            sequence_tokens=context_input.sequence_tokens,
            residue_mask=context_input.residue_mask,
            context_coordinates=context_input.context_coordinates,
            context_atomic_numbers=context_input.context_atomic_numbers,
            context_mask=context_input.context_mask,
        )
        if entry == "encode":
            arguments.update(residue_index=context_input.residue_index, chain_index=context_input.chain_index)
        else:
            arguments["neighbor_indices"] = torch.zeros((1, 1, 1), dtype=torch.int64)
        with pytest.raises(InvalidRequestError):
            getattr(small_packer.model.feature_encoder, entry)(**arguments)
