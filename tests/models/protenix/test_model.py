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

"""Behavior of ofoldx.models.protenix.model."""

import pytest
import torch

from ofoldx.models.protenix.model import ProtenixNoisyStructureEmbedder


def test_unavailable_structure_payload():
    model = ProtenixNoisyStructureEmbedder(pair_size=8)
    pair = torch.randn(1, 2, 2, 8)
    positions = torch.tensor([[[1.0, 2.0, 3.0], [float("nan"), float("inf"), 0.0]]])
    mask = torch.tensor([[True, False]])
    output = model(pair, positions, mask)
    expected = model(pair, torch.nan_to_num(positions, nan=100.0, posinf=200.0), mask)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.isnan(positions[0, 1, 0])


def test_msa_input_availability(peptide):
    from ofoldx.errors import InvalidRequestError
    from ofoldx.models.protenix import ProtenixProcessor
    from ofoldx.models.protenix.model import normalize_protenix_input

    value = ProtenixProcessor().prepare_input(peptide("ALA", "GLY")).features
    value.msa_tokens = torch.tensor([[[1, -10000]]])
    value.token_mask = torch.tensor([[True, False]])
    value.msa_deletion_counts = torch.tensor([[[100000.0, torch.nan]]], dtype=torch.float64)
    normalized = normalize_protenix_input(value, dtype=torch.float16)
    assert normalized.msa_tokens.tolist() == [[[1, 0]]]
    assert value.msa_tokens.tolist() == [[[1, -10000]]]
    assert normalized.msa_deletion_counts.dtype == torch.float64
    assert normalized.msa_deletion_counts.tolist() == [[[100000.0, 0.0]]]
    value.msa_tokens[0, 0, 0] = -1
    with pytest.raises(InvalidRequestError):
        normalize_protenix_input(value)


def test_pair_local_calibration() -> None:
    from ofoldx.models.protenix.model import _protenix_tm_scores

    probability = torch.zeros(30, 30, 64, dtype=torch.float64)
    probability[..., 4] = 1.0
    chain = torch.arange(30) // 10
    global_ptm, _, chain_ptm, partner_mean, pair_complex, pair_partner_mean = _protenix_tm_scores(
        probability,
        token_has_frame=torch.ones(30, dtype=torch.bool),
        token_chain=chain,
        atom_to_token=torch.arange(30),
        atom_is_polymer=torch.ones(30, dtype=torch.bool),
    )
    center = 2.25
    pair_value = 1.0 / (1.0 + (center / (1.24 * (20 - 15) ** (1.0 / 3.0) - 1.8)) ** 2)
    global_value = 1.0 / (1.0 + (center / (1.24 * (30 - 15) ** (1.0 / 3.0) - 1.8)) ** 2)
    torch.testing.assert_close(global_ptm, probability.new_tensor(global_value))
    expected = probability.new_full((3, 3), pair_value)
    expected.fill_diagonal_(0)
    torch.testing.assert_close(pair_complex, expected)
    torch.testing.assert_close(partner_mean, probability.new_full((3,), pair_value))
    torch.testing.assert_close(pair_partner_mean, expected)
    assert chain_ptm.shape == (3,)
    assert not torch.isclose(global_ptm, pair_complex[0, 1])


def test_partner_frame_eligibility() -> None:
    from ofoldx.models.protenix.model import _protenix_tm_scores

    bins = torch.full((30, 30), 4, dtype=torch.long)
    bins[:10, 10:20] = bins[10:20, :10] = 8
    bins[:10, 20:] = bins[20:, :10] = 1
    probability = torch.nn.functional.one_hot(bins, 64).double().requires_grad_()
    frames = torch.arange(30) < 20
    chain = torch.arange(30) // 10
    _, _, _, partner_mean, pair_complex, _ = _protenix_tm_scores(
        probability,
        token_has_frame=frames,
        token_chain=chain,
        atom_to_token=torch.arange(30),
        atom_is_polymer=torch.ones(30, dtype=torch.bool),
    )
    scale = 1.24 * (20 - 15) ** (1.0 / 3.0) - 1.8
    ab = 1.0 / (1.0 + (4.25 / scale) ** 2)
    ac = 1.0 / (1.0 + (0.75 / scale) ** 2)
    torch.testing.assert_close(partner_mean[0], probability.new_tensor((ab + ac) / 2))
    actual_gradient = torch.autograd.grad(partner_mean[0], probability, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad((pair_complex[0, 1] + pair_complex[0, 2]) / 2, probability)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    _, _, _, absent, _, _ = _protenix_tm_scores(
        probability.detach(),
        token_has_frame=torch.zeros(30, dtype=torch.bool),
        token_chain=chain,
        atom_to_token=torch.arange(30),
        atom_is_polymer=torch.ones(30, dtype=torch.bool),
    )
    assert torch.isnan(absent).all()
