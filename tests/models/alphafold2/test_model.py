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

"""Behavior of ofoldx.models.alphafold2.model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.models.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2Processor,
    alphafold2_predicted_interface_tm_score,
    alphafold2_predicted_tm_score,
)
from ofoldx.models.alphafold2.model import normalize_alphafold2_input


class TestNormalize:
    def test_masked_msa_cells_cleared(self, peptide):
        value = AlphaFold2Processor().prepare_input(peptide("ALA", "GLY")).features
        value.msa_tokens = torch.tensor([[[1, -10000]]])
        value.residue_mask = torch.tensor([[1.0, 0.0]])
        value.msa_deletion_counts = torch.tensor([[[3.0, torch.nan]]])
        normalized = normalize_alphafold2_input(value)
        assert normalized.msa_tokens.tolist() == [[[1, 0]]]
        assert normalized.msa_deletion_counts.tolist() == [[[3.0, 0.0]]]
        assert value.msa_tokens.tolist() == [[[1, -10000]]]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


@pytest.fixture(scope="class")
def tm_case(floating_dtype: torch.dtype):
    index = torch.arange(3 * 3 * 7, dtype=floating_dtype).reshape(3, 3, 7)
    logits = (index * 0.37).sin() + (index.remainder(5) * 0.173)
    weights = torch.tensor([0.25, 0.70, 1.13], dtype=floating_dtype)
    roles = torch.tensor([0, 0, 1])
    config = AlphaFold2Config({"heads": {"predicted_aligned_error": {"num_pae_bins": 7}}})
    breaks = np.linspace(0.0, 31.0, 6)
    step = breaks[1] - breaks[0]
    centers = np.r_[breaks + step / 2, breaks[-1] + 3 * step / 2]
    scale = 1.24 * (19 - 15) ** (1.0 / 3.0) - 1.8
    terms = _softmax(logits.double().numpy()) @ (1 / (1 + (centers / scale) ** 2))
    source = weights.double().numpy()
    target = np.broadcast_to(source, terms.shape)
    means = (terms * target).sum(axis=-1) / target.sum(axis=-1)
    ptm = means[np.argmax(means * source)]
    cross = target * (roles.numpy()[:, None] != roles.numpy()[None, :])
    means = (terms * cross).sum(axis=-1) / cross.sum(axis=-1)
    iptm = means[np.argmax(means * source)]
    tolerance = 2e-13 if floating_dtype == torch.float64 else 2e-6
    return logits, weights, roles, config, ptm, iptm, tolerance


class TestTmScore:
    def test_ptm(self, tm_case):
        logits, weights, _, config, expected, _, tolerance = tm_case
        score = alphafold2_predicted_tm_score(logits, weights, config, backend="eager")
        assert score == pytest.approx(expected, rel=tolerance, abs=tolerance)

    def test_iptm(self, tm_case):
        logits, weights, roles, config, _, expected, tolerance = tm_case
        score = alphafold2_predicted_interface_tm_score(logits, roles, weights, config, backend="eager")
        assert score == pytest.approx(expected, rel=tolerance, abs=tolerance)


def test_point_norm() -> None:
    from ofoldx.models.alphafold2.model import AlphaFold2MultimerInvariantPointAttention, AlphaFold2Rigid

    attention = AlphaFold2MultimerInvariantPointAttention(
        hidden_size=2,
        pair_size=1,
        num_attention_heads=1,
        scalar_query_key_size=1,
        scalar_value_size=1,
        num_point_qk=1,
        num_point_v=1,
    ).double()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
        attention.value_point_projection.point_projection.weight[:2].copy_(torch.eye(2))
        attention.output_projection.weight[0, 4] = 1
    single = torch.tensor(
        [[[0.0, 0.0]], [[3e-9, 4e-9]], [[3e-6, 4e-6]], [[3.0, 4.0]]], dtype=torch.float64, requires_grad=True
    )
    result = attention(
        single=single,
        pair=single.new_zeros(4, 1, 1, 1),
        mask=single.new_ones(4, 1),
        rigid=AlphaFold2Rigid.identity((4, 1), device=single.device, dtype=single.dtype),
    )[..., 0]
    torch.testing.assert_close(result, single.new_tensor([[1e-8], [1e-8], [5e-6], [5.0]]))
    gradient = torch.autograd.grad(result.sum(), single)[0]
    torch.testing.assert_close(gradient, single.new_tensor([[[0, 0]], [[0, 0]], [[0.6, 0.8]], [[0.6, 0.8]]]))
