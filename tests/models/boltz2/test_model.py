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

"""Behavior of ofoldx.models.boltz2.model on prepared inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.errors import InvalidRequestError
from ofoldx.models.boltz2 import Boltz2Model, Boltz2Processor
from ofoldx.models.boltz2.model import normalize_boltz2_input


class TestSampling:
    def test_repository_roundtrip(self, tmp_path, boltz2_config, native_repository_roundtrip, peptide):
        torch.manual_seed(779)
        model = Boltz2Model(boltz2_config()).double().eval().set_backend("eager")
        processor = Boltz2Processor(dtype=torch.float64)
        structure = peptide("GLY", "GLY", "GLY", "GLY")
        input = processor.prepare_input(structure, generator=np.random.default_rng(408)).features
        loaded, loaded_processor = native_repository_roundtrip(model, processor, tmp_path / "native")
        restored_input = loaded_processor.prepare_input(structure, generator=np.random.default_rng(408)).features
        options = {"num_samples": 3, "num_sampling_steps": 1, "num_recycles": 1}
        with torch.no_grad():
            actual = model.sample_with_confidence(input, generator=torch.Generator().manual_seed(208), **options)
            restored = loaded.sample_with_confidence(
                restored_input, generator=torch.Generator().manual_seed(208), **options
            )
        torch.testing.assert_close(actual.coordinates, restored.coordinates, rtol=2e-10, atol=2e-10)
        for name in ("plddt_logits", "pae_logits", "pde_logits", "resolved_logits"):
            torch.testing.assert_close(getattr(actual, name), getattr(restored, name), rtol=2e-10, atol=2e-10)
        assert torch.isfinite(actual.coordinates).all()

    def test_unavailable_input_gradients(self, boltz2_config, peptide):
        input = Boltz2Processor().prepare_input(peptide("ALA"), generator=np.random.default_rng(13)).features
        input.atom_mask[..., -1] = False
        positions = input.reference_positions.detach().clone()
        positions[..., -1, :] = float("nan")
        features = input.atom_features.detach().clone()
        features[..., -1, :] = float("inf")
        input.reference_positions, input.atom_features = positions.requires_grad_(), features.requires_grad_()
        model = Boltz2Model(boltz2_config()).eval().set_backend("eager")
        output = model.representations(input, num_recycles=1, generator=torch.Generator().manual_seed(17))
        assert torch.isfinite(output.single).all()
        for gradient in torch.autograd.grad(output.single.square().sum(), (positions, features)):
            assert torch.isfinite(gradient).all() and not gradient[..., -1, :].any()


class TestNormalize:
    def test_msa_input_availability(self, peptide):
        value = Boltz2Processor().prepare_input(peptide("ALA", "GLY")).features
        value.msa_tokens = torch.tensor([[[1, -10000]]])
        value.msa_mask = torch.tensor([[[True, False]]])
        value.token_mask = torch.tensor([[True, False]])
        value.has_deletion = torch.tensor([[[1.0, torch.nan]]])
        value.deletion_value = torch.tensor([[[0.5, torch.nan]]])
        value.msa_paired = torch.tensor([[[0.0, torch.nan]]])
        normalized = normalize_boltz2_input(value, dtype=torch.float16)
        assert normalized.msa_tokens.tolist() == [[[1, 0]]]
        assert value.msa_tokens.tolist() == [[[1, -10000]]]
        assert normalized.deletion_value.tolist() == [[[0.5, 0.0]]]
        value.msa_tokens[0, 0, 0] = -1
        with pytest.raises(InvalidRequestError):
            normalize_boltz2_input(value)
