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

"""Boltz-1 sampling survives public native repository save and load."""

import numpy as np
import pytest
import torch

from ofoldx.errors import InvalidRequestError, UnsupportedError
from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model, Boltz1Processor
from ofoldx.models.boltz1.model import Boltz1PoseBustersPotential, Boltz1VdwOverlapPotential, normalize_boltz1_input


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("element", ["unknown", "unsupported", "soft", "unused", "masked"])
def test_physical_pair_elements(geometry, element):
    elements = torch.nn.functional.one_hot(torch.full((1, 5), 6), 128).double()
    atom_mask = torch.tensor([[True, True, True, True, False]])
    elements[0, 4] = float("nan")
    row = 4 if element == "unused" else 0
    elements[0, row] = 0
    elements[0, row, 119 if element == "unsupported" else 0] = 1
    if element == "soft":
        elements[0, row] = 0
        elements[0, row, 6] = elements[0, row, 8] = 0.5
    if element == "masked":
        atom_mask[0, row] = False
        elements[0, row] = float("nan")
    feats = {
        "reference_elements": elements,
        "atom_mask": atom_mask,
        "atom_to_token": torch.arange(5).unsqueeze(0),
        "chain_index": torch.tensor([[0, 0, 1, 1, 2]]),
        "connected_chain_index": torch.empty(1, 2, 0, dtype=torch.long),
        "geometry_pair_index": torch.tensor([[[0], [2]]]),
        "geometry_lower_bounds": torch.tensor([[1.0]]),
        "geometry_upper_bounds": torch.tensor([[2.0]]),
        "geometry_bond_mask": torch.tensor([[True]]),
        "geometry_angle_mask": torch.tensor([[False]]),
    }
    if geometry:
        potential = Boltz1PoseBustersPotential({})
        parameters = {"bond_buffer": 0.125, "angle_buffer": 0.125, "clash_buffer": 0.1}
    else:
        potential = Boltz1VdwOverlapPotential({})
        parameters = {"buffer": 0.225}
    if element in {"unused", "masked"}:
        index, (_, lower, upper), _ = potential.compute_args(feats, parameters)
        assert lower is not None and torch.isfinite(lower).all()
        if upper is not None:
            assert torch.isfinite(upper).all()
        assert not bool((index == row).any())
    else:
        error = {
            "unknown": InvalidRequestError,
            "unsupported": UnsupportedError,
            "soft": InvalidRequestError,
        }[element]
        with pytest.raises(error):
            potential.compute_args(feats, parameters)


def _tiny_boltz_config(imitate_trunk, use_single_diffusion):
    return Boltz1Config(
        {
            "single_size": 8,
            "pair_size": 8,
            "atom_hidden_size": 8,
            "atom_pair_hidden_size": 4,
            "fourier_embedding_size": 8,
            "num_distogram_bins": 4,
            "num_recycles": 1,
            "use_msa": False,
            "input_embeddings": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 1},
            "trunk": {
                "num_pairformer_blocks": 1,
                "num_pair_attention_heads": 1,
                "triangle_attention_head_size": 4,
                "num_triangle_attention_heads": 1,
                "pair_dropout": 0,
            },
            "denoiser": {
                "atom_hidden_size": 8,
                "atom_pair_hidden_size": 4,
                "num_query_atoms": 4,
                "num_key_atoms": 4,
                "num_atom_encoder_blocks": 1,
                "num_atom_decoder_blocks": 1,
                "num_token_encoder_blocks": 1,
                "num_atom_attention_heads": 1,
                "num_token_encoder_attention_heads": 1,
                "num_conditioning_transitions": 2,
            },
            "confidence": {
                "imitate_trunk": imitate_trunk,
                "num_pairformer_blocks": 1,
                "num_pair_attention_heads": 1,
                "triangle_attention_head_size": 4,
                "num_triangle_attention_heads": 1,
                "num_distogram_bins": 4,
                "num_plddt_bins": 4,
                "num_pae_bins": 4,
                "num_pde_bins": 4,
                "pair_dropout": 0,
                "use_single_diffusion": use_single_diffusion,
            },
        }
    )


@pytest.mark.parametrize("imitate_trunk", [False, True])
@pytest.mark.parametrize("use_single_diffusion", [False, True])
def test_repository_roundtrip(imitate_trunk, use_single_diffusion, tmp_path, native_repository_roundtrip, peptide):
    torch.manual_seed(614)
    model = Boltz1Model(_tiny_boltz_config(imitate_trunk, use_single_diffusion)).double().eval().set_backend("eager")
    processor = Boltz1Processor(dtype=torch.float64)
    structure = peptide("GLY", "GLY", "GLY", "GLY")
    input = processor.prepare_input(structure, generator=np.random.default_rng(116)).features
    loaded, loaded_processor = native_repository_roundtrip(model, processor, tmp_path / "native")
    restored_input = loaded_processor.prepare_input(structure, generator=np.random.default_rng(116)).features
    with torch.no_grad():
        actual = model.sample_with_confidence(
            input, num_samples=3, num_sampling_steps=1, num_recycles=1, generator=torch.Generator().manual_seed(223)
        )
        restored = loaded.sample_with_confidence(
            restored_input,
            num_samples=3,
            num_sampling_steps=1,
            num_recycles=1,
            generator=torch.Generator().manual_seed(223),
        )
    torch.testing.assert_close(actual.coordinates, restored.coordinates, rtol=2e-10, atol=2e-10)
    for name in ("plddt_logits", "pae_logits", "pde_logits", "resolved_logits"):
        torch.testing.assert_close(
            getattr(actual.confidence, name), getattr(restored.confidence, name), rtol=2e-10, atol=2e-10
        )
        assert torch.isfinite(getattr(actual.confidence, name)).all()
    assert torch.isfinite(actual.coordinates).all()


def test_msa_input_availability(peptide):
    value = Boltz1Processor().prepare_input(peptide("ALA", "GLY")).features
    value.msa_tokens = torch.tensor([[[1, -10000]]])
    value.msa_mask = torch.tensor([[[True, False]]])
    value.token_mask = torch.tensor([[True, False]])
    value.has_deletion = torch.tensor([[[1.0, torch.nan]]])
    value.deletion_value = torch.tensor([[[0.5, torch.nan]]])
    value.msa_paired = torch.tensor([[[0.0, torch.nan]]])
    normalized = normalize_boltz1_input(value, dtype=torch.float16)
    assert normalized.msa_tokens.tolist() == [[[1, 0]]]
    assert value.msa_tokens.tolist() == [[[1, -10000]]]
    assert normalized.deletion_value.dtype == torch.float16
    assert normalized.deletion_value.tolist() == [[[0.5, 0.0]]]
    value.msa_tokens[0, 0, 0] = -1
    with pytest.raises(InvalidRequestError):
        normalize_boltz1_input(value)
