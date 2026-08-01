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
    @pytest.mark.parametrize("enabled", [False, True])
    def test_repository_roundtrip(self, enabled, tmp_path, boltz2_config, native_repository_roundtrip, peptide):
        torch.manual_seed(779)
        config = boltz2_config()
        config.input_embeddings.add_molecule_type_feature = enabled
        config.denoiser.align_reverse_diffusion = enabled
        config.trunk.trunk_pair_dropout = 0.17
        config.trunk.msa_pair_dropout = 0.43
        if not enabled:
            config.template = None
        model = Boltz2Model(config).double().eval().set_backend("eager")
        processor = Boltz2Processor(dtype=torch.float64)
        structure = peptide("GLY", "GLY", "GLY", "GLY")
        input = processor.prepare_input(structure, generator=np.random.default_rng(408)).features
        loaded, loaded_processor = native_repository_roundtrip(model, processor, tmp_path / "native")
        assert loaded.config.input_embeddings.add_molecule_type_feature is enabled
        assert loaded.config.denoiser.align_reverse_diffusion is enabled
        assert loaded.config.trunk.trunk_pair_dropout == 0.17
        assert loaded.config.trunk.msa_pair_dropout == 0.43
        assert (loaded.config.template is not None) is enabled
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


@pytest.mark.parametrize("affinity", [False, True])
@pytest.mark.parametrize("use_template", [False, True])
@pytest.mark.parametrize("has_template_args", [False, True])
def test_template_config_from_upstream_checkpoint(affinity, use_template, has_template_args):
    from ofoldx.models.boltz2.convert import boltz2_config_from_checkpoint

    pairformer = {
        "num_blocks": 1,
        "num_heads": 1,
        "dropout": 0.17,
        "pairwise_head_width": 4,
        "pairwise_num_heads": 2,
    }
    hyperparameters = {
        "token_s": 8,
        "token_z": 8,
        "atom_s": 8,
        "atom_z": 4,
        "atom_feature_dim": 388,
        "num_bins": 4,
        "atoms_per_window_queries": 4,
        "atoms_per_window_keys": 4,
        "conditioning_cutoff_min": 4.0,
        "conditioning_cutoff_max": 20.0,
        "predict_bfactor": False,
        "use_templates": use_template,
        "training_args": {"recycling_steps": 0},
        "embedder_args": {"atom_encoder_depth": 1, "atom_encoder_heads": 1},
        "pairformer_args": pairformer,
        "msa_args": {
            "msa_blocks": 1,
            "msa_s": 8,
            "msa_dropout": 0.0,
            "z_dropout": 0.43,
            "use_paired_feature": True,
            "pairwise_head_width": 8,
            "pairwise_num_heads": 1,
        },
        "score_model_args": {
            "dim_fourier": 8,
            "token_transformer_depth": 1,
            "token_transformer_heads": 1,
            "atom_encoder_depth": 1,
            "atom_decoder_depth": 1,
            "atom_encoder_heads": 1,
            "conditioning_transition_layers": 1,
            "sigma_data": 16.0,
        },
        "diffusion_process_args": {"alignment_reverse_diff": False},
        "confidence_model_args": {
            "pairformer_args": pairformer,
            "confidence_args": {"num_plddt_bins": 4, "num_pde_bins": 4, "num_pae_bins": 4},
            "num_dist_bins": 4,
            "max_dist": 20.0,
        },
    }
    if has_template_args:
        hyperparameters["template_args"] = {"template_dim": 8, "template_blocks": 1}
    checkpoint = {
        "hyper_parameters": hyperparameters,
        "state_dict": {"input_embedder.mol_type_conditioning_init.weight": torch.empty(0)},
    }
    config = boltz2_config_from_checkpoint(checkpoint, affinity=affinity)
    assert config.input_embeddings.add_molecule_type_feature is True
    assert config.denoiser.align_reverse_diffusion is False
    assert config.trunk.trunk_pair_dropout == 0.17
    assert config.trunk.msa_pair_dropout == 0.43
    assert config.trunk.trunk_triangle_attention_head_size == 4
    assert config.trunk.num_trunk_triangle_attention_heads == 2
    assert config.trunk.msa_triangle_attention_head_size == 8
    assert config.trunk.num_msa_triangle_attention_heads == 1
    assert (config.template is not None) is use_template
    if use_template and has_template_args:
        assert config.template.hidden_size == 8
        assert config.template.num_blocks == 1


def test_model_template_defaults():
    from ofoldx.models.boltz2 import Boltz2AffinityConfig, Boltz2Config

    assert Boltz2Config().template is not None
    assert Boltz2AffinityConfig().template is None
