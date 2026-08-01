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

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ofoldx import AutoModel, AutoProcessor, save_model
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.models.vocabulary import FOLDING_SOURCE_TOKEN_INDEX


@pytest.fixture(scope="class", params=[torch.float32, torch.float64])
def floating_dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture
def boltz2_config():
    from ofoldx.models.boltz2 import Boltz2Config  # a family still to migrate (WU14) must not break collection

    def make_config():
        return Boltz2Config(
            {
                "single_size": 8,
                "pair_size": 8,
                "atom_hidden_size": 8,
                "atom_pair_hidden_size": 4,
                "fourier_embedding_size": 8,
                "num_distogram_bins": 4,
                "num_recycles": 1,
                "input_embeddings": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 1},
                "trunk": {
                    "num_pairformer_blocks": 1,
                    "num_pair_attention_heads": 1,
                    "num_msa_blocks": 1,
                    "msa_size": 8,
                    "msa_attention_head_size": 4,
                    "num_msa_attention_heads": 1,
                    "outer_product_hidden_size": 4,
                    "msa_triangle_attention_head_size": 4,
                    "num_msa_triangle_attention_heads": 1,
                    "triangle_attention_head_size": 4,
                    "num_triangle_attention_heads": 1,
                    "pair_dropout": 0,
                    "msa_dropout": 0,
                    "msa_pair_dropout": 0,
                },
                "template": {
                    "hidden_size": 8,
                    "num_blocks": 1,
                    "num_distogram_bins": 4,
                    "triangle_attention_head_size": 4,
                    "num_triangle_attention_heads": 1,
                    "dropout": 0,
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
                    "num_pairformer_blocks": 1,
                    "num_pair_attention_heads": 1,
                    "triangle_attention_head_size": 4,
                    "num_triangle_attention_heads": 1,
                    "num_distogram_bins": 4,
                    "num_plddt_bins": 4,
                    "num_pae_bins": 4,
                    "num_pde_bins": 4,
                    "pair_dropout": 0,
                },
            }
        )

    return make_config


@pytest.fixture
def mpnn_packer():
    from ofoldx.models.mpnn_packer import (
        MpnnPackerConfig,
        MpnnPackerModel,
        MpnnPackerPipeline,
        MpnnPackerProcessor,
        MpnnPackerProcessorConfig,
    )

    values = {
        "hidden_size": 8,
        "edge_size": 8,
        "num_encoder_blocks": 1,
        "num_decoder_blocks": 1,
        "num_neighbors": 3,
        "num_context_atoms": 1,
    }
    return MpnnPackerPipeline(
        model=MpnnPackerModel(MpnnPackerConfig(values)).cuda().eval(),
        processor=MpnnPackerProcessor(MpnnPackerProcessorConfig(num_context_atoms=1), device="cuda"),
    )


_SOURCE_COLUMNS = [TOKEN_INDEX[token] for token in FOLDING_SOURCE_TOKEN_INDEX]


@pytest.fixture
def source_feature_planes():
    def project(features, prefix_size):
        """The two canonical planes read on the upstream folding alphabet (features supported on its classes)."""
        size = len(TOKENS)
        planes = [features[..., start + np.array(_SOURCE_COLUMNS)] for start in (prefix_size, prefix_size + size)]
        return torch.cat((features[..., :prefix_size], *planes, features[..., prefix_size + 2 * size :]), dim=-1)

    return project


@pytest.fixture
def native_token_features():
    def make_features(prefix_size, suffix_size, num_tokens, generator):
        """Canonical features whose residue one-hot and profile lie on the upstream folding alphabet's classes."""
        size = len(TOKENS)
        prefix = torch.randn(2, num_tokens, prefix_size, dtype=torch.float64, generator=generator)
        residue = torch.zeros(2, num_tokens, size, dtype=torch.float64)
        residue[..., torch.arange(num_tokens), torch.tensor(_SOURCE_COLUMNS)[torch.arange(num_tokens) % 33]] = 1
        profile = torch.zeros(2, num_tokens, size, dtype=torch.float64)
        profile[..., _SOURCE_COLUMNS] = torch.rand(2, num_tokens, 33, dtype=torch.float64, generator=generator)
        profile = profile / profile.sum(dim=-1, keepdim=True)
        suffix = torch.randn(2, num_tokens, suffix_size, dtype=torch.float64, generator=generator)
        if suffix_size == 5:
            suffix[..., 1:] = F.one_hot(torch.arange(num_tokens) % 4, num_classes=4).to(suffix)
        return torch.cat((prefix, residue, profile, suffix), dim=-1)

    return make_features


@pytest.fixture
def placed() -> Callable[..., Structure]:
    """``placed(system)``: every atom at its reference coordinates shifted 4 Å per component, evidence OBSERVED."""

    def build(system) -> Structure:
        reference, _ = system.reference_coordinates()
        owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
        coordinates = reference + 4.0 * owner[:, None] * np.array([1.0, 0.0, 0.0])
        every = np.ones(len(owner), dtype=bool)
        structure, _ = replace_coordinates(
            Structure.unobserved(system),
            np.arange(len(owner)),
            coordinates,
            scope=every,
            evidence=CoordinateEvidence.OBSERVED,
        )
        return structure

    return build


@pytest.fixture
def native_repository_roundtrip():
    def roundtrip(model, processor, directory):
        # Synthetic state validates public save and the real meta/assign loader.
        # Released source weights remain a separate acceptance gate.
        save_model(directory, model=model, processor=processor)
        loaded = AutoModel.from_pretrained(directory, device="cpu", dtype=torch.float64, backend="eager")
        loaded_processor = AutoProcessor.from_pretrained(directory, device="cpu", dtype=torch.float64)
        assert type(loaded) is type(model)
        assert type(loaded_processor) is type(processor)
        expected_state, loaded_state = model.state_dict(), loaded.state_dict()
        assert expected_state.keys() == loaded_state.keys()
        for name, expected in expected_state.items():
            torch.testing.assert_close(loaded_state[name], expected, rtol=0, atol=0)
        return loaded, loaded_processor

    return roundtrip
