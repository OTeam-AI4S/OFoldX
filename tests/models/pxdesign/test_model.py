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

"""Behavior of ofoldx.models.pxdesign.model."""

from __future__ import annotations

import pytest
import torch

from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import InvalidRequestError
from ofoldx.models.pxdesign import PxDesignConfig, PxDesignModel, PxDesignProcessor
from ofoldx.models.pxdesign.processor import from_binder

SMALL = {
    "single_size": 8,
    "pair_size": 8,
    "token_feature_size": 8,
    "num_template_bins": 4,
    "condition_embeddings": {
        "atom_hidden_size": 8,
        "atom_pair_hidden_size": 4,
        "token_hidden_size": 8,
        "num_atom_encoder_blocks": 1,
        "num_atom_encoder_attention_heads": 1,
        "num_query_atoms": 4,
        "num_key_atoms": 4,
    },
    "denoiser": {
        "token_hidden_size": 8,
        "atom_hidden_size": 8,
        "atom_pair_hidden_size": 4,
        "num_token_encoder_blocks": 1,
        "num_token_encoder_attention_heads": 1,
        "num_atom_encoder_blocks": 1,
        "num_atom_encoder_attention_heads": 1,
        "num_atom_decoder_blocks": 1,
        "num_atom_decoder_attention_heads": 1,
        "num_query_atoms": 4,
        "num_key_atoms": 4,
    },
}


@pytest.fixture
def model():
    torch.manual_seed(17)
    return PxDesignModel(PxDesignConfig(SMALL)).eval()


@pytest.fixture
def input(design_library):
    return PxDesignProcessor().prepare_input(placement=from_binder(2), library=design_library).features


class TestModel:
    def test_canonical_token_axis(self, model, input):
        assert input.residue_type_features.shape[-1] == len(TOKENS)
        assert model.config.condition_embeddings.vocab_size == len(TOKENS)

    def test_denoised_shape(self, model, input):
        noisy = input.reference_positions[:, None]
        with torch.inference_mode():
            output = model(input, noisy_coordinates=noisy, noise_level=torch.ones(1, 1))
        assert output.coordinates.shape == noisy.shape

    def test_sampled_candidates(self, model, input):
        with torch.inference_mode():
            output = model.sample(input, num_samples=2, num_sampling_steps=1, sample_chunk_size=None)
        assert output.coordinates.shape == (1, 2, input.atom_mask.shape[-1], 3)

    @pytest.mark.parametrize("num_samples,chunk", [(0, 1), (1, 0)])
    def test_invalid_sampling_request(self, model, input, num_samples, chunk):
        with pytest.raises(InvalidRequestError, match="value_invalid"):
            model.sample(input, num_samples=num_samples, num_sampling_steps=1, sample_chunk_size=chunk)
