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

"""Behavior of ofoldx.models.odesign.model."""

from __future__ import annotations

import json

import pytest
import torch

from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.errors import InvalidRequestError
from ofoldx.models.odesign import ODesignBackboneComponent, ODesignConfig, ODesignModel
from ofoldx.models.odesign.model import ODESIGN_SOURCE_TOKEN_INDEX, normalize_odesign_input, odesign_source_map
from ofoldx.models.odesign.processor import ODesignProcessor


def test_dropout_configuration_roundtrip_and_training_rng(tmp_path, peptide):
    config = ODesignConfig(
        single_size=16,
        pair_size=8,
        constraint_embeddings={"hidden_size": 8, "num_blocks": 1, "num_distogram_bins": 4, "dropout": 0.0},
        trunk={
            "num_pairformer_blocks": 1,
            "num_msa_stages": 2,
            "trunk_pair_dropout": 0.11,
            "msa_dropout": 0.0,
            "msa_pair_dropout": 0.37,
        },
        denoiser={
            "num_token_encoder_blocks": 1,
            "num_token_encoder_attention_heads": 1,
            "atom_hidden_size": 8,
            "atom_pair_hidden_size": 4,
            "token_hidden_size": 16,
            "num_atom_encoder_blocks": 1,
            "num_atom_encoder_attention_heads": 1,
            "num_atom_decoder_blocks": 1,
            "num_atom_decoder_attention_heads": 1,
            "num_query_atoms": 4,
            "num_key_atoms": 4,
        },
        pairwise={"num_distogram_bins": 4, "bond_hidden_size": 8},
    )
    processor = ODesignProcessor()
    model = ODesignModel(config).set_backend("eager")
    component = ODesignBackboneComponent(model=model, processor=processor)
    component.save(tmp_path)
    payload = json.loads((tmp_path / "model.json").read_text())
    assert payload["trunk"]["trunk_pair_dropout"] == 0.11
    assert payload["trunk"]["msa_pair_dropout"] == 0.37
    assert "pairformer_dropout" not in payload["trunk"] and "pair_dropout" not in payload["trunk"]
    restored = ODesignBackboneComponent.from_pretrained(tmp_path, backend="eager")
    assert restored.model.config.trunk.trunk_pair_dropout == 0.11
    assert restored.model.config.trunk.msa_pair_dropout == 0.37
    structure = peptide("ALA", "GLY")
    prepared, target = processor.prepare(structure, target=structure)

    def run(candidate):
        candidate.train()
        torch.manual_seed(171)
        generator = torch.Generator().manual_seed(771)
        with torch.no_grad():
            output = candidate(prepared.features, target, num_recycles=1, num_samples=1, generator=generator)
        return output, torch.get_rng_state(), generator.get_state()

    expected, expected_global_rng, expected_generator_rng = run(model)
    actual, actual_global_rng, actual_generator_rng = run(restored.model)
    for name in ("coordinates", "distogram_logits", "token_bond_logits", "noise_level"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    assert torch.equal(actual_global_rng, expected_global_rng)
    assert torch.equal(actual_generator_rng, expected_generator_rng)


class TestSourceMap:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("A", {"A": 1.0}),
            ("OTHER", {"X": 1.0}),
            ("RNA_X", {"RNA_N": 1.0}),
            ("B", {"D": 0.5, "N": 0.5}),
            ("<mask>", {}),
        ],
    )
    def test_canonical_columns(self, token, expected):
        column = odesign_source_map()[:, TOKEN_INDEX[token]]
        rows = {name: float(column[row]) for name, row in ODESIGN_SOURCE_TOKEN_INDEX.items() if column[row]}
        assert rows == expected


class TestNormalize:
    def test_masked_msa_projected(self, peptide):
        prepared = ODesignProcessor().prepare_input(peptide("ALA", "GLY"))
        input = prepared.features
        input.msa_mask = torch.zeros_like(input.msa_mask)
        input.msa_tokens = torch.full_like(input.msa_tokens, -10000)
        normalized = normalize_odesign_input(input)
        assert int(normalized.msa_tokens.max()) == 0
        assert not bool(normalized.msa_mask.any())

    def test_fixed_atom_needs_observation(self, peptide):
        prepared = ODesignProcessor().prepare_input(peptide("ALA", "GLY"))
        input = prepared.features
        input.fixed_coordinate_mask = torch.ones_like(input.fixed_coordinate_mask)
        input.observed_mask = torch.zeros_like(input.observed_mask)
        with pytest.raises(InvalidRequestError) as caught:
            normalize_odesign_input(input)
        assert caught.value.refs[0].argument == "fixed_atoms"
