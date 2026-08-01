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

import pytest

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.structure import Structure
from ofoldx.models.alphafold2 import (
    AlphaFold2Config,
    AlphaFold2InitialGuessConfig,
    AlphaFold2InitialGuessModel,
    AlphaFold2InitialGuessPipeline,
    AlphaFold2InitialGuessProcessor,
    AlphaFold2Model,
    AlphaFold2MultimerConfig,
    AlphaFold2MultimerModel,
    AlphaFold2MultimerPipeline,
    AlphaFold2MultimerProcessor,
    AlphaFold2Pipeline,
    AlphaFold2Processor,
)


@pytest.fixture
def chains(library, placed) -> Callable[..., Structure]:
    """``chains(A=("ALA", "GLY"), B=("GLY",))``: one protein instance per keyword (identical sequences share an
    entity), ``placed``."""

    def build(**instances: tuple[str, ...]) -> Structure:
        entities: dict = {}
        for components in instances.values():
            entities.setdefault(components, f"P{len(entities)}")
        recipe = Recipe(
            {name: PolymerSequence("polypeptide(L)", components) for components, name in entities.items()},
            {instance: entities[components] for instance, components in instances.items()},
        )
        system, _, _ = build_system(recipe, library)
        return placed(system)

    return build


def _values(multimer: bool, pae: bool) -> dict:
    return {
        "template": {"enabled": False},
        "evoformer": {
            "num_blocks": 1,
            "num_extra_msa_blocks": 0,
            "msa_size": 8,
            "extra_msa_size": 8,
            "pair_size": 8,
            "single_size": 8,
            "block": {
                "msa_row_attention_with_pair_bias": {"num_attention_heads": 1, "dropout": 0.0},
                "msa_transition": {"expansion": 2},
                "outer_product_mean": {"first": multimer, "hidden_size": 4},
                "triangle_multiplication_outgoing": {"hidden_size": 4},
                "triangle_attention_starting_node": {"num_attention_heads": 1, "head_size": 4, "dropout": 0.0},
                "pair_transition": {"expansion": 2},
            },
        },
        "structure_module": {
            "dropout": 0.0,
            "hidden_size": 8,
            "num_attention_heads": 1,
            "num_blocks": 1,
            "num_transition_projections": 1,
            "num_point_qk": 1,
            "num_point_v": 1,
            "scalar_query_key_size": 2,
            "scalar_value_size": 2,
            "sidechain": {"hidden_size": 8, "num_blocks": 1},
        },
        "heads": {
            "distogram": {"num_distogram_bins": 4},
            "predicted_aligned_error": {"enabled": pae, "num_pae_bins": 4},
            "predicted_lddt": {"num_plddt_bins": 4, "hidden_size": 8},
        },
        "msa": {
            "max_msa_clusters": 1,
            "max_extra_msa": 0,
            "masked_msa_enabled": False,
            "resample_msa_in_recycling": False,
        },
        "num_recycles": 1,
    }


@pytest.fixture
def tiny_alphafold2():
    """``tiny_alphafold2("monomer" | "multimer" | "initial_guess", **config)``: a one-block pipeline of that variant,
    with ``config`` overriding top-level configuration values."""

    def build(kind: str, **config):
        if kind == "initial_guess":
            model = AlphaFold2InitialGuessModel(AlphaFold2InitialGuessConfig({**_values(False, True), **config}))
            model = model.eval().set_backend("eager")
            return AlphaFold2InitialGuessPipeline(model=model, processor=AlphaFold2InitialGuessProcessor())
        if kind == "multimer":
            model = AlphaFold2MultimerModel(AlphaFold2MultimerConfig({**_values(True, True), **config}))
            return AlphaFold2MultimerPipeline(model=model.eval(), processor=AlphaFold2MultimerProcessor())
        model = AlphaFold2Model(AlphaFold2Config({**_values(False, False), **config}))
        return AlphaFold2Pipeline(model=model.eval(), processor=AlphaFold2Processor())

    return build
