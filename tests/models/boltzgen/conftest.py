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

r"""
Builders of the BoltzGen tests.

``design_library`` adds ``UNK`` to the shared CCD subset for generated positions and decoded residue identities.
"""

from __future__ import annotations

import pytest

from ofoldx.chemistry.definitions import ComponentLibrary, define_component, load_library, write_artifact
from ofoldx.data.molecular import DefinitionScope
from ofoldx.models.boltzgen.config import BoltzGenConfig, BoltzGenInverseFoldingConfig
from tests.chemistry.builders import ATOM_ITEMS, make_items, provenance

#: CCD ``UNK`` atoms in ``ATOM_ITEMS`` order: id, element, charge, aromatic, leaving, stereo, N- and C-terminal
#: flags, and the ideal conformer.
UNKNOWN_ATOMS = """
N   N 0 N N N Y N  1.827 -0.498 -1.278
CA  C 0 N N S N N  2.145 -1.162 -2.517
C   C 0 N N N N Y  1.576 -0.429 -3.719
O   O 0 N N N N Y  0.850  0.554 -3.635
CB  C 0 N N N N N  1.719 -2.634 -2.491
CG  C 0 N N N N N  0.211 -2.855 -2.469
OXT O 0 N Y N N Y  1.959 -0.964 -4.903
H   H 0 N N N Y N  0.900 -0.131 -1.145
H2  H 0 N Y N Y N  2.576 -0.233 -0.660
HA  H 0 N N N N N  3.236 -1.108 -2.608
HB1 H 0 N N N N N  2.152 -3.122 -1.609
HB2 H 0 N N N N N  2.132 -3.146 -3.370
HG1 H 0 N N N N N -0.250 -2.387 -1.594
HG2 H 0 N N N N N -0.007 -3.927 -2.427
HG3 H 0 N N N N N -0.268 -2.456 -3.368
HXT H 0 N Y N N Y  1.587 -0.471 -5.665
"""
UNKNOWN_BONDS = (
    "N CA SING, N H SING, N H2 SING, CA C SING, CA CB SING, CA HA SING, C O DOUB, C OXT SING, CB CG SING, "
    "CB HB1 SING, CB HB2 SING, CG HG1 SING, CG HG2 SING, CG HG3 SING, OXT HXT SING"
)
#: A BoltzGen trunk, denoiser and template stack small enough to run in a test.
SMALL_DESIGN = {
    "single_size": 8,
    "pair_size": 8,
    "atom_hidden_size": 8,
    "atom_pair_hidden_size": 4,
    "num_distogram_bins": 4,
    "fourier_embedding_size": 8,
    "num_recycles": 1,
    "input_embeddings": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 1},
    "trunk": {
        "num_trunk_blocks": 1,
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
    "template": {"num_distogram_bins": 4, "triangle_attention_head_size": 4, "num_triangle_attention_heads": 1},
    "denoiser": {
        "num_token_encoder_blocks": 1,
        "num_token_encoder_attention_heads": 1,
        "num_atom_encoder_blocks": 1,
        "num_atom_decoder_blocks": 1,
        "num_atom_attention_heads": 1,
        "atom_hidden_size": 8,
        "atom_pair_hidden_size": 4,
        "num_query_atoms": 4,
        "num_key_atoms": 4,
        "num_conditioning_transitions": 1,
        "transformer_size": 8,
    },
    "design": {"template_size": 8, "num_template_blocks": 1, "use_token_distance": False, "template_dropout": 0},
}
#: The inverse folder's atom encoder and residue graph, small and without dropout.
SMALL_INVERSE_FOLDING = {
    "single_size": 8,
    "atom_hidden_size": 8,
    "atom_pair_hidden_size": 4,
    "input_embeddings": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 1},
    "inverse_folding": {
        "node_size": 8,
        "edge_size": 8,
        "hidden_size": 8,
        "num_encoder_blocks": 1,
        "num_decoder_blocks": 1,
        "num_attention_heads": 1,
        "num_neighbors": 2,
        "num_atom_encoder_blocks": 1,
        "num_atom_encoder_attention_heads": 1,
        "num_query_atoms": 4,
        "num_key_atoms": 4,
        "dropout": 0,
        "softmax_dropout": 0,
        "coordinate_noise": 0,
    },
}


def unknown_columns() -> dict:
    r"""Columns of CCD ``UNK`` for ``make_items``, in the shape ``glycine_columns`` has."""
    rows = [line.split() for line in UNKNOWN_ATOMS.strip().splitlines()]
    atoms = {item: [row[k] for row in rows] for k, item in enumerate(ATOM_ITEMS)}
    for item in ATOM_ITEMS[8:] + ("charge",):
        atoms[item] = [float(token) for token in atoms[item]]
    pairs = [bond.split() for bond in UNKNOWN_BONDS.split(", ")]
    bonds = {item: [bond[k] for bond in pairs] for k, item in enumerate(("atom_id_1", "atom_id_2", "value_order"))}
    bonds |= {"pdbx_aromatic_flag": ["N"] * len(pairs), "pdbx_stereo_config": ["N"] * len(pairs)}
    component = {"id": "UNK", "type": "L-PEPTIDE LINKING", "one_letter_code": "X", "mon_nstd_parent_comp_id": "?"}
    return {"component": component, "atoms": atoms, "bonds": bonds}


@pytest.fixture(scope="session")
def design_library(tmp_path_factory, ccd) -> ComponentLibrary:
    r"""The shared CCD subset plus ``UNK`` for generated BoltzGen chains."""
    definitions = [
        define_component(items, scope=DefinitionScope.ARCHIVE) for name, items in ccd.items() if name != "UNK"
    ]
    definitions.append(define_component(make_items(**unknown_columns()), scope=DefinitionScope.ARCHIVE))
    path = tmp_path_factory.mktemp("boltzgen") / "design.safetensors"
    write_artifact(definitions, path, provenance("design"))
    return load_library(path)


@pytest.fixture
def design_config() -> BoltzGenConfig:
    return BoltzGenConfig(SMALL_DESIGN)


@pytest.fixture
def inverse_folding_config() -> BoltzGenInverseFoldingConfig:
    return BoltzGenInverseFoldingConfig(SMALL_INVERSE_FOLDING)
