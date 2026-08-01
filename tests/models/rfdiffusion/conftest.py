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
Builders of the RFdiffusion tests.

The library fixture uses a specific ``UNK`` definition for generated rows, replacing any subset entry of the same id.
Its atoms and bonds below are the CCD entry's (``_chem_comp_atom`` ideal coordinates, leaving and terminal flags;
``_chem_comp_bond`` orders).
"""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.chemistry.definitions import ComponentLibrary, define_component, load_library, write_artifact
from ofoldx.data.molecular import DefinitionScope
from ofoldx.data.structure import Structure
from ofoldx.provenance import Basis, Resolution, state
from tests.chemistry.builders import ATOM_ITEMS, make_items, provenance

_ATOMS = """
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
_BONDS = (
    "N CA SING, N H SING, N H2 SING, CA C SING, CA CB SING, CA HA SING, C O DOUB, C OXT SING, CB CG SING,"
    " CB HB1 SING, CB HB2 SING, CG HG1 SING, CG HG2 SING, CG HG3 SING, OXT HXT SING"
)


def _unknown_items():
    r"""The ``ComponentItems`` of CCD ``UNK``."""
    rows = [line.split() for line in _ATOMS.strip().splitlines()]
    atoms = {item: [row[index] for row in rows] for index, item in enumerate(ATOM_ITEMS)}
    for item in (*ATOM_ITEMS[8:], "charge"):
        atoms[item] = [float(token) for token in atoms[item]]
    pairs = [bond.split() for bond in _BONDS.split(", ")]
    items = ("atom_id_1", "atom_id_2", "value_order")
    bonds = {item: [bond[index] for bond in pairs] for index, item in enumerate(items)}
    bonds |= {"pdbx_aromatic_flag": ["N"] * len(pairs), "pdbx_stereo_config": ["N"] * len(pairs)}
    component = {"id": "UNK", "type": "L-PEPTIDE LINKING", "one_letter_code": "X", "mon_nstd_parent_comp_id": "?"}
    return make_items(component, atoms, bonds)


@pytest.fixture(scope="session")
def library(tmp_path_factory, ccd) -> ComponentLibrary:
    r"""The CCD subset of ``tests/fixtures/ccd`` with ``UNK`` as an artifact."""
    definitions = [
        define_component(items, scope=DefinitionScope.ARCHIVE) for name, items in ccd.items() if name != "UNK"
    ]
    definitions.append(define_component(_unknown_items(), scope=DefinitionScope.ARCHIVE))
    path = tmp_path_factory.mktemp("rfdiffusion") / "ccd.safetensors"
    write_artifact(definitions, path, provenance("subset-with-unknown"))
    return load_library(path)


@pytest.fixture
def numbered(peptide):
    r"""``numbered("ALA", "GLY")``: ``peptide`` addressed as a source addresses it -- author chain ``A`` and author
    numbers ``1..k``, the identity the contig grammar and the template chains are read by."""

    def build(*components: str) -> Structure:
        structure = peptide(*components)
        arrays = structure.build()
        declared = state(Resolution.KNOWN, Basis.DECLARED)
        arrays.symbols["chain"] = ("A",)
        arrays.symbols["author_number"] = tuple(str(number) for number in range(1, len(components) + 1))
        arrays.columns["instance_author_chain"][:] = 0
        arrays.columns["instance_author_chain_state"][:] = declared
        arrays.columns["component_author_text"] = np.arange(len(components), dtype=np.int32)
        arrays.columns["component_author_number"] = np.arange(1, len(components) + 1, dtype=np.int64)
        arrays.columns["component_author_number_state"][:] = declared
        return Structure.from_arrays(structure.system, arrays)

    return build
