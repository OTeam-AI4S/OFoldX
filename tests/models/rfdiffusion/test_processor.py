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

"""Behavior of ofoldx.models.rfdiffusion.processor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.chemistry.construction import INHERITED_COLUMNS
from ofoldx.data.molecular import AtomSite, ComponentRef, MolecularSystem
from ofoldx.data.projection import project_atoms
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.data.tokenization import TOKEN_INDEX
from ofoldx.io import parse_source, read_structure
from ofoldx.models.conditions import ExistingSpan, GeneratedSegment, SymbolicBond
from ofoldx.models.design import PositionSelection
from ofoldx.models.rfdiffusion.processor import (
    RfDiffusionPlacements,
    RfDiffusionPointGroupSymmetry,
    RfDiffusionProcessor,
    from_contig_string,
    to_contig_string,
)
from ofoldx.templates import Template, TemplateAlignment, chain_coordinates

MASK = TOKEN_INDEX["<mask>"]


def ids(structure) -> list[str]:
    system = structure.system
    return system.text("component", system.component_symbol).tolist()


def scoped_components(prepared) -> list[int]:
    r"""The target components the regeneration scope covers."""
    pointer = prepared.context.structure.system.component_atom_ptr
    owner = np.repeat(np.arange(len(pointer) - 1), np.diff(pointer))
    return sorted(set(owner[prepared.context.scope.atoms].tolist()))


@pytest.fixture
def scaffold():
    r"""One motif residue followed by two generated ones."""
    return RfDiffusionPlacements(
        {"chain": {"motif": ExistingSpan("A", 0, 1), "new": GeneratedSegment("protein", 2, 2)}}
    )


MOTIF_ATOMS = """
CYS 1 N N 0.000 0.000 0.000
CYS 1 CA C 1.458 0.000 0.000
CYS 1 C C 2.009 1.420 0.000
CYS 1 O O 1.251 2.390 0.000
CYS 1 CB C 1.988 -0.773 1.199
CYS 1 SG S 3.785 -0.773 1.199
GLY 2 N N 3.332 1.536 0.000
GLY 2 CA C 3.988 2.839 0.000
GLY 2 C C 5.504 2.704 0.000
GLY 2 O O 6.073 1.611 0.000
GLY 3 N N 6.168 3.858 0.000
GLY 3 CA C 7.624 3.888 0.000
GLY 3 C C 8.171 5.308 0.000
GLY 3 O O 7.412 6.278 0.000
CYS 4 N N 9.494 5.424 0.000
CYS 4 CA C 10.150 6.727 0.000
CYS 4 C C 11.666 6.592 0.000
CYS 4 O O 12.235 5.499 0.000
CYS 4 CB C 9.620 7.500 1.199
CYS 4 SG S 5.835 -0.773 1.199
"""
MOTIF = """data_MOTIF
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 CYS
1 2 GLY
1 3 GLY
1 4 CYS
_struct_asym.id A
_struct_asym.entity_id 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.label_atom_id
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.occupancy
_atom_site.pdbx_PDB_model_num
{rows}
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_symmetry
disulf1 disulf A CYS 1 SG A CYS 4 SG 1_555 1_555
metalc1 metalc A GLY 2 O A GLY 3 O 1_555 1_555
"""


@pytest.fixture
def motif(library):
    r"""Chain ``A`` CYS GLY GLY CYS, read from mmCIF: a disulfide between the two CYS and a coordination edge between
    the two GLY O."""
    rows = [f"ATOM {k} {line} A 1 1.00 1" for k, line in enumerate(MOTIF_ATOMS.strip().splitlines(), 1)]
    structure, _ = read_structure(parse_source(MOTIF.format(rows="\n".join(rows)), format="mmcif"), library=library)
    return structure


class TestPrepare:
    def test_generated_rows_masked(self, peptide, library, scaffold):
        prepared = RfDiffusionProcessor().prepare_input(peptide("ALA", "GLY"), placement=scaffold, library=library)
        assert ids(prepared.context.structure) == ["ALA", "UNK", "UNK"]
        assert prepared.features.residue_type_features[0].argmax(-1).tolist() == [TOKEN_INDEX["A"], MASK, MASK]
        assert prepared.features.sequence_visible_mask.tolist() == [[True, False, False]]

    def test_motif_outside_scope(self, peptide, library, scaffold):
        prepared = RfDiffusionProcessor().prepare_input(peptide("ALA", "GLY"), placement=scaffold, library=library)
        assert prepared.features.motif_mask.tolist() == [[True, False, False]]
        assert prepared.context.scope.rule == "components"
        assert scoped_components(prepared) == [1, 2]

    def test_motif_keeps_coordinates(self, peptide, library, scaffold):
        source = peptide("ALA", "GLY")
        prepared = RfDiffusionProcessor().prepare_input(source, placement=scaffold, library=library)
        assert prepared.features.noisy_coordinate_mask[0, 0, :4].all()
        assert not prepared.features.noisy_coordinate_mask[0, 1].any()
        alpha = int(source.system.atom_index([AtomSite("A", 0, "CA")])[0])
        expected = torch.from_numpy(source.coordinates[alpha]).to(prepared.features.noisy_coordinates.dtype)
        torch.testing.assert_close(prepared.features.noisy_coordinates[0, 0, 1], expected)

    def test_copied_evidence_kept(self, peptide, library, scaffold):
        source = peptide("ALA", "GLY")
        every = np.ones(len(source.coordinates), dtype=bool)
        predicted, _ = replace_coordinates(
            source,
            np.arange(len(source.coordinates)),
            source.coordinates,
            scope=every,
            evidence=CoordinateEvidence.PREDICTED,
        )
        prepared = RfDiffusionProcessor().prepare_input(predicted, placement=scaffold, library=library)
        alpha = int(prepared.context.structure.system.atom_index([AtomSite("chain_0", 0, "CA")])[0])
        assert prepared.context.structure.coordinate_evidence[alpha] == CoordinateEvidence.PREDICTED

    def test_copied_definition_must_match(self, peptide, library, scaffold, invalid):
        source = peptide("ALA", "GLY")
        arrays = source.system.build()
        arrays.columns["component_definition_digest"][0] += 1
        other = Structure.from_arrays(MolecularSystem.from_arrays(arrays), source.build())
        processor = RfDiffusionProcessor()
        invalid(processor.prepare, other, placement=scaffold, library=library, code="definition_set_mismatch")

    def test_rows_are_linked(self, peptide, library, scaffold):
        prepared = RfDiffusionProcessor().prepare_input(peptide("ALA", "GLY"), placement=scaffold, library=library)
        system = prepared.context.structure.system
        carbon = int(system.atom_index([AtomSite("chain_0", 0, "C")])[0])
        nitrogen = int(system.atom_index([AtomSite("chain_0", 1, "N")])[0])
        first, second = system.edge_index
        assert ((first == carbon) & (second == nitrogen)).any() or ((first == nitrogen) & (second == carbon)).any()

    def test_inpaint_sequence_hides_identity(self, peptide, library):
        placement = RfDiffusionPlacements({"chain": {"motif": ExistingSpan("A", 0, 2)}})
        prepared = RfDiffusionProcessor().prepare_input(
            peptide("ALA", "GLY"),
            placement=placement,
            inpaint_sequence=PositionSelection(components=(ComponentRef("A", 1),)),
            library=library,
        )
        assert prepared.features.residue_type_features[0].argmax(-1).tolist() == [TOKEN_INDEX["A"], MASK]
        assert prepared.features.motif_mask.all()

    def test_inpaint_structure_diffuses(self, peptide, library):
        placement = RfDiffusionPlacements({"chain": {"motif": ExistingSpan("A", 0, 2)}})
        prepared = RfDiffusionProcessor().prepare_input(
            peptide("ALA", "GLY"),
            placement=placement,
            inpaint_structure=PositionSelection(components=(ComponentRef("A", 1),)),
            library=library,
        )
        assert prepared.features.motif_mask.tolist() == [[True, False]]
        assert scoped_components(prepared) == [1]

    def test_symmetry_units(self, peptide, library):
        placement = RfDiffusionPlacements({"chain": {"new": GeneratedSegment("protein", 2, 2)}})
        prepared = RfDiffusionProcessor().prepare_input(
            peptide("ALA"), placement=placement, symmetry=RfDiffusionPointGroupSymmetry("cyclic", 2), library=library
        )
        assert prepared.features.symmetry_residue_index.tolist() == [[0], [1]]
        assert len(prepared.context.structure.system.instance_symbol) == 2

    def test_partial_diffusion_correspondence(self, peptide, library, invalid, scaffold):
        processor = RfDiffusionProcessor()
        invalid(processor.prepare, peptide("ALA", "GLY"), placement=scaffold, partial_diffusion=True, library=library)

    def test_length_bounds(self, peptide, library):
        placement = RfDiffusionPlacements({"chain": {"new": GeneratedSegment("protein", 1, 5)}})
        prepared = RfDiffusionProcessor().prepare_input(
            peptide("ALA"),
            placement=placement,
            length=(3, 3),
            generator=np.random.default_rng(7),
            library=library,
        )
        assert len(ids(prepared.context.structure)) == 3


class TestTemplates:
    def test_labels_cover_copied_rows(self, numbered, library):
        source = numbered("ALA", "GLY", "LYS")
        alignment = TemplateAlignment("P", "A", "A", ((0, 0), (1, 1), (2, 2)))
        template = Template(chain_coordinates(source, ["A"]), (alignment,))
        placement = RfDiffusionPlacements({"chain": {"motif": ExistingSpan("A", 0, 3)}})
        prepared = RfDiffusionProcessor().prepare_input(
            source, placement=placement, templates=(template,), library=library
        )
        assert (prepared.features.secondary_structure[0].argmax(-1) != 3).all()


class TestConnections:
    SG = (AtomSite("A", 0, "SG"), AtomSite("A", 3, "SG"))

    @pytest.mark.parametrize(
        ("stop", "ends", "bonds", "compared", "projected"),
        [
            (4, SG, (), INHERITED_COLUMNS, False),
            (4, SG, (SymbolicBond(*SG),), ("bond_kind", "bond_aromatic_state"), False),
            (4, (AtomSite("A", 1, "O"), AtomSite("A", 2, "O")), (), INHERITED_COLUMNS, False),
            (3, SG, (), INHERITED_COLUMNS, False),
            (3, SG, (), INHERITED_COLUMNS, True),
        ],
        ids=["disulfide", "declared_too", "coordination", "partner_cut", "source_external"],
    )
    def test_copied_connection_kept(self, motif, library, stop, ends, bonds, compared, projected):
        r"""A connection at a copied residue keeps its resolved states, once: an edge between two copied residues of
        two placements, an external connection naming a partner the output does not copy or the source does not
        hold."""
        placement = RfDiffusionPlacements(
            {
                "chain": {
                    "head": ExistingSpan("A", 0, 2),
                    "new": GeneratedSegment("protein", 2, 2),
                    "tail": ExistingSpan("A", 2, stop),
                }
            }
        )
        keep = ~np.isin(np.arange(len(motif.coordinates)), motif.system.atom_index([ends[1]]))
        value = project_atoms(motif, keep)[0] if projected else motif
        prepared = RfDiffusionProcessor().prepare_input(value, placement=placement, bonds=bonds, library=library)
        rows = {0: 0, 1: 1, 2: 4, 3: 5}
        source, target = motif.system, prepared.context.structure.system
        before = np.flatnonzero(np.isin(source.edge_index, source.atom_index(list(ends))).all(axis=0))
        copied = target.atom_index([AtomSite("chain_0", rows[site.component], site.atom) for site in ends])
        prefix = "bond_" if (copied >= 0).all() else "external_"
        if prefix == "bond_":
            after = np.flatnonzero(np.isin(target.edge_index, copied).all(axis=0))
        else:
            after = np.flatnonzero(target.external_atom == copied[0])
            symbols = target.symbols()
            partner = (
                symbols["instance"][target.external_partner_instance[after[0]]],
                int(target.external_partner_component[after[0]]),
                symbols["atom"][target.external_partner_atom[after[0]]],
            )
            assert partner == (ends[1].instance, ends[1].component, ends[1].atom)
        assert [len(before), len(after)] == [1, 1]
        for name in compared:
            assert getattr(target, name.replace("bond_", prefix))[after[0]] == getattr(source, name)[before[0]], name
        types = {
            source.symbols()["connection_type"][source.bond_connection_type[before[0]]],
            target.symbols()["connection_type"][getattr(target, f"{prefix}connection_type")[after[0]]],
        }
        assert len(types) == 1
        for atom in copied[copied >= 0]:
            component = int(np.searchsorted(target.component_atom_ptr, atom, side="right") - 1)
            first, last = target.component_atom_ptr[component : component + 2]
            assert "HG" not in target.text("atom", target.atom_symbol[first:last]).tolist()


class TestContigs:
    def test_author_numbered_span(self, numbered):
        source = numbered("ALA", "GLY")
        placements = from_contig_string("A1-2/10-10", source=source)
        group = placements.placements["chain_1"]
        assert tuple(group.values()) == (ExistingSpan("A", 0, 2), GeneratedSegment("protein", 10, 10))
        assert to_contig_string(placements, source=source) == "A1-2/10-10"

    def test_absent_author_number(self, numbered, invalid):
        invalid(from_contig_string, "A3-4", source=numbered("ALA", "GLY"))
