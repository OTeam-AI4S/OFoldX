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

"""Behavior of ofoldx.models.preparation."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from ofoldx.chemistry.construction import PolymerSequence, Recipe, build_system
from ofoldx.data.molecular import EndpointState, MolecularSystem, MonomerType
from ofoldx.data.structure import CoordinateEvidence, Structure
from ofoldx.data.tokenization import TOKEN_INDEX, TOKENS
from ofoldx.errors import ContradictionError, InsufficientInformationError, InvalidRequestError
from ofoldx.models.alphafold3 import AlphaFold3Processor
from ofoldx.models.layout import LayoutTable, slot_map, token_layout
from ofoldx.models.preparation import (
    ConformerSource,
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    decode_coordinates,
    gate,
    missing_reference,
    reference_conformers,
)
from ofoldx.models.protenix import ProtenixProcessor
from ofoldx.msa import MSA, IndexedMsaLoader

PROTEIN = MonomerType.PROTEIN
TABLE = LayoutTable(
    family="test",
    monomer_types={PROTEIN: 0},
    residue_slots=None,
    monomer_slots={PROTEIN: ("N", "CA", "C", "O")},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)


def prepared(structure: Structure, scope: RegenerationScope) -> PreparedInput[None]:
    slots = slot_map(structure.system, token_layout(structure.system, TABLE), TABLE)
    record = PreparationRecord(gate(structure, slots, GatingPolicy()), np.zeros(0, dtype=np.uint8))
    return PreparedInput(None, PreparationContext(structure, slots, scope, record))


def rebuilt(structure: Structure, **cells) -> Structure:
    arrays = structure.system.build()
    for column, (index, value) in cells.items():
        arrays.columns[column][index] = value
    return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())


class TestMsaPreparation:
    @pytest.fixture(params=[AlphaFold3Processor, ProtenixProcessor], ids=["af3", "protenix"])
    def processor(self, request):
        return request.param()

    @pytest.fixture
    def structure(self, library, placed):
        def build(length):
            system, _, _ = build_system(
                Recipe(
                    {
                        "P": PolymerSequence("polypeptide(L)", ("ALA",) * 5),
                        "Q": PolymerSequence("polypeptide(L)", ("GLY",) * length),
                    },
                    {"A": "P", "B": "Q"},
                ),
                library,
            )
            return placed(system)

        return build

    @pytest.mark.parametrize(
        "length,cropped,read",
        [(2, False, False), (5, True, False), (5, False, True)],
        ids=["short-chain", "cropped-chain", "fed-chain"],
    )
    def test_lazy_pools(self, structure, tmp_path, processor, length, cropped, read):
        target = structure(length)
        directory = tmp_path / "active"
        directory.mkdir()
        (directory / "pairing.a3m").write_text(">query\nAAAAA\n")
        (directory / "non_pairing.a3m").write_text(">query\nAAAAA\n>hit\nGAAAA\n")
        pools = IndexedMsaLoader({"AAAAA": "active", "G" * length: "missing"}, tmp_path)(target)
        cropper = (lambda structure, layout, **kwargs: layout.token_component < 5) if cropped else None
        with pytest.raises(FileNotFoundError) if read else nullcontext():
            prepared = processor.prepare_input(target, msa=pools, cropper=cropper, generator=np.random.default_rng(0))
            assert prepared.features.msa_tokens[0, 1, 0] == TOKEN_INDEX["G"]
            profile = prepared.features.token_features[0, 0, 384 + len(TOKENS) : 384 + 2 * len(TOKENS)]
            assert profile[TOKEN_INDEX["A"]] == profile[TOKEN_INDEX["G"]] == 0.5

    @pytest.mark.parametrize("length,paired", [(2, False), (4, False), (5, True)])
    def test_pairing_admission(self, structure, processor, length, paired):
        target = structure(length)
        pools = {}
        for name, entity, query, unpaired, paired_row in (
            ("A", "P", "AAAAA", "DDDDD", "CCCCC"),
            ("B", "Q", "G" * length, "V" * length, "T" * length),
        ):
            counts = np.zeros((2, len(query)), dtype=np.uint32)
            counts[1, 0] = 3
            pools[name] = MSA(
                entity,
                (query, unpaired),
                counts,
                (query, paired_row),
                counts.copy(),
                ("query", "UniRef100_A_9606"),
            )
        reference = processor.prepare_input(target, msa={"A": pools["A"]}).features
        actual = processor.prepare_input(target, msa=pools).features
        rows = ["AAAAA", "CCCCC", "DDDDD"] if paired else ["AAAAA", "DDDDD"]
        assert actual.msa_tokens[0, :, :5].tolist() == [[TOKEN_INDEX[letter] for letter in row] for row in rows]
        profile = slice(384 + len(TOKENS), 384 + 2 * len(TOKENS) + 1)
        assert actual.token_features[0, :5, profile].tolist() == reference.token_features[0, :5, profile].tolist()
        if not paired:
            assert actual.msa_tokens.tolist() == reference.msa_tokens.tolist()
            assert actual.msa_deletion_counts.tolist() == reference.msa_deletion_counts.tolist()


class TestRegenerationScope:
    @pytest.mark.parametrize(
        "make,count,frame",
        [(RegenerationScope.none, 0, "source"), (RegenerationScope.all, None, "new")],
    )
    def test_whole_rules(self, peptide, make, count, frame):
        system = peptide("ALA").system
        scope = make(system)
        assert int(scope.atoms.sum()) == (len(system.atom_symbol) if count is None else count)
        assert scope.frame == frame

    def test_components_keep_atoms(self, peptide):
        system = peptide("ALA", "GLY").system
        keep = np.zeros(len(system.atom_symbol), dtype=bool)
        keep[system.component_atom_ptr[1]] = True
        scope = RegenerationScope.components(system, np.array([1]), keep_atoms=keep)
        assert int(scope.atoms.sum()) == int(np.diff(system.component_atom_ptr)[1]) - 1


class TestDecode:
    def test_unpredicted_atoms_invalid(self, peptide):
        structure = peptide("ALA")
        task = prepared(structure, RegenerationScope.all(structure.system))
        decoded, report = decode_coordinates(task, np.ones((len(task.context.slots.atom), 3)))
        assert (decoded.coordinate_evidence[task.context.slots.atom] == CoordinateEvidence.PREDICTED).all()
        assert len(report.invalidated_unwritten) == len(structure.coordinates) - len(task.context.slots.atom)
        assert report.frame_withdrawn

    def test_input_coordinates_ignored(self, peptide):
        structure = peptide("ALA", "GLY")
        moved = structure.build()
        moved.columns["coordinates"] += 10.0
        output = np.arange(8 * 3, dtype=np.float64).reshape(8, 3)
        first = decode_coordinates(prepared(structure, RegenerationScope.all(structure.system)), output)[0]
        other = Structure.from_arrays(structure.system, moved)
        second = decode_coordinates(prepared(other, RegenerationScope.all(other.system)), output)[0]
        assert first.digest() == second.digest()

    def test_outside_scope_refused(self, peptide):
        structure = peptide("ALA")
        task = prepared(structure, RegenerationScope.none(structure.system))
        with pytest.raises(InvalidRequestError, match="scope_violation"):
            decode_coordinates(task, np.zeros((len(task.context.slots.atom), 3)))


class TestGate:
    def test_conflict_outside_slots(self, peptide):
        structure = peptide("ALA", ligands=("ACT",))
        ligand_atom = int(structure.system.component_atom_ptr[1])
        conflicted = rebuilt(
            structure, atom_endpoint_state=(ligand_atom, EndpointState.CONFLICT), component_topology_complete=(1, False)
        )
        task = prepared(conflicted, RegenerationScope.none(conflicted.system))
        assert task.context.record.records.counts() == {"prepare_accepted_incomplete": 1}

    def test_conflict_bonded_to_slot(self, peptide):
        structure = peptide("ALA")
        beta = int(np.flatnonzero(structure.system.text("atom", structure.system.atom_symbol) == "CB")[0])
        conflicted = rebuilt(
            structure, atom_endpoint_state=(beta, EndpointState.CONFLICT), component_topology_complete=(0, False)
        )
        with pytest.raises(ContradictionError, match="prepare_source_conflict"):
            prepared(conflicted, RegenerationScope.none(conflicted.system))

    def test_incomplete_outside_slots(self, peptide):
        structure = peptide("ALA", ligands=("ACT",))
        open_ligand = rebuilt(structure, component_topology_complete=(1, False))
        task = prepared(open_ligand, RegenerationScope.none(open_ligand.system))
        assert task.context.record.records.counts() == {"prepare_accepted_incomplete": 1}

    def test_unknown_required_state(self, peptide):
        structure = peptide("ALA")
        unknown = rebuilt(structure, formal_charge_state=(1, 8), formal_charge=(1, 0))
        slots = slot_map(unknown.system, token_layout(unknown.system, TABLE), TABLE)
        with pytest.raises(InsufficientInformationError, match="prepare_unknown_feature"):
            gate(unknown, slots, GatingPolicy(required_states=frozenset({"formal_charge"})))


class TestReferenceConformers:
    def test_pool_coordinates(self, peptide):
        structure = peptide("ALA")
        slots = prepared(structure, RegenerationScope.none(structure.system)).context.slots
        coordinates, source, _ = reference_conformers(structure, slots, embed=np.zeros(1, dtype=bool), seed=0)
        pool, _ = structure.system.reference_coordinates()
        np.testing.assert_array_equal(coordinates, pool[slots.atom])
        assert (source == ConformerSource.REFERENCE).all()

    def test_unconsumed_atom_ignored(self, peptide):
        structure = peptide("ALA")
        beta = int(np.flatnonzero(structure.system.text("atom", structure.system.atom_symbol) == "CB")[0])
        extra = rebuilt(structure, atom_reference_row=(beta, -1))
        slots = prepared(extra, RegenerationScope.none(extra.system)).context.slots
        _, source, _ = reference_conformers(extra, slots, embed=np.zeros(1, dtype=bool), seed=0)
        assert (source == ConformerSource.REFERENCE).all()

    def test_missing_not_embedded(self, peptide):
        structure = peptide("ALA")
        missing = rebuilt(structure, atom_reference_row=(slice(None), -1))
        assert missing_reference(missing.system).tolist() == [True]
        slots = prepared(missing, RegenerationScope.none(missing.system)).context.slots
        with pytest.raises(InsufficientInformationError, match="prepare_reference_missing"):
            reference_conformers(missing, slots, embed=np.zeros(1, dtype=bool), seed=0)
