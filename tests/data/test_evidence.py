from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data import (
    MSA,
    MSA_ALPHABET,
    MSA_INDEX,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    MsaCollection,
    PolymerSequence,
    Structure,
    Template,
    TemplateAlignment,
    encode_msa_sequences,
    msa_component_index,
    msa_residue_index,
)
from ofoldx.data.evidence import resolve_msa_bindings


def _structure() -> Structure:
    system = MolecularSystem.from_entities(
        {"target": PolymerSequence.from_one_letter("AGV")},
        {"A": "target"},
    )
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros((atom_count, 3)),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _ambiguous_system() -> MolecularSystem:
    components = tuple(
        ChemicalComponent(component, (ChemicalAtom("CA", "C"),), frozenset())
        for component in ("ASP", "ASN", "ASX", "GLU", "GLN", "GLX", "ILE", "LEU", "XLE", "SEC", "PYL")
    )
    return MolecularSystem.from_entities(
        {"target": PolymerSequence("protein", components, linked=(False,) * 10)},
        {"A": "target"},
    )


class TestMSA:
    def test_canonical_alphabet(self) -> None:
        assert MSA_ALPHABET == (
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
            "X",
            "Z",
            "B",
            "J",
            "U",
            "O",
            "-",
            "RNA_A",
            "RNA_G",
            "RNA_C",
            "RNA_U",
            "RNA_N",
            "DNA_A",
            "DNA_G",
            "DNA_C",
            "DNA_T",
            "DNA_N",
            "OTHER",
        )

    def test_ambiguous_residues(self) -> None:
        assert [msa_residue_index(symbol, "protein") for symbol in "BZJUO"] == [MSA_INDEX[symbol] for symbol in "BZJUO"]
        assert [msa_component_index(component, "protein") for component in ("ASX", "GLX", "XLE", "SEC", "PYL")] == [
            MSA_INDEX[symbol] for symbol in "BZJUO"
        ]
        assert msa_residue_index("U", "protein") != msa_residue_index("U", "rna")

    @pytest.mark.parametrize("polymer_type", ["protein", "rna", "dna"])
    def test_sequence_encoding(self, polymer_type: str) -> None:
        sequences = ("AG-", "?U-")
        expected = np.asarray(
            [[msa_residue_index(symbol, polymer_type) for symbol in sequence] for sequence in sequences]
        )

        assert np.array_equal(encode_msa_sequences(sequences, polymer_type), expected)

    def test_sequence_pools(self) -> None:
        msa = MSA(
            "target",
            ("AGV", "AG-"),
            np.zeros((2, 3), dtype=np.uint32),
            ("AGV",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        assert msa.unpaired_sequences == ("AGV", "AG-")
        assert msa.paired_sequences == ("AGV",)
        assert msa.unpaired_deletion_counts.shape == (2, 3)

    def test_rejects_misaligned_rows(self) -> None:
        with pytest.raises(ValueError, match="query width"):
            MSA(
                "target",
                ("AGV", "AG"),
                np.zeros((2, 3), dtype=np.uint32),
                ("AGV",),
                np.zeros((1, 3), dtype=np.uint32),
            )


class TestMsaCollection:
    def test_pairing_plan(self) -> None:
        msa = MSA(
            "target",
            ("AGV", "AG-"),
            np.zeros((2, 3), dtype=np.uint32),
            ("AGV",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        collection = MsaCollection(
            (msa,),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[True], [False]], dtype=np.bool_),
            target_instance_ids=("A",),
        )
        assert collection.row_pairing.tolist() == [[0], [1]]

    def test_rejects_paired_gap(self) -> None:
        msa = MSA(
            "target",
            ("AGV", "AG-"),
            np.zeros((2, 3), dtype=np.uint32),
            ("AGV",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        with pytest.raises(ValueError, match="gap cell"):
            MsaCollection(
                (msa,),
                np.array([[0], [-1]], dtype=np.int64),
                np.array([[True], [True]], dtype=np.bool_),
            )


class TestResolveMsaBindings:
    @pytest.mark.parametrize("query", ["ABBZZZJJJUO", "BBDZZZJJJUO"])
    def test_mismatch(self, query: str) -> None:
        msa = MSA(
            "target",
            (query,),
            np.zeros((1, len(query)), dtype=np.uint32),
            (query,),
            np.zeros((1, len(query)), dtype=np.uint32),
        )

        with pytest.raises(ValueError, match="target sequence at column"):
            resolve_msa_bindings(msa, _ambiguous_system())

    def test_ambiguous_residues(self) -> None:
        query = "BBBZZZJJJUO"
        msa = MSA(
            "target",
            (query,),
            np.zeros((1, len(query)), dtype=np.uint32),
            (query,),
            np.zeros((1, len(query)), dtype=np.uint32),
        )

        assert resolve_msa_bindings(msa, _ambiguous_system())[0][2] == 0

    def test_pairing_plan(self) -> None:
        msa = MSA(
            "target",
            ("AGV", "AG-"),
            np.zeros((2, 3), dtype=np.uint32),
            ("AGV",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        collection = MsaCollection(
            (msa,),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[True], [False]], dtype=np.bool_),
            target_instance_ids=("A",),
        )

        resolved = resolve_msa_bindings(collection, _structure().system)

        assert resolved[0][1:] == (0, 0)
        assert resolved[0][0].entity_id == "target"
        assert resolved[0][0].unpaired_sequences == ("AGV", "AG-")

    def test_instance_realization(self) -> None:
        declared = PolymerSequence(
            "protein",
            (ChemicalComponent("ALA", (ChemicalAtom("CA", "C"),), frozenset()),),
        )
        alternate = PolymerSequence(
            "protein",
            (ChemicalComponent("GLY", (ChemicalAtom("CA", "C"),), frozenset()),),
        )
        system = MolecularSystem.from_entities(
            {"target": declared},
            {"A": "target", "B": "target"},
            instance_values={"B": alternate},
        )
        glycine = MSA(
            "target",
            ("G",),
            np.zeros((1, 1), dtype=np.uint32),
            ("G",),
            np.zeros((1, 1), dtype=np.uint32),
        )
        alanine = MSA(
            "target",
            ("A",),
            np.zeros((1, 1), dtype=np.uint32),
            ("A",),
            np.zeros((1, 1), dtype=np.uint32),
        )
        targeted = MsaCollection(
            (glycine,),
            np.array([[0]], dtype=np.int64),
            np.array([[True]], dtype=np.bool_),
            target_instance_ids=("B",),
        )

        assert resolve_msa_bindings(targeted, system)[0][1:] == (1, 0)
        with pytest.raises(ValueError, match="target sequence at column 0"):
            resolve_msa_bindings(alanine, system)


class TestTemplateAlignment:
    def test_requires_increasing_indices(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            TemplateAlignment("query", None, "A", ((1, 1), (0, 2)))


class TestTemplate:
    def test_alignment_bounds(self) -> None:
        structure = _structure()
        with pytest.raises(ValueError, match="outside the template instance"):
            Template(
                structure,
                (TemplateAlignment("query", None, "A", ((0, 3),)),),
            )

    def test_rejects_conflicting_alignments(self) -> None:
        structure = _structure()
        with pytest.raises(ValueError, match="same target position"):
            Template(
                structure,
                (
                    TemplateAlignment("query", None, "A", ((0, 0),)),
                    TemplateAlignment("query", "chain", "A", ((0, 2),)),
                ),
            )
