from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from ofoldx.data import (
    MSA,
    MSA_ALPHABET,
    MSA_INDEX,
    ChainCoordinates,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    MsaCollection,
    PolymerCoordinates,
    PolymerSequence,
    Structure,
    Template,
    TemplateAlignment,
    encode_msa_sequences,
    load_msa_collection,
    load_templates,
    msa_component_index,
    msa_residue_index,
    save_msa_collection,
    save_templates,
)
from ofoldx.data.evidence import resolve_msa_bindings, template_atoms


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

    def test_artifact(self, tmp_path: Path) -> None:
        left = MSA(
            "left",
            ("AG", "A-", "-G"),
            np.array([[0, 0], [0, 2], [1, 0]], dtype=np.uint32),
            ("AG", "A-"),
            np.array([[0, 0], [0, 2]], dtype=np.uint32),
        )
        right = MSA(
            "right",
            ("C", "-"),
            np.array([[0], [3]], dtype=np.uint32),
            ("C",),
            np.array([[0]], dtype=np.uint32),
        )
        collection = MsaCollection(
            (left, right),
            np.array([[0, 0], [1, -1], [-1, 1]], dtype=np.int64),
            np.array([[True, True], [True, False], [False, False]], dtype=np.bool_),
            target_instance_ids=("A", "B"),
        )
        path = tmp_path / "example.msa.safetensors"

        save_msa_collection(collection, path)
        restored = load_msa_collection(path)
        with safe_open(str(path), framework="pt") as artifact:
            assert artifact.metadata().keys() == {"format", "entity_ids", "target_instance_ids"}
            assert artifact.metadata()["format"] == "ofoldx.msa"

        assert restored.target_instance_ids == ("A", "B")
        assert tuple(msa.entity_id for msa in restored.msas) == ("left", "right")
        for actual, expected in zip(restored.msas, collection.msas, strict=True):
            assert actual.unpaired_sequences == expected.unpaired_sequences
            assert actual.paired_sequences == expected.paired_sequences
            np.testing.assert_array_equal(actual.unpaired_deletion_counts, expected.unpaired_deletion_counts)
            np.testing.assert_array_equal(actual.paired_deletion_counts, expected.paired_deletion_counts)
        np.testing.assert_array_equal(restored.row_pairing, collection.row_pairing)
        np.testing.assert_array_equal(restored.row_is_paired, collection.row_is_paired)

    def test_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "example.safetensors"
        msa = MSA(
            "target",
            ("A",),
            np.zeros((1, 1), dtype=np.uint32),
            ("A",),
            np.zeros((1, 1), dtype=np.uint32),
        )
        collection = MsaCollection(
            (msa,),
            np.zeros((1, 1), dtype=np.int64),
            np.ones((1, 1), dtype=np.bool_),
        )

        with pytest.raises(ValueError, match=r"\.msa\.safetensors"):
            save_msa_collection(collection, path)
        with pytest.raises(ValueError, match=r"\.msa\.safetensors"):
            load_msa_collection(path)


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

    def test_owns_residue_pairs(self) -> None:
        pairs = [[0, np.int64(1)], [1, 2]]
        alignment = TemplateAlignment("query", None, "A", pairs)
        pairs[0][1] = 2
        assert alignment.residue_pairs == ((0, 1), (1, 2))
        for pair in ((0, 1.5), (True, 0), (0, False), (np.bool_(True), 0), (0, np.bool_(False))):
            with pytest.raises(TypeError):
                TemplateAlignment("query", None, "A", (pair,))


class TestTemplate:
    def test_alignment_bounds(self) -> None:
        structure = _structure()
        with pytest.raises(ValueError, match="outside the template instance"):
            Template.from_structure(
                structure,
                (TemplateAlignment("query", None, "A", ((0, 3),)),),
            )

    def test_rejects_conflicting_alignments(self) -> None:
        structure = _structure()
        with pytest.raises(ValueError, match="same target position"):
            Template.from_structure(
                structure,
                (
                    TemplateAlignment("query", None, "A", ((0, 0),)),
                    TemplateAlignment("query", "chain", "A", ((0, 2),)),
                ),
            )

    def test_snapshot_lifecycle(self, tmp_path: Path) -> None:
        source = _structure()
        alignments = [TemplateAlignment("query", None, "A", ((0, 0),))]
        template = Template.from_structure(source, alignments)
        coordinates = template.coordinates
        alignments.clear()
        source.coordinates[:] = 7
        source.coordinate_mask[:] = False
        assert template.coordinates is coordinates
        assert template.coordinates.chains["A"].coordinate_mask.all()
        assert not template.coordinates.chains["A"].coordinates.any()
        assert len(template.alignments) == 1
        with pytest.raises(FrozenInstanceError):
            template.coordinates = PolymerCoordinates({})
        path = tmp_path / "snapshot.templates.safetensors"
        save_templates((template,), path)
        restored = load_templates(path)[0]
        assert restored.coordinates.chains["A"].coordinate_mask.all()
        assert not restored.coordinates.chains["A"].coordinates.any()
        replay = pickle.loads(pickle.dumps(template))
        assert replay.alignments == template.alignments
        np.testing.assert_array_equal(replay.coordinates.chains["A"].coordinates, coordinates.chains["A"].coordinates)

    def test_artifact(self, tmp_path: Path) -> None:
        coordinates = PolymerCoordinates(
            {
                "A": ChainCoordinates(
                    polymer_type="protein",
                    component_ids=("ALA", "MSE", "MOD"),
                    atom_names=("CA", "CA", "CA"),
                    component_index=np.array([0, 1, 2]),
                    coordinates=np.array([[1.123456789, 2, 3], [np.nan, 0, 0], [4, 5, 6]], dtype=np.float64),
                    coordinate_mask=np.array([True, False, True]),
                    parent_component_ids=((), ("MET",), ("ALA", "GLY")),
                ),
                "R": ChainCoordinates(
                    polymer_type="rna",
                    component_ids=("A",),
                    atom_names=("P",),
                    component_index=np.array([0]),
                    coordinates=np.array([[7, 8, 9]], dtype=np.float32),
                    coordinate_mask=np.array([True]),
                ),
            }
        )
        templates = (
            Template(
                coordinates,
                (TemplateAlignment("query", None, "A", ((0, 0), (1, 1))),),
            ),
            Template(
                coordinates,
                (TemplateAlignment("other", "target-B", "R", ((2, 0),)),),
            ),
        )
        path = tmp_path / "example.templates.safetensors"

        save_templates(templates, path)
        restored = load_templates(path)
        with safe_open(str(path), framework="pt") as artifact:
            assert artifact.metadata().keys() == {
                "format",
                "template_coordinate_chains",
                "alignment_target_entity_ids",
                "alignment_target_instance_ids",
                "alignment_template_instance_ids",
            }
            assert artifact.metadata()["format"] == "ofoldx.templates"
        path.unlink()

        assert tuple(template.alignments for template in restored) == tuple(
            template.alignments for template in templates
        )
        for actual, expected in zip(restored, templates, strict=True):
            assert tuple(actual.coordinates.chains) == tuple(expected.coordinates.chains)
            for identifier, expected_chain in expected.coordinates.chains.items():
                chain = actual.coordinates.chains[identifier]
                assert chain.polymer_type == expected_chain.polymer_type
                assert chain.component_ids == expected_chain.component_ids
                assert chain.atom_names == expected_chain.atom_names
                assert chain.parent_component_ids == expected_chain.parent_component_ids
                assert chain.coordinates.dtype == expected_chain.coordinates.dtype
                for field in ("component_index", "coordinates", "coordinate_mask"):
                    np.testing.assert_array_equal(getattr(chain, field), getattr(expected_chain, field))
                with pytest.raises(ValueError):
                    chain.coordinates.setflags(write=True)

    def test_empty_artifact(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.templates.safetensors"
        save_templates((), path)
        assert load_templates(path) == ()

    def test_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "example.safetensors"
        template = Template.from_structure(
            _structure(),
            (TemplateAlignment("query", None, "A", ((0, 0),)),),
        )

        with pytest.raises(ValueError, match=r"\.templates\.safetensors"):
            save_templates((template,), path)
        with pytest.raises(ValueError, match=r"\.templates\.safetensors"):
            load_templates(path)


def test_template_atoms_own_storage():
    source = _structure().double()
    source.coordinates[:] = 1.123456789
    chain = PolymerCoordinates.from_structure(source).chains["A"]
    coordinates, mask = template_atoms(chain, 0, ("CA", None))
    assert coordinates.dtype == torch.float64 and mask.dtype == torch.float32
    assert coordinates[0, 0].item() == 1.123456789
    assert mask.tolist() == [1.0, 0.0]
    coordinates[:] = 0
    mask[:] = 0
    assert (chain.coordinates == 1.123456789).all() and chain.coordinate_mask.all()
