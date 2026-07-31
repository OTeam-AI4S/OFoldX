from __future__ import annotations

import json

import numpy as np
import torch

from ofoldx.data import (
    MSA,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    MSACollection,
    PolymerSequence,
    Structure,
    Template,
    TemplateAlignment,
    decode_msa,
    decode_msa_collection,
    decode_structure,
    decode_system,
    decode_template,
    encode_msa,
    encode_msa_collection,
    encode_structure,
    encode_system,
    encode_template,
)


def _structure() -> Structure:
    component = ChemicalComponent(
        "MSE",
        (
            ChemicalAtom("CA", "C", formal_charge=0),
            ChemicalAtom("SE", "Se", isotope=80),
        ),
        frozenset(),
        reference_coordinates=((0.0, 0.0, 0.0), None),
    )
    system = MolecularSystem.from_entities(
        {"target": PolymerSequence("protein", (component,))},
        {"A": "target"},
    )
    return Structure(
        system=system,
        coordinates=torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        coordinate_mask=torch.tensor([True, False]),
        occupancy=torch.tensor([0.75, 0.0]),
        occupancy_mask=torch.tensor([True, False]),
        b_factor=torch.tensor([12.0, 0.0]),
        b_factor_mask=torch.tensor([True, False]),
        source_symbols=("7", "B"),
        residue_id=torch.tensor([0]),
        insertion_code=torch.tensor([1]),
        batch_size=[],
    )


class TestTensorCodecs:
    def test_system_round_trip_preserves_every_named_tensor(self) -> None:
        system = _structure().system
        metadata, arrays = encode_system(system)
        encoded_json = json.loads(json.dumps(metadata))
        restored = decode_system(encoded_json, arrays)
        restored_metadata, restored_arrays = encode_system(restored)

        assert restored_metadata == metadata
        assert restored.symbols == system.symbols
        assert arrays.keys() == restored_arrays.keys()
        for name, value in arrays.items():
            assert torch.equal(restored_arrays[name], value)

    def test_structure_round_trip_preserves_graph_and_aligned_evidence(self) -> None:
        structure = _structure()
        metadata, arrays = encode_structure(structure, array_prefix="structure")
        restored = decode_structure(json.loads(json.dumps(metadata)), arrays)
        restored_metadata, restored_arrays = encode_structure(
            restored,
            array_prefix="structure",
        )

        assert restored_metadata == metadata
        assert restored.source_symbols == ("7", "B")
        for name, value in arrays.items():
            assert torch.equal(restored_arrays[name], value)


class TestEvidenceCodecs:
    def test_msa_and_pairing_round_trip(self) -> None:
        msa = MSA(
            "target",
            ("A", "G"),
            np.zeros((2, 1), dtype=np.uint32),
            ("A",),
            np.zeros((1, 1), dtype=np.uint32),
        )
        restored_msa = decode_msa(json.loads(json.dumps(encode_msa(msa))))
        assert restored_msa.unpaired_sequences == ("A", "G")
        np.testing.assert_array_equal(
            restored_msa.unpaired_deletion_counts,
            msa.unpaired_deletion_counts,
        )

        collection = MSACollection(
            (msa,),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[True], [False]], dtype=np.bool_),
            target_instance_ids=("A",),
        )
        restored_collection = decode_msa_collection(json.loads(json.dumps(encode_msa_collection(collection))))
        assert restored_collection.target_instance_ids == ("A",)
        np.testing.assert_array_equal(
            restored_collection.row_pairing,
            collection.row_pairing,
        )

    def test_template_round_trip_preserves_structure_and_alignment(self) -> None:
        template = Template(
            _structure(),
            (TemplateAlignment("query", None, "A", ((0, 0),)),),
        )
        metadata, arrays = encode_template(template, array_prefix="template")
        restored = decode_template(json.loads(json.dumps(metadata)), arrays)

        assert restored.alignments == template.alignments
        assert restored.structure.source_symbols == template.structure.source_symbols
        assert torch.equal(
            restored.structure.coordinates,
            template.structure.coordinates,
        )
