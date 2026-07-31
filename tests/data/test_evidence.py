from __future__ import annotations

import numpy as np
import pytest
import torch

from ofoldx.data import (
    MSA,
    MolecularSystem,
    MSACollection,
    PolymerSequence,
    Structure,
    Template,
    TemplateAlignment,
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
        batch_size=[],
    )


class TestMSA:
    def test_keeps_paired_and_unpaired_pools(self) -> None:
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

        with pytest.raises(ValueError, match="query width"):
            MSA(
                "target",
                ("AGV", "AG"),
                np.zeros((2, 3), dtype=np.uint32),
                ("AGV",),
                np.zeros((1, 3), dtype=np.uint32),
            )


class TestMSACollection:
    def test_pairing_plan_resolves_against_the_tensor_graph(self) -> None:
        msa = MSA(
            "target",
            ("AGV", "AG-"),
            np.zeros((2, 3), dtype=np.uint32),
            ("AGV",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        collection = MSACollection(
            (msa,),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[True], [False]], dtype=np.bool_),
            target_instance_ids=("A",),
        )
        resolved = resolve_msa_bindings(collection, _structure().system)

        assert resolved[0][1:] == (0, 0)
        assert resolved[0][0] is collection.msas[0]
        assert collection.row_pairing.tolist() == [[0], [1]]

        with pytest.raises(ValueError, match="gap cell"):
            MSACollection(
                (msa,),
                np.array([[0], [-1]], dtype=np.int64),
                np.array([[True], [True]], dtype=np.bool_),
            )


class TestTemplateAlignment:
    def test_uses_strictly_increasing_component_indices(self) -> None:
        alignment = TemplateAlignment(
            "query",
            None,
            "A",
            ((0, 0), (2, 2)),
        )
        assert alignment.residue_pairs == ((0, 0), (2, 2))

        with pytest.raises(ValueError, match="strictly increasing"):
            TemplateAlignment("query", None, "A", ((1, 1), (0, 2)))


class TestTemplate:
    def test_binds_alignment_to_a_polymer_instance(self) -> None:
        structure = _structure()
        template = Template(
            structure,
            (TemplateAlignment("query", None, "A", ((0, 0), (2, 2))),),
        )
        assert template.structure is structure
        assert template.alignments[0].template_instance_id == "A"

        with pytest.raises(ValueError, match="outside the template instance"):
            Template(
                structure,
                (TemplateAlignment("query", None, "A", ((0, 3),)),),
            )
