from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ofoldx.data import (
    ENTITY_PROTEIN,
    MSA,
    MolecularSystem,
    MsaCollection,
    PolymerSequence,
    Structure,
    Template,
    TemplateAlignment,
    save_msa_collection,
    save_structure,
    save_templates,
)
from ofoldx.pipelines.input import (
    resolve_msa_input,
    resolve_polymer_input,
    resolve_structure_input,
    resolve_template_input,
)


class TestPolymerInput:
    @pytest.mark.parametrize(
        "input",
        (
            "ACD",
            PolymerSequence.from_one_letter("ACD"),
            (PolymerSequence.from_one_letter("ACD"),),
        ),
    )
    def test_anonymous(
        self,
        input: str | PolymerSequence | tuple[PolymerSequence, ...],
    ) -> None:
        named = resolve_polymer_input(input)

        assert isinstance(named, tuple)
        assert named[0][0] == "chain_1"
        assert isinstance(named[0][1], PolymerSequence)

    def test_fasta(self, tmp_path: Path) -> None:
        path = tmp_path / "input.faa"
        path.write_text(">first\nMK\n>second\nWQ\n", encoding="ascii")

        named = resolve_polymer_input(path)

        assert isinstance(named, tuple)
        assert [name for name, _sequence in named] == ["first", "second"]

    def test_fna(self, tmp_path: Path) -> None:
        path = tmp_path / "input.fna"
        path.write_text(">chain\nACG\n", encoding="ascii")

        with pytest.raises(ValueError, match="does not distinguish DNA from RNA"):
            resolve_polymer_input(path)

    def test_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "input.pdb"
        path.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C\nEND\n",
            encoding="ascii",
        )

        assert isinstance(resolve_polymer_input(path), Structure)
        assert isinstance(resolve_structure_input(path), Structure)

    def test_native_structure(self, tmp_path: Path) -> None:
        path = tmp_path / "input.structure.safetensors"
        save_structure(Structure.from_sequence("ACD"), path)

        assert isinstance(resolve_polymer_input(path), Structure)
        assert isinstance(resolve_structure_input(path), Structure)


class TestMsaInput:
    def test_binding(self, tmp_path: Path) -> None:
        path = tmp_path / "input.a3m"
        path.write_text(">query\nACD\n>first\nAGD\n>second\nADD\n", encoding="ascii")
        system = MolecularSystem.from_entities(
            {"target": PolymerSequence.from_one_letter("ACD")},
            {"A": "target"},
        )

        msa = resolve_msa_input(path, system, entity_types={ENTITY_PROTEIN}, max_rows=2)

        assert isinstance(msa, MSA)
        assert msa.entity_id == "target"
        assert msa.unpaired_sequences == ("ACD", "AGD")

    def test_ambiguity(self, tmp_path: Path) -> None:
        path = tmp_path / "input.a3m"
        path.write_text(">query\nACD\n", encoding="ascii")
        system = MolecularSystem.from_entities(
            {
                "first": PolymerSequence.from_one_letter("ACD"),
                "second": PolymerSequence.from_one_letter("EFG"),
            },
            {"A": "first", "B": "second"},
        )

        with pytest.raises(ValueError, match="explicit entity_id"):
            resolve_msa_input(path, system, entity_types={ENTITY_PROTEIN})

    def test_native_collection(self, tmp_path: Path) -> None:
        msa = MSA(
            "target",
            ("ACD", "AGD"),
            np.zeros((2, 3), dtype=np.uint32),
            ("ACD",),
            np.zeros((1, 3), dtype=np.uint32),
        )
        collection = MsaCollection(
            (msa,),
            np.array([[0], [1]], dtype=np.int64),
            np.array([[True], [False]], dtype=np.bool_),
        )
        path = tmp_path / "input.msa.safetensors"
        save_msa_collection(collection, path)

        restored = resolve_msa_input(path, None, entity_types={ENTITY_PROTEIN})

        assert isinstance(restored, MsaCollection)
        assert restored.msas[0].unpaired_sequences == msa.unpaired_sequences


class TestTemplateInput:
    def test_native_artifact(self, tmp_path: Path) -> None:
        structure = Structure.from_sequence("ACD")
        template = Template.from_structure(
            structure,
            (TemplateAlignment("target", None, structure.system.instance_id(0), ((0, 0),)),),
        )
        path = tmp_path / "input.templates.safetensors"
        save_templates((template,), path)

        assert resolve_template_input(path)[0].alignments == template.alignments
