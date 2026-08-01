from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import ENTITY_PROTEIN, MSA, MolecularSystem, PolymerSequence, Structure
from ofoldx.pipelines.input import resolve_msa_input, resolve_polymer_input


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
