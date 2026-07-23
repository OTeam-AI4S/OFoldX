from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import ChemicalAtom, ChemicalComponent, Molecule
from ofoldx.io import read_structure, write_structure
from tests.data.test_codec import _heterogeneous_structure


def test_pdb_element_symbols_are_normalized_before_graph_comparison(tmp_path: Path) -> None:
    path = tmp_path / "elements.pdb"
    path.write_text(
        "HETATM    1 ZN   ZN  Z   1       0.000   0.000   0.000  1.00 10.00          ZN  \n"
        "HETATM    2 CL   CL  C   2       2.000   0.000   0.000  1.00 10.00          CL  \n"
        "END\n",
        encoding="ascii",
    )
    components = {
        "ZN": ChemicalComponent("ZN", atoms=(ChemicalAtom("ZN", "Zn"),), bonds=frozenset()),
        "CL": ChemicalComponent("CL", atoms=(ChemicalAtom("CL", "Cl"),), bonds=frozenset()),
    }

    structure = read_structure(path, components=components)

    assert tuple(structure.system.atom(site).element for site in structure.sites) == ("Zn", "Cl")


def test_pdb_keeps_separate_het_residues_as_separate_single_component_molecules(tmp_path: Path) -> None:
    path = tmp_path / "separate-ligands.pdb"
    path.write_text(
        "HETATM    1  C1  LFT L   1       0.000   0.000   0.000  1.00 10.00           C  \n"
        "HETATM    2  C1  RGT L   2       2.000   0.000   0.000  1.00 10.00           C  \n"
        "END\n",
        encoding="ascii",
    )
    components = {
        "LFT": ChemicalComponent("LFT", atoms=(ChemicalAtom("C1", "C"),), bonds=frozenset()),
        "RGT": ChemicalComponent("RGT", atoms=(ChemicalAtom("C1", "C"),), bonds=frozenset()),
    }

    structure = read_structure(path, components=components)

    assert len(structure.system.entities) == 2
    assert len(structure.system.instances) == 2
    assert all(
        type(entity) is Molecule and len(entity.components) == 1 for entity in structure.system.entities.values()
    )
    assert {site.component_index for site in structure.sites} == {0}


@pytest.mark.xfail(
    reason="reopened §3.5: the LossReport contract (dropped/rewritten field vocabulary and the "
    "removed format field) is the reopened §3.5 lossy-write reporting decision",
    strict=True,
)
def test_pdb_fails_by_default_and_returns_exact_loss_report_when_acknowledged(tmp_path: Path) -> None:
    structure = _heterogeneous_structure()
    path = tmp_path / "lossy.pdb"

    with pytest.raises(ValueError, match="allow_lossy=True"):
        write_structure(structure, path)
    assert not path.exists()

    report = write_structure(structure, path, allow_lossy=True)
    assert report is not None
    # §3.5.1: LossReport.format was removed; the report exposes only dropped/rewritten field
    # vocabularies, whose exact contents stay the reopened §3.5 lossy-write reporting decision.
    assert "system.nonpolymer_graph" in report.dropped_fields
    assert "system.bond_origin" in report.rewritten_fields
    assert path.read_text(encoding="ascii").endswith("END\n")


def test_source_duplicate_atoms_fail_before_canonical_frozensets(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.pdb"
    line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  \n"
    path.write_text(line + line + "END\n", encoding="ascii")
    with pytest.raises(ValueError, match="duplicate source atom declaration"):
        read_structure(path)


def test_pdb_assembly_materializes_distinct_instances_of_one_entity(tmp_path: Path) -> None:
    path = tmp_path / "assembly.pdb"
    path.write_text(
        """REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 APPLY THE FOLLOWING TO CHAINS: A
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2  1.000000  0.000000  0.000000       10.00000
REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
END
""",
        encoding="ascii",
    )

    structure = read_structure(path, assembly_id="1")

    assert tuple(structure.system.instances) == ("A#0", "A#1")
    assert len(structure.system.entities) == 1
    assert structure.coordinates[:, 0].tolist() == [0.0, 10.0]
