from __future__ import annotations

import pytest
import torch

from ofoldx.data import AtomRecord, ResidueKey, Structure, coords_to_structure
from ofoldx.processors.structure import StructureProcessor


def _protein_structure() -> Structure:
    residue = ResidueKey("A", 1)
    return Structure(
        (
            AtomRecord("N", "N", "ALA", residue, -1.2, 0.0, 0.0, mol_type="protein"),
            AtomRecord("CA", "C", "ALA", residue, 0.0, 0.0, 0.0, mol_type="protein"),
            AtomRecord("C", "C", "ALA", residue, 1.2, 0.0, 0.0, mol_type="protein"),
        )
    )


def test_base_source_structure_returns_input_atoms_in_order() -> None:
    reference = StructureProcessor().source_structure(_protein_structure())
    assert reference is not None
    assert [atom.name for atom in reference.atoms] == ["N", "CA", "C"]


def test_base_source_structure_reads_a_path(tmp_path) -> None:
    from ofoldx.io import write_structure

    path = tmp_path / "one.pdb"
    write_structure(_protein_structure(), path, format="pdb")
    reference = StructureProcessor().source_structure(path)
    assert reference is not None
    assert len(reference.atoms) == 3


def test_base_source_structure_is_none_for_a_bare_sequence() -> None:
    assert StructureProcessor().source_structure("MKVLA") is None


def test_base_source_structure_matches_featurized_atom_count() -> None:
    # The reference the coordinates overlay onto must have exactly as many atoms as the processor featurizes,
    # so a per-atom predicted-coordinate output lines up one-to-one. Checked against a real family featurizer.
    from ofoldx.models.protenix.processor import ProtenixProcessor

    processor = ProtenixProcessor()
    structure = _protein_structure()
    reference = processor.source_structure(structure)
    assert reference is not None
    assert len(reference.atoms) == processor(structure).atom_mask.shape[1]


def test_esmfold2_source_structure_builds_heavy_atom_layout() -> None:
    from ofoldx.models.esmfold2.processor import ESMFold2Processor

    # ALA(5) + CYS(6) + ASP(8) heavy atoms.
    reference = ESMFold2Processor().source_structure("ACD")
    assert reference is not None
    assert len(reference.atoms) == 5 + 6 + 8
    assert [atom.name for atom in reference.atoms[:5]] == ["N", "CA", "C", "O", "CB"]


def test_esmfold2_source_structure_is_none_for_non_sequence() -> None:
    from ofoldx.models.esmfold2.processor import ESMFold2Processor

    assert ESMFold2Processor().source_structure("12345") is None


def test_esmfold2_padded_coordinates_overlay_onto_real_atoms() -> None:
    from ofoldx.models.esmfold2.processor import ESMFold2Processor

    reference = ESMFold2Processor().source_structure("A")  # ALA -> 5 heavy atoms
    assert reference is not None
    # The model pads its atom axis to a multiple of 32; only the leading 5 rows are real.
    coords = torch.zeros(1, 1, 32, 3)
    coords[0, 0, :5] = torch.arange(5, dtype=torch.float32).unsqueeze(-1).expand(5, 3)
    moved = coords_to_structure(reference, coords, sample=0)
    assert len(moved.atoms) == 5
    assert (moved.atoms[4].x, moved.atoms[4].y, moved.atoms[4].z) == pytest.approx((4.0, 4.0, 4.0))
