from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx.data import MolecularSystem, Molecule, Structure, get_component
from ofoldx.io import read_structure, write_structure


def _structure() -> Structure:
    component = get_component("ALA").without_hydrogens()
    system = MolecularSystem.from_entities(
        {"alanine": Molecule((component,))},
        {"A": "alanine"},
    )
    atom_count = system.atom_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros((atom_count, 3), dtype=torch.float32),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.ones(atom_count, dtype=torch.float32),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count, dtype=torch.float32),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((1,), -1, dtype=torch.int64),
        insertion_code=torch.full((1,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _write_resolution_source(
    path: Path,
    fields: tuple[tuple[str, str, str], ...],
) -> Path:
    from biotite.structure.io import pdbx

    write_structure(_structure(), path)
    cif_file = pdbx.CIFFile.read(str(path))
    for category_name, column_name, value in fields:
        cif_file.block[category_name] = pdbx.CIFCategory(
            {column_name: pdbx.CIFColumn([value])}
        )
    cif_file.write(str(path))
    return path


@pytest.mark.parametrize(
    ("category_name", "column_name"),
    (
        ("refine", "ls_d_res_high"),
        ("em_3d_reconstruction", "resolution"),
        ("reflns", "d_resolution_high"),
    ),
)
def test_read_structure_transports_standard_mmcif_resolution_fields(
    tmp_path: Path,
    category_name: str,
    column_name: str,
) -> None:
    path = _write_resolution_source(
        tmp_path / f"{category_name}.cif",
        ((category_name, column_name, "1.75"),),
    )

    structure = read_structure(path)

    assert structure.resolution is not None
    assert structure.resolution.shape == ()
    assert structure.resolution.item() == pytest.approx(1.75)
    assert structure.resolution_is_known is not None
    assert structure.resolution_is_known.shape == ()
    assert structure.resolution_is_known.item()


@pytest.mark.parametrize("invalid_value", (None, ".", "?", "0", "-1", "nan", "inf", "invalid"))
def test_read_structure_marks_absent_or_invalid_mmcif_resolution_unknown(
    tmp_path: Path,
    invalid_value: str | None,
) -> None:
    fields = () if invalid_value is None else (("refine", "ls_d_res_high", invalid_value),)
    path = _write_resolution_source(tmp_path / "unknown.cif", fields)

    structure = read_structure(path)

    assert structure.resolution is not None
    assert structure.resolution.item() == -1.0
    assert structure.resolution_is_known is not None
    assert not structure.resolution_is_known.item()


def test_read_structure_uses_first_valid_standard_resolution_field(tmp_path: Path) -> None:
    path = _write_resolution_source(
        tmp_path / "priority.cif",
        (
            ("refine", "ls_d_res_high", "?"),
            ("em_3d_reconstruction", "resolution", "2.6"),
            ("reflns", "d_resolution_high", "1.2"),
        ),
    )

    structure = read_structure(path)

    assert structure.resolution is not None
    assert structure.resolution.item() == pytest.approx(2.6)
    assert structure.resolution_is_known is not None
    assert structure.resolution_is_known.item()
