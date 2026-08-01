"""Native PDB observations. This module has no chemical graph or tensor dependency."""

from __future__ import annotations

import io
from functools import cached_property

import numpy as np

from .records import atom_rows_from_array, reject_duplicate_atom_rows
from .source import ModelTable


class PdbAtomTable(ModelTable):
    def __init__(self, lines: tuple[str, ...]) -> None:
        super().__init__()
        self.lines = lines

    @cached_property
    def model_ids(self) -> tuple[int | None, ...]:
        identifiers = []
        for line in self.lines:
            if line.startswith("MODEL "):
                try:
                    identifiers.append(int(line[10:14]))
                except ValueError:
                    raise ValueError("PDB MODEL requires a serial in columns 11–14") from None
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("PDB MODEL serials must be unique")
        return tuple(identifiers) if identifiers else (None,)

    def _read_model(self, model_id: int | None) -> np.recarray:
        model_index = self.model_ids.index(model_id) + 1
        return decode_pdb_model(self.lines, model_index)


def decode_pdb_model(lines: tuple[str, ...], model_index: int) -> np.recarray:
    """Decode identities with Biotite and float64 observations from original fields."""
    from biotite.structure.io.pdb import PDBFile

    pdb_file = PDBFile.read(io.StringIO("\n".join(lines) + "\n"))
    parser_lines = list(lines)
    for index, line in enumerate(parser_lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if not line[54:60].strip():
            line = line[:54] + "  0.00" + line[60:]
        if not line[60:66].strip():
            line = line[:60] + "  0.00" + line[66:]
        parser_lines[index] = line
    if parser_lines != list(lines):
        pdb_file = PDBFile.read(io.StringIO("\n".join(parser_lines) + "\n"))
    try:
        array = pdb_file.get_structure(
            model=model_index,
            altloc="all",
            extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
            include_bonds=False,
        )
    except IndexError as error:
        raise ValueError(f"PDB does not contain requested model {model_index}") from error
    rows = atom_rows_from_array(array, format_name="pdb")
    source_atom_lines = _pdb_model_atom_lines(lines, model_index)
    if len(source_atom_lines) != len(rows):
        raise ValueError("PDB atom records do not form a consistent model")
    for row, line in zip(rows, source_atom_lines, strict=True):
        row.coordinates = [float(line[start : start + 8]) for start in (30, 38, 46)]
        occupancy = line[54:60].strip()
        b_factor = line[60:66].strip()
        row.occupancy = float(occupancy) if occupancy else None
        row.b_factor = float(b_factor) if b_factor else None
    coordinate_present = np.isfinite(rows.coordinates).all(axis=1)
    rows.coordinate_present = coordinate_present
    rows.coordinates[~coordinate_present] = 0.0
    reject_duplicate_atom_rows(rows)
    return rows


def _pdb_model_atom_lines(pdb_lines: tuple[str, ...], model: int) -> tuple[str, ...]:
    r"""Return the atom records belonging to one one-based PDB model."""

    has_models = any(line.startswith("MODEL ") for line in pdb_lines)
    if not has_models:
        return tuple(line for line in pdb_lines if line.startswith(("ATOM  ", "HETATM"))) if model == 1 else ()

    current_model = 0
    selected = False
    atom_lines: list[str] = []
    for line in pdb_lines:
        if line.startswith("MODEL "):
            current_model += 1
            selected = current_model == model
        elif line.startswith("ENDMDL"):
            selected = False
        elif selected and line.startswith(("ATOM  ", "HETATM")):
            atom_lines.append(line)
    return tuple(atom_lines)
