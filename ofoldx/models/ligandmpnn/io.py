r"""
Packed side-chain PDB output for the LigandMPNN family.

OFoldX-native replacement for the reference LigandMPNN ``write_full_PDB``: emit a plain PDB from the
side-chain LigandMPNNPacker's atom14 output ([`pack_side_chains`][ofoldx.models.ligandmpnn.model.pack_side_chains])
without a ProDy dependency. This is LigandMPNN-specific output (it consumes the LigandMPNNPacker's atom14 tensors and
the LigandMPNN sequence alphabet), so it lives with the model family rather than in the shared ``io`` package.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from ...data import residue_constants as rc
from .processor import _MPNN_ALPHABET


def write_packed_pdb(
    path: str | PathLike[str],
    *,
    atom14_coords,
    atom14_mask,
    b_factors,
    residue_index,
    chain_ids: Sequence[str],
    sequence_tokens,
    insertion_codes: Sequence[str] | None = None,
) -> None:
    r"""
    Write packed atom14 side-chain coordinates to a PDB file.

    Only atoms flagged present in ``atom14_mask`` are written, named via the atom14 name table in
    [`residue_constants`][ofoldx.data.residue_constants].

    Args:
        path: Destination PDB path.
        atom14_coords: Per-residue atom14 coordinates with shape ``(L, 14, 3)``.
        atom14_mask: Per-residue atom14 existence mask with shape ``(L, 14)``.
        b_factors: Per-residue atom14 B-factors with shape ``(L, 14)``.
        residue_index: Author residue numbers with shape ``(L,)``.
        chain_ids: Per-residue chain identifiers with length ``L``.
        sequence_tokens: LigandMPNN sequence tokens with shape ``(L,)``.
        insertion_codes: Optional per-residue insertion codes with length ``L``.

    Examples:
        >>> from ofoldx.models.ligandmpnn import pack_side_chains  # doctest: +SKIP
        >>> packed = pack_side_chains(...)  # doctest: +SKIP
        >>> write_packed_pdb(  # doctest: +SKIP
        ...     "packed.pdb",
        ...     atom14_coords=packed["atom14_coords"],
        ...     atom14_mask=packed["atom14_mask"],
        ...     b_factors=packed["b_factors"],
        ...     residue_index=residue_index,
        ...     chain_ids=chain_ids,
        ...     sequence_tokens=sequence_tokens,
        ... )
    """

    import numpy as np

    atom14_coords = np.asarray(atom14_coords, dtype=np.float64)
    atom14_mask = np.asarray(atom14_mask) > 0.5
    b_factors = np.asarray(b_factors, dtype=np.float64)
    residue_index = np.asarray(residue_index).astype(int)
    sequence_tokens = np.asarray(sequence_tokens).astype(int)
    insertion_codes = list(insertion_codes) if insertion_codes is not None else [""] * len(sequence_tokens)

    lines: list[str] = []
    serial = 1
    for position, token in enumerate(sequence_tokens):
        one_letter = _MPNN_ALPHABET[int(token)]
        residue_name = rc.restype_1to3.get(one_letter, "UNK")
        atom_names = rc.restype_name_to_atom14_names[residue_name]
        for atom_index, atom_name in enumerate(atom_names):
            if not atom_name or not atom14_mask[position, atom_index]:
                continue
            x, y, z = atom14_coords[position, atom_index]
            name = _pdb_atom_name(atom_name)
            lines.append(
                f"{'ATOM':<6}{serial:>5} {name:<4}{'':>1}"
                f"{residue_name:>3} {str(chain_ids[position])[:1]:>1}"
                f"{int(residue_index[position]):>4}{str(insertion_codes[position])[:1]:>1}   "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{1.00:>6.2f}{float(b_factors[position, atom_index]):>6.2f}          "
                f"{atom_name[0]:>2}{'':>2}"
            )
            serial += 1
    lines.append("END")
    lines.append("")
    Path(path).write_text("\n".join(line.ljust(80) if line else line for line in lines))


def _pdb_atom_name(name: str) -> str:
    r"""
    Return the PDB-column atom name; 4-character names fill the column, shorter names are left-padded.

    Args:
        name: Atom name.

    Returns:
        The atom name left-padded with one space unless it already fills the 4-character column.

    Examples:
        >>> _pdb_atom_name("CA"), _pdb_atom_name("FE1X")
        (' CA', 'FE1X')
    """

    return name if len(name) == 4 else f" {name}"  # noqa: PLR2004
