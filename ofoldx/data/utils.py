r"""
Structure data utilities -- pure in-memory transforms over the [`Structure`][ofoldx.data.structure.Structure]
objects. Serialization lives in [`ofoldx.io`][ofoldx.io]; the data objects themselves live in
[`ofoldx.data.structure`][ofoldx.data.structure].
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .structure import Structure


def coords_to_structure(reference: Structure, coords: Any, *, sample: int | None = None) -> Structure:
    r"""
    Overlay predicted coordinates onto a reference structure's atoms.

    Puts ``coords`` (``[..., N_atom, 3]`` predicted per-atom coordinates, with any leading sample/batch axes
    reduced) onto ``reference``'s atom records in order, returning a new [`Structure`][ofoldx.data.structure.Structure]
    with the same atom identities (names/residues/chains) and bonds. This is how a structure-prediction output
    becomes a writable structure: the model predicts coordinates for the atom layout the processor built from
    ``reference``, so reference atom ``i`` receives predicted coordinate ``i``. Pair it with
    [`write_structure`][ofoldx.io.write_structure] to serialize the result.

    A model may pad its atom axis to a fixed block size, so ``coords`` can carry *more* rows than the reference
    has atoms; the reference defines the real atoms, so the leading ``len(reference.atoms)`` predicted rows are
    used and any trailing padding rows are dropped. Fewer predicted rows than reference atoms is an error --
    coordinates would be missing, so the model reorders/filters atoms and needs its own structure writer.

    Args:
        reference: The structure whose atom identities and bonds to keep.
        coords: Predicted coordinates ``[..., N_atom, 3]`` (torch tensor, numpy array, or nested list); a padded
            atom axis longer than ``reference`` is truncated to the real atoms.
        sample: Leading sample/batch index to select when ``coords`` has more than two axes; ``None`` selects the
            first of each leading axis.

    Returns:
        ``reference`` with every atom's coordinate replaced by its predicted coordinate.

    Raises:
        ValueError: If there are fewer predicted coordinates than ``reference`` has atoms.

    Examples:
        >>> import torch
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey
        >>> reference = Structure((AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0),))
        >>> moved = coords_to_structure(reference, torch.tensor([[[7.0, 8.0, 9.0]]]))
        >>> (moved.atoms[0].x, moved.atoms[0].y, moved.atoms[0].z)
        (7.0, 8.0, 9.0)
    """

    rows = _coordinate_rows(coords, sample)
    atoms = reference.atoms
    if len(rows) < len(atoms):
        raise ValueError(
            f"predicted atom count {len(rows)} is fewer than the reference structure's {len(atoms)} atoms; "
            "this model reorders/filters atoms, so use its model-specific structure writer"
        )
    # zip stops at the shorter iterable, dropping any trailing padding rows beyond the real atoms.
    moved = tuple(replace(atom, x=float(row[0]), y=float(row[1]), z=float(row[2])) for atom, row in zip(atoms, rows))
    return Structure(moved, bonds=reference.bonds)


def _coordinate_rows(coords: Any, sample: int | None) -> list[Any]:
    r"""Reduce ``coords`` (``[..., N_atom, 3]`` tensor/array/list) to a list of ``[x, y, z]`` rows for one sample."""

    if hasattr(coords, "dim"):  # torch tensor
        reduced = coords.detach().to("cpu")
        while reduced.dim() > 2:
            reduced = reduced[sample] if sample is not None and sample < reduced.shape[0] else reduced[0]
        return reduced.tolist()
    values = coords.tolist() if hasattr(coords, "tolist") else coords
    while values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], list):
        values = values[sample] if sample is not None and sample < len(values) else values[0]
    return values


def index_to_chain_id(index: int) -> str:
    r"""
    Convert a one-based index to a spreadsheet-style chain label (``1 -> A``, ``27 -> AA``).

    The shared chain-label utility used by the family JSON parsers to auto-assign chain ids, matching
    upstream ``data.utils.int_to_letters``.

    Args:
        index: One-based chain index.

    Returns:
        Uppercase-letter chain label.

    Examples:
        >>> index_to_chain_id(1), index_to_chain_id(27), index_to_chain_id(703)
        ('A', 'AA', 'AAA')
    """

    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label
