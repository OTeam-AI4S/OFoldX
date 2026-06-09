r"""
PyTorch datasets for OFoldX model inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar

from torch.utils.data import Dataset

from ..io import read_structure
from .structure import Structure

StructureInput: TypeAlias = Structure | str | PathLike[str]
OutputT = TypeVar("OutputT")


class StructureDataset(Dataset[OutputT], Generic[OutputT]):
    r"""
    Dataset that reads structure files and returns processor outputs.

    Args:
        root: Directory containing structure files.
        format: Structure format filter. ``"auto"`` accepts PDB and mmCIF suffixes.
        max_tokens: Optional token crop length applied before processing.
        processor: Callable that converts a parsed structure into the model-local input object.

    Raises:
        NotADirectoryError: If ``root`` is not an existing directory.
        ValueError: If no structure files matching ``format`` are found.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        >>> dataset = StructureDataset("structures/", processor=processor, max_tokens=384)  # doctest: +SKIP
        >>> dataset[0]  # doctest: +SKIP
        {'token_index': ..., 'restype': ...}
    """

    def __init__(
        self,
        root: str | PathLike[str],
        *,
        format: str = "auto",
        max_tokens: int | None = None,
        processor: Callable[[Structure], OutputT],
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise NotADirectoryError(f"structure dataset root is not a directory: {self.root}")
        self.paths = _structure_files(self.root, format=format)
        self.format = format
        self.processor = processor
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> OutputT:
        structure = read_structure(self.paths[index], format=self.format)
        if self.max_tokens is not None:
            structure = crop_structure(structure, max_tokens=self.max_tokens)
        return self.processor(structure)

    def __getitems__(self, indices: list[int]) -> list[OutputT]:
        return [self[index] for index in indices]


def crop_structure(structure: Structure, *, max_tokens: int) -> Structure:
    r"""
    Return atoms covered by the first ``max_tokens`` model tokens.

    Args:
        structure: Parsed structure to crop.
        max_tokens: Positive number of model tokens to keep.

    Returns:
        Cropped structure with bonds remapped to the retained atom indices.

    Raises:
        ValueError: If ``max_tokens`` is not positive.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> atoms = tuple(
        ...     AtomRecord(name, name[0], "ALA", ResidueKey("A", num), float(num), 0.0, 0.0, mol_type="protein")
        ...     for num in (1, 2, 3)
        ...     for name in ("N", "CA", "C", "O")
        ... )
        >>> len(crop_structure(Structure(atoms), max_tokens=2).atoms)
        8
        >>> crop_structure(Structure(atoms), max_tokens=0)
        Traceback (most recent call last):
            ...
        ValueError: max_tokens must be positive, got 0
    """

    from ..processors import tokenize_structure

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    tokens = tokenize_structure(structure)
    if len(tokens) <= max_tokens:
        return structure
    kept_atoms = {atom_index for token in tokens[:max_tokens] for atom_index in token.atom_indices}
    index_map = {old_index: new_index for new_index, old_index in enumerate(sorted(kept_atoms))}
    atoms = tuple(atom for atom_index, atom in enumerate(structure.atoms) if atom_index in kept_atoms)
    bonds = tuple((index_map[i], index_map[j]) for i, j in structure.bonds if i in kept_atoms and j in kept_atoms)
    return Structure(atoms, bonds=bonds)


def _structure_files(root: Path, *, format: str) -> tuple[Path, ...]:
    r"""
    Return the sorted structure files under ``root`` matching ``format``.

    Args:
        root: Directory to scan (non-recursively).
        format: Structure format filter passed to `_structure_suffixes`.

    Returns:
        Sorted paths of the matching structure files.

    Raises:
        ValueError: If no matching structure files are found.
    """

    suffixes = _structure_suffixes(format)
    paths = tuple(sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes))
    if not paths:
        raise ValueError(f"no structure files found under {root}")
    return paths


def _structure_suffixes(format: str) -> set[str]:
    r"""
    Map a structure-format filter to the accepted lowercase file suffixes.

    Args:
        format: One of ``"auto"``, ``"pdb"``, ``"cif"``, or ``"mmcif"`` (case-insensitive).

    Returns:
        Accepted lowercase file suffixes (including the leading dot).

    Raises:
        ValueError: If ``format`` is not a recognized structure format.
    """

    normalized = format.lower()
    if normalized == "auto":
        return {".pdb", ".cif", ".mmcif"}
    if normalized == "pdb":
        return {".pdb"}
    if normalized in {"cif", "mmcif"}:
        return {".cif", ".mmcif"}
    raise ValueError(f"unsupported structure format: {format!r}")
