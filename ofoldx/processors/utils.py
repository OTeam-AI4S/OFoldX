r"""
Shared structure-processing utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import Any, Protocol, TypeAlias

import torch
from torch import Tensor

from ..data.structure import ResidueKey, Structure
from ..io import read_structure

StructureInput: TypeAlias = Structure | str | PathLike[str]


class ProcessorOutput(Protocol):
    r"""
    Structural typing for the tensor bundle a structure processor emits and a model consumes.

    Any object whose tensors move as a group under a ``to`` call (the model-local dataclasses) satisfies
    this protocol; it exists so [`AutoProcessor`][ofoldx.processors.auto.AutoProcessor] can be typed
    without importing every model input class.
    """

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Any:
        r"""
        Return a copy of the bundle with every contained tensor moved to a device and/or dtype.

        Args:
            device: Target device, or ``None`` to leave the device unchanged.
            dtype: Target dtype, or ``None`` to leave dtypes unchanged.
            non_blocking: If ``True``, request an asynchronous host-to-device copy where supported.
            copy: If ``True``, force a copy even when the move is a no-op.

        Returns:
            A bundle of the same type with its tensors relocated.

        Examples:
            >>> model_input.to("cuda")  # doctest: +SKIP
        """

        ...


@dataclass(frozen=True, slots=True)
class StructureToken:
    r"""
    Token-level view over one or more atoms in a structure.

    Attributes:
        residue: Residue key owning this token.
        atom_indices: Atom indices included in this token.
        representative_atom_index: Atom index used as the token representative.

    Examples:
        >>> from ofoldx.data.structure import ResidueKey
        >>> token = StructureToken(residue=ResidueKey("A", 1, ""), atom_indices=(0, 1),
        ...                        representative_atom_index=1)
        >>> token.atom_indices
        (0, 1)
    """

    residue: ResidueKey
    atom_indices: tuple[int, ...]
    representative_atom_index: int


@dataclass(frozen=True, slots=True)
class StructuralFields:
    r"""
    Model-agnostic structural pieces shared by every processor.

    Processors build this once, then layer model-specific encodings on top. It holds the post-materialization
    structure, tokenization, atom/token partition, identity indices, reference-conformer geometry, frames, and
    target-coordinate labels.

    Attributes:
        structure: Parsed structure after optional materialization.
        tokens: Tokenized structure view.
        residues: Residue key for each token.
        residue_names: Residue names keyed by residue id.
        chain_to_index: Mapping from chain id to contiguous chain index.
        num_tokens: Number of tokens.
        num_atoms: Number of atoms.
        atom_to_token_values: Python list mapping atom index to token index.
        atom_to_token: Tensor mapping atom index to token index.
        atom_mask: Boolean atom validity mask.
        token_mask: Boolean token validity mask.
        representative_atom_mask: Mask selecting representative atoms.
        atom_token_index: Per-token atom offset for each atom.
        token_to_rep_atom: Representative atom index for each token.
        entity_values: Entity id for each token.
        reference_space_values: Reference-conformer group id for each atom.
        reference_positions: Centered reference-conformer atom coords.
        reference_mask: Boolean mask for valid reference-conformer atoms.
        reference_charges: Per-atom reference formal charges.
        molecule_type_values: Molecule type id for each token.
        frame_atom_index_values: Frame atom indices for each token.
        frame_mask_values: Frame validity mask values.
        target_positions: Experimental atom coords.
        target_mask: Experimental-coordinate validity mask.

    Examples:
        >>> fields = processor._structural_fields(structure, ccd)  # doctest: +SKIP
        >>> fields.num_tokens  # doctest: +SKIP
        128
    """

    structure: Structure
    tokens: tuple[StructureToken, ...]
    residues: tuple[ResidueKey, ...]
    residue_names: dict[ResidueKey, str]
    chain_to_index: dict[str, int]
    num_tokens: int
    num_atoms: int
    atom_to_token_values: list[int]
    atom_to_token: Tensor
    atom_mask: Tensor
    token_mask: Tensor
    representative_atom_mask: Tensor
    atom_token_index: Tensor
    token_to_rep_atom: Tensor
    entity_values: list[int]
    reference_space_values: list[int]
    reference_positions: Tensor
    reference_mask: Tensor
    reference_charges: Tensor
    molecule_type_values: list[int]
    frame_atom_index_values: list[list[int]]
    frame_mask_values: list[bool]
    target_positions: Tensor
    target_mask: Tensor


def structure_from_input(value: StructureInput) -> Structure:
    r"""
    Return a parsed structure from either a ``Structure`` object or a structure file path.

    Args:
        value: Parsed structure object or structure file path.

    Returns:
        Parsed structure object.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> atom = AtomRecord(name="CA", element="C", residue_name="ALA",
        ...                   residue=ResidueKey("A", 1, ""), x=0.0, y=0.0, z=0.0)
        >>> structure = Structure((atom,), bonds=())
        >>> structure_from_input(structure) is structure
        True
        >>> structure_from_input("complex.cif")  # doctest: +SKIP
        Structure(...)
    """

    if isinstance(value, Structure):
        return value
    return read_structure(value)


# Polymer residue names tokenized one-token-per-residue. Everything else is atom-tokenized.
_PROTEIN_RESIDUE_NAMES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
_RNA_RESIDUE_NAMES = frozenset({"A", "G", "C", "U"})
_DNA_RESIDUE_NAMES = frozenset({"DA", "DG", "DC", "DT"})
_STANDARD_RESIDUE_NAMES = _PROTEIN_RESIDUE_NAMES | _RNA_RESIDUE_NAMES | _DNA_RESIDUE_NAMES


def tokenize_structure(structure: Structure) -> tuple[StructureToken, ...]:
    r"""
    Tokenize a structure into residue tokens for standard polymers and atom tokens otherwise.

    Args:
        structure: Parsed structure to tokenize.

    Returns:
        Structure tokens in atom order.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> residue = ResidueKey("A", 1, "")
        >>> atoms = (
        ...     AtomRecord(name="N", element="N", residue_name="ALA", residue=residue, x=0.0, y=0.0, z=0.0),
        ...     AtomRecord(name="CA", element="C", residue_name="ALA", residue=residue, x=1.0, y=0.0, z=0.0),
        ... )
        >>> tokens = tokenize_structure(Structure(atoms, bonds=()))
        >>> len(tokens), tokens[0].atom_indices
        (1, (0, 1))
    """

    tokens: list[StructureToken] = []
    for atom_indices in _residue_atom_groups(structure):
        first_atom = structure.atoms[atom_indices[0]]
        if _is_standard_residue_token(first_atom.residue_name, first_atom.hetero):
            tokens.append(
                StructureToken(
                    residue=first_atom.residue,
                    atom_indices=atom_indices,
                    representative_atom_index=_representative_atom_index(structure, atom_indices),
                )
            )
        else:
            tokens.extend(
                StructureToken(
                    residue=first_atom.residue,
                    atom_indices=(atom_index,),
                    representative_atom_index=atom_index,
                )
                for atom_index in atom_indices
            )
    return tuple(tokens)


def _residue_atom_groups(structure: Structure) -> list[tuple[int, ...]]:
    r"""
    Group atom indices by ``(residue, residue_name, hetero)``, preserving first-seen order.

    Args:
        structure: Parsed structure to group.

    Returns:
        Atom-index tuples, one per residue group.
    """

    groups: dict[tuple[ResidueKey, str, bool], list[int]] = {}
    for atom_index, atom in enumerate(structure.atoms):
        group_key = (atom.residue, atom.residue_name, atom.hetero)
        groups.setdefault(group_key, []).append(atom_index)
    return [tuple(atom_indices) for atom_indices in groups.values()]


def _atom_to_token(tokens: tuple[StructureToken, ...], num_atoms: int) -> list[int]:
    r"""
    Return the token index owning each atom.

    Args:
        tokens: Structure tokens.
        num_atoms: Total atom count.

    Returns:
        Token index per atom.

    Raises:
        ValueError: If any atom is not covered by exactly one token.
    """

    atom_to_token = [-1] * num_atoms
    for token_index, token in enumerate(tokens):
        for atom_index in token.atom_indices:
            atom_to_token[atom_index] = token_index
    if any(token_index < 0 for token_index in atom_to_token):
        raise ValueError("every atom must belong to exactly one token")
    return atom_to_token


def _atom_token_fields(tokens: tuple[StructureToken, ...], num_atoms: int) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the representative-atom mask, per-atom within-token offset, and per-token representative atom.

    Args:
        tokens: Structure tokens.
        num_atoms: Total atom count.

    Returns:
        ``(representative_atom_mask, atom_token_index, token_to_rep_atom)`` tensors.
    """

    representative_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)
    atom_token_index = torch.zeros(num_atoms, dtype=torch.long)
    token_to_rep_atom = torch.zeros(len(tokens), dtype=torch.long)
    for token_index, token in enumerate(tokens):
        representative_atom_mask[token.representative_atom_index] = True
        token_to_rep_atom[token_index] = token.representative_atom_index
        for atom_offset, atom_index in enumerate(token.atom_indices):
            atom_token_index[atom_index] = atom_offset
    return representative_atom_mask, atom_token_index, token_to_rep_atom


def _token_residue_names(structure: Structure, tokens: tuple[StructureToken, ...]) -> dict[ResidueKey, str]:
    r"""
    Map each token's residue key to its residue name (first-seen name per residue).

    Args:
        structure: Parsed structure.
        tokens: Structure tokens.

    Returns:
        Residue name keyed by residue key.
    """

    names: dict[ResidueKey, str] = {}
    for token in tokens:
        atom = structure.atoms[token.atom_indices[0]]
        names.setdefault(token.residue, atom.residue_name)
    return names


def _reference_space_ids(structure: Structure) -> list[int]:
    r"""
    Assign a unique reference-space id per residue, in first-seen order.

    Args:
        structure: Parsed structure.

    Returns:
        Reference-space id per atom (atoms of the same residue share an id).
    """

    space_ids: dict[ResidueKey, int] = {}
    return [space_ids.setdefault(atom.residue, len(space_ids)) for atom in structure.atoms]


def _is_standard_residue_token(residue_name: str, hetero: bool) -> bool:
    r"""
    Return whether a residue is a non-hetero standard polymer residue (tokenized one token per residue).

    Args:
        residue_name: Residue name to classify.
        hetero: Whether the residue is a HETATM record.

    Returns:
        ``True`` for standard polymer residues, ``False`` otherwise.
    """

    return not hetero and residue_name in _STANDARD_RESIDUE_NAMES


def _representative_atom_index(structure: Structure, atom_indices: tuple[int, ...]) -> int:
    r"""
    Pick a residue token's representative atom: protein ``CA``, nucleic ``C1'``, else the first atom.

    Args:
        structure: Parsed structure.
        atom_indices: Atom indices belonging to the token.

    Returns:
        The chosen representative atom index.
    """

    residue_name = structure.atoms[atom_indices[0]].residue_name
    preferred_atom_name = "CA" if len(residue_name) == 3 else "C1'"
    for atom_index in atom_indices:
        if structure.atoms[atom_index].name == preferred_atom_name:
            return atom_index
    return atom_indices[0]


def _index_values(values: Iterable[str]) -> dict[str, int]:
    r"""
    Assign a contiguous 0-based index to each distinct value, in first-seen order.

    Args:
        values: Iterable of values to index.

    Returns:
        Mapping from value to its first-seen index.
    """

    indices: dict[str, int] = {}
    for value in values:
        if value not in indices:
            indices[value] = len(indices)
    return indices
