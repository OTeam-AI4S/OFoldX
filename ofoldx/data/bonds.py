r"""
Covalent-bond validation and inter-chain bond detection.

Validates the explicit inter-residue covalent bonds parsed from a structure (mmCIF ``struct_conn``,
e.g. disulfides and covalent ligand attachments) by bond length, and classifies each as intra- or
inter-chain. This exposes as a tested, reusable API the bond-length gate the featurization applies
inline (``processors.structure._token_bonds`` at 2.4 Angstrom) plus the inter-chain detection the
upstream AlphaFold3 structure-cleaning step performs (``structure/bonds.py`` restriction/validation).
Metal-coordination bonds are already excluded upstream at parse time
([`read_mmcif`][ofoldx.io.structure.read_mmcif]), so only covalent links reach here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .structure import AtomRecord, Structure

# Longest accepted covalent bond in Angstrom; matches the featurization cutoff. Bonds longer than this
# are treated as spurious ``struct_conn`` records and dropped (upstream applies the same gate).
MAX_COVALENT_BOND_LENGTH = 2.4


@dataclass(frozen=True, slots=True)
class CovalentBond:
    r"""
    A validated inter-residue covalent bond between two atoms.

    Attributes:
        first_atom_index: Index into ``Structure.atoms`` of the first atom.
        second_atom_index: Index into ``Structure.atoms`` of the second atom.
        length: Bond length in Angstrom.
        inter_chain: Whether the two atoms lie in different chains.

    Examples:
        >>> bond = CovalentBond(first_atom_index=0, second_atom_index=1, length=2.03, inter_chain=True)
        >>> bond.first_atom_index, bond.inter_chain
        (0, True)
    """

    first_atom_index: int
    second_atom_index: int
    length: float
    inter_chain: bool


def bond_length(first: AtomRecord, second: AtomRecord) -> float:
    r"""
    Return the Euclidean distance between two atoms in Angstrom.

    Args:
        first: First atom.
        second: Second atom.

    Returns:
        Distance between the two atoms.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey
        >>> key = ResidueKey("A", 1)
        >>> a0 = AtomRecord("SG", "S", "CYS", key, 0.0, 0.0, 0.0)
        >>> a1 = AtomRecord("SG", "S", "CYS", key, 2.0, 0.0, 0.0)
        >>> round(bond_length(a0, a1), 4)
        2.0
    """

    return float(((first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2) ** 0.5)


def validate_covalent_bonds(
    structure: Structure, *, max_bond_length: float = MAX_COVALENT_BOND_LENGTH
) -> list[CovalentBond]:
    r"""
    Validate a structure's explicit covalent bonds by length and classify them.

    Bonds whose atoms are farther apart than ``max_bond_length`` are dropped as spurious. Each retained
    bond is tagged as inter-chain when its two atoms belong to different chains.

    Args:
        structure: Parsed structure whose ``bonds`` are index pairs into ``atoms``.
        max_bond_length: Maximum accepted bond length in Angstrom.

    Returns:
        Validated covalent bonds with their lengths and inter-chain flags.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> a0 = AtomRecord("SG", "S", "CYS", ResidueKey("A", 1), 0.0, 0.0, 0.0)
        >>> a1 = AtomRecord("SG", "S", "CYS", ResidueKey("B", 1), 2.0, 0.0, 0.0)
        >>> bonds = validate_covalent_bonds(Structure((a0, a1), bonds=((0, 1),)))
        >>> [(bond.length, bond.inter_chain) for bond in bonds]
        [(2.0, True)]
    """

    atoms = structure.atoms
    validated: list[CovalentBond] = []
    for first_index, second_index in structure.bonds:
        first, second = atoms[first_index], atoms[second_index]
        length = bond_length(first, second)
        if length > max_bond_length:
            continue
        validated.append(
            CovalentBond(
                first_atom_index=first_index,
                second_atom_index=second_index,
                length=length,
                inter_chain=first.residue.chain_id != second.residue.chain_id,
            )
        )
    return validated


def inter_chain_bonds(structure: Structure, *, max_bond_length: float = MAX_COVALENT_BOND_LENGTH) -> list[CovalentBond]:
    r"""
    Return only the validated inter-chain covalent bonds of a structure.

    These are the covalent links that join distinct chains (e.g. an inter-chain disulfide or a ligand
    covalently attached to a different chain); they must be preserved through featurization because the
    standard polymer connectivity encoded by ``residue_index`` does not represent them.

    Args:
        structure: Parsed structure.
        max_bond_length: Maximum accepted bond length in Angstrom.

    Returns:
        Validated inter-chain covalent bonds.

    Examples:
        >>> from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
        >>> a0 = AtomRecord("SG", "S", "CYS", ResidueKey("A", 1), 0.0, 0.0, 0.0)
        >>> a1 = AtomRecord("SG", "S", "CYS", ResidueKey("B", 1), 2.0, 0.0, 0.0)
        >>> len(inter_chain_bonds(Structure((a0, a1), bonds=((0, 1),))))
        1
    """

    return [bond for bond in validate_covalent_bonds(structure, max_bond_length=max_bond_length) if bond.inter_chain]
