r"""
Structure data objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResidueKey:
    r"""
    Stable residue identity inside one structure.

    Attributes:
        chain_id: Author chain identifier.
        residue_number: Author residue number.
        insertion_code: Optional insertion code.

    Examples:
        >>> key = ResidueKey("A", 1)
        >>> key.chain_id, key.residue_number, key.insertion_code
        ('A', 1, '')
    """

    chain_id: str
    residue_number: int
    insertion_code: str = ""


@dataclass(frozen=True, slots=True)
class AtomRecord:
    r"""
    Atom coords and residue identity parsed from a structure file.

    ``mol_type`` is the entity-level molecule type ("protein"/"rna"/"dna"/"ligand") when the source
    provides it (e.g. mmCIF ``_entity_poly.type``), which is how a modified polymer residue (MSE,
    SEP, ...) is distinguished from a true ligand. ``None`` means unknown (e.g. plain PDB), in which
    case the processor falls back to classifying by residue name.

    Attributes:
        name: Atom name within the residue/component.
        element: Uppercase element symbol.
        residue_name: Residue or CCD component name.
        residue: Stable residue identity.
        x: Cartesian x coordinate in Angstrom.
        y: Cartesian y coordinate in Angstrom.
        z: Cartesian z coordinate in Angstrom.
        charge: Parsed or synthesized formal charge.
        hetero: Whether the atom came from a hetero record.
        mol_type: Optional entity-level molecule type.

    Examples:
        >>> atom = AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 1.0, 2.0, 3.0)
        >>> atom.name, atom.element, atom.hetero, atom.mol_type
        ('CA', 'C', False, None)
    """

    name: str
    element: str
    residue_name: str
    residue: ResidueKey
    x: float
    y: float
    z: float
    charge: float = 0.0
    hetero: bool = False
    mol_type: str | None = None


@dataclass(frozen=True, slots=True)
class Structure:
    r"""
    Ordered atom records from a biomolecular structure.

    ``bonds`` holds explicit inter-residue covalent bonds (e.g. mmCIF ``struct_conn`` links such as
    covalent ligand attachments) as pairs of indices into ``atoms``; intra-residue bonds come from
    the CCD and are not listed here. Empty for inputs without such records (e.g. plain PDB).

    Attributes:
        atoms: Ordered atom records.
        bonds: Explicit inter-residue covalent bonds as pairs of indices into ``atoms``.

    Raises:
        ValueError: If ``atoms`` is empty.

    Examples:
        >>> atom = AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0)
        >>> Structure((atom,)).bonds
        ()
        >>> Structure(())
        Traceback (most recent call last):
            ...
        ValueError: structure must contain at least one atom
    """

    atoms: tuple[AtomRecord, ...]
    bonds: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("structure must contain at least one atom")

    @property
    def residues(self) -> tuple[ResidueKey, ...]:
        r"""
        Return residues in first-atom order.

        Returns:
            Unique residue keys ordered by their first atom in ``atoms``.

        Examples:
            >>> a0 = AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0)
            >>> a1 = AtomRecord("CA", "C", "GLY", ResidueKey("B", 1), 1.0, 0.0, 0.0)
            >>> [key.chain_id for key in Structure((a0, a1)).residues]
            ['A', 'B']
        """

        residues = []
        seen = set()
        for atom in self.atoms:
            if atom.residue not in seen:
                residues.append(atom.residue)
                seen.add(atom.residue)
        return tuple(residues)
