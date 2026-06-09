r"""
Structure IO.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ofoldx.data.structure import AtomRecord, Structure


def read_structure(path: str | PathLike[str], *, format: str = "auto") -> Structure:
    r"""
    Read a biomolecular structure file.

    Args:
        path: PDB or mmCIF file path.
        format: Input format. ``"auto"`` infers from the suffix; supported explicit values are ``"pdb"``,
            ``"cif"``, and ``"mmcif"``.

    Returns:
        Parsed structure containing kept heavy atoms and explicit inter-residue covalent bonds when available.

    Raises:
        ValueError: If ``format`` or the inferred suffix is unsupported.

    Examples:
        >>> structure = read_structure("complex.cif")  # doctest: +SKIP
        >>> len(structure.atoms)  # doctest: +SKIP
        1234
    """

    path = Path(path)
    structure_format = path.suffix.lower().lstrip(".") if format == "auto" else format.lower()
    if structure_format == "pdb":
        return read_pdb(path)
    if structure_format in {"cif", "mmcif"}:
        return read_mmcif(path)
    raise ValueError(f"unsupported structure format: {structure_format!r}")


def read_pdb(path: str | PathLike[str]) -> Structure:
    r"""
    Read ATOM/HETATM records from a PDB file.

    Args:
        path: PDB file path.

    Returns:
        Parsed structure from the first ``MODEL`` block, or from the whole file when no model records are present.

    Examples:
        >>> structure = read_pdb("complex.pdb")  # doctest: +SKIP
        >>> structure.atoms[0].name  # doctest: +SKIP
        'N'
    """

    from ofoldx.data.structure import Structure

    atoms = []
    seen_model = False
    active_model = True
    with Path(path).open() as file:
        for line in file:
            if line.startswith("MODEL"):
                seen_model = True
                active_model = _pdb_model_number(line) == 1
                continue
            if seen_model and line.startswith("ENDMDL"):
                if active_model:
                    break
                active_model = False
                continue
            if not active_model:
                continue
            if line.startswith(("ATOM  ", "HETATM")) and _accept_altloc(line):
                atom = _parse_atom_line(line)
                if _keep_atom(atom.name, atom.element, atom.residue_name):
                    atoms.append(atom)
    return Structure(tuple(atoms))


def read_mmcif(path: str | PathLike[str]) -> Structure:
    r"""
    Read atom records from an mmCIF file.

    Args:
        path: mmCIF file path.

    Returns:
        Parsed structure from model 1, including accepted inter-residue covalent bonds from the structure table.

    Examples:
        >>> structure = read_mmcif("complex.cif")  # doctest: +SKIP
        >>> structure.bonds  # doctest: +SKIP
        ((10, 42), ...)
    """

    from biotite.structure.io import pdbx

    from ofoldx.data.structure import AtomRecord, ResidueKey, Structure

    cif = pdbx.CIFFile.read(path)
    atom_array = pdbx.get_structure(
        cif,
        model=1,
        altloc="first",
        use_author_fields=True,
        include_bonds=True,
        extra_fields=["label_entity_id"],
    )
    entity_mol_types = _entity_mol_types(cif)
    atoms: list[AtomRecord] = []
    residues: list[ResidueKey] = []
    index_map: dict[int, int] = {}  # original atom-array index -> kept-atom index
    for original_index, atom in enumerate(atom_array):
        name = str(atom.atom_name).strip()
        element = str(atom.element).strip().upper() or _infer_element(name)
        residue_name = str(atom.res_name).strip().upper()
        if not _keep_atom(name, element, residue_name):
            continue
        residue = ResidueKey(
            chain_id=str(atom.chain_id).strip(),
            residue_number=int(atom.res_id),
            insertion_code=str(getattr(atom, "ins_code", "")).strip(),
        )
        index_map[original_index] = len(atoms)
        residues.append(residue)
        atoms.append(
            AtomRecord(
                name=name,
                element=element,
                residue_name=residue_name,
                residue=residue,
                x=float(atom.coord[0]),
                y=float(atom.coord[1]),
                z=float(atom.coord[2]),
                hetero=bool(getattr(atom, "hetero", False)),
                mol_type=_atom_mol_type(str(getattr(atom, "label_entity_id", "")).strip(), entity_mol_types),
            ),
        )
    return Structure(tuple(atoms), bonds=_inter_residue_bonds(atom_array, index_map, residues))


def _pdb_model_number(line: str) -> int:
    r"""
    Return the PDB ``MODEL`` serial, defaulting malformed/blank serials to the first model.
    """

    try:
        return int(line[10:14].strip() or "1")
    except ValueError:
        return 1


def _entity_mol_types(cif: object) -> dict[str, str]:
    r"""
    Map ``label_entity_id`` -> mol_type from the mmCIF ``_entity_poly.type`` (empty if absent).
    """

    try:
        block = cif.block  # type: ignore[attr-defined]
        if "entity_poly" not in block:
            return {}
        entity_poly = block["entity_poly"]
        entity_ids = entity_poly["entity_id"].as_array(str)
        poly_types = entity_poly["type"].as_array(str)
    except Exception:  # noqa: BLE001 - biotite version / missing-category robustness
        return {}
    return {str(entity_id): _poly_type_to_mol_type(poly_type) for entity_id, poly_type in zip(entity_ids, poly_types)}


def _poly_type_to_mol_type(poly_type: str) -> str:
    r"""
    Map an mmCIF ``_entity_poly.type`` string to a house molecule type.

    Args:
        poly_type: The mmCIF polymer type (e.g. ``"polypeptide(L)"``).

    Returns:
        ``"protein"`` / ``"dna"`` / ``"rna"``, defaulting to ``"ligand"`` for unrecognized types. DNA is
        checked before RNA because its type string also contains ``"ribonucleotide"``.
    """

    poly_type = poly_type.lower()
    if "polypeptide" in poly_type:
        return "protein"
    if "polydeoxyribonucleotide" in poly_type:  # checked before rna (it contains "ribonucleotide")
        return "dna"
    if "polyribonucleotide" in poly_type:
        return "rna"
    return "ligand"


def _atom_mol_type(entity_id: str, entity_mol_types: dict[str, str]) -> str | None:
    r"""
    Resolve an atom's molecule type from its entity id, or ``None`` when no entity table exists.

    Args:
        entity_id: The atom's ``label_entity_id``.
        entity_mol_types: Entity-id to mol-type map from the mmCIF ``entity_poly`` category.

    Returns:
        The molecule type, ``"ligand"`` for an unlisted entity, or ``None`` when no entity table is
        available (so the caller falls back to residue-name classification).
    """

    if not entity_mol_types:
        return None  # no entity_poly category: fall back to residue-name classification
    return entity_mol_types.get(entity_id, "ligand")


def _inter_residue_bonds(
    atom_array: object,
    index_map: dict[int, int],
    residues: Sequence[object],
) -> tuple[tuple[int, int], ...]:
    r"""
    Extract inter-residue covalent bonds (struct_conn) from a biotite atom array.

    Uses biotite's own bond reader and keeps only bonds whose atoms survived filtering and lie in
    different residues (intra-residue bonds come from the CCD). Defensive: any absence/incompatibility
    degrades to no bonds. (Validated only where biotite is installed.)
    """

    bonds = getattr(atom_array, "bonds", None)
    if bonds is None:
        return ()
    try:
        from biotite.structure import BondType

        coordination = int(BondType.COORDINATION)
        bond_records = bonds.as_array()
    except Exception:  # noqa: BLE001 - biotite version / empty-bond robustness
        return ()
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for record in bond_records:
        if int(record[2]) == coordination:
            continue  # exclude metal-coordination bonds (AF3 keeps only covalent links)
        first, second = index_map.get(int(record[0])), index_map.get(int(record[1]))
        if first is None or second is None or residues[first] == residues[second]:
            continue
        key = (first, second) if first < second else (second, first)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return tuple(pairs)


def _accept_altloc(line: str) -> bool:
    r"""
    Return whether a PDB atom line is the primary alternate location (blank or ``"A"``).

    Args:
        line: A raw PDB ``ATOM``/``HETATM`` line.

    Returns:
        ``True`` for the blank or ``"A"`` altloc, keeping a single copy of alt-loc atoms.
    """

    altloc = line[16:17]
    return altloc in {" ", "A"}


def _parse_atom_line(line: str) -> AtomRecord:
    r"""
    Parse one fixed-column PDB ``ATOM``/``HETATM`` line into an [`AtomRecord`][ofoldx.data.structure.AtomRecord].

    Args:
        line: A raw PDB ``ATOM``/``HETATM`` line.

    Returns:
        The parsed atom record, with the element inferred from the atom name when the element column
        is blank and ``hetero`` set for ``HETATM`` lines.
    """

    from ofoldx.data.structure import AtomRecord, ResidueKey

    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    insertion_code = line[26:27].strip()
    element = line[76:78].strip() or _infer_element(atom_name)
    return AtomRecord(
        name=atom_name,
        element=element.upper(),
        residue_name=residue_name.upper(),
        residue=ResidueKey(
            chain_id=chain_id,
            residue_number=int(line[22:26]),
            insertion_code=insertion_code,
        ),
        x=float(line[30:38]),
        y=float(line[38:46]),
        z=float(line[46:54]),
        hetero=line.startswith("HETATM"),
    )


def _infer_element(atom_name: str) -> str:
    r"""
    Infer an element symbol from a PDB atom name (its first alphabetic character).

    Args:
        atom_name: PDB-style atom name.

    Returns:
        Single-letter uppercase element symbol.

    Raises:
        ValueError: If the atom name contains no letters.
    """

    for character in atom_name:
        if character.isalpha():
            return character.upper()
    raise ValueError(f"cannot infer element from atom name {atom_name!r}")


def _keep_atom(atom_name: str, element: str, residue_name: str) -> bool:
    r"""
    Return whether an atom survives filtering: heavy, non-water, non-hydrogen.

    Hydrogen / deuterium / unknown-element (``X``) atoms and water residues are dropped so only heavy
    structural atoms are kept.

    Args:
        atom_name: PDB-style atom name.
        element: Element symbol (may be blank).
        residue_name: Three-letter residue name.

    Returns:
        ``True`` to keep the atom, ``False`` to drop it.
    """

    element = element.upper()
    if element in {"H", "D", "X"}:
        return False
    if residue_name.upper() in {"HOH", "DOD", "WAT", "H2O", "D2O"}:
        return False
    return not atom_name.upper().startswith(("H", "D")) or element not in {"", "H", "D"}
