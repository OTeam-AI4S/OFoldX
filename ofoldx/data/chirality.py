r"""
Ligand chirality validation against reference molecules.

Compares the chirality of a ligand -- built from its experimental coordinates and a reference bond
graph -- against the reference molecule's chiral centers, flagging inversions such as a D-sugar
predicted as an L-sugar. Faithful port of upstream AlphaFold3 ``model/scoring/chirality.py``
(``_find_chiral_centres``, ``_chiral_match``, ``compare_chirality``); it reuses the RDKit CCD Mol
builder in [`ccd`][ofoldx.data.ccd] for reference molecules. RDKit is a declared dependency, so
this module imports it at load; it operates on the flat [`Structure`][ofoldx.data.structure.Structure]
atom list rather than upstream's chain-filtered structure object.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
from rdkit import Chem

from .structure import ResidueKey, Structure

logger = logging.getLogger(__name__)

# Only carbon and sulfur centers are checked (upstream ``_CHIRAL_ELEMENTS``); other stereocenters are
# either uncommon or unreliably assigned from coordinates alone.
_CHIRAL_ELEMENTS = frozenset({"C", "S"})


def find_chiral_centres(mol: Chem.Mol) -> dict[str, str]:
    r"""
    Return the chirality (``R``/``S``/``?``) of each carbon/sulfur chiral center in a molecule.

    Args:
        mol: Molecule whose atoms carry an ``atom_name`` property.

    Returns:
        Mapping from chiral-center atom name to its assigned chirality.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.AddHs(Chem.MolFromSmiles("[C@H](F)(Cl)Br"))
        >>> Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        >>> for index, atom in enumerate(mol.GetAtoms()):
        ...     atom.SetProp("atom_name", f"{atom.GetSymbol()}{index}")
        >>> find_chiral_centres(mol)
        {'C0': 'S'}
    """

    chiral_centres = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=False, useLegacyImplementation=True)
    name_by_index = {atom.GetIdx(): atom.GetProp("atom_name") for atom in mol.GetAtoms() if atom.HasProp("atom_name")}
    chirality_by_name = {
        name_by_index[index]: chirality for index, chirality in chiral_centres if index in name_by_index
    }
    return {
        name: chirality
        for name, chirality in chirality_by_name.items()
        if any(name[: len(element)].upper() == element for element in _CHIRAL_ELEMENTS)
    }


def chiral_match(test_mol: Chem.Mol, reference_mol: Chem.Mol) -> bool:
    r"""
    Return whether the test molecule's chiral centers match the reference.

    The test molecule may cover a subset of the reference atoms/centers; every shared center whose
    reference chirality is assigned must agree.

    Args:
        test_mol: Molecule built from experimental coordinates.
        reference_mol: Reference molecule with known chirality.

    Returns:
        ``True`` if every comparable chiral center matches, else ``False``.

    Raises:
        ValueError: If the test atoms are not a subset of the reference atoms.

    Examples:
        >>> from rdkit import Chem
        >>> def named(smiles):
        ...     mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        ...     Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        ...     for index, atom in enumerate(mol.GetAtoms()):
        ...         atom.SetProp("atom_name", f"{atom.GetSymbol()}{index}")
        ...     return mol
        >>> s_form, r_form = named("[C@H](F)(Cl)Br"), named("[C@@H](F)(Cl)Br")
        >>> chiral_match(s_form, s_form)
        True
        >>> chiral_match(s_form, r_form)
        False
    """

    test_names = {atom.GetProp("atom_name") for atom in test_mol.GetAtoms() if atom.HasProp("atom_name")}
    reference_names = {atom.GetProp("atom_name") for atom in reference_mol.GetAtoms() if atom.HasProp("atom_name")}
    if test_names != reference_names and not test_names.issubset(reference_names):
        raise ValueError("test atoms are not a subset of reference atoms")

    test_centres = find_chiral_centres(test_mol)
    reference_centres = find_chiral_centres(reference_mol)
    if set(test_centres) != set(reference_centres) and not set(test_centres).issubset(reference_centres):
        return False
    return all(
        chirality == reference_centres[name]
        for name, chirality in test_centres.items()
        if reference_centres[name] != "?"
    )


def mol_from_ligand_coords(reference_mol: Chem.Mol, coords_by_atom_name: Mapping[str, np.ndarray]) -> Chem.Mol | None:
    r"""
    Build a molecule from a reference bond graph and experimental ligand coordinates.

    Drops reference atoms not present in ``coords_by_atom_name``, attaches the given coordinates as a
    conformer, and assigns stereochemistry from 3D. Returns ``None`` when RDKit cannot normalize the
    geometry (e.g. degenerate/colinear atoms), matching the upstream degenerate-vector guard.

    Args:
        reference_mol: Reference molecule with per-atom ``atom_name`` properties.
        coords_by_atom_name: Mapping from atom name to its ``[3]`` experimental coordinate.

    Returns:
        Molecule with an assigned 3D conformer, or ``None`` when stereochemistry cannot be assigned.

    Examples:
        >>> from ofoldx.data.ccd import mol_from_ccd_block  # doctest: +SKIP
        >>> reference = mol_from_ccd_block(ccd_blocks["NAG"])  # doctest: +SKIP
        >>> coords = {"C1": [0.0, 0.0, 0.0], "C2": [1.5, 0.0, 0.0]}  # doctest: +SKIP
        >>> mol_from_ligand_coords(reference, coords)  # doctest: +SKIP
        <rdkit.Chem.rdchem.Mol object at ...>
    """

    mol = Chem.Mol(reference_mol)
    mol.RemoveAllConformers()
    to_remove = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not atom.HasProp("atom_name") or atom.GetProp("atom_name") not in coords_by_atom_name
    ]
    editable = Chem.EditableMol(mol)
    for index in reversed(to_remove):
        editable.RemoveAtom(index)
    mol = editable.GetMol()

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, atom in enumerate(mol.GetAtoms()):
        conformer.SetAtomPosition(index, [float(value) for value in coords_by_atom_name[atom.GetProp("atom_name")]])
    mol.AddConformer(conformer)
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except RuntimeError as error:
        if "Cannot normalize a zero length vector" in str(error):
            return None
        raise
    return mol


def compare_chirality(
    structure: Structure,
    reference_mol_by_residue: Mapping[ResidueKey, Chem.Mol],
) -> dict[ResidueKey, bool]:
    r"""
    Compare each single-residue ligand's chirality against its reference molecule.

    Only single-residue ligands with an available reference molecule are compared; ligands that span
    multiple residues, have no reference, or whose geometry is degenerate are skipped (a warning is
    logged). Reference molecules must carry per-atom ``atom_name`` properties matching the CCD naming
    (see [`mol_from_ccd_block`][ofoldx.data.ccd.mol_from_ccd_block]).

    Args:
        structure: Parsed structure whose ligand residues should be checked.
        reference_mol_by_residue: Mapping from a ligand residue key to its reference molecule.

    Returns:
        Mapping from a ligand residue key to whether its chirality matches the reference.

    Examples:
        >>> from ofoldx.io import read_structure  # doctest: +SKIP
        >>> structure = read_structure("complex.cif")  # doctest: +SKIP
        >>> compare_chirality(structure, reference_mol_by_residue)  # doctest: +SKIP
        {ResidueKey(chain_id='B', residue_number=401, insertion_code=''): True}
    """

    residue_atoms: dict[ResidueKey, list[int]] = {}
    for index, atom in enumerate(structure.atoms):
        residue_atoms.setdefault(atom.residue, []).append(index)

    matches: dict[ResidueKey, bool] = {}
    for residue, reference_mol in reference_mol_by_residue.items():
        indices = residue_atoms.get(residue)
        if indices is None:
            continue
        coords_by_atom_name = {
            structure.atoms[index].name: np.asarray(
                [structure.atoms[index].x, structure.atoms[index].y, structure.atoms[index].z], dtype=np.float64
            )
            for index in indices
        }
        test_mol = mol_from_ligand_coords(reference_mol, coords_by_atom_name)
        if test_mol is None:
            logger.warning("failed to build ligand mol for residue %s; skipping chirality check", residue)
            continue
        matches[residue] = chiral_match(test_mol, reference_mol)
    return matches
