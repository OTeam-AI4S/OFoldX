r"""
Raw-SMILES chemistry for Boltz-2 affinity prediction.
"""

from __future__ import annotations

from ...data.molecular import Molecule


def standardize_boltz2_affinity_smiles(
    smiles: str,
) -> Molecule:
    r"""
    Standardize a SMILES string for Boltz-2 affinity scoring.

    Args:
        smiles: Ligand represented as a SMILES string.

    Returns:
        The largest ChEMBL-standardized fragment as a ``Molecule`` with heavy-atom
        names suitable for Boltz-2 features.

    Raises:
        ValueError: If ChEMBL excludes the molecule, standardization produces an
            invalid molecule, or an atom name exceeds four characters.
    """

    from chembl_structure_pipeline.exclude_flag import exclude_flag
    from chembl_structure_pipeline.standardizer import standardize_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if exclude_flag(molecule, includeRDKitSanitization=False):
        raise ValueError("the affinity SMILES is excluded by ChEMBL standardization")
    molecule = rdMolStandardize.LargestFragmentChooser().choose(molecule)
    molecule = standardize_mol(molecule)
    standardized_smiles = Chem.MolToSmiles(molecule)
    if Chem.MolFromSmiles(standardized_smiles) is None:
        raise ValueError("Boltz-2 affinity standardization produced an invalid molecule")

    molecule = AllChem.MolFromSmiles(standardized_smiles)
    molecule = AllChem.AddHs(molecule)
    canonical_order = AllChem.CanonicalRankAtoms(molecule)
    atom_ids: list[str] = []
    for atom, canonical_index in zip(molecule.GetAtoms(), canonical_order):
        atom.SetProp("name", atom.GetSymbol().upper() + str(canonical_index + 1))
        atom_id = atom.GetProp("name")
        if len(atom_id) > 4:
            raise ValueError(f"Boltz-2 atom name cannot be encoded in four characters: {atom_id!r}")
        if atom.GetAtomicNum() == 1:
            continue
        atom_ids.append(atom_id)

    return Molecule.from_smiles(
        standardized_smiles,
        atom_ids=tuple(atom_ids),
    )
