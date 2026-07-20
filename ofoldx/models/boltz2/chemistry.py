r"""Released raw-SMILES chemistry for Boltz-2 affinity prediction."""

from __future__ import annotations

from ...data.molecular import Molecule


def standardize_boltz2_affinity_smiles(
    smiles: str,
    *,
    entity_id: str,
) -> Molecule:
    r"""Return the ChEMBL-standardized Boltz-2 affinity ligand graph.

    Boltz-2 applies ChEMBL structure standardization only to raw SMILES affinity
    ligands. The largest standardized fragment is parsed into the canonical
    ``Molecule`` returned here, including the atom names used by Boltz-2.
    """

    from chembl_structure_pipeline.exclude_flag import exclude_flag
    from chembl_structure_pipeline.standardizer import standardize_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if exclude_flag(molecule, includeRDKitSanitization=False):
        raise ValueError("the affinity SMILES is excluded by the released ChEMBL standardization path")
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
