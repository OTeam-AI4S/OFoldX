r"""Private lossless SMILES import for the canonical molecular graph."""

from __future__ import annotations

import re

from .molecular import (
    ChemicalAtom,
    ChemicalBond,
    ChemicalComponent,
    Molecule,
)

_RDKIT_TETRAHEDRAL_STEREO = frozenset({"CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"})
_RDKIT_BOND_STEREO = {
    "STEREOE": "E",
    "STEREOZ": "Z",
    "STEREOCIS": "cis",
    "STEREOTRANS": "trans",
}
_RDKIT_BOND_TYPES = frozenset({"SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", "AROMATIC"})
_RDKIT_DIRECTIONAL_BONDS = frozenset({"ENDUPRIGHT", "ENDDOWNRIGHT"})
_BRACKET_ATOM = re.compile(r"\[([^\[\]]*)\]")
_BRACKET_PREFIX = re.compile(r"(?P<isotope>\d+)?(?:\*|[A-Z][a-z]?|[a-z]{1,2})(?P<rest>.*)")


def molecule_from_smiles(
    smiles: str,
    *,
    atom_ids: tuple[str, ...] | None,
) -> Molecule:
    r"""Build a canonical Molecule from the exactly representable SMILES subset."""

    _validate_bracket_atom_numbers(smiles)

    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    parameters = Chem.SmilesParserParams()
    parameters.allowCXSMILES = False
    parameters.parseName = False
    parameters.removeHs = False
    parameters.sanitize = False
    molecule = Chem.MolFromSmiles(smiles, parameters)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    declared_atom_stereo = tuple(str(atom.GetChiralTag()) for atom in molecule.GetAtoms())
    declared_bond_directions = tuple(str(bond.GetBondDir()) for bond in molecule.GetBonds())
    try:
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid SMILES chemistry: {smiles!r}") from error
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    for index, declared in enumerate(declared_atom_stereo):
        retained = str(molecule.GetAtomWithIdx(index).GetChiralTag())
        if declared != "CHI_UNSPECIFIED" and retained != declared:
            raise ValueError(f"SMILES contains invalid or unsupported atom stereo at atom index {index}")
    for index, declared in enumerate(declared_bond_directions):
        retained = str(molecule.GetBondWithIdx(index).GetBondDir())
        if declared != "NONE" and retained != declared:
            raise ValueError(f"SMILES contains incomplete or invalid directional stereo at bond index {index}")

    if molecule.GetStereoGroups():
        raise ValueError("enhanced SMILES stereo groups are not supported by the canonical graph")

    # AssignStereochemistry() performs declaration cleanup, but its legacy
    # CIP labels are approximate for dependent/para stereochemistry. Keep
    # only labels produced by RDKit's true CIP implementation.
    for atom in molecule.GetAtoms():
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    try:
        rdCIPLabeler.AssignCIPLabels(molecule)
    except RuntimeError as error:
        raise ValueError("SMILES stereochemistry has no exact CIP assignment") from error

    if atom_ids is None:
        element_counts: dict[str, int] = {}
        generated_ids: list[str] = []
        for atom in molecule.GetAtoms():
            element = atom.GetSymbol().upper()
            element_counts[element] = element_counts.get(element, 0) + 1
            generated_ids.append(f"{element}{element_counts[element]}")
        stable_ids = tuple(generated_ids)
    else:
        if not isinstance(atom_ids, tuple):
            raise TypeError("atom_ids must be a tuple of strings or None")
        stable_ids = atom_ids
        if any(not isinstance(value, str) or not value for value in stable_ids):
            raise ValueError("atom_ids must contain non-empty strings")
        if len(stable_ids) != molecule.GetNumAtoms():
            raise ValueError(f"atom_ids has length {len(stable_ids)}, expected {molecule.GetNumAtoms()}")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("atom_ids must be unique")

    atoms: list[ChemicalAtom] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            raise ValueError(f"SMILES atom mapping is not supported at atom index {atom.GetIdx()}")
        chiral_tag = str(atom.GetChiralTag())
        stereo = None
        if chiral_tag != "CHI_UNSPECIFIED" and chiral_tag not in _RDKIT_TETRAHEDRAL_STEREO:
            raise ValueError(
                f"unsupported non-tetrahedral SMILES stereo {chiral_tag!r} " f"at atom index {atom.GetIdx()}"
            )
        if chiral_tag in _RDKIT_TETRAHEDRAL_STEREO:
            if not atom.HasProp("_CIPCode"):
                raise ValueError(
                    f"SMILES tetrahedral stereo at atom index {atom.GetIdx()} " "has no exact CIP assignment"
                )
            stereo = atom.GetProp("_CIPCode")
            if stereo not in {"R", "S", "r", "s"}:
                raise ValueError(f"unsupported SMILES CIP stereo {stereo!r} at atom index {atom.GetIdx()}")
        if atom.GetAtomicNum() == 0:
            raise ValueError(f"SMILES dummy atom at index {atom.GetIdx()} is not a chemical element")
        atoms.append(
            ChemicalAtom(
                atom_id=stable_ids[atom.GetIdx()],
                element=atom.GetSymbol(),
                isotope=atom.GetIsotope() or None,
                formal_charge=atom.GetFormalCharge(),
                radical_electrons=atom.GetNumRadicalElectrons(),
                explicit_hydrogens=atom.GetNumExplicitHs(),
                implicit_hydrogens=atom.GetNumImplicitHs(),
                implicit_hydrogens_allowed=not atom.GetNoImplicit(),
                aromatic=atom.GetIsAromatic(),
                stereo=stereo,
            )
        )

    bonds: list[ChemicalBond] = []
    for bond in molecule.GetBonds():
        bond_type = str(bond.GetBondType())
        if bond_type not in _RDKIT_BOND_TYPES:
            raise ValueError(f"unsupported SMILES bond type {bond_type!r} at bond index {bond.GetIdx()}")
        bond_stereo = str(bond.GetStereo())
        if bond_stereo == "STEREONONE":
            stereo = None
        elif bond.HasProp("_CIPCode"):
            stereo = bond.GetProp("_CIPCode")
        else:
            stereo = _RDKIT_BOND_STEREO.get(bond_stereo)
        if bond_stereo != "STEREONONE" and stereo is None:
            raise ValueError(f"unsupported SMILES bond stereo {bond_stereo!r} at bond index {bond.GetIdx()}")
        stereo_indices = tuple(bond.GetStereoAtoms())
        stereo_atoms = None
        if stereo is not None:
            if len(stereo_indices) != 2:
                raise ValueError(f"SMILES bond stereo at bond index {bond.GetIdx()} lacks two reference atoms")
            stereo_atoms = (stable_ids[stereo_indices[0]], stable_ids[stereo_indices[1]])
        bonds.append(
            ChemicalBond(
                atom_a=stable_ids[bond.GetBeginAtomIdx()],
                atom_b=stable_ids[bond.GetEndAtomIdx()],
                order=bond.GetBondTypeAsDouble(),
                aromatic=bond.GetIsAromatic(),
                stereo=stereo,
                stereo_atoms=stereo_atoms,
            )
        )

    for bond in molecule.GetBonds():
        direction = str(bond.GetBondDir())
        if direction == "NONE":
            continue
        if direction not in _RDKIT_DIRECTIONAL_BONDS or not _direction_is_encoded(molecule, bond):
            raise ValueError(f"unsupported SMILES bond direction {direction!r} at bond index {bond.GetIdx()}")

    return Molecule(
        components=(
            ChemicalComponent(
                component_id="UNL",
                atoms=tuple(atoms),
                bonds=frozenset(bonds),
            ),
        ),
    )


def _validate_bracket_atom_numbers(smiles: str) -> None:
    r"""Reject bracket fields that RDKit would narrow before exposing them."""

    for bracket in _BRACKET_ATOM.finditer(smiles):
        contents = bracket.group(1)
        if re.search(r":\d+", contents):
            raise ValueError("SMILES atom mapping is not supported")
        match = _BRACKET_PREFIX.fullmatch(contents)
        if match is None:
            continue

        isotope_text = match.group("isotope")
        if isotope_text is not None:
            normalized_isotope = isotope_text.lstrip("0") or "0"
            if (
                normalized_isotope == "0"
                or len(normalized_isotope) > 5
                or (len(normalized_isotope) == 5 and normalized_isotope > "65535")
            ):
                raise ValueError("SMILES bracket isotope must be within 1..65535")

        rest = match.group("rest")
        hydrogen_match = re.search(r"H(?P<count>\d*)", rest)
        if hydrogen_match is not None:
            count_text = hydrogen_match.group("count")
            normalized_count = count_text.lstrip("0") or "0"
            if len(normalized_count) > 3 or (len(normalized_count) == 3 and normalized_count > "255"):
                raise ValueError("SMILES bracket explicit hydrogen count must be at most 255")

        numeric_charge = re.search(r"(?P<sign>[+-])(?P<magnitude>\d+)$", rest)
        if numeric_charge is not None:
            magnitude = numeric_charge.group("magnitude").lstrip("0") or "0"
            maximum = "127" if numeric_charge.group("sign") == "+" else "128"
            if len(magnitude) > 3 or (len(magnitude) == 3 and magnitude > maximum):
                raise ValueError("SMILES bracket formal charge must be within -128..127")
            continue
        repeated_charge = re.search(r"(?P<sign>\++|-+)$", rest)
        if repeated_charge is not None:
            token = repeated_charge.group("sign")
            charge = len(token) if token[0] == "+" else -len(token)
            if charge < -128 or charge > 127:
                raise ValueError("SMILES bracket formal charge must be within -128..127")


def _direction_is_encoded(molecule: object, directed_bond: object) -> bool:
    r"""Return whether a directional RDKit bond is captured by stored E/Z stereo."""

    from rdkit import Chem

    directed_endpoints = frozenset((directed_bond.GetBeginAtomIdx(), directed_bond.GetEndAtomIdx()))
    for bond in molecule.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        stereo_atoms = tuple(bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            continue
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if directed_endpoints in {
            frozenset((stereo_atoms[0], begin)),
            frozenset((stereo_atoms[1], end)),
        }:
            return True
    return False
