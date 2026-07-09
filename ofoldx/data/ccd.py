r"""
Chemical Component Dictionary (CCD) reference lookup.

Provides per-component reference-conformer positions, validity masks, formal charges, and the covalent bond graph
used to build token-bond features. Two providers satisfy the same [`CcdProvider`][ofoldx.data.ccd.CcdProvider]
contract: [`RdkitPickleCcdProvider`][ofoldx.data.ccd.RdkitPickleCcdProvider] consumes the prebaked RDKit-Mol pickle
used by the checkpoint conversion path, so reference coords stay byte-exact when the same artifact is configured;
[`RdkitCifCcdProvider`][ofoldx.data.ccd.RdkitCifCcdProvider] builds each component from a CCD ``components.cif`` via
RDKit (see the "CIF -> Mol construction" section below). Tests inject a small in-memory provider instead, so the
committed suite stays self-contained.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .residue_constants import HYDROGEN_ELEMENTS, LEAVING_ATOMS

logger = logging.getLogger(__name__)

CCD_RDKIT_MOL_ENV = "OFOLDX_CCD_RDKIT_MOL_FILE"
# Path to the prebaked RDKit-Mol pickle for the pickle provider (external artifact, not shipped); unset means that
# provider disables itself and returns ``None`` for every lookup.
DEFAULT_CCD_RDKIT_MOL_FILE = ""
# Path to a CCD ``components.cif`` for the RDKit CIF provider (external reference data, not shipped); unset means
# that provider disables itself and returns ``None`` for every lookup.
CCD_CIF_ENV = "OFOLDX_CCD_CIF_FILE"
BondRecord = tuple[str, str, str]
CifBlock = Mapping[str, Sequence[str]]


@dataclass(frozen=True, slots=True)
class ReferenceComponent:
    r"""
    Reference-conformer geometry and bond graph for one CCD component.

    Attributes:
        atom_map: Mapping from atom name to reference conformer index.
        coords: ``[N_atom, 3]`` ideal/reference coords in Angstrom.
        mask: ``[N_atom]`` validity mask for reference coords.
        charge: ``[N_atom]`` formal charges.
        bonds: Covalent bonds as ``(atom_name, atom_name, rdkit_bond_type)`` records.
        elements: Element symbols aligned to reference conformer indices.

    Examples:
        >>> import numpy as np
        >>> component = ReferenceComponent(
        ...     atom_map={"C1": 0},
        ...     coords=np.zeros((1, 3), dtype=np.float32),
        ...     mask=np.ones(1, dtype=bool),
        ...     charge=np.zeros(1, dtype=np.float32),
        ...     bonds=(),
        ...     elements=("C",),
        ... )
        >>> component.atom_map["C1"], component.coords.shape, component.elements
        (0, (1, 3), ('C',))
    """

    atom_map: Mapping[str, int]  # atom name -> index into the conformer
    coords: np.ndarray  # [N, 3] ideal/reference conformer coords (Angstrom)
    mask: np.ndarray  # [N] bool: the reference coordinate is valid
    charge: np.ndarray  # [N] per-atom formal charge
    bonds: tuple[BondRecord, ...]  # covalent bonds as (atom_name, atom_name, RDKit bond type) records
    elements: tuple[str, ...] = ()  # per-atom element symbol (aligned to the conformer index)


class CcdProvider(Protocol):
    r"""
    Structural interface for a CCD lookup: map a residue / component name to its reference component.

    Any object exposing a ``get(component_name)`` method that returns a
    [`ReferenceComponent`][ofoldx.data.ccd.ReferenceComponent] (or ``None`` for an unknown code)
    satisfies this protocol. Both the pickle-backed
    [`RdkitPickleCcdProvider`][ofoldx.data.ccd.RdkitPickleCcdProvider] and the mmCIF-backed
    [`RdkitCifCcdProvider`][ofoldx.data.ccd.RdkitCifCcdProvider] conform, so a test can
    substitute a tiny in-memory dictionary wrapper without touching the featurization code.
    """

    def get(self, component_name: str) -> ReferenceComponent | None:
        r"""
        Return the reference component for a CCD code.

        Args:
            component_name: Residue or CCD component name.

        Returns:
            Reference component, or ``None`` when the component is unknown.

        Examples:
            >>> provider.get("ALA")  # doctest: +SKIP
            ReferenceComponent(...)
        """

        ...


def _reference_component(
    mol: Chem.Mol, atom_map: Mapping[str, int], coords: np.ndarray, mask: np.ndarray
) -> ReferenceComponent:
    r"""
    Assemble a reference component from an RDKit ``Mol`` and its already-extracted conformer geometry.

    Source-agnostic tail shared by the pickle and CIF providers: given a molecule, its ``atom_name`` ->
    index map, and the reference coordinates/validity mask, it reads per-atom formal charges and element
    symbols from the molecule and materializes the covalent bond graph as ``(name, name, bond_type)``
    records (dropping bonds to atoms outside ``atom_map``).

    Args:
        mol: RDKit ``Mol`` whose atoms and bonds define charges, elements, and the bond graph.
        atom_map: Mapping from atom name to its conformer index.
        coords: ``[N_atom, 3]`` reference coordinates.
        mask: ``[N_atom]`` validity mask for the reference coordinates.

    Returns:
        Reference conformer geometry and bond graph.
    """

    inverse = {index: name for name, index in atom_map.items()}
    charge = np.asarray([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.float32)
    elements = tuple(atom.GetSymbol().upper() for atom in mol.GetAtoms())
    bonds: tuple[BondRecord, ...] = tuple(
        (
            inverse[bond.GetBeginAtomIdx()],
            inverse[bond.GetEndAtomIdx()],
            str(bond.GetBondType()).upper(),
        )
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in inverse and bond.GetEndAtomIdx() in inverse
    )
    return ReferenceComponent(
        atom_map=atom_map, coords=coords, mask=mask, charge=charge, bonds=bonds, elements=elements
    )


def _component_from_mol(mol: object | None) -> ReferenceComponent | None:
    r"""
    Convert a prebaked RDKit ``Mol`` into a reference component using its baked-in reference conformer.

    Reads the ``atom_map`` (name -> index), ``ref_conf_id`` (reference conformer), and ``ref_mask``
    attributes the pickle bakes onto each molecule, keeping the reference coordinates byte-exact.

    Args:
        mol: RDKit ``Mol`` from the backing pickle, or ``None`` for a missing code.

    Returns:
        Reference conformer geometry and bond graph, or ``None`` when ``mol`` is missing or has no atoms.
    """

    if mol is None or mol.GetNumAtoms() == 0:  # type: ignore[attr-defined]
        return None
    atom_map = dict(mol.atom_map)  # type: ignore[attr-defined]
    coords = np.asarray(
        mol.GetConformer(mol.ref_conf_id).GetPositions(),  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    mask = np.asarray(mol.ref_mask).astype(np.bool_)  # type: ignore[attr-defined]
    return _reference_component(mol, atom_map, coords, mask)  # type: ignore[arg-type]


def reference_component_from_mol(mol: Chem.Mol) -> ReferenceComponent | None:
    r"""
    Convert an RDKit ``Mol`` (with ``atom_name`` properties and a conformer) into a reference component.

    Args:
        mol: Molecule carrying per-atom ``atom_name`` properties and at least one conformer.

    Returns:
        Reference conformer geometry and bond graph, or ``None`` when the molecule is empty.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        >>> for index, atom in enumerate(mol.GetAtoms()):
        ...     atom.SetProp("atom_name", f"{atom.GetSymbol()}{index}")
        >>> component = reference_component_from_mol(mol)
        >>> sorted(component.atom_map)
        ['C0', 'H2', 'H3', 'H4', 'H5', 'O1']
    """

    if mol.GetNumAtoms() == 0:
        return None
    atom_map = {atom.GetProp("atom_name"): atom.GetIdx() for atom in mol.GetAtoms() if atom.HasProp("atom_name")}
    if mol.GetNumConformers():
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float32)
        mask = np.ones(mol.GetNumAtoms(), dtype=bool)
    else:
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    return _reference_component(mol, atom_map, coords, mask)


class RdkitPickleCcdProvider:
    r"""
    CCD provider backed by the prebaked RDKit-Mol pickle.

    The pickle maps each CCD code to an RDKit ``Mol`` carrying ``atom_map`` (name -> index),
    ``ref_conf_id`` (the reference conformer), and ``ref_mask`` (valid-coordinate flags). Loading is
    lazy and tolerant: a missing file yields an empty store (every lookup returns ``None``) so the
    processor degrades gracefully rather than failing where the artifact is unavailable.

    Args:
        path: Optional path to the prebaked RDKit-Mol pickle. Defaults to ``OFOLDX_CCD_RDKIT_MOL_FILE`` when set.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or os.environ.get(CCD_RDKIT_MOL_ENV, DEFAULT_CCD_RDKIT_MOL_FILE)
        self._mols: dict[str, object] | None = None
        self._cache: dict[str, ReferenceComponent | None] = {}

    def _load(self) -> dict[str, object]:
        r"""
        Lazily read and memoize the backing pickle, tolerating a missing file with an empty store.

        Returns:
            Mapping from CCD code to its prebaked RDKit ``Mol`` (empty when the pickle is absent).
        """

        if self._mols is None:
            if not self._path:
                logger.warning("CCD RDKit-Mol pickle path unset; reference features disabled")
                self._mols = {}
                return self._mols
            try:
                with open(self._path, "rb") as handle:
                    self._mols = pickle.load(handle)
            except FileNotFoundError:
                logger.warning("CCD RDKit-Mol pickle not found at %s; reference features disabled", self._path)
                self._mols = {}
        return self._mols

    def get(self, component_name: str) -> ReferenceComponent | None:
        r"""
        Return a cached reference component for a CCD code.

        Args:
            component_name: Residue or CCD component name.

        Returns:
            Reference component, or ``None`` when the component is absent from the backing pickle.

        Examples:
            >>> provider = RdkitPickleCcdProvider("components.rdkit_mol.pkl")  # doctest: +SKIP
            >>> provider.get("GLY").coords.shape  # doctest: +SKIP
            (4, 3)
        """

        if component_name not in self._cache:
            self._cache[component_name] = _component_from_mol(self._load().get(component_name))
        return self._cache[component_name]


@functools.lru_cache(maxsize=1)
def default_ccd_provider() -> RdkitPickleCcdProvider:
    r"""
    Return the process-wide default provider (lazily backed by the configured pickle).

    The provider is constructed once and memoized; the backing pickle is only read on the first
    [`RdkitPickleCcdProvider.get`][] call, so obtaining the provider is cheap and side-effect free.

    Returns:
        Cached default CCD provider.

    Examples:
        >>> provider = default_ccd_provider()
        >>> isinstance(provider, RdkitPickleCcdProvider)
        True
        >>> provider is default_ccd_provider()
        True
    """

    return RdkitPickleCcdProvider()


def heavy_atom_names(component: ReferenceComponent, *, mol_type: str | None, keep_leaving: bool) -> list[str]:
    r"""
    Return a component's heavy-atom names in reference-conformer order.

    Hydrogens (and deuteriums) are always dropped. Internal-residue leaving atoms (protein ``OXT``,
    nucleic ``OP3``) are dropped unless ``keep_leaving`` is set (a chain terminus or a non-polymer
    component keeps them). This is the shared heavy-atom selection rule the family JSON parsers use to
    materialize sequence-only chains from CCD reference conformers.

    Args:
        component: Reference component whose atoms are selected.
        mol_type: Molecule-type name (``"protein"`` / ``"rna"`` / ``"dna"``) used to pick the leaving-atom
            set, or ``None`` to keep every heavy atom.
        keep_leaving: Whether to keep the residue's leaving atom (terminal or non-polymer residues do).

    Returns:
        Heavy-atom names ordered by their reference-conformer index.

    Examples:
        >>> import numpy as np
        >>> component = ReferenceComponent(
        ...     atom_map={"N": 0, "CA": 1, "C": 2, "O": 3, "OXT": 4, "H": 5},
        ...     coords=np.zeros((6, 3), dtype=np.float32),
        ...     mask=np.ones(6, dtype=bool),
        ...     charge=np.zeros(6, dtype=np.float32),
        ...     bonds=(),
        ...     elements=("N", "C", "C", "O", "O", "H"),
        ... )
        >>> heavy_atom_names(component, mol_type="protein", keep_leaving=False)
        ['N', 'CA', 'C', 'O']
        >>> heavy_atom_names(component, mol_type="protein", keep_leaving=True)
        ['N', 'CA', 'C', 'O', 'OXT']
    """

    leaving = frozenset() if keep_leaving or mol_type is None else LEAVING_ATOMS.get(mol_type, frozenset())
    return [
        name
        for name in sorted(component.atom_map, key=lambda name: component.atom_map[name])
        if component.elements[component.atom_map[name]].upper() not in HYDROGEN_ELEMENTS and name not in leaving
    ]


# --------------------------------------------------------------------------------------------------------------------
# CIF -> Mol construction
#
# Builds an RDKit ``Mol`` for a CCD code directly from its mmCIF data block -- atoms, bonds, formal charges, atom
# names, and the deposited ideal conformer -- and converts it into a ``ReferenceComponent``. When a component ships
# no ideal coordinates, an RDKit ETKDGv3 conformer is generated on the fly. This is a faithful port of upstream
# AlphaFold3 ``data/tools/rdkit_utils.py`` (``mol_from_ccd_cif``, ``parse_atom_data``, ``parse_bond_data``,
# ``get_random_conformer``, ``assign_atom_names_from_graph``); the only substitution is the mmCIF reader (upstream's
# C++ ``cif_dict`` -> ``biotite``), so the parsed field dictionary the port operates on is unchanged. The heavy CCD
# dictionary (``components.cif``, ~430 MB) is external reference data and is not shipped.
# --------------------------------------------------------------------------------------------------------------------

# CCD ``_chem_comp_bond.value_order`` -> RDKit bond type (upstream ``_RDKIT_MMCIF_TO_BOND_TYPE``).
_MMCIF_TO_BOND_TYPE: Mapping[str, Chem.BondType] = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
}


class MolFromCcdError(Exception):
    r"""
    Raised when an RDKit ``Mol`` cannot be built from a CCD mmCIF block.

    Wraps the underlying parsing / sanitization failure (a missing required column, an unknown bond
    order, or an RDKit kekulization / valence error) so callers can distinguish an unbuildable
    component from other exceptions and fall back to disabling reference features for that code.
    """


def parse_atom_data(block: CifBlock, force_parse: bool) -> tuple[list[str], list[str], list[int], list[str]]:
    r"""
    Read atom names, element types, formal charges, and leaving-atom flags from a CCD block.

    Args:
        block: Mapping from CCD mmCIF column name to its per-atom string values.
        force_parse: If ``True``, replace missing charges with ``0`` and deuterium with hydrogen.

    Returns:
        Tuple of atom names, capitalized element symbols, integer charges, and leaving-atom flags.

    Examples:
        >>> block = {
        ...     "_chem_comp_atom.type_symbol": ["C", "O"],
        ...     "_chem_comp_atom.atom_id": ["C1", "O1"],
        ...     "_chem_comp_atom.charge": ["0", "?"],
        ... }
        >>> parse_atom_data(block, force_parse=True)
        (['C1', 'O1'], ['C', 'O'], [0, 0], ['?', '?'])
    """

    atom_types = [t.capitalize() for t in block["_chem_comp_atom.type_symbol"]]
    atom_names = list(block["_chem_comp_atom.atom_id"])
    atom_charges = list(block["_chem_comp_atom.charge"])
    leaving_flags = list(block.get("_chem_comp_atom.pdbx_leaving_atom_flag", ["?"] * len(atom_names)))
    if force_parse:
        atom_charges = [charge if charge not in {"?", "."} else "0" for charge in atom_charges]
        atom_types = [element if element != "D" else "H" for element in atom_types]
    charges = [int(charge) if charge not in {"?", "."} else 0 for charge in atom_charges]
    return atom_names, atom_types, charges, leaving_flags


def parse_bond_data(block: CifBlock, force_parse: bool) -> tuple[list[str], list[str], list[Chem.BondType], list[bool]]:
    r"""
    Read bond endpoints, orders, and aromatic flags from a CCD block.

    Args:
        block: Mapping from CCD mmCIF column name to its per-bond string values.
        force_parse: If ``True``, treat missing aromatic flags as non-aromatic.

    Returns:
        Tuple of begin-atom names, end-atom names, RDKit bond types, and aromatic flags.

    Examples:
        >>> block = {
        ...     "_chem_comp_bond.atom_id_1": ["C1"],
        ...     "_chem_comp_bond.atom_id_2": ["O1"],
        ...     "_chem_comp_bond.value_order": ["DOUB"],
        ... }
        >>> begins, ends, bond_types, aromatic = parse_bond_data(block, force_parse=True)
        >>> begins, ends, [str(bond_type) for bond_type in bond_types], aromatic
        (['C1'], ['O1'], ['DOUBLE'], [False])
    """

    begins = list(block.get("_chem_comp_bond.atom_id_1", []))
    ends = list(block.get("_chem_comp_bond.atom_id_2", []))
    orders = list(block.get("_chem_comp_bond.value_order", []))
    bond_types = [_MMCIF_TO_BOND_TYPE[order] for order in orders]
    aromatic_flags = block.get("_chem_comp_bond.pdbx_aromatic_flag")
    if aromatic_flags is None:
        if not force_parse:
            raise KeyError("_chem_comp_bond.pdbx_aromatic_flag")
        is_aromatic = [False] * len(begins)
    else:
        is_aromatic = [flag == "Y" for flag in aromatic_flags]
    return begins, ends, bond_types, is_aromatic


def _ideal_conformer(block: CifBlock) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Return the deposited ideal coordinates and a per-atom validity mask from a CCD block.

    Coordinates are ``NaN``/``?``-tolerant: an unreadable position yields a zero coordinate with a
    ``False`` mask entry, so downstream masking mirrors the pickle provider's behavior.

    Args:
        block: Parsed CCD mmCIF data block for one component.

    Returns:
        Tuple of ``[N_atom, 3]`` ideal coordinates and an ``[N_atom]`` validity mask.
    """

    x = block.get("_chem_comp_atom.pdbx_model_Cartn_x_ideal", [])
    y = block.get("_chem_comp_atom.pdbx_model_Cartn_y_ideal", [])
    z = block.get("_chem_comp_atom.pdbx_model_Cartn_z_ideal", [])
    coords = np.zeros((len(x), 3), dtype=np.float32)
    mask = np.zeros(len(x), dtype=bool)
    for index, position in enumerate(zip(x, y, z)):
        try:
            coords[index] = [float(value) for value in position]
            mask[index] = True
        except ValueError:
            continue
    return coords, mask


def mol_from_ccd_block(
    block: CifBlock,
    *,
    force_parse: bool = False,
    remove_hydrogens: bool = True,
) -> Chem.Mol:
    r"""
    Build an RDKit ``Mol`` from a CCD mmCIF data block.

    Atoms carry the CCD ``atom_name`` as a property, bonds carry orders/aromaticity, and the deposited
    ideal conformer (when present) is attached and used to assign stereochemistry. Non-essential
    hydrogens are removed by default (stereochemistry-defining hydrogens are retained by RDKit).

    Args:
        block: Parsed CCD mmCIF data block for one component.
        force_parse: If ``True``, tolerate missing charges/aromatic flags/conformers.
        remove_hydrogens: If ``True``, drop non-essential hydrogens after sanitization.

    Returns:
        Sanitized RDKit ``Mol`` with per-atom ``atom_name`` properties.

    Raises:
        MolFromCcdError: If the block cannot be interpreted as a molecule.

    Examples:
        >>> block = {
        ...     "_chem_comp_atom.type_symbol": ["C", "O"],
        ...     "_chem_comp_atom.atom_id": ["C1", "O1"],
        ...     "_chem_comp_atom.charge": ["0", "0"],
        ...     "_chem_comp_bond.atom_id_1": ["C1"],
        ...     "_chem_comp_bond.atom_id_2": ["O1"],
        ...     "_chem_comp_bond.value_order": ["DOUB"],
        ...     "_chem_comp_bond.pdbx_aromatic_flag": ["N"],
        ...     "_chem_comp_atom.pdbx_model_Cartn_x_ideal": ["0.0", "1.2"],
        ...     "_chem_comp_atom.pdbx_model_Cartn_y_ideal": ["0.0", "0.0"],
        ...     "_chem_comp_atom.pdbx_model_Cartn_z_ideal": ["0.0", "0.0"],
        ... }
        >>> mol = mol_from_ccd_block(block, force_parse=True)
        >>> [atom.GetProp("atom_name") for atom in mol.GetAtoms()]
        ['C1', 'O1']
    """

    try:
        atom_names, atom_types, atom_charges, leaving_flags = parse_atom_data(block, force_parse)
        bond_begins, bond_ends, bond_orders, bond_aromatics = parse_bond_data(block, force_parse)
    except (KeyError, ValueError) as error:
        raise MolFromCcdError(str(error)) from error

    mol = Chem.RWMol()
    name_to_index = {name: index for index, name in enumerate(atom_names)}
    for name, element, charge, leaving in zip(atom_names, atom_types, atom_charges, leaving_flags):
        atom = Chem.Atom("*" if element == "X" else element)
        atom.SetNoImplicit(True)
        atom.SetProp("atom_name", name)
        atom.SetProp("atom_leaving_flag", leaving)
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)
    for begin, end, bond_type, is_aromatic in zip(bond_begins, bond_ends, bond_orders, bond_aromatics):
        if begin not in name_to_index or end not in name_to_index:
            continue
        bond_index = mol.AddBond(name_to_index[begin], name_to_index[end], bond_type)
        mol.GetBondWithIdx(bond_index - 1).SetIsAromatic(is_aromatic)

    coords, coord_mask = _ideal_conformer(block)
    mol.UpdatePropertyCache(strict=False)
    if coord_mask.all() and len(coords) == mol.GetNumAtoms():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for index, position in enumerate(coords):
            conformer.SetAtomPosition(index, [float(value) for value in position])
        mol.AddConformer(conformer)
    try:
        Chem.SanitizeMol(mol)
        if mol.GetNumConformers():
            Chem.AssignStereochemistryFrom3D(mol)
        if remove_hydrogens:
            mol = Chem.RemoveHs(mol)
    except (ValueError, Chem.KekulizeException, Chem.AtomValenceException) as error:
        if not force_parse:
            raise MolFromCcdError(str(error)) from error
    return mol


def ensure_conformer(mol: Chem.Mol, random_seed: int = 0, max_iterations: int | None = None) -> Chem.Mol:
    r"""
    Return a copy of ``mol`` that carries at least one 3D conformer.

    When ``mol`` has no conformer (e.g. a component with no deposited ideal coordinates), an RDKit
    ETKDGv3 conformer is embedded, matching upstream ``get_random_conformer``. If embedding fails the
    molecule is returned unchanged (the caller masks out its reference coordinates).

    Args:
        mol: Input molecule.
        random_seed: ETKDG random seed for reproducibility.
        max_iterations: Optional override for the ETKDG iteration budget.

    Returns:
        Molecule with an embedded conformer when one could be produced.

    Examples:
        >>> from rdkit import Chem
        >>> mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        >>> ensure_conformer(mol, random_seed=0).GetNumConformers()
        1
    """

    if mol.GetNumConformers():
        return mol
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if max_iterations is not None:
        params.maxIterations = max_iterations
    mol_copy = Chem.Mol(mol)
    try:
        if AllChem.EmbedMolecule(mol_copy, params) == 0:
            return mol_copy
    except (ValueError, RuntimeError):
        logger.warning("failed to embed conformer for component")
    return mol


def read_ccd_blocks(path: str | os.PathLike[str]) -> dict[str, CifBlock]:
    r"""
    Read every component data block from a CCD ``components.cif`` file.

    Args:
        path: Path to a CCD mmCIF file (one ``data_<code>`` block per component).

    Returns:
        Mapping from CCD code to its parsed mmCIF field dictionary.

    Examples:
        >>> blocks = read_ccd_blocks("components.cif")  # doctest: +SKIP
        >>> blocks["ALA"]["_chem_comp_atom.atom_id"][:3]  # doctest: +SKIP
        ['N', 'CA', 'C']
    """

    from biotite.structure.io.pdbx import CIFFile

    cif = CIFFile.read(str(path))
    blocks: dict[str, CifBlock] = {}
    for code, block in cif.items():
        fields: dict[str, list[str]] = {}
        for category_name, category in block.items():
            for column_name, column in category.items():
                fields[f"_{category_name}.{column_name}"] = list(column.as_array(str))
        blocks[code] = fields
    return blocks


class RdkitCifCcdProvider:
    r"""
    CCD provider that builds reference components from a CCD ``components.cif`` via RDKit.

    Components are parsed lazily on first access and cached. A missing CCD file makes every lookup
    return ``None`` (features disabled) rather than raising, mirroring the pickle-backed provider.

    Args:
        path: Path to a CCD mmCIF file. Defaults to ``OFOLDX_CCD_CIF_FILE`` when set.
        embed_missing_conformers: If ``True``, embed an ETKDG conformer for components lacking deposited
            ideal coordinates.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, embed_missing_conformers: bool = True) -> None:
        self._path = str(path) if path is not None else os.environ.get(CCD_CIF_ENV, "")
        self._embed_missing_conformers = embed_missing_conformers
        self._blocks: dict[str, CifBlock] | None = None
        self._cache: dict[str, ReferenceComponent | None] = {}

    def _load(self) -> dict[str, CifBlock]:
        r"""
        Lazily read and memoize the CCD mmCIF blocks, tolerating a missing file with an empty store.

        Returns:
            Mapping from CCD code to its parsed mmCIF field dictionary (empty when the file is absent).
        """

        if self._blocks is None:
            if not self._path or not os.path.exists(self._path):
                logger.warning("CCD CIF not found at %r; RDKit reference features disabled", self._path)
                self._blocks = {}
            else:
                self._blocks = read_ccd_blocks(self._path)
        return self._blocks

    def get(self, component_name: str) -> ReferenceComponent | None:
        r"""
        Return the reference component for a CCD code, building it via RDKit on first access.

        Args:
            component_name: Residue or CCD component name.

        Returns:
            Reference component, or ``None`` when the component is absent or unparsable.

        Examples:
            >>> provider = RdkitCifCcdProvider("components.cif")  # doctest: +SKIP
            >>> provider.get("ALA").coords.shape  # doctest: +SKIP
            (5, 3)
        """

        if component_name not in self._cache:
            self._cache[component_name] = self._build(component_name)
        return self._cache[component_name]

    def _build(self, component_name: str) -> ReferenceComponent | None:
        r"""
        Build a reference component for a CCD code from its mmCIF block, embedding a conformer if asked.

        Args:
            component_name: Residue or CCD component name.

        Returns:
            Reference component, or ``None`` when the code is absent or its block cannot be built.
        """

        block = self._load().get(component_name)
        if block is None:
            return None
        try:
            mol = mol_from_ccd_block(block, force_parse=True)
        except MolFromCcdError as error:
            logger.warning("failed to build mol for %s: %s", component_name, error)
            return None
        if self._embed_missing_conformers:
            mol = ensure_conformer(mol)
        return reference_component_from_mol(mol)
