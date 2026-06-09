r"""
Chemical Component Dictionary (CCD) reference lookup.

Provides per-component reference-conformer positions, validity masks, formal charges, and the covalent bond graph
used to build token-bond features. The production provider consumes the prebaked RDKit-Mol pickle used by the
checkpoint conversion path, so reference coords remain byte-exact when the same artifact is configured. Tests
inject a small in-memory provider instead, so the committed suite stays self-contained.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .residue_constants import HYDROGEN_ELEMENTS, LEAVING_ATOMS

logger = logging.getLogger(__name__)

CCD_RDKIT_MOL_ENV = "OFOLDX_CCD_RDKIT_MOL_FILE"
DEFAULT_CCD_RDKIT_MOL_FILE = "/data/node1/mini_dataset/components.v20240608.cif.rdkit_mol.pkl"
# Path to a CCD ``components.cif`` for the RDKit provider in ``rdkit_ccd`` (external reference data,
# not shipped); unset means that provider disables itself and returns ``None`` for every lookup.
CCD_CIF_ENV = "OFOLDX_CCD_CIF_FILE"
BondRecord = tuple[str, str, str]


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
    [`RdkitCifCcdProvider`][ofoldx.data.rdkit_ccd.RdkitCifCcdProvider] conform, so a test can
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


class RdkitPickleCcdProvider:
    r"""
    CCD provider backed by the prebaked RDKit-Mol pickle.

    The pickle maps each CCD code to an RDKit ``Mol`` carrying ``atom_map`` (name -> index),
    ``ref_conf_id`` (the reference conformer), and ``ref_mask`` (valid-coordinate flags). Loading is
    lazy and tolerant: a missing file yields an empty store (every lookup returns ``None``) so the
    processor degrades gracefully rather than failing where the artifact is unavailable.

    Args:
        path: Optional path to the prebaked RDKit-Mol pickle. Defaults to ``OFOLDX_CCD_RDKIT_MOL_FILE`` or the
            project default path.
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
    inverse = {index: name for name, index in atom_map.items()}
    coords = np.asarray(
        mol.GetConformer(mol.ref_conf_id).GetPositions(),  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    mask = np.asarray(mol.ref_mask).astype(np.bool_)  # type: ignore[attr-defined]
    charge = np.asarray(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()],  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    elements = tuple(atom.GetSymbol().upper() for atom in mol.GetAtoms())  # type: ignore[attr-defined]
    bonds = tuple(
        (
            inverse[bond.GetBeginAtomIdx()],  # type: ignore[attr-defined]
            inverse[bond.GetEndAtomIdx()],  # type: ignore[attr-defined]
            str(bond.GetBondType()).upper(),  # type: ignore[attr-defined]
        )
        for bond in mol.GetBonds()  # type: ignore[attr-defined]
        if bond.GetBeginAtomIdx() in inverse and bond.GetEndAtomIdx() in inverse
    )
    return ReferenceComponent(
        atom_map=atom_map, coords=coords, mask=mask, charge=charge, bonds=bonds, elements=elements
    )


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
