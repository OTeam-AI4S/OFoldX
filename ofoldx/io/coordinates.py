"""Coordinate projection of declared CIF polymers, without chemical graph construction."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..data.coordinates import ChainCoordinates, PolymerCoordinates
from .polymer import POLYMER_TYPES, resolve_monomers
from .selection import (
    _ALTLOC_POLICIES,
    _polymer_declarations,
    _select_altloc,
    _select_mmcif_polymer_microheterogeneity,
)
from .source import StructureView, _require_cif_source


def read_polymer_coordinates(
    view: StructureView,
    *,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_hydrogens: bool = True,
) -> PolymerCoordinates:
    r"""Select deposited coordinate observations against declared sequence positions.

    CIF is required for this projection: PDB observations remain available through
    source.view().rows without guessing a SEQRES-to-author-number correspondence.
    Unknown polymer types and ambiguous unobserved monomers raise explicitly.

    Parent component declarations are retained as tuples; consumers choose how to
    interpret multiple parents. Missing sequence/type evidence is a data error.

    Args:
        view: Selected CIF model and label chains, without an assembly request.
        altloc: Residue-level alternate-location policy, shared with structure reading.
        include_hydrogens: Whether to retain observed hydrogen atoms, including isotopes.

    Returns:
        Coordinates keyed by label chain ID, retaining declared sequence positions
            without observed atoms and the complete parent list at each position.
            Atom positions index the declared sequence, independent of author numbering.

    Raises:
        ofoldx.io.UnsupportedFormatError: If the source is not CIF.
        KeyError: If an explicitly selected label chain is not declared.
        ValueError: If options are invalid, required evidence is missing or
            contradictory, or a polymer type or unobserved monomer cannot be resolved.

    Examples:
        >>> from .source import parse_source
        >>> read_polymer_coordinates(parse_source("END\n", format="pdb").view())
        Traceback (most recent call last):
            ...
        ofoldx.io.source.UnsupportedFormatError: this operation supports CIF sources only
    """
    source = _require_cif_source(view.source)
    if view.assembly_id is not None:
        raise ValueError("polymer coordinate projection uses deposited chains")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    declarations = _polymer_declarations(view)
    source.author_chains(view.model_index)
    monomers = declarations.polymer_monomers or {}
    rows = _select_altloc(_select_mmcif_polymer_microheterogeneity(view.rows, monomers, altloc), altloc)
    if not include_hydrogens:
        rows = rows[rows.element != "H"]
    parents = source.component_parents
    chains = {}
    for chain, entity in declarations.asym_entities.items():
        if entity not in declarations.polymer_types:
            continue
        declared = monomers[entity]
        kind = POLYMER_TYPES[declarations.polymer_types[entity].casefold()]
        atoms = rows[rows.chain_id == chain]
        if any(value is None for value in atoms.label_seq_id):
            raise ValueError("observed mmCIF polymer residue lacks label_seq_id")
        if any(value != entity for value in atoms.entity_id):
            raise ValueError("observed polymer entity contradicts its declared chain")
        positions = np.asarray(atoms.label_seq_id, dtype=np.int64) - 1
        observed: dict[int, str] = {}
        identities: dict[int, tuple[str, str | None, str | None]] = {}
        for index, identifier, author, insertion in zip(
            positions.tolist(), atoms.component_id, atoms.auth_residue_id, atoms.insertion_code, strict=True
        ):
            identity = identifier, author, insertion
            if index in identities and identities[index] != identity:
                raise ValueError(f"multiple observed mmCIF residues map to label_seq_id {index + 1}")
            identities[index] = identity
            observed[index] = identifier
        identifiers = resolve_monomers(entity, declared, observed)
        chains[chain] = ChainCoordinates(
            polymer_type=kind,
            component_ids=identifiers,
            parent_component_ids=tuple(parents.get(identifier, ()) for identifier in identifiers),
            atom_names=tuple(atoms.atom_name),
            component_index=positions,
            coordinates=atoms.coordinates,
            coordinate_mask=atoms.coordinate_present,
        )
    return PolymerCoordinates(chains)
