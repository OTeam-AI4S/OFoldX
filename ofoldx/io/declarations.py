"""Typed interpretation of original mmCIF declarations, without chemical completion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from ..data.metadata import (
    RESOLUTION_KIND_EM_RECONSTRUCTION,
    RESOLUTION_KIND_REFINEMENT,
    RESOLUTION_KIND_REFLECTIONS,
)
from ..data.references import AtomSite
from .cif import CifBlock, CifCategory, column_integers

_MMCIF_RESOLUTION_FIELDS = (
    ("refine", "ls_d_res_high", RESOLUTION_KIND_REFINEMENT),
    ("reflns", "d_resolution_high", RESOLUTION_KIND_REFLECTIONS),
    ("em_3d_reconstruction", "resolution", RESOLUTION_KIND_EM_RECONSTRUCTION),
)


def _cif_column(category: CifCategory, name: str) -> tuple[str | None, ...]:
    return category.strings(name)


def _cif_string(value: object | None) -> str | None:
    return None if value is None else str(value)


def read_component_parents(category: CifCategory | None) -> dict[str, tuple[str, ...]]:
    r"""Read the declared parent components without choosing a canonical monomer.

    A component may derive from several parents. The tuple retains their source
    order; an absent declaration adds no relationship. Lexical null states remain
    available on the original category.

    Args:
        category: The ``chem_comp`` category, if present.

    Returns:
        Component identifiers mapped to their complete parent lists.

    Raises:
        ValueError: A declared relationship has no component ID, an empty parent,
            or conflicts with another declaration for the same component.

    Examples:
        >>> category = CifCategory({"id": ["MSE", "CRO"],
        ...                         "mon_nstd_parent_comp_id": ["MET", "SER,TYR,GLY"]})
        >>> read_component_parents(category)
        {'MSE': ('MET',), 'CRO': ('SER', 'TYR', 'GLY')}
    """
    if category is None:
        return {}
    parents: dict[str, tuple[str, ...]] = {}
    for identifier, value in zip(category.strings("id"), category.strings("mon_nstd_parent_comp_id"), strict=True):
        if value is None:
            continue
        if not identifier:
            raise ValueError("mmCIF parent relationship requires a component ID")
        components = tuple(part.strip() for part in value.split(","))
        if not all(components):
            raise ValueError(f"mmCIF component {identifier!r} has an empty parent identifier")
        if identifier in parents and parents[identifier] != components:
            raise ValueError(f"mmCIF component {identifier!r} has conflicting parent declarations")
        parents[identifier] = components
    return parents


def read_mmcif_resolutions(block: CifBlock) -> tuple[tuple[float, int], ...]:
    r"""Read every typed experimental-resolution value, in angstroms."""

    resolutions: list[tuple[float, int]] = []
    for category_name, column_name, kind in _MMCIF_RESOLUTION_FIELDS:
        if category_name not in block or column_name not in block[category_name]:
            continue
        column = block[category_name][column_name]
        for resolution in column.as_array(np.float64, np.nan)[column.present]:
            if not math.isfinite(resolution) or resolution <= 0.0:
                raise ValueError(f"invalid mmCIF experimental resolution {resolution!r}")
            resolutions.append((float(resolution), kind))
    return tuple(resolutions)


def _read_mmcif_atom_provenance(block: CifBlock) -> frozenset[AtomSite] | None:
    r"""Read explicit formal-charge source markers preserved by the OFoldX writer."""

    marker = block.get("ofoldx_provenance")
    if marker is None:
        return None
    if _cif_column(marker, "formal_charge_source_complete") != ("Y",):
        raise ValueError("mmCIF _ofoldx_provenance.formal_charge_source_complete must be one Y value")
    category = block.get("ofoldx_source_formal_charge")
    if category is None:
        return frozenset()
    required = {"label_asym_id", "component_index", "label_atom_id"}
    if not required.issubset(category):
        raise ValueError("mmCIF _ofoldx_source_formal_charge is incomplete")

    provenance: set[AtomSite] = set()
    for asym_id, component_index, atom_id in zip(
        _cif_column(category, "label_asym_id"),
        column_integers(category, "component_index"),
        _cif_column(category, "label_atom_id"),
        strict=True,
    ):
        index = component_index
        if asym_id is None or index is None or index < 0 or atom_id is None:
            raise ValueError("mmCIF _ofoldx_source_formal_charge has an invalid atom identity")
        site = AtomSite(asym_id, index, atom_id)
        if site in provenance:
            raise ValueError(f"mmCIF _ofoldx_source_formal_charge repeats {site!r}")
        provenance.add(site)
    return frozenset(provenance)


def _cif_row_value(category: CifCategory, name: str, index: int) -> str | None:
    return category.value(name, index)


PolymerMonomers: TypeAlias = Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]]
AuthorNumbering: TypeAlias = Mapping[tuple[str, int] | tuple[str, int, str], tuple[str | None, str | None]]
NonpolymerOccurrences: TypeAlias = Mapping[str, tuple[tuple[str, str, str, str | None, str | None, str | None], ...]]


@dataclass(frozen=True, kw_only=True)
class EntityDeclarations:
    """Source entity and numbering facts, before component or graph completion.

    None for polymer_monomers means the category is absent; an empty mapping
    means it was present without positions. All mappings are owned snapshots.
    Source categories remain available for finer lexical null distinctions.
    """

    entity_types: Mapping[str, str]
    polymer_monomers: PolymerMonomers | None
    polymer_types: Mapping[str, str]
    author_numbering: AuthorNumbering
    asym_entities: Mapping[str, str]
    nonpolymer_components: Mapping[str, str]
    nonpolymer_occurrences: NonpolymerOccurrences

    def __post_init__(self) -> None:
        for name in ("entity_types", "polymer_types", "asym_entities", "nonpolymer_components"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.polymer_monomers is not None:
            monomers = {
                entity: tuple((number, tuple(candidates)) for number, candidates in positions)
                for entity, positions in self.polymer_monomers.items()
            }
            object.__setattr__(self, "polymer_monomers", MappingProxyType(monomers))
        object.__setattr__(
            self,
            "author_numbering",
            MappingProxyType({key: tuple(value) for key, value in self.author_numbering.items()}),
        )
        object.__setattr__(
            self,
            "nonpolymer_occurrences",
            MappingProxyType(
                {asym: tuple(tuple(row) for row in rows) for asym, rows in self.nonpolymer_occurrences.items()}
            ),
        )

    def select_asym_ids(self, asym_ids: Set[str]) -> EntityDeclarations:
        """Select declarations while retaining the referenced entities and numbering."""
        asym_entities = {asym: entity for asym, entity in self.asym_entities.items() if asym in asym_ids}
        entities = set(asym_entities.values())
        return replace(
            self,
            entity_types={key: value for key, value in self.entity_types.items() if key in entities},
            polymer_monomers=(
                None
                if self.polymer_monomers is None
                else {key: value for key, value in self.polymer_monomers.items() if key in entities}
            ),
            polymer_types={key: value for key, value in self.polymer_types.items() if key in entities},
            author_numbering={key: value for key, value in self.author_numbering.items() if key[0] in asym_ids},
            asym_entities=asym_entities,
            nonpolymer_components={key: value for key, value in self.nonpolymer_components.items() if key in entities},
            nonpolymer_occurrences={
                key: value for key, value in self.nonpolymer_occurrences.items() if key in asym_ids
            },
        )


def read_entity_declarations(block: CifBlock) -> EntityDeclarations:
    r"""
    Decode mmCIF entity-type, polymer-sequence, and author-numbering declarations.

    The ``_entity.type`` declaration is the authoritative entity category, and
    ``_entity_poly_seq`` is the ordered canonical monomer sequence per polymer entity,
    including residues unmodeled in the coordinates.

    Args:
        block: Parsed mmCIF block.

    Returns:
        Entity declarations gathered from the source. An absent polymer sequence
            category yields None; other absent categories yield empty mappings.
            Non-polymer occurrence tuples contain NDB identity, entity, monomer,
            author number, insertion code, and author strand in declaration order.
    """

    entity_types: dict[str, str] = {}
    has_entity_declarations = "entity" in block
    if has_entity_declarations:
        category = block["entity"]
        for entity_id, entity_type in zip(_cif_column(category, "id"), _cif_column(category, "type")):
            if entity_id is None or entity_type is None:
                raise ValueError("mmCIF _entity row is missing id or type")
            if entity_id in entity_types:
                raise ValueError(f"mmCIF _entity contains duplicate id {entity_id!r}")
            normalized_type = entity_type.lower()
            if normalized_type not in {"branched", "macrolide", "non-polymer", "polymer", "water"}:
                raise ValueError(f"unsupported mmCIF entity type {entity_type!r}")
            entity_types[entity_id] = normalized_type

    polymer_monomers: dict[str, dict[int, list[str]]] = {}
    polymer_monomer_hetero: dict[tuple[str, int, str], list[str | None]] = {}
    has_polymer_sequences = "entity_poly_seq" in block
    if has_polymer_sequences:
        category = block["entity_poly_seq"]
        for entity_id, number, monomer, hetero in zip(
            _cif_column(category, "entity_id"),
            column_integers(category, "num"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "hetero"),
        ):
            index = number
            if entity_id is None or index is None or monomer is None:
                raise ValueError("mmCIF _entity_poly_seq row is missing entity_id, num, or mon_id")
            if index < 1:
                raise ValueError(f"mmCIF _entity_poly_seq.num must be positive, got {index}")
            candidates = polymer_monomers.setdefault(entity_id, {}).setdefault(index, [])
            if monomer not in candidates:
                candidates.append(monomer)
            polymer_monomer_hetero.setdefault((entity_id, index, monomer), []).append(
                None if hetero is None else hetero.casefold()
            )
        for declared_entity_id, positions in polymer_monomers.items():
            numbers = sorted(positions)
            if numbers != list(range(1, len(numbers) + 1)):
                raise ValueError(f"mmCIF polymer entity {declared_entity_id!r} has a non-contiguous declared sequence")
            for position_number, candidates in positions.items():
                if len(candidates) > 1 and any(
                    flag not in {"y", "yes"}
                    for monomer in candidates
                    for flag in polymer_monomer_hetero[(declared_entity_id, position_number, monomer)]
                ):
                    raise ValueError(
                        f"mmCIF polymer entity {declared_entity_id!r} position {position_number} has conflicting "
                        f"non-heterogeneous monomer declarations {candidates!r}"
                    )

    entity_poly_types: dict[str, str] = {}
    if "entity_poly" in block:
        category = block["entity_poly"]
        for entity_id, poly_type in zip(_cif_column(category, "entity_id"), _cif_column(category, "type")):
            if entity_id is None or poly_type is None:
                raise ValueError("mmCIF _entity_poly row is missing entity_id or type")
            if entity_id in entity_poly_types:
                raise ValueError(f"mmCIF _entity_poly contains duplicate entity_id {entity_id!r}")
            entity_poly_types[entity_id] = poly_type.lower()

    author_numbering: dict[
        tuple[str, int] | tuple[str, int, str],
        tuple[str | None, str | None],
    ] = {}
    if "pdbx_poly_seq_scheme" in block:
        category = block["pdbx_poly_seq_scheme"]
        polymer_numbering: dict[tuple[str, int, str | None], tuple[str | None, str | None]] = {}
        for asym_id, entity_id, seq_id, monomer, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "entity_id"),
            column_integers(category, "seq_id"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            seq = seq_id
            if asym_id is None or seq is None:
                raise ValueError("mmCIF _pdbx_poly_seq_scheme row is missing asym_id or seq_id")
            # ``pdb_seq_num`` is the namespace ``_atom_site.auth_seq_id`` numbers the observed
            # rows in, so it is the one that identifies a residue. ``auth_seq_num`` names the
            # nearest observed residue for an unmodeled position, which repeats its neighbour.
            author_residue_number = _cif_string(pdb_number)
            if author_residue_number is None:
                author_residue_number = _cif_string(auth_number)
            if entity_id is not None and monomer is not None:
                declared_candidates = polymer_monomers.get(entity_id, {}).get(seq)
                if declared_candidates is None or monomer not in declared_candidates:
                    raise ValueError(
                        f"mmCIF _pdbx_poly_seq_scheme monomer {monomer!r} is not declared "
                        f"at entity {entity_id!r} position {seq}"
                    )
            scheme_key = (asym_id, seq, monomer)
            numbering = (author_residue_number, ins_code)
            if scheme_key in polymer_numbering and polymer_numbering[scheme_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_poly_seq_scheme has conflicting numbering for {scheme_key!r}")
            polymer_numbering[scheme_key] = numbering

        for (asym_id, seq, monomer), numbering in polymer_numbering.items():
            if monomer is None:
                author_numbering[(asym_id, seq)] = numbering
            else:
                author_numbering[(asym_id, seq, monomer)] = numbering

    if "pdbx_branch_scheme" in block:
        category = block["pdbx_branch_scheme"]
        for asym_id, number, auth_number, pdb_number, ins_code in zip(
            _cif_column(category, "asym_id"),
            column_integers(category, "num"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
        ):
            component_number = number
            if asym_id is None or component_number is None:
                raise ValueError("mmCIF _pdbx_branch_scheme row is missing asym_id or num")
            author_residue_number = pdb_number or auth_number
            numbering_key = (asym_id, component_number)
            numbering = (author_residue_number, ins_code)
            if numbering_key in author_numbering and author_numbering[numbering_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_branch_scheme has conflicting numbering for {numbering_key!r}")
            author_numbering[numbering_key] = numbering

    nonpoly_occurrences: dict[
        str,
        list[tuple[str, str, str, str | None, str | None, str | None]],
    ] = {}
    if "pdbx_nonpoly_scheme" in block:
        category = block["pdbx_nonpoly_scheme"]
        ordinal_by_asym: dict[str, int] = {}
        occurrence_keys: set[tuple[str, str]] = set()
        for asym_id, entity_id, monomer, ndb_number, auth_number, pdb_number, ins_code, strand_id in zip(
            _cif_column(category, "asym_id"),
            _cif_column(category, "entity_id"),
            _cif_column(category, "mon_id"),
            _cif_column(category, "ndb_seq_num"),
            _cif_column(category, "auth_seq_num"),
            _cif_column(category, "pdb_seq_num"),
            _cif_column(category, "pdb_ins_code"),
            _cif_column(category, "pdb_strand_id"),
        ):
            if asym_id is None or entity_id is None or monomer is None:
                raise ValueError("mmCIF _pdbx_nonpoly_scheme row is missing asym_id, entity_id, or mon_id")
            if ndb_number is None:
                # ModelCIF producers such as AF3 omit the optional NDB numbering.
                # Preserve the declared PDB/author occurrence identity, including
                # its insertion code, instead of requiring an unrelated namespace.
                occurrence_number = _cif_string(pdb_number) or _cif_string(auth_number)
                if occurrence_number is None:
                    raise ValueError("mmCIF nonpoly occurrence has no NDB, PDB, or author number")
                ndb_number = occurrence_number if ins_code is None else f"{occurrence_number}:{ins_code}"
            occurrence_key = asym_id, ndb_number
            if occurrence_key in occurrence_keys:
                raise ValueError(f"mmCIF _pdbx_nonpoly_scheme contains duplicate key {occurrence_key!r}")
            occurrence_keys.add(occurrence_key)
            author_residue_number = _cif_string(pdb_number) or _cif_string(auth_number)
            # One non-polymer asym holds many occurrences (every water of a chain), each
            # with its own opaque NDB identity and author number. The canonical component
            # axis follows declaration order; the NDB code is not an integer component index.
            ordinal = ordinal_by_asym[asym_id] = ordinal_by_asym.get(asym_id, 0) + 1
            numbering_key = asym_id, ordinal
            numbering = (author_residue_number, ins_code)
            if numbering_key in author_numbering and author_numbering[numbering_key] != numbering:
                raise ValueError(f"mmCIF _pdbx_nonpoly_scheme has conflicting numbering for {numbering_key!r}")
            author_numbering[numbering_key] = numbering
            nonpoly_occurrences.setdefault(asym_id, []).append(
                (ndb_number, entity_id, monomer, author_residue_number, ins_code, strand_id)
            )

    asym_entities: dict[str, str] = {}
    if "struct_asym" in block:
        category = block["struct_asym"]
        for asym_id, entity_id in zip(
            _cif_column(category, "id"),
            _cif_column(category, "entity_id"),
        ):
            if asym_id is None or entity_id is None:
                raise ValueError("mmCIF _struct_asym row is missing id or entity_id")
            previous = asym_entities.get(asym_id)
            if previous is not None and previous != entity_id:
                raise ValueError(f"mmCIF _struct_asym has conflicting entity for {asym_id!r}")
            asym_entities[asym_id] = entity_id

        for category_name in ("pdbx_poly_seq_scheme", "pdbx_branch_scheme"):
            if category_name not in block:
                continue
            scheme = block[category_name]
            for asym_id, entity_id in zip(
                _cif_column(scheme, "asym_id"),
                _cif_column(scheme, "entity_id"),
            ):
                if asym_id is None or asym_id not in asym_entities:
                    raise ValueError(f"mmCIF {category_name} references undeclared asym {asym_id!r}")
                if entity_id is not None and asym_entities[asym_id] != entity_id:
                    raise ValueError(f"mmCIF {category_name} disagrees with _struct_asym for {asym_id!r}")

    nonpoly_components: dict[str, str] = {}
    if "pdbx_entity_nonpoly" in block:
        category = block["pdbx_entity_nonpoly"]
        for entity_id, component_id in zip(
            _cif_column(category, "entity_id"),
            _cif_column(category, "comp_id"),
        ):
            if entity_id is None or component_id is None:
                raise ValueError("mmCIF _pdbx_entity_nonpoly row is missing entity_id or comp_id")
            previous = nonpoly_components.get(entity_id)
            if previous is not None and previous != component_id:
                raise ValueError(f"mmCIF non-polymer entity {entity_id!r} has conflicting components")
            nonpoly_components[entity_id] = component_id

    if has_entity_declarations:
        for asym_id, entity_id in asym_entities.items():
            if entity_id not in entity_types:
                raise ValueError(f"mmCIF _struct_asym {asym_id!r} references undeclared entity {entity_id!r}")

    for asym_id, occurrences in nonpoly_occurrences.items():
        declared_entity = asym_entities.get(asym_id)
        if "struct_asym" in block and declared_entity is None:
            raise ValueError(f"mmCIF non-polymer scheme references undeclared asym {asym_id!r}")
        for _, entity_id, component_id, *_ in occurrences:
            if declared_entity is not None and declared_entity != entity_id:
                raise ValueError(f"mmCIF non-polymer scheme for asym {asym_id!r} disagrees with _struct_asym")
            declared_component = nonpoly_components.get(entity_id)
            if "pdbx_entity_nonpoly" in block and declared_component is None:
                raise ValueError(f"mmCIF non-polymer scheme references undeclared entity {entity_id!r}")
            if declared_component is not None and declared_component != component_id:
                raise ValueError(f"mmCIF non-polymer scheme for entity {entity_id!r} disagrees with its component")

    if has_entity_declarations:
        for entity_id in {*polymer_monomers, *entity_poly_types}:
            if entity_types.get(entity_id) != "polymer":
                raise ValueError(f"mmCIF polymer declaration belongs to non-polymer entity {entity_id!r}")
        molecule_types = {"macrolide", "non-polymer", "water"}
        for entity_id in nonpoly_components:
            if entity_types.get(entity_id) not in molecule_types:
                raise ValueError(f"mmCIF non-polymer declaration belongs to entity {entity_id!r} of another type")
        for occurrences in nonpoly_occurrences.values():
            for _, entity_id, *_ in occurrences:
                if entity_types.get(entity_id) not in molecule_types:
                    raise ValueError(f"mmCIF non-polymer occurrence belongs to entity {entity_id!r} of another type")
        if "pdbx_entity_branch_list" in block:
            for entity_id in _cif_column(block["pdbx_entity_branch_list"], "entity_id"):
                if entity_id is None:
                    raise ValueError("mmCIF branch declaration is missing entity_id")
                entity_type = entity_types.get(entity_id)
                if entity_type != "branched":
                    raise ValueError(
                        f"mmCIF entity {entity_id!r} is declared {entity_type!r} "
                        "but also has a branched-entity definition"
                    )

    if "atom_site" in block:
        atom_site = block["atom_site"]
        if has_entity_declarations and "label_entity_id" in atom_site:
            for entity_id in _cif_column(atom_site, "label_entity_id"):
                if entity_id is not None and entity_id not in entity_types:
                    raise ValueError(f"mmCIF atom_site references undeclared entity {entity_id!r}")
        if "struct_asym" in block and "label_asym_id" in atom_site:
            for asym_id in _cif_column(atom_site, "label_asym_id"):
                if asym_id is not None and asym_id not in asym_entities:
                    raise ValueError(f"mmCIF atom_site references undeclared asym {asym_id!r}")

    return EntityDeclarations(
        entity_types=entity_types,
        polymer_monomers=(
            {
                entity_id: tuple((number, tuple(positions[number])) for number in sorted(positions))
                for entity_id, positions in polymer_monomers.items()
            }
            if has_polymer_sequences
            else None
        ),
        polymer_types=entity_poly_types,
        author_numbering=author_numbering,
        asym_entities=asym_entities,
        nonpolymer_components=nonpoly_components,
        nonpolymer_occurrences={asym_id: tuple(occurrences) for asym_id, occurrences in nonpoly_occurrences.items()},
    )
