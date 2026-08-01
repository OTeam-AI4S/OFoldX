"""Explicit alternate-location, microheterogeneity and water selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, cast

import numpy as np

from ..data._placed_atoms import PlacedAtoms, _AtomRow
from .polymer import POLYMER_TYPES
from .source import StructureView, _require_cif_source

if TYPE_CHECKING:
    from .declarations import EntityDeclarations

RowsT = TypeVar("RowsT", bound=np.recarray | PlacedAtoms)

_ALTLOC_POLICIES = frozenset({"highest_occupancy", "first"})
_MmcifPolymerPosition: TypeAlias = tuple[str | None, str, str, int]


def _select_cif_declarations(view: StructureView) -> EntityDeclarations:
    """Select referenced CIF entities without adding evidence requirements."""
    declarations = _require_cif_source(view.source).declarations
    return declarations if view.asym_ids is None else declarations.select_asym_ids(view.asym_ids)


def _polymer_declarations(view: StructureView) -> EntityDeclarations:
    """Require evidence that distinguishes absent polymers from unknown types."""
    source = _require_cif_source(view.source)
    if "struct_asym" not in source.block:
        raise ValueError("polymer coordinate projection requires _struct_asym declarations")
    if view.asym_ids is not None:
        missing = view.asym_ids - source.declarations.asym_entities.keys()
        if missing:
            raise KeyError(f"no declared CIF chains {sorted(missing)!r}")
    declarations = _select_cif_declarations(view)
    for entity in dict.fromkeys(declarations.asym_entities.values()):
        entity_type = declarations.entity_types.get(entity)
        if entity_type is not None and entity_type != "polymer":
            continue
        if entity not in declarations.polymer_types:
            raise ValueError(f"polymer coordinate projection requires _entity_poly.type for entity {entity!r}")
        if declarations.polymer_types[entity].casefold() not in POLYMER_TYPES:
            raise ValueError(f"unsupported polymer type {declarations.polymer_types[entity]!r}")
        if declarations.polymer_monomers is None or not declarations.polymer_monomers.get(entity):
            raise ValueError(f"polymer coordinate projection requires _entity_poly_seq for entity {entity!r}")
    return declarations


def _altloc_position_key(row: np.record | _AtomRow) -> tuple[object, ...]:
    # Component identity is deliberately absent: microheterogeneous components at one
    # structural position compete as alternate realizations.
    return (
        row.symmetry_id,
        row.chain_id,
        row.residue_id,
        row.auth_residue_id,
        row.insertion_code,
    )


def _mean_present_occupancy(values: Sequence[float | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / max(len(present), 1)


def _select_mmcif_polymer_microheterogeneity(
    rows: RowsT,
    polymer_monomers: Mapping[str, tuple[tuple[int, tuple[str, ...]], ...]] | None,
    policy: Literal["highest_occupancy", "first"],
) -> RowsT:
    r"""Select one deterministic deposited identity at each polymer position."""

    if polymer_monomers is None:
        return rows
    declared = {
        (entity_id, number): frozenset(candidates)
        for entity_id, positions in polymer_monomers.items()
        for number, candidates in positions
    }
    microheterogeneous = {key for key, candidates in declared.items() if len(candidates) > 1}
    if not microheterogeneous:
        return rows

    row: np.record | _AtomRow
    position_rows: dict[_MmcifPolymerPosition, list[np.record | _AtomRow]] = {}
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            continue
        entity_id = str(row.entity_id)
        number = int(row.label_seq_id)
        if (entity_id, number) not in microheterogeneous:
            continue
        key = (
            None if row.symmetry_id is None else str(row.symmetry_id),
            str(row.chain_id),
            entity_id,
            number,
        )
        position_rows.setdefault(key, []).append(row)

    selected: dict[_MmcifPolymerPosition, tuple[str | None, str]] = {}
    for position, source_rows in position_rows.items():
        _, _, entity_id, number = position
        observed_components = {str(row.component_id) for row in source_rows}
        if not observed_components <= declared[(entity_id, number)]:
            unexpected = sorted(observed_components - declared[(entity_id, number)])
            raise ValueError(
                f"observed mmCIF components {unexpected!r} are not declared at entity {entity_id!r} position {number}"
            )

        rows_by_label: dict[str, list[np.record | _AtomRow]] = {}
        for row in source_rows:
            if row.altloc_id is not None:
                rows_by_label.setdefault(str(row.altloc_id), []).append(row)
        labels: dict[str, str] = {}
        for label, label_rows in rows_by_label.items():
            components = {str(row.component_id) for row in label_rows}
            if len(components) != 1:
                raise ValueError(f"mmCIF polymer alternate label {label!r} maps to multiple monomers at {position!r}")
            labels[label] = next(iter(components))

        if labels:
            if observed_components - set(labels.values()):
                raise ValueError(
                    f"mmCIF polymer microheterogeneity at {position!r} mixes labelled and unlabelled monomers"
                )
            choices = list(labels)
            if policy == "first":
                label = choices[0]
            else:
                label = max(
                    choices,
                    key=lambda candidate: _mean_present_occupancy([row.occupancy for row in rows_by_label[candidate]]),
                )
            selected[position] = (label, labels[label])
            continue

        rows_by_component: dict[str, list[np.record | _AtomRow]] = {}
        for row in source_rows:
            rows_by_component.setdefault(str(row.component_id), []).append(row)
        choices = list(rows_by_component)
        if policy == "first":
            component_id = choices[0]
        else:
            component_id = max(
                choices,
                key=lambda candidate: _mean_present_occupancy([row.occupancy for row in rows_by_component[candidate]]),
            )
        selected[position] = (None, component_id)

    if not selected:
        return rows
    keep = []
    for row in rows:
        if row.entity_id is None or row.label_seq_id is None:
            keep.append(True)
            continue
        position = (
            None if row.symmetry_id is None else str(row.symmetry_id),
            str(row.chain_id),
            str(row.entity_id),
            int(row.label_seq_id),
        )
        realization = selected.get(position)
        if realization is None:
            keep.append(True)
            continue
        selected_label, component_id = realization
        row_label = None if row.altloc_id is None else str(row.altloc_id)
        keep.append(
            str(row.component_id) == component_id
            and (selected_label is None or row_label is None or row_label == selected_label)
        )
    # NumPy preserves the recarray subclass under boolean selection; its stubs
    # describe the base ndarray return instead.
    return cast(RowsT, rows[np.asarray(keep, dtype=np.bool_)])


def _select_altloc(
    rows: RowsT,
    policy: Literal["highest_occupancy", "first"],
) -> RowsT:
    if not any(label is not None for label in rows.altloc_id):
        return rows

    labels_by_residue: dict[tuple[object, ...], list[str]] = {}
    occupancies_by_label: dict[tuple[tuple[object, ...], str], list[float]] = {}
    for row in rows:
        if row.altloc_id is None:
            continue
        residue_key = _altloc_position_key(row)
        label = str(row.altloc_id)
        labels = labels_by_residue.setdefault(residue_key, [])
        if label not in labels:
            labels.append(label)
        key = residue_key, label
        if row.occupancy is not None:
            occupancies_by_label.setdefault(key, []).append(float(row.occupancy))
    selected: dict[tuple[object, ...], str] = {}
    for residue_key, labels in labels_by_residue.items():
        if policy == "first":
            selected[residue_key] = labels[0]
        else:
            selected[residue_key] = max(
                labels,
                key=lambda label: _mean_present_occupancy(occupancies_by_label.get((residue_key, label), ())),
            )
    keep = np.asarray(
        [
            row.altloc_id is None or selected.get(_altloc_position_key(row), row.altloc_id) == row.altloc_id
            for row in rows
        ],
        dtype=np.bool_,
    )
    return cast(RowsT, rows[keep])
