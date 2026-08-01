"""Chemical water selection; observation selection is shared with coordinate consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ...io.selection import RowsT
from ..molecular import ChemicalComponent, is_water_component


def _exclude_water_rows(
    rows: RowsT,
    component_table: Mapping[str, ChemicalComponent],
    entity_types: Mapping[str, str] | None = None,
) -> RowsT:
    declared_entities = entity_types
    entity_types = {} if entity_types is None else entity_types
    water_entities = {entity_id for entity_id, entity_type in entity_types.items() if entity_type == "water"}
    keep: NDArray[np.bool_] = np.ones(len(rows), dtype=np.bool_)
    fallback_groups: dict[tuple[object, ...], tuple[list[int], bool]] = {}
    for index, (
        entity_id,
        symmetry_id,
        chain_id,
        residue_id,
        auth_residue_id,
        insertion_code,
        component_id,
    ) in enumerate(
        zip(
            rows.entity_id,
            rows.symmetry_id,
            rows.chain_id,
            rows.residue_id,
            rows.auth_residue_id,
            rows.insertion_code,
            rows.component_id,
            strict=True,
        )
    ):
        if entity_id in water_entities:
            keep[index] = False
            continue
        if entity_id is not None and entity_id in entity_types:
            continue
        key = (
            ("entity", entity_id)
            if declared_entities is not None and entity_id is not None
            else (
                "residue",
                symmetry_id,
                chain_id,
                residue_id,
                auth_residue_id,
                insertion_code,
                component_id,
            )
        )
        indices, all_water = fallback_groups.setdefault(key, ([], True))
        indices.append(index)
        if all_water and not is_water_component(component_table[str(component_id)]):
            fallback_groups[key] = (indices, False)
    for indices, all_water in fallback_groups.values():
        if all_water:
            keep[indices] = False
    return cast(RowsT, rows[keep])
