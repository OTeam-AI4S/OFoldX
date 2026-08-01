"""Explicit molecular consumers of source documents; no training preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from ..io.selection import _ALTLOC_POLICIES
from ..io.source import CifSource, PdbSource, StructureSource, StructureView
from ._materialization.template import materialize_template as materialize_template
from .molecular import ChemicalComponent
from .structure import Structure

__all__ = ["materialize_structure", "materialize_template"]


def materialize_structure(
    source: StructureSource | StructureView,
    *,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = True,
    include_hydrogens: bool = True,
    components: Mapping[str, ChemicalComponent] | None = None,
    component_resolver: Callable[[str], ChemicalComponent] | None = None,
) -> Structure:
    """Realize an explicit source view, requiring sufficient chemical evidence."""
    if isinstance(source, StructureSource):
        source = StructureView(source)
    if not isinstance(source, StructureView):
        raise TypeError("materialize_structure requires StructureSource or StructureView")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if isinstance(source.source, CifSource):
        from ._materialization.cif import materialize_cif

        return materialize_cif(
            source,
            altloc=altloc,
            include_water=include_water,
            include_hydrogens=include_hydrogens,
            components=components,
            component_resolver=component_resolver,
        )
    if isinstance(source.source, PdbSource):
        from ._materialization.pdb import materialize_pdb

        return materialize_pdb(
            source,
            altloc=altloc,
            include_water=include_water,
            include_hydrogens=include_hydrogens,
            components=components,
            component_resolver=component_resolver,
        )
    raise TypeError(f"unsupported molecular source: {type(source.source).__name__}")
