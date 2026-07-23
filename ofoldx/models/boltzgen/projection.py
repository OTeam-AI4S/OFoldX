r"""Project BoltzGen residue design masks to and from placement records.

``from_design_mask`` collapses a per-residue mask into maximal generated-polymer segments
and contiguous spans of existing components in a
[`PlacementSpec`][ofoldx.data.placement.PlacementSpec]. ``to_design_mask`` expands those placements
through a [`PlacementMap`][ofoldx.data.placement.PlacementMap] to recover designedness for each
realized row. The representation cannot encode generated non-polymers such as de-novo
ligands, and per-residue pinned identity remains in a separate conditioning record.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ...data.molecular import ComponentReference, PolymerType
from ...data.placement import (
    ExistingSpan,
    GeneratedSegment,
    PlacementMap,
    PlacementSpec,
    PlacementValue,
)


@dataclass(frozen=True)
class DesignMaskResidue:
    r"""One per-output-residue cell of a design-mask surface (``from_design_mask`` input unit).

    Exactly one of the two modes must hold:

    * a DESIGNED (generated) residue — ``designed=True``, ``polymer_type`` set, ``source=None``;
    * a FIXED (source/context) residue — ``designed=False``, ``source=(instance_id, index)`` set.

    ``designed=True`` with ``polymer_type=None`` represents a generated non-polymer and is
    rejected by [`from_design_mask`][] because [`PlacementValue`][] cannot encode it.
    """

    designed: bool
    polymer_type: PolymerType | None = None
    source: ComponentReference | None = None


def _placement_value_of_run(placement_id: str, run: list[DesignMaskResidue]) -> PlacementValue:
    r"""Collapse one maximal same-``placement_id`` run of residues into one core value.

    A designed run → one [`GeneratedSegment`][] (one polymer type, exact length); a fixed
    run → one [`ExistingSpan`][] (one contiguous half-open source interval). Mixing, a
    designed non-polymer, or a non-contiguous / multi-instance source run is rejected.
    """

    designed = run[0].designed
    if any(residue.designed != designed for residue in run):
        raise ValueError(
            f"placement {placement_id!r} mixes designed and fixed residues; a placement_id must be "
            "either wholly designed (GeneratedSegment) or wholly fixed (ExistingSpan)"
        )

    if designed:
        polymer_type = run[0].polymer_type
        if polymer_type is None:
            raise ValueError(
                f"designed placement {placement_id!r} has no polymer_type: a designed non-polymer "
                "(GAP-1 de-novo ligand) has no PlacementValue and must be routed out-of-band"
            )
        if any(residue.polymer_type != polymer_type for residue in run):
            raise ValueError(
                f"designed placement {placement_id!r} spans more than one polymer type; a maximal "
                "designed run is one polymer type"
            )
        if any(residue.source is not None for residue in run):
            raise ValueError(f"designed placement {placement_id!r} carries a source reference")
        return GeneratedSegment(polymer_type, len(run), len(run))

    sources: list[ComponentReference] = []
    for residue in run:
        if residue.source is None:
            raise ValueError(f"fixed placement {placement_id!r} has a residue with no source reference")
        sources.append(residue.source)
    instance_id = sources[0][0]
    start = sources[0][1]
    for offset, source in enumerate(sources):
        if source[0] != instance_id or source[1] != start + offset:
            raise ValueError(
                f"fixed placement {placement_id!r} is not one contiguous source interval of a single "
                f"instance: {sources!r}"
            )
    return ExistingSpan(instance_id, start, start + len(run))


def from_design_mask(
    residues: Sequence[DesignMaskResidue],
    *,
    group_of: Callable[[int], str],
    placement_id_of: Callable[[int], str],
) -> PlacementSpec:
    r"""Project a per-output-residue design mask onto a [`PlacementSpec`][].

    Args:
        residues: The per-output-residue mask cells, in output order (see [`DesignMaskResidue`][]).
        group_of: Maps a residue index to its declaration group (chain) label — the outer
            [`PlacementSpec`][ofoldx.data.placement.PlacementSpec] key.
        placement_id_of: Maps a residue index to its ``placement_id`` — the globally-unique inner
            key. Consecutive residues sharing one ``placement_id`` form one placement; the
            ``placement_id`` must be contiguous (a value may not reappear after a different one).

    Returns:
        The realized [`PlacementSpec`][ofoldx.data.placement.PlacementSpec]: maximal designed runs →
        [`GeneratedSegment`][] (exact length), maximal fixed runs → [`ExistingSpan`][].
        Interleaving is preserved as ordered placements within each group. The projection
        preserves the information expressible by [`GeneratedSegment`][] and
        [`ExistingSpan`][]; see the module docstring for its representational limits.
    """

    placement: PlacementSpec = {}
    seen_placement_ids: set[str] = set()
    run: list[DesignMaskResidue] = []
    run_group: str | None = None
    run_placement_id: str | None = None

    def _flush() -> None:
        nonlocal run
        if not run:
            return
        assert run_group is not None and run_placement_id is not None
        if run_placement_id in seen_placement_ids:
            raise ValueError(
                f"placement_id {run_placement_id!r} is non-contiguous: it reappears after another "
                "placement; a placement_id must name one contiguous run"
            )
        seen_placement_ids.add(run_placement_id)
        placement.setdefault(run_group, {})[run_placement_id] = _placement_value_of_run(run_placement_id, run)
        run = []

    for index, residue in enumerate(residues):
        group = group_of(index)
        placement_id = placement_id_of(index)
        if placement_id != run_placement_id or group != run_group:
            _flush()
            run_group = group
            run_placement_id = placement_id
        run.append(residue)
    _flush()

    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""Project realized placements back onto a per-realized-row design mask.

    Walks ``placement`` in declaration order (outer group, then inner placement) and, for
    each placement, emits one row per realized occurrence in ``placement_map``: ``True`` for a
    [`GeneratedSegment`][] occurrence, ``False`` for an [`ExistingSpan`][] occurrence, with
    the declaration-group label carried per row. Symmetry/copy expansion (a placement whose
    occurrences span more than one realized instance id) yields one row per occurrence, so the
    result covers the realized rows in ``placement_map`` occurrence order.

    Returns:
        ``(res_design_mask, per_row_group)`` — a per-realized-row designed-ness flag and the
        owning declaration-group label. ``placement_map`` supplies the realized expansion;
        per-residue pinned identity is carried separately, as described in the module docstring.
    """

    res_design_mask: list[bool] = []
    per_row_group: list[str] = []
    for group, placements in placement.items():
        for placement_id, value in placements.items():
            designed = isinstance(value, GeneratedSegment)
            occurrences = placement_map[group][placement_id]
            for _ in occurrences:
                res_design_mask.append(designed)
                per_row_group.append(group)
    return res_design_mask, per_row_group
