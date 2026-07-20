r"""Design-mask <-> placement-core projection for BoltzGen structure design.

The lossless projection between a per-output-residue design-mask surface and the frozen
shared placement core (:data:`~ofoldx.data.placement.PlacementSpec`). ``from_design_mask``
folds a per-residue mask into maximal designed / fixed runs; ``to_design_mask`` unfolds a
realized placement core back onto per-realized-row designed-ness. Named losses: a designed
non-polymer (de-novo ligand) has no core spelling (GAP-1) and per-residue pinned identity is
carried by a separate conditioning record, not by this projection.
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
    validate_placement,
)


@dataclass(frozen=True)
class DesignMaskResidue:
    r"""One per-output-residue cell of a design-mask surface (``from_design_mask`` input unit).

    Exactly one of the two modes must hold:

    * a DESIGNED (generated) residue — ``designed=True``, ``polymer_type`` set, ``source=None``;
    * a FIXED (source/context) residue — ``designed=False``, ``source=(instance_id, index)`` set.

    ``designed=True`` with ``polymer_type=None`` is the GAP-1 designed-non-polymer case and is
    rejected by :func:`from_design_mask` (a de-novo ligand is not spellable in the core).
    """

    designed: bool
    polymer_type: PolymerType | None = None
    source: ComponentReference | None = None


def _placement_value_of_run(placement_id: str, run: list[DesignMaskResidue]) -> PlacementValue:
    r"""Collapse one maximal same-``placement_id`` run of residues into one core value.

    A designed run → one :class:`GeneratedSegment` (one polymer type, exact length); a fixed
    run → one :class:`ExistingSpan` (one contiguous half-open source interval). Mixing, a
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

    sources = [residue.source for residue in run]
    if any(source is None for source in sources):
        raise ValueError(f"fixed placement {placement_id!r} has a residue with no source reference")
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
    r"""Project a per-output-residue design mask onto the shared placement core.

    Args:
        residues: The per-output-residue mask cells, in output order (see :class:`DesignMaskResidue`).
        group_of: Maps a residue index to its declaration group (chain) label — the outer
            :data:`~ofoldx.data.placement.PlacementSpec` key.
        placement_id_of: Maps a residue index to its ``placement_id`` — the globally-unique inner
            key. Consecutive residues sharing one ``placement_id`` form one placement; the
            ``placement_id`` must be contiguous (a value may not reappear after a different one).

    Returns:
        The realized :data:`~ofoldx.data.placement.PlacementSpec`: maximal designed runs →
        :class:`GeneratedSegment` (exact length), maximal fixed runs → :class:`ExistingSpan`.
        Interleaving is preserved as ordered placements within each group. Lossless for the
        polymer surface; see module docstring for the named losses.
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

    validate_placement(placement)  # frozen §3.7 global-uniqueness gate (redundant but explicit)
    return placement


def to_design_mask(
    placement: PlacementSpec,
    placement_map: PlacementMap,
) -> tuple[list[bool], list[str]]:
    r"""Project a realized placement core back onto a per-realized-row design mask.

    Walks ``placement`` in frozen declared order (outer group, then inner placement) and, for
    each placement, emits one row per realized occurrence in ``placement_map``: ``True`` for a
    :class:`GeneratedSegment` occurrence, ``False`` for an :class:`ExistingSpan` occurrence, with
    the declaration-group label carried per row. Symmetry/copy expansion (a placement whose
    occurrences span more than one realized instance id) yields one row per occurrence, so the
    result is over the REALIZED rows, matching the frozen ``placement_map`` order.

    Returns:
        ``(res_design_mask, per_row_group)`` — a per-realized-row designed-ness flag and the
        owning declaration-group label. Lossless given ``placement_map``; per-residue pinned
        identity is NOT carried (a separate conditioning record — see module docstring).
    """

    res_design_mask: list[bool] = []
    per_row_group: list[str] = []
    for group, placements in placement.items():
        for placement_id, value in placements.items():
            designed = type(value) is GeneratedSegment
            occurrences = placement_map[group][placement_id]
            for _ in occurrences:
                res_design_mask.append(designed)
                per_row_group.append(group)
    return res_design_mask, per_row_group


__all__ = [
    "DesignMaskResidue",
    "from_design_mask",
    "to_design_mask",
]
