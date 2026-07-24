r"""
Project BoltzGen residue design masks to and from placement records.

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
    r"""
    Describe one generated or retained residue in a design mask.

    Args:
        designed: Whether the output residue is generated rather than copied.
        polymer_type: Polymer chemistry for a generated residue.
        source: Source component occurrence for a retained residue.

    Raises:
        ValueError: When converted if the fields do not describe exactly one of a
            generated polymer residue or retained source residue.
    """

    designed: bool
    polymer_type: PolymerType | None = None
    source: ComponentReference | None = None


def _placement_value_of_run(placement_id: str, run: list[DesignMaskResidue]) -> PlacementValue:
    r"""
    Collapse one maximal same-``placement_id`` run of residues into one core value.

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
    r"""
    Convert residue design specifications to placements.

    Args:
        residues: Residue specifications in output order.
        group_of: Function returning the placement group for each residue index.
        placement_id_of: Function returning the placement identifier for each residue
            index. Each identifier must occupy one contiguous run.

    Returns:
        Placements containing exact-length generated segments for designed runs and
        existing spans for fixed runs.

    Raises:
        ValueError: If a placement identifier is noncontiguous or a fixed run does not
            describe one contiguous source span.
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
    r"""
    Convert realized placements to a residue design mask.

    Args:
        placement: Declared generated segments and existing spans.
        placement_map: Realized residue references for each placement.

    Returns:
        A pair containing the designed-status flag and declaration-group label for
        each realized residue, in placement declaration order.
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
