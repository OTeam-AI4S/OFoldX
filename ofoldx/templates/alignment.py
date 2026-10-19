# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Structural templates: target-to-template alignments, resolving target instances, and placement frames.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from ..data.molecular import MolecularSystem, MonomerType
from ..data.tokenization import CLASS_MONOMER, token_classes
from ..errors import RequestRef, StructureRef, fail
from .coordinates import PolymerCoordinates


@dataclass(frozen=True)
class TemplateAlignment:
    r"""
    A target-to-template polymer alignment.

    ``target_instance_id=None`` broadcasts the alignment to every instance of ``target_entity_id``.
    ``template_instance_id`` names a chain of the template's ``PolymerCoordinates``. ``residue_pairs`` are zero-based
    ``(target, template)`` positions, strictly increasing on both axes.

    Examples:
        >>> TemplateAlignment("query", None, "A", ((0, 2), (1, 3))).residue_pairs
        ((0, 2), (1, 3))
    """

    target_entity_id: str
    target_instance_id: str | None
    template_instance_id: str
    residue_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        pairs = self.residue_pairs
        array = np.array(pairs).reshape(-1, 2)
        if array.dtype.kind not in "iu":
            fail("value_invalid", RequestRef("residue_pairs", str(pairs)), invariant="integer")
        if not len(pairs) or (array < 0).any() or (np.diff(array, axis=0) <= 0).any():
            fail("value_invalid", RequestRef("residue_pairs", str(pairs)), invariant="strictly_increasing")


@dataclass(frozen=True, eq=False)
class Template:
    r"""
    Structural template evidence: chains in one frame and the alignments that place them on a target.

    Alignments may cover disjoint target ranges, or place the same template position on one target position; two
    alignments placing *different* template positions on one target position would make the template depend on their
    order and are refused. A broadcast alignment covers every instance of its entity, so it overlaps an alignment naming
    one instance of that entity.
    """

    coordinates: PolymerCoordinates
    alignments: tuple[TemplateAlignment, ...]

    def __post_init__(self) -> None:
        if not self.alignments:
            fail("value_invalid", RequestRef("alignments", "()"), invariant="nonempty")
        placements: dict[tuple[str, int], dict[str | None, tuple[str, int]]] = {}
        for alignment in self.alignments:
            chain = self.coordinates.chains.get(alignment.template_instance_id)
            if chain is None or alignment.residue_pairs[-1][1] >= len(chain.component_ids):
                fail("value_invalid", RequestRef("template_instance_id", alignment.template_instance_id))
            for target, position in alignment.residue_pairs:
                source = (alignment.template_instance_id, position)
                covered = placements.setdefault((alignment.target_entity_id, target), {})
                for instance, placed in covered.items():
                    if placed != source and (instance is None or alignment.target_instance_id in (None, instance)):
                        fail(
                            "value_invalid",
                            RequestRef("alignments", f"{alignment.target_entity_id}:{target}"),
                            invariant="one_template_position_per_target",
                        )
                covered[alignment.target_instance_id] = source


@dataclass(frozen=True)
class ResolvedTemplateAlignment:
    r"""One template chain aligned to one target instance: zero-based ``(target, template)`` position pairs, stated
    sorted and unique by whoever builds it (repeated pairs and pair order carry no evidence). The class checks
    nothing: ``resolve_template`` passes on a ``TemplateAlignment``'s pairs, which are strictly increasing, and a
    family remap sorts its own."""

    template_instance_id: str
    target_instance_index: int
    residue_pairs: tuple[tuple[int, int], ...]


def resolve_template(
    template: Template, system: MolecularSystem, *, monomer_types: Collection[MonomerType]
) -> tuple[ResolvedTemplateAlignment, ...]:
    r"""
    The one target resolution: each alignment bound to the instances of its entity in ``system`` (all of them when
    ``target_instance_id`` is ``None``), in alignment order, then system order.

    Positions index the instance's components in the uncut system (a crop acts on the layout; the family drops the
    tokens outside it). An unknown entity or instance, an instance of another entity, or a target position past its
    instance is ``InvalidRequestError(template_target_unknown)``; a target position whose monomer type is not in
    ``monomer_types`` (the family's template channels) is ``UnsupportedError(template_target_unsupported)``. Nothing
    is skipped.
    """

    s = system.arrays()
    entity_text = np.asarray(system.symbols()["entity"], dtype=object)[s["entity_symbol"]]
    instance_entity = entity_text[s["instance_entity"]]
    pointer = s["instance_component_ptr"]
    monomer = CLASS_MONOMER[token_classes(system).token]
    allowed = np.isin(monomer, [int(m) for m in monomer_types])
    resolved = []
    for alignment in template.alignments:
        target = RequestRef("target", f"{alignment.target_entity_id}/{alignment.target_instance_id}")
        if alignment.target_instance_id is None:
            instances = np.flatnonzero(instance_entity == alignment.target_entity_id)
        else:
            instances = system.instance_index([alignment.target_instance_id])
            if instances[0] < 0 or instance_entity[instances[0]] != alignment.target_entity_id:
                fail("template_target_unknown", target)
        if not len(instances):
            fail("template_target_unknown", target)
        positions = np.array([t for t, _ in alignment.residue_pairs], dtype=np.int64)
        for instance in instances.tolist():
            start, stop = int(pointer[instance]), int(pointer[instance + 1])
            if positions[-1] >= stop - start:
                fail("template_target_unknown", target, position=int(positions[-1]))
            if not allowed[start + positions].all():
                position = int(positions[~allowed[start + positions]][0])
                fail(
                    "template_target_unsupported",
                    StructureRef(instance=instance, component=start + position),
                    monomer_type=MonomerType(int(monomer[start + position])).name,
                )
            resolved.append(
                ResolvedTemplateAlignment(alignment.template_instance_id, instance, alignment.residue_pairs)
            )
    return tuple(resolved)


def template_alignment_frames(alignments: Sequence[ResolvedTemplateAlignment]) -> tuple[int, ...]:
    r"""
    The placement frame of each alignment from one template (called after any family remap).

    Disjoint, uniquely placed template positions share frame 0. A template position placed on several target sites
    makes each alignment that uses it an independent frame; exact duplicate alignments share one. A target site that
    would need two frames is ``UnsupportedError(template_frame_incompatible)``: the one-frame-per-token template
    carriers cannot represent it.
    """

    source_targets: dict[tuple[str, int], set[tuple[int, int]]] = {}
    target_sources: dict[tuple[int, int], tuple[str, int]] = {}
    for alignment in alignments:
        for target_position, source_position in alignment.residue_pairs:
            source = (alignment.template_instance_id, source_position)
            target = (alignment.target_instance_index, target_position)
            # one template position per target site: unreachable from a valid Template (Template.__post_init__)
            assert target_sources.setdefault(target, source) == source, f"two template positions on {target}"
            source_targets.setdefault(source, set()).add(target)

    independent: dict[ResolvedTemplateAlignment, int] = {}
    target_frames: dict[tuple[int, int], int] = {}
    frames = []
    for alignment in alignments:
        reused = any(
            len(source_targets[(alignment.template_instance_id, position)]) > 1
            for _, position in alignment.residue_pairs
        )
        frame = independent.setdefault(alignment, len(independent) + 1) if reused else 0
        for target_position, _ in alignment.residue_pairs:
            target = (alignment.target_instance_index, target_position)
            if target_frames.setdefault(target, frame) != frame:
                fail(
                    "template_frame_incompatible",
                    StructureRef(instance=alignment.target_instance_index),
                    position=target_position,
                )
        frames.append(frame)
    return tuple(frames)
