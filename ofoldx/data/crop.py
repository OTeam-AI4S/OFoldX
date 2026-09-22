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
Training crops over a ``TokenLayout`` (protocol and policies kept from the earlier ``data/crop.py``; a numpy
``Generator`` instead of a torch one; masks instead of index tensors).

A crop keeps whole component groups (every token of a component or none). ``required`` tokens pull in their whole
groups; ``anchor`` tokens, when given, guarantee that the crop holds at least one of them. Geometric policies read token
positions from the structure's coordinates where their evidence is not ``NONE``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..errors import RequestRef, fail
from .schema import check_mask
from .structure import Structure
from .tokenization import TokenLayout, select_tokens

_BLOCK = 1 << 20  # pairwise-distance elements per block


class Cropper(Protocol):
    r"""A configured crop policy over one token axis: the mask of the tokens it keeps."""

    def __call__(
        self,
        structure: Structure,
        layout: TokenLayout,
        *,
        required: NDArray[np.bool_] | None,
        generator: np.random.Generator,
    ) -> NDArray[np.bool_]: ...


def crop_token_layout(
    structure: Structure,
    layout: TokenLayout,
    cropper: Cropper,
    *,
    required: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> TokenLayout:
    r"""Apply one configured crop policy."""
    return select_tokens(layout, cropper(structure, layout, required=required, generator=generator))


def _token_instance(structure: Structure, layout: TokenLayout) -> NDArray[np.int64]:
    pointer = structure.system.instance_component_ptr
    return np.searchsorted(pointer, layout.token_component, side="right") - 1


def resolved_token_mask(structure: Structure, layout: TokenLayout) -> NDArray[np.bool_]:
    r"""Tokens holding at least one atom whose coordinates are not ``NONE``."""
    count = len(layout.token_component)
    owner = np.repeat(np.arange(count), np.diff(layout.token_atom_ptr))
    return np.bincount(owner[structure.coordinate_mask[layout.token_atoms]], minlength=count) > 0


def _token_coordinates(structure: Structure, layout: TokenLayout) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    r"""Per token: its centre atom's coordinates when present, else the mean of its present atoms; and presence."""
    count = len(layout.token_component)
    owner = np.repeat(np.arange(count), np.diff(layout.token_atom_ptr))
    present = structure.coordinate_mask[layout.token_atoms]
    sums = np.zeros((count, 3))
    np.add.at(sums, owner[present], structure.coordinates[layout.token_atoms[present]])
    counts = np.bincount(owner[present], minlength=count)
    resolved = counts > 0
    coordinates = np.where(resolved[:, None], sums / np.maximum(counts, 1)[:, None], 0.0)
    centre = layout.token_centre_atom
    centred = structure.coordinate_mask[centre]
    coordinates[centred] = structure.coordinates[centre[centred]]
    return coordinates, resolved


def _state(layout: TokenLayout, max_tokens: int, required, anchor):
    r"""Component groups, the required tokens grown to whole groups, and the token limit (``required`` must fit)."""
    count = len(layout.token_component)
    _, group, sizes = np.unique(layout.token_component, return_inverse=True, return_counts=True)
    required = np.zeros(count, dtype=bool) if required is None else check_mask("required", required, count)
    required_groups = np.zeros(len(sizes), dtype=bool)
    required_groups[group[required]] = True
    required = required_groups[group]
    limit = min(max_tokens, count)
    if int(required.sum()) > limit:
        fail("value_invalid", RequestRef("required", str(int(required.sum()))), invariant="fits", limit=limit)
    return required, None if anchor is None else check_mask("anchor", anchor, count), limit, group, sizes


def _add_group(base, candidates, group, sizes, limit, generator, argument):
    if (base & candidates).any():
        return base
    groups = np.zeros(len(sizes), dtype=bool)
    groups[group[candidates]] = True
    groups &= sizes <= limit - int(base.sum())
    feasible = np.flatnonzero(candidates & groups[group])
    if not len(feasible):
        fail("value_invalid", RequestRef(argument, "groups"), invariant="fits", limit=limit)
    chosen = group[feasible[generator.integers(len(feasible))]]
    return base | (group == chosen)


def _complete(base, scores, limit, group, sizes, generator) -> NDArray[np.bool_]:
    selected = np.zeros(len(sizes), dtype=bool)
    selected[group[base]] = True
    remaining = limit - int(sizes[selected].sum())
    if remaining:
        candidates = generator.permutation(np.flatnonzero(~selected))
        best = np.full(len(sizes), np.inf)
        np.minimum.at(best, group, scores)
        for candidate in candidates[np.argsort(best[candidates], kind="stable")]:
            if sizes[candidate] <= remaining:
                selected[candidate] = True
                remaining -= int(sizes[candidate])
                if not remaining:
                    break
    return selected[group]


def _distances(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1))


def _spatial_scores(coordinates, resolved, seeds) -> NDArray[np.float64]:
    seeds = seeds[resolved[seeds]]
    scores = np.full(len(coordinates), np.inf)
    rows = np.flatnonzero(resolved)
    step = max(1, _BLOCK // max(len(seeds), 1))
    for start in range(0, len(rows), step):
        block = rows[start : start + step]
        scores[block] = _distances(coordinates[block], coordinates[seeds]).min(axis=1)
    return scores


def _reservoir(selected, seen: int, candidates, generator):
    if not len(candidates):
        return selected, seen
    total = seen + len(candidates)
    offset = int(generator.integers(total))
    return (candidates[offset] if offset < len(candidates) else selected), total


def continuous_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    required: NDArray[np.bool_] | None = None,
    anchor: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> NDArray[np.bool_]:
    r"""Contiguous token-axis neighbourhoods: required tokens seed their instances, other instances a random token."""
    required, anchor, limit, group, sizes = _state(layout, max_tokens, required, anchor)
    if limit == len(group):
        return np.ones(limit, dtype=bool)
    base = _add_group(
        required.copy(), np.ones_like(required) if anchor is None else anchor, group, sizes, limit, generator, "anchor"
    )
    instance = _token_instance(structure, layout)
    tokens = np.arange(len(base))
    scores = np.full(len(base), np.inf)
    for value in np.unique(instance):
        members = tokens[instance == value]
        seeds = members[base[members]]
        if not len(seeds):
            seeds = members[[generator.integers(len(members))]]
        scores[members] = np.abs(members[:, None] - seeds[None, :]).min(axis=1)
    return _complete(base, scores, limit, group, sizes, generator)


def spatial_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    required: NDArray[np.bool_] | None = None,
    anchor: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> NDArray[np.bool_]:
    r"""The tokens nearest to a random resolved token (one from ``anchor`` and ``required`` when given)."""
    required, anchor, limit, group, sizes = _state(layout, max_tokens, required, anchor)
    if limit == len(group):
        return np.ones(limit, dtype=bool)
    coordinates, resolved = _token_coordinates(structure, layout)
    if not resolved.any():
        fail("value_invalid", RequestRef("structure", "no resolved token"), invariant="spatial")
    base = required.copy()
    if anchor is not None and not (base & anchor).any():
        joint = anchor & resolved
        base = _add_group(base, joint if joint.any() else anchor, group, sizes, limit, generator, "anchor")
    base = _add_group(base, resolved, group, sizes, limit, generator, "resolved")
    scores = _spatial_scores(coordinates, resolved, np.flatnonzero(base & resolved))
    return _complete(base, scores, limit, group, sizes, generator)


def interface_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    interface_cutoff: float = 15.0,
    required: NDArray[np.bool_] | None = None,
    anchor: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> NDArray[np.bool_]:
    r"""Around a random resolved contact between distinct instances within ``interface_cutoff``; without any contact,
    ``spatial_crop``."""
    required_mask, anchor, limit, group, sizes = _state(layout, max_tokens, required, anchor)
    if limit == len(group):
        return np.ones(limit, dtype=bool)
    coordinates, resolved = _token_coordinates(structure, layout)
    instance = _token_instance(structure, layout)
    tokens = np.flatnonzero(resolved)
    base_size = int(required_mask.sum())
    selected, seen = None, 0
    step = max(1, _BLOCK // max(len(tokens), 1))
    for start in range(0, len(tokens), step):
        source = tokens[start : start + step]
        close = _distances(coordinates[source], coordinates[tokens]) <= interface_cutoff
        close &= instance[source][:, None] != instance[tokens][None, :]
        close &= np.arange(start, start + len(source))[:, None] < np.arange(len(tokens))[None, :]
        pairs = np.stack((source[np.nonzero(close)[0]], tokens[np.nonzero(close)[1]]), axis=1)
        if not len(pairs):
            continue
        pair_groups = group[pairs]
        growth = ((~required_mask[pairs]) * sizes[pair_groups]).sum(axis=1)
        growth -= ((pair_groups[:, 0] == pair_groups[:, 1]) & ~required_mask[pairs[:, 0]]) * sizes[pair_groups[:, 0]]
        selected, seen = _reservoir(selected, seen, pairs[base_size + growth <= limit], generator)
    if seen == 0:
        return spatial_crop(
            structure, layout, max_tokens=max_tokens, required=required_mask, anchor=anchor, generator=generator
        )
    if selected is None:
        fail("value_invalid", RequestRef("max_tokens", str(max_tokens)), invariant="interface_fits")
    base = required_mask | np.isin(group, group[selected])
    if anchor is not None:
        base = _add_group(base, anchor, group, sizes, limit, generator, "anchor")
    scores = _spatial_scores(coordinates, resolved, np.flatnonzero(base))
    return _complete(base, scores, limit, group, sizes, generator)


def instance_anchored_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    cropper,
    instances: Sequence[str],
    required: NDArray[np.bool_] | None = None,
    anchor: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> NDArray[np.bool_]:
    r"""A configured policy (accepting ``anchor``) whose anchors are restricted to the named instances."""
    if instances:
        restricted = np.isin(_token_instance(structure, layout), structure.system.instance_index(list(instances)))
        anchor = restricted if anchor is None else restricted & check_mask("anchor", anchor, len(restricted))
    return cropper(structure, layout, required=required, anchor=anchor, generator=generator)


def interface_anchored_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    cropper,
    instances: Sequence[str],
    interface_cutoff: float = 15.0,
    required: NDArray[np.bool_] | None = None,
    anchor: NDArray[np.bool_] | None = None,
    generator: np.random.Generator,
) -> NDArray[np.bool_]:
    r"""A configured policy that must keep one resolved contact between the two named instances."""
    if len(instances) != 2 or instances[0] == instances[1]:
        fail("value_invalid", RequestRef("instances", ",".join(instances)), invariant="two_distinct")
    left, right = structure.system.instance_index(list(instances))
    coordinates, resolved = _token_coordinates(structure, layout)
    instance = _token_instance(structure, layout)
    first, second = (np.flatnonzero(resolved & (instance == value)) for value in (left, right))
    selected, seen = None, 0
    step = max(1, _BLOCK // max(len(second), 1))
    for start in range(0, len(first), step):
        source = first[start : start + step]
        close = np.nonzero(_distances(coordinates[source], coordinates[second]) <= interface_cutoff)
        selected, seen = _reservoir(selected, seen, np.stack((source[close[0]], second[close[1]]), axis=1), generator)
    if selected is None:
        fail(
            "value_invalid", RequestRef("instances", ",".join(instances)), invariant="contact", cutoff=interface_cutoff
        )
    count = len(layout.token_component)
    required = np.zeros(count, dtype=bool) if required is None else check_mask("required", required, count).copy()
    required[selected] = True
    return cropper(structure, layout, required=required, anchor=anchor, generator=generator)
