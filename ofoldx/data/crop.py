r"""
Shared structural-token crop policies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor

from .structure import Structure
from .tokenization import TokenLayout, select_tokens

_INTERFACE_DISTANCE_BLOCK_ELEMENTS = 1 << 20


class Cropper(Protocol):
    r"""A configured crop policy over one canonical structural token axis."""

    def __call__(
        self,
        structure: Structure,
        layout: TokenLayout,
        *,
        required_token_mask: Tensor | None = None,
        anchor_token_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Return sorted, unique indices into ``layout``."""

        ...


def crop_token_layout(
    structure: Structure,
    layout: TokenLayout,
    cropper: Cropper,
    *,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> TokenLayout:
    r"""Apply one configured crop policy."""

    selection = cropper(
        structure,
        layout,
        required_token_mask=required_token_mask,
        anchor_token_mask=anchor_token_mask,
        generator=generator,
    )
    return select_tokens(layout, selection)


def instance_anchored_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    cropper: Cropper,
    instance_ids: Sequence[str],
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Restrict a configured cropper's anchors to selected structure instances."""

    if not instance_ids:
        return cropper(
            structure,
            layout,
            required_token_mask=required_token_mask,
            anchor_token_mask=anchor_token_mask,
            generator=generator,
        )

    selected_instances = layout.token_instance_index.new_tensor(_instance_indices(structure, instance_ids))
    anchor = torch.isin(layout.token_instance_index, selected_instances)
    if anchor_token_mask is not None:
        processor_anchor = anchor_token_mask.to(device=anchor.device, dtype=torch.bool)
        if processor_anchor.shape != anchor.shape:
            raise ValueError(
                f"anchor_token_mask must have shape {tuple(anchor.shape)}, got {tuple(processor_anchor.shape)}"
            )
        anchor &= processor_anchor
    return cropper(
        structure,
        layout,
        required_token_mask=required_token_mask,
        anchor_token_mask=anchor,
        generator=generator,
    )


def interface_anchored_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    cropper: Cropper,
    instance_ids: Sequence[str],
    interface_cutoff: float = 15.0,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Require a resolved contact from one exact instance pair in a configured crop."""

    if len(instance_ids) != 2 or instance_ids[0] == instance_ids[1]:
        raise ValueError("an interface crop requires two distinct instance IDs")
    left, right = _instance_indices(structure, instance_ids)
    coordinates, resolved = _token_coordinates(structure, layout)
    left_tokens = torch.nonzero(
        resolved & (layout.token_instance_index == left),
        as_tuple=False,
    ).flatten()
    right_tokens = torch.nonzero(
        resolved & (layout.token_instance_index == right),
        as_tuple=False,
    ).flatten()

    selected: Tensor | None = None
    seen = 0
    if right_tokens.numel():
        block_size = max(1, _INTERFACE_DISTANCE_BLOCK_ELEMENTS // right_tokens.numel())
        for start in range(0, left_tokens.numel(), block_size):
            source = left_tokens[start : start + block_size]
            contacts = (torch.cdist(coordinates[source], coordinates[right_tokens]) <= interface_cutoff).nonzero(
                as_tuple=False
            )
            pairs = torch.stack(
                (source[contacts[:, 0]], right_tokens[contacts[:, 1]]),
                dim=-1,
            )
            selected, seen = _reservoir_choice(selected, seen, pairs, generator)
    if selected is None:
        raise ValueError(
            f"sampled interface {instance_ids[0]!r}-{instance_ids[1]!r} "
            f"has no resolved contact within {interface_cutoff:g} A"
        )

    num_tokens = layout.token_component_index.numel()
    if required_token_mask is None:
        required = torch.zeros(num_tokens, dtype=torch.bool, device=layout.token_component_index.device)
    else:
        required = required_token_mask.to(device=layout.token_component_index.device, dtype=torch.bool)
        if required.shape != (num_tokens,):
            raise ValueError(f"required_token_mask must have shape ({num_tokens},), got {tuple(required.shape)}")
        required = required.clone()
    required[selected] = True
    return cropper(
        structure,
        layout,
        required_token_mask=required,
        anchor_token_mask=anchor_token_mask,
        generator=generator,
    )


def _instance_indices(structure: Structure, instance_ids: Sequence[str]) -> tuple[int, ...]:
    index = {structure.system.instance_id(offset): offset for offset in range(structure.system.instance_symbol.numel())}
    unknown = [instance_id for instance_id in dict.fromkeys(instance_ids) if instance_id not in index]
    if unknown:
        raise ValueError(f"crop instance(s) do not exist: {', '.join(unknown)}")
    return tuple(index[instance_id] for instance_id in instance_ids)


def _random_choice(candidates: Tensor, generator: torch.Generator | None) -> Tensor:
    device = candidates.device if generator is None else generator.device
    offset = torch.randint(candidates.numel(), (), device=device, generator=generator)
    return candidates[offset.to(candidates.device)]


def resolved_token_mask(structure: Structure, layout: TokenLayout) -> Tensor:
    r"""Return tokens containing at least one resolved canonical atom."""

    num_tokens = layout.token_component_index.numel()
    atom_counts = layout.token_atom_ptr[1:] - layout.token_atom_ptr[:-1]
    atom_token_index = torch.repeat_interleave(
        torch.arange(num_tokens, device=atom_counts.device),
        atom_counts,
    )
    resolved = torch.zeros(num_tokens, dtype=torch.bool, device=atom_counts.device)
    resolved[atom_token_index[structure.coordinate_mask[layout.atom_index]]] = True
    return resolved


def _crop_state(
    layout: TokenLayout,
    max_tokens: int,
    required_token_mask: Tensor | None,
    anchor_token_mask: Tensor | None,
) -> tuple[Tensor, Tensor | None, int, Tensor, Tensor]:
    num_tokens = layout.token_component_index.numel()
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if num_tokens == 0:
        raise ValueError("cannot crop an empty token layout")

    _, component_group, group_sizes = torch.unique(
        layout.token_component_index,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )

    if required_token_mask is None:
        required = torch.zeros(num_tokens, dtype=torch.bool, device=layout.token_component_index.device)
    else:
        required = required_token_mask.to(device=layout.token_component_index.device, dtype=torch.bool)
        if required.shape != (num_tokens,):
            raise ValueError(f"required_token_mask must have shape ({num_tokens},), got {tuple(required.shape)}")
    required_groups = torch.zeros_like(group_sizes, dtype=torch.bool)
    required_groups[component_group[required]] = True
    required = required_groups[component_group]

    limit = min(max_tokens, num_tokens)
    required_count = int(required.sum())
    if required_count > limit:
        raise ValueError(f"{required_count} tokens in required component groups do not fit in a {limit}-token crop")

    anchor = None
    if anchor_token_mask is not None:
        anchor = anchor_token_mask.to(device=layout.token_component_index.device, dtype=torch.bool)
        if anchor.shape != (num_tokens,):
            raise ValueError(f"anchor_token_mask must have shape ({num_tokens},), got {tuple(anchor.shape)}")
        if not bool(anchor.any()):
            raise ValueError("anchor_token_mask does not select any token")
        if required_count == limit and not bool((required & anchor).any()):
            raise ValueError("the required component groups fill the crop but contain no anchor token")
        if not bool((required & anchor).any()):
            anchor_groups = torch.zeros_like(group_sizes, dtype=torch.bool)
            anchor_groups[component_group[anchor]] = True
            if not bool((group_sizes[anchor_groups] <= limit - required_count).any()):
                raise ValueError(f"no complete anchor component group fits in a {limit}-token crop")
    return required, anchor, limit, component_group, group_sizes


def _add_candidate_group(
    base: Tensor,
    candidates: Tensor,
    component_group: Tensor,
    group_sizes: Tensor,
    limit: int,
    generator: torch.Generator | None,
    error: str,
) -> Tensor:
    if bool((base & candidates).any()):
        return base

    candidate_groups = torch.zeros_like(group_sizes, dtype=torch.bool)
    candidate_groups[component_group[candidates]] = True
    candidate_groups &= group_sizes <= limit - int(base.sum())
    feasible = candidates & candidate_groups[component_group]
    if not bool(feasible.any()):
        raise ValueError(error)
    group = component_group[_random_choice(torch.nonzero(feasible, as_tuple=False).flatten(), generator)]
    return base | (component_group == group)


def _selection_base(
    required: Tensor,
    anchor: Tensor | None,
    component_group: Tensor,
    group_sizes: Tensor,
    limit: int,
    generator: torch.Generator | None,
) -> Tensor:
    candidates = torch.ones_like(required) if anchor is None else anchor
    error = (
        f"no complete component group fits in a {limit}-token crop"
        if anchor is None
        else f"no complete anchor component group fits in a {limit}-token crop"
    )
    return _add_candidate_group(
        required.clone(),
        candidates,
        component_group,
        group_sizes,
        limit,
        generator,
        error,
    )


def _complete_selection(
    base: Tensor,
    scores: Tensor,
    limit: int,
    component_group: Tensor,
    group_sizes: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    selected_groups = torch.zeros_like(group_sizes, dtype=torch.bool)
    selected_groups[component_group[base]] = True
    remaining = limit - int(group_sizes[selected_groups].sum())
    if remaining:
        candidates = torch.nonzero(~selected_groups, as_tuple=False).flatten()
        device = candidates.device if generator is None else generator.device
        permutation = torch.randperm(candidates.numel(), device=device, generator=generator).to(candidates.device)
        candidates = candidates[permutation]
        group_scores = scores.new_full(group_sizes.shape, torch.inf)
        group_scores.scatter_reduce_(0, component_group, scores, reduce="amin", include_self=True)
        order = torch.argsort(group_scores[candidates], stable=True)
        sizes = group_sizes.detach().cpu().tolist()
        additions = []
        for group in candidates[order].detach().cpu().tolist():
            if sizes[group] <= remaining:
                additions.append(group)
                remaining -= sizes[group]
                if not remaining:
                    break
        if additions:
            selected_groups[group_sizes.new_tensor(additions)] = True
    return torch.nonzero(selected_groups[component_group], as_tuple=False).flatten()


def _token_coordinates(structure: Structure, layout: TokenLayout) -> tuple[Tensor, Tensor]:
    num_tokens = layout.token_component_index.numel()
    atom_counts = layout.token_atom_ptr[1:] - layout.token_atom_ptr[:-1]
    atom_token_index = torch.repeat_interleave(
        torch.arange(num_tokens, device=atom_counts.device),
        atom_counts,
    )
    resolved_atom = structure.coordinate_mask[layout.atom_index]
    resolved_token_index = atom_token_index[resolved_atom]
    coordinates = structure.coordinates[layout.atom_index[resolved_atom]].float()

    token_coordinates = coordinates.new_zeros((num_tokens, 3))
    token_coordinates.index_add_(0, resolved_token_index, coordinates)
    resolved_counts = coordinates.new_zeros(num_tokens)
    resolved_counts.index_add_(0, resolved_token_index, torch.ones_like(resolved_token_index, dtype=coordinates.dtype))
    resolved = resolved_counts > 0
    token_coordinates[resolved] /= resolved_counts[resolved, None]
    center_atom_index = layout.token_center_atom_index
    resolved_center = structure.coordinate_mask[center_atom_index]
    token_coordinates[resolved_center] = structure.coordinates[center_atom_index[resolved_center]].float()
    return token_coordinates, resolved


def _spatial_scores(coordinates: Tensor, resolved: Tensor, seeds: Tensor) -> Tensor:
    scores = coordinates.new_full((coordinates.shape[0],), torch.inf)
    resolved_seeds = seeds[resolved[seeds]]
    scores[resolved] = torch.cdist(coordinates[resolved], coordinates[resolved_seeds]).min(dim=-1).values
    return scores


def _reservoir_choice(
    selected: Tensor | None,
    seen: int,
    candidates: Tensor,
    generator: torch.Generator | None,
) -> tuple[Tensor | None, int]:
    count = candidates.shape[0]
    if not count:
        return selected, seen
    total = seen + count
    device = candidates.device if generator is None else generator.device
    offset = torch.randint(total, (), device=device, generator=generator)
    if int(offset) < count:
        selected = candidates[offset.to(candidates.device)]
    return selected, total


def continuous_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Crop contiguous token-axis neighbourhoods.

    Required tokens seed their instances. An anchor mask, when supplied, guarantees that
    the selection contains at least one token from the caller-defined scoreable set.
    """

    required, anchor, limit, component_group, group_sizes = _crop_state(
        layout,
        max_tokens,
        required_token_mask,
        anchor_token_mask,
    )
    if limit == layout.token_component_index.numel():
        return torch.arange(limit, device=layout.token_component_index.device)

    base = _selection_base(required, anchor, component_group, group_sizes, limit, generator)
    token_index = torch.arange(base.numel(), device=base.device)
    scores = torch.full((base.numel(),), torch.inf, device=base.device)
    for instance in torch.unique(layout.token_instance_index):
        instance_tokens = token_index[layout.token_instance_index == instance]
        seeds = instance_tokens[base[instance_tokens]]
        if not seeds.numel():
            seeds = _random_choice(instance_tokens, generator).reshape(1)
        scores[instance_tokens] = (instance_tokens[:, None] - seeds).abs().min(dim=-1).values.float()
    return _complete_selection(base, scores, limit, component_group, group_sizes, generator)


def spatial_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Crop tokens nearest to a randomly selected resolved token."""

    required, anchor, limit, component_group, group_sizes = _crop_state(
        layout,
        max_tokens,
        required_token_mask,
        anchor_token_mask,
    )
    if limit == layout.token_component_index.numel():
        return torch.arange(limit, device=layout.token_component_index.device)

    coordinates, coordinate_mask = _token_coordinates(structure, layout)
    if not bool(coordinate_mask.any()):
        raise ValueError("spatial crop requires at least one resolved token")

    base = required.clone()
    needs_resolved = not bool((base & coordinate_mask).any())
    needs_anchor = anchor is not None and not bool((base & anchor).any())
    if needs_resolved and needs_anchor:
        group_has_resolved = torch.zeros_like(group_sizes, dtype=torch.bool)
        group_has_resolved[component_group[coordinate_mask]] = True
        group_has_anchor = torch.zeros_like(group_sizes, dtype=torch.bool)
        assert anchor is not None
        group_has_anchor[component_group[anchor]] = True
        remaining = limit - int(base.sum())
        joint_groups = group_has_resolved & group_has_anchor & (group_sizes <= remaining)
        if bool(joint_groups.any()):
            base = _add_candidate_group(
                base,
                coordinate_mask & joint_groups[component_group],
                component_group,
                group_sizes,
                limit,
                generator,
                "no complete resolved anchor component group fits in the crop",
            )
        else:
            anchor_reserve = int(group_sizes[group_has_anchor].min())
            base = _add_candidate_group(
                base,
                coordinate_mask,
                component_group,
                group_sizes,
                limit - anchor_reserve,
                generator,
                "no complete resolved and anchor component groups fit in the crop",
            )
            base = _add_candidate_group(
                base,
                anchor,
                component_group,
                group_sizes,
                limit,
                generator,
                "no complete resolved and anchor component groups fit in the crop",
            )
    elif needs_resolved:
        base = _add_candidate_group(
            base,
            coordinate_mask,
            component_group,
            group_sizes,
            limit,
            generator,
            "the crop constraints leave no room for a complete resolved component group",
        )
    elif needs_anchor:
        assert anchor is not None
        base = _add_candidate_group(
            base,
            anchor,
            component_group,
            group_sizes,
            limit,
            generator,
            "the crop constraints leave no room for a complete anchor component group",
        )
    resolved_seeds = torch.nonzero(base & coordinate_mask, as_tuple=False).flatten()
    scores = _spatial_scores(coordinates, coordinate_mask, resolved_seeds)
    return _complete_selection(base, scores, limit, component_group, group_sizes, generator)


def interface_crop(
    structure: Structure,
    layout: TokenLayout,
    *,
    max_tokens: int,
    interface_cutoff: float = 15.0,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""
    Crop around a random resolved contact between distinct molecular instances.

    Structures without an inter-instance contact inside ``interface_cutoff`` fall back to
    :func:`spatial_crop`.
    """

    required, anchor, limit, component_group, group_sizes = _crop_state(
        layout,
        max_tokens,
        required_token_mask,
        anchor_token_mask,
    )
    if limit == layout.token_component_index.numel():
        return torch.arange(limit, device=layout.token_component_index.device)

    coordinates, resolved = _token_coordinates(structure, layout)
    resolved_tokens = torch.nonzero(resolved, as_tuple=False).flatten()
    if not resolved_tokens.numel():
        raise ValueError("interface crop requires at least one resolved token")
    base = required.clone()
    base_count = int(base.sum())
    needs_anchor = anchor is not None and not bool((base & anchor).any())
    group_has_anchor = torch.zeros_like(group_sizes, dtype=torch.bool)
    if anchor is not None:
        group_has_anchor[component_group[anchor]] = True
    smallest_anchor_group = group_sizes[group_has_anchor].min() if needs_anchor else group_sizes.new_zeros(())
    selected_pair: Tensor | None = None
    selected_anchor_pair: Tensor | None = None
    seen_pairs = 0
    seen_anchor_pairs = 0
    has_contact = False
    num_resolved = resolved_tokens.numel()
    block_size = max(1, _INTERFACE_DISTANCE_BLOCK_ELEMENTS // num_resolved)
    target_position = torch.arange(num_resolved, device=resolved_tokens.device)
    resolved_instances = layout.token_instance_index[resolved_tokens]
    for start in range(0, num_resolved, block_size):
        stop = min(start + block_size, num_resolved)
        source_tokens = resolved_tokens[start:stop]
        distances = torch.cdist(coordinates[source_tokens], coordinates[resolved_tokens])
        source_position = torch.arange(start, stop, device=resolved_tokens.device)
        contacts = (
            (source_position[:, None] < target_position)
            & (layout.token_instance_index[source_tokens, None] != resolved_instances[None, :])
            & (distances <= interface_cutoff)
        ).nonzero(as_tuple=False)
        if not contacts.numel():
            continue
        has_contact = True
        pairs = torch.stack(
            (source_tokens[contacts[:, 0]], resolved_tokens[contacts[:, 1]]),
            dim=-1,
        )
        pair_groups = component_group[pairs]
        pair_growth = ((~base[pairs]) * group_sizes[pair_groups]).sum(dim=-1)
        same_group = (pair_groups[:, 0] == pair_groups[:, 1]) & ~base[pairs[:, 0]]
        pair_growth -= same_group * group_sizes[pair_groups[:, 0]]
        pair_has_anchor = group_has_anchor[pair_groups].any(dim=-1) if needs_anchor else None
        if pair_has_anchor is not None:
            pair_growth += (~pair_has_anchor) * smallest_anchor_group
        feasible = base_count + pair_growth <= limit
        pairs = pairs[feasible]
        if not pairs.numel():
            continue
        selected_pair, seen_pairs = _reservoir_choice(selected_pair, seen_pairs, pairs, generator)
        if pair_has_anchor is not None:
            pair_has_anchor = pair_has_anchor[feasible]
            selected_anchor_pair, seen_anchor_pairs = _reservoir_choice(
                selected_anchor_pair,
                seen_anchor_pairs,
                pairs[pair_has_anchor],
                generator,
            )

    if not has_contact:
        return spatial_crop(
            structure,
            layout,
            max_tokens=max_tokens,
            required_token_mask=required,
            anchor_token_mask=anchor,
            generator=generator,
        )
    pair = selected_anchor_pair if selected_anchor_pair is not None else selected_pair
    if pair is None:
        raise ValueError("the crop constraints leave no room for complete interface component groups")
    pair_mask = torch.zeros_like(base)
    pair_mask[pair] = True
    pair_groups = torch.zeros_like(group_sizes, dtype=torch.bool)
    pair_groups[component_group[pair_mask]] = True
    base |= pair_groups[component_group]
    base = _selection_base(base, anchor, component_group, group_sizes, limit, generator)
    scores = _spatial_scores(coordinates, resolved, torch.nonzero(base, as_tuple=False).flatten())
    return _complete_selection(base, scores, limit, component_group, group_sizes, generator)
