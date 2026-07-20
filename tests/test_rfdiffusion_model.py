r"""
RFdiffusion combined-length (``ContigMap.length``) behaviour.

RFdiffusion's ``ContigMap.length`` (§3.7) is threaded onto ``RFdiffusionLayout.combined_length`` as the
combined generated length across a placement's ``GeneratedSegment`` rows. Upstream re-samples the
per-subcon generated lengths until their inpaint total lands in the requested ``[min, max]`` window and
errors when it never can (``contigs.py`` ``get_sampled_mask``); ``_sample_variable_lengths`` honours it
the same way -- reject-and-resample the per-segment draws, and raise rather than silently drop a window
that cannot be realised. These tests lock that contract on the CPU length-sampling path (no network,
weights, or GPU): the window is honoured (per-segment and coupled across segments), an unrealisable
window raises a clear error, and ``None`` leaves the draw unconstrained within each segment's own bounds.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from ofoldx.data.placement import GeneratedSegment
from ofoldx.models.rfdiffusion.model import (
    RFdiffusionInput,
    RFdiffusionLayout,
    RFdiffusionPlacements,
    _sample_variable_lengths,
)
from ofoldx.models.rfdiffusion.processor import _resolve_placement


def _batch(placement: RFdiffusionPlacements, combined_length: tuple[int, int] | None) -> RFdiffusionInput:
    output_residues, output_instance_ids = _resolve_placement(placement)
    length = len(output_residues)
    mask = torch.zeros((1, length), dtype=torch.bool)
    layout = RFdiffusionLayout(
        system=None,
        structure=None,
        templates=(),
        conditions=(placement,),
        output_residues=output_residues,
        output_instance_ids=output_instance_ids,
        combined_length=combined_length,
    )
    return RFdiffusionInput(
        noisy_coords=torch.zeros((1, length, 14, 3)),
        sequence=torch.zeros((1, length, 22)),
        residue_index=torch.zeros((1, length), dtype=torch.int64),
        fixed_structure_mask=mask,
        fixed_sequence_mask=mask.clone(),
        hotspot_mask=mask.clone(),
        cyclic_mask=mask.clone(),
        secondary_structure=torch.zeros((1, length, 4)),
        block_adjacency=torch.zeros((1, length, length, 3)),
        timestep=torch.tensor([50], dtype=torch.int64),
        layout=layout,
    )


def _generated_total(batch: RFdiffusionInput) -> int:
    # Every realised generated position is a SymbolicResidueSite; existing/fixed positions are not.
    return sum(type(residue).__name__ == "SymbolicResidueSite" for residue in batch.layout.output_residues)


def _single_segment() -> RFdiffusionPlacements:
    # One free segment realising anywhere in [50, 150].
    return RFdiffusionPlacements(placements={"chain_1": {"chain_1:0": GeneratedSegment("protein", 50, 150)}})


def _two_segments() -> RFdiffusionPlacements:
    # Two free segments on separate chains, each realising in [20, 60] (combined [40, 120]).
    return RFdiffusionPlacements(
        placements={
            "chain_1": {"chain_1:0": GeneratedSegment("protein", 20, 60)},
            "chain_2": {"chain_2:0": GeneratedSegment("protein", 20, 60)},
        }
    )


@pytest.mark.parametrize(
    ("placement_factory", "window"),
    [
        (_single_segment, (100, 110)),  # single segment, window strictly inside its own [50, 150]
        (_two_segments, (90, 95)),  # coupled across two segments whose per-segment sum spans [40, 120]
    ],
)
def test_combined_length_window_is_honoured(
    placement_factory: Callable[[], RFdiffusionPlacements], window: tuple[int, int]
) -> None:
    generator = torch.Generator().manual_seed(0)
    low, high = window
    for _ in range(32):
        realized = _generated_total(_sample_variable_lengths(_batch(placement_factory(), window), generator))
        assert low <= realized <= high


@pytest.mark.parametrize("window", [(10, 20), (200, 300)])  # below and above the [50, 150] envelope
def test_infeasible_combined_length_raises(window: tuple[int, int]) -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="cannot be realized"):
        _sample_variable_lengths(_batch(_single_segment(), window), generator)


def test_combined_length_none_is_unconstrained() -> None:
    generator = torch.Generator().manual_seed(0)
    realized = _generated_total(_sample_variable_lengths(_batch(_single_segment(), None), generator))
    assert 50 <= realized <= 150
