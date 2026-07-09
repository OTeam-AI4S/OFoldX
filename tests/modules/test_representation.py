from __future__ import annotations

import pytest
import torch

from ofoldx.modules.geometry import align_coords, random_rotation_frames
from ofoldx.modules.representation import Coords, Frames, Pair, Single


def test_coords_axis_check():
    Coords(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="last axis must be 3"):
        Coords(torch.zeros(4, 2))


def test_coords_centered_zeros_the_mean():
    coords = Coords(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert torch.allclose(coords.centered().tensor.mean(dim=-2), torch.zeros(3), atol=1e-6)


def test_frames_identity_apply_is_noop():
    local = Coords(torch.randn(2, 5, 3))
    assert torch.allclose(Frames.identity(2).apply(local).tensor, local.tensor, atol=1e-5)


def test_frames_axis_checks():
    with pytest.raises(ValueError, match="rotations must end with"):
        Frames(torch.zeros(3, 2), torch.zeros(3))
    with pytest.raises(ValueError, match="share leading"):
        Frames(torch.zeros(2, 3, 3), torch.zeros(3, 3))


def test_frames_inverse_roundtrip():
    frames = random_rotation_frames(3, generator=torch.Generator().manual_seed(0))
    points = Coords(torch.randn(3, 4, 3))
    recovered = frames.inverse().apply(frames.apply(points))
    assert torch.allclose(recovered.tensor, points.tensor, atol=1e-5)


def test_frames_compose_matches_sequential_apply():
    left = random_rotation_frames(2, generator=torch.Generator().manual_seed(1))
    right = random_rotation_frames(2, generator=torch.Generator().manual_seed(2))
    points = Coords(torch.randn(2, 6, 3))
    composed = left.compose(right).apply(points)
    sequential = left.apply(right.apply(points))
    assert torch.allclose(composed.tensor, sequential.tensor, atol=1e-5)


def test_structure_output_coords_accessor():
    # Real consumer #1: the contract output exposes a typed Coords accessor.
    from ofoldx.models.contract import StructurePredictionOutput

    output = StructurePredictionOutput(pred_coords=torch.zeros(1, 7, 3), distogram_logits=torch.zeros(1, 7, 7, 4))
    assert isinstance(output.coords, Coords)
    assert output.coords.num_points == 7


def test_align_coords_typed_wrapper():
    # Real consumer #2: geometry's typed alignment API.
    src = Coords(torch.randn(5, 3))
    aligned = align_coords(src, src, weight=torch.ones(5))
    assert isinstance(aligned, Coords)
    assert torch.allclose(aligned.tensor, src.tensor, atol=1e-4)


def test_single_axes():
    assert Single(torch.zeros(1, 8, 32)).num_tokens == 8
    assert Single(torch.zeros(1, 8, 32)).channels == 32


def test_pair_named_axis_ops():
    pair = Pair(torch.randn(1, 4, 4, 8))
    assert pair.num_tokens == 4
    # transpose_ij is an involution
    assert torch.allclose(pair.transpose_ij().transpose_ij().tensor, pair.tensor)
    # symmetrize is i<->j invariant
    sym = pair.symmetrize()
    assert torch.allclose(sym.tensor, sym.transpose_ij().tensor, atol=1e-6)


def test_pair_requires_square_ij():
    with pytest.raises(ValueError, match="i and j axes must have equal length"):
        Pair(torch.zeros(1, 4, 5, 8))


def test_carriers_are_pytree_nodes():
    # torch.compile traces through the carriers because they flatten to their raw tensor children.
    from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

    coords = Coords(torch.randn(2, 5, 3))
    leaves, spec = tree_flatten(coords)
    assert [leaf.shape for leaf in leaves] == [coords.tensor.shape]
    rebuilt = tree_unflatten(leaves, spec)
    assert isinstance(rebuilt, Coords)
    assert torch.equal(rebuilt.tensor, coords.tensor)

    frames = Frames.identity(2)
    frame_leaves, frame_spec = tree_flatten(frames)
    assert len(frame_leaves) == 2  # rotations + translations
    doubled = tree_map(lambda t: t * 2, frames)
    assert isinstance(doubled, Frames)
    assert torch.equal(doubled.translations, frames.translations * 2)

    pair = Pair(torch.randn(1, 4, 4, 8))
    single = Single(torch.randn(1, 4, 8))
    for carrier in (pair, single):
        flat, carrier_spec = tree_flatten(carrier)
        assert len(flat) == 1
        assert isinstance(tree_unflatten(flat, carrier_spec), type(carrier))
