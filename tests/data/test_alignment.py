from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.alignment import align, deviations_from_coords, rmsd_from_coords, transform_ls


def _rotation_z(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])


def test_identical_coords_have_zero_rmsd() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 3))
    assert rmsd_from_coords(x, x) == pytest.approx(0.0, abs=1e-9)


def test_rigid_transform_recovers_zero_rmsd() -> None:
    # A pure rotation + translation is undone by the Kabsch alignment -> RMSD ~ 0.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 3))
    y = x @ _rotation_z(0.9).T + np.array([3.0, -2.0, 1.0])
    assert rmsd_from_coords(x, y) == pytest.approx(0.0, abs=1e-6)


def test_transform_ls_recovers_known_rotation() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((30, 3))
    rotation = _rotation_z(0.5)
    y = x @ rotation.T
    # transform_ls centers internally via align; here fit directly on centered clouds.
    fitted = transform_ls(x - x.mean(0), y - y.mean(0))
    np.testing.assert_allclose(fitted, rotation, atol=1e-6)


def test_rmsd_with_noise_is_positive_and_bounded() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((15, 3))
    y = x @ _rotation_z(0.3).T + rng.standard_normal((15, 3)) * 0.1
    value = rmsd_from_coords(x, y)
    assert 0.0 < value < 1.0


def test_align_only_uses_selected_indices() -> None:
    # Aligning on a subset still returns transformed coordinates for all atoms.
    rng = np.random.default_rng(4)
    x = rng.standard_normal((10, 3))
    y = x @ _rotation_z(0.4).T + np.array([1.0, 1.0, 1.0])
    idx = np.array([0, 1, 2, 3, 4])
    aligned = align(x=x, y=y, x_indices=idx, y_indices=idx)
    np.testing.assert_allclose(aligned[idx], y[idx], atol=1e-6)


def test_align_rejects_mismatched_index_lengths() -> None:
    x = np.zeros((5, 3))
    with pytest.raises(ValueError, match="must match"):
        align(x=x, y=x, x_indices=np.array([0, 1]), y_indices=np.array([0]))


def test_deviations_shape_matches_included_indices() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, 3))
    deviations = deviations_from_coords(x, x, include_idxs=np.array([0, 2, 4]))
    assert deviations.shape == (3,)
    np.testing.assert_allclose(deviations, 0.0, atol=1e-9)
