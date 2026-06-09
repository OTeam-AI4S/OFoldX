r"""
Rigid alignment and RMSD scoring.

Kabsch-style least-squares superposition of two coordinate sets and the aligned RMSD used to score a
predicted structure against a reference. Faithful port of upstream AlphaFold3
``model/scoring/alignment.py`` (``transform_ls``, ``align``, ``deviations_from_coords``,
``rmsd_from_coords``); pure NumPy, so it runs and parity-checks without external tools.
"""

from __future__ import annotations

import numpy as np


def transform_ls(x: np.ndarray, b: np.ndarray, *, allow_reflection: bool = False) -> np.ndarray:
    r"""
    Return the least-squares rotation mapping ``x`` onto ``b``.

    Solves ``A x^T = b^T`` for the rotation ``A`` via the SVD of the cross-covariance, optionally
    forbidding reflection by fixing the sign of the last singular direction to keep ``det(A) > 0``.

    Args:
        x: ``[N, D]`` source coordinates.
        b: ``[N, D]`` target coordinates.
        allow_reflection: If ``True``, allow the result to reflect as well as rotate.

    Returns:
        ``[D, D]`` rotation matrix mapping ``x`` onto ``b``.

    Raises:
        ValueError: If ``x`` has fewer columns than ``b``, or the two arrays have different row counts.

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> transform_ls(x, x).round(4).tolist()
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """

    if x.shape[1] < b.shape[1]:
        raise ValueError(f"x dimension {x.shape[1]} must be at least b dimension {b.shape[1]}")
    if b.shape[0] != x.shape[0]:
        raise ValueError(f"x and b must have the same number of points, got {x.shape[0]} and {b.shape[0]}")
    cross_covariance = np.dot(b.transpose(), x) / b.shape[0]
    u, _, v = np.linalg.svd(cross_covariance)
    rotation = np.dot(u, v)
    if not allow_reflection:
        flip = np.ones((v.shape[1], 1))
        flip[v.shape[1] - 1, 0] = np.sign(np.linalg.det(rotation))
        rotation = np.dot(u, v * flip)
    return rotation


def align(*, x: np.ndarray, y: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray) -> np.ndarray:
    r"""
    Rigidly superpose ``x`` onto ``y`` using the selected index correspondence.

    Args:
        x: ``[N, D]`` coordinates to transform.
        y: ``[N, D]`` reference coordinates.
        x_indices: Indices into ``x`` used to fit the transform.
        y_indices: Matching indices into ``y``; must be the same length as ``x_indices``.

    Returns:
        ``[N, D]`` coordinates from applying the fitted rigid transform to all of ``x``.

    Raises:
        ValueError: If the two index arrays differ in length.

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> y = x + np.array([1.0, 0.0, 0.0])
        >>> idx = np.array([0, 1, 2])
        >>> align(x=x, y=y, x_indices=idx, y_indices=idx).round(4).tolist()
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    """

    if len(x_indices) != len(y_indices):
        raise ValueError(f"number of aligned indices must match, got x: {len(x_indices)}, y: {len(y_indices)}")
    x_mean = np.mean(x[x_indices, :], axis=0)
    y_mean = np.mean(y[y_indices, :], axis=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    rotation = transform_ls(centered_x[x_indices, :], centered_y[y_indices, :])
    return np.dot(centered_x, rotation.transpose()) + y_mean


def deviations_from_coords(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    align_idxs: np.ndarray | None = None,
    include_idxs: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Return per-atom deviations after aligning ``decoy_coords`` onto ``gt_coords``.

    Args:
        decoy_coords: ``[M, 3]`` predicted coordinates.
        gt_coords: ``[M, 3]`` reference coordinates.
        align_idxs: Indices used to fit the alignment; defaults to ``include_idxs``.
        include_idxs: Indices scored after alignment; defaults to all atoms.

    Returns:
        ``[len(include_idxs)]`` per-atom distances between the aligned decoy and reference.

    Raises:
        ValueError: If the two coordinate arrays differ in shape.

    Examples:
        >>> import numpy as np
        >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> deviations_from_coords(coords, coords).round(4).tolist()
        [0.0, 0.0, 0.0]
    """

    if decoy_coords.shape != gt_coords.shape:
        raise ValueError(f"decoy and gt coords must match, got {decoy_coords.shape} and {gt_coords.shape}")
    if include_idxs is None:
        include_idxs = np.arange(decoy_coords.shape[0])
    if align_idxs is None:
        align_idxs = include_idxs
    aligned = align(x=decoy_coords, y=gt_coords, x_indices=align_idxs, y_indices=align_idxs)
    return np.linalg.norm(aligned[include_idxs] - gt_coords[include_idxs], axis=1)


def rmsd_from_coords(
    decoy_coords: np.ndarray,
    gt_coords: np.ndarray,
    align_idxs: np.ndarray | None = None,
    include_idxs: np.ndarray | None = None,
) -> float:
    r"""
    Return the aligned RMSD between predicted and reference coordinates.

    Args:
        decoy_coords: ``[M, 3]`` predicted coordinates.
        gt_coords: ``[M, 3]`` reference coordinates.
        align_idxs: Indices used to fit the alignment; defaults to ``include_idxs``.
        include_idxs: Indices scored after alignment; defaults to all atoms.

    Returns:
        Aligned root-mean-square deviation.

    Examples:
        >>> import numpy as np
        >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> round(rmsd_from_coords(coords, coords), 4)
        0.0
    """

    deviations = deviations_from_coords(decoy_coords, gt_coords, align_idxs, include_idxs)
    return float(np.sqrt(np.mean(np.square(deviations))))
