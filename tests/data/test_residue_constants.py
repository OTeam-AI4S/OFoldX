from __future__ import annotations

import numpy as np

from ofoldx.data import residue_constants as rc


def test_derived_array_shapes() -> None:
    assert rc.restype_atom14_to_rigid_group.shape == (21, 14)
    assert rc.restype_atom14_mask.shape == (21, 14)
    assert rc.restype_atom14_rigid_group_positions.shape == (21, 14, 3)
    assert rc.restype_rigid_group_default_frame.shape == (21, 8, 4, 4)


def test_backbone_atoms_present_for_every_residue() -> None:
    # N/CA/C/O occupy atom14 slots 0..3 for all 20 standard residues.
    assert np.all(rc.restype_atom14_mask[:20, :4] == 1.0)


def test_default_frames_are_rigid() -> None:
    # Populated rigid-group frames (nonzero bottom-right) must be proper homogeneous transforms;
    # unused chi frames for shorter side chains stay all-zeros.
    frames = rc.restype_rigid_group_default_frame
    populated = frames[..., 3, 3] == 1.0
    assert np.allclose(frames[populated][..., 3, :], np.array([0.0, 0.0, 0.0, 1.0]))
    rotations = frames[populated][..., :3, :3]
    identity = np.einsum("...ij,...kj->...ik", rotations, rotations)
    assert np.allclose(identity, np.eye(3), atol=1e-4)


def test_glycine_has_no_chi_atoms() -> None:
    glycine = rc.restypes.index("G")
    assert rc.restype_atom14_mask[glycine, 4:].sum() == 0.0  # only backbone N/CA/C/O
