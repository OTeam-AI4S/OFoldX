from __future__ import annotations

import numpy as np

from ofoldx.data import residue_constants as rc

# --------------------------------------------------------------------------------------
# Vendored plain-Python vocabulary tables (the module's real public surface)
# --------------------------------------------------------------------------------------


def test_restype_vocabulary_is_the_twenty_standard_residues() -> None:
    assert rc.restypes == list("ARNDCQEGHILKMFPSTWYV")
    assert rc.restype_num == 20
    assert len(rc.restypes) == rc.restype_num


def test_atom_type_ordering_is_the_thirty_seven_af2_names() -> None:
    assert rc.atom_type_num == 37
    assert len(rc.atom_types) == rc.atom_type_num
    assert len(set(rc.atom_types)) == rc.atom_type_num
    assert rc.atom_order == {atom_type: index for index, atom_type in enumerate(rc.atom_types)}
    assert rc.atom_order["N"] == 0
    assert rc.atom_types[0:4] == ["N", "CA", "C", "CB"]
    assert rc.atom_types[-1] == "OXT"
    assert rc.atom_order["OXT"] == 36


def test_three_and_one_letter_maps_are_consistent_inverses() -> None:
    assert len(rc.restype_1to3) == 20
    assert set(rc.restype_1to3) == set(rc.restypes)
    assert rc.restype_3to1 == {three: one for one, three in rc.restype_1to3.items()}
    assert rc.restype_1to3["A"] == "ALA"


def test_unknown_residue_map_extends_the_standard_map_with_x() -> None:
    assert rc.restype_1to3_with_unknown == {**rc.restype_1to3, "X": "UNK"}
    assert rc.restype_1to3_with_unknown["X"] == "UNK"
    assert len(rc.restype_1to3_with_unknown) == 21
    # The standard map itself stays free of the unknown placeholder.
    assert "X" not in rc.restype_1to3


def test_nucleic_one_letter_maps_carry_unknown_placeholders() -> None:
    assert rc.rna_1to3 == {"A": "A", "G": "G", "C": "C", "U": "U", "X": "N", "I": "I", "N": "N"}
    assert rc.dna_1to3 == {
        "A": "DA",
        "G": "DG",
        "C": "DC",
        "T": "DT",
        "X": "DN",
        "I": "DI",
        "N": "DN",
        "U": "DU",
    }


def test_leaving_atom_and_hydrogen_tables() -> None:
    assert rc.LEAVING_ATOMS == {
        "protein": frozenset({"OXT"}),
        "rna": frozenset({"OP3"}),
        "dna": frozenset({"OP3"}),
    }
    assert rc.HYDROGEN_ELEMENTS == frozenset({"H", "D"})


def test_atom14_name_table_has_fixed_width_and_backbone_prefix() -> None:
    for name in rc.restype_1to3.values():
        atom14 = rc.restype_name_to_atom14_names[name]
        assert len(atom14) == 14
        assert atom14[:4] == ["N", "CA", "C", "O"]
    # The UNK row is present but carries no atom14 names.
    assert rc.restype_name_to_atom14_names["UNK"] == [""] * 14


def test_chi_mask_counts_match_declared_chi_atom_sets() -> None:
    assert len(rc.chi_angles_mask) == rc.restype_num
    for letter, mask_row in zip(rc.restypes, rc.chi_angles_mask, strict=True):
        resname = rc.restype_1to3[letter]
        assert len(mask_row) == 4
        assert int(sum(mask_row)) == len(rc.chi_angles_atoms[resname])


# --------------------------------------------------------------------------------------
# Derived rigid-group arrays (built at import by _make_rigid_group_constants)
# --------------------------------------------------------------------------------------


def test_derived_array_shapes() -> None:
    assert rc.restype_atom14_to_rigid_group.shape == (21, 14)
    assert rc.restype_atom14_mask.shape == (21, 14)
    assert rc.restype_atom14_rigid_group_positions.shape == (21, 14, 3)
    assert rc.restype_rigid_group_default_frame.shape == (21, 8, 4, 4)


def test_backbone_atoms_present_for_every_standard_residue() -> None:
    # N/CA/C/O occupy atom14 slots 0..3 for all 20 standard residues.
    assert np.all(rc.restype_atom14_mask[:20, :4] == 1.0)


def test_unknown_residue_row_is_all_zeros() -> None:
    # UNK (row 20) is never populated by _make_rigid_group_constants.
    assert rc.restype_atom14_mask[20].sum() == 0.0


def test_atom14_mask_matches_named_atom_counts() -> None:
    for restype, letter in enumerate(rc.restypes):
        names = rc.restype_name_to_atom14_names[rc.restype_1to3[letter]]
        assert rc.restype_atom14_mask[restype].sum() == float(sum(1 for name in names if name))


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
