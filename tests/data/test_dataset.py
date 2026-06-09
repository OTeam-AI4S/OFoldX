from __future__ import annotations

import pytest

from ofoldx.data import AtomRecord, ResidueKey, Structure, StructureDataset
from ofoldx.data.dataset import crop_structure


@pytest.fixture
def interleaved() -> Structure:
    # res_a's atoms (0, 2) interleave res_b's atom 1, forcing a non-identity remap; intra-res_a bond 0-2 survives,
    # cross-residue bond 0-1 is filtered when res_b drops.
    res_a, res_b = ResidueKey("A", 1), ResidueKey("A", 2)
    return Structure(
        atoms=(
            AtomRecord("N", "N", "ALA", res_a, 0.0, 0.0, 0.0),
            AtomRecord("N", "N", "ALA", res_b, 1.0, 0.0, 0.0),
            AtomRecord("CA", "C", "ALA", res_a, 2.0, 0.0, 0.0),
        ),
        bonds=((0, 2), (0, 1)),
    )


def test_crop(interleaved: Structure) -> None:
    cropped = crop_structure(interleaved, max_tokens=1)

    assert len(cropped.atoms) == 2  # res_a's two atoms (old indices 0 and 2)
    assert cropped.bonds == ((0, 1),)  # old (0, 2) remapped onto the new contiguous indices; (0, 1) dropped
    # at or below the token count, crop is a no-op and returns the input unchanged
    assert crop_structure(interleaved, max_tokens=8) is interleaved


def test_structure_dataset_reads_crops_and_batches(tmp_path) -> None:
    (tmp_path / "tiny.pdb").write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  GLY A   2       2.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n"
    )
    seen = []

    def processor(structure: Structure) -> tuple[int, float]:
        seen.append(structure)
        return len(structure.atoms), structure.atoms[0].x

    dataset = StructureDataset(tmp_path, processor=processor, max_tokens=1)

    assert len(dataset) == 1
    assert dataset[0] == (1, 1.0)
    assert dataset.__getitems__([0]) == [(1, 1.0)]
    assert len(seen) == 2


def test_structure_dataset_validates_roots_and_formats(tmp_path) -> None:
    def processor(structure: Structure) -> Structure:
        return structure

    with pytest.raises(NotADirectoryError, match="not a directory"):
        StructureDataset(tmp_path / "missing", processor=processor)
    with pytest.raises(ValueError, match="unsupported structure format"):
        StructureDataset(tmp_path, format="xyz", processor=processor)
    (tmp_path / "notes.txt").write_text("not a structure")
    with pytest.raises(ValueError, match="no structure files"):
        StructureDataset(tmp_path, processor=processor)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_invalid_inputs(interleaved: Structure, max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        crop_structure(interleaved, max_tokens=max_tokens)
