from __future__ import annotations

import pytest
import torch

from ofoldx.data.structure import AtomRecord, ResidueKey, Structure
from ofoldx.models.contract import DesignOutput, SequenceDesignCandidate, StructurePredictionOutput
from ofoldx.modules.representation import Coords
from ofoldx.results import DesignResult, StructurePredictionResult


def _structure() -> Structure:
    return Structure(
        (
            AtomRecord("N", "N", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0),
            AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 1.0, 1.0, 1.0),
        )
    )


def _prediction() -> StructurePredictionOutput:
    return StructurePredictionOutput(
        pred_coords=torch.tensor([[[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]]),
        distogram_logits=torch.zeros(1, 2, 2, 4),
        plddt_logits=torch.zeros(1, 2, 50),
    )


def test_fold_result_accessors():
    result = StructurePredictionResult(_prediction(), source=_structure())
    assert isinstance(result.coords, Coords)
    assert result.coords.num_points == 2
    assert result.plddt.shape == (1, 2)
    assert isinstance(result.raw, StructurePredictionOutput)


def test_fold_result_save_pdb_puts_predicted_coords_on_input_atoms(tmp_path):
    from ofoldx.io import read_pdb

    result = StructurePredictionResult(_prediction(), source=_structure())
    path = tmp_path / "out.pdb"
    result.save_pdb(path)
    reloaded = read_pdb(path)
    assert len(reloaded.atoms) == 2
    assert (reloaded.atoms[0].x, reloaded.atoms[0].y, reloaded.atoms[0].z) == pytest.approx((5.0, 5.0, 5.0))
    assert reloaded.atoms[0].name == "N"


def test_fold_result_sequence_input_has_no_source(tmp_path):
    result = StructurePredictionResult(_prediction(), source=None)
    with pytest.raises(ValueError, match="needs a structure/path input"):
        result.save_pdb(tmp_path / "out.pdb")


def test_fold_result_fewer_predicted_atoms_is_flagged(tmp_path):
    output = StructurePredictionOutput(pred_coords=torch.zeros(1, 1, 3), distogram_logits=torch.zeros(1, 1, 1, 4))
    result = StructurePredictionResult(output, source=_structure())  # source has 2 atoms, prediction has only 1
    with pytest.raises(ValueError, match="fewer than"):
        result.save_pdb(tmp_path / "out.pdb")


def test_fold_result_drops_trailing_atom_padding(tmp_path):
    from ofoldx.io import read_pdb

    # source has 2 real atoms; the model padded its atom axis to 4 -- the trailing padding rows are dropped.
    output = StructurePredictionOutput(
        pred_coords=torch.tensor([[[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        distogram_logits=torch.zeros(1, 4, 4, 4),
    )
    path = tmp_path / "out.pdb"
    StructurePredictionResult(output, source=_structure()).save_pdb(path)
    reloaded = read_pdb(path)
    assert len(reloaded.atoms) == 2
    assert (reloaded.atoms[1].x, reloaded.atoms[1].y, reloaded.atoms[1].z) == pytest.approx((6.0, 6.0, 6.0))


def test_design_result_wraps_candidates():
    output = DesignOutput([SequenceDesignCandidate(sequence_tokens=torch.zeros(4, dtype=torch.long))])
    result = DesignResult(output, source=_structure())
    assert len(result) == 1
    assert result.candidates[0].sequence_tokens is not None


def test_design_result_sequence_decode_via_registry():
    import ofoldx.models.proteinmpnn  # noqa: F401  -- triggers the family's decoder self-registration

    output = DesignOutput(
        [SequenceDesignCandidate(sequence_tokens=torch.tensor([0, 1, 2, 20]), metadata={"architecture": "proteinmpnn"})]
    )
    result = DesignResult(output)
    assert result.sequence == "ACDX"
    assert result.sequences == ["ACDX"]


def test_design_result_sequence_none_without_registered_decoder():
    output = DesignOutput([SequenceDesignCandidate(sequence_tokens=torch.tensor([0, 1]), metadata=None)])
    assert DesignResult(output).sequence is None
    assert DesignResult(output).sequences == []
