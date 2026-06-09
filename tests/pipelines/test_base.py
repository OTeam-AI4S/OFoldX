from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ofoldx.data import AtomRecord, ResidueKey, Structure
from ofoldx.models import BaseInput
from ofoldx.models.proteinmpnn import ProteinMPNNConfig, ProteinMPNNDesignGenerator, ProteinMPNNProcessor
from ofoldx.pipelines.base import Pipeline, is_structure_input
from ofoldx.pipelines.design import DesignPipeline


@dataclass(slots=True, kw_only=True)
class CustomInput(BaseInput):
    r"""
    Minimal external input used to verify base pipeline behavior.
    """

    value: torch.Tensor


def test_is_structure_input_separates_raw_structures_from_model_inputs() -> None:
    structure = Structure(
        atoms=(AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0),),
    )

    assert is_structure_input(structure)
    assert is_structure_input("tiny.pdb")
    assert is_structure_input(Path("tiny.cif"))
    assert not is_structure_input(CustomInput(value=torch.ones(1)))
    assert not is_structure_input(torch.ones(1))


def test_pipeline_sets_eval_mode_and_reports_parameter_device() -> None:
    model = nn.Linear(2, 2)
    model.train()

    pipeline: Pipeline[CustomInput, object] = Pipeline(model)
    empty_pipeline: Pipeline[CustomInput, object] = Pipeline(nn.Module())

    assert not model.training
    assert pipeline.device == torch.device("cpu")
    assert empty_pipeline.device == torch.device("cpu")


def test_pipeline_moves_base_input_to_model_device() -> None:
    pipeline: Pipeline[CustomInput, object] = Pipeline(nn.Linear(1, 1))
    batch = CustomInput(value=torch.ones(2))

    moved = pipeline._move_to_model(batch)

    assert isinstance(moved, CustomInput)
    assert moved is not batch
    assert moved.value.device == pipeline.device
    torch.testing.assert_close(moved.value, batch.value)


def test_pipeline_from_pretrained_runs_bare_proteinmpnn_model_with_default_processor(tmp_path) -> None:
    model = ProteinMPNNDesignGenerator(
        ProteinMPNNConfig(
            {
                "hidden_size": 16,
                "edge_feature_size": 16,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "k_neighbors": 2,
            }
        )
    )
    atoms: list[AtomRecord] = []
    for residue_number, residue_name in enumerate(("ALA", "GLY", "SER", "THR"), start=1):
        residue = ResidueKey("A", residue_number)
        x = float(residue_number) * 3.8
        atoms.extend(
            (
                AtomRecord("N", "N", residue_name, residue, x - 1.2, 0.0, 0.0),
                AtomRecord("CA", "C", residue_name, residue, x, 0.0, 0.0),
                AtomRecord("C", "C", residue_name, residue, x + 1.2, 0.0, 0.0),
                AtomRecord("O", "O", residue_name, residue, x + 1.8, 0.6, 0.0),
            )
        )

    model.save_pretrained(tmp_path)
    pipeline = Pipeline.from_pretrained(tmp_path)
    output = pipeline(Structure(tuple(atoms)), num_samples=1)

    assert isinstance(pipeline, DesignPipeline)
    assert isinstance(pipeline.processor, ProteinMPNNProcessor)
    assert len(output.candidates) == 1
    assert output.candidates[0].sequence_logits is not None
    assert output.candidates[0].sequence_logits.shape == (1, 4, 21)
