from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from chanfig import Config
from torch import nn

from ofoldx.artifacts import ARTIFACT_MANIFEST_NAME
from ofoldx.data import AtomRecord, ResidueKey, Structure
from ofoldx.models import BaseInput, StructurePredictionOutput
from ofoldx.models.alphafold2 import AlphaFold2Config, AlphaFold2Model
from ofoldx.models.alphafold2.processor import AlphaFold2Processor
from ofoldx.models.alphafold3 import AlphaFold3Config, AlphaFold3Model
from ofoldx.models.boltz1 import Boltz1Config, Boltz1Model
from ofoldx.models.boltz2 import Boltz2Config, Boltz2Model
from ofoldx.models.odesign import ODesignConfig, ODesignModel
from ofoldx.models.protenix import ProtenixConfig, ProtenixModel
from ofoldx.pipelines import Pipeline, StructurePredictionPipeline, StructureProcessor
from ofoldx.pipelines.auto import PIPELINE_MANIFEST_NAME
from ofoldx.processors import AutoProcessor
from ofoldx.processors.utils import StructureInput


@dataclass(slots=True, kw_only=True)
class CustomInput(BaseInput):
    r"""
    Minimal external model input used to verify pipeline extensibility.
    """

    value: torch.Tensor


class StructureModel(nn.Module):
    def forward(self, inputs: CustomInput) -> StructurePredictionOutput:
        return StructurePredictionOutput(
            pred_coords=inputs.value.reshape(1, 1, 1, 3),
            distogram_logits=torch.zeros(1, 1, 1, 2),
        )


class BadStructureModel(nn.Module):
    def forward(self, inputs: CustomInput) -> torch.Tensor:
        return inputs.value


class SaveableStructureModel(nn.Module):
    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


class SaveableProcessor:
    def __call__(self, structure: StructureInput) -> CustomInput:
        return CustomInput(value=torch.zeros(3))

    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


def tiny_structure() -> Structure:
    return Structure(
        atoms=(AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0),),
    )


def protein_backbone_structure() -> Structure:
    atoms: list[AtomRecord] = []
    for residue_number, residue_name in enumerate(("ALA", "GLY", "SER"), start=1):
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
    return Structure(tuple(atoms))


def tiny_alphafold_family_config() -> dict[str, object]:
    return {
        "single_size": 16,
        "pair_size": 8,
        "recycling_steps": 1,
        "trunk": {
            "num_pairformer_blocks": 1,
            "blocks_per_checkpoint": None,
            "operator_backend": "sdpa",
            "pairformer_dropout": 0.0,
            "msa_dropout": 0.0,
            "pair_dropout": 0.0,
        },
        "template": {"num_blocks": 0, "hidden_size": 8},
        "denoiser": {
            "num_token_encoder_blocks": 1,
            "num_token_encoder_attention_heads": 2,
            "blocks_per_checkpoint": None,
            "operator_backend": "sdpa",
            "atom_hidden_size": 8,
            "atom_pair_hidden_size": 4,
            "token_hidden_size": 16,
            "num_atom_encoder_blocks": 1,
            "num_atom_encoder_attention_heads": 2,
            "num_atom_decoder_blocks": 1,
            "num_atom_decoder_attention_heads": 2,
            "num_query_atoms": 4,
            "num_key_atoms": 8,
        },
        "sampler": {"num_sampling_steps": 1, "center_random_augmentation": False},
        "diffusion_training": {"num_samples": 1, "rollout_steps": 1, "rollout_num_samples": 1},
        "confidence": {
            "num_blocks": 1,
            "blocks_per_checkpoint": None,
            "operator_backend": "sdpa",
            "num_pair_attention_heads": 2,
            "triangle_multiplication_hidden_size": 8,
            "triangle_attention_hidden_size": 4,
            "num_triangle_attention_heads": 2,
            "expansion": 2,
            "pairformer_dropout": 0.0,
            "max_atoms_per_token": 4,
        },
    }


def tiny_alphafold3_model() -> AlphaFold3Model:
    return AlphaFold3Model(AlphaFold3Config(tiny_alphafold_family_config()))


def tiny_alphafold2_model() -> AlphaFold2Model:
    return AlphaFold2Model(
        AlphaFold2Config(
            {
                "recycling_steps": 1,
                "evoformer": {
                    "msa_size": 8,
                    "pair_size": 4,
                    "single_size": 8,
                    "extra_msa_size": 4,
                    "num_blocks": 1,
                    "extra_msa_num_blocks": 0,
                    "operator_backend": "sdpa",
                    "blocks_per_checkpoint": None,
                    "iteration": {
                        "msa_row_attention_with_pair_bias": {
                            "num_head": 2,
                            "value_dim": 4,
                            "dropout_rate": 0.0,
                        },
                        "outer_product_mean": {"num_outer_channel": 4},
                        "triangle_multiplication_outgoing": {"num_intermediate_channel": 4},
                        "triangle_multiplication_incoming": {"num_intermediate_channel": 4},
                        "triangle_attention_starting_node": {
                            "num_head": 2,
                            "value_dim": 2,
                            "dropout_rate": 0.0,
                        },
                        "triangle_attention_ending_node": {
                            "num_head": 2,
                            "value_dim": 2,
                            "dropout_rate": 0.0,
                        },
                        "pair_transition": {"num_intermediate_factor": 2},
                    },
                },
                "template": {"enabled": False, "num_blocks": 0},
                "heads": {
                    "structure_module": {"num_channel": 8, "num_layer": 1, "num_layer_in_transition": 1},
                    "predicted_lddt": {"num_channels": 8},
                },
            }
        )
    )


def tiny_odesign_model() -> ODesignModel:
    return ODesignModel(ODesignConfig(tiny_alphafold_family_config()))


def tiny_protenix_model() -> ProtenixModel:
    config = ProtenixConfig(tiny_alphafold_family_config())
    config.constraint_embedder.constraint_pair_size = 8
    config.constraint_embedder.substructure_embedder.hidden_size = 8
    config.constraint_embedder.substructure_embedder.n_layers = 1
    return ProtenixModel(config)


def tiny_boltz1_model() -> Boltz1Model:
    return Boltz1Model(
        Boltz1Config(
            {
                "single_size": 16,
                "pair_size": 8,
                "atom_hidden_size": 8,
                "atom_pair_hidden_size": 4,
                "fourier_embedding_size": 16,
                "recycling_steps": 0,
                "no_msa": True,
                "input_embedder": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 2},
                "trunk": {
                    "num_pairformer_blocks": 1,
                    "num_pair_attention_heads": 2,
                    "num_msa_blocks": 0,
                    "msa_size": 8,
                    "msa_head_dim": 4,
                    "num_msa_attention_heads": 2,
                    "outer_product_hidden_size": 4,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 2,
                    "operator_backend": "sdpa",
                    "dropout": 0.0,
                    "msa_dropout": 0.0,
                },
                "diffusion": {
                    "num_token_transformer_blocks": 1,
                    "num_token_transformer_heads": 2,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_decoder_blocks": 1,
                    "num_atom_attention_heads": 2,
                    "atom_hidden_size": 8,
                    "atom_pair_hidden_size": 4,
                    "num_query_atoms": 4,
                    "num_key_atoms": 8,
                    "num_conditioning_transitions": 1,
                    "operator_backend": "sdpa",
                    "alignment_reverse_diff": False,
                },
                "sampler": {"num_sampling_steps": 1},
                "confidence": {"enabled": False},
            }
        )
    )


def tiny_boltz2_model() -> Boltz2Model:
    return Boltz2Model(
        Boltz2Config(
            {
                "single_size": 16,
                "pair_size": 8,
                "atom_hidden_size": 8,
                "atom_pair_hidden_size": 4,
                "fourier_embedding_size": 16,
                "recycling_steps": 0,
                "input_embedder": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 2},
                "template": {
                    "hidden_size": 8,
                    "num_blocks": 0,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 2,
                },
                "trunk": {
                    "num_pairformer_blocks": 1,
                    "num_pair_attention_heads": 2,
                    "num_msa_blocks": 0,
                    "msa_size": 8,
                    "msa_head_dim": 4,
                    "num_msa_attention_heads": 2,
                    "outer_product_hidden_size": 4,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 2,
                    "operator_backend": "sdpa",
                    "dropout": 0.0,
                    "msa_dropout": 0.0,
                },
                "diffusion": {
                    "num_token_transformer_blocks": 1,
                    "num_token_transformer_heads": 2,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_decoder_blocks": 1,
                    "num_atom_attention_heads": 2,
                    "atom_hidden_size": 8,
                    "atom_pair_hidden_size": 4,
                    "num_query_atoms": 4,
                    "num_key_atoms": 8,
                    "num_conditioning_transitions": 1,
                    "operator_backend": "sdpa",
                    "alignment_reverse_diff": False,
                },
                "sampler": {"num_sampling_steps": 1},
                "confidence": {"num_pairformer_blocks": 1, "num_pair_attention_heads": 2},
            }
        )
    )


def test_structure_prediction_pipeline_accepts_custom_base_input() -> None:
    pipeline = StructurePredictionPipeline(StructureModel())
    batch = CustomInput(value=torch.ones(3))

    output = pipeline(batch)

    torch.testing.assert_close(output.pred_coords, torch.ones(1, 1, 1, 3))


def test_structure_prediction_pipeline_accepts_processed_structure_input() -> None:
    def processor(structure: StructureInput) -> CustomInput:
        assert isinstance(structure, Structure)
        return CustomInput(value=torch.ones(3))

    pipeline = StructurePredictionPipeline(StructureModel(), processor=processor)

    output = pipeline(tiny_structure())

    torch.testing.assert_close(output.pred_coords, torch.ones(1, 1, 1, 3))


def test_structure_prediction_pipeline_rejects_invalid_inputs_and_outputs() -> None:
    pipeline = StructurePredictionPipeline(StructureModel())

    with pytest.raises(ValueError, match="processor is required"):
        pipeline(tiny_structure())
    with pytest.raises(TypeError, match="BaseInput or structure input"):
        pipeline(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return StructurePredictionOutput"):
        StructurePredictionPipeline(BadStructureModel())(CustomInput(value=torch.ones(3)))


def test_structure_prediction_pipeline_save_writes_manifest(tmp_path) -> None:
    StructurePredictionPipeline(SaveableStructureModel(), processor=SaveableProcessor()).save_pretrained(tmp_path)

    manifest = Config.from_json(tmp_path / PIPELINE_MANIFEST_NAME)
    assert manifest["kind"] == "pipeline"
    assert manifest["task"] == "structure_prediction"
    assert manifest["entrypoint"] == "ofoldx.pipelines.structure_prediction.StructurePredictionPipeline"
    assert manifest["components"]["predictor"]["path"] == "predictor"
    assert manifest["components"]["processor"]["path"] == "processor"


@pytest.mark.parametrize(
    ("architecture", "model_factory"),
    [
        ("alphafold2", tiny_alphafold2_model),
        ("alphafold3", tiny_alphafold3_model),
        ("boltz1", tiny_boltz1_model),
        ("boltz2", tiny_boltz2_model),
        ("odesign", tiny_odesign_model),
        ("protenix", tiny_protenix_model),
    ],
)
def test_structure_prediction_pipeline_e2e_runs_saved_registered_packages(
    tmp_path,
    architecture: str,
    model_factory: Callable[[], nn.Module],
) -> None:
    model = model_factory()
    pipeline = StructurePredictionPipeline(
        model,
        processor=cast(StructureProcessor[BaseInput], AutoProcessor.from_config(model.config)),
    )

    pipeline.save_pretrained(tmp_path / architecture)
    loaded = Pipeline.from_pretrained(tmp_path / architecture)
    output = loaded(protein_backbone_structure(), recycling_steps=1, num_samples=1, num_sampling_steps=1)

    assert isinstance(loaded, StructurePredictionPipeline)
    assert output.pred_coords.ndim == 4
    assert output.pred_coords.shape[:2] == (1, 1)
    assert output.pred_coords.shape[-1] == 3
    assert output.distogram_logits.shape[0] == 1


def test_pipeline_from_plain_model_artifact_uses_processor_component(tmp_path: Path) -> None:
    model = tiny_alphafold2_model()
    root = tmp_path / "alphafold2"
    model.save_pretrained(root)
    AlphaFold2Processor(protein_only=False).save_pretrained(root / "processor")
    manifest = Config.from_json(root / ARTIFACT_MANIFEST_NAME)
    components = dict(manifest.get("components", {}))
    components["processor"] = {"path": "processor"}
    manifest["components"] = components
    manifest.json(root / ARTIFACT_MANIFEST_NAME)

    loaded = Pipeline.from_pretrained(root)

    assert isinstance(loaded, StructurePredictionPipeline)
    assert isinstance(loaded.processor, AlphaFold2Processor)
    assert loaded.processor.protein_only is False
