from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from chanfig import Config
from torch import nn

from ofoldx.data import AtomRecord, ResidueKey, Structure
from ofoldx.models import AutoModel, BaseInput, DesignGeneratorOutput, StructurePredictionOutput
from ofoldx.models.boltz2 import Boltz2AffinityConfig, Boltz2AffinityModel
from ofoldx.models.boltzgen import (
    BoltzGenAdherenceConfig,
    BoltzGenConfig,
    BoltzGenDesignGenerator,
    BoltzGenDiverseConfig,
    BoltzGenIFoldConfig,
    BoltzGenIFoldGenerator,
    BoltzGenStructureTrainedSmallConfig,
)
from ofoldx.models.ligandmpnn import LigandMPNNConfig, LigandMPNNDesignGenerator, LigandMPNNProcessor
from ofoldx.models.odesign import ODesignDesignConfig, ODesignDesignGenerator
from ofoldx.models.proteinmpnn import ProteinMPNNConfig, ProteinMPNNDesignGenerator, ProteinMPNNProcessor
from ofoldx.models.pxdesign import PxDesignConfig, PxDesignDesignGenerator
from ofoldx.models.rfdiffusion import RfDiffusionConfig, RfDiffusionDesignGenerator
from ofoldx.pipelines import DesignPipeline, Pipeline, StructureProcessor
from ofoldx.pipelines.auto import PIPELINE_MANIFEST_NAME
from ofoldx.processors import AutoProcessor
from ofoldx.processors.utils import StructureInput


@dataclass(slots=True, kw_only=True)
class CustomInput(BaseInput):
    r"""
    Minimal external model input used to verify design-pipeline extensibility.
    """

    value: torch.Tensor


class DesignModel(nn.Module):
    def generate(self, inputs: CustomInput) -> DesignGeneratorOutput:
        return DesignGeneratorOutput(sequence_logits=inputs.value)


class ForwardDesignModel(nn.Module):
    def forward(self, inputs: CustomInput) -> StructurePredictionOutput:
        return StructurePredictionOutput(
            pred_coords=inputs.value.reshape(1, 1, 1, 3),
            distogram_logits=torch.zeros(1, 1, 1, 2),
        )


class BadDesignModel(nn.Module):
    def forward(self, inputs: CustomInput) -> torch.Tensor:
        return inputs.value


class AssignmentModel(nn.Module):
    def forward(self, inputs: BaseInput, **_: object) -> DesignGeneratorOutput:
        length = inputs.token_mask.shape[-1]  # type: ignore[attr-defined]
        return DesignGeneratorOutput(sequence_tokens=torch.ones(1, length, dtype=torch.long))


class SaveableDesignModel(nn.Module):
    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


class SaveableProcessor:
    def __call__(self, structure: StructureInput) -> CustomInput:
        return CustomInput(value=torch.zeros(1, 1, 21))

    def save_pretrained(self, save_directory: Path, **kwargs: object) -> None:
        save_directory.mkdir(parents=True)


def tiny_structure() -> Structure:
    return Structure(
        atoms=(AtomRecord("CA", "C", "ALA", ResidueKey("A", 1), 0.0, 0.0, 0.0),),
    )


def protein_backbone_structure() -> Structure:
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
    return Structure(tuple(atoms))


def protein_ligand_structure() -> Structure:
    atoms = list(protein_backbone_structure().atoms)
    atoms.append(
        AtomRecord(
            "C1",
            "C",
            "LIG",
            ResidueKey("L", 1),
            8.0,
            4.0,
            0.0,
            hetero=True,
            mol_type="ligand",
        )
    )
    return Structure(tuple(atoms))


def tiny_proteinmpnn_model() -> ProteinMPNNDesignGenerator:
    return ProteinMPNNDesignGenerator(
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


def tiny_ligandmpnn_model(_workdir: Path) -> LigandMPNNDesignGenerator:
    return LigandMPNNDesignGenerator(
        LigandMPNNConfig(
            {
                "hidden_size": 16,
                "node_size": 16,
                "edge_size": 16,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "k_neighbors": 2,
                "atom_context_num": 1,
            }
        )
    )


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


def tiny_odesign_design_model(_workdir: Path) -> ODesignDesignGenerator:
    return ODesignDesignGenerator(ODesignDesignConfig(tiny_alphafold_family_config()))


def tiny_boltz2_config() -> dict[str, object]:
    return {
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


def tiny_boltz2_affinity_model(_workdir: Path) -> Boltz2AffinityModel:
    config_dict = tiny_boltz2_config()
    config_dict.update(
        {
            "affinity_ensemble": False,
            "affinity": {
                "num_pairformer_blocks": 1,
                "num_pair_attention_heads": 2,
                "triangle_attention_hidden_size": 4,
                "num_triangle_attention_heads": 2,
                "num_conditioning_transitions": 1,
            },
        }
    )
    return Boltz2AffinityModel(Boltz2AffinityConfig(config_dict))


def tiny_boltzgen_config() -> dict[str, object]:
    return {
        "single_size": 16,
        "pair_size": 8,
        "atom_hidden_size": 8,
        "atom_pair_hidden_size": 4,
        "fourier_embedding_size": 16,
        "recycling_steps": 0,
        "use_templates": False,
        "use_templates_v2": False,
        "bond_type_feature": False,
        "predict_bfactor": False,
        "input_embedder": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 2},
        "template": {
            "hidden_size": 8,
            "num_blocks": 0,
            "triangle_attention_hidden_size": 4,
            "num_triangle_attention_heads": 2,
            "miniformer_blocks": False,
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
            "use_miniformer": False,
            "msa_miniformer_blocks": False,
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
            "conditioning_transition_layers": 1,
            "operator_backend": "sdpa",
            "alignment_reverse_diff": False,
        },
        "sampler": {"num_sampling_steps": 1},
        "design": {
            "template_size": 8,
            "template_blocks": 0,
            "token_distance_size": 8,
            "token_distance_blocks": 0,
            "distance_gaussian_dim": 4,
            "use_token_distance": False,
        },
    }


def force_tiny_boltzgen_variant(config: Any) -> Any:
    config.trunk.num_pairformer_blocks = 1
    config.trunk.num_msa_blocks = 0
    config.trunk.use_miniformer = False
    config.trunk.msa_miniformer_blocks = False
    config.template.num_blocks = 0
    config.template.miniformer_blocks = False
    config.design.template_blocks = 0
    config.design.token_distance_blocks = 0
    config.design.use_token_distance = False
    config.diffusion.num_token_transformer_blocks = 1
    config.diffusion.num_atom_encoder_blocks = 1
    config.diffusion.num_atom_decoder_blocks = 1
    return config


def tiny_boltzgen_model(_workdir: Path) -> BoltzGenDesignGenerator:
    return BoltzGenDesignGenerator(BoltzGenConfig(tiny_boltzgen_config()))


def tiny_boltzgen_diverse_model(_workdir: Path) -> BoltzGenDesignGenerator:
    return BoltzGenDesignGenerator(force_tiny_boltzgen_variant(BoltzGenDiverseConfig(tiny_boltzgen_config())))


def tiny_boltzgen_adherence_model(_workdir: Path) -> BoltzGenDesignGenerator:
    return BoltzGenDesignGenerator(force_tiny_boltzgen_variant(BoltzGenAdherenceConfig(tiny_boltzgen_config())))


def tiny_boltzgen_structuretrained_small_model(_workdir: Path) -> BoltzGenDesignGenerator:
    return BoltzGenDesignGenerator(
        force_tiny_boltzgen_variant(BoltzGenStructureTrainedSmallConfig(tiny_boltzgen_config()))
    )


def tiny_boltzgen_ifold_model(_workdir: Path) -> BoltzGenIFoldGenerator:
    return BoltzGenIFoldGenerator(
        BoltzGenIFoldConfig(
            {
                "single_size": 16,
                "pair_size": 8,
                "atom_hidden_size": 8,
                "atom_pair_hidden_size": 4,
                "fourier_embedding_size": 16,
                "input_embedder": {"num_atom_encoder_blocks": 1, "num_atom_encoder_attention_heads": 2},
                "diffusion": {"num_query_atoms": 4, "num_key_atoms": 8},
                "ifold": {
                    "node_size": 16,
                    "edge_size": 16,
                    "hidden_size": 16,
                    "num_encoder_layers": 1,
                    "num_decoder_layers": 1,
                    "num_attention_heads": 2,
                    "k_neighbors": 2,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_encoder_attention_heads": 2,
                    "operator_backend": "sdpa",
                },
            }
        )
    )


def tiny_pxdesign_model(_workdir: Path) -> PxDesignDesignGenerator:
    return PxDesignDesignGenerator(
        PxDesignConfig(
            {
                "single_size": 16,
                "pair_size": 8,
                "token_feature_size": 16,
                "atom_feature_size": 385,
                "condition_embedder": {
                    "atom_hidden_size": 8,
                    "atom_pair_hidden_size": 4,
                    "token_hidden_size": 8,
                    "token_feature_size": 16,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_encoder_attention_heads": 2,
                    "num_query_atoms": 4,
                    "num_key_atoms": 8,
                    "operator_backend": "sdpa",
                },
                "conditional_template": {"pair_size": 8},
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
                "sampler": {
                    "num_sampling_steps": 1,
                    "num_samples": 1,
                    "sample_chunk_size": None,
                    "center_random_augmentation": False,
                },
            }
        )
    )


def tiny_rfdiffusion_model(workdir: Path) -> RfDiffusionDesignGenerator:
    se3 = {
        "num_layers": 1,
        "num_channels": 4,
        "num_degrees": 2,
        "num_attention_heads": 1,
        "div": 1,
        "l0_in_features": 4,
        "l0_out_features": 4,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 4,
    }
    return RfDiffusionDesignGenerator(
        RfDiffusionConfig(
            {
                "model": {
                    "num_extra_blocks": 0,
                    "num_main_blocks": 1,
                    "num_refinement_blocks": 0,
                    "msa_size": 8,
                    "full_msa_size": 4,
                    "pair_size": 8,
                    "template_size": 4,
                    "num_msa_attention_heads": 1,
                    "num_pair_attention_heads": 1,
                    "num_template_attention_heads": 1,
                    "hidden_size": 4,
                    "template_hidden_size": 4,
                    "dropout": 0.0,
                    "se3_param_full": se3,
                    "se3_param_topk": se3,
                },
                "diffuser": {
                    "num_timesteps": 15,
                    "truncation_level": 20,
                    "igso3_num_omega": 8,
                    "igso3_num_sigma": 4,
                },
                "preprocess": {"template_1d_size": 22, "template_2d_size": 44},
                "inference": {"final_step": 1, "schedule_directory_path": str(workdir / "schedules")},
            }
        )
    )


@dataclass(frozen=True, slots=True)
class RegisteredDesignCase:
    architecture: str
    model_factory: Callable[[Path], nn.Module]
    processor_factory: Callable[[Any], object]
    structure_factory: Callable[[], Structure]
    model_kwargs: dict[str, object]
    expects_sequence: bool = False
    expects_structure: bool = False
    expects_scores: bool = False


def tiny_proteinmpnn_model_for_case(_workdir: Path) -> ProteinMPNNDesignGenerator:
    return tiny_proteinmpnn_model()


def auto_processor(config: Any) -> object:
    return AutoProcessor.from_config(config)


def pxdesign_processor(config: Any) -> object:
    return AutoProcessor.from_config(config, design_residues=(("A", 2),))


def rfdiffusion_processor(config: Any) -> object:
    return AutoProcessor.from_config(config, preserve_input_motif=False)


REGISTERED_DESIGN_CASES = [
    RegisteredDesignCase(
        "proteinmpnn",
        tiny_proteinmpnn_model_for_case,
        auto_processor,
        protein_backbone_structure,
        {"num_samples": 1},
        expects_sequence=True,
    ),
    RegisteredDesignCase(
        "ligandmpnn",
        tiny_ligandmpnn_model,
        auto_processor,
        protein_backbone_structure,
        {"num_samples": 1},
        expects_sequence=True,
    ),
    RegisteredDesignCase(
        "boltzgen",
        tiny_boltzgen_model,
        auto_processor,
        protein_backbone_structure,
        {"recycling_steps": 1, "num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "boltzgen_diverse",
        tiny_boltzgen_diverse_model,
        auto_processor,
        protein_backbone_structure,
        {"recycling_steps": 1, "num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "boltzgen_adherence",
        tiny_boltzgen_adherence_model,
        auto_processor,
        protein_backbone_structure,
        {"recycling_steps": 1, "num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "boltzgen_structuretrained_small",
        tiny_boltzgen_structuretrained_small_model,
        auto_processor,
        protein_backbone_structure,
        {"recycling_steps": 1, "num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "boltzgen_ifold",
        tiny_boltzgen_ifold_model,
        auto_processor,
        protein_backbone_structure,
        {},
        expects_sequence=True,
    ),
    RegisteredDesignCase(
        "odesign_design",
        tiny_odesign_design_model,
        auto_processor,
        protein_backbone_structure,
        {"recycling_steps": 1, "num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "pxdesign",
        tiny_pxdesign_model,
        pxdesign_processor,
        protein_backbone_structure,
        {"num_samples": 1, "num_sampling_steps": 1},
        expects_structure=True,
    ),
    RegisteredDesignCase(
        "rfdiffusion",
        tiny_rfdiffusion_model,
        rfdiffusion_processor,
        protein_backbone_structure,
        {"num_samples": 1, "num_sampling_steps": 1},
        expects_sequence=True,
        expects_structure=True,
    ),
]


def test_design_pipeline_accepts_custom_base_input() -> None:
    pipeline = DesignPipeline(DesignModel())
    batch = CustomInput(value=torch.ones(2, 3))

    output = pipeline(batch)

    assert len(output.candidates) == 1
    torch.testing.assert_close(output.candidates[0].sequence_logits, torch.ones(2, 3))


def test_design_pipeline_accepts_processed_structure_input() -> None:
    def processor(structure: StructureInput) -> CustomInput:
        assert isinstance(structure, Structure)
        return CustomInput(value=torch.ones(2, 3))

    pipeline = DesignPipeline(DesignModel(), processor=processor)

    output = pipeline(tiny_structure())

    torch.testing.assert_close(output.candidates[0].sequence_logits, torch.ones(2, 3))


def test_design_pipeline_normalizes_structure_prediction_outputs() -> None:
    pipeline = DesignPipeline(ForwardDesignModel())

    output = pipeline(CustomInput(value=torch.ones(3)))

    assert len(output.candidates) == 1
    structure_prediction = output.candidates[0].structure_prediction
    assert structure_prediction is not None
    torch.testing.assert_close(structure_prediction.pred_coords, torch.ones(1, 1, 1, 3))


def test_design_pipeline_rejects_invalid_inputs_and_outputs() -> None:
    pipeline = DesignPipeline(DesignModel())

    with pytest.raises(ValueError, match="processor is required"):
        pipeline(tiny_structure())
    with pytest.raises(TypeError, match="BaseInput or structure input"):
        pipeline(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="design generators must return"):
        DesignPipeline(BadDesignModel())(CustomInput(value=torch.ones(3)))


def test_design_pipeline_save_requires_processor(tmp_path) -> None:
    with pytest.raises(ValueError, match="processor is required"):
        DesignPipeline(SaveableDesignModel()).save_pretrained(tmp_path)


def test_design_pipeline_save_writes_manifest(tmp_path) -> None:
    DesignPipeline(SaveableDesignModel(), processor=SaveableProcessor()).save_pretrained(tmp_path)

    manifest = Config.from_json(tmp_path / PIPELINE_MANIFEST_NAME)
    assert manifest["kind"] == "pipeline"
    assert manifest["task"] == "design_generation"
    assert manifest["entrypoint"] == "ofoldx.pipelines.design.DesignPipeline"
    assert manifest["components"]["generator"]["path"] == "generator"
    assert manifest["components"]["processor"]["path"] == "processor"


def test_design_pipeline_e2e_runs_saved_proteinmpnn_package(tmp_path) -> None:
    pipeline = DesignPipeline(tiny_proteinmpnn_model(), processor=ProteinMPNNProcessor())

    pipeline.save_pretrained(tmp_path)
    loaded = Pipeline.from_pretrained(tmp_path)
    loaded_model = AutoModel.from_pretrained(tmp_path)
    loaded_processor = AutoProcessor.from_pretrained(tmp_path)
    output = loaded(protein_backbone_structure(), num_samples=1)

    assert isinstance(loaded, DesignPipeline)
    assert isinstance(loaded.model, ProteinMPNNDesignGenerator)
    assert isinstance(loaded.processor, ProteinMPNNProcessor)
    assert isinstance(loaded_model, ProteinMPNNDesignGenerator)
    assert isinstance(loaded_processor, ProteinMPNNProcessor)
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert candidate.sequence_logits is not None
    assert candidate.sequence_logits.shape == (1, 4, 21)
    assert candidate.sequence_tokens is not None
    assert candidate.sequence_tokens.shape == (1, 4)
    assert candidate.scores is not None
    assert candidate.scores["log_probs"].shape == (1, 4, 21)
    assert candidate.scores["sampling_probs"].shape == (1, 4, 20)


@pytest.mark.parametrize("case", REGISTERED_DESIGN_CASES, ids=lambda case: case.architecture)
def test_design_pipeline_e2e_runs_saved_registered_packages(tmp_path, case: RegisteredDesignCase) -> None:
    model = case.model_factory(tmp_path / case.architecture)
    pipeline = DesignPipeline(
        model,
        processor=cast(StructureProcessor[BaseInput], case.processor_factory(model.config)),
    )

    pipeline.save_pretrained(tmp_path / case.architecture)
    loaded = Pipeline.from_pretrained(tmp_path / case.architecture)
    output = loaded(case.structure_factory(), **case.model_kwargs)

    assert isinstance(loaded, DesignPipeline)
    assert len(output.candidates) == 1
    if case.architecture == "ligandmpnn":
        assert isinstance(loaded.processor, LigandMPNNProcessor)

    candidate = output.candidates[0]
    if case.expects_sequence:
        assert candidate.sequence_tokens is not None or candidate.sequence_logits is not None
    if candidate.sequence_tokens is not None:
        assert candidate.sequence_tokens.ndim >= 2
    if candidate.sequence_logits is not None:
        assert candidate.sequence_logits.shape[-1] > 1
    if case.expects_structure:
        assert candidate.structure_prediction is not None
        assert candidate.structure_prediction.pred_coords.ndim >= 3
        assert candidate.structure_prediction.pred_coords.shape[-1] == 3
    if case.expects_scores:
        assert candidate.scores is not None


def test_pxdesign_pipeline_runs_sequence_assignment_and_filtering_stages(tmp_path: Path) -> None:
    model = tiny_pxdesign_model(tmp_path)
    pipeline = DesignPipeline(
        model,
        processor=cast(StructureProcessor[BaseInput], pxdesign_processor(model.config)),
    )

    output = pipeline(
        protein_backbone_structure(),
        num_samples=1,
        num_sampling_steps=1,
        sequence_assignment={"model": AssignmentModel(), "num_samples": 1},
        filtering={"prediction": {"quality": torch.tensor([0.7])}},
    )

    candidate = output.candidates[0]
    assert candidate.sequence_tokens is not None
    assert candidate.scores is not None
    assert "filter_quality" in candidate.scores
    assert candidate.metadata is not None
    assert candidate.metadata["sequence_assignment"]["backend"] == "native_proteinmpnn"
    assert candidate.metadata["filtering"]["backend"] == "native_structure_predictor_hook"
