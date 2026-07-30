r"""
Strict translation of the official PxDesign v0.1 checkpoint.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor, nn

from .config import PxDesignConfig
from .model import PxDesignModel

_ReferenceKey: TypeAlias = str | tuple[str, ...]
_ReferenceValue: TypeAlias = Tensor | tuple[Tensor, ...]

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def translate_pxdesign_state(
    model: nn.Module,
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Translate every official tensor to one exact native model tensor.
    """

    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    reference_state = _extract_reference_state(checkpoint_or_state_dict)
    converted = _translate_reference_state(model, reference_state, _pxdesign_reference_key)

    expected_reference_keys: set[str] = set()
    for model_key in model.state_dict():
        reference_key = _pxdesign_reference_key(model_key)
        expected_reference_keys.update(reference_key if isinstance(reference_key, tuple) else (reference_key,))
    unexpected = sorted(set(reference_state) - expected_reference_keys)
    if unexpected:
        raise RuntimeError(
            "checkpoint translation failed\n"
            f"unexpected: {len(unexpected)}\n" + "\n".join(f"  unexpected {key}" for key in unexpected[:20])
        )

    if dtype is None:
        return converted
    return {key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in converted.items()}


def convert_pxdesign_model(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> PxDesignModel:
    r"""
    Build a PxDesign model and strict-load the complete official checkpoint.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)
    config = PxDesignConfig()
    with torch.device("meta"):
        model = PxDesignModel(config)
    state_dict = translate_pxdesign_state(model, checkpoint, dtype=dtype)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    return model


def convert_pxdesign_backbone_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert a PXDesign-d checkpoint and write its independent backbone repository.

    Strict-loads the checkpoint via
    [`convert_pxdesign_model`][ofoldx.models.pxdesign.convert.convert_pxdesign_model], pairs
    it with [`PxDesignProcessor`][], and saves the registered
    [`PxDesignBackbonePipeline`][]. The complete released operation is converted
    separately with [`convert_pxdesign_checkpoint`][] because it owns six fixed
    model--processor pairs.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from .pipeline import PxDesignBackbonePipeline
    from .processor import PxDesignProcessor

    model = convert_pxdesign_model(checkpoint_path, map_location=map_location, dtype=dtype)
    processor_dtype = model.condition_embeddings.input_projection.weight.dtype
    PxDesignBackbonePipeline(model=model, processor=PxDesignProcessor(dtype=processor_dtype)).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_pxdesign_checkpoint(
    checkpoint_path: str | PathLike[str],
    proteinmpnn_checkpoint_path: str | PathLike[str],
    alphafold2_checkpoint_path: str | PathLike[str],
    protenix_checkpoint_path: str | PathLike[str],
    protenix_structure_checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert and write the released fixed six-pair PXDesign repository.

    Args:
        checkpoint_path: Released PXDesign-d checkpoint.
        proteinmpnn_checkpoint_path: Released ProteinMPNN ``v_48_020`` checkpoint.
        alphafold2_checkpoint_path: Official AlphaFold parameters containing
            ``model_1_ptm``. The fixed complex and binder pairs use the same
            released weights with their distinct Model graphs.
        protenix_checkpoint_path: ``protenix_base_default_v0.5.0`` checkpoint.
        protenix_structure_checkpoint_path:
            ``protenix_mini_tmpl_v0.5.0`` checkpoint.
        output_dir: Destination composite repository.
        map_location: Device mapping used to load the PXDesign-d checkpoint.
        dtype: Optional common floating-point dtype.
        repo_id: Optional destination Hub repository.
        token: Optional Hub token for destination upload.
        private: Whether a newly created destination repository is private.
    """

    from ..alphafold2.convert import (
        _own_parameter_storage,
        convert_alphafold2_initial_guess_model,
        convert_alphafold2_model,
    )
    from ..alphafold2.pipeline import (
        AlphaFold2InitialGuessPipeline,
        AlphaFold2Pipeline,
    )
    from ..alphafold2.processor import (
        AlphaFold2InitialGuessProcessor,
        AlphaFold2Processor,
    )
    from ..proteinmpnn.config import ProteinMpnnProcessorConfig
    from ..proteinmpnn.convert import convert_proteinmpnn_model
    from ..proteinmpnn.pipeline import ProteinMpnnPipeline
    from ..proteinmpnn.processor import ProteinMpnnProcessor
    from ..protenix.convert import convert_protenix_model
    from ..protenix.model import ProtenixModel, ProtenixStructureConditionModel
    from ..protenix.pipeline import (
        ProtenixPipeline,
        ProtenixStructureConditionPipeline,
    )
    from ..protenix.processor import (
        ProtenixProcessor,
        ProtenixStructureConditionProcessor,
    )
    from .pipeline import PxDesignBackbonePipeline, PxDesignPipeline
    from .processor import PxDesignProcessor

    backbone_model = convert_pxdesign_model(
        checkpoint_path,
        map_location=map_location,
        dtype=dtype,
    )
    proteinmpnn_model = convert_proteinmpnn_model(
        proteinmpnn_checkpoint_path,
        dtype=dtype,
        expected_noise_level=0.2,
    )
    af2_complex_model = convert_alphafold2_initial_guess_model(
        alphafold2_checkpoint_path,
        dtype=dtype,
    )
    af2_binder_model = convert_alphafold2_model(
        alphafold2_checkpoint_path,
        model_name="model_1_ptm",
        dtype=dtype,
    )
    _own_parameter_storage(af2_complex_model)
    _own_parameter_storage(af2_binder_model)
    protenix_model = cast(
        ProtenixModel,
        convert_protenix_model(
            Path(protenix_checkpoint_path),
            variant="protenix_base_default_v0.5.0",
            dtype=dtype,
        ),
    )
    protenix_structure_model = cast(
        ProtenixStructureConditionModel,
        convert_protenix_model(
            Path(protenix_structure_checkpoint_path),
            variant="protenix_mini_tmpl_v0.5.0",
            dtype=dtype,
        ),
    )
    backbone_dtype = backbone_model.condition_embeddings.input_projection.weight.dtype
    PxDesignPipeline(
        backbone=PxDesignBackbonePipeline(
            model=backbone_model,
            processor=PxDesignProcessor(dtype=backbone_dtype),
        ),
        proteinmpnn=ProteinMpnnPipeline(
            model=proteinmpnn_model,
            processor=ProteinMpnnProcessor(
                ProteinMpnnProcessorConfig(
                    ca_only=proteinmpnn_model.config.ca_only,
                ),
                dtype=dtype,
            ),
        ),
        af2_complex=AlphaFold2InitialGuessPipeline(
            model=af2_complex_model,
            processor=AlphaFold2InitialGuessProcessor(dtype=dtype),
        ),
        af2_binder=AlphaFold2Pipeline(
            model=af2_binder_model,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
        protenix=ProtenixPipeline(
            model=protenix_model,
            processor=ProtenixProcessor(dtype=dtype),
        ),
        protenix_structure=ProtenixStructureConditionPipeline(
            model=protenix_structure_model,
            processor=ProtenixStructureConditionProcessor(dtype=dtype),
        ),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Convert the independent backbone pair or complete fixed PXDesign composition.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    backbone = commands.add_parser(
        "backbone",
        help="convert only the independently callable PXDesign-d pair",
    )
    backbone.add_argument(
        "checkpoint",
        type=Path,
        help="released PXDesign-d checkpoint (e.g. pxdesign_v0.1.0.pt)",
    )
    backbone.add_argument("output_dir", type=Path)
    backbone.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    backbone.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    backbone.add_argument(
        "--token",
        default=None,
        help="optional Hugging Face token for --repo-id",
    )
    backbone.add_argument(
        "--private",
        action="store_true",
        help="create the --repo-id repo as private",
    )

    complete = commands.add_parser(
        "complete",
        help="convert the released fixed six-pair Preview/Extended composition",
    )
    complete.add_argument("checkpoint", type=Path, help="released PXDesign-d checkpoint")
    complete.add_argument(
        "proteinmpnn_checkpoint",
        type=Path,
        help="released ProteinMPNN v_48_020 checkpoint",
    )
    complete.add_argument(
        "alphafold2_checkpoint",
        type=Path,
        help="official AlphaFold parameters containing model_1_ptm",
    )
    complete.add_argument(
        "protenix_checkpoint",
        type=Path,
        help="protenix_base_default_v0.5.0 checkpoint",
    )
    complete.add_argument(
        "protenix_structure_checkpoint",
        type=Path,
        help="protenix_mini_tmpl_v0.5.0 checkpoint",
    )
    complete.add_argument("output_dir", type=Path)
    complete.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    complete.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    complete.add_argument(
        "--token",
        default=None,
        help="optional Hugging Face token for --repo-id",
    )
    complete.add_argument(
        "--private",
        action="store_true",
        help="create the --repo-id repo as private",
    )

    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    if args.command == "backbone":
        convert_pxdesign_backbone_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
    else:
        convert_pxdesign_checkpoint(
            args.checkpoint,
            args.proteinmpnn_checkpoint,
            args.alphafold2_checkpoint,
            args.protenix_checkpoint,
            args.protenix_structure_checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )


def _extract_reference_state(checkpoint_or_state_dict: Mapping[str, Any]) -> dict[str, Tensor]:
    state = checkpoint_or_state_dict.get("model", checkpoint_or_state_dict)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a model mapping or be a state dict")
    normalized = {}
    for key, value in state.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized:
            raise ValueError(f"checkpoint keys collide after removing module. prefix: {normalized_key!r}")
        normalized[normalized_key] = value
    return normalized


def _pxdesign_reference_key(model_key: str) -> _ReferenceKey:
    if model_key.startswith("condition_embeddings.input_projection."):
        return "design_condition_embedder.input_embedder.input_map." + model_key.removeprefix(
            "condition_embeddings.input_projection."
        )
    if model_key.startswith("condition_embeddings.template_embedder.embedding."):
        return "design_condition_embedder.condition_template_embedder.embedder." + model_key.removeprefix(
            "condition_embeddings.template_embedder.embedding."
        )
    if model_key.startswith("condition_embeddings.atom_encoder."):
        reference_key = _reference_key(
            "input_embeddings.atom_encoder." + model_key.removeprefix("condition_embeddings.atom_encoder.")
        )
        source = "input_embedder.atom_attention_encoder."
        target = "design_condition_embedder.input_embedder.atom_attention_encoder."
        if isinstance(reference_key, tuple):
            return tuple(target + key[len(source) :] if key.startswith(source) else key for key in reference_key)
        return target + reference_key[len(source) :] if reference_key.startswith(source) else reference_key
    if model_key.startswith("coordinate_denoiser."):
        return _reference_key(model_key)
    return model_key


def _reference_key(model_key: str) -> _ReferenceKey:
    r"""
    Return the released PxDesign checkpoint key for an OFoldX model key.
    """

    key = _replace_top_level(model_key)
    key = _replace_token_stack_key(key)
    key = _replace_denoiser_key(key)
    key = _replace_atom_stack_key(key)
    key = _replace_atom_key(key)
    key = _replace_attention_key(key)
    key = _replace_attention_projection_key(key)
    key = _replace_transition_key(key)
    return _replace_conditioned_norm_key(key)


def _lookup_reference_value(reference_state: Mapping[str, Tensor], reference_key: _ReferenceKey) -> _ReferenceValue:
    r"""
    Read the source tensor or tensors named by one target key.
    """

    if isinstance(reference_key, tuple):
        return tuple(reference_state[key] for key in reference_key)
    return reference_state[reference_key]


def _transform_reference_value(model_key: str, reference_value: _ReferenceValue) -> Tensor:
    r"""
    Apply tensor transforms required by the released PxDesign format.
    """

    if isinstance(reference_value, tuple):
        if model_key.endswith(".conditioning_projection.weight"):
            return torch.cat(reference_value, dim=0)
        raise RuntimeError(f"unexpected multi-tensor reference mapping for {model_key}")

    if model_key.endswith(".noise_embedding.projection.weight") and reference_value.dim() == 1:
        return reference_value.unsqueeze(-1)
    return reference_value


def _translate_reference_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], _ReferenceKey],
) -> dict[str, Tensor]:
    r"""
    Translate source names and values for one target state dict.
    """

    translated = {}
    for model_key in model.state_dict():
        reference_key = key_fn(model_key)
        reference_value = _lookup_reference_value(reference_state, reference_key)
        translated[model_key] = _transform_reference_value(model_key, reference_value)
    return translated


def _remap_token_transformer_blocks(key: str) -> str:
    r"""
    Map a consolidated token transformer onto Protenix block names.
    """

    rules = [
        (r"\.pair_norms\.(\d+)\.", r".blocks.\1.attention_pair_bias.layernorm_z."),
        (r"\.bias_projections\.(\d+)\.", r".blocks.\1.attention_pair_bias.linear_nobias_z."),
        (r"\.transformer\.blocks\.(\d+)\.transition\.", r".blocks.\1.conditioned_transition_block."),
        (r"\.transformer\.blocks\.(\d+)\.conditioned_norm\.", r".blocks.\1.attention_pair_bias.layernorm_a."),
        (r"\.transformer\.blocks\.(\d+)\.attention\.", r".blocks.\1.attention_pair_bias.attention."),
        (
            r"\.transformer\.blocks\.(\d+)\.output_gate_projection\.",
            r".blocks.\1.attention_pair_bias.linear_a_last.",
        ),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_top_level(key: str) -> str:
    replacements = {
        "input_embeddings.atom_encoder": "input_embedder.atom_attention_encoder",
        "coordinate_denoiser": "diffusion_module",
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return key


def _replace_token_stack_key(key: str) -> str:
    if ".token_encoder." not in key:
        return key
    key = key.replace(".token_refiner.token_encoder.", ".diffusion_transformer.")
    return _remap_token_transformer_blocks(key)


def _replace_denoiser_key(key: str) -> str:
    if not key.startswith("diffusion_module."):
        return key
    return _replace_many(
        key,
        {
            ".conditioner.relative_position_embedder.": ".diffusion_conditioning.relpe.",
            ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
            ".conditioner.pair_input_norm.": ".diffusion_conditioning.layernorm_z.",
            ".conditioner.pair_projection.": ".diffusion_conditioning.linear_no_bias_z.",
            ".conditioner.pair_transitions.0.": ".diffusion_conditioning.transition_z1.",
            ".conditioner.pair_transitions.1.": ".diffusion_conditioning.transition_z2.",
            ".conditioner.single_conditioning_norm.": ".diffusion_conditioning.layernorm_s.",
            ".conditioner.single_conditioning_projection.": ".diffusion_conditioning.linear_no_bias_s.",
            ".conditioner.noise_embedding.projection.weight": ".diffusion_conditioning.fourier_embedding.w",
            ".conditioner.noise_embedding.projection.bias": ".diffusion_conditioning.fourier_embedding.b",
            ".conditioner.noise_norm.": ".diffusion_conditioning.layernorm_n.",
            ".conditioner.noise_projection.": ".diffusion_conditioning.linear_no_bias_n.",
            ".conditioner.single_transitions.0.": ".diffusion_conditioning.transition_s1.",
            ".conditioner.single_transitions.1.": ".diffusion_conditioning.transition_s2.",
            ".atom_encoder.": ".atom_attention_encoder.",
            ".token_refiner.single_norm.": ".layernorm_s.",
            ".token_refiner.single_to_token_projection.": ".linear_no_bias_s.",
            ".token_refiner.token_encoder.": ".diffusion_transformer.",
            ".token_refiner.token_norm.": ".layernorm_a.",
            ".atom_decoder.": ".atom_attention_decoder.",
        },
    )


def _replace_atom_stack_key(key: str) -> str:
    if ".atom_self_attention." not in key:
        return key
    block = r".atom_transformer.diffusion_transformer.blocks.\1"
    source_block = r"\.atom_self_attention\.atom_attention\.transformer\.blocks\.(\d+)\."
    rules = [
        (r"\.atom_self_attention\.pair_norms\.(\d+)\.", block + r".attention_pair_bias.layernorm_z."),
        (
            r"\.atom_self_attention\.bias_projections\.(\d+)\.",
            block + r".attention_pair_bias.linear_nobias_z.",
        ),
        (source_block + r"transition\.", block + r".conditioned_transition_block."),
        (source_block + r"conditioned_norm\.", block + r".attention_pair_bias.layernorm_a."),
        (source_block + r"key_value_norm\.", block + r".attention_pair_bias.layernorm_kv."),
        (source_block + r"attention\.", block + r".attention_pair_bias.attention."),
        (source_block + r"output_gate_projection\.", block + r".attention_pair_bias.linear_a_last."),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_atom_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".atom_conditioner.reference_position_projection.": ".linear_no_bias_ref_pos.",
            ".atom_conditioner.reference_charge_projection.": ".linear_no_bias_ref_charge.",
            ".atom_conditioner.atom_feature_projection.": ".linear_no_bias_f.",
            ".atom_pair_encoder.distance_vector_projection.": ".linear_no_bias_d.",
            ".atom_pair_encoder.inverse_distance_projection.": ".linear_no_bias_invd.",
            ".atom_pair_encoder.distance_mask_projection.": ".linear_no_bias_v.",
            ".atom_pair_encoder.left_atom_projection.": ".linear_no_bias_cl.",
            ".atom_pair_encoder.right_atom_projection.": ".linear_no_bias_cm.",
            ".atom_pair_encoder.feed_forward_input_projection.": ".small_mlp.1.",
            ".atom_pair_encoder.feed_forward_hidden_projection.": ".small_mlp.3.",
            ".atom_pair_encoder.feed_forward_output_projection.": ".small_mlp.5.",
            ".atom_self_attention.": ".atom_transformer.diffusion_transformer.",
            ".atom_attention_encoder.token_projection.": ".atom_attention_encoder.linear_no_bias_q.",
            ".atom_attention_decoder.token_projection.": ".atom_attention_decoder.linear_no_bias_a.",
            ".single_norm.": ".layernorm_s.",
            ".single_to_atom_projection.": ".linear_no_bias_s.",
            ".atom_attention_encoder.pair_norm.": ".atom_attention_encoder.layernorm_z.",
            ".pair_projection.": ".linear_no_bias_z.",
            ".noisy_position_projection.": ".linear_no_bias_r.",
            ".atom_attention_decoder.output_projection.": ".atom_attention_decoder.linear_no_bias_out.",
            ".atom_attention_decoder.query_norm.": ".atom_attention_decoder.layernorm_q.",
        },
    )


def _replace_attention_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".pair_attention.": ".attention_pair_bias.",
            ".hidden_norm.": ".layernorm_a.",
            ".atom_norm.": ".layernorm_a.",
            ".key_value_norm.": ".layernorm_kv.",
            ".attention_pair_bias.pair_norm.": ".attention_pair_bias.layernorm_z.",
            ".attention_pair_bias.bias_projection.": ".attention_pair_bias.linear_nobias_z.",
            ".attention_pair_bias.conditioning_gate_projection.": ".attention_pair_bias.linear_a_last.",
        },
    )


def _replace_attention_projection_key(key: str) -> str:
    return _replace_many(
        key,
        {
            "relative_position_encoding.output_projection.": "relative_position_encoding.linear_no_bias.",
            ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
            ".fourier_embedding.frequency": ".fourier_embedding.w",
            ".fourier_embedding.phase": ".fourier_embedding.b",
            ".attention.projection.query_projection.": ".attention.linear_q.",
            ".attention.projection.key_projection.": ".attention.linear_k.",
            ".attention.projection.value_projection.": ".attention.linear_v.",
            ".attention.projection.output_projection.": ".attention.linear_o.",
            ".attention.projection.gate_projection.": ".attention.linear_g.",
            ".mha.query_projection.": ".mha.linear_q.",
            ".mha.key_projection.": ".mha.linear_k.",
            ".mha.value_projection.": ".mha.linear_v.",
            ".mha.output_projection.": ".mha.linear_o.",
            ".mha.gate_projection.": ".mha.linear_g.",
            ".relative_position_encoding.output_projection.": ".relative_position_encoding.linear_no_bias.",
        },
    )


def _replace_transition_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".input_norm.": ".layernorm1.",
            ".conditioned_norm.": ".adaln.",
            (
                ".conditioned_transition_block.feed_forward.gate_projection."
            ): ".conditioned_transition_block.linear_nobias_a1.",
            (
                ".conditioned_transition_block.feed_forward.value_projection."
            ): ".conditioned_transition_block.linear_nobias_a2.",
            (
                ".conditioned_transition_block.feed_forward.output_projection."
            ): ".conditioned_transition_block.linear_nobias_b.",
            ".conditioned_transition_block.conditioning_gate_projection.": ".conditioned_transition_block.linear_s.",
            ".transition_z1.feed_forward.gate_projection.": ".transition_z1.linear_no_bias_a.",
            ".transition_z1.feed_forward.value_projection.": ".transition_z1.linear_no_bias_b.",
            ".transition_z1.feed_forward.output_projection.": ".transition_z1.linear_no_bias.",
            ".transition_z2.feed_forward.gate_projection.": ".transition_z2.linear_no_bias_a.",
            ".transition_z2.feed_forward.value_projection.": ".transition_z2.linear_no_bias_b.",
            ".transition_z2.feed_forward.output_projection.": ".transition_z2.linear_no_bias.",
            ".transition_s1.feed_forward.gate_projection.": ".transition_s1.linear_no_bias_a.",
            ".transition_s1.feed_forward.value_projection.": ".transition_s1.linear_no_bias_b.",
            ".transition_s1.feed_forward.output_projection.": ".transition_s1.linear_no_bias.",
            ".transition_s2.feed_forward.gate_projection.": ".transition_s2.linear_no_bias_a.",
            ".transition_s2.feed_forward.value_projection.": ".transition_s2.linear_no_bias_b.",
            ".transition_s2.feed_forward.output_projection.": ".transition_s2.linear_no_bias.",
        },
    )


def _replace_conditioned_norm_key(key: str) -> _ReferenceKey:
    if key.endswith(".conditioning_projection.weight"):
        base = key[: -len("conditioning_projection.weight")]
        return (base + "linear_s.weight", base + "linear_nobias_s.weight")
    if key.endswith(".conditioning_scale_bias"):
        base = key[: -len("conditioning_scale_bias")]
        return base + "linear_s.bias"
    return _replace_many(
        key,
        {
            ".hidden_norm.": ".layernorm_a.",
            ".conditioning_norm.": ".layernorm_s.",
            ".gate_projection.": ".linear_s.",
            ".shift_projection.": ".linear_nobias_s.",
        },
    )


def _replace_many(key: str, replacements: Mapping[str, str]) -> str:
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


if __name__ == "__main__":
    main()
