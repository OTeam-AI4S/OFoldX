# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
PXDesign checkpoint conversion.

Strict translation of the official PxDesign v0.1 checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor, nn

from ...conversion import SourceCheckpoint, save_repository
from ...data.tokenization import TOKEN_INDEX
from ...errors import RequestRef, SourceRef, fail
from ..alphafold2.config import AlphaFold2MsaConfig
from ..alphafold2.convert import (
    alphafold2_own_parameter_storage,
    convert_alphafold2_initial_guess_model,
    convert_alphafold2_model,
)
from ..alphafold2.model import AlphaFold2Model
from ..alphafold2.pipeline import (
    AlphaFold2InitialGuessPipeline,
    AlphaFold2Pipeline,
)
from ..alphafold2.processor import (
    AlphaFold2InitialGuessProcessor,
    AlphaFold2Processor,
)
from ..convert import (
    atom_stack_reference_key,
    conditioned_norm_reference_key,
    token_transformer_reference_key,
    translate_target_state,
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
from ..vocabulary import canonical_word_embeddings
from .config import PxDesignConfig
from .model import PxDesignModel
from .pipeline import PxDesignBackbonePipeline, PxDesignPipeline
from .processor import PxDesignProcessor

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
        fail("value_invalid", RequestRef("dtype", str(dtype)))
    reference_state = _extract_reference_state(checkpoint_or_state_dict)
    converted = translate_target_state(
        model, reference_state, _pxdesign_reference_key, _lookup_reference_value, _transform_reference_value
    )

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
    if dtype is not None:
        model.to(dtype=dtype)
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

    model = convert_pxdesign_model(checkpoint_path, map_location=map_location, dtype=dtype)
    processor_dtype = model.condition_embeddings.input_projection.weight.dtype
    save_repository(
        PxDesignBackbonePipeline(model=model, processor=PxDesignProcessor(dtype=processor_dtype)),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter="ofoldx.models.pxdesign.convert.convert_pxdesign_backbone_checkpoint",
        options={"map_location": str(map_location), "dtype": None if dtype is None else str(dtype)},
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
    af2_binder_model = cast(
        AlphaFold2Model,
        convert_alphafold2_model(
            alphafold2_checkpoint_path,
            model_name="model_1_ptm",
            dtype=dtype,
            msa_config=AlphaFold2MsaConfig(
                {
                    "max_msa_clusters": 1,
                    "max_extra_msa": 0,
                    "resample_msa_in_recycling": False,
                    "masked_msa_enabled": False,
                }
            ),
        ),
    )
    alphafold2_own_parameter_storage(af2_complex_model)
    alphafold2_own_parameter_storage(af2_binder_model)
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
    save_repository(
        PxDesignPipeline(
            backbone=PxDesignBackbonePipeline(
                model=backbone_model,
                processor=PxDesignProcessor(dtype=backbone_dtype),
            ),
            inverse_folder=ProteinMpnnPipeline(
                model=proteinmpnn_model,
                processor=ProteinMpnnProcessor(
                    ProteinMpnnProcessorConfig(
                        ca_only=proteinmpnn_model.config.ca_only,
                    ),
                    dtype=dtype,
                ),
            ),
            complex_folder=AlphaFold2InitialGuessPipeline(
                model=af2_complex_model,
                processor=AlphaFold2InitialGuessProcessor(dtype=dtype),
            ),
            binder_folder=AlphaFold2Pipeline(
                model=af2_binder_model,
                processor=AlphaFold2Processor(dtype=dtype),
            ),
            protenix_folder=ProtenixPipeline(
                model=protenix_model,
                processor=ProtenixProcessor(dtype=dtype),
            ),
            structure_conditioned_folder=ProtenixStructureConditionPipeline(
                model=protenix_structure_model,
                processor=ProtenixStructureConditionProcessor(dtype=dtype),
            ),
        ),
        output_dir,
        sources={
            "backbone": checkpoint_path,
            "proteinmpnn": proteinmpnn_checkpoint_path,
            "alphafold2": SourceCheckpoint(alphafold2_checkpoint_path, members=("params_model_1_ptm.npz",)),
            "protenix": protenix_checkpoint_path,
            "protenix_structure": protenix_structure_checkpoint_path,
        },
        converter="ofoldx.models.pxdesign.convert.convert_pxdesign_checkpoint",
        options={"map_location": str(map_location), "dtype": None if dtype is None else str(dtype)},
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
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item="model"))
    normalized = {}
    for key, value in state.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized:
            fail(
                "checkpoint_invalid", SourceRef("checkpoint", 0, item=normalized_key)
            )  # keys collide after removing the module. prefix
        normalized[normalized_key] = value
    return normalized


_CONDITION_ATOM_PREFIXES = tuple(
    f"condition_embeddings.{child}."
    for child in ("atom_conditioner", "atom_pair_encoder", "atom_pair_bias_projection", "atom_encoder")
)


def _pxdesign_reference_key(model_key: str) -> _ReferenceKey:
    if model_key.startswith("condition_embeddings.input_projection."):
        return "design_condition_embedder.input_embedder.input_map." + model_key.removeprefix(
            "condition_embeddings.input_projection."
        )
    if model_key.startswith("condition_embeddings.template_embedder.embedding."):
        return "design_condition_embedder.condition_template_embedder.embedder." + model_key.removeprefix(
            "condition_embeddings.template_embedder.embedding."
        )
    if model_key.startswith(_CONDITION_ATOM_PREFIXES):
        # The local-atom condition stack mirrors the Protenix input embeddings under the design embedder.
        reference_key = _reference_key("input_embeddings." + model_key.removeprefix("condition_embeddings."))
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

    key = _replace_atom_module_key(model_key)
    key = _replace_top_level(key)
    key = _replace_token_stack_key(key)
    key = _replace_denoiser_key(key)
    key = atom_stack_reference_key(key)
    key = _replace_atom_key(key)
    key = _replace_attention_key(key)
    key = _replace_attention_projection_key(key)
    key = _replace_transition_key(key)
    return conditioned_norm_reference_key(key)


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

    if model_key == "condition_embeddings.input_projection.weight":
        return _convert_condition_input_axis(reference_value)
    if model_key.endswith(".noise_embedding.projection.weight") and reference_value.dim() == 1:
        return reference_value.unsqueeze(-1)
    return reference_value


#: The official residue plane's rows, in canonical spelling: 21 protein classes, the five RNA and five DNA standard
#: bases, then the gap. The official plane carries four more columns after them; the design flag is the first.
PXDESIGN_VOCABULARY: tuple[str, ...] = (
    *"ARNDCQEGHILKMFPSTWYVX",
    *(f"RNA_{base}" for base in "AGCUN"),
    *(f"DNA_{base}" for base in "AGCTN"),
    "-",
)
#: Width of the official residue plane (its 32 rows and the four columns after them, the design flag first), and of
#: the token columns that follow it (pLDDT, hotspot and the two added condition one-hots).
_SOURCE_RESIDUE_SIZE = 36
_SUFFIX_SIZE = 10


def _convert_condition_input_axis(source: Tensor) -> Tensor:
    r"""
    Regenerate the condition projection's residue columns from the official plane onto the 72 canonical classes,
    keeping its design column: the official plane states a generated token as its unknown protein
    class and the flag together, where the native features state the flag alone.
    """

    plane = _SOURCE_RESIDUE_SIZE + _SUFFIX_SIZE
    prefix, residue, suffix = source[..., :-plane], source[..., -plane:-_SUFFIX_SIZE], source[..., -_SUFFIX_SIZE:]
    size = len(PXDESIGN_VOCABULARY)
    (classes,) = canonical_word_embeddings(residue[..., :size], vocabulary=PXDESIGN_VOCABULARY, dim=-1)
    unknown = PXDESIGN_VOCABULARY.index("X")
    classes[..., TOKEN_INDEX["OTHER"]] = classes[..., TOKEN_INDEX["X"]]
    design = residue[..., size : size + 1] - residue[..., unknown : unknown + 1]
    return torch.cat((prefix, classes, design, suffix), dim=-1)


def _replace_top_level(key: str) -> str:
    replacements = {
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
    return token_transformer_reference_key(key)


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
            ".conditioner.single_norm.": ".diffusion_conditioning.layernorm_s.",
            ".conditioner.single_projection.": ".diffusion_conditioning.linear_no_bias_s.",
            ".conditioner.noise_embedding.projection.weight": ".diffusion_conditioning.fourier_embedding.w",
            ".conditioner.noise_embedding.projection.bias": ".diffusion_conditioning.fourier_embedding.b",
            ".conditioner.noise_norm.": ".diffusion_conditioning.layernorm_n.",
            ".conditioner.noise_projection.": ".diffusion_conditioning.linear_no_bias_n.",
            ".conditioner.transitions.0.": ".diffusion_conditioning.transition_s1.",
            ".conditioner.transitions.1.": ".diffusion_conditioning.transition_s2.",
            ".token_refiner.single_norm.": ".layernorm_s.",
            ".token_refiner.single_to_token_projection.": ".linear_no_bias_s.",
            ".token_refiner.token_encoder.": ".diffusion_transformer.",
            ".token_refiner.token_norm.": ".layernorm_a.",
        },
    )


# The Protenix atom attention encoder or decoder that holds each local-atom child of the input embeddings
# and the coordinate denoiser. The children split one Protenix module by concern; the prefixes they keep
# resolve in ``atom_stack_reference_key`` and ``_replace_atom_key``.
_ATOM_MODULE_PREFIXES = (
    ("input_embeddings.atom_encoder.", "input_embedder.atom_attention_encoder."),
    ("input_embeddings.", "input_embedder.atom_attention_encoder."),
    ("coordinate_denoiser.atom_conditioner.", "diffusion_module.atom_attention_encoder.atom_conditioner."),
    ("coordinate_denoiser.trunk_conditioning.", "diffusion_module.atom_attention_encoder.trunk_conditioning."),
    ("coordinate_denoiser.atom_pair_encoder.", "diffusion_module.atom_attention_encoder.atom_pair_encoder."),
    (
        "coordinate_denoiser.atom_encoder_pair_bias_projection.",
        "diffusion_module.atom_attention_encoder.atom_pair_bias_projection.",
    ),
    ("coordinate_denoiser.atom_encoder.", "diffusion_module.atom_attention_encoder."),
    (
        "coordinate_denoiser.atom_decoder_pair_bias_projection.",
        "diffusion_module.atom_attention_decoder.atom_pair_bias_projection.",
    ),
    ("coordinate_denoiser.atom_decoder.", "diffusion_module.atom_attention_decoder."),
)


def _replace_atom_module_key(key: str) -> str:
    for source, target in _ATOM_MODULE_PREFIXES:
        if key.startswith(source):
            return target + key.removeprefix(source)
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
            ".atom_pair_encoder.feed_forward.input_projection.": ".small_mlp.1.",
            ".atom_pair_encoder.feed_forward.intermediate_projections.0.": ".small_mlp.3.",
            ".atom_pair_encoder.feed_forward.output_projection.": ".small_mlp.5.",
            ".trunk_conditioning.single_norm.": ".layernorm_s.",
            ".trunk_conditioning.single_projection.": ".linear_no_bias_s.",
            ".trunk_conditioning.pair_norm.": ".layernorm_z.",
            ".trunk_conditioning.pair_projection.": ".linear_no_bias_z.",
            ".atom_attention_encoder.noisy_position_projection.": ".atom_attention_encoder.linear_no_bias_r.",
            ".atom_attention_encoder.token_projection.": ".atom_attention_encoder.linear_no_bias_q.",
            ".atom_attention_decoder.token_projection.": ".atom_attention_decoder.linear_no_bias_a.",
            ".atom_attention_decoder.position_norm.": ".atom_attention_decoder.layernorm_q.",
            ".atom_attention_decoder.position_update_projection.": ".atom_attention_decoder.linear_no_bias_out.",
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


def _replace_many(key: str, replacements: Mapping[str, str]) -> str:
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


if __name__ == "__main__":
    main()
