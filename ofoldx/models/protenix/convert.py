r"""
Protenix checkpoint converter.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import chanfig
import torch
from torch import Tensor, nn

from ...artifacts import save_model
from ...processor import Processor

if TYPE_CHECKING:
    from .model import ProtenixModel


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_ReferenceKey: TypeAlias = str | tuple[str, ...]
_ReferenceValue: TypeAlias = Tensor | tuple[Tensor, ...]


def load_reference_state(checkpoint_path: str | Any, *, map_location: str | Any = "cpu") -> dict[str, Tensor]:
    r"""
    Load a reference checkpoint's tensor state, unwrapping the ``model`` container and ``module.`` prefixes.

    Reads the checkpoint, unwraps a ``{"model": ...}`` wrapper when present, drops the distributed-training
    ``module.`` key prefix, and keeps only tensor-valued entries.

    Args:
        checkpoint_path: Path (or file-like object) accepted by `torch.load`.
        map_location: Device mapping forwarded to `torch.load`; defaults to CPU.

    Returns:
        Mapping from clean checkpoint key to tensor.

    Examples:
        >>> state = load_reference_state("af3.pt")  # doctest: +SKIP
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    reference_state: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = source_key.removeprefix("module.")
        if key in reference_state:
            raise ValueError(f"multiple Protenix source tensors normalize to {key!r}")
        reference_state[key] = value
    return reference_state


def translate_reference_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], _ReferenceKey],
) -> dict[str, Tensor]:
    r"""Translate every target and require every source tensor to participate."""

    translated: dict[str, Tensor] = {}
    consumed: set[str] = set()
    for model_key in model.state_dict():
        reference_key = key_fn(model_key)
        reference_value = _lookup_reference_value(reference_state, reference_key)
        translated[model_key] = _transform_reference_value(model_key, reference_value)
        if isinstance(reference_key, tuple):
            consumed.update(reference_key)
        else:
            consumed.add(reference_key)

    unconsumed = sorted(set(reference_state).difference(consumed))
    if unconsumed:
        preview = ", ".join(repr(key) for key in unconsumed[:20])
        suffix = "" if len(unconsumed) <= 20 else f", ... ({len(unconsumed)} total)"
        raise ValueError(f"unconsumed Protenix source tensors: {preview}{suffix}")
    return translated


def translate_protenix_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every tensor in one released Protenix graph."""

    return translate_reference_state(model, reference_state, _protenix_reference_key)


def convert_protenix_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
) -> ProtenixModel:
    r"""
    Convert a Protenix base checkpoint into a strict-loaded native model.

    Args:
        checkpoint_path: Path to the released Protenix base checkpoint.
        dtype: Optional floating-point dtype for converted weights.

    Returns:
        The converted, strict-loaded `ProtenixModel`. Use `save_protenix_checkpoint` to persist the paired
        model/processor repository locally or on the Hub.

    Examples:
        >>> model = convert_protenix_checkpoint("protenix_base_default_v0.5.0.pt")  # doctest: +SKIP
    """

    from .config import ProtenixConfig, ProtenixConstraintConfig
    from .model import ProtenixConstraintModel, ProtenixModel

    reference_state = load_reference_state(Path(checkpoint_path))
    if "constraint_embedder.pocket_z_embedder.weight" in reference_state:
        model: ProtenixModel = ProtenixConstraintModel(ProtenixConstraintConfig())
    else:
        model = ProtenixModel(ProtenixConfig())
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_state_dict(translate_protenix_state(model, reference_state), strict=True)
    model.eval()
    return model


def save_protenix_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert a Protenix checkpoint and write a routable OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_protenix_checkpoint`][ofoldx.models.protenix.convert.convert_protenix_checkpoint] — which
    auto-selects the default [`ProtenixModel`][] or the constraint
    [`ProtenixConstraintModel`][] from the checkpoint tensors — pairs it with the matching processor,
    and writes a self-contained ``model.json`` / ``processor.json`` / safetensors repository
    via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model].

    The default pair routes through the generic standard ``fold`` route (no marker). The constraint
    pair additionally writes a ``pipeline.json = {"pipeline": "ofoldx.protenix_constraint"}`` marker so
    ``pipeline("fold", model=dir)`` routes to [`ProtenixConstraintPipeline`][], whose ``fold``
    accepts and forwards the released ``atom_contacts`` / ``token_contacts`` / ``pocket`` channels that
    the generic ``FoldPipeline`` cannot.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from .model import ProtenixConstraintModel
    from .processor import ProtenixConstraintProcessor, ProtenixProcessor

    model = convert_protenix_checkpoint(Path(checkpoint_path).expanduser(), dtype=dtype)
    destination = Path(output_dir).expanduser()
    if isinstance(model, ProtenixConstraintModel):
        processor: Processor = ProtenixConstraintProcessor()
    else:
        processor = ProtenixProcessor()
    save_model(destination, model=model, processor=processor)
    if isinstance(model, ProtenixConstraintModel):
        from ...pipelines.registry import PIPELINES
        from .pipeline import ProtenixConstraintPipeline

        pipeline_id = PIPELINES.id_for(ProtenixConstraintPipeline)
        chanfig.NestedDict({"pipeline": pipeline_id}).dump(destination / "pipeline.json")
    if repo_id is not None:
        from ...hub import upload_repository

        upload_repository(destination, repo_id, token=token, private=private)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="released Protenix base checkpoint (e.g. protenix_base_default_v0.5.0.pt)",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    save_protenix_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


def _protenix_reference_key(model_key: str) -> _ReferenceKey:
    if model_key.startswith("constraint_embeddings."):
        relative_key = model_key.removeprefix("constraint_embeddings.")
        transformer_match = re.fullmatch(
            r"substructure_embeddings\.transformer\.blocks\.(\d+)\.(.+)",
            relative_key,
        )
        if transformer_match is not None:
            block_index, suffix = transformer_match.groups()
            source_suffix = {
                "attention.query_projection.weight": "self_attn.in_proj_weight",
                "attention.query_projection.bias": "self_attn.in_proj_bias",
                "attention.key_projection.weight": "self_attn.in_proj_weight",
                "attention.key_projection.bias": "self_attn.in_proj_bias",
                "attention.value_projection.weight": "self_attn.in_proj_weight",
                "attention.value_projection.bias": "self_attn.in_proj_bias",
                "attention.output_projection.weight": "self_attn.out_proj.weight",
                "attention.output_projection.bias": "self_attn.out_proj.bias",
                "attention_norm.weight": "norm1.weight",
                "attention_norm.bias": "norm1.bias",
                "transition.input_projection.weight": "linear1.weight",
                "transition.input_projection.bias": "linear1.bias",
                "transition.output_projection.weight": "linear2.weight",
                "transition.output_projection.bias": "linear2.bias",
                "transition_norm.weight": "norm2.weight",
                "transition_norm.bias": "norm2.bias",
            }[suffix]
            return "constraint_embedder.substructure_z_embedder.transformer.layers." f"{block_index}.{source_suffix}"

        key = "constraint_embedder." + relative_key
        key = _replace_many(
            key,
            {
                ".pocket_projection.": ".pocket_z_embedder.",
                ".token_contact_projection.": ".contact_z_embedder.",
                ".atom_contact_projection.": ".contact_atom_z_embedder.",
                ".substructure_embeddings.": ".substructure_z_embedder.",
            },
        )
        return key.replace(".input_projection.", ".input_proj.").replace(".output_projection.", ".output_proj.")
    return _reference_key(model_key)


def _reference_key(model_key: str) -> _ReferenceKey:
    r"""Return the released Protenix checkpoint key for an OFoldX model key."""

    key = _replace_top_level(model_key)
    key = _replace_template_key(key)
    key = _replace_confidence_key(key)
    key = _replace_token_stack_key(key)
    key = _replace_denoiser_key(key)
    key = _replace_atom_stack_key(key)
    key = _replace_atom_key(key)
    key = _replace_pairformer_key(key)
    key = _replace_attention_key(key)
    key = _replace_attention_projection_key(key)
    key = _replace_transition_key(key)
    return _replace_conditioned_norm_key(key)


def _lookup_reference_value(reference_state: Mapping[str, Tensor], reference_key: _ReferenceKey) -> _ReferenceValue:
    r"""Read the source tensor or tensors named by one target key."""

    if isinstance(reference_key, tuple):
        return tuple(reference_state[key] for key in reference_key)
    return reference_state[reference_key]


def _transform_reference_value(model_key: str, reference_value: _ReferenceValue) -> Tensor:
    r"""Apply tensor transforms required by the released Protenix format."""

    if isinstance(reference_value, tuple):
        if model_key.endswith(".conditioning_projection.weight"):
            return torch.cat(reference_value, dim=0)
        raise RuntimeError(f"unexpected multi-tensor reference mapping for {model_key}")

    if model_key.startswith("constraint_embeddings.substructure_embeddings.transformer.blocks."):
        projection_index = {
            ".attention.query_projection.": 0,
            ".attention.key_projection.": 1,
            ".attention.value_projection.": 2,
        }
        for projection, index in projection_index.items():
            if projection in model_key:
                return reference_value.chunk(3, dim=0)[index]

    if model_key.endswith(".noise_embedding.projection.weight") and reference_value.dim() == 1:
        return reference_value.unsqueeze(-1)
    return reference_value


def _remap_token_transformer_blocks(key: str) -> str:
    r"""Map a consolidated token transformer onto Protenix block names."""

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
        "relative_position_embedder": "relative_position_encoding",
        "single_init_projection": "linear_no_bias_sinit",
        "pair_left_projection": "linear_no_bias_zinit1",
        "pair_right_projection": "linear_no_bias_zinit2",
        "token_bond_projection": "linear_no_bias_token_bond",
        "pair_recycle_projection": "linear_no_bias_z_cycle",
        "single_recycle_projection": "linear_no_bias_s",
        "pair_recycle_norm": "layernorm_z_cycle",
        "single_recycle_norm": "layernorm_s",
        "distogram_head.projection": "distogram_head.linear",
        "input_embeddings.atom_encoder": "input_embedder.atom_attention_encoder",
        "msa_trunk": "msa_module",
        "coordinate_denoiser": "diffusion_module",
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return key


def _replace_template_key(key: str) -> str:
    if not key.startswith("template_embedder."):
        return key
    return _replace_many(
        key,
        {
            ".pair_norm.": ".layernorm_z.",
            ".pair_projection.": ".linear_no_bias_z.",
            ".template_projection.": ".linear_no_bias_a.",
            ".hidden_norm.": ".layernorm_v.",
            ".output_projection.": ".linear_no_bias_u.",
        },
    )


def _replace_confidence_key(key: str) -> str:
    if not key.startswith("confidence_head."):
        return key
    replacements = {
        "confidence_head.pair_embedder.lower_bins": "confidence_head.lower_bins",
        "confidence_head.pair_embedder.upper_bins": "confidence_head.upper_bins",
        "confidence_head.pair_embedder.left_token_projection": "confidence_head.linear_no_bias_s2",
        "confidence_head.pair_embedder.right_token_projection": "confidence_head.linear_no_bias_s1",
        "confidence_head.pair_embedder.distance_one_hot_projection": "confidence_head.linear_no_bias_d",
        "confidence_head.pair_embedder.distance_projection": "confidence_head.linear_no_bias_d_wo_onehot",
        "confidence_head.single_norm": "confidence_head.input_strunk_ln",
        "confidence_head.atom_projector.plddt_weight": "confidence_head.plddt_weight",
        "confidence_head.atom_projector.resolved_weight": "confidence_head.resolved_weight",
        "confidence_head.atom_projector.plddt_norm": "confidence_head.plddt_ln",
        "confidence_head.atom_projector.resolved_norm": "confidence_head.resolved_ln",
        "confidence_head.pair_projector.pae_norm": "confidence_head.pae_ln",
        "confidence_head.pair_projector.pde_norm": "confidence_head.pde_ln",
        "confidence_head.pair_projector.pae_projection": "confidence_head.linear_no_bias_pae",
        "confidence_head.pair_projector.pde_projection": "confidence_head.linear_no_bias_pde",
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
    target_block = r"\.atom_self_attention\.atom_attention\.transformer\.blocks\.(\d+)\."
    rules = [
        (r"\.atom_self_attention\.pair_norms\.(\d+)\.", block + r".attention_pair_bias.layernorm_z."),
        (
            r"\.atom_self_attention\.bias_projections\.(\d+)\.",
            block + r".attention_pair_bias.linear_nobias_z.",
        ),
        (target_block + r"transition\.", block + r".conditioned_transition_block."),
        (target_block + r"conditioned_norm\.", block + r".attention_pair_bias.layernorm_a."),
        (target_block + r"key_value_norm\.", block + r".attention_pair_bias.layernorm_kv."),
        (target_block + r"attention\.", block + r".attention_pair_bias.attention."),
        (target_block + r"output_gate_projection\.", block + r".attention_pair_bias.linear_a_last."),
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
            ".atom_pair_encoder.feed_forward.1.": ".small_mlp.1.",
            ".atom_pair_encoder.feed_forward.3.": ".small_mlp.3.",
            ".atom_pair_encoder.feed_forward.5.": ".small_mlp.5.",
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


def _replace_pairformer_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".triangle_multiplication_outgoing.": ".tri_mul_out.",
            ".triangle_multiplication_incoming.": ".tri_mul_in.",
            ".triangle_attention_starting_node.": ".tri_att_start.",
            ".triangle_attention_ending_node.": ".tri_att_end.",
            ".pair_attention.": ".attention_pair_bias.",
            ".tri_mul_out.input_norm.": ".tri_mul_out.layer_norm_in.",
            ".tri_mul_in.input_norm.": ".tri_mul_in.layer_norm_in.",
            ".tri_mul_out.output_norm.": ".tri_mul_out.layer_norm_out.",
            ".tri_mul_in.output_norm.": ".tri_mul_in.layer_norm_out.",
            ".tri_mul_out.left_projection.": ".tri_mul_out.linear_a_p.",
            ".tri_mul_in.left_projection.": ".tri_mul_in.linear_a_p.",
            ".tri_mul_out.left_gate_projection.": ".tri_mul_out.linear_a_g.",
            ".tri_mul_in.left_gate_projection.": ".tri_mul_in.linear_a_g.",
            ".tri_mul_out.right_projection.": ".tri_mul_out.linear_b_p.",
            ".tri_mul_in.right_projection.": ".tri_mul_in.linear_b_p.",
            ".tri_mul_out.right_gate_projection.": ".tri_mul_out.linear_b_g.",
            ".tri_mul_in.right_gate_projection.": ".tri_mul_in.linear_b_g.",
            ".tri_mul_out.output_projection.": ".tri_mul_out.linear_z.",
            ".tri_mul_in.output_projection.": ".tri_mul_in.linear_z.",
            ".tri_mul_out.gate_projection.": ".tri_mul_out.linear_g.",
            ".tri_mul_in.gate_projection.": ".tri_mul_in.linear_g.",
            ".tri_att_start.pair_norm.": ".tri_att_start.layer_norm.",
            ".tri_att_end.pair_norm.": ".tri_att_end.layer_norm.",
            ".tri_att_start.bias_projection.": ".tri_att_start.linear.",
            ".tri_att_end.bias_projection.": ".tri_att_end.linear.",
            ".tri_att_start.attention.projection.": ".tri_att_start.mha.",
            ".tri_att_end.attention.projection.": ".tri_att_end.mha.",
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
            "msa_module.msa_projection.": "msa_module.linear_no_bias_m.",
            "msa_module.token_projection.": "msa_module.linear_no_bias_s.",
            ".outer_product_mean.": ".outer_product_mean_msa.",
            ".outer_product_mean_msa.msa_norm.": ".outer_product_mean_msa.layer_norm.",
            ".outer_product_mean_msa.left_projection.": ".outer_product_mean_msa.linear_1.",
            ".outer_product_mean_msa.right_projection.": ".outer_product_mean_msa.linear_2.",
            ".outer_product_mean_msa.output_projection.": ".outer_product_mean_msa.linear_out.",
            ".msa_pair_weighted_averaging.msa_norm.": ".msa_pair_weighted_averaging.layernorm_m.",
            ".msa_pair_weighted_averaging.value_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mv.",
            ".msa_pair_weighted_averaging.pair_norm.": ".msa_pair_weighted_averaging.layernorm_z.",
            ".msa_pair_weighted_averaging.pair_projection.": ".msa_pair_weighted_averaging.linear_no_bias_z.",
            ".msa_pair_weighted_averaging.gate_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mg.",
            ".msa_pair_weighted_averaging.output_projection.": ".msa_pair_weighted_averaging.linear_no_bias_out.",
            ".msa_stack.transition.": ".msa_stack.transition_m.",
            ".pair_stack.transition.": ".pair_stack.pair_transition.",
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
            ".conditioned_transition_block.gate_projection.": ".conditioned_transition_block.linear_nobias_a1.",
            ".conditioned_transition_block.value_projection.": ".conditioned_transition_block.linear_nobias_a2.",
            ".conditioned_transition_block.output_projection.": ".conditioned_transition_block.linear_nobias_b.",
            ".conditioned_transition_block.conditioning_gate_projection.": ".conditioned_transition_block.linear_s.",
            ".pair_transition.gate_projection.": ".pair_transition.linear_no_bias_a.",
            ".pair_transition.value_projection.": ".pair_transition.linear_no_bias_b.",
            ".pair_transition.output_projection.": ".pair_transition.linear_no_bias.",
            ".single_transition.gate_projection.": ".single_transition.linear_no_bias_a.",
            ".single_transition.value_projection.": ".single_transition.linear_no_bias_b.",
            ".single_transition.output_projection.": ".single_transition.linear_no_bias.",
            ".transition_m.gate_projection.": ".transition_m.linear_no_bias_a.",
            ".transition_m.value_projection.": ".transition_m.linear_no_bias_b.",
            ".transition_m.output_projection.": ".transition_m.linear_no_bias.",
            ".transition_z1.gate_projection.": ".transition_z1.linear_no_bias_a.",
            ".transition_z1.value_projection.": ".transition_z1.linear_no_bias_b.",
            ".transition_z1.output_projection.": ".transition_z1.linear_no_bias.",
            ".transition_z2.gate_projection.": ".transition_z2.linear_no_bias_a.",
            ".transition_z2.value_projection.": ".transition_z2.linear_no_bias_b.",
            ".transition_z2.output_projection.": ".transition_z2.linear_no_bias.",
            ".transition_s1.gate_projection.": ".transition_s1.linear_no_bias_a.",
            ".transition_s1.value_projection.": ".transition_s1.linear_no_bias_b.",
            ".transition_s1.output_projection.": ".transition_s1.linear_no_bias.",
            ".transition_s2.gate_projection.": ".transition_s2.linear_no_bias_a.",
            ".transition_s2.value_projection.": ".transition_s2.linear_no_bias_b.",
            ".transition_s2.output_projection.": ".transition_s2.linear_no_bias.",
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
