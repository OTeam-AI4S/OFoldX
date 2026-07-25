r"""
Boltz-2 checkpoint → OFoldX key converter.

OFoldX modules use clean, idiomatic parameter names; the public ``boltz2_conf.ckpt`` uses Boltz's
own names. This module owns both the released Boltz-2 source-format rules and the strict target routes.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .config import Boltz2AffinityConfig, Boltz2Config
from .model import Boltz2AffinityModel, Boltz2Model
from .processor import Boltz2AffinityProcessor, Boltz2Processor

Rule = tuple[re.Pattern[str], str | None]


Converter = list[Rule] | Callable[[str], str | None]


TIED_OUTPUT_PROJECTION = re.compile(
    r"(?:\.diffusion_transformer\.layers\.\d+|\.token_transformer(?:_layers\.0)?\.layers\.\d+)"
    r"\.(?:transition\.)?output_projection\.0\.(?:weight|bias)$"
)


def is_tied_output_projection(key: str) -> bool:
    r"""
    Return whether ``key`` is a released Boltz checkpoint's tied projection duplicate.
    """

    return TIED_OUTPUT_PROJECTION.search(key) is not None


def compile_rules(pairs: Iterable[tuple[str, str | None]]) -> list[Rule]:
    r"""
    Compile ordered string regex pairs into conversion rules.
    """

    return [(re.compile(pattern), replacement) for pattern, replacement in pairs]


def _unpack_triple_product_projection(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    marker = ".gate_value_projection."
    if marker not in key:
        return [(key, tensor)]
    prefix, suffix = key.rsplit(marker, 1)
    value, gate = tensor.chunk(2, dim=0)
    return [
        (f"{prefix}.value_projection.{suffix}", value),
        (f"{prefix}.gate_projection.{suffix}", gate),
    ]


CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES: list[Rule] = compile_rules(
    [
        (r"^transition\.adaln\.s_norm\.", "transition.conditioned_norm.conditioning_norm."),
        (r"^transition\.adaln\.s_scale\.", "transition.conditioned_norm.gate_projection."),
        (r"^transition\.adaln\.s_bias\.", "transition.conditioned_norm.shift_projection."),
        (r"^transition\.swish_gate\.0\.", "transition.gate_value_projection."),
        (r"^transition\.a_to_b\.", "transition.expand_projection."),
        (r"^transition\.b_to_a\.", "transition.output_projection."),
        (r"^transition\.output_projection\.0\.", "transition.conditioning_gate_projection."),
        (r"^adaln\.s_norm\.", "conditioned_norm.conditioning_norm."),
        (r"^adaln\.s_scale\.", "conditioned_norm.gate_projection."),
        (r"^adaln\.s_bias\.", "conditioned_norm.shift_projection."),
        (r"^pair_bias_attn\.proj_q\.", "attention.projection.query_projection."),
        (r"^pair_bias_attn\.proj_k\.", "attention.projection.key_projection."),
        (r"^pair_bias_attn\.proj_v\.", "attention.projection.value_projection."),
        (r"^pair_bias_attn\.proj_g\.", "attention.projection.gate_projection."),
        (r"^pair_bias_attn\.proj_o\.", "attention.projection.output_projection."),
        (r"^output_projection_linear\.", "output_gate_projection."),
        (r"^output_projection\.0\.", None),
    ]
)


DISTOGRAM_RULES: list[Rule] = compile_rules([(r"^distogram\.", "projection.")])


BFACTOR_RULES: list[Rule] = compile_rules([(r"^bfactor\.", "")])


ATOM_ENCODER_RULES: list[Rule] = compile_rules(
    [
        (r"^embed_atom_features\.", "atom_feature_projection."),
        (r"^embed_atompair_ref_pos\.", "distance_vector_projection."),
        (r"^embed_atompair_ref_dist\.", "inverse_distance_projection."),
        (r"^embed_atompair_mask\.", "distance_mask_projection."),
        (r"^c_to_p_trans_q\.1\.", "left_atom_projection."),
        (r"^c_to_p_trans_k\.1\.", "right_atom_projection."),
        (r"^p_mlp\.1\.", "feed_forward_input_projection."),
        (r"^p_mlp\.3\.", "feed_forward_hidden_projection."),
        (r"^p_mlp\.5\.", "feed_forward_output_projection."),
        (r"^s_to_c_trans\.0\.", "single_trunk_norm."),
        (r"^s_to_c_trans\.1\.", "single_trunk_projection."),
        (r"^z_to_p_trans\.0\.", "pair_trunk_norm."),
        (r"^z_to_p_trans\.1\.", "pair_trunk_projection."),
    ]
)


def _transition_index_rules() -> list[tuple[str, str | None]]:
    return [
        (r"^transitions\.(\d+)\.norm\.", r"transitions.\1.input_norm."),
        (r"^transitions\.(\d+)\.fc1\.", r"transitions.\1.feed_forward.gate_projection."),
        (r"^transitions\.(\d+)\.fc2\.", r"transitions.\1.feed_forward.value_projection."),
        (r"^transitions\.(\d+)\.fc3\.", r"transitions.\1.feed_forward.output_projection."),
    ]


SINGLE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^norm_single\.", "single_norm."),
        (r"^single_embed\.", "single_projection."),
        (r"^fourier_embed\.proj\.", "noise_fourier.projection."),
        (r"^norm_fourier\.", "fourier_norm."),
        (r"^fourier_to_single\.", "fourier_projection."),
        *_transition_index_rules(),
    ]
)


PAIRWISE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^dim_pairwise_init_proj\.0\.", "input_norm."),
        (r"^dim_pairwise_init_proj\.1\.", "input_projection."),
        *_transition_index_rules(),
    ]
)


def _layer_route(key: str, old: str, new: str, rules: Converter) -> str | None:
    index, _, remainder = key[len(old) :].partition(".")
    converted = _apply_converter(remainder, rules)
    return None if converted is None else f"{new}{index}.{converted}"


def _convert_diffusion_conditioning(key: str) -> str | None:
    if key.startswith("pairwise_conditioner."):
        inner = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return "pairwise_conditioner." + inner if inner is not None else None
    if key.startswith("atom_encoder."):
        inner = convert_keys(key[len("atom_encoder.") :], ATOM_ENCODER_RULES)
        return "atom_encoder." + inner if inner is not None else None
    for boltz, norm_house, projection_house in (
        (
            "atom_enc_proj_z.",
            "atom_encoder_pair_norms.",
            "atom_encoder_pair_bias_projections.",
        ),
        (
            "atom_dec_proj_z.",
            "atom_decoder_pair_norms.",
            "atom_decoder_pair_bias_projections.",
        ),
        (
            "token_trans_proj_z.",
            "token_encoder_pair_norms.",
            "token_encoder_pair_bias_projections.",
        ),
    ):
        if key.startswith(boltz):
            index, _, remainder = key[len(boltz) :].partition(".")
            if remainder.startswith("0."):
                return f"{norm_house}{index}.{remainder.removeprefix('0.')}"
            if remainder.startswith("1."):
                return f"{projection_house}{index}.{remainder.removeprefix('1.')}"
            return None
    return key


def _convert_atom_attention_encoder(key: str) -> str | None:
    if key.startswith("atom_encoder.diffusion_transformer.layers."):
        return _layer_route(
            key,
            "atom_encoder.diffusion_transformer.layers.",
            "atom_transformer.transformer.blocks.",
            CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES,
        )
    if key.startswith("atom_to_token_trans.0."):
        return key.replace("atom_to_token_trans.0.", "token_projection.", 1)
    if key.startswith("r_to_q_trans."):
        return key.replace("r_to_q_trans.", "noisy_position_projection.", 1)
    return key


def _convert_atom_attention_decoder(key: str) -> str | None:
    if key.startswith("atom_decoder.diffusion_transformer.layers."):
        return _layer_route(
            key,
            "atom_decoder.diffusion_transformer.layers.",
            "atom_transformer.transformer.blocks.",
            CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES,
        )
    if key.startswith("a_to_q_trans."):
        return key.replace("a_to_q_trans.", "token_projection.", 1)
    if key.startswith("atom_feat_to_atom_pos_update.0."):
        return key.replace("atom_feat_to_atom_pos_update.0.", "position_norm.", 1)
    if key.startswith("atom_feat_to_atom_pos_update.1."):
        return key.replace("atom_feat_to_atom_pos_update.1.", "position_update_projection.", 1)
    return key


def _convert_score_model(key: str) -> str | None:
    if key.startswith("single_conditioner."):
        inner = convert_keys(key[len("single_conditioner.") :], SINGLE_CONDITIONING_RULES)
        return "single_conditioner." + inner if inner is not None else None
    if key.startswith("atom_attention_encoder."):
        inner = _convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    if key.startswith("atom_attention_decoder."):
        inner = _convert_atom_attention_decoder(key[len("atom_attention_decoder.") :])
        return None if inner is None else "atom_attention_decoder." + inner
    if key.startswith("token_transformer.layers."):
        return _layer_route(
            key,
            "token_transformer.layers.",
            "token_encoder.blocks.",
            CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES,
        )
    if key.startswith("s_to_a_linear.0."):
        return key.replace("s_to_a_linear.0.", "single_to_token_norm.", 1)
    if key.startswith("s_to_a_linear.1."):
        return key.replace("s_to_a_linear.1.", "single_to_token_projection.", 1)
    if key.startswith("a_norm."):
        return key.replace("a_norm.", "token_norm.", 1)
    return key


CONTACT_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^fourier_embedding\.proj\.", "threshold_fourier.projection."),
        (r"^encoder\.", "constraint_encoder."),
        (r"^encoding_unspecified$", "unspecified_bias"),
        (r"^encoding_unselected$", "unselected_bias"),
    ]
)


def _triangle_attention_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.layer_norm\.", f"{clean}.pair_norm."),
        (rf"^{boltz}\.linear\.", f"{clean}.bias_projection."),
        (rf"^{boltz}\.mha\.linear_q\.", f"{clean}.attention.projection.query_projection."),
        (rf"^{boltz}\.mha\.linear_k\.", f"{clean}.attention.projection.key_projection."),
        (rf"^{boltz}\.mha\.linear_v\.", f"{clean}.attention.projection.value_projection."),
        (rf"^{boltz}\.mha\.linear_o\.", f"{clean}.attention.projection.output_projection."),
        (rf"^{boltz}\.mha\.linear_g\.", f"{clean}.attention.projection.gate_projection."),
    ]


def _transition_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.norm\.", f"{clean}.input_norm."),
        (rf"^{boltz}\.fc1\.", f"{clean}.feed_forward.gate_projection."),
        (rf"^{boltz}\.fc2\.", f"{clean}.feed_forward.value_projection."),
        (rf"^{boltz}\.fc3\.", f"{clean}.feed_forward.output_projection."),
    ]


def _triangle_multiplication_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.norm_in\.", f"{clean}.input_norm."),
        (rf"^{boltz}\.norm_out\.", f"{clean}.output_norm."),
        (rf"^{boltz}\.p_out\.", f"{clean}.output_projection."),
        (rf"^{boltz}\.g_out\.", f"{clean}.gate_projection."),
    ]


PAIRFORMER_LAYER_RULES: list[Rule] = compile_rules(
    [
        (r"^pre_norm_s\.", "pair_attention.hidden_norm."),
        (r"^attention\.norm_s\.", "pair_attention.hidden_norm."),
        (r"^attention\.proj_q\.", "pair_attention.attention.projection.query_projection."),
        (r"^attention\.proj_k\.", "pair_attention.attention.projection.key_projection."),
        (r"^attention\.proj_v\.", "pair_attention.attention.projection.value_projection."),
        (r"^attention\.proj_g\.", "pair_attention.attention.projection.gate_projection."),
        (r"^attention\.proj_o\.", "pair_attention.attention.projection.output_projection."),
        (r"^attention\.proj_z\.0\.", "pair_attention.pair_norm."),
        (r"^attention\.proj_z\.1\.", "pair_attention.bias_projection."),
        *_triangle_multiplication_rules("tri_mul_out", "pair_update.triangle_multiplication_outgoing"),
        *_triangle_multiplication_rules("tri_mul_in", "pair_update.triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "pair_update.triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "pair_update.triangle_attention_ending_node"),
        *_transition_rules("transition_s", "single_transition"),
        *_transition_rules("transition_z", "pair_update.transition"),
    ]
)


FUSED_TRIANGLE_PROJECTIONS = {
    "tri_mul_out.p_in.weight": (
        "triangle_multiplication_outgoing",
        "left_projection",
        "right_projection",
    ),
    "tri_mul_out.g_in.weight": (
        "triangle_multiplication_outgoing",
        "left_gate_projection",
        "right_gate_projection",
    ),
    "tri_mul_in.p_in.weight": (
        "triangle_multiplication_incoming",
        "left_projection",
        "right_projection",
    ),
    "tri_mul_in.g_in.weight": (
        "triangle_multiplication_incoming",
        "left_gate_projection",
        "right_gate_projection",
    ),
}


def _split_fused_projection(key: str, tensor: Tensor) -> list[tuple[str, Tensor]] | None:
    if key not in FUSED_TRIANGLE_PROJECTIONS:
        return None
    block, left, right = FUSED_TRIANGLE_PROJECTIONS[key]
    half = tensor.shape[0] // 2
    return [
        (f"{block}.{left}.weight", tensor[:half].clone()),
        (f"{block}.{right}.weight", tensor[half:].clone()),
    ]


def _convert_pairformer_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = _split_fused_projection(key, tensor)
    if split is not None:
        return [(f"pair_update.{new_key}", value) for new_key, value in split]
    new_key = convert_keys(key, PAIRFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


PAIR_ONLY_LAYER_RULES: list[Rule] = compile_rules(
    [
        *_triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_z", "transition"),
    ]
)


def _convert_pair_only_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = _split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MSA_DIRECT_TRIANGLE_RULES: list[Rule] = compile_rules(
    [
        *_triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("z_transition", "transition"),
    ]
)


def _convert_msa_direct_triangle(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = _split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, MSA_DIRECT_TRIANGLE_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MSA_PAIR_WEIGHTED_AVERAGING_RULES = compile_rules(
    [
        (r"^norm_m\.", "msa_norm."),
        (r"^norm_z\.", "pair_norm."),
        (r"^proj_m\.", "value_projection."),
        (r"^proj_g\.", "gate_projection."),
        (r"^proj_z\.", "pair_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)


MSA_TRANSITION_RULES = compile_rules(
    [
        (r"^norm\.", "input_norm."),
        (r"^fc1\.", "feed_forward.gate_projection."),
        (r"^fc2\.", "feed_forward.value_projection."),
        (r"^fc3\.", "feed_forward.output_projection."),
    ]
)


OUTER_PRODUCT_MEAN_RULES = compile_rules(
    [
        (r"^norm\.", "msa_norm."),
        (r"^proj_a\.", "left_projection."),
        (r"^proj_b\.", "right_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)


def _convert_msa_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    if key.startswith("pair_weighted_averaging."):
        converted = _reprefix(
            key,
            "pair_weighted_averaging.",
            "pair_weighted_averaging.",
            MSA_PAIR_WEIGHTED_AVERAGING_RULES,
        )
        return [(cast(str, converted), tensor)]
    if key.startswith("msa_transition."):
        converted = _reprefix(key, "msa_transition.", "msa_transition.", MSA_TRANSITION_RULES)
        return [(cast(str, converted), tensor)]
    if key.startswith("outer_product_mean."):
        converted = _reprefix(key, "outer_product_mean.", "outer_product_mean.", OUTER_PRODUCT_MEAN_RULES)
        return [(cast(str, converted), tensor)]
    if key.startswith("pairformer_layer."):
        inner = _convert_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{new_key}", value) for new_key, value in inner]
    if key.startswith(("tri_mul_out.", "tri_mul_in.", "tri_att_start.", "tri_att_end.", "z_transition.")):
        return [
            (f"triangle_pair_update.{new_key}", value) for new_key, value in _convert_msa_direct_triangle(key, tensor)
        ]
    raise KeyError(f"unmapped MSA layer key: {key}")


def _convert_msa_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"blocks.{index}.{new_key}", value) for new_key, value in _convert_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


def convert_keys(key: str, rules: list[Rule]) -> str | None:
    r"""
    Apply the first matching rename/drop rule to ``key``.
    """

    for pattern, replacement in rules:
        if pattern.search(key):
            return None if replacement is None else pattern.sub(replacement, key)
    return key


def _apply_converter(key: str, converter: Converter) -> str | None:
    if callable(converter):
        return converter(key)
    return convert_keys(key, converter)


def _reprefix(key: str, old_prefix: str, new_prefix: str, converter: Converter) -> str | None:
    converted = _apply_converter(key[len(old_prefix) :], converter)
    return None if converted is None else new_prefix + converted


TRUNK_FLAT = {
    "s_init.weight": "single_init_projection.weight",
    "z_init_1.weight": "pair_left_projection.weight",
    "z_init_2.weight": "pair_right_projection.weight",
    "rel_pos.linear_layer.weight": "relative_position_embedder.output_projection.weight",
    "token_bonds.weight": "token_bond_projection.weight",
    "token_bonds_type.weight": "token_bond_type_embedding.weight",
    "s_norm.weight": "single_recycle_norm.weight",
    "s_norm.bias": "single_recycle_norm.bias",
    "z_norm.weight": "pair_recycle_norm.weight",
    "z_norm.bias": "pair_recycle_norm.bias",
    "s_recycle.weight": "single_recycle_projection.weight",
    "z_recycle.weight": "pair_recycle_projection.weight",
}


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Boltz TemplateV2Module -> OFoldX Boltz2TemplateV2Embeddings.
TEMPLATE_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^z_norm\.", "pair_norm."),
        (r"^v_norm\.", "hidden_norm."),
        (r"^z_proj\.", "pair_projection."),
        (r"^a_proj\.", "template_projection."),
        (r"^u_proj\.", "output_projection."),
    ]
)


def convert_template_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for the whole ``template_module.*`` (prefix-stripped).

    Args:
        key: A ``template_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX template-embedder names (empty when dropped).
    """

    if key.startswith("pairformer.layers."):
        index, _, remainder = key[len("pairformer.layers.") :].partition(".")
        return [
            (f"pair_update_stack.blocks.{index}.{nk}", nv) for nk, nv in _convert_pair_only_layer(remainder, tensor)
        ]
    new_key = convert_keys(key, TEMPLATE_FLAT_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Token-level encodings + conditioning embeddings at the input-embedder top level.
INPUT_EMBEDDER_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^res_type_encoding\.", "residue_type_projection."),
        (r"^msa_profile_encoding\.", "msa_profile_projection."),
        (r"^method_conditioning_init\.", "method_embedding."),
        (r"^modified_conditioning_init\.", "modified_embedding."),
        (r"^cyclic_conditioning_init\.", "cyclic_projection."),
        (r"^mol_type_conditioning_init\.", "molecule_type_embedding."),
        (r"^atom_enc_proj_z\.0\.", "atom_pair_norm."),
        (r"^atom_enc_proj_z\.1\.", "atom_pair_bias_projection."),
        (
            r"^atom_attention_encoder\.atom_to_token_trans\.0\.",
            "atom_attention_encoder.token_projection.",
        ),
    ]
)

_ATTN_OLD = "atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
_ATTN_NEW = "atom_attention_encoder.atom_transformer.transformer.blocks."


def convert_input_embedder_key(key: str) -> str | None:
    r"""
    Route a Boltz ``input_embedder.*`` (prefix-stripped) key to its OFoldX name.

    Args:
        key: An ``input_embedder``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
    """

    if key.startswith("atom_encoder."):
        return _reprefix(key, "atom_encoder.", "atom_encoder.", ATOM_ENCODER_RULES)
    if key.startswith(_ATTN_OLD):
        return reprefix_indexed(key, _ATTN_OLD, _ATTN_NEW, CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES)
    return convert_keys(key, INPUT_EMBEDDER_FLAT_RULES)


def reprefix_indexed(key: str, old_prefix: str, new_prefix: str, sub: Converter) -> str | None:
    r"""
    Like ``_reprefix`` but preserves a ``<index>.`` segment that follows the prefix.

    Args:
        key: The full key beginning with ``old_prefix`` followed by an ``<index>.`` segment.
        old_prefix: The prefix to strip before the index.
        new_prefix: The prefix to prepend to the ``<index>.`` and converted remainder.
        sub: Converter applied to the remainder after the index.

    Returns:
        ``new_prefix<index>.`` + converted remainder, or ``None`` when ``sub`` drops the remainder.
    """

    index, _, remainder = key[len(old_prefix) :].partition(".")
    converted = _apply_converter(remainder, sub)
    return None if converted is None else f"{new_prefix}{index}.{converted}"


_AFFINITY_PREFIXES = ("affinity_module1.", "affinity_module2.")


CONFIDENCE_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "distance_embedding."),
        (r"^s_inputs_norm\.", "input_single_norm."),
        (r"^s_norm\.", "single_norm."),
        (r"^z_norm\.", "pair_norm."),
        (r"^s_input_to_s\.", "input_single_projection."),
        (r"^s_to_z_transpose\.", "single_to_pair_col."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^s_to_z_prod_out\.", "single_to_pair_product_out."),
        (r"^s_to_z\.", "single_to_pair_row."),
        (r"^rel_pos\.linear_layer\.", "relative_position_embedder.output_projection."),
        (r"^token_bonds_type\.", "token_bond_type_embedding."),
        (r"^token_bonds\.", "token_bond_projection."),
        (r"^boundaries$", "boundaries"),
    ]
)
CONFIDENCE_HEAD_RULES: list[Rule] = compile_rules(
    [
        (r"^to_pae_intra_logits\.", "pae_intra_projection."),
        (r"^to_pae_inter_logits\.", "pae_inter_projection."),
        (r"^to_pde_intra_logits\.", "pde_intra_projection."),
        (r"^to_pde_inter_logits\.", "pde_inter_projection."),
        (r"^to_plddt_logits\.", "plddt_projection."),
        (r"^to_resolved_logits\.", "resolved_projection."),
    ]
)


def convert_confidence_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for source ``confidence_module.*`` (prefix-stripped).

    Args:
        key: A ``confidence_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX confidence-module names (empty when dropped).
    """

    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in _convert_pairformer_layer(remainder, tensor)
        ]
    if key.startswith("confidence_heads."):
        new = convert_keys(key[len("confidence_heads.") :], CONFIDENCE_HEAD_RULES)
        return [("output_head." + new, tensor)] if new is not None else []
    new = convert_keys(key, CONFIDENCE_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


AFFINITY_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "dist_bin_pairwise_embed."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^z_norm\.", "pair_norm."),
        (r"^z_linear\.", "pair_projection."),
        (r"^boundaries$", "boundaries"),
    ]
)
AFFINITY_HEAD_RULES: list[Rule] = compile_rules(
    [
        (r"^affinity_out_mlp\.0\.", "pair_feature_input_projection."),
        (r"^affinity_out_mlp\.2\.", "pair_feature_output_projection."),
        (r"^to_affinity_pred_value\.0\.", "affinity_value_input_projection."),
        (r"^to_affinity_pred_value\.2\.", "affinity_value_hidden_projection."),
        (r"^to_affinity_pred_value\.4\.", "affinity_value_output_projection."),
        (r"^to_affinity_pred_score\.0\.", "affinity_score_input_projection."),
        (r"^to_affinity_pred_score\.2\.", "affinity_score_hidden_projection."),
        (r"^to_affinity_pred_score\.4\.", "affinity_score_output_projection."),
        (r"^to_affinity_logits_binary\.", "binder_logit_projection."),
    ]
)
AFFINITY_PAIR_UPDATE_BLOCK_RULES: list[Rule] = compile_rules(
    [
        *_triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_z", "transition"),
    ]
)


def convert_affinity_pair_update_block(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one upstream affinity Pairformer layer into a triangle pair-update block.

    The affinity head uses ``TrianglePairUpdateStack``, whose blocks own one pair ``transition``.
    """

    split = _split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, AFFINITY_PAIR_UPDATE_BLOCK_RULES)
    return [] if new_key is None else [(new_key, tensor)]


def convert_affinity_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for a prefix-stripped upstream ``affinity_module*.*`` key.

    Args:
        key: Checkpoint key with the ``affinity_module*.`` prefix already stripped.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX affinity-module names.
    """

    if key.startswith("pairwise_conditioner."):
        new = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return [("pairwise_conditioner." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"pair_update_stack.blocks.{index}.{nk}", nv)
            for nk, nv in convert_affinity_pair_update_block(remainder, tensor)
        ]
    if key.startswith("affinity_heads."):
        new = convert_keys(key[len("affinity_heads.") :], AFFINITY_HEAD_RULES)
        return [("output_head." + new, tensor)] if new is not None else []
    new = convert_keys(key, AFFINITY_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


def convert_affinity_modules(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert upstream affinity head weights into ``Boltz2AffinityModel`` module names.

    Args:
        state_dict: A full Boltz-2 affinity checkpoint ``state_dict``.

    Returns:
        Converted affinity head tensors under ``affinity_heads.*``.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for index, module in enumerate(("affinity_module1", "affinity_module2")):
            prefix = module + "."
            if key.startswith(prefix):
                entries = convert_affinity_module(key[len(prefix) :], value)
                if not entries:
                    raise KeyError(f"unmapped affinity checkpoint key: {key}")
                for packed_key, packed_value in entries:
                    for new_key, new_value in _unpack_triple_product_projection(packed_key, packed_value):
                        converted[f"affinity_heads.{index}.{new_key}"] = new_value
                break
    return converted


def convert_trunk_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Map one Boltz checkpoint key to its ``Boltz2Model`` trunk param(s); ``[]`` drops non-trunk keys.

    Routes the key to the per-subsystem converter matching its prefix (input embedder, template, MSA,
    Pairformer, contact conditioning, distogram, diffusion, score model, confidence, B-factor) or the top-level
    trunk rename table.

    Args:
        key: A full Boltz checkpoint key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the ``Boltz2Model`` param names; empty for non-trunk keys.
    """

    if key.startswith("input_embedder."):
        new = convert_input_embedder_key(key[len("input_embedder.") :])
        return [("input_embeddings." + new, tensor)] if new is not None else []
    if key.startswith("template_module."):
        inner = convert_template_module(key[len("template_module.") :], tensor)
        return [("template_embedder." + nk, nv) for nk, nv in inner]
    if key.startswith("msa_module."):
        inner = _convert_msa_module(key[len("msa_module.") :], tensor)
        return [("msa_trunk." + nk, nv) for nk, nv in inner]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        inner = _convert_pairformer_layer(remainder, tensor)
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in inner]
    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("distogram_module."):
        new = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = _convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("denoiser_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = _convert_score_model(key[len("structure_module.score_model.") :])
        return [("coordinate_diffusion.denoiser." + new, tensor)] if new is not None else []
    if key.startswith("confidence_module."):
        inner = convert_confidence_module(key[len("confidence_module.") :], tensor)
        return [("confidence_head." + nk, nv) for nk, nv in inner]
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in TRUNK_FLAT:
        return [(TRUNK_FLAT[key], tensor)]
    return []


def convert_trunk(state_dict: Mapping[str, torch.Tensor], *, allow_affinity: bool = False) -> dict[str, torch.Tensor]:
    r"""
    Convert a full Boltz-2 ``state_dict`` to the ``Boltz2Model`` parameter tree.

    Applies [`convert_trunk_key`][ofoldx.models.boltz2.convert.convert_trunk_key] to every entry, then packs
    the AdaLN scale/shift projections so the result ``strict``-loads into the OFoldX modules. Unknown keys fail;
    ``allow_affinity`` skips only the two exact affinity-module prefixes handled by the affinity converter.

    Args:
        state_dict: A full Boltz-2 checkpoint ``state_dict``.
        allow_affinity: Whether the released affinity-module prefixes may be routed by a separate converter.

    Returns:
        The converted trunk ``state_dict`` under the ``Boltz2Model`` param names.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        entries = convert_trunk_key(key, value)
        if not entries:
            if is_tied_output_projection(key):
                continue
            if allow_affinity and key.startswith(_AFFINITY_PREFIXES):
                continue
            raise KeyError(f"unmapped Boltz-2 checkpoint key: {key}")
        for packed_key, packed_value in entries:
            for new_key, new_value in _unpack_triple_product_projection(packed_key, packed_value):
                converted[new_key] = new_value
    _pack_conditioned_adaptive_norms(converted)
    return converted


def _pack_conditioned_adaptive_norms(state_dict: dict[str, torch.Tensor]) -> None:
    r"""
    Pack upstream Boltz AdaLN scale/shift projections into the shared fused layout.
    """

    # The fused module splits scale first and shift second; only upstream s_scale carries a bias.
    scale_weight_suffix = ".conditioned_norm.gate_projection.weight"
    for scale_weight_key in tuple(key for key in state_dict if key.endswith(scale_weight_suffix)):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state_dict.pop(scale_weight_key)
        scale_bias = state_dict.pop(prefix + "gate_projection.bias")
        shift_weight = state_dict.pop(prefix + "shift_projection.weight")
        state_dict[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state_dict[prefix + "conditioning_scale_bias"] = scale_bias


def convert_boltz2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2Model:
    r"""
    Convert a released Boltz-2 structure-prediction checkpoint.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    converted = _cast_floating_tensors(convert_trunk(checkpoint["state_dict"]), dtype)
    with torch.device("meta"):
        model = Boltz2Model(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_boltz2_affinity_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2AffinityConfig | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2AffinityModel:
    r"""
    Convert the released two-head Boltz-2 affinity checkpoint.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = convert_trunk(state, allow_affinity=True)
    converted.update(convert_affinity_modules(state))
    converted = _cast_floating_tensors(converted, dtype)
    with torch.device("meta"):
        model = Boltz2AffinityModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def _cast_floating_tensors(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    r"""
    Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested.
    """

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def save_boltz2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert a Boltz-2 structure-prediction checkpoint and write a standard OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_boltz2_checkpoint`][ofoldx.models.boltz2.convert.convert_boltz2_checkpoint], pairs the
    model with the config-less [`Boltz2Processor`][], and writes a self-contained ``model.json`` /
    ``processor.json`` / safetensors repository marked with the exact [`Boltz2Pipeline`][].
    The Processor repository metadata records its floating-point dtype even though it has no
    family configuration.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository. No
    organization or token is assumed and nothing is published without an explicit ``repo_id``.
    """

    from .pipeline import Boltz2Pipeline

    model = convert_boltz2_checkpoint(checkpoint_path, config=config, dtype=dtype)
    Boltz2Pipeline(model=model, processor=Boltz2Processor(dtype=dtype)).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def save_boltz2_affinity_checkpoint(
    affinity_checkpoint_path: str | PathLike[str],
    structure_checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    affinity_config: Boltz2AffinityConfig | None = None,
    structure_config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the Boltz-2 affinity + structure checkpoints and write the composite repository.

    The affinity deliverable is a multi-pair [`Boltz2AffinityPipeline`][]: the affinity
    head runs only after a structure/confidence pass, so it is not a standalone Pipeline.
    This writes the composite repository — the
    affinity pair at the root, the structure/confidence pair ([`Boltz2Model`][] /
    [`Boltz2Processor`][]) at the fixed ``structure/`` sub-directory, and the one-field
    ``pipeline.json`` marker — via ``Boltz2AffinityPipeline.save`` so ``pipeline(..., model=dir)``
    routes it back through the multi-pair loader.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the composite repository.
    """

    from .pipeline import Boltz2AffinityPipeline

    affinity_model = convert_boltz2_affinity_checkpoint(
        affinity_checkpoint_path,
        config=affinity_config,
        dtype=dtype,
    )
    structure_model = convert_boltz2_checkpoint(
        structure_checkpoint_path,
        config=structure_config,
        dtype=dtype,
    )
    Boltz2AffinityPipeline(
        model=affinity_model,
        processor=Boltz2AffinityProcessor(dtype=dtype),
        structure_model=structure_model,
        structure_processor=Boltz2Processor(dtype=dtype),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run boltz2 checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold = subparsers.add_parser("fold", help="convert the Boltz-2 structure/confidence checkpoint")
    fold.add_argument("checkpoint", type=Path, help="released Boltz-2 fold .ckpt")
    fold.add_argument("output_dir", type=Path)
    fold.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    fold.add_argument("--repo-id", default=None, help="optional Hub repo id; local-only when omitted")
    fold.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    fold.add_argument("--private", action="store_true", help="create the --repo-id repo as private")

    affinity = subparsers.add_parser("affinity", help="convert the Boltz-2 affinity + structure checkpoints")
    affinity.add_argument("affinity_checkpoint", type=Path, help="released Boltz-2 affinity .ckpt")
    affinity.add_argument("structure_checkpoint", type=Path, help="released Boltz-2 structure/confidence .ckpt")
    affinity.add_argument("output_dir", type=Path)
    affinity.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    affinity.add_argument("--repo-id", default=None, help="optional Hub repo id; local-only when omitted")
    affinity.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    affinity.add_argument("--private", action="store_true", help="create the --repo-id repo as private")

    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    if args.command == "fold":
        save_boltz2_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
    else:
        save_boltz2_affinity_checkpoint(
            args.affinity_checkpoint,
            args.structure_checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )


if __name__ == "__main__":
    main()
