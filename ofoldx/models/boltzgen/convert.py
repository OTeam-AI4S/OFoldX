r"""
Checkpoint converter: BoltzGen inverse-folding checkpoint -> OFoldX clean names.

Routes the three top-level prefixes of ``boltzgen1_ifold.ckpt`` (input_embedder / inverse_folding_encoder /
structure_module) to the matching OFoldX modules. The input embedder reuses the Boltz-2 atom converters;
the encoder/decoder GNN layers use the clean MLP/transition renames. Strict-load after conversion = parity.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch

from ...hub import add_hub_upload_arguments, upload_artifact_from_args

Rule = tuple[re.Pattern[str], str | None]
Converter = list[Rule] | Callable[[str], "str | None"]
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_STATE_KEYS = ("state_dict", "model_state_dict", "model")
_STRUCTURE_DESIGN_ARCHITECTURES = (
    "boltzgen",
    "boltzgen_diverse",
    "boltzgen_adherence",
    "boltzgen_structuretrained_small",
)
_BOLTZGEN_ARCHITECTURES = (*_STRUCTURE_DESIGN_ARCHITECTURES, "boltzgen_ifold")


def _rules(pairs: Iterable[tuple[str, str | None]]) -> list[Rule]:
    return [(re.compile(pattern), replacement) for pattern, replacement in pairs]


CONDITIONED_TRANSFORMER_LAYER_RULES: list[Rule] = _rules(
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

DISTOGRAM_RULES: list[Rule] = _rules([(r"^distogram\.", "linear.")])
BFACTOR_RULES: list[Rule] = _rules([(r"^bfactor\.", "projection.")])
_ATOM_ENCODER_RULES: list[Rule] = _rules(
    [
        (r"^embed_atom_features\.", "atom_feature_projection."),
        (r"^embed_atompair_ref_pos\.", "distance_vector_projection."),
        (r"^embed_atompair_ref_dist\.", "inverse_distance_projection."),
        (r"^embed_atompair_mask\.", "distance_mask_projection."),
        (r"^c_to_p_trans_q\.", "left_atom_projection."),
        (r"^c_to_p_trans_k\.", "right_atom_projection."),
        (r"^p_mlp\.", "feed_forward."),
        (r"^s_to_c_trans\.", "single_trunk_projection."),
        (r"^z_to_p_trans\.", "pair_trunk_projection."),
    ]
)


def _transition_index_rules() -> list[tuple[str, str | None]]:
    return [
        (r"^transitions\.(\d+)\.norm\.", r"transitions.\1.norm_input."),
        (r"^transitions\.(\d+)\.fc1\.", r"transitions.\1.gate_projection."),
        (r"^transitions\.(\d+)\.fc2\.", r"transitions.\1.value_projection."),
        (r"^transitions\.(\d+)\.fc3\.", r"transitions.\1.output_projection."),
    ]


SINGLE_CONDITIONING_RULES: list[Rule] = _rules(
    [
        (r"^norm_single\.", "single_norm."),
        (r"^single_embed\.", "single_projection."),
        (r"^fourier_embed\.proj\.", "noise_fourier.projection."),
        (r"^norm_fourier\.", "fourier_norm."),
        (r"^fourier_to_single\.", "fourier_projection."),
        *_transition_index_rules(),
    ]
)
PAIRWISE_CONDITIONING_RULES: list[Rule] = _rules(
    [(r"^dim_pairwise_init_proj\.", "init_projection."), *_transition_index_rules()]
)


def _layer_route(key: str, old: str, new: str, rules: Converter) -> str | None:
    r"""
    Route an indexed-layer key ``<old><idx>.<rest>`` to ``<new><idx>.<converted rest>``.
    """

    index, _, remainder = key[len(old) :].partition(".")
    converted = apply_converter(remainder, rules)
    return None if converted is None else f"{new}{index}.{converted}"


def convert_diffusion_conditioning(key: str) -> str | None:
    r"""
    Convert a prefix-stripped upstream ``diffusion_conditioning.*`` key to its OFoldX name.

    Args:
        key: Checkpoint key with the ``diffusion_conditioning.`` prefix already stripped.

    Returns:
        The renamed key, or ``None`` when the key is dropped.
    """

    if key.startswith("pairwise_conditioner."):
        inner = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return "pairwise_conditioner." + inner if inner is not None else None
    if key.startswith("atom_encoder."):
        inner = convert_keys(key[len("atom_encoder.") :], _ATOM_ENCODER_RULES)
        return "atom_encoder." + inner if inner is not None else None
    head = key.split(".")[0]
    if head in _DIFFUSION_CONDITIONING_BIAS_PROJECTIONS:
        return _DIFFUSION_CONDITIONING_BIAS_PROJECTIONS[head] + key[len(head) :]
    return key


# Upstream per-block pair-bias projections; house names spell out z -> pair.
_DIFFUSION_CONDITIONING_BIAS_PROJECTIONS = {
    "atom_enc_proj_z": "atom_enc_proj_pair",
    "atom_dec_proj_z": "atom_dec_proj_pair",
    "token_trans_proj_z": "token_trans_proj_pair",
}


def _convert_atom_attention_encoder(key: str) -> str | None:
    r"""
    Convert a prefix-stripped atom attention encoder key.
    """

    if key.startswith("atom_encoder.diffusion_transformer.layers."):
        return _layer_route(
            key,
            "atom_encoder.diffusion_transformer.layers.",
            "atom_transformer.transformer.layers.",
            CONDITIONED_TRANSFORMER_LAYER_RULES,
        )
    if key.startswith("atom_to_token_trans."):
        return key.replace("atom_to_token_trans.", "token_projection.", 1)
    if key.startswith("r_to_q_trans."):
        return key.replace("r_to_q_trans.", "noisy_position_projection.", 1)
    return key


def convert_atom_attention_decoder(key: str) -> str | None:
    r"""
    Convert a prefix-stripped upstream atom-attention-decoder key to its OFoldX name.

    Args:
        key: Checkpoint key with the ``atom_attention_decoder.`` prefix already stripped.

    Returns:
        The renamed key, or ``None`` when the key is dropped.
    """

    if key.startswith("atom_decoder.diffusion_transformer.layers."):
        return _layer_route(
            key,
            "atom_decoder.diffusion_transformer.layers.",
            "atom_transformer.transformer.layers.",
            CONDITIONED_TRANSFORMER_LAYER_RULES,
        )
    if key.startswith("a_to_q_trans."):
        return key.replace("a_to_q_trans.", "token_projection.", 1)
    if key.startswith("atom_feat_to_atom_pos_update."):
        return key.replace("atom_feat_to_atom_pos_update.", "position_update.", 1)
    return key


def convert_score_model(key: str) -> str | None:
    r"""
    Convert a prefix-stripped upstream ``score_model.*`` key to its OFoldX name.

    Routes the single conditioner, atom attention encoder/decoder, and token transformer to their sub-converters
    and applies the flat top-level renames to everything else.

    Args:
        key: Checkpoint key with the ``structure_module.score_model.`` prefix already stripped.

    Returns:
        The renamed key, or ``None`` when the key is dropped.
    """

    if key.startswith("single_conditioner."):
        inner = convert_keys(key[len("single_conditioner.") :], SINGLE_CONDITIONING_RULES)
        return "single_conditioner." + inner if inner is not None else None
    if key.startswith("atom_attention_encoder."):
        inner = _convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    if key.startswith("atom_attention_decoder."):
        inner = convert_atom_attention_decoder(key[len("atom_attention_decoder.") :])
        return None if inner is None else "atom_attention_decoder." + inner
    if key.startswith("token_transformer.layers."):
        return _layer_route(
            key, "token_transformer.layers.", "token_transformer.layers.", CONDITIONED_TRANSFORMER_LAYER_RULES
        )
    if key.startswith("s_to_a_linear."):
        return key.replace("s_to_a_linear.", "single_to_token.", 1)
    if key.startswith("a_norm."):
        return key.replace("a_norm.", "token_norm.", 1)
    return key


CONTACT_CONDITIONING_RULES: list[Rule] = _rules(
    [
        (r"^fourier_embedding\.proj\.", "threshold_fourier.projection."),
        (r"^encoder\.", "constraint_encoder."),
        (r"^encoding_unspecified$", "unspecified_bias"),
        (r"^encoding_unselected$", "unselected_bias"),
    ]
)


def _triangle_attention_rules(source: str, target: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source}\.layer_norm\.", f"{target}.norm_pair."),
        (rf"^{source}\.linear\.", f"{target}.bias_projection."),
        (rf"^{source}\.mha\.linear_q\.", f"{target}.attention.projection.query_projection."),
        (rf"^{source}\.mha\.linear_k\.", f"{target}.attention.projection.key_projection."),
        (rf"^{source}\.mha\.linear_v\.", f"{target}.attention.projection.value_projection."),
        (rf"^{source}\.mha\.linear_o\.", f"{target}.attention.projection.output_projection."),
        (rf"^{source}\.mha\.linear_g\.", f"{target}.attention.projection.gate_projection."),
    ]


def _transition_rules(source: str, target: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source}\.norm\.", f"{target}.norm_input."),
        (rf"^{source}\.fc1\.", f"{target}.gate_projection."),
        (rf"^{source}\.fc2\.", f"{target}.value_projection."),
        (rf"^{source}\.fc3\.", f"{target}.output_projection."),
    ]


def _triangle_mult_rename_rules(source: str, target: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source}\.norm_in\.", f"{target}.norm_input."),
        (rf"^{source}\.norm_out\.", f"{target}.norm_output."),
        (rf"^{source}\.p_out\.", f"{target}.output_projection."),
        (rf"^{source}\.g_out\.", f"{target}.gate_projection."),
    ]


PAIRFORMER_LAYER_RULES: list[Rule] = _rules(
    [
        (r"^pre_norm_s\.", "pair_attention.hidden_norm."),
        (r"^attention\.proj_q\.", "pair_attention.attention.projection.query_projection."),
        (r"^attention\.proj_k\.", "pair_attention.attention.projection.key_projection."),
        (r"^attention\.proj_v\.", "pair_attention.attention.projection.value_projection."),
        (r"^attention\.proj_g\.", "pair_attention.attention.projection.gate_projection."),
        (r"^attention\.proj_o\.", "pair_attention.attention.projection.output_projection."),
        (r"^attention\.proj_z\.0\.", "pair_attention.norm_pair."),
        (r"^attention\.proj_z\.1\.", "pair_attention.bias_projection."),
        *_triangle_mult_rename_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_mult_rename_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_s", "single_feed_forward"),
        *_transition_rules("transition_z", "pair_feed_forward"),
    ]
)
_FUSED_SPLIT = {
    "tri_mul_out.p_in.weight": ("triangle_multiplication_outgoing", "left_projection", "right_projection"),
    "tri_mul_out.g_in.weight": ("triangle_multiplication_outgoing", "left_gate_projection", "right_gate_projection"),
    "tri_mul_in.p_in.weight": ("triangle_multiplication_incoming", "left_projection", "right_projection"),
    "tri_mul_in.g_in.weight": ("triangle_multiplication_incoming", "left_gate_projection", "right_gate_projection"),
}


def _split_fused(key: str, tensor: Any) -> list[tuple[str, object]] | None:
    if key not in _FUSED_SPLIT:
        return None
    block, left, right = _FUSED_SPLIT[key]
    half = tensor.shape[0] // 2
    return [(f"{block}.{left}.weight", tensor[:half]), (f"{block}.{right}.weight", tensor[half:])]


def convert_pairformer_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one full Pairformer layer.

    Splits the fused triangle-multiplication projections into their left/right halves; otherwise applies the flat
    rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor (needed to split fused projections).

    Returns:
        List of ``(new_key, tensor)`` pairs (one, or two for split fused weights; empty when dropped).
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIRFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MINIFORMER_LAYER_RULES: list[Rule] = _rules(
    [
        (r"^pre_norm_s\.", "attention.hidden_norm."),
        (r"^attention\.proj_q\.", "attention.attention.projection.query_projection."),
        (r"^attention\.proj_k\.", "attention.attention.projection.key_projection."),
        (r"^attention\.proj_v\.", "attention.attention.projection.value_projection."),
        (r"^attention\.proj_g\.", "attention.attention.projection.gate_projection."),
        (r"^attention\.proj_o\.", "attention.attention.projection.output_projection."),
        (r"^attention\.proj_z\.0\.", "attention.norm_pair."),
        (r"^attention\.proj_z\.1\.", "attention.bias_projection."),
        *_transition_rules("transition_s", "transition_single"),
        *_transition_rules("transition_z", "transition_pair"),
    ]
)


def convert_miniformer_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one compact Miniformer layer.

    Passes ``triangular.*`` keys through unchanged (the compact triangular update keeps upstream names); otherwise
    applies the Miniformer rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("triangular."):
        return [(key, tensor)]
    new_key = convert_keys(key, MINIFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


PAIR_ONLY_LAYER_RULES: list[Rule] = _rules(
    [
        *_triangle_mult_rename_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_mult_rename_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_z", "feed_forward"),
    ]
)


def convert_pair_only_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one pair-only triangle layer.

    Splits the fused triangle-multiplication projections; otherwise applies the pair-only rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor (needed to split fused projections).

    Returns:
        List of ``(new_key, tensor)`` pairs (one, or two for split fused weights; empty when dropped).
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


CONFIDENCE_FLAT_RULES: list[Rule] = _rules(
    [
        (r"^dist_bin_pairwise_embed\.", "distance_embedding."),
        (r"^s_inputs_norm\.", "input_single_norm."),
        (r"^s_norm\.", "single_norm."),
        (r"^z_norm\.", "norm_pair."),
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
CONFIDENCE_HEADS_RULES: list[Rule] = _rules(
    [
        (r"^to_pae_intra_logits\.", "pae_intra_projection."),
        (r"^to_pae_inter_logits\.", "pae_inter_projection."),
        (r"^to_pde_intra_logits\.", "pde_intra_projection."),
        (r"^to_pde_inter_logits\.", "pde_inter_projection."),
        (r"^to_plddt_logits\.", "plddt_projection."),
        (r"^to_resolved_logits\.", "resolved_projection."),
    ]
)


def convert_confidence_head(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for a prefix-stripped upstream ``confidence_module.*`` key onto the ported
    ``ConfidenceHead``. Contact conditioning, the Pairformer stack, and the projection heads route to their clean
    names; the remaining flat projections use ``CONFIDENCE_FLAT_RULES``.

    Args:
        key: Checkpoint key with the ``confidence_module.`` prefix already stripped.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in convert_pairformer_layer(remainder, tensor)]
    if key.startswith("confidence_heads."):
        new = convert_keys(key[len("confidence_heads.") :], CONFIDENCE_HEADS_RULES)
        return [("confidence_heads." + new, tensor)] if new is not None else []
    new = convert_keys(key, CONFIDENCE_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


AFFINITY_FLAT_RULES: list[Rule] = _rules(
    [
        (r"^dist_bin_pairwise_embed\.", "dist_bin_pairwise_embed."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^z_norm\.", "pair_norm."),
        (r"^z_linear\.", "pair_linear."),
        (r"^boundaries$", "boundaries"),
    ]
)
AFFINITY_HEADS_RULES: list[Rule] = _rules(
    [
        (r"^affinity_out_mlp\.", "affinity_out_mlp."),
        (r"^to_affinity_pred_value\.", "to_affinity_pred_value."),
        (r"^to_affinity_pred_score\.", "to_affinity_pred_score."),
        (r"^to_affinity_logits_binary\.", "to_affinity_logits_binary."),
    ]
)


def convert_affinity_head(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for a prefix-stripped upstream ``affinity_module*.*`` key onto the ported
    ``AffinityHead``. The pairwise conditioner, the pair-only Pairformer stack, and the scalar heads route to their
    clean names.

    Args:
        key: Checkpoint key with the ``affinity_module*.`` prefix already stripped.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("pairwise_conditioner."):
        new = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return [("pairwise_conditioner." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in convert_pair_only_layer(remainder, tensor)]
    if key.startswith("affinity_heads."):
        new = convert_keys(key[len("affinity_heads.") :], AFFINITY_HEADS_RULES)
        return [("affinity_heads." + new, tensor)] if new is not None else []
    new = convert_keys(key, AFFINITY_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


MINIFORMER_PAIR_ONLY_LAYER_RULES: list[Rule] = _rules([*_transition_rules("transition_z", "transition_pair")])


def convert_miniformer_pair_only_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one compact pair-only Miniformer layer.

    Passes ``triangular.*`` keys through unchanged; otherwise applies the compact pair-only rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("triangular."):
        return [(key, tensor)]
    new_key = convert_keys(key, MINIFORMER_PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


_PWA_INNER = _rules(
    [
        (r"^norm_m\.", "norm_msa."),
        (r"^norm_z\.", "norm_pair."),
        (r"^proj_m\.", "value_projection."),
        (r"^proj_g\.", "gate_projection."),
        (r"^proj_z\.", "pair_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)
_SWIGLU_INNER = _rules(
    [
        (r"^norm\.", "norm_input."),
        (r"^fc1\.", "gate_projection."),
        (r"^fc2\.", "value_projection."),
        (r"^fc3\.", "output_projection."),
    ]
)
_OPM_INNER = _rules(
    [
        (r"^norm\.", "norm_msa."),
        (r"^proj_a\.", "left_projection."),
        (r"^proj_b\.", "right_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)


def convert_msa_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one MSA layer (Pairformer pair update).

    Routes the pair-weighted-averaging, MSA transition, outer-product-mean, and pair-only Pairformer sub-blocks to
    their clean names.

    Args:
        key: Checkpoint key relative to the MSA-layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA sub-block.
    """

    if key.startswith("pair_weighted_averaging."):
        return [(cast(str, reprefix(key, "pair_weighted_averaging.", "pair_weighted_averaging.", _PWA_INNER)), tensor)]
    if key.startswith("msa_transition."):
        return [(cast(str, reprefix(key, "msa_transition.", "msa_transition.", _SWIGLU_INNER)), tensor)]
    if key.startswith("outer_product_mean."):
        return [(cast(str, reprefix(key, "outer_product_mean.", "outer_product_mean.", _OPM_INNER)), tensor)]
    if key.startswith("pairformer_layer."):
        inner = convert_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in inner]
    raise KeyError(f"unmapped MSA layer key: {key}")


def convert_miniformer_msa_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one MSA layer whose pair update uses MiniformerNoSeq blocks.

    Same routing as [`convert_msa_layer`][ofoldx.models.boltzgen.convert.convert_msa_layer] but the pair-only
    sub-block uses the compact Miniformer converter.

    Args:
        key: Checkpoint key relative to the MSA-layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA sub-block.
    """

    if key.startswith("pair_weighted_averaging."):
        return [(cast(str, reprefix(key, "pair_weighted_averaging.", "pair_weighted_averaging.", _PWA_INNER)), tensor)]
    if key.startswith("msa_transition."):
        return [(cast(str, reprefix(key, "msa_transition.", "msa_transition.", _SWIGLU_INNER)), tensor)]
    if key.startswith("outer_product_mean."):
        return [(cast(str, reprefix(key, "outer_product_mean.", "outer_product_mean.", _OPM_INNER)), tensor)]
    if key.startswith("pairformer_layer."):
        inner = convert_miniformer_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in inner]
    raise KeyError(f"unmapped Miniformer MSA layer key: {key}")


def convert_msa_module(key: str, tensor: object, *, miniformer_blocks: bool = False) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for the whole MSA module.

    Renames the token / MSA input projections and dispatches per-layer keys to the Pairformer or Miniformer MSA
    layer converter.

    Args:
        key: Checkpoint key with the ``msa_module.`` prefix already stripped.
        tensor: The associated weight tensor.
        miniformer_blocks: Whether the layers use MiniformerNoSeq pair updates.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA-module sub-block.
    """

    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        converter = convert_miniformer_msa_layer if miniformer_blocks else convert_msa_layer
        return [(f"layers.{index}.{nk}", nv) for nk, nv in converter(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


def convert_keys(key: str, rules: list[Rule]) -> str | None:
    r"""
    Apply the first matching rename/drop rule to ``key``.

    Args:
        key: Checkpoint key to convert.
        rules: Ordered ``(pattern, replacement)`` rules; a ``None`` replacement drops the key.

    Returns:
        The renamed key, ``None`` if a matching rule drops it, or ``key`` unchanged when no rule matches.

    Examples:
        >>> import re
        >>> rules = [(re.compile(r"^old\."), "new."), (re.compile(r"^drop\."), None)]
        >>> convert_keys("old.weight", rules)
        'new.weight'
        >>> convert_keys("drop.weight", rules) is None
        True
        >>> convert_keys("keep.weight", rules)
        'keep.weight'
    """

    for pattern, replacement in rules:
        if pattern.search(key):
            return None if replacement is None else pattern.sub(replacement, key)
    return key


def apply_converter(key: str, converter: Converter) -> str | None:
    r"""
    Dispatch ``key`` through a ``Converter``: call it if a callable, else apply it as a ``convert_keys`` rule list.

    Args:
        key: Checkpoint key to convert.
        converter: Either a callable ``str -> str | None`` or a list of ``convert_keys`` rules.

    Returns:
        The renamed key, ``None`` to drop the key, or ``key`` unchanged when no rule matches.
    """

    if callable(converter):
        return converter(key)
    return convert_keys(key, converter)


def reprefix(key: str, old_prefix: str, new_prefix: str, sub: Converter) -> str | None:
    r"""
    Swap ``old_prefix`` for ``new_prefix`` and convert the remainder through ``sub``.

    Args:
        key: Checkpoint key to convert.
        old_prefix: Prefix that ``key`` must start with.
        new_prefix: Prefix to prepend to the converted remainder.
        sub: Converter applied to the remainder after the prefix.

    Returns:
        ``None`` if ``key`` lacks ``old_prefix`` or ``sub`` drops the remainder; else ``new_prefix`` + converted.
    """

    if not key.startswith(old_prefix):
        return None
    inner = apply_converter(key[len(old_prefix) :], sub)
    return None if inner is None else new_prefix + inner


def convert_subset_tensors(
    subset: dict[str, object], converter: Callable[[str, object], list[tuple[str, object]]]
) -> dict[str, object]:
    r"""
    Convert a prefix-stripped tensor subset with a tensor-aware converter.

    Args:
        subset: Mapping of prefix-stripped checkpoint keys to tensors.
        converter: Tensor-aware converter mapping one ``(key, tensor)`` to a list of output pairs.

    Returns:
        New mapping of converted keys to tensors (fused-weight splits expanded, dropped keys omitted).
    """

    out: dict[str, object] = {}
    for key, value in subset.items():
        for new_key, new_value in converter(key, value):
            out[new_key] = new_value
    return out


_ADAPTIVE_NORM_MODULES = ("conditioned_norm", "key_value_norm")


def _adaptive_norm_base(key: str, projection: str, param: str) -> str | None:
    suffix = f".{projection}.{param}"
    if not key.endswith(suffix):
        return None
    base = key[: -len(suffix)]
    module = base.rsplit(".", 1)[-1]
    if module not in _ADAPTIVE_NORM_MODULES:
        return None
    return base


def pack_adaptive_layer_norm_state(state_dict: dict[str, object]) -> dict[str, object]:
    r"""
    Keep converted AdaptiveLayerNorm gate/shift projections in the explicit Boltz layout.

    Args:
        state_dict: Already-renamed state dict with split gate/shift projection weights.

    Returns:
        A shallow copy of ``state_dict``.
    """

    return dict(state_dict)


_TRUNK_FLAT = {
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


def convert_trunk_key(
    key: str,
    tensor: object,
    *,
    miniformer_pairformer: bool = False,
    miniformer_msa: bool = False,
) -> list[tuple[str, object]]:
    r"""
    Convert one structure-design trunk checkpoint key onto the OFoldX trunk modules.

    Dispatches the shared backbone prefixes (contact conditioning, MSA module, Pairformer stack, distogram,
    diffusion conditioning, score model, B-factor head) to their sub-converters and applies the flat trunk renames
    to the top-level projections.

    Args:
        key: Checkpoint key relative to the trunk root.
        tensor: The associated weight tensor.
        miniformer_pairformer: Whether the Pairformer stack uses Miniformer layers.
        miniformer_msa: Whether the MSA module uses Miniformer pair updates.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is not part of the trunk).
    """

    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("msa_module."):
        inner = convert_msa_module(key[len("msa_module.") :], tensor, miniformer_blocks=miniformer_msa)
        return [(f"msa_module.{new_key}", new_value) for new_key, new_value in inner]
    if key.startswith(("pairformer_stack.layers.", "pairformer_module.layers.")):
        prefix = (
            "pairformer_stack.layers." if key.startswith("pairformer_stack.layers.") else "pairformer_module.layers."
        )
        index, _, remainder = key[len(prefix) :].partition(".")
        converter = convert_miniformer_layer if miniformer_pairformer else convert_pairformer_layer
        return [
            (f"pairformer_stack.blocks.{index}.{new_key}", new_value)
            for new_key, new_value in converter(remainder, tensor)
        ]
    if key.startswith("distogram_module."):
        new = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("diffusion_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = convert_score_model(key[len("structure_module.score_model.") :])
        return [("structure_module.score_model." + new, tensor)] if new is not None else []
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in _TRUNK_FLAT:
        return [(_TRUNK_FLAT[key], tensor)]
    return []


INPUT_EMBEDDER_CONDITIONING = {
    "atom_enc_proj_z": "atom_pair_bias_projection",
    "res_type_encoding": "residue_type_projection",
    "msa_profile_encoding": "msa_profile_projection",
    "method_conditioning_init": "method_embedding",
    "modified_conditioning_init": "modified_embedding",
    "cyclic_conditioning_init": "cyclic_projection",
    "mol_type_conditioning_init": "molecule_type_embedding",
    "binding_specification_conditioning_init": "binding_specification_embedding",
    "design_mask_conditioning_init": "design_mask_embedding",
    "ss_specification_init": "ss_specification_embedding",
}


def convert_input_embedder(key: str) -> str | None:
    r"""
    Convert one ``input_embedder.*`` key onto its AtomConditionedInputEmbedder name.

    Args:
        key: Checkpoint key with the ``input_embedder.`` prefix already stripped.

    Returns:
        The renamed key, or ``None`` to drop the tied duplicate.
    """
    if key.startswith("atom_encoder."):
        inner = convert_keys(key[len("atom_encoder.") :], _ATOM_ENCODER_RULES)
        return None if inner is None else "atom_encoder." + inner
    if key.startswith("atom_attention_encoder."):
        inner = _convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    head = key.split(".")[0]
    if head in INPUT_EMBEDDER_CONDITIONING:
        return INPUT_EMBEDDER_CONDITIONING[head] + key[len(head) :]
    return key


GNN_LAYER_RENAME = {
    "attn_weight_mlp": "attention_weight_mlp",
    "attn_value_mlp": "attention_value_mlp",
    "attn_output_linear": "attention_output",
    "attn_FFN": "node_transition",
    "edge_FFN": "edge_transition",
}


def _convert_gnn_layer(suffix: str) -> str:
    head, rest = suffix.split(".", 1)
    return f"{GNN_LAYER_RENAME.get(head, head)}.{rest}"


ENCODER_TOP = {
    "linear_token_to_node": "node_input_projection",
    "linear_token_to_pair": "edge_input_projection",
    "distance_gaussian_smearing": "distance_smearing",
}

DECODER_TOP = {
    "seq_to_s": "residue_type_projection",
    "predictor": "residue_type_head",
}


def convert_encoder(key: str) -> str:
    r"""
    Convert one ``inverse_folding_encoder.*`` key onto its InverseFoldingEncoder name.

    Args:
        key: Checkpoint key with the ``inverse_folding_encoder.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("encoder_layers."):
        index, suffix = key[len("encoder_layers.") :].split(".", 1)
        return f"layers.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return ENCODER_TOP.get(head, head) + key[len(head) :]


def convert_decoder(key: str) -> str:
    r"""
    Convert one ``structure_module.*`` key onto its InverseFoldingDecoder name.

    Args:
        key: Checkpoint key with the ``structure_module.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("decoder_layers."):
        index, suffix = key[len("decoder_layers.") :].split(".", 1)
        return f"layers.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return DECODER_TOP.get(head, head) + key[len(head) :]


TOKEN_DISTANCE_TOP = {
    "z_norm": "norm_pair",
    "v_norm": "hidden_norm",
    "z_proj": "pair_projection",
    "a_proj": "distance_feature_projection",
    "u_proj": "output_projection",
}

TOKEN_DISTANCE_ENCODER = {
    "distance_gaussian_smearing": "distance_smearing",
    "distance_token_bias_trans.fc1": "distance_token_bias_trans.gate_projection",
    "distance_token_bias_trans.fc2": "distance_token_bias_trans.value_projection",
    "distance_token_bias_trans.fc3": "distance_token_bias_trans.output_projection",
}


def convert_token_distance(state: dict) -> dict:
    r"""
    Convert a ``token_distance_module.*`` checkpoint subset onto the TokenDistanceModule (clean names).

    The pair-only pairformer layers are split tensor-aware; the distance encoder and top-level projections use the
    local rename maps.

    Args:
        state: Mapping of ``token_distance_module``-relative keys to tensors.

    Returns:
        New mapping of converted keys to tensors.
    """
    out: dict = {}
    for key, value in state.items():
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            for new_suffix, new_value in convert_pair_only_layer(suffix, value):
                out[f"pairformer.{index}.{new_suffix}"] = new_value
        elif key.startswith("token_distance_encoder."):
            inner = key[len("token_distance_encoder.") :]
            for ref, clean in TOKEN_DISTANCE_ENCODER.items():
                if inner.startswith(ref):
                    inner = clean + inner[len(ref) :]
                    break
            out["distance_encoder." + inner] = value
        else:
            head = key.split(".")[0]
            out[TOKEN_DISTANCE_TOP.get(head, head) + key[len(head) :]] = value
    return out


def convert_template_v1(state: dict, *, miniformer_blocks: bool = False) -> dict:
    r"""
    Convert a ``template_module.*`` (BoltzGen v1) checkpoint subset onto the TemplateModule (clean names).

    The pair-only pairformer layers are split tensor-aware (Miniformer variant when ``miniformer_blocks`` is set);
    the top-level projections use the local rename map.

    Args:
        state: Mapping of ``template_module``-relative keys to tensors.
        miniformer_blocks: Whether the template pairformer uses compact Miniformer pair updates.

    Returns:
        New mapping of converted keys to tensors.
    """
    rename = {**TOKEN_DISTANCE_TOP, "a_proj": "template_feature_projection"}
    out: dict = {}
    for key, value in state.items():
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            converter = convert_miniformer_pair_only_layer if miniformer_blocks else convert_pair_only_layer
            for new_suffix, new_value in converter(suffix, value):
                out[f"pairformer.{index}.{new_suffix}"] = new_value
        else:
            head = key.split(".")[0]
            out[rename.get(head, head) + key[len(head) :]] = value
    return out


def convert_inverse_folding(state: dict) -> dict:
    r"""
    Convert a whole inverse-folding checkpoint state dict onto a BoltzGenInverseFoldDesigner state dict.

    Routes the three top-level prefixes (``input_embedder`` / ``inverse_folding_encoder`` / ``structure_module``)
    to their sub-converters and packs the adaptive-layer-norm projections into the fused OOps parameters. A clean
    strict-load after conversion is the parity check.

    Args:
        state: Upstream inverse-folding checkpoint state dict (e.g. ``boltzgen1_ifold.ckpt``).

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> state = torch.load("boltzgen1_ifold.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_inverse_folding(state)  # doctest: +SKIP
    """
    out: dict = {}
    for key, value in state.items():
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            new_key = None if inner is None else "input_embedder." + inner
        elif key.startswith("inverse_folding_encoder."):
            new_key = "encoder." + convert_encoder(key[len("inverse_folding_encoder.") :])
        elif key.startswith("structure_module."):
            new_key = "decoder." + convert_decoder(key[len("structure_module.") :])
        else:
            new_key = key
        if new_key is not None:
            out[new_key] = value
    return pack_adaptive_layer_norm_state(out)


def convert_structure_design(state: dict) -> dict:
    r"""
    Convert a whole BoltzGen structure-design checkpoint (e.g. ``boltzgen1_diverse``) onto OFoldX clean names.

    The shared backbone prefixes route through this file's ``convert_trunk_key``. The input embedder carries
    design conditioning, and ``token_distance_module`` plus the v1 template module use local sub-converters.
    Tied ``output_projection.0`` atom-layer duplicates are dropped. The Miniformer variants are auto-detected from
    the presence of ``triangular.*`` keys. A clean strict-load after conversion is the parity check.

    Args:
        state: Upstream structure-design checkpoint state dict.

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> state = torch.load("boltzgen1_diverse.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_structure_design(state)  # doctest: +SKIP
    """

    def _sub(prefix: str) -> dict:
        return {key[len(prefix) + 1 :]: value for key, value in state.items() if key.startswith(prefix + ".")}

    miniformer_pairformer = any(key.startswith("pairformer_module.layers.") and ".triangular." in key for key in state)
    miniformer_msa = any(
        key.startswith("msa_module.layers.") and ".pairformer_layer.triangular." in key for key in state
    )
    miniformer_template = any(
        key.startswith("template_module.pairformer.layers.") and ".triangular." in key for key in state
    )

    # Modules the boltz2 trunk converter doesn't cover (or covers as v2): route to the BoltzGen converters.
    out = {f"token_distance.{k}": v for k, v in convert_token_distance(_sub("token_distance_module")).items()}
    out.update(
        {
            f"template_module.{k}": v
            for k, v in convert_template_v1(_sub("template_module"), miniformer_blocks=miniformer_template).items()
        }
    )
    for key, value in state.items():
        if key.startswith(("token_distance_module.", "template_module.")):
            continue
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            if inner is not None:
                out["input_embedder." + inner] = value
        else:
            # BoltzGen wraps the diffusion token transformer in an outer ModuleList (token_layers=1):
            # token_transformer_layers.0.layers.<i> == boltz2's flat token_transformer.layers.<i>.
            key = key.replace(
                "structure_module.score_model.token_transformer_layers.0.layers.",
                "structure_module.score_model.token_transformer.layers.",
                1,
            )
            for new_key, new_value in convert_trunk_key(
                key,
                value,
                miniformer_pairformer=miniformer_pairformer,
                miniformer_msa=miniformer_msa,
            ):
                out[new_key] = new_value
    return pack_adaptive_layer_norm_state(out)


def convert_boltzgen_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    architecture: str | None = None,
    kind: str = "auto",
    state_key: str = "auto",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    r"""
    Convert a BoltzGen checkpoint into a pretrained OFoldX artifact directory.

    Args:
        checkpoint_path: Path to a BoltzGen ``.ckpt`` checkpoint.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        architecture: Artifact architecture key. Defaults to ``"boltzgen_diverse"`` for structure design and
            ``"boltzgen_ifold"`` for inverse folding.
        kind: ``"auto"``, ``"structure-design"``, or ``"ifold"``.
        state_key: Checkpoint state-dict key: ``"auto"``, ``"state_dict"``, ``"model_state_dict"``,
            ``"model"``, or ``"root"``.
        dtype: Optional floating-point dtype for persisted weights.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_boltzgen_checkpoint("boltzgen1_diverse.ckpt", "boltzgen_diverse")  # doctest: +SKIP
        >>> convert_boltzgen_checkpoint("boltzgen1_ifold.ckpt", "boltzgen_ifold", kind="ifold")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = _load_boltz_lightning_checkpoint(checkpoint_path)
    raw_state, selected_state_key = _select_state_dict(checkpoint, state_key=state_key)
    model, converter, resolved_kind = _build_boltzgen_target(architecture=architecture, kind=kind)
    converted = _cast_floating_tensors(_tensor_state_dict(converter(dict(raw_state))), dtype)
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_state_dict(converted, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": model.config.architecture,
        "kind": resolved_kind,
        "state_key": selected_state_key,
        "tensor_count": len(converted),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(converted),
    }


def _load_boltz_lightning_checkpoint(checkpoint_path: Path) -> Mapping[str, Any]:
    r"""
    Load BoltzGen Lightning checkpoints without requiring the original ``boltzgen`` package.

    Released checkpoints pickle small validator objects alongside the tensor ``state_dict``. OFoldX only needs the
    tensors, so when the upstream package is absent this loader registers minimal validator stubs and retries.

    Examples:
        >>> checkpoint = _load_boltz_lightning_checkpoint(Path("boltzgen1_ifold.ckpt"))  # doctest: +SKIP
        >>> "state_dict" in checkpoint  # doctest: +SKIP
        True
    """

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as error:
        if error.name != "boltzgen":
            raise
        checkpoint = _load_with_boltzgen_unpickle_stubs(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    return checkpoint


def _load_with_boltzgen_unpickle_stubs(checkpoint_path: Path) -> object:
    r"""Load a checkpoint with temporary ``boltzgen`` validator stubs."""

    modules = _boltzgen_unpickle_stub_modules()
    previous: dict[str, types.ModuleType | None] = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def _boltzgen_unpickle_stub_modules() -> dict[str, types.ModuleType]:
    r"""Build minimal modules/classes needed to unpickle released BoltzGen checkpoint metadata."""

    class DesignValidator:
        pass

    class RCSBValidator:
        pass

    DesignValidator.__module__ = "boltzgen.model.validation.design"
    RCSBValidator.__module__ = "boltzgen.model.validation.rcsb"
    modules = {
        "boltzgen": types.ModuleType("boltzgen"),
        "boltzgen.model": types.ModuleType("boltzgen.model"),
        "boltzgen.model.validation": types.ModuleType("boltzgen.model.validation"),
        "boltzgen.model.validation.design": types.ModuleType("boltzgen.model.validation.design"),
        "boltzgen.model.validation.rcsb": types.ModuleType("boltzgen.model.validation.rcsb"),
    }
    design_module = cast(Any, modules["boltzgen.model.validation.design"])
    rcsb_module = cast(Any, modules["boltzgen.model.validation.rcsb"])
    root_module = cast(Any, modules["boltzgen"])
    model_module = cast(Any, modules["boltzgen.model"])
    validation_module = cast(Any, modules["boltzgen.model.validation"])
    design_module.DesignValidator = DesignValidator
    rcsb_module.RCSBValidator = RCSBValidator
    root_module.model = modules["boltzgen.model"]
    model_module.validation = modules["boltzgen.model.validation"]
    validation_module.design = modules["boltzgen.model.validation.design"]
    validation_module.rcsb = modules["boltzgen.model.validation.rcsb"]
    return modules


def _build_boltzgen_target(
    *,
    architecture: str | None,
    kind: str,
) -> tuple[Any, Callable[[dict], dict], str]:
    r"""Instantiate the target BoltzGen model and select its checkpoint converter."""

    from .config import (
        BoltzGenAdherenceConfig,
        BoltzGenConfig,
        BoltzGenDiverseConfig,
        BoltzGenIFoldConfig,
        BoltzGenStructureTrainedSmallConfig,
    )
    from .model import BoltzGenDesigner, BoltzGenInverseFoldDesigner

    if kind not in {"auto", "structure-design", "ifold"}:
        raise ValueError("kind must be 'auto', 'structure-design', or 'ifold'")
    arch = None if architecture is None else architecture.lower()
    if arch is None:
        arch = "boltzgen_ifold" if kind == "ifold" else "boltzgen_diverse"
    if arch not in _BOLTZGEN_ARCHITECTURES:
        options = ", ".join(_BOLTZGEN_ARCHITECTURES)
        raise ValueError(f"unsupported BoltzGen architecture {architecture!r}; expected one of: {options}")

    resolved_kind = "ifold" if kind == "auto" and arch == "boltzgen_ifold" else kind
    if resolved_kind == "auto":
        resolved_kind = "structure-design"
    if resolved_kind == "ifold":
        if arch != "boltzgen_ifold":
            raise ValueError("--kind ifold requires --architecture boltzgen_ifold")
        return BoltzGenInverseFoldDesigner(BoltzGenIFoldConfig()), convert_inverse_folding, "ifold"

    if arch == "boltzgen_ifold":
        raise ValueError("--architecture boltzgen_ifold requires --kind ifold")
    config_classes = {
        "boltzgen": BoltzGenConfig,
        "boltzgen_diverse": BoltzGenDiverseConfig,
        "boltzgen_adherence": BoltzGenAdherenceConfig,
        "boltzgen_structuretrained_small": BoltzGenStructureTrainedSmallConfig,
    }
    return BoltzGenDesigner(config_classes[arch]()), convert_structure_design, "structure-design"


def _select_state_dict(checkpoint: Mapping[str, Any], *, state_key: str) -> tuple[Mapping[str, Any], str]:
    r"""Locate the tensor state dict within a checkpoint mapping."""

    if state_key == "auto":
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value, key
        if _looks_like_state_dict(checkpoint):
            return checkpoint, "root"
    if state_key == "root" and _looks_like_state_dict(checkpoint):
        return checkpoint, "root"
    value = checkpoint.get(state_key)
    if isinstance(value, Mapping):
        return value, state_key
    raise TypeError(f"checkpoint does not contain tensor state dict {state_key!r}")


def _looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    r"""Return whether ``value`` looks like a raw tensor state dict."""

    return bool(value) and any(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items())


def _tensor_state_dict(state_dict: Mapping[str, object]) -> dict[str, torch.Tensor]:
    r"""Keep only tensor-valued entries from ``state_dict``."""

    return {key: value for key, value in state_dict.items() if isinstance(key, str) and torch.is_tensor(value)}


def _cast_floating_tensors(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    r"""Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def _dtype_counts(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    r"""Return a sorted count of tensors per dtype name in ``state_dict``."""

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for BoltzGen checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a BoltzGen .ckpt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--architecture", choices=_BOLTZGEN_ARCHITECTURES, default=None)
    parser.add_argument("--kind", choices=("auto", "structure-design", "ifold"), default="auto")
    parser.add_argument(
        "--state-key", default="auto", help="Checkpoint state key: auto, state_dict, model_state_dict, model, or root"
    )
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for BoltzGen checkpoint conversion.

    Examples:
        >>> main(["boltzgen1_diverse.ckpt", "out/boltzgen_diverse"])  # doctest: +SKIP
        >>> main(["boltzgen1_ifold.ckpt", "out/boltzgen_ifold", "--kind", "ifold"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_boltzgen_checkpoint(
        args.checkpoint,
        args.output_dir,
        architecture=args.architecture,
        kind=args.kind,
        state_key=args.state_key,
        dtype=dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted BoltzGen checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
