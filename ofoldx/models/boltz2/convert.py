r"""
Boltz-2 checkpoint → OFoldX key converter.

OFoldX modules use clean, idiomatic parameter names; the public ``boltz2_conf.ckpt`` uses Boltz's
own names. This module remaps Boltz keys onto OFoldX names (and drops weight-tied duplicates) so a
converted ``state_dict`` ``strict``-loads into the OFoldX modules. Rules are grouped per subsystem
and grow as the port advances; [`convert_keys`][ofoldx.models.boltz2.convert.convert_keys] applies an ordered
rule list to one key.

A rule is ``(compiled_regex, replacement_or_None)``: the first matching rule rewrites the key via
``re.sub`` (or drops it when the replacement is ``None``). Order matters — list the most specific rules
first.
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
# A converter maps one Boltz key to its OFoldX name, or to ``None`` to drop it. It is either an
# ordered rule list or a callable (used for nested-prefix routing).
Converter = list[Rule] | Callable[[str], "str | None"]
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_STATE_KEYS = ("state_dict", "model_state_dict", "model")


def _rules(pairs: Iterable[tuple[str, str | None]]) -> list[Rule]:
    r"""
    Compile ``(pattern, replacement)`` string pairs into a rule list.

    Args:
        pairs: Iterable of ``(regex_pattern, replacement_or_None)`` string pairs.

    Returns:
        Rule list with each pattern compiled to a [`re.Pattern`][re.Pattern].
    """

    return [(re.compile(pattern), replacement) for pattern, replacement in pairs]


# A single Boltz ``ConditionedTransformerLayer`` (AdaLN + gated pair-bias attention + triple-product
# transition). Applied to keys already stripped to the per-layer suffix (e.g. ``adaln.s_scale.weight``).
# Transition rules come before the layer-level rules so ``transition.output_projection.0`` is handled
# before the layer-level ``output_projection.0`` drop.
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
        # output_projection.0 is the tied duplicate of output_projection_linear — drop it.
        (r"^output_projection\.0\.", None),
    ]
)

# Boltz DistogramModule.distogram (a single Linear) -> OFoldX DistogramHead.linear.
DISTOGRAM_RULES: list[Rule] = _rules([(r"^distogram\.", "linear.")])

# Boltz BFactorModule.bfactor (a single Linear) -> OFoldX BFactorHead.projection.
BFACTOR_RULES: list[Rule] = _rules([(r"^bfactor\.", "projection.")])


# Boltz AtomEncoder -> OFoldX AtomEncoder (clean names). Applied to the per-encoder suffix. The
# s_to_c_trans / z_to_p_trans rules only fire for the diffusion (structure_prediction=True) encoder.
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
    r"""
    Rename rules for a Boltz Transition inside an indexed ``transitions.N.*`` ModuleList.

    Returns:
        String ``(pattern, replacement)`` pairs mapping each indexed transition submodule to its OFoldX name.
    """

    # Boltz Transition inside an indexed ModuleList (transitions.N.*) -> transition module.
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
    Route an indexed-layer key ``<old><idx>.<rest>`` -> ``<new><idx>.<converted rest>``.

    Args:
        key: The full Boltz key beginning with ``old``.
        old: The Boltz prefix preceding the ``<idx>.`` segment.
        new: The OFoldX prefix to substitute for ``old``.
        rules: Converter applied to the per-layer remainder after the index.

    Returns:
        The rerouted OFoldX key, or ``None`` when the remainder converter drops the key.
    """

    index, _, remainder = key[len(old) :].partition(".")
    converted = apply_converter(remainder, rules)
    return None if converted is None else f"{new}{index}.{converted}"


def convert_diffusion_conditioning(key: str) -> str | None:
    r"""
    Boltz ``diffusion_conditioning.*`` (prefix-stripped) -> OFoldX DiffusionConditioning.

    Args:
        key: A ``diffusion_conditioning``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
    """

    if key.startswith("pairwise_conditioner."):
        inner = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return "pairwise_conditioner." + inner if inner is not None else None
    if key.startswith("atom_encoder."):
        inner = convert_keys(key[len("atom_encoder.") :], _ATOM_ENCODER_RULES)
        return "atom_encoder." + inner if inner is not None else None
    # Boltz names the per-block bias projections *_proj_z (pair track); OFoldX spells "pair". Each is a
    # Sequential(LayerNorm, Linear), so only the attribute prefix is renamed.
    for boltz, house in (
        ("atom_enc_proj_z.", "atom_enc_proj_pair."),
        ("atom_dec_proj_z.", "atom_dec_proj_pair."),
        ("token_trans_proj_z.", "token_trans_proj_pair."),
    ):
        if key.startswith(boltz):
            return house + key[len(boltz) :]
    return key


def _convert_atom_attention_encoder(key: str) -> str | None:
    r"""
    Convert a score-model ``atom_attention_encoder``-relative Boltz key to its OFoldX name.

    Args:
        key: An ``atom_attention_encoder``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
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


def _convert_atom_attention_decoder(key: str) -> str | None:
    r"""
    Convert a score-model ``atom_attention_decoder``-relative Boltz key to its OFoldX name.

    Args:
        key: An ``atom_attention_decoder``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
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
    Boltz ``structure_module.score_model.*`` (prefix-stripped) -> OFoldX DiffusionScoreModel.

    Args:
        key: A ``structure_module.score_model``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
    """

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
            key, "token_transformer.layers.", "token_transformer.layers.", CONDITIONED_TRANSFORMER_LAYER_RULES
        )
    if key.startswith("s_to_a_linear."):
        return key.replace("s_to_a_linear.", "single_to_token.", 1)
    if key.startswith("a_norm."):
        return key.replace("a_norm.", "token_norm.", 1)
    return key


# Boltz ContactConditioning -> OFoldX ContactConditioning (clean names).
CONTACT_CONDITIONING_RULES: list[Rule] = _rules(
    [
        (r"^fourier_embedding\.proj\.", "threshold_fourier.projection."),
        (r"^encoder\.", "constraint_encoder."),
        (r"^encoding_unspecified$", "unspecified_bias"),
        (r"^encoding_unselected$", "unselected_bias"),
    ]
)


def _triangle_attention_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    r"""
    Rename rules for one Boltz triangle-attention block.

    Args:
        boltz: The Boltz submodule prefix (e.g. ``"tri_att_start"``).
        clean: The OFoldX submodule prefix to substitute.

    Returns:
        String ``(pattern, replacement)`` pairs mapping the norm / bias / q,k,v,o,g projections.
    """

    return [
        (rf"^{boltz}\.layer_norm\.", f"{clean}.norm_pair."),
        (rf"^{boltz}\.linear\.", f"{clean}.bias_projection."),
        (rf"^{boltz}\.mha\.linear_q\.", f"{clean}.attention.projection.query_projection."),
        (rf"^{boltz}\.mha\.linear_k\.", f"{clean}.attention.projection.key_projection."),
        (rf"^{boltz}\.mha\.linear_v\.", f"{clean}.attention.projection.value_projection."),
        (rf"^{boltz}\.mha\.linear_o\.", f"{clean}.attention.projection.output_projection."),
        (rf"^{boltz}\.mha\.linear_g\.", f"{clean}.attention.projection.gate_projection."),
    ]


def _transition_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    r"""
    Rename rules for one Boltz Transition block.

    Args:
        boltz: The Boltz submodule prefix (e.g. ``"transition_z"``).
        clean: The OFoldX submodule prefix to substitute.

    Returns:
        String ``(pattern, replacement)`` pairs mapping ``norm``/``fc1``/``fc2``/``fc3`` to their OFoldX names.
    """

    # Boltz Transition (norm/fc1/fc2/fc3) -> OFoldX Transition.
    return [
        (rf"^{boltz}\.norm\.", f"{clean}.norm_input."),
        (rf"^{boltz}\.fc1\.", f"{clean}.gate_projection."),
        (rf"^{boltz}\.fc2\.", f"{clean}.value_projection."),
        (rf"^{boltz}\.fc3\.", f"{clean}.output_projection."),
    ]


def _triangle_mult_rename_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    r"""
    Rename rules for the non-fused keys of one Boltz triangle-multiplication block.

    The fused ``p_in``/``g_in`` projections are split separately (tensor-aware); only the remaining
    rename-only keys are covered here.

    Args:
        boltz: The Boltz submodule prefix (e.g. ``"tri_mul_out"``).
        clean: The OFoldX submodule prefix to substitute.

    Returns:
        String ``(pattern, replacement)`` pairs mapping the input/output norms and output/gate projections.
    """

    # Non-fused triangle-multiplication keys (p_in/g_in are split separately, tensor-aware).
    return [
        (rf"^{boltz}\.norm_in\.", f"{clean}.norm_input."),
        (rf"^{boltz}\.norm_out\.", f"{clean}.norm_output."),
        (rf"^{boltz}\.p_out\.", f"{clean}.output_projection."),
        (rf"^{boltz}\.g_out\.", f"{clean}.gate_projection."),
    ]


# Boltz trunk PairformerLayer -> OFoldX PairformerBlock (rename-only keys; the fused p_in/g_in are
# handled tensor-aware by [`convert_pairformer_layer`][]).
PAIRFORMER_LAYER_RULES: list[Rule] = _rules(
    [
        (r"^pre_norm_s\.", "pair_attention.hidden_norm."),
        # Boltz-1 keeps the single-input norm inside AttentionPairBias (``attention.norm_s``); Boltz-2 pulls
        # it out into ``pre_norm_s``. Both land on the same OFoldX ``hidden_norm``.
        (r"^attention\.norm_s\.", "pair_attention.hidden_norm."),
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

# Boltz fuses the left/right (and their gates) into one 2*hidden Linear; OFoldX keeps them split.
_FUSED_SPLIT = {
    "tri_mul_out.p_in.weight": ("triangle_multiplication_outgoing", "left_projection", "right_projection"),
    "tri_mul_out.g_in.weight": ("triangle_multiplication_outgoing", "left_gate_projection", "right_gate_projection"),
    "tri_mul_in.p_in.weight": ("triangle_multiplication_incoming", "left_projection", "right_projection"),
    "tri_mul_in.g_in.weight": ("triangle_multiplication_incoming", "left_gate_projection", "right_gate_projection"),
}


def _split_fused(key: str, tensor: Any) -> list[tuple[str, object]] | None:
    r"""
    Split a fused Boltz triangle-multiplication projection into its OFoldX left/right halves.

    Args:
        key: The Boltz key; must be one of the fused ``p_in``/``g_in`` weights.
        tensor: The fused weight tensor whose first axis is split in half.

    Returns:
        Two ``(key, tensor)`` pairs for the left and right halves, or ``None`` when ``key`` is not fused.
    """

    if key not in _FUSED_SPLIT:
        return None
    block, left, right = _FUSED_SPLIT[key]
    half = tensor.shape[0] // 2
    return [(f"{block}.{left}.weight", tensor[:half]), (f"{block}.{right}.weight", tensor[half:])]


def convert_pairformer_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one PairformerLayer: split fused projections, rename the rest.

    Args:
        key: A PairformerLayer-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs (two when the fused projection is split, one after a rename, or empty
        when the key is dropped).
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIRFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Boltz PairformerNoSeqLayer (pair-only) -> OFoldX TrianglePairUpdateBlock. Same triangle ops as the
# full layer, but the pair FF is named ``feed_forward`` (not ``pair_feed_forward``) and there is no
# single track.
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
    Tensor-aware converter for one pair-only triangle layer (MSA / template).

    Args:
        key: A pair-only-layer-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs (two when a fused projection is split, one after a rename, or empty
        when the key is dropped).
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Boltz-1's MSA layer holds its triangle ops as direct attributes (with the pair transition named
# ``z_transition``); Boltz-2 nests the same ops under ``pairformer_layer.``.
MSA_DIRECT_TRIANGLE_RULES: list[Rule] = _rules(
    [
        *_triangle_mult_rename_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_mult_rename_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("z_transition", "feed_forward"),
    ]
)


def convert_msa_direct_triangle(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for one Boltz-1 MSA-layer direct triangle-op key.

    Args:
        key: A ``tri_mul_out.`` / ``tri_mul_in.`` / ``tri_att_start.`` / ``tri_att_end.`` /
            ``z_transition.`` key relative to one MSA layer.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX ``triangle_pair_update`` submodule names.
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, MSA_DIRECT_TRIANGLE_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Boltz MSALayer submodule inner-rule sets (relative to each submodule).
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
    Tensor-aware converter for one MSALayer (suffix relative to ``msa_module.layers.N.``).

    Args:
        key: A key relative to one MSA layer (``pair_weighted_averaging.``, ``msa_transition.``,
            ``outer_product_mean.``, or ``pairformer_layer.``).
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX MSA-layer names.

    Raises:
        KeyError: If ``key`` does not belong to a known MSA-layer submodule.
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
    if key.startswith(("tri_mul_out.", "tri_mul_in.", "tri_att_start.", "tri_att_end.", "z_transition.")):
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in convert_msa_direct_triangle(key, tensor)]
    raise KeyError(f"unmapped MSA layer key: {key}")


def convert_msa_module(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for the whole ``msa_module.*`` (prefix-stripped).

    Args:
        key: An ``msa_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX MSA-module names.

    Raises:
        KeyError: If ``key`` does not belong to a known MSA-module submodule.
    """

    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"layers.{index}.{nk}", nv) for nk, nv in convert_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


# Boltz TemplateV2Module -> OFoldX TemplateV2Embedder.
TEMPLATE_FLAT_RULES: list[Rule] = _rules(
    [
        (r"^z_norm\.", "norm_pair."),
        (r"^v_norm\.", "hidden_norm."),
        (r"^z_proj\.", "pair_projection."),
        (r"^a_proj\.", "template_projection."),
        (r"^u_proj\.", "output_projection."),
    ]
)


def convert_template_module(key: str, tensor: object) -> list[tuple[str, object]]:
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
        return [(f"blocks.{index}.{nk}", nv) for nk, nv in convert_pair_only_layer(remainder, tensor)]
    new_key = convert_keys(key, TEMPLATE_FLAT_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Token-level encodings + conditioning embeddings at the input-embedder top level.
INPUT_EMBEDDER_FLAT_RULES: list[Rule] = _rules(
    [
        (r"^res_type_encoding\.", "residue_type_projection."),
        (r"^msa_profile_encoding\.", "msa_profile_projection."),
        (r"^method_conditioning_init\.", "method_embedding."),
        (r"^modified_conditioning_init\.", "modified_embedding."),
        (r"^cyclic_conditioning_init\.", "cyclic_projection."),
        (r"^mol_type_conditioning_init\.", "molecule_type_embedding."),
        (r"^atom_enc_proj_z\.", "atom_pair_bias_projection."),
        (r"^atom_attention_encoder\.atom_to_token_trans\.", "atom_attention_encoder.token_projection."),
    ]
)

_ATTN_OLD = "atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
_ATTN_NEW = "atom_attention_encoder.atom_transformer.transformer.layers."


def convert_input_embedder_key(key: str) -> str | None:
    r"""
    Route a Boltz ``input_embedder.*`` (prefix-stripped) key to its OFoldX name.

    Args:
        key: An ``input_embedder``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
    """

    if key.startswith("atom_encoder."):
        return reprefix(key, "atom_encoder.", "atom_encoder.", _ATOM_ENCODER_RULES)
    if key.startswith(_ATTN_OLD):
        return reprefix_indexed(key, _ATTN_OLD, _ATTN_NEW, CONDITIONED_TRANSFORMER_LAYER_RULES)
    return convert_keys(key, INPUT_EMBEDDER_FLAT_RULES)


def convert_keys(key: str, rules: list[Rule]) -> str | None:
    r"""
    Rewrite ``key`` by the first matching rule; return ``None`` to drop the key.

    Rules are tried in order, so more specific patterns must be listed first. An unmatched key passes through
    unchanged, while a rule with a ``None`` replacement drops it.

    Args:
        key: The state-dict key to rewrite.
        rules: An ordered rule list of ``(compiled_regex, replacement_or_None)``.

    Returns:
        The rewritten key, the original key when no rule matches, or ``None`` when a matching rule drops it.

    Examples:
        >>> import re
        >>> rules = [(re.compile(r"^distogram\."), "linear.")]
        >>> convert_keys("distogram.weight", rules)
        'linear.weight'
        >>> convert_keys("other.weight", rules)
        'other.weight'
        >>> convert_keys("drop.weight", [(re.compile(r"^drop\."), None)]) is None
        True
    """

    for pattern, replacement in rules:
        if pattern.search(key):
            return None if replacement is None else pattern.sub(replacement, key)
    return key


def apply_converter(key: str, converter: Converter) -> str | None:
    r"""
    Apply a :data:`Converter` (rule list or callable) to a single key.

    Args:
        key: The state-dict key to convert.
        converter: Either an ordered rule list or a callable ``key -> key_or_None``.

    Returns:
        The converted key, or ``None`` to drop it.

    Examples:
        >>> import re
        >>> apply_converter("distogram.weight", [(re.compile(r"^distogram\."), "linear.")])
        'linear.weight'
        >>> apply_converter("a.b", lambda key: key.upper())
        'A.B'
    """

    if callable(converter):
        return converter(key)
    return convert_keys(key, converter)


def reprefix(key: str, old_prefix: str, new_prefix: str, sub: Converter) -> str | None:
    r"""
    Convert the remainder of ``key`` after ``old_prefix`` with ``sub`` and re-prefix it.

    Args:
        key: The full key beginning with ``old_prefix``.
        old_prefix: The prefix to strip before applying ``sub``.
        new_prefix: The prefix to prepend to the converted remainder.
        sub: Converter applied to the stripped remainder.

    Returns:
        ``new_prefix`` + converted remainder, or ``None`` when ``sub`` drops the remainder.
    """

    converted = apply_converter(key[len(old_prefix) :], sub)
    return None if converted is None else new_prefix + converted


def reprefix_indexed(key: str, old_prefix: str, new_prefix: str, sub: Converter) -> str | None:
    r"""
    Like [`reprefix`][ofoldx.models.boltz2.convert.reprefix] but preserves a ``<index>.`` segment that follows
    the prefix.

    Args:
        key: The full key beginning with ``old_prefix`` followed by an ``<index>.`` segment.
        old_prefix: The prefix to strip before the index.
        new_prefix: The prefix to prepend to the ``<index>.`` and converted remainder.
        sub: Converter applied to the remainder after the index.

    Returns:
        ``new_prefix<index>.`` + converted remainder, or ``None`` when ``sub`` drops the remainder.
    """

    index, _, remainder = key[len(old_prefix) :].partition(".")
    converted = apply_converter(remainder, sub)
    return None if converted is None else f"{new_prefix}{index}.{converted}"


def convert_subset(subset: dict[str, object], converter: Converter) -> dict[str, object]:
    r"""
    Apply ``converter`` to every key of ``subset``, dropping keys mapped to ``None``.

    Args:
        subset: A ``{key: tensor}`` sub-dict of a Boltz state dict.
        converter: Converter applied to each key.

    Returns:
        A new dict keyed by the converted names, with dropped keys omitted.

    Examples:
        >>> import re
        >>> convert_subset({"distogram.weight": 1}, [(re.compile(r"^distogram\."), "linear.")])
        {'linear.weight': 1}
    """

    converted: dict[str, object] = {}
    for key, value in subset.items():
        new_key = apply_converter(key, converter)
        if new_key is not None:
            converted[new_key] = value
    return converted


def convert_subset_tensors(
    subset: dict[str, object], converter: Callable[[str, Any], list[tuple[str, object]]]
) -> dict[str, object]:
    r"""
    Apply a tensor-aware converter ``(key, tensor) -> [(key, tensor), ...]`` to ``subset``.

    Used by subsystems that split or otherwise reshape weights (e.g. the fused triangle-multiplication
    projections), where one source key can map to several targets.

    Args:
        subset: A ``{key: tensor}`` sub-dict of a Boltz state dict.
        converter: A tensor-aware converter mapping one ``(key, tensor)`` to a list of output pairs.

    Returns:
        A new dict keyed by the converted names (one source key may contribute several entries).
    """

    converted: dict[str, object] = {}
    for key, value in subset.items():
        for new_key, new_value in converter(key, value):
            converted[new_key] = new_value
    return converted


_ADAPTIVE_NORM_MODULES = ("conditioned_norm", "key_value_norm")


def _adaptive_norm_base(key: str, projection: str, param: str) -> str | None:
    r"""
    Return the module base of an AdaLN scale/shift key, or ``None`` if ``key`` is not such a key.

    Args:
        key: A converted state-dict key.
        projection: The projection name to match (``"gate_projection"`` or ``"shift_projection"``).
        param: The parameter suffix to match (``"weight"`` or ``"bias"``).

    Returns:
        The key prefix up to and including the adaptive-norm module (so ``base + tail`` reconstructs ``key``),
        or ``None`` when ``key`` does not end in the given adaptive-norm projection/param.
    """

    tail = f"{projection}.{param}"
    for module in _ADAPTIVE_NORM_MODULES:
        suffix = f"{module}.{tail}"
        if key == suffix or key.endswith("." + suffix):
            return key[: -len(tail)]
    return None


def pack_adaptive_layer_norm_state(state_dict: dict[str, object]) -> dict[str, object]:
    r"""
    Keep converted AdaLN scale/shift projections in the explicit Boltz layout.

    Args:
        state_dict: A converted state dict carrying split ``gate_projection`` / ``shift_projection`` AdaLN entries.

    Returns:
        A shallow copy of ``state_dict``.
    """

    return dict(state_dict)


# Boltz ConfidenceModule -> OFoldX ConfidenceHead (clean names).
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
    Tensor-aware converter for source ``confidence_module.*`` (prefix-stripped).

    Args:
        key: A ``confidence_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX confidence-head names (empty when dropped).
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
AFFINITY_PAIR_ONLY_LAYER_RULES: list[Rule] = _rules(
    [
        *_triangle_mult_rename_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_mult_rename_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_z", "pair_feed_forward"),
    ]
)


def convert_affinity_pairformer_layer(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for an affinity-head pair-only Pairformer layer.

    The affinity head uses ``PairformerStack(single_size=0)``, whose pair transition remains named
    ``pair_feed_forward``. Template/MSA pair-only stacks use ``TrianglePairUpdateBlock.feed_forward`` instead.
    """

    split = _split_fused(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, AFFINITY_PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


def convert_affinity_head(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Tensor-aware converter for a prefix-stripped upstream ``affinity_module*.*`` key.

    Args:
        key: Checkpoint key with the ``affinity_module*.`` prefix already stripped.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX affinity-head names.
    """

    if key.startswith("pairwise_conditioner."):
        new = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return [("pairwise_conditioner." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"pairformer_stack.blocks.{index}.{nk}", nv)
            for nk, nv in convert_affinity_pairformer_layer(remainder, tensor)
        ]
    if key.startswith("affinity_heads."):
        new = convert_keys(key[len("affinity_heads.") :], AFFINITY_HEADS_RULES)
        return [("affinity_heads." + new, tensor)] if new is not None else []
    new = convert_keys(key, AFFINITY_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


def convert_affinity_modules(state_dict: dict[str, object]) -> dict[str, object]:
    r"""
    Convert upstream affinity head weights into ``Boltz2AffinityModel`` module names.

    Args:
        state_dict: A full Boltz-2 affinity checkpoint ``state_dict``.

    Returns:
        Converted affinity head tensors under ``affinity_module*``.
    """

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        for module in ("affinity_module1", "affinity_module2", "affinity_module"):
            prefix = module + "."
            if key.startswith(prefix):
                for new_key, new_value in convert_affinity_head(key[len(prefix) :], value):
                    converted[f"{module}.{new_key}"] = new_value
                break
    return converted


# Top-level trunk renames (Boltz key -> Boltz2Model attribute path).
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


def convert_trunk_key(key: str, tensor: object) -> list[tuple[str, object]]:
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
        return [("input_embedder." + new, tensor)] if new is not None else []
    if key.startswith("template_module."):
        inner = convert_template_module(key[len("template_module.") :], tensor)
        return [("template_embedder." + nk, nv) for nk, nv in inner]
    if key.startswith("msa_module."):
        inner = convert_msa_module(key[len("msa_module.") :], tensor)
        return [("msa_module." + nk, nv) for nk, nv in inner]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        inner = convert_pairformer_layer(remainder, tensor)
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in inner]
    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("distogram_module."):
        new = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("diffusion_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = convert_score_model(key[len("structure_module.score_model.") :])
        return [("structure_module.score_model." + new, tensor)] if new is not None else []
    if key.startswith("confidence_module."):
        inner = convert_confidence_head(key[len("confidence_module.") :], tensor)
        return [("confidence_head." + nk, nv) for nk, nv in inner]
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in _TRUNK_FLAT:
        return [(_TRUNK_FLAT[key], tensor)]
    return []


def convert_trunk(state_dict: dict[str, object]) -> dict[str, object]:
    r"""
    Convert a full Boltz-2 ``state_dict`` to the ``Boltz2Model`` trunk param tree (non-trunk dropped).

    Applies [`convert_trunk_key`][ofoldx.models.boltz2.convert.convert_trunk_key] to every entry, then packs
    the AdaLN scale/shift projections so the result ``strict``-loads into the OFoldX modules.

    Args:
        state_dict: A full Boltz-2 checkpoint ``state_dict``.

    Returns:
        The converted trunk ``state_dict`` under the ``Boltz2Model`` param names.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> ckpt = torch.load("boltz2_conf.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_trunk(ckpt)  # doctest: +SKIP
        >>> model.load_state_dict(converted, strict=True)  # doctest: +SKIP
    """

    converted: dict[str, object] = {}
    for key, value in state_dict.items():
        for new_key, new_value in convert_trunk_key(key, value):
            converted[new_key] = new_value
    return pack_adaptive_layer_norm_state(converted)


def convert_boltz2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    architecture: str = "boltz2",
    state_key: str = "auto",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    r"""
    Convert a public Boltz-2 checkpoint into a pretrained OFoldX artifact directory.

    Args:
        checkpoint_path: Path to ``boltz2_conf.ckpt``.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        architecture: Target architecture: ``"boltz2"`` or ``"boltz2_affinity"``.
        state_key: Checkpoint state-dict key: ``"auto"``, ``"state_dict"``, ``"model_state_dict"``,
            ``"model"``, or ``"root"``.
        dtype: Optional floating-point dtype for persisted weights.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_boltz2_checkpoint("boltz2_conf.ckpt", "boltz2")  # doctest: +SKIP
    """

    from .config import Boltz2AffinityConfig, Boltz2Config
    from .model import Boltz2AffinityModel, Boltz2Model

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = _load_boltz_lightning_checkpoint(checkpoint_path)
    raw_state, selected_state_key = _select_state_dict(checkpoint, state_key=state_key)
    if architecture == "boltz2":
        model = Boltz2Model(Boltz2Config())
        converted_raw = convert_trunk(dict(raw_state))
    elif architecture == "boltz2_affinity":
        model = Boltz2AffinityModel(Boltz2AffinityConfig())
        converted_raw = convert_trunk(dict(raw_state))
        converted_raw.update(convert_affinity_modules(dict(raw_state)))
    else:
        raise ValueError(f"unsupported Boltz-2 architecture: {architecture!r}")
    converted = _cast_floating_tensors(_tensor_state_dict(converted_raw), dtype)
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_state_dict(converted, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": model.config.architecture,
        "state_key": selected_state_key,
        "tensor_count": len(converted),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(converted),
    }


def _load_boltz_lightning_checkpoint(checkpoint_path: Path) -> Mapping[str, Any]:
    r"""
    Load Boltz Lightning checkpoints without requiring the original ``boltzgen`` package.

    Some released checkpoints pickle validator objects from ``boltzgen.model.validation`` next to the tensor
    ``state_dict``. OFoldX only consumes the tensors, so this loader stubs those metadata classes when needed.
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
    r"""Build minimal modules/classes needed to unpickle released Boltz checkpoint metadata."""

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
    r"""Parse command-line arguments for Boltz-2 checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a public Boltz-2 .ckpt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument(
        "--architecture",
        choices=("boltz2", "boltz2_affinity"),
        default="boltz2",
        help="Target OFoldX architecture to save.",
    )
    parser.add_argument(
        "--state-key", default="auto", help="Checkpoint state key: auto, state_dict, model_state_dict, model, or root"
    )
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for Boltz-2 checkpoint conversion.

    Examples:
        >>> main(["boltz2_conf.ckpt", "out/boltz2"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_boltz2_checkpoint(
        args.checkpoint,
        args.output_dir,
        architecture=args.architecture,
        state_key=args.state_key,
        dtype=dtype,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted Boltz-2 checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
