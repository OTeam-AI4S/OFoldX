r"""
Checkpoint converter: BoltzGen inverse-folding checkpoint -> OFoldX clean names.

Routes the three top-level prefixes of ``boltzgen1_ifold.ckpt`` (input_embedder / inverse_folding_encoder /
structure_module) to the matching OFoldX modules. The input embedder reuses the Boltz-2 atom converters;
the encoder/decoder GNN layers use the clean MLP/transition renames. Strict-load after conversion = parity.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch

from .config import BoltzGenConfig, BoltzGenIFoldConfig
from .model import BoltzGenIFoldModel, BoltzGenModel

# --------------------------------------------------------------------------------------
# Boltz checkpoint-name conversion helpers, duplicated verbatim into this family so BoltzGen
# carries no cross-family import (family-boundary converter code is self-contained per family,
# not shared). Sourced from the Boltz-1/Boltz-2/BoltzGen shared converter; the whole-MSA-module
# converter is spelled ``convert_standard_msa_module`` here (BoltzGen owns its own tensor-aware
# ``convert_msa_module`` below, which dispatches to this standard one for the non-Miniformer path).
# --------------------------------------------------------------------------------------

Rule = tuple[re.Pattern[str], str | None]
Converter = list[Rule] | Callable[[str], "str | None"]
_TIED_OUTPUT_PROJECTION = re.compile(
    r"(?:\.diffusion_transformer\.layers\.\d+|\.token_transformer(?:_layers\.0)?\.layers\.\d+)"
    r"\.(?:transition\.)?output_projection\.0\.(?:weight|bias)$"
)


def is_tied_output_projection(key: str) -> bool:
    r"""Return whether ``key`` is a released Boltz checkpoint's tied projection duplicate."""

    return _TIED_OUTPUT_PROJECTION.search(key) is not None


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


def convert_standard_msa_module(key: str, tensor: object) -> list[tuple[str, object]]:
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

    if not miniformer_blocks:
        return convert_standard_msa_module(key, tensor)
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"layers.{index}.{nk}", nv) for nk, nv in convert_miniformer_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


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


def convert_template(state: dict, *, miniformer_blocks: bool = False) -> dict:
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


def convert_inverse_folding(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole inverse-folding checkpoint state dict onto a BoltzGenIFoldModel state dict.

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
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            new_key = None if inner is None else "input_embedder." + inner
        elif key.startswith("inverse_folding_encoder."):
            new_key = "encoder." + convert_encoder(key[len("inverse_folding_encoder.") :])
        elif key.startswith("structure_module."):
            new_key = "decoder." + convert_decoder(key[len("structure_module.") :])
        else:
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        if new_key is None:
            if is_tied_output_projection(key):
                continue
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        out[new_key] = value
    return out


def convert_structure_design(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def _sub(prefix: str) -> dict[str, torch.Tensor]:
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
            for k, v in convert_template(_sub("template_module"), miniformer_blocks=miniformer_template).items()
        }
    )
    for key, value in state.items():
        if key.startswith(("token_distance_module.", "template_module.")):
            continue
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            if inner is None:
                if is_tied_output_projection(key):
                    continue
                raise KeyError(f"unmapped structure-design checkpoint key: {key}")
            out["input_embedder." + inner] = value
        else:
            # BoltzGen wraps the diffusion token transformer in an outer ModuleList (token_layers=1):
            # token_transformer_layers.0.layers.<i> == boltz2's flat token_transformer.layers.<i>.
            source_key = key
            key = key.replace(
                "structure_module.score_model.token_transformer_layers.0.layers.",
                "structure_module.score_model.token_transformer.layers.",
                1,
            )
            converted = convert_trunk_key(
                key,
                value,
                miniformer_pairformer=miniformer_pairformer,
                miniformer_msa=miniformer_msa,
            )
            if not converted:
                if is_tied_output_projection(source_key):
                    continue
                raise KeyError(f"unmapped structure-design checkpoint key: {source_key}")
            for new_key, new_value in converted:
                out[new_key] = new_value
    return out


def convert_boltzgen_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: BoltzGenConfig | None = None,
    dtype: torch.dtype | None = None,
) -> BoltzGenModel:
    r"""Convert a released BoltzGen structure-design checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_structure_design`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = _cast_floating_tensors(convert_structure_design(state), dtype)
    with torch.device("meta"):
        model = BoltzGenModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_boltzgen_ifold_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: BoltzGenIFoldConfig | None = None,
    dtype: torch.dtype | None = None,
) -> BoltzGenIFoldModel:
    r"""Convert a released BoltzGen inverse-folding checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_inverse_folding`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = _cast_floating_tensors(convert_inverse_folding(state), dtype)
    with torch.device("meta"):
        model = BoltzGenIFoldModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def _cast_floating_tensors(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    r"""Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }
