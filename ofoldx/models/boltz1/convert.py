r"""
Boltz-1 checkpoint converter.

The public Boltz-1 checkpoints store the older monolithic score model. OFoldX routes invariant
pair/atom conditioning into ``denoiser_conditioning`` and learned denoising layers into
``coordinate_diffusion.denoiser``. This converter preserves checkpoint tensor meaning for strict loading.
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

from ...artifacts import save_model
from .config import Boltz1Config
from .model import Boltz1Model
from .processor import Boltz1Processor

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
    first, second = tensor.chunk(2, dim=0)
    if prefix.endswith("output_token_feature_update.transition"):
        projections = (("gate_projection", first), ("value_projection", second))
    else:
        projections = (("value_projection", first), ("gate_projection", second))
    return [(f"{prefix}.{projection}.{suffix}", value) for projection, value in projections]


CONDITIONED_TRANSFORMER_BLOCK_RULES: list[Rule] = compile_rules(
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


ATOM_ENCODER_RULES: list[Rule] = compile_rules(
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
    [(r"^dim_pairwise_init_proj\.", "init_projection."), *_transition_index_rules()]
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
        *_triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_transition_rules("transition_s", "single_transition"),
        *_transition_rules("transition_z", "pair_transition"),
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
        return split
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


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def convert_boltz1_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Convert a full Boltz-1 ``state_dict`` to [`Boltz1Model`][ofoldx.models.boltz1.model.Boltz1Model] keys.

    Every source key is stripped of a leading ``module.`` and routed through
    [`convert_boltz1_key`][ofoldx.models.boltz1.convert.convert_boltz1_key]; unrecognized, non-ignorable
    keys fail immediately. Adaptive-layer-norm parameters are packed into the OFoldX layout.

    Args:
        state_dict: The source Boltz-1 parameter mapping (checkpoint ``state_dict``).
        dtype: Optional dtype to cast floating-point tensors to; leaves them unchanged when ``None``.

    Returns:
        Mapping from OFoldX parameter name to tensor (plus any packed/synthesized entries).

    Raises:
        KeyError: If a source key is unmapped.
    """

    converted: dict[str, Tensor] = {}
    for source_key, value in state_dict.items():
        key = source_key.removeprefix("module.")
        items = convert_boltz1_key(key, value)
        if not items and not is_tied_output_projection(key):
            raise KeyError(f"unmapped Boltz-1 checkpoint key: {key}")
        for packed_key, packed_value in items:
            for target_key, target_value in _unpack_triple_product_projection(packed_key, packed_value):
                if dtype is not None and target_value.is_floating_point():
                    target_value = target_value.to(dtype=dtype)
                converted[target_key] = target_value
    _pack_conditioned_adaptive_norms(converted)
    return converted


def _pack_conditioned_adaptive_norms(state_dict: dict[str, Tensor]) -> None:
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


def convert_boltz1_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Route one Boltz-1 checkpoint key to zero or more OFoldX keys.

    The key's top-level prefix selects a sub-converter (input embedder, MSA module, Pairformer, distogram,
    score model, out-token-feature update, or confidence module); flat trunk keys map through a lookup table.
    Keys that no rule recognizes yield an empty list, letting the caller decide whether that is an error.

    Args:
        key: A single Boltz-1 checkpoint key, already stripped of any ``module.`` prefix.
        tensor: The associated parameter value, passed through to the emitted target key(s).

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    if key.startswith("input_embedder."):
        input_inner = convert_input_embedder_key(key[len("input_embedder.") :], tensor)
        return [] if input_inner is None else [(input_inner[0], input_inner[1])]
    if key.startswith("msa_module."):
        msa_items = _convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_trunk.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = _convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("distogram_module."):
        new_key = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [] if new_key is None else [(f"distogram_head.{new_key}", tensor)]
    if key.startswith("structure_module.score_model."):
        return convert_score_model_key(key[len("structure_module.score_model.") :], tensor)
    if key.startswith("structure_module.out_token_feat_update."):
        return convert_output_token_feature_update_key(key[len("structure_module.out_token_feat_update.") :], tensor)
    if key.startswith("confidence_module."):
        confidence_items = convert_confidence_module_key(key[len("confidence_module.") :], tensor)
        return [(f"confidence_head.{new_key}", value) for new_key, value in confidence_items]
    if key in _TRUNK_FLAT:
        return [(_TRUNK_FLAT[key], tensor)]
    return []


def convert_input_embedder_key(key: str, tensor: Tensor) -> tuple[str, Tensor] | None:
    r"""
    Convert one ``input_embedder.*`` key from the old Boltz-1 atom embedder.

    Handles the atom-attention-encoder base parameters, the per-layer transformer/pair-bias routing, and the
    atom-to-token transition rename.

    Args:
        key: The ``input_embedder``-relative key (leading ``input_embedder.`` already removed).
        tensor: The associated parameter value.

    Returns:
        A single ``(target_key, value)`` pair, or ``None`` when the key is not an input-embedder parameter.
    """

    prefix = "input_embeddings."
    if key.startswith("atom_attention_encoder."):
        inner = key[len("atom_attention_encoder.") :]
        atom_key = _convert_atom_encoder_base_key(inner)
        if atom_key is not None:
            return (prefix + atom_key, tensor)
        routed = _convert_transformer_block_key(
            inner,
            old_prefix="atom_encoder.diffusion_transformer.layers.",
            block_prefix="input_embeddings.atom_attention_encoder.atom_transformer.transformer.blocks.",
            pair_bias_prefix="input_embeddings.atom_pair_bias_projection.bias_projections.",
            tensor=tensor,
        )
        if routed is not None:
            return routed
        if inner.startswith("atom_to_token_trans."):
            return (
                prefix + inner.replace("atom_to_token_trans.", "atom_attention_encoder.token_projection.", 1),
                tensor,
            )
    return None


def convert_score_model_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one ``structure_module.score_model.*`` key from the old Boltz-1 score model.

    This is where the OFoldX split happens: the invariant pairwise conditioner moves to
    ``denoiser_conditioning`` while the single conditioner, atom encoder/decoder, token transformer, and
    the single-to-token / token-norm layers stay under ``coordinate_diffusion.denoiser``.

    Args:
        key: The ``score_model``-relative key (leading ``structure_module.score_model.`` already removed).
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    score_prefix = "coordinate_diffusion.denoiser."
    if key.startswith("single_conditioner."):
        new = convert_keys(key[len("single_conditioner.") :], SINGLE_CONDITIONING_RULES)
        return [] if new is None else [(score_prefix + "single_conditioner." + new, tensor)]
    if key.startswith("pairwise_conditioner."):
        new = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return [] if new is None else [("denoiser_conditioning.pairwise_conditioner." + new, tensor)]
    if key.startswith("atom_attention_encoder."):
        return _convert_score_atom_encoder_key(key[len("atom_attention_encoder.") :], tensor)
    if key.startswith("token_transformer.layers."):
        routed = _convert_transformer_block_key(
            key,
            old_prefix="token_transformer.layers.",
            block_prefix=score_prefix + "token_transformer.blocks.",
            pair_bias_prefix="denoiser_conditioning.token_transformer_pair_bias_projections.",
            tensor=tensor,
        )
        return [] if routed is None else [routed]
    if key.startswith("atom_attention_decoder."):
        return _convert_score_atom_decoder_key(key[len("atom_attention_decoder.") :], tensor)
    if key.startswith("s_to_a_linear."):
        return [(score_prefix + key.replace("s_to_a_linear.", "single_to_token.", 1), tensor)]
    if key.startswith("a_norm."):
        return [(score_prefix + key.replace("a_norm.", "token_norm.", 1), tensor)]
    return []


def convert_output_token_feature_update_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one ``structure_module.out_token_feat_update.*`` key (the diffusion-token accumulator).

    Args:
        key: The ``out_token_feat_update``-relative key (its checkpoint prefix already removed).
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    prefix = "coordinate_diffusion.output_token_feature_update."
    if key.startswith("norm_next."):
        return [(prefix + key.replace("norm_next.", "next_norm.", 1), tensor)]
    if key.startswith("fourier_embed.proj."):
        return [(prefix + key.replace("fourier_embed.proj.", "noise_fourier.projection.", 1), tensor)]
    if key.startswith("norm_fourier."):
        return [(prefix + key.replace("norm_fourier.", "fourier_norm.", 1), tensor)]
    if key.startswith("transition_block."):
        new = _convert_conditioned_transition_key(key[len("transition_block.") :])
        return [] if new is None else [(prefix + "transition." + new, tensor)]
    return []


def convert_confidence_module_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one ``confidence_module.*`` key from the official Boltz-1 confidence module.

    The confidence module embeds its own copies of the input embedder, MSA module, and Pairformer stack;
    this routes each of those (plus the confidence projection heads, transition block, flat trunk keys, and
    single/pair conditioning prefixes) into the OFoldX ``confidence_head.*`` namespace.

    Args:
        key: The ``confidence_module``-relative key (leading ``confidence_module.`` already removed).
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    if key.startswith("input_embedder."):
        input_inner = convert_input_embedder_key(key[len("input_embedder.") :], tensor)
        return [] if input_inner is None else [(input_inner[0], input_inner[1])]
    if key.startswith("msa_module."):
        msa_items = _convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_trunk.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = _convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        pairformer_items = _convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("confidence_heads."):
        confidence_key = _convert_boltz1_confidence_head_key(key[len("confidence_heads.") :])
        return [] if confidence_key is None else [(f"output_head.{confidence_key}", tensor)]
    if key.startswith("transition_block."):
        transition_key = _convert_conditioned_transition_key(key[len("transition_block.") :])
        return [] if transition_key is None else [(f"transition_block.{transition_key}", tensor)]
    flat = _CONFIDENCE_FLAT.get(key)
    if flat is not None:
        return [(flat, tensor)]
    for old, new in _CONFIDENCE_PREFIXES:
        if key.startswith(old):
            return [(key.replace(old, new, 1), tensor)]
    return []


def convert_boltz1_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz1Config | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz1Model:
    r"""
    Convert an original Boltz-1 ``.ckpt`` into a strict-loaded model.

    Loads the checkpoint (taking its ``state_dict`` entry when present), remaps the parameters via
    [`convert_boltz1_state_dict`][ofoldx.models.boltz1.convert.convert_boltz1_state_dict], and strict-loads the
    resulting tensors into a native model. The Boltz-1 Processor and ``FoldResult`` decoder provide the surrounding
    inference path; the converter itself saves nothing implicitly and declares no dtype policy beyond the optional
    cast. Persistence is available through [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model].

    Args:
        checkpoint_path: Path to the original Boltz-1 ``.ckpt`` file.
        config: Optional Boltz-1 config (or mapping); defaults to [`Boltz1Config`][].
        dtype: Optional dtype to cast floating-point tensors to during conversion.

    Returns:
        The converted, strict-loaded model.

    Raises:
        KeyError: If a source key is unmapped.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    converted = convert_boltz1_state_dict(checkpoint["state_dict"], dtype=dtype)
    with torch.device("meta"):
        model = Boltz1Model(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


_TRUNK_FLAT = {
    "s_init.weight": "single_init_projection.weight",
    "z_init_1.weight": "pair_left_projection.weight",
    "z_init_2.weight": "pair_right_projection.weight",
    "rel_pos.linear_layer.weight": "relative_position_embedder.output_projection.weight",
    "token_bonds.weight": "token_bond_projection.weight",
    "s_norm.weight": "single_recycle_norm.weight",
    "s_norm.bias": "single_recycle_norm.bias",
    "z_norm.weight": "pair_recycle_norm.weight",
    "z_norm.bias": "pair_recycle_norm.bias",
    "s_recycle.weight": "single_recycle_projection.weight",
    "z_recycle.weight": "pair_recycle_projection.weight",
}

_CONFIDENCE_FLAT = {
    "boundaries": "boundaries",
    "s_init.weight": "single_init_projection.weight",
    "z_init_1.weight": "pair_left_projection.weight",
    "z_init_2.weight": "pair_right_projection.weight",
    "rel_pos.linear_layer.weight": "relative_position_embedder.output_projection.weight",
    "token_bonds.weight": "token_bond_projection.weight",
    "s_norm.weight": "single_recycle_norm.weight",
    "s_norm.bias": "single_recycle_norm.bias",
    "z_norm.weight": "pair_recycle_norm.weight",
    "z_norm.bias": "pair_recycle_norm.bias",
    "s_recycle.weight": "single_recycle_projection.weight",
    "z_recycle.weight": "pair_recycle_projection.weight",
    "final_s_norm.weight": "final_single_norm.weight",
    "final_s_norm.bias": "final_single_norm.bias",
    "final_z_norm.weight": "final_pair_norm.weight",
    "final_z_norm.bias": "final_pair_norm.bias",
}

_CONFIDENCE_PREFIXES = (
    ("dist_bin_pairwise_embed.", "distance_embedding."),
    ("s_diffusion_norm.", "diffusion_single_norm."),
    ("s_diffusion_to_s.", "diffusion_single_projection."),
    ("s_inputs_norm.", "input_single_norm."),
    ("s_input_to_s.", "input_single_projection."),
    ("s_to_z_transpose.", "single_to_pair_col."),
    ("s_to_z_prod_in1.", "single_to_pair_product_left."),
    ("s_to_z_prod_in2.", "single_to_pair_product_right."),
    ("s_to_z_prod_out.", "single_to_pair_product_out."),
    ("s_to_z.", "single_to_pair_row."),
)


def _convert_score_atom_encoder_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    atom_key = _convert_atom_encoder_base_key(key)
    if atom_key is not None:
        if atom_key.startswith("atom_attention_encoder.noisy_position_projection."):
            if key == "r_to_q_trans.weight":
                # Clone the 3-column slice so it owns its storage; a bare view would not cover the
                # source weight's storage and safetensors refuses to serialize it on save (persistence).
                tensor = tensor[:, :3].clone()
            return [("coordinate_diffusion.denoiser." + atom_key, tensor)]
        return [("denoiser_conditioning." + atom_key, tensor)]
    routed = _convert_transformer_block_key(
        key,
        old_prefix="atom_encoder.diffusion_transformer.layers.",
        block_prefix="coordinate_diffusion.denoiser.atom_attention_encoder.atom_transformer.transformer.blocks.",
        pair_bias_prefix="denoiser_conditioning.atom_encoder_pair_bias_projections.",
        tensor=tensor,
    )
    if routed is not None:
        return [routed]
    if key.startswith("atom_to_token_trans."):
        return [
            (
                "coordinate_diffusion.denoiser."
                + key.replace("atom_to_token_trans.", "atom_attention_encoder.token_projection.", 1),
                tensor,
            )
        ]
    return []


def _convert_score_atom_decoder_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    routed = _convert_transformer_block_key(
        key,
        old_prefix="atom_decoder.diffusion_transformer.layers.",
        block_prefix="coordinate_diffusion.denoiser.atom_attention_decoder.atom_transformer.transformer.blocks.",
        pair_bias_prefix="denoiser_conditioning.atom_decoder_pair_bias_projections.",
        tensor=tensor,
    )
    if routed is not None:
        return [routed]
    if key.startswith("a_to_q_trans."):
        return [
            (
                "coordinate_diffusion.denoiser."
                + key.replace("a_to_q_trans.", "atom_attention_decoder.token_projection.", 1),
                tensor,
            )
        ]
    if key.startswith("atom_feat_to_atom_pos_update."):
        return [
            (
                "coordinate_diffusion.denoiser."
                + key.replace("atom_feat_to_atom_pos_update.", "atom_attention_decoder.position_update.", 1),
                tensor,
            )
        ]
    return []


def _convert_atom_encoder_base_key(key: str) -> str | None:
    new = convert_keys(key, ATOM_ENCODER_RULES)
    if new is not None and new != key:
        return "atom_encoder." + new
    if key.startswith("r_to_q_trans."):
        return key.replace("r_to_q_trans.", "atom_attention_encoder.noisy_position_projection.", 1)
    return None


def _convert_transformer_block_key(
    key: str,
    *,
    old_prefix: str,
    block_prefix: str,
    pair_bias_prefix: str,
    tensor: Tensor,
) -> tuple[str, Tensor] | None:
    if not key.startswith(old_prefix):
        return None
    index, _, remainder = key[len(old_prefix) :].partition(".")
    pair_bias = _convert_pair_bias_projection_suffix(remainder)
    if pair_bias is not None:
        return (f"{pair_bias_prefix}{index}.{pair_bias}", tensor)
    new = convert_keys(remainder, CONDITIONED_TRANSFORMER_BLOCK_RULES)
    return None if new is None else (f"{block_prefix}{index}.{new}", tensor)


def _convert_boltz1_confidence_head_key(key: str) -> str | None:
    replacements = {
        "to_pde_logits.": "pde_projection.",
        "to_plddt_logits.": "plddt_projection.",
        "to_resolved_logits.": "resolved_projection.",
        "to_pae_logits.": "pae_projection.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            return key.replace(old, new, 1)
    return None


def _convert_conditioned_transition_key(key: str) -> str | None:
    if key.startswith("adaln.a_norm."):
        return None
    replacements = (
        ("adaln.s_norm.", "conditioned_norm.conditioning_norm."),
        ("adaln.s_scale.", "conditioned_norm.gate_projection."),
        ("adaln.s_bias.", "conditioned_norm.shift_projection."),
        ("swish_gate.0.", "gate_value_projection."),
        ("a_to_b.", "expand_projection."),
        ("b_to_a.", "output_projection."),
        ("output_projection.0.", "conditioning_gate_projection."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return key.replace(old, new, 1)
    return None


def _convert_pair_bias_projection_suffix(key: str) -> str | None:
    if key.startswith("pair_bias_attn.proj_z.0."):
        return "0." + key[len("pair_bias_attn.proj_z.0.") :]
    if key.startswith("pair_bias_attn.proj_z.1."):
        return "1." + key[len("pair_bias_attn.proj_z.1.") :]
    return None


def _prefix_index(index: str, items: list[tuple[str, Tensor]]) -> list[tuple[str, Tensor]]:
    return [(f"{index}.{key}", value) for key, value in items]


def save_boltz1_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    config: Boltz1Config | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert a Boltz-1 ``.ckpt`` and write a standard OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_boltz1_checkpoint`][ofoldx.models.boltz1.convert.convert_boltz1_checkpoint], pairs
    the model with the config-less [`Boltz1Processor`][], and writes a self-contained
    ``model.json`` / safetensors repository via
    [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model]. The Processor carries no config,
    so no ``processor.json`` is written.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository to
    that Hub repo. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    model = convert_boltz1_checkpoint(checkpoint_path, config=config, dtype=dtype)
    save_model(
        output_dir,
        model=model,
        processor=Boltz1Processor(),
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run boltz1 checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="original Boltz-1 .ckpt (e.g. boltz1_conf.ckpt)")
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
    save_boltz1_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
