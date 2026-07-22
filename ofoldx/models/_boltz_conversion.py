r"""Key and tensor mapping shared by released Boltz-format checkpoints."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import cast

from torch import Tensor

Rule = tuple[re.Pattern[str], str | None]
Converter = list[Rule] | Callable[[str], str | None]

TIED_OUTPUT_PROJECTION = re.compile(
    r"(?:\.diffusion_transformer\.layers\.\d+|\.token_transformer(?:_layers\.0)?\.layers\.\d+)"
    r"\.(?:transition\.)?output_projection\.0\.(?:weight|bias)$"
)


def is_tied_output_projection(key: str) -> bool:
    r"""Return whether ``key`` is a released Boltz checkpoint's tied projection duplicate."""

    return TIED_OUTPUT_PROJECTION.search(key) is not None


def compile_rules(pairs: Iterable[tuple[str, str | None]]) -> list[Rule]:
    r"""Compile ordered string regex pairs into conversion rules."""

    return [(re.compile(pattern), replacement) for pattern, replacement in pairs]


CONDITIONED_TRANSFORMER_LAYER_RULES: list[Rule] = compile_rules(
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

DISTOGRAM_RULES: list[Rule] = compile_rules([(r"^distogram\.", "linear.")])
BFACTOR_RULES: list[Rule] = compile_rules([(r"^bfactor\.", "projection.")])
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


def transition_index_rules() -> list[tuple[str, str | None]]:
    return [
        (r"^transitions\.(\d+)\.norm\.", r"transitions.\1.norm_input."),
        (r"^transitions\.(\d+)\.fc1\.", r"transitions.\1.gate_projection."),
        (r"^transitions\.(\d+)\.fc2\.", r"transitions.\1.value_projection."),
        (r"^transitions\.(\d+)\.fc3\.", r"transitions.\1.output_projection."),
    ]


SINGLE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^norm_single\.", "single_norm."),
        (r"^single_embed\.", "single_projection."),
        (r"^fourier_embed\.proj\.", "noise_fourier.projection."),
        (r"^norm_fourier\.", "fourier_norm."),
        (r"^fourier_to_single\.", "fourier_projection."),
        *transition_index_rules(),
    ]
)
PAIRWISE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [(r"^dim_pairwise_init_proj\.", "init_projection."), *transition_index_rules()]
)


def layer_route(key: str, old: str, new: str, rules: Converter) -> str | None:
    index, _, remainder = key[len(old) :].partition(".")
    converted = apply_converter(remainder, rules)
    return None if converted is None else f"{new}{index}.{converted}"


def convert_diffusion_conditioning(key: str) -> str | None:
    if key.startswith("pairwise_conditioner."):
        inner = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return "pairwise_conditioner." + inner if inner is not None else None
    if key.startswith("atom_encoder."):
        inner = convert_keys(key[len("atom_encoder.") :], ATOM_ENCODER_RULES)
        return "atom_encoder." + inner if inner is not None else None
    for boltz, house in (
        ("atom_enc_proj_z.", "atom_enc_proj_pair."),
        ("atom_dec_proj_z.", "atom_dec_proj_pair."),
        ("token_trans_proj_z.", "token_trans_proj_pair."),
    ):
        if key.startswith(boltz):
            return house + key[len(boltz) :]
    return key


def convert_atom_attention_encoder(key: str) -> str | None:
    if key.startswith("atom_encoder.diffusion_transformer.layers."):
        return layer_route(
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
    if key.startswith("atom_decoder.diffusion_transformer.layers."):
        return layer_route(
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
    if key.startswith("single_conditioner."):
        inner = convert_keys(key[len("single_conditioner.") :], SINGLE_CONDITIONING_RULES)
        return "single_conditioner." + inner if inner is not None else None
    if key.startswith("atom_attention_encoder."):
        inner = convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    if key.startswith("atom_attention_decoder."):
        inner = convert_atom_attention_decoder(key[len("atom_attention_decoder.") :])
        return None if inner is None else "atom_attention_decoder." + inner
    if key.startswith("token_transformer.layers."):
        return layer_route(
            key,
            "token_transformer.layers.",
            "token_transformer.layers.",
            CONDITIONED_TRANSFORMER_LAYER_RULES,
        )
    if key.startswith("s_to_a_linear."):
        return key.replace("s_to_a_linear.", "single_to_token.", 1)
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


def triangle_attention_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.layer_norm\.", f"{clean}.norm_pair."),
        (rf"^{boltz}\.linear\.", f"{clean}.bias_projection."),
        (rf"^{boltz}\.mha\.linear_q\.", f"{clean}.attention.projection.query_projection."),
        (rf"^{boltz}\.mha\.linear_k\.", f"{clean}.attention.projection.key_projection."),
        (rf"^{boltz}\.mha\.linear_v\.", f"{clean}.attention.projection.value_projection."),
        (rf"^{boltz}\.mha\.linear_o\.", f"{clean}.attention.projection.output_projection."),
        (rf"^{boltz}\.mha\.linear_g\.", f"{clean}.attention.projection.gate_projection."),
    ]


def transition_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.norm\.", f"{clean}.norm_input."),
        (rf"^{boltz}\.fc1\.", f"{clean}.gate_projection."),
        (rf"^{boltz}\.fc2\.", f"{clean}.value_projection."),
        (rf"^{boltz}\.fc3\.", f"{clean}.output_projection."),
    ]


def triangle_multiplication_rules(boltz: str, clean: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{boltz}\.norm_in\.", f"{clean}.norm_input."),
        (rf"^{boltz}\.norm_out\.", f"{clean}.norm_output."),
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
        (r"^attention\.proj_z\.0\.", "pair_attention.norm_pair."),
        (r"^attention\.proj_z\.1\.", "pair_attention.bias_projection."),
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("transition_s", "single_feed_forward"),
        *transition_rules("transition_z", "pair_feed_forward"),
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


def split_fused_projection(key: str, tensor: Tensor) -> list[tuple[str, Tensor]] | None:
    if key not in FUSED_TRIANGLE_PROJECTIONS:
        return None
    block, left, right = FUSED_TRIANGLE_PROJECTIONS[key]
    half = tensor.shape[0] // 2
    return [
        (f"{block}.{left}.weight", tensor[:half].clone()),
        (f"{block}.{right}.weight", tensor[half:].clone()),
    ]


def convert_pairformer_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIRFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


PAIR_ONLY_LAYER_RULES: list[Rule] = compile_rules(
    [
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("transition_z", "feed_forward"),
    ]
)


def convert_pair_only_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MSA_DIRECT_TRIANGLE_RULES: list[Rule] = compile_rules(
    [
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("z_transition", "feed_forward"),
    ]
)


def convert_msa_direct_triangle(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    split = split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = convert_keys(key, MSA_DIRECT_TRIANGLE_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MSA_PAIR_WEIGHTED_AVERAGING_RULES = compile_rules(
    [
        (r"^norm_m\.", "norm_msa."),
        (r"^norm_z\.", "norm_pair."),
        (r"^proj_m\.", "value_projection."),
        (r"^proj_g\.", "gate_projection."),
        (r"^proj_z\.", "pair_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)
MSA_TRANSITION_RULES = compile_rules(
    [
        (r"^norm\.", "norm_input."),
        (r"^fc1\.", "gate_projection."),
        (r"^fc2\.", "value_projection."),
        (r"^fc3\.", "output_projection."),
    ]
)
OUTER_PRODUCT_MEAN_RULES = compile_rules(
    [
        (r"^norm\.", "norm_msa."),
        (r"^proj_a\.", "left_projection."),
        (r"^proj_b\.", "right_projection."),
        (r"^proj_o\.", "output_projection."),
    ]
)


def convert_msa_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    if key.startswith("pair_weighted_averaging."):
        converted = reprefix(
            key,
            "pair_weighted_averaging.",
            "pair_weighted_averaging.",
            MSA_PAIR_WEIGHTED_AVERAGING_RULES,
        )
        return [(cast(str, converted), tensor)]
    if key.startswith("msa_transition."):
        converted = reprefix(key, "msa_transition.", "msa_transition.", MSA_TRANSITION_RULES)
        return [(cast(str, converted), tensor)]
    if key.startswith("outer_product_mean."):
        converted = reprefix(key, "outer_product_mean.", "outer_product_mean.", OUTER_PRODUCT_MEAN_RULES)
        return [(cast(str, converted), tensor)]
    if key.startswith("pairformer_layer."):
        inner = convert_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{new_key}", value) for new_key, value in inner]
    if key.startswith(("tri_mul_out.", "tri_mul_in.", "tri_att_start.", "tri_att_end.", "z_transition.")):
        return [
            (f"triangle_pair_update.{new_key}", value) for new_key, value in convert_msa_direct_triangle(key, tensor)
        ]
    raise KeyError(f"unmapped MSA layer key: {key}")


def convert_msa_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"layers.{index}.{new_key}", value) for new_key, value in convert_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


def convert_keys(key: str, rules: list[Rule]) -> str | None:
    r"""Apply the first matching rename/drop rule to ``key``."""

    for pattern, replacement in rules:
        if pattern.search(key):
            return None if replacement is None else pattern.sub(replacement, key)
    return key


def apply_converter(key: str, converter: Converter) -> str | None:
    if callable(converter):
        return converter(key)
    return convert_keys(key, converter)


def reprefix(key: str, old_prefix: str, new_prefix: str, converter: Converter) -> str | None:
    converted = apply_converter(key[len(old_prefix) :], converter)
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
