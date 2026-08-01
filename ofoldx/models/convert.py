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

r"""Checkpoint tensor transforms shared by family converters."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import cast

import torch
from torch import Tensor

from ..errors import SourceRef, fail


def pack_conditioned_adaptive_norms(state_dict: dict[str, Tensor]) -> None:
    r"""Pack scale then shift weights into the conditioned-normalization state layout.

    Only the scale projection has a bias; it becomes ``conditioning_scale_bias``.
    Required source entries retain their ordinary dictionary lookup semantics.
    """

    scale_weight_suffix = ".conditioned_norm.gate_projection.weight"
    for scale_weight_key in tuple(key for key in state_dict if key.endswith(scale_weight_suffix)):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state_dict.pop(scale_weight_key)
        scale_bias = state_dict.pop(prefix + "gate_projection.bias")
        shift_weight = state_dict.pop(prefix + "shift_projection.weight")
        state_dict[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state_dict[prefix + "conditioning_scale_bias"] = scale_bias


def cast_floating_state(state_dict: Mapping[str, Tensor], dtype: torch.dtype | None) -> dict[str, Tensor]:
    r"""Cast requested floating state tensors, retaining all other tensor identities."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


Rule = tuple[re.Pattern[str], str | None]
Converter = list[Rule] | Callable[[str], str | None]


def token_transformer_reference_key(key: str) -> str:
    r"""Map a pair-biased token encoder to its released block parameter names."""

    rules = (
        (r"\.pair_norms\.(\d+)\.", r".blocks.\1.attention_pair_bias.layernorm_z."),
        (r"\.bias_projections\.(\d+)\.", r".blocks.\1.attention_pair_bias.linear_nobias_z."),
        (r"\.blocks\.(\d+)\.transition\.", r".blocks.\1.conditioned_transition_block."),
        (r"\.blocks\.(\d+)\.conditioned_norm\.", r".blocks.\1.attention_pair_bias.layernorm_a."),
        (r"\.blocks\.(\d+)\.attention\.", r".blocks.\1.attention_pair_bias.attention."),
        (r"\.blocks\.(\d+)\.output_gate_projection\.", r".blocks.\1.attention_pair_bias.linear_a_last."),
    )
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def compile_rules(pairs: Iterable[tuple[str, str | None]]) -> list[Rule]:
    r"""
    Compile ordered string regex pairs into conversion rules.
    """

    return [(re.compile(pattern), replacement) for pattern, replacement in pairs]


def convert_keys(key: str, rules: list[Rule]) -> str | None:
    r"""
    Apply the first matching rename/drop rule to ``key``.
    """

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


def reprefix_indexed(key: str, old_prefix: str, new_prefix: str, sub: Converter) -> str | None:
    r"""
    Like ``reprefix`` but preserves a ``<index>.`` segment that follows the prefix.

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


def transition_index_rules() -> list[tuple[str, str | None]]:
    return [
        (r"^transitions\.(\d+)\.norm\.", r"transitions.\1.input_norm."),
        (r"^transitions\.(\d+)\.fc1\.", r"transitions.\1.feed_forward.gate_projection."),
        (r"^transitions\.(\d+)\.fc2\.", r"transitions.\1.feed_forward.value_projection."),
        (r"^transitions\.(\d+)\.fc3\.", r"transitions.\1.feed_forward.output_projection."),
    ]


def triangle_attention_rules(source_prefix: str, target_prefix: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source_prefix}\.layer_norm\.", f"{target_prefix}.pair_norm."),
        (rf"^{source_prefix}\.linear\.", f"{target_prefix}.bias_projection."),
        (rf"^{source_prefix}\.mha\.linear_q\.", f"{target_prefix}.attention.projection.query_projection."),
        (rf"^{source_prefix}\.mha\.linear_k\.", f"{target_prefix}.attention.projection.key_projection."),
        (rf"^{source_prefix}\.mha\.linear_v\.", f"{target_prefix}.attention.projection.value_projection."),
        (rf"^{source_prefix}\.mha\.linear_o\.", f"{target_prefix}.attention.projection.output_projection."),
        (rf"^{source_prefix}\.mha\.linear_g\.", f"{target_prefix}.attention.projection.gate_projection."),
    ]


def transition_rules(source_prefix: str, target_prefix: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source_prefix}\.norm\.", f"{target_prefix}.input_norm."),
        (rf"^{source_prefix}\.fc1\.", f"{target_prefix}.feed_forward.gate_projection."),
        (rf"^{source_prefix}\.fc2\.", f"{target_prefix}.feed_forward.value_projection."),
        (rf"^{source_prefix}\.fc3\.", f"{target_prefix}.feed_forward.output_projection."),
    ]


def triangle_multiplication_rules(source_prefix: str, target_prefix: str) -> list[tuple[str, str | None]]:
    return [
        (rf"^{source_prefix}\.norm_in\.", f"{target_prefix}.input_norm."),
        (rf"^{source_prefix}\.norm_out\.", f"{target_prefix}.output_norm."),
        (rf"^{source_prefix}\.p_out\.", f"{target_prefix}.output_projection."),
        (rf"^{source_prefix}\.g_out\.", f"{target_prefix}.gate_projection."),
    ]


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


def split_fused_triangle_projection(key: str, tensor: Tensor) -> list[tuple[str, Tensor]] | None:
    if key not in FUSED_TRIANGLE_PROJECTIONS:
        return None
    block, left, right = FUSED_TRIANGLE_PROJECTIONS[key]
    half = tensor.shape[0] // 2
    return [
        (f"{block}.{left}.weight", tensor[:half].clone()),
        (f"{block}.{right}.weight", tensor[half:].clone()),
    ]


def convert_rule_tensor(
    key: str, tensor: Tensor, rules: list[Rule], fused_prefix: str = ""
) -> list[tuple[str, Tensor]]:
    r"""Route fused triangle rows or apply the actual ordered rename/drop rules."""

    split = split_fused_triangle_projection(key, tensor)
    if split is not None:
        return [(fused_prefix + target, value) for target, value in split]
    target = convert_keys(key, rules)
    return [] if target is None else [(target, tensor)]


def unpack_gate_value_projection(
    key: str,
    tensor: Tensor,
    projection_names: tuple[str, str] = ("value_projection", "gate_projection"),
) -> list[tuple[str, Tensor]]:
    r"""Unpack the native dim-zero chunks into their actual ordered projections."""

    marker = ".gate_value_projection."
    if marker not in key:
        return [(key, tensor)]
    prefix, suffix = key.rsplit(marker, 1)
    first, second = tensor.chunk(2, dim=0)
    return [
        (f"{prefix}.{projection_names[0]}.{suffix}", first.clone()),
        (f"{prefix}.{projection_names[1]}.{suffix}", second.clone()),
    ]


PAIR_ONLY_LAYER_RULES: list[Rule] = compile_rules(
    [
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("transition_z", "transition"),
    ]
)


MSA_DIRECT_TRIANGLE_RULES: list[Rule] = compile_rules(
    [
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("z_transition", "transition"),
    ]
)


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


def convert_standard_msa_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
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
        inner = convert_rule_tensor(key[len("pairformer_layer.") :], tensor, PAIR_ONLY_LAYER_RULES)
        return [(f"triangle_pair_update.{new_key}", value) for new_key, value in inner]
    if key.startswith(("tri_mul_out.", "tri_mul_in.", "tri_att_start.", "tri_att_end.", "z_transition.")):
        return [
            (f"triangle_pair_update.{new_key}", value)
            for new_key, value in convert_rule_tensor(key, tensor, MSA_DIRECT_TRIANGLE_RULES)
        ]
    fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))


def convert_msa_module(
    key: str,
    tensor: Tensor,
    layer_converter: Callable[[str, Tensor], list[tuple[str, Tensor]]] = convert_standard_msa_layer,
) -> list[tuple[str, Tensor]]:
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"blocks.{index}.{new_key}", value) for new_key, value in layer_converter(remainder, tensor)]
    fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))


def atom_stack_reference_key(key: str) -> str:
    block = r".atom_transformer.diffusion_transformer.blocks.\1"
    block_path = r"\.atom_transformer\.transformer\.blocks\.(\d+)\."
    rules = [
        (r"\.atom_pair_bias_projection\.pair_norms\.(\d+)\.", block + r".attention_pair_bias.layernorm_z."),
        (
            r"\.atom_pair_bias_projection\.bias_projections\.(\d+)\.",
            block + r".attention_pair_bias.linear_nobias_z.",
        ),
        (block_path + r"transition\.", block + r".conditioned_transition_block."),
        (block_path + r"conditioned_norm\.", block + r".attention_pair_bias.layernorm_a."),
        (block_path + r"key_value_norm\.", block + r".attention_pair_bias.layernorm_kv."),
        (block_path + r"attention\.", block + r".attention_pair_bias.attention."),
        (block_path + r"output_gate_projection\.", block + r".attention_pair_bias.linear_a_last."),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def conditioned_norm_reference_key(key: str) -> str | tuple[str, ...]:
    if key.endswith(".conditioning_projection.weight"):
        base = key[: -len("conditioning_projection.weight")]
        return (base + "linear_s.weight", base + "linear_nobias_s.weight")
    if key.endswith(".conditioning_scale_bias"):
        base = key[: -len("conditioning_scale_bias")]
        return base + "linear_s.bias"
    for source, target in {
        ".hidden_norm.": ".layernorm_a.",
        ".conditioning_norm.": ".layernorm_s.",
        ".gate_projection.": ".linear_s.",
        ".shift_projection.": ".linear_nobias_s.",
    }.items():
        key = key.replace(source, target)
    return key


def attention_reference_key(key: str) -> str:
    for source, target in {
        ".pair_attention.": ".attention_pair_bias.",
        ".hidden_norm.": ".layernorm_a.",
        ".atom_norm.": ".layernorm_a.",
        ".key_value_norm.": ".layernorm_kv.",
        ".attention_pair_bias.pair_norm.": ".attention_pair_bias.layernorm_z.",
        ".attention_pair_bias.bias_projection.": ".attention_pair_bias.linear_nobias_z.",
        ".attention_pair_bias.conditioning_gate_projection.": ".attention_pair_bias.linear_a_last.",
    }.items():
        key = key.replace(source, target)
    return key


def attention_projection_reference_key(key: str) -> str:
    for source, target in {
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
    }.items():
        key = key.replace(source, target)
    return key


def translate_target_state(
    model: torch.nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], str | tuple[str, ...]],
    lookup: Callable[[Mapping[str, Tensor], str | tuple[str, ...]], Tensor | tuple[Tensor, ...]],
    transform: Callable[[str, Tensor | tuple[Tensor, ...]], Tensor],
) -> dict[str, Tensor]:
    r"""Translate one target state in registration order through its actual source operations."""

    translated = {}
    for model_key in model.state_dict():
        reference_key = key_fn(model_key)
        reference_value = lookup(reference_state, reference_key)
        translated[model_key] = transform(model_key, reference_value)
    return translated
