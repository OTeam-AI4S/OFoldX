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
Boltz-2 checkpoint conversion.

OFoldX modules use clean, idiomatic parameter names; the public ``boltz2_conf.ckpt`` uses Boltz's
own names. This module owns both the released Boltz-2 source-format rules and the strict target routes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import RequestRef, fail
from ..boltz1.convert import is_tied_output_projection
from ..convert import (
    PAIR_ONLY_LAYER_RULES,
    Rule,
    cast_floating_state,
    compile_rules,
    convert_keys,
    convert_msa_module,
    convert_rule_tensor,
    pack_conditioned_adaptive_norms,
    reprefix_indexed,
    split_fused_triangle_projection,
    transition_index_rules,
    transition_rules,
    triangle_attention_rules,
    triangle_multiplication_rules,
    unpack_gate_value_projection,
)
from ..vocabulary import FOLDING_SOURCE_TOKEN_INDEX, folding_word_embeddings
from .config import Boltz2AffinityConfig, Boltz2Config
from .model import Boltz2AffinityModel, Boltz2Model
from .pipeline import Boltz2AffinityPipeline, Boltz2AffinityScorePipeline, Boltz2Pipeline
from .processor import Boltz2AffinityProcessor, Boltz2Processor

BOLTZ_PAIRWISE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^dim_pairwise_init_proj\.0\.", "input_norm."),
        (r"^dim_pairwise_init_proj\.1\.", "input_projection."),
        *transition_index_rules(),
    ]
)


BOLTZ_SINGLE_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^norm_single\.", "single_norm."),
        (r"^single_embed\.", "single_projection."),
        (r"^fourier_embed\.proj\.", "noise_embedding.projection."),
        (r"^norm_fourier\.", "noise_norm."),
        (r"^fourier_to_single\.", "noise_projection."),
        *transition_index_rules(),
    ]
)


def convert_boltz_diffusion_conditioning(key: str, pairwise_rules: list[Rule]) -> str | None:
    if key.startswith("pairwise_conditioner."):
        inner = convert_keys(key[len("pairwise_conditioner.") :], pairwise_rules)
        return "pairwise_conditioner." + inner if inner is not None else None
    for boltz, norm_house, projection_house in (
        (
            "atom_enc_proj_z.",
            "atom_encoder_pair_bias_projection.pair_norms.",
            "atom_encoder_pair_bias_projection.bias_projections.",
        ),
        (
            "atom_dec_proj_z.",
            "atom_decoder_pair_bias_projection.pair_norms.",
            "atom_decoder_pair_bias_projection.bias_projections.",
        ),
        (
            "token_trans_proj_z.",
            "token_encoder_pair_bias_projection.pair_norms.",
            "token_encoder_pair_bias_projection.bias_projections.",
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


def convert_boltz_score_model(
    key: str,
    single_rules: list[Rule],
    block_rules: list[Rule],
    atom_encoder: Callable[[str], str | None],
    atom_decoder: Callable[[str], str | None],
) -> str | None:
    if key.startswith("single_conditioner."):
        inner = convert_keys(key[len("single_conditioner.") :], single_rules)
        return "single_conditioner." + inner if inner is not None else None
    if key.startswith("atom_attention_encoder."):
        inner = atom_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_encoder." + inner
    if key.startswith("atom_attention_decoder."):
        inner = atom_decoder(key[len("atom_attention_decoder.") :])
        return None if inner is None else "atom_decoder." + inner
    if key.startswith("token_transformer.layers."):
        return reprefix_indexed(
            key,
            "token_transformer.layers.",
            "token_encoder.blocks.",
            block_rules,
        )
    if key.startswith("s_to_a_linear.0."):
        return key.replace("s_to_a_linear.0.", "single_to_token_norm.", 1)
    if key.startswith("s_to_a_linear.1."):
        return key.replace("s_to_a_linear.1.", "single_to_token_projection.", 1)
    if key.startswith("a_norm."):
        return key.replace("a_norm.", "token_norm.", 1)
    return key


def _convert_vocabulary_axes(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Regenerate the residue-type, MSA-profile, MSA one-hot and template residue-type input columns from the 33-class
    upstream alphabet onto the 72 canonical classes; every other column keeps its place.
    """

    size = len(FOLDING_SOURCE_TOKEN_INDEX)
    for key in (
        "input_embeddings.residue_type_projection.weight",
        "input_embeddings.msa_profile_projection.weight",
        "msa_trunk.msa_projection.weight",
    ):
        if key in state_dict:
            tensor = state_dict[key]
            (tokens,) = folding_word_embeddings(tensor[..., :size], dim=-1)
            state_dict[key] = torch.cat((tokens, tensor[..., size:]), dim=-1)
    key = "template_embedder.template_projection.weight"
    if key in state_dict:
        tensor = state_dict[key]
        start = tensor.shape[-1] - 2 * size
        left, right = folding_word_embeddings(tensor[..., start : start + size], tensor[..., start + size :], dim=-1)
        state_dict[key] = torch.cat((tensor[..., :start], left, right), dim=-1)
    return state_dict


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
        (r"^post_lnorm\.", "post_norm."),
    ]
)


DISTOGRAM_RULES: list[Rule] = compile_rules([(r"^distogram\.", "projection.")])


B_FACTOR_RULES: list[Rule] = compile_rules([(r"^bfactor\.", "")])


ATOM_ENCODER_RULES: list[Rule] = compile_rules(
    [
        (r"^embed_atom_features\.", "atom_feature_projection."),
        (r"^embed_atompair_ref_pos\.", "atom_pair_encoder.distance_vector_projection."),
        (r"^embed_atompair_ref_dist\.", "atom_pair_encoder.inverse_distance_projection."),
        (r"^embed_atompair_mask\.", "atom_pair_encoder.distance_mask_projection."),
        (r"^c_to_p_trans_q\.1\.", "atom_pair_encoder.left_atom_projection."),
        (r"^c_to_p_trans_k\.1\.", "atom_pair_encoder.right_atom_projection."),
        (r"^p_mlp\.1\.", "atom_pair_encoder.feed_forward.input_projection."),
        (r"^p_mlp\.3\.", "atom_pair_encoder.feed_forward.intermediate_projections.0."),
        (r"^p_mlp\.5\.", "atom_pair_encoder.feed_forward.output_projection."),
        (r"^s_to_c_trans\.0\.", "trunk_conditioning.single_norm."),
        (r"^s_to_c_trans\.1\.", "trunk_conditioning.single_projection."),
        (r"^z_to_p_trans\.0\.", "trunk_conditioning.pair_norm."),
        (r"^z_to_p_trans\.1\.", "trunk_conditioning.pair_projection."),
    ]
)

# Boltz embeds key-minus-query reference offsets, OFoldX query-minus-key ones; the weight changes sign.
_KEY_MINUS_QUERY_OFFSET_WEIGHT = "embed_atompair_ref_pos.weight"
_INPUT_ATOM_LAYER_PREFIX = "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers."


def _convert_atom_encoder(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""Convert one prefix-stripped Boltz ``AtomEncoder`` key, flipping the reference-offset weight."""

    new_key = convert_keys(key, ATOM_ENCODER_RULES)
    if new_key is None:
        return []
    return [(new_key, -tensor if key == _KEY_MINUS_QUERY_OFFSET_WEIGHT else tensor)]


def _count_layers(state_dict: Mapping[str, Tensor], prefix: str) -> int:
    r"""Return the number of indexed layers under ``prefix`` in a released state dict."""

    return len({key[len(prefix) :].partition(".")[0] for key in state_dict if key.startswith(prefix)})


def _convert_atom_attention_encoder(key: str) -> str | None:
    if key.startswith("atom_encoder.diffusion_transformer.layers."):
        return reprefix_indexed(
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
        return reprefix_indexed(
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


CONTACT_CONDITIONING_RULES: list[Rule] = compile_rules(
    [
        (r"^fourier_embedding\.proj\.", "threshold_fourier.projection."),
        (r"^encoder\.", "constraint_encoder."),
        (r"^encoding_unspecified$", "unspecified_bias"),
        (r"^encoding_unselected$", "unselected_bias"),
    ]
)


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
        (r"^s_post_norm\.", "single_post_norm."),
        *triangle_multiplication_rules("tri_mul_out", "pair_update.triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "pair_update.triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "pair_update.triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "pair_update.triangle_attention_ending_node"),
        *transition_rules("transition_s", "single_transition"),
        *transition_rules("transition_z", "pair_update.transition"),
    ]
)


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
            (f"pair_update_stack.blocks.{index}.{nk}", nv)
            for nk, nv in convert_rule_tensor(remainder, tensor, PAIR_ONLY_LAYER_RULES)
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
        (
            r"^atom_attention_encoder\.atom_to_token_trans\.0\.",
            "atom_encoder.token_projection.",
        ),
    ]
)

_ATTN_OLD = "atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
_ATTN_NEW = "atom_encoder.atom_transformer.transformer.blocks."


def convert_input_embedder_key(key: str, tensor: Tensor, *, num_atom_encoder_blocks: int) -> list[tuple[str, Tensor]]:
    r"""
    Route a Boltz ``input_embedder.*`` (prefix-stripped) key to its OFoldX parameters.

    The released input embedder normalizes the atom pairs once and projects them to every atom-encoder
    block's heads at once; OFoldX keeps one normalization and one head projection per block. The shared
    normalization is copied to every block, and the projection rows split block-major.

    Args:
        key: An ``input_embedder``-relative Boltz key.
        tensor: The associated weight tensor.
        num_atom_encoder_blocks: Number of atom-encoder Transformer blocks of the checkpoint.

    Returns:
        ``(key, tensor)`` pairs under the OFoldX names; empty when dropped.
    """

    if key.startswith("atom_encoder."):
        return _convert_atom_encoder(key[len("atom_encoder.") :], tensor)
    if key.startswith("atom_enc_proj_z.0."):
        name = key[len("atom_enc_proj_z.0.") :]
        return [
            (f"atom_pair_bias_projection.pair_norms.{block}.{name}", tensor.clone())
            for block in range(num_atom_encoder_blocks)
        ]
    if key == "atom_enc_proj_z.1.weight":
        rows = tensor.reshape(num_atom_encoder_blocks, -1, tensor.shape[-1])
        return [
            (f"atom_pair_bias_projection.bias_projections.{block}.weight", rows[block].clone())
            for block in range(num_atom_encoder_blocks)
        ]
    if key.startswith(_ATTN_OLD):
        new_key = reprefix_indexed(key, _ATTN_OLD, _ATTN_NEW, CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES)
    else:
        new_key = convert_keys(key, INPUT_EMBEDDER_FLAT_RULES)
    return [] if new_key is None else [(new_key, tensor)]


_AFFINITY_PREFIXES = ("affinity_module1.", "affinity_module2.")

# Released tensors the model computes from its configuration: source key -> model buffer.
_COMPUTED_BUFFERS = {
    "confidence_module.boundaries": "confidence_head.boundaries",
    "affinity_module1.boundaries": "affinity_heads.0.boundaries",
    "affinity_module2.boundaries": "affinity_heads.1.boundaries",
}


CONFIDENCE_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "distance_embedding."),
        (r"^s_inputs_norm\.", "input_single_norm."),
        (r"^s_norm\.", "single_norm."),
        (r"^z_norm\.", "pair_norm."),
        (r"^s_input_to_s\.", "input_single_projection."),
        (r"^s_to_z_transpose\.", "pair_right_projection."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^s_to_z_prod_out\.", "single_to_pair_product_out."),
        (r"^s_to_z\.", "pair_left_projection."),
        (r"^rel_pos\.linear_layer\.", "relative_position_embedder.output_projection."),
        (r"^token_bonds_type\.", "token_bond_type_embedding."),
        (r"^token_bonds\.", "token_bond_projection."),
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

    Resolved-class projection rows are exchanged from source ``[resolved,
    unresolved]`` to canonical ``[unresolved, resolved]`` order.

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
            (f"pairformer_stack.blocks.{index}.{nk}", nv)
            for nk, nv in convert_rule_tensor(remainder, tensor, PAIRFORMER_LAYER_RULES, "pair_update.")
        ]
    if key.startswith("confidence_heads."):
        new = convert_keys(key[len("confidence_heads.") :], CONFIDENCE_HEAD_RULES)
        if new is not None and new.startswith("resolved_projection."):
            tensor = tensor.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
        return [("output_head." + new, tensor)] if new is not None else []
    new = convert_keys(key, CONFIDENCE_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


AFFINITY_FLAT_RULES: list[Rule] = compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "distance_embedding."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^z_norm\.", "pair_norm."),
        (r"^z_linear\.", "pair_projection."),
    ]
)
AFFINITY_HEAD_RULES: list[Rule] = compile_rules(
    [
        (r"^affinity_out_mlp\.0\.", "pair_feature_feed_forward.input_projection."),
        (r"^affinity_out_mlp\.2\.", "pair_feature_feed_forward.output_projection."),
        (r"^to_affinity_pred_value\.0\.", "affinity_value_feed_forward.input_projection."),
        (r"^to_affinity_pred_value\.2\.", "affinity_value_feed_forward.intermediate_projections.0."),
        (r"^to_affinity_pred_value\.4\.", "affinity_value_feed_forward.output_projection."),
        (r"^to_affinity_pred_score\.0\.", "affinity_score_feed_forward.input_projection."),
        (r"^to_affinity_pred_score\.2\.", "affinity_score_feed_forward.intermediate_projections.0."),
        (r"^to_affinity_pred_score\.4\.", "affinity_score_feed_forward.output_projection."),
        (r"^to_affinity_logits_binary\.", "binder_logit_projection."),
    ]
)
AFFINITY_PAIR_UPDATE_BLOCK_RULES: list[Rule] = compile_rules(
    [
        *triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *transition_rules("transition_z", "transition"),
    ]
)


def convert_affinity_pair_update_block(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one upstream affinity Pairformer layer into a triangle pair-update block.

    The affinity head uses ``TrianglePairUpdateStack``, whose blocks own one pair ``transition``.
    """

    split = split_fused_triangle_projection(key, tensor)
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
        new = convert_keys(key[len("pairwise_conditioner.") :], BOLTZ_PAIRWISE_CONDITIONING_RULES)
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

    A released copy of a tensor the model computes from its configuration is not returned;
    [`convert_boltz2_affinity_model`][ofoldx.models.boltz2.convert.convert_boltz2_affinity_model] checks it
    instead.

    Args:
        state_dict: A full Boltz-2 affinity checkpoint ``state_dict``.

    Returns:
        Converted affinity head tensors under ``affinity_heads.*``.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in _COMPUTED_BUFFERS:
            continue
        for index, module in enumerate(("affinity_module1", "affinity_module2")):
            prefix = module + "."
            if key.startswith(prefix):
                entries = convert_affinity_module(key[len(prefix) :], value)
                if not entries:
                    fail("value_invalid", RequestRef("checkpoint", key))
                for packed_key, packed_value in entries:
                    for new_key, new_value in unpack_gate_value_projection(packed_key, packed_value):
                        converted[f"affinity_heads.{index}.{new_key}"] = new_value
                break
    return converted


def convert_trunk_key(key: str, tensor: Tensor, *, num_atom_encoder_blocks: int) -> list[tuple[str, Tensor]]:
    r"""
    Map one Boltz checkpoint key to its ``Boltz2Model`` trunk param(s); ``[]`` drops non-trunk keys.

    Routes the key to the per-subsystem converter matching its prefix (input embedder, template, MSA,
    Pairformer, contact conditioning, distogram, diffusion, score model, confidence, B-factor) or the top-level
    trunk rename table.

    Args:
        key: A full Boltz checkpoint key.
        tensor: The associated weight tensor.
        num_atom_encoder_blocks: Number of input-embedder atom-encoder Transformer blocks of the checkpoint.

    Returns:
        A list of ``(key, tensor)`` pairs under the ``Boltz2Model`` param names; empty for non-trunk keys.
    """

    if key.startswith("input_embedder."):
        inner = convert_input_embedder_key(
            key[len("input_embedder.") :],
            tensor,
            num_atom_encoder_blocks=num_atom_encoder_blocks,
        )
        return [("input_embeddings." + nk, nv) for nk, nv in inner]
    if key.startswith("template_module."):
        inner = convert_template_module(key[len("template_module.") :], tensor)
        return [("template_embedder." + nk, nv) for nk, nv in inner]
    if key.startswith("msa_module."):
        inner = convert_msa_module(key[len("msa_module.") :], tensor)
        return [("msa_trunk." + nk, nv) for nk, nv in inner]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        inner = convert_rule_tensor(remainder, tensor, PAIRFORMER_LAYER_RULES, "pair_update.")
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in inner]
    if key.startswith("contact_conditioning."):
        new = convert_keys(key[len("contact_conditioning.") :], CONTACT_CONDITIONING_RULES)
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("distogram_module."):
        new = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        # The source projects a symmetrized pair; the head symmetrizes affine logits.
        return (
            [("distogram_head." + new, tensor / 2 if new == "projection.bias" else tensor)] if new is not None else []
        )
    if key.startswith("diffusion_conditioning.atom_encoder."):
        inner = _convert_atom_encoder(key[len("diffusion_conditioning.atom_encoder.") :], tensor)
        return [("denoiser_conditioning." + nk, nv) for nk, nv in inner]
    if key.startswith("diffusion_conditioning."):
        new = convert_boltz_diffusion_conditioning(
            key[len("diffusion_conditioning.") :], BOLTZ_PAIRWISE_CONDITIONING_RULES
        )
        return [("denoiser_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = convert_boltz_score_model(
            key[len("structure_module.score_model.") :],
            BOLTZ_SINGLE_CONDITIONING_RULES,
            CONDITIONED_TRANSFORMER_ENCODER_BLOCK_RULES,
            _convert_atom_attention_encoder,
            _convert_atom_attention_decoder,
        )
        return [("coordinate_diffusion.denoiser." + new, tensor)] if new is not None else []
    if key.startswith("confidence_module."):
        inner = convert_confidence_module(key[len("confidence_module.") :], tensor)
        return [("confidence_head." + nk, nv) for nk, nv in inner]
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], B_FACTOR_RULES)
        return [("b_factor_head." + new, tensor)] if new is not None else []
    if key in TRUNK_FLAT:
        return [(TRUNK_FLAT[key], tensor)]
    return []


def convert_trunk(state_dict: Mapping[str, torch.Tensor], *, allow_affinity: bool = False) -> dict[str, torch.Tensor]:
    r"""
    Convert a full Boltz-2 ``state_dict`` to the ``Boltz2Model`` parameter tree.

    Applies [`convert_trunk_key`][ofoldx.models.boltz2.convert.convert_trunk_key] to every entry, then packs
    the AdaLN scale/shift projections so the result ``strict``-loads into the OFoldX modules. Unknown keys fail;
    ``allow_affinity`` skips only the two exact affinity-module prefixes handled by the affinity converter. A
    released copy of a tensor the model computes from its configuration is not returned; the model converters
    check it instead.

    Args:
        state_dict: A full Boltz-2 checkpoint ``state_dict``.
        allow_affinity: Whether the released affinity-module prefixes may be routed by a separate converter.

    Returns:
        The converted trunk ``state_dict`` under the ``Boltz2Model`` param names.
    """

    converted: dict[str, torch.Tensor] = {}
    num_atom_encoder_blocks = _count_layers(state_dict, _INPUT_ATOM_LAYER_PREFIX)
    for key, value in state_dict.items():
        if key in _COMPUTED_BUFFERS:
            continue
        entries = convert_trunk_key(key, value, num_atom_encoder_blocks=num_atom_encoder_blocks)
        if not entries:
            if is_tied_output_projection(key):
                continue
            if allow_affinity and key.startswith(_AFFINITY_PREFIXES):
                continue
            fail("value_invalid", RequestRef("checkpoint", key))
        for packed_key, packed_value in entries:
            for new_key, new_value in unpack_gate_value_projection(packed_key, packed_value):
                converted[new_key] = new_value
    pack_conditioned_adaptive_norms(converted)
    return _convert_vocabulary_axes(converted)


def boltz2_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    affinity: bool = False,
) -> Boltz2Config:
    r"""
    Reconstruct a released Boltz-2 architecture and its independent training equations.
    """

    hyperparameters: Mapping[str, Any] = checkpoint["hyper_parameters"]
    state: Mapping[str, Tensor] = checkpoint["state_dict"]
    training_args: Mapping[str, Any] = hyperparameters["training_args"]
    embedder_args: Mapping[str, Any] = hyperparameters["embedder_args"]
    msa_args: Mapping[str, Any] = hyperparameters["msa_args"]
    pairformer_args: Mapping[str, Any] = hyperparameters["pairformer_args"]
    score_model_args: Mapping[str, Any] = hyperparameters["score_model_args"]
    diffusion_process_args: Mapping[str, Any] = hyperparameters["diffusion_process_args"]
    confidence_model_args: Mapping[str, Any] = hyperparameters["confidence_model_args"]

    config: Boltz2Config = Boltz2AffinityConfig() if affinity else Boltz2Config()
    config.single_size = hyperparameters["token_s"]
    config.pair_size = hyperparameters["token_z"]
    config.atom_hidden_size = hyperparameters["atom_s"]
    config.atom_pair_hidden_size = hyperparameters["atom_z"]
    config.atom_feature_size = hyperparameters["atom_feature_dim"]
    config.num_distogram_bins = hyperparameters["num_bins"]
    config.fourier_embedding_size = score_model_args["dim_fourier"]
    config.num_recycles = training_args["recycling_steps"] + 1
    config.use_templates = hyperparameters["use_templates"]
    config.predict_b_factor = hyperparameters["predict_bfactor"]
    config.min_conditioning_cutoff = hyperparameters["conditioning_cutoff_min"]
    config.max_conditioning_cutoff = hyperparameters["conditioning_cutoff_max"]

    config.input_embeddings.num_atom_encoder_blocks = embedder_args["atom_encoder_depth"]
    config.input_embeddings.num_atom_encoder_attention_heads = embedder_args["atom_encoder_heads"]
    config.input_embeddings.add_method_conditioning = "input_embedder.method_conditioning_init.weight" in state
    config.input_embeddings.add_modified_flag = "input_embedder.modified_conditioning_init.weight" in state
    config.input_embeddings.add_cyclic_flag = "input_embedder.cyclic_conditioning_init.weight" in state
    config.input_embeddings.add_mol_type_feat = "input_embedder.mol_type_conditioning_init.weight" in state

    config.trunk.num_pairformer_blocks = pairformer_args["num_blocks"]
    config.trunk.num_pair_attention_heads = pairformer_args["num_heads"]
    config.trunk.num_msa_blocks = msa_args["msa_blocks"]
    config.trunk.msa_size = msa_args["msa_s"]
    config.trunk.msa_triangle_attention_head_size = msa_args.get("pairwise_head_width", 32)
    config.trunk.num_msa_triangle_attention_heads = msa_args.get("pairwise_num_heads", 4)
    config.trunk.triangle_attention_head_size = pairformer_args.get("pairwise_head_width", 32)
    config.trunk.num_triangle_attention_heads = pairformer_args.get("pairwise_num_heads", 4)
    config.trunk.use_paired_feature = msa_args["use_paired_feature"]
    config.trunk.pair_dropout = pairformer_args["dropout"]
    config.trunk.msa_dropout = msa_args["msa_dropout"]
    config.trunk.msa_pair_dropout = msa_args["z_dropout"]
    config.trunk.post_layer_norm = pairformer_args.get("post_layer_norm", False)

    template_args = hyperparameters.get("template_args")
    if template_args is not None:
        config.template.hidden_size = template_args["template_dim"]
        config.template.num_blocks = template_args["template_blocks"]
        config.template.num_distogram_bins = template_args.get("num_bins", 38)
        config.template.dropout = template_args.get("dropout", 0.25)
        config.template.min_distance = template_args.get("min_dist", 3.25)
        config.template.max_distance = template_args.get("max_dist", 50.75)
        config.template.triangle_attention_head_size = template_args.get("pairwise_head_width", 32)
        config.template.num_triangle_attention_heads = template_args.get("pairwise_num_heads", 4)

    config.denoiser.num_token_encoder_blocks = score_model_args["token_transformer_depth"]
    config.denoiser.num_token_encoder_attention_heads = score_model_args["token_transformer_heads"]
    config.denoiser.num_atom_encoder_blocks = score_model_args["atom_encoder_depth"]
    config.denoiser.num_atom_decoder_blocks = score_model_args["atom_decoder_depth"]
    config.denoiser.num_atom_attention_heads = score_model_args["atom_encoder_heads"]
    config.denoiser.post_layer_norm = score_model_args.get("transformer_post_ln", False)
    config.denoiser.atom_hidden_size = config.atom_hidden_size
    config.denoiser.atom_pair_hidden_size = config.atom_pair_hidden_size
    config.denoiser.num_query_atoms = hyperparameters["atoms_per_window_queries"]
    config.denoiser.num_key_atoms = hyperparameters["atoms_per_window_keys"]
    config.denoiser.num_conditioning_transitions = score_model_args["conditioning_transition_layers"]
    config.denoiser.sigma_data = score_model_args["sigma_data"]
    config.denoiser.alignment_reverse_diff = diffusion_process_args["alignment_reverse_diff"]

    confidence_pairformer: Mapping[str, Any] = confidence_model_args["pairformer_args"]
    confidence_args: Mapping[str, Any] = confidence_model_args["confidence_args"]
    config.confidence.num_pairformer_blocks = confidence_pairformer["num_blocks"]
    config.confidence.num_pair_attention_heads = confidence_pairformer["num_heads"]
    config.confidence.triangle_attention_head_size = confidence_pairformer.get("pairwise_head_width", 32)
    config.confidence.num_triangle_attention_heads = confidence_pairformer.get("pairwise_num_heads", 4)
    config.confidence.pair_dropout = confidence_pairformer["dropout"]
    config.confidence.post_layer_norm = confidence_pairformer.get("post_layer_norm", False)
    config.confidence.normalize_single_input = not confidence_pairformer.get("no_update_s", False)
    config.confidence.num_distogram_bins = confidence_model_args["num_dist_bins"]
    config.confidence.max_distance = confidence_model_args["max_dist"]
    config.confidence.num_plddt_bins = confidence_args["num_plddt_bins"]
    config.confidence.num_pae_bins = confidence_args["num_pae_bins"]
    config.confidence.num_pde_bins = confidence_args["num_pde_bins"]
    return config


def convert_boltz2_model(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2Model:
    r"""
    Convert a released Boltz-2 structure-prediction checkpoint.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = boltz2_config_from_checkpoint(checkpoint) if config is None else config
    state = checkpoint["state_dict"]
    converted = cast_floating_state(convert_trunk(state), dtype)
    with torch.device("meta"):
        model = Boltz2Model(config)
    _check_computed_buffers(model, state)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def convert_boltz2_affinity_model(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2AffinityConfig | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2AffinityModel:
    r"""
    Convert the released two-head Boltz-2 affinity checkpoint.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    config = (
        cast(Boltz2AffinityConfig, boltz2_config_from_checkpoint(checkpoint, affinity=True))
        if config is None
        else config
    )
    state = checkpoint["state_dict"]
    converted = convert_trunk(state, allow_affinity=True)
    converted.update(convert_affinity_modules(state))
    converted = cast_floating_state(converted, dtype)
    with torch.device("meta"):
        model = Boltz2AffinityModel(config)
    _check_computed_buffers(model, state)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def _check_computed_buffers(model: Boltz2Model | Boltz2AffinityModel, state_dict: Mapping[str, Tensor]) -> None:
    r"""
    Check a release's copies of the tensors the model computes from its configuration.

    The release serialized these tensors although its configuration determines them, so they are evidence
    about that configuration rather than state to load: a disagreement means the configuration does not
    describe the checkpoint.
    """

    for source_key, released in state_dict.items():
        buffer_name = _COMPUTED_BUFFERS.get(source_key)
        if buffer_name is not None and not torch.equal(released, model.get_buffer(buffer_name).to(released)):
            fail("value_invalid", RequestRef("checkpoint", source_key), disagrees_with=buffer_name)


def convert_boltz2_checkpoint(
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
    [`convert_boltz2_model`][ofoldx.models.boltz2.convert.convert_boltz2_model], pairs the
    model with the config-less [`Boltz2Processor`][], and writes a self-contained ``model.json`` /
    ``processor.json`` / safetensors repository marked with the exact [`Boltz2Pipeline`][].
    The Processor repository metadata records its floating-point dtype even though it has no
    family configuration.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository. No
    organization or token is assumed and nothing is published without an explicit ``repo_id``.
    """

    model = convert_boltz2_model(checkpoint_path, config=config, dtype=dtype)
    save_repository(
        Boltz2Pipeline(model=model, processor=Boltz2Processor(dtype=dtype)),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter=__name__ + ".convert_boltz2_checkpoint",
        options={"dtype": None if dtype is None else str(dtype)},
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_boltz2_affinity_checkpoint(
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

    [`Boltz2AffinityPipeline`][] composes independently deployable affinity-scoring
    and structure-prediction Pipelines. The saved repository contains these children
    in ``affinity/`` and ``structure/``, each with its own Pipeline marker, and a
    parent marker retaining the complete workflow identity.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the composite repository.
    """

    affinity_model = convert_boltz2_affinity_model(
        affinity_checkpoint_path,
        config=affinity_config,
        dtype=dtype,
    )
    structure_model = convert_boltz2_model(
        structure_checkpoint_path,
        config=structure_config,
        dtype=dtype,
    )
    save_repository(
        Boltz2AffinityPipeline(
            affinity_scorer=Boltz2AffinityScorePipeline(
                model=affinity_model,
                processor=Boltz2AffinityProcessor(dtype=dtype),
            ),
            folder=Boltz2Pipeline(
                model=structure_model,
                processor=Boltz2Processor(dtype=dtype),
            ),
        ),
        output_dir,
        sources={"affinity": affinity_checkpoint_path, "structure": structure_checkpoint_path},
        converter=__name__ + ".convert_boltz2_affinity_checkpoint",
        options={"dtype": None if dtype is None else str(dtype)},
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
        convert_boltz2_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
    else:
        convert_boltz2_affinity_checkpoint(
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
