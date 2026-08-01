r"""
Checkpoint conversion for released BoltzGen structure design and inverse folding.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ...data.evidence import MSA_ALPHABET
from ...data.tokenization import TOKEN_ALPHABET
from ..boltz2.config import Boltz2AffinityConfig, Boltz2Config
from ..boltz2.convert import convert_boltz2_affinity_model, convert_boltz2_model
from ..boltz2.processor import Boltz2AffinityProcessor, Boltz2Processor
from .config import (
    BoltzGenConfidenceConfig,
    BoltzGenConfig,
    BoltzGenInverseFoldingConfig,
    BoltzGenMiniformerConfig,
    BoltzGenMiniformerResidueTypeConfig,
    BoltzGenResidueTypeConfig,
)
from .model import (
    BoltzGenConfidenceModel,
    BoltzGenInverseFoldingModel,
    BoltzGenMiniformerModel,
    BoltzGenMiniformerResidueTypeModel,
    BoltzGenModel,
    BoltzGenResidueTypeModel,
)
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor

Rule = tuple[re.Pattern[str], str | None]


Converter = list[Rule] | Callable[[str], str | None]

_SOURCE_TOKEN_INDEX = {
    "OTHER": 0,
    "-": 1,
    "A": 2,
    "R": 3,
    "N": 4,
    "D": 5,
    "C": 6,
    "Q": 7,
    "E": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "L": 12,
    "K": 13,
    "M": 14,
    "F": 15,
    "P": 16,
    "S": 17,
    "T": 18,
    "W": 19,
    "Y": 20,
    "V": 21,
    "X": 22,
    "RNA_A": 23,
    "RNA_G": 24,
    "RNA_C": 25,
    "RNA_U": 26,
    "RNA_N": 27,
    "DNA_A": 28,
    "DNA_G": 29,
    "DNA_C": 30,
    "DNA_T": 31,
    "DNA_N": 32,
}


def _convert_source_token_values(tensor: Tensor, alphabet: Sequence[str]) -> Tensor:
    old_size = len(_SOURCE_TOKEN_INDEX)
    converted = tensor.new_zeros(*tensor.shape[:-1], len(alphabet) + tensor.shape[-1] - old_size)
    for target, symbol in enumerate(alphabet):
        if symbol == "B":
            converted[..., target] = (tensor[..., _SOURCE_TOKEN_INDEX["D"]] + tensor[..., _SOURCE_TOKEN_INDEX["N"]]) / 2
        elif symbol == "Z":
            converted[..., target] = (tensor[..., _SOURCE_TOKEN_INDEX["E"]] + tensor[..., _SOURCE_TOKEN_INDEX["Q"]]) / 2
        elif symbol == "J":
            converted[..., target] = (tensor[..., _SOURCE_TOKEN_INDEX["I"]] + tensor[..., _SOURCE_TOKEN_INDEX["L"]]) / 2
        elif symbol == "U":
            converted[..., target] = tensor[..., _SOURCE_TOKEN_INDEX["C"]]
        elif symbol == "O":
            converted[..., target] = tensor[..., _SOURCE_TOKEN_INDEX["K"]]
        else:
            converted[..., target] = tensor[..., _SOURCE_TOKEN_INDEX[symbol]]
    converted[..., len(alphabet) :] = tensor[..., old_size:]
    return converted


def _convert_source_token_axis(tensor: Tensor, alphabet: Sequence[str], axis: int) -> Tensor:
    axis %= tensor.ndim
    converted = _convert_source_token_values(tensor.movedim(axis, -1), alphabet)
    return converted.movedim(-1, axis).contiguous()


def _convert_vocabulary_axes(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    structural_input_keys = (
        "input_embeddings.residue_type_projection.weight",
        "inverse_folding_decoder.residue_type_projection.weight",
    )
    structural_output_keys = (
        "coordinate_diffusion.denoiser.atom_attention_decoder.residue_type_projection.weight",
        "coordinate_diffusion.denoiser.atom_attention_decoder.residue_type_projection.bias",
        "inverse_folding_decoder.residue_type_head.weight",
    )
    for key in structural_input_keys:
        if key in state_dict:
            state_dict[key] = _convert_source_token_axis(state_dict[key], TOKEN_ALPHABET, -1)
    for key in structural_output_keys:
        if key in state_dict:
            state_dict[key] = _convert_source_token_axis(state_dict[key], TOKEN_ALPHABET, 0)
    for key in ("input_embeddings.msa_profile_projection.weight", "msa_trunk.msa_projection.weight"):
        if key in state_dict:
            state_dict[key] = _convert_source_token_axis(state_dict[key], MSA_ALPHABET, -1)
    template_key = "template_embedder.template_feature_projection.weight"
    if template_key in state_dict:
        tensor = state_dict[template_key]
        source_vocab_size = len(_SOURCE_TOKEN_INDEX)
        residue_start = tensor.shape[-1] - 2 * source_vocab_size
        state_dict[template_key] = torch.cat(
            (
                tensor[..., :residue_start],
                _convert_source_token_values(
                    tensor[..., residue_start : residue_start + source_vocab_size],
                    TOKEN_ALPHABET,
                ),
                _convert_source_token_values(
                    tensor[..., residue_start + source_vocab_size :],
                    TOKEN_ALPHABET,
                ),
            ),
            dim=-1,
        )
    return state_dict


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
        (r"^pair_bias_attn\.q_norm\.", "attention.projection.query_norm."),
        (r"^pair_bias_attn\.k_norm\.", "attention.projection.key_norm."),
        (r"^output_projection_linear\.", "output_gate_projection."),
        (r"^output_projection\.0\.", None),
        (r"^post_lnorm\.", "post_norm."),
    ]
)


DISTOGRAM_RULES: list[Rule] = compile_rules([(r"^distogram\.", "projection.")])


BFACTOR_RULES: list[Rule] = compile_rules([(r"^bfactor\.", "")])


ATOM_ENCODER_RULES: list[Rule] = compile_rules(
    [
        (r"^embed_atom_features\.", "atom_feature_projection."),
        (r"^embed_atompair_ref_pos\.", "pair_encoding.distance_vector_projection."),
        (r"^embed_atompair_ref_dist\.", "pair_encoding.inverse_distance_projection."),
        (r"^embed_atompair_mask\.", "pair_encoding.distance_mask_projection."),
        (r"^c_to_p_trans_q\.1\.", "pair_encoding.left_atom_projection."),
        (r"^c_to_p_trans_k\.1\.", "pair_encoding.right_atom_projection."),
        (r"^p_mlp\.1\.", "pair_encoding.feed_forward_input_projection."),
        (r"^p_mlp\.3\.", "pair_encoding.feed_forward_hidden_projection."),
        (r"^p_mlp\.5\.", "pair_encoding.feed_forward_output_projection."),
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
        (r"^token_s_to_tfmr_s\.", "input_projection."),
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
    if key.startswith("encoding_3d."):
        return key.replace("encoding_3d.", "position_encoding.", 1)
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
    if key.startswith("res_type_predictor."):
        return key.replace("res_type_predictor.", "residue_type_projection.", 1)
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
        (r"^s_post_norm\.", "single_post_norm."),
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


CONFIDENCE_KEY_RENAMES = {
    "boundaries": "boundaries",
    "dist_bin_pairwise_embed": "distance_embedding",
    "s_to_z": "single_to_pair_row",
    "s_to_z_transpose": "single_to_pair_col",
    "s_to_z_prod_in1": "single_to_pair_product_left",
    "s_to_z_prod_in2": "single_to_pair_product_right",
    "s_to_z_prod_out": "single_to_pair_product_out",
    "s_input_to_s": "input_single_projection",
    "s_inputs_norm": "input_single_norm",
    "s_norm": "single_norm",
    "z_norm": "pair_norm",
}


CONFIDENCE_OUTPUT_KEY_RENAMES = {
    "to_pae_logits": "pae_projection",
    "to_pae_intra_logits": "pae_intra_projection",
    "to_pae_inter_logits": "pae_inter_projection",
    "to_pde_logits": "pde_projection",
    "to_pde_intra_logits": "pde_intra_projection",
    "to_pde_inter_logits": "pde_inter_projection",
    "to_plddt_logits": "plddt_projection",
    "to_resolved_logits": "resolved_projection",
}


def convert_confidence_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Convert one ``confidence_module`` parameter onto the distinct confidence head.
    """

    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"confidence_head.pairformer_stack.blocks.{index}.{new_key}", value)
            for new_key, value in _convert_pairformer_layer(remainder, tensor)
        ]
    if key.startswith("confidence_heads."):
        inner = key[len("confidence_heads.") :]
        head, separator, remainder = inner.partition(".")
        renamed = CONFIDENCE_OUTPUT_KEY_RENAMES.get(head)
        if renamed is None:
            raise KeyError(f"unmapped confidence output key: {key}")
        suffix = separator + remainder if separator else ""
        return [(f"confidence_head.output_head.{renamed}{suffix}", tensor)]
    if key.startswith(("rel_pos.", "token_bonds.", "token_bonds_type.", "contact_conditioning.")):
        return []
    head, separator, remainder = key.partition(".")
    renamed = CONFIDENCE_KEY_RENAMES.get(head)
    if renamed is None:
        raise KeyError(f"unmapped confidence checkpoint key: {key}")
    suffix = separator + remainder if separator else ""
    return [(f"confidence_head.{renamed}{suffix}", tensor)]


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


def _convert_standard_msa_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
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

MINIFORMER_LAYER_RULES: list[Rule] = compile_rules(
    [
        (r"^pre_norm_s\.", "attention.hidden_norm."),
        (r"^attention\.proj_q\.", "attention.attention.projection.query_projection."),
        (r"^attention\.proj_k\.", "attention.attention.projection.key_projection."),
        (r"^attention\.proj_v\.", "attention.attention.projection.value_projection."),
        (r"^attention\.proj_g\.", "attention.attention.projection.gate_projection."),
        (r"^attention\.proj_o\.", "attention.attention.projection.output_projection."),
        (r"^attention\.proj_z\.0\.", "attention.pair_norm."),
        (r"^attention\.proj_z\.1\.", "attention.bias_projection."),
        (r"^s_to_z_1\.", "single_to_pair_left_projection."),
        (r"^s_to_z_2\.", "single_to_pair_right_projection."),
        (r"^s_post_norm\.", "single_post_norm."),
        *_transition_rules("transition_s", "single_transition"),
        *_transition_rules("transition_z", "pair_transition"),
    ]
)


def _convert_miniformer_triangle_key(key: str) -> str | None:
    if not key.startswith("triangular."):
        return None
    converted = "triangle_multiplication." + key.removeprefix("triangular.")
    for source, target in {
        ".norm_in.": ".input_norm.",
        ".p_in.": ".input_projection.",
        ".g_in.": ".input_gate_projection.",
        ".norm_out.": ".output_norm.",
        ".p_out.": ".output_projection.",
        ".g_out.": ".output_gate_projection.",
    }.items():
        converted = converted.replace(source, target)
    return converted


def convert_miniformer_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one compact Miniformer layer.

    Maps the compact triangle multiplication and applies the remaining Miniformer rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    triangle_key = _convert_miniformer_triangle_key(key)
    if triangle_key is not None:
        return [(triangle_key, tensor)]
    new_key = convert_keys(key, MINIFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MINIFORMER_PAIR_ONLY_LAYER_RULES: list[Rule] = compile_rules(
    [
        (r"^z_post_norm\.", "pair_post_norm."),
        *_transition_rules("transition_z", "pair_transition"),
    ]
)


def convert_miniformer_pair_only_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one compact pair-only Miniformer layer.

    Maps the compact triangle multiplication and applies the remaining pair-only rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    triangle_key = _convert_miniformer_triangle_key(key)
    if triangle_key is not None:
        return [(triangle_key, tensor)]
    new_key = convert_keys(key, MINIFORMER_PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


def convert_miniformer_msa_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one MSA layer whose pair update uses MiniformerNoSeq blocks.

    Routes MSA attention, transition, and outer-product weights by sub-block name;
    pair-only updates use the compact Miniformer converter.

    Args:
        key: Checkpoint key relative to the MSA-layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA sub-block.
    """

    if key.startswith("pair_weighted_averaging."):
        return [
            (
                cast(
                    str,
                    _reprefix(
                        key,
                        "pair_weighted_averaging.",
                        "pair_weighted_averaging.",
                        MSA_PAIR_WEIGHTED_AVERAGING_RULES,
                    ),
                ),
                tensor,
            )
        ]
    if key.startswith("msa_transition."):
        return [
            (
                cast(
                    str,
                    _reprefix(key, "msa_transition.", "msa_transition.", MSA_TRANSITION_RULES),
                ),
                tensor,
            )
        ]
    if key.startswith("outer_product_mean."):
        return [
            (
                cast(
                    str,
                    _reprefix(key, "outer_product_mean.", "outer_product_mean.", OUTER_PRODUCT_MEAN_RULES),
                ),
                tensor,
            )
        ]
    if key.startswith("pairformer_layer."):
        inner = convert_miniformer_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in inner]
    raise KeyError(f"unmapped Miniformer MSA layer key: {key}")


def convert_msa_module(
    key: str,
    tensor: Tensor,
    *,
    miniformer_blocks: bool = False,
) -> list[tuple[str, Tensor]]:
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
        return _convert_standard_msa_module(key, tensor)
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"blocks.{index}.{nk}", nv) for nk, nv in convert_miniformer_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


def convert_trunk_key(
    key: str,
    tensor: Tensor,
    *,
    miniformer_pairformer: bool = False,
    miniformer_msa: bool = False,
) -> list[tuple[str, Tensor]]:
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
        return [(f"msa_trunk.{new_key}", new_value) for new_key, new_value in inner]
    if key.startswith(("pairformer_stack.layers.", "pairformer_module.layers.")):
        prefix = (
            "pairformer_stack.layers." if key.startswith("pairformer_stack.layers.") else "pairformer_module.layers."
        )
        index, _, remainder = key[len(prefix) :].partition(".")
        converter = convert_miniformer_layer if miniformer_pairformer else _convert_pairformer_layer
        return [
            (f"trunk_stack.blocks.{index}.{new_key}", new_value) for new_key, new_value in converter(remainder, tensor)
        ]
    if key.startswith("distogram_module."):
        new = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = _convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("denoiser_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = _convert_score_model(key[len("structure_module.score_model.") :])
        return [("coordinate_diffusion.denoiser." + new, tensor)] if new is not None else []
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in TRUNK_FLAT:
        return [(TRUNK_FLAT[key], tensor)]
    return []


INPUT_EMBEDDER_CONDITIONING = {
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
        inner = convert_keys(key[len("atom_encoder.") :], ATOM_ENCODER_RULES)
        return None if inner is None else "atom_encoder." + inner
    if key.startswith("atom_attention_encoder."):
        inner = _convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    if key.startswith("atom_enc_proj_z.0."):
        return key.replace("atom_enc_proj_z.0.", "atom_pair_norm.", 1)
    if key.startswith("atom_enc_proj_z.1."):
        return key.replace("atom_enc_proj_z.1.", "atom_pair_bias_projection.", 1)
    head = key.split(".")[0]
    if head in INPUT_EMBEDDER_CONDITIONING:
        return INPUT_EMBEDDER_CONDITIONING[head] + key[len(head) :]
    return key


GNN_LAYER_RENAME = {
    "attn_weight_mlp": {
        "0": "attention.weight_feed_forward.input_projection",
        "2": "attention.weight_feed_forward.hidden_projections.0",
        "4": "attention.weight_feed_forward.output_projection",
    },
    "attn_value_mlp": {
        "0": "attention.value_feed_forward.input_projection",
        "2": "attention.value_feed_forward.hidden_projections.0",
        "4": "attention.value_feed_forward.output_projection",
    },
    "attn_output_linear": {
        "0": "attention.output_projection",
        "2": "attention_norm",
    },
    "attn_FFN": {
        "0": "node_feed_forward.input_projection",
        "2": "node_feed_forward.output_projection",
        "4": "node_norm",
    },
    "edge_FFN": {
        "0": "edge_feed_forward.input_projection",
        "2": "edge_feed_forward.output_projection",
        "4": "edge_norm",
    },
}


def _convert_gnn_layer(suffix: str) -> str:
    head, rest = suffix.split(".", 1)
    index, parameter = rest.split(".", 1)
    return f"{GNN_LAYER_RENAME[head][index]}.{parameter}"


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
    Convert one ``inverse_folding_encoder.*`` key onto the BoltzGen inverse-folding encoder.

    Args:
        key: Checkpoint key with the ``inverse_folding_encoder.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("encoder_layers."):
        index, suffix = key[len("encoder_layers.") :].split(".", 1)
        return f"blocks.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return ENCODER_TOP.get(head, head) + key[len(head) :]


def convert_decoder(key: str) -> str:
    r"""
    Convert one ``structure_module.*`` key onto the BoltzGen inverse-folding decoder.

    Args:
        key: Checkpoint key with the ``structure_module.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("decoder_layers."):
        index, suffix = key[len("decoder_layers.") :].split(".", 1)
        return f"blocks.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return DECODER_TOP.get(head, head) + key[len(head) :]


TOKEN_DISTANCE_TOP = {
    "z_norm": "pair_norm",
    "v_norm": "hidden_norm",
    "z_proj": "pair_projection",
    "a_proj": "distance_feature_projection",
    "u_proj": "output_projection",
}

TOKEN_DISTANCE_ENCODER = {
    "distance_gaussian_smearing": "distance_smearing",
    "distance_token_bias_trans.norm": "distance_norm",
    "distance_token_bias_trans.fc1": "distance_feed_forward.gate_projection",
    "distance_token_bias_trans.fc2": "distance_feed_forward.value_projection",
    "distance_token_bias_trans.fc3": "distance_feed_forward.output_projection",
}


def convert_token_distance(state: dict, *, miniformer_blocks: bool = False) -> dict:
    r"""
    Convert a ``token_distance_module.*`` checkpoint subset onto ``BoltzGenTokenDistanceEmbedder``.

    The pair-only pairformer layers are split tensor-aware (Miniformer variant when
    ``miniformer_blocks`` is set); the distance encoder and top-level projections use the local rename maps.

    Args:
        state: Mapping of ``token_distance_module``-relative keys to tensors.
        miniformer_blocks: Whether the token-distance pairformer uses compact Miniformer pair updates.

    Returns:
        New mapping of converted keys to tensors.
    """
    out: dict = {}
    for key, value in state.items():
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            converter = convert_miniformer_pair_only_layer if miniformer_blocks else _convert_pair_only_layer
            for new_suffix, new_value in converter(suffix, value):
                out[f"pair_stack.blocks.{index}.{new_suffix}"] = new_value
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
    Convert a ``template_module.*`` checkpoint subset onto ``BoltzGenTemplateEmbedder``.

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
            converter = convert_miniformer_pair_only_layer if miniformer_blocks else _convert_pair_only_layer
            for new_suffix, new_value in converter(suffix, value):
                out[f"pair_stack.blocks.{index}.{new_suffix}"] = new_value
        else:
            head = key.split(".")[0]
            out[rename.get(head, head) + key[len(head) :]] = value
    return out


def _pack_conditioned_adaptive_norms(state_dict: dict[str, torch.Tensor]) -> None:
    r"""
    Pack upstream BoltzGen AdaLN scale/shift projections into the shared fused layout.
    """

    scale_weight_suffix = ".conditioned_norm.gate_projection.weight"
    for scale_weight_key in tuple(key for key in state_dict if key.endswith(scale_weight_suffix)):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state_dict.pop(scale_weight_key)
        scale_bias = state_dict.pop(prefix + "gate_projection.bias")
        shift_weight = state_dict.pop(prefix + "shift_projection.weight")
        state_dict[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state_dict[prefix + "conditioning_scale_bias"] = scale_bias


def convert_inverse_folding(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole inverse-folding checkpoint state dict onto a BoltzGenInverseFoldingModel state dict.

    Routes the three top-level prefixes (``input_embedder`` / ``inverse_folding_encoder`` / ``structure_module``)
    to their sub-converters and packs the adaptive-layer-norm projections into the shared parameter layout.

    Args:
        state: Upstream inverse-folding checkpoint state dict (e.g. ``boltzgen1_ifold.ckpt``).

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            new_key = None if inner is None else "input_embeddings." + inner
        elif key.startswith("inverse_folding_encoder."):
            new_key = "inverse_folding_encoder." + convert_encoder(key[len("inverse_folding_encoder.") :])
        elif key.startswith("structure_module."):
            new_key = "inverse_folding_decoder." + convert_decoder(key[len("structure_module.") :])
        else:
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        if new_key is None:
            if is_tied_output_projection(key):
                continue
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        for unpacked_key, unpacked_value in _unpack_triple_product_projection(new_key, value):
            out[unpacked_key] = unpacked_value
    _pack_conditioned_adaptive_norms(out)
    return _convert_vocabulary_axes(out)


def convert_structure_design(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole BoltzGen structure-design checkpoint (e.g. ``boltzgen1_diverse``) onto OFoldX clean names.

    The shared backbone prefixes route through this file's ``convert_trunk_key``. The input embedder carries
    design conditioning, and ``token_distance_module`` plus the v1 template module use local sub-converters.
    Tied ``output_projection.0`` atom-layer duplicates are dropped. The Miniformer variants are auto-detected from
    the presence of ``triangular.*`` keys.

    Args:
        state: Upstream structure-design checkpoint state dict.

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.
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
    miniformer_token_distance = any(
        key.startswith("token_distance_module.pairformer.layers.") and ".triangular." in key for key in state
    )

    # Modules the boltz2 trunk converter doesn't cover (or covers as v2): route to the BoltzGen converters.
    out = {
        f"token_distance_embedder.{key}": value
        for key, value in convert_token_distance(
            _sub("token_distance_module"),
            miniformer_blocks=miniformer_token_distance,
        ).items()
    }
    out.update(
        {
            f"template_embedder.{k}": v
            for k, v in convert_template(_sub("template_module"), miniformer_blocks=miniformer_template).items()
        }
    )
    for key, value in state.items():
        if key.startswith(("token_distance_module.", "template_module.")):
            continue
        if key.startswith("confidence_module."):
            for new_key, new_value in convert_confidence_key(
                key[len("confidence_module.") :],
                value,
            ):
                out[new_key] = new_value
            continue
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            if inner is None:
                if is_tied_output_projection(key):
                    continue
                raise KeyError(f"unmapped structure-design checkpoint key: {key}")
            for new_key, new_value in _unpack_triple_product_projection("input_embeddings." + inner, value):
                out[new_key] = new_value
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
            for packed_key, packed_value in converted:
                for new_key, new_value in _unpack_triple_product_projection(packed_key, packed_value):
                    out[new_key] = new_value
    _pack_conditioned_adaptive_norms(out)
    return _convert_vocabulary_axes(out)


def boltzgen_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> (
    BoltzGenConfig
    | BoltzGenConfidenceConfig
    | BoltzGenResidueTypeConfig
    | BoltzGenMiniformerConfig
    | BoltzGenMiniformerResidueTypeConfig
):
    r"""
    Reconstruct the released structure-design architecture from Lightning hyperparameters.
    """

    hyperparameters: Mapping[str, Any] = checkpoint["hyper_parameters"]
    state: Mapping[str, torch.Tensor] = checkpoint["state_dict"]
    training_args: Mapping[str, Any] = hyperparameters["training_args"]
    embedder_args: Mapping[str, Any] = hyperparameters["embedder_args"]
    msa_args: Mapping[str, Any] = hyperparameters["msa_args"]
    pairformer_args: Mapping[str, Any] = hyperparameters["pairformer_args"]
    score_model_args: Mapping[str, Any] = hyperparameters["score_model_args"]
    diffusion_process_args: Mapping[str, Any] = hyperparameters["diffusion_process_args"]
    template_args: Mapping[str, Any] = hyperparameters["template_args"]

    has_confidence = any(key.startswith("confidence_module.") for key in state)
    miniformer = bool(hyperparameters["use_miniformer"])
    predict_residue_type = bool(hyperparameters.get("predict_res_type", False))
    if has_confidence and (miniformer or predict_residue_type):
        raise ValueError("confidence checkpoints cannot select a different structure-model topology")
    if has_confidence:
        config: BoltzGenConfig = BoltzGenConfidenceConfig()
    elif miniformer and predict_residue_type:
        config = BoltzGenMiniformerResidueTypeConfig()
    elif miniformer:
        config = BoltzGenMiniformerConfig()
    elif predict_residue_type:
        config = BoltzGenResidueTypeConfig()
    else:
        config = BoltzGenConfig()
    config.single_size = hyperparameters["token_s"]
    config.pair_size = hyperparameters["token_z"]
    config.atom_hidden_size = hyperparameters["atom_s"]
    config.atom_pair_hidden_size = hyperparameters["atom_z"]
    config.atom_feature_size = hyperparameters["atom_feature_dim"]
    config.num_distogram_bins = hyperparameters["num_bins"]
    config.fourier_embedding_size = score_model_args["dim_fourier"]
    config.recycling_steps = training_args["recycling_steps"] + 1
    config.conditioning_cutoff_min = hyperparameters["conditioning_cutoff_min"]
    config.conditioning_cutoff_max = hyperparameters["conditioning_cutoff_max"]

    config.input_embeddings.num_atom_encoder_blocks = embedder_args["atom_encoder_depth"]
    config.input_embeddings.num_atom_encoder_attention_heads = embedder_args["atom_encoder_heads"]
    config.input_embeddings.add_method_conditioning = "input_embedder.method_conditioning_init.weight" in state
    config.input_embeddings.add_modified_flag = "input_embedder.modified_conditioning_init.weight" in state
    config.input_embeddings.add_cyclic_flag = "input_embedder.cyclic_conditioning_init.weight" in state
    config.input_embeddings.add_molecule_type_feature = "input_embedder.mol_type_conditioning_init.weight" in state
    config.input_embeddings.add_binding_specification = (
        "input_embedder.binding_specification_conditioning_init.weight" in state
    )
    config.input_embeddings.add_design_mask = "input_embedder.design_mask_conditioning_init.weight" in state
    config.input_embeddings.add_ss_specification = "input_embedder.ss_specification_init.weight" in state

    config.trunk.num_trunk_blocks = pairformer_args["num_blocks"]
    config.trunk.num_pair_attention_heads = pairformer_args["num_heads"]
    config.trunk.num_msa_blocks = msa_args["msa_blocks"]
    config.trunk.msa_size = msa_args["msa_s"]
    config.trunk.msa_triangle_attention_hidden_size = msa_args["pairwise_head_width"]
    config.trunk.num_msa_triangle_attention_heads = msa_args["pairwise_num_heads"]
    config.trunk.triangle_attention_hidden_size = pairformer_args.get("pairwise_head_width", 32)
    config.trunk.num_triangle_attention_heads = pairformer_args.get("pairwise_num_heads", 4)
    config.trunk.use_paired_feature = msa_args["use_paired_feature"]
    if bool(msa_args["miniformer_blocks"]) != miniformer:
        raise ValueError("mixed Pairformer and Miniformer MSA topology is not a concrete BoltzGen architecture")
    config.trunk.pair_dropout = pairformer_args["dropout"]
    config.trunk.msa_pair_dropout = msa_args["z_dropout"]
    config.trunk.msa_dropout = msa_args["msa_dropout"]
    config.trunk.post_layer_norm = pairformer_args.get("post_layer_norm", False)
    config.trunk.use_single_to_pair = pairformer_args.get("use_s_to_z", False)

    config.design.template_size = template_args["template_dim"]
    config.design.num_template_blocks = template_args["template_blocks"]
    config.design.template_dropout = template_args.get("dropout", 0.25)
    config.design.template_post_layer_norm = template_args.get("post_layer_norm", False)
    if bool(template_args.get("miniformer_blocks", False)) != miniformer:
        raise ValueError("mixed Pairformer and Miniformer template topology is not a concrete BoltzGen architecture")
    config.design.use_token_distance = any(key.startswith("token_distance_module.") for key in state)
    if config.design.use_token_distance:
        token_distance_args: Mapping[str, Any] = hyperparameters["token_distance_args"]
        config.design.token_distance_size = token_distance_args["token_distance_dim"]
        config.design.num_token_distance_blocks = token_distance_args["token_distance_blocks"]
        config.design.token_distance_dropout = token_distance_args.get("dropout", 0.25)
        config.design.token_distance_post_layer_norm = token_distance_args.get("post_layer_norm", False)
        if bool(token_distance_args.get("miniformer_blocks", False)) != miniformer:
            raise ValueError(
                "mixed Pairformer and Miniformer token-distance topology is not a concrete BoltzGen architecture"
            )
        config.design.distance_gaussian_dim = token_distance_args["distance_gaussian_dim"]
        config.design.use_token_distance_features = token_distance_args.get("use_token_distance_feats", True)

    config.denoiser.num_token_encoder_blocks = score_model_args["token_transformer_depth"]
    config.denoiser.num_token_encoder_attention_heads = score_model_args["token_transformer_heads"]
    config.denoiser.num_atom_encoder_blocks = score_model_args["atom_encoder_depth"]
    config.denoiser.num_atom_decoder_blocks = score_model_args["atom_decoder_depth"]
    config.denoiser.num_atom_attention_heads = score_model_args["atom_encoder_heads"]
    config.denoiser.atom_hidden_size = config.atom_hidden_size
    config.denoiser.atom_pair_hidden_size = config.atom_pair_hidden_size
    config.denoiser.num_query_atoms = hyperparameters["atoms_per_window_queries"]
    config.denoiser.num_key_atoms = hyperparameters["atoms_per_window_keys"]
    config.denoiser.num_conditioning_transitions = score_model_args["conditioning_transition_layers"]
    config.denoiser.transformer_size = score_model_args.get("tfmr_s")
    config.denoiser.post_layer_norm = score_model_args.get("transformer_post_ln", False)
    config.denoiser.use_qk_norm = score_model_args.get("use_qk_norm", False)
    config.denoiser.gaussian_random_3d_encoding_dim = score_model_args.get(
        "gaussian_random_3d_encoding_dim",
        0,
    )
    config.denoiser.sigma_data = score_model_args["sigma_data"]
    config.denoiser.alignment_reverse_diff = diffusion_process_args["alignment_reverse_diff"]
    if has_confidence:
        confidence_config = cast(BoltzGenConfidenceConfig, config).confidence
        confidence_model_args = hyperparameters["confidence_model_args"]
        if not isinstance(confidence_model_args, Mapping):
            raise TypeError("confidence_model_args must describe a confidence-bearing checkpoint")
        confidence_pairformer_args: Mapping[str, Any] = confidence_model_args["pairformer_args"]
        confidence_output_args: Mapping[str, Any] = confidence_model_args.get("confidence_args", {})

        confidence_config.num_pairformer_blocks = confidence_pairformer_args["num_blocks"]
        confidence_config.num_pair_attention_heads = confidence_pairformer_args["num_heads"]
        confidence_config.triangle_attention_hidden_size = confidence_pairformer_args.get(
            "pairwise_head_width",
            32,
        )
        confidence_config.num_triangle_attention_heads = confidence_pairformer_args.get(
            "pairwise_num_heads",
            4,
        )
        confidence_config.pair_dropout = confidence_pairformer_args["dropout"]
        confidence_config.post_layer_norm = confidence_pairformer_args.get("post_layer_norm", False)
        confidence_config.num_distogram_bins = confidence_model_args.get("num_dist_bins", 64)
        confidence_config.max_distance = confidence_model_args.get("max_dist", 22.0)
        confidence_config.token_level_confidence = hyperparameters.get("token_level_confidence", True)
        confidence_config.add_single_input = bool(confidence_model_args.get("add_s_input_to_s", False))
        confidence_config.add_single_to_pair_product = confidence_model_args.get("add_s_to_z_prod", False)
        confidence_config.use_separate_heads = confidence_output_args.get(
            "use_separate_heads",
            any(key.startswith("confidence_module.confidence_heads.to_pae_intra_logits.") for key in state),
        )
        confidence_config.num_plddt_bins = confidence_output_args.get("num_plddt_bins", 50)
        confidence_config.num_pae_bins = confidence_output_args.get("num_pae_bins", 64)
        confidence_config.num_pde_bins = confidence_output_args.get("num_pde_bins", 64)
        confidence_config.return_latent_features = confidence_model_args.get("return_latent_feats", False)
    return config


def convert_boltzgen_model(
    checkpoint_path: str | PathLike[str],
    *,
    config: (
        BoltzGenConfig
        | BoltzGenConfidenceConfig
        | BoltzGenResidueTypeConfig
        | BoltzGenMiniformerConfig
        | BoltzGenMiniformerResidueTypeConfig
        | None
    ) = None,
    dtype: torch.dtype | None = None,
) -> (
    BoltzGenModel
    | BoltzGenConfidenceModel
    | BoltzGenResidueTypeModel
    | BoltzGenMiniformerModel
    | BoltzGenMiniformerResidueTypeModel
):
    r"""
    Convert a released BoltzGen structure-design checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_structure_design`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    has_confidence = any(key.startswith("confidence_module.") for key in state)
    converted = _cast_floating_tensors(convert_structure_design(state), dtype)
    config = boltzgen_config_from_checkpoint(checkpoint) if config is None else config
    if has_confidence and type(config) is not BoltzGenConfidenceConfig:
        raise TypeError("confidence-bearing checkpoints require BoltzGenConfidenceConfig")
    if not has_confidence and type(config) is BoltzGenConfidenceConfig:
        raise TypeError("BoltzGenConfidenceConfig requires confidence_module checkpoint parameters")
    with torch.device("meta"):
        model: (
            BoltzGenModel
            | BoltzGenConfidenceModel
            | BoltzGenResidueTypeModel
            | BoltzGenMiniformerModel
            | BoltzGenMiniformerResidueTypeModel
        )
        if type(config) is BoltzGenConfidenceConfig:
            model = BoltzGenConfidenceModel(cast(BoltzGenConfidenceConfig, config))
        elif type(config) is BoltzGenResidueTypeConfig:
            model = BoltzGenResidueTypeModel(cast(BoltzGenResidueTypeConfig, config))
        elif type(config) is BoltzGenMiniformerConfig:
            model = BoltzGenMiniformerModel(cast(BoltzGenMiniformerConfig, config))
        elif type(config) is BoltzGenMiniformerResidueTypeConfig:
            model = BoltzGenMiniformerResidueTypeModel(cast(BoltzGenMiniformerResidueTypeConfig, config))
        elif type(config) is BoltzGenConfig:
            model = BoltzGenModel(config)
        else:
            raise TypeError(f"unsupported BoltzGen configuration: {type(config).__name__}")
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_boltzgen_miniformer_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    config: BoltzGenMiniformerConfig | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert and save the released structure-trained small checkpoint."""

    from .pipeline import BoltzGenMiniformerPipeline

    model = convert_boltzgen_model(checkpoint_path, config=config, dtype=dtype)
    if type(model) is not BoltzGenMiniformerModel:
        raise TypeError("structure-trained small checkpoints require the Miniformer coordinate architecture")
    BoltzGenMiniformerPipeline(
        model=cast(BoltzGenMiniformerModel, model),
        processor=BoltzGenProcessor(dtype=dtype),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_boltzgen_inverse_folding_model(
    checkpoint_path: str | PathLike[str],
    *,
    config: BoltzGenInverseFoldingConfig | None = None,
    dtype: torch.dtype | None = None,
) -> BoltzGenInverseFoldingModel:
    r"""
    Convert a released BoltzGen inverse-folding checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_inverse_folding`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = _cast_floating_tensors(convert_inverse_folding(state), dtype)
    with torch.device("meta"):
        model = BoltzGenInverseFoldingModel(config)
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


def convert_boltzgen_checkpoint(
    diverse_checkpoint_path: str | PathLike[str],
    adherence_checkpoint_path: str | PathLike[str],
    inverse_folding_checkpoint_path: str | PathLike[str],
    fold_checkpoint_path: str | PathLike[str],
    affinity_checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    diverse_config: BoltzGenConfig | None = None,
    adherence_config: BoltzGenConfig | None = None,
    inverse_folding_config: BoltzGenInverseFoldingConfig | None = None,
    fold_config: Boltz2Config | None = None,
    affinity_config: Boltz2AffinityConfig | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert all five released BoltzGen checkpoint roles into one fixed repository.

    The diverse structure designer owns the root. The adherence designer,
    inverse-folding model, Boltz-2 folding model, and Boltz-2 affinity model live
    at ``adherence/``, ``ifold/``, ``fold/``, and ``affinity/`` respectively.
    Every pair is required by the concrete [`BoltzGenPipeline`][] lifecycle.
    """

    from ..boltz2.pipeline import Boltz2AffinityScorePipeline, Boltz2Pipeline
    from .pipeline import (
        BoltzGenBackbonePipeline,
        BoltzGenInverseFoldingPipeline,
        BoltzGenPipeline,
    )

    diverse_model = cast(
        BoltzGenModel,
        convert_boltzgen_model(
            diverse_checkpoint_path,
            config=diverse_config,
            dtype=dtype,
        ),
    )
    adherence_model = cast(
        BoltzGenModel,
        convert_boltzgen_model(
            adherence_checkpoint_path,
            config=adherence_config,
            dtype=dtype,
        ),
    )
    inverse_folding_model = convert_boltzgen_inverse_folding_model(
        inverse_folding_checkpoint_path,
        config=inverse_folding_config,
        dtype=dtype,
    )
    fold_model = convert_boltz2_model(
        fold_checkpoint_path,
        config=fold_config,
        dtype=dtype,
    )
    affinity_model = convert_boltz2_affinity_model(
        affinity_checkpoint_path,
        config=affinity_config,
        dtype=dtype,
    )
    BoltzGenPipeline(
        design_diverse=BoltzGenBackbonePipeline(
            model=diverse_model,
            processor=BoltzGenProcessor(dtype=dtype),
        ),
        design_adherence=BoltzGenBackbonePipeline(
            model=adherence_model,
            processor=BoltzGenProcessor(dtype=dtype),
        ),
        inverse_fold=BoltzGenInverseFoldingPipeline(
            model=inverse_folding_model,
            processor=BoltzGenInverseFoldingProcessor(dtype=dtype),
        ),
        fold=Boltz2Pipeline(
            model=fold_model,
            processor=Boltz2Processor(dtype=dtype),
        ),
        affinity=Boltz2AffinityScorePipeline(
            model=affinity_model,
            processor=Boltz2AffinityProcessor(dtype=dtype),
        ),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run boltzgen checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "diverse_checkpoint",
        type=Path,
        help="released BoltzGen diverse-design .ckpt",
    )
    parser.add_argument(
        "adherence_checkpoint",
        type=Path,
        help="released BoltzGen adherence-design .ckpt",
    )
    parser.add_argument(
        "inverse_folding_checkpoint",
        type=Path,
        help="released BoltzGen inverse-folding .ckpt",
    )
    parser.add_argument(
        "fold_checkpoint",
        type=Path,
        help="released Boltz-2 confidence/folding .ckpt",
    )
    parser.add_argument(
        "affinity_checkpoint",
        type=Path,
        help="released Boltz-2 affinity .ckpt",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="optional Hugging Face token for --repo-id",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the --repo-id repo as private",
    )
    args = parser.parse_args(argv)
    convert_boltzgen_checkpoint(
        args.diverse_checkpoint,
        args.adherence_checkpoint,
        args.inverse_folding_checkpoint,
        args.fold_checkpoint,
        args.affinity_checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
