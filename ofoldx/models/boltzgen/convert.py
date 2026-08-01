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
BoltzGen structure-design and inverse-folding checkpoint conversion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ...conversion import SourceCheckpoint, save_repository
from ...errors import RequestRef, fail
from ...references import classify_reference
from ..boltz1.convert import is_tied_output_projection
from ..boltz2.config import Boltz2AffinityConfig, Boltz2Config
from ..boltz2.convert import (
    BOLTZ_PAIRWISE_CONDITIONING_RULES,
    BOLTZ_SINGLE_CONDITIONING_RULES,
    convert_boltz2_affinity_model,
    convert_boltz2_model,
    convert_boltz_diffusion_conditioning,
    convert_boltz_score_model,
)
from ..boltz2.pipeline import Boltz2AffinityScorePipeline, Boltz2Pipeline
from ..boltz2.processor import Boltz2AffinityProcessor, Boltz2Processor
from ..convert import (
    MSA_PAIR_WEIGHTED_AVERAGING_RULES,
    MSA_TRANSITION_RULES,
    OUTER_PRODUCT_MEAN_RULES,
    PAIR_ONLY_LAYER_RULES,
    Rule,
    cast_floating_state,
    compile_rules,
    convert_keys,
    convert_msa_module,
    convert_rule_tensor,
    convert_standard_msa_layer,
    pack_conditioned_adaptive_norms,
    reprefix,
    reprefix_indexed,
    transition_rules,
    triangle_attention_rules,
    triangle_multiplication_rules,
    unpack_gate_value_projection,
)
from ..esmfold2.pipeline import EsmFold2Pipeline
from ..proteinmpnn.config import ProteinMpnnProcessorConfig
from ..proteinmpnn.convert import convert_proteinmpnn_model
from ..proteinmpnn.pipeline import ProteinMpnnPipeline
from ..proteinmpnn.processor import ProteinMpnnProcessor
from ..vocabulary import FOLDING_SOURCE_TOKEN_INDEX, folding_word_embeddings
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
from .pipeline import (
    BoltzGenBackbonePipeline,
    BoltzGenInverseFoldingPipeline,
    BoltzGenMiniformerPipeline,
    BoltzGenPipeline,
)
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor


def _convert_vocabulary_axes(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Regenerate every residue-type and MSA axis from the 33-class upstream folding alphabet onto the 72 canonical
    classes; each tensor's other columns keep their place.

    Every BoltzGen path reading these blocks is linear (the input embeddings sum projections, the decoder projects the
    decoded one-hot), so no layer needs ``folding_source_map``.
    """

    size = len(FOLDING_SOURCE_TOKEN_INDEX)
    for key in (
        "input_embeddings.residue_type_projection.weight",
        "inverse_folding_decoder.residue_type_projection.weight",
    ):
        if key in state_dict:
            (state_dict[key],) = folding_word_embeddings(state_dict[key], dim=-1)
    for key in (
        "coordinate_diffusion.denoiser.atom_decoder.residue_type_projection.weight",
        "coordinate_diffusion.denoiser.atom_decoder.residue_type_projection.bias",
        "inverse_folding_decoder.residue_type_head.weight",
    ):
        if key in state_dict:
            (state_dict[key],) = folding_word_embeddings(state_dict[key], dim=0)
    # the profile and MSA one-hot blocks lead their feature vector; the deletion and pairing columns follow them
    for key in ("input_embeddings.msa_profile_projection.weight", "msa_trunk.msa_projection.weight"):
        if key in state_dict:
            tensor = state_dict[key]
            (tokens,) = folding_word_embeddings(tensor[..., :size], dim=-1)
            state_dict[key] = torch.cat((tokens, tensor[..., size:]), dim=-1)
    template_key = "template_embedder.template_feature_projection.weight"
    if template_key in state_dict:
        tensor = state_dict[template_key]
        start = tensor.shape[-1] - 2 * size
        first, second = folding_word_embeddings(tensor[..., start : start + size], tensor[..., start + size :], dim=-1)
        state_dict[template_key] = torch.cat((tensor[..., :start], first, second), dim=-1)
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
        (r"^pair_bias_attn\.q_norm\.", "attention.projection.query_norm."),
        (r"^pair_bias_attn\.k_norm\.", "attention.projection.key_norm."),
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


def _fold_single_width_adapter(state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    r"""
    Fold the ``token_s_to_tfmr_s`` width adapter of a widened diffusion Transformer into ``single_embed``.

    Upstream applies the adapter and then ``single_embed``, two affine maps with nothing between them, when
    ``tfmr_s`` differs from twice the trunk width. Their composition is the single projection that
    [`SingleConditioning`][ofoldx.modules.diffusion.conditioning.SingleConditioning] owns; it is formed in
    float64 and returned in the released dtype.
    """

    prefix = "structure_module.score_model.single_conditioner."
    adapter, embed = prefix + "token_s_to_tfmr_s.", prefix + "single_embed."
    if adapter + "weight" not in state:
        return state
    folded = {key: value for key, value in state.items() if not key.startswith(adapter)}
    inner_weight, inner_bias = state[adapter + "weight"].double(), state[adapter + "bias"].double()
    outer_weight, outer_bias = state[embed + "weight"], state[embed + "bias"]
    folded[embed + "weight"] = (outer_weight.double() @ inner_weight).to(outer_weight.dtype)
    folded[embed + "bias"] = (outer_weight.double() @ inner_bias + outer_bias.double()).to(outer_bias.dtype)
    return folded


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
    if key == "encoding_3d.std":
        return "position_encoding.variance"
    if key.startswith("encoding_3d."):
        return key.replace("encoding_3d.", "position_encoding.", 1)
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
    if key.startswith("res_type_predictor."):
        return key.replace("res_type_predictor.", "residue_type_projection.", 1)
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


# Released tensors the model computes from its configuration: source key -> model buffer.
_COMPUTED_BUFFERS = {
    "confidence_module.boundaries": "confidence_head.boundaries",
    "token_distance_module.token_distance_encoder.distance_gaussian_smearing.offset": (
        "token_distance_embedder.distance_encoder.distance_smearing.centers"
    ),
    "inverse_folding_encoder.distance_gaussian_smearing.offset": "inverse_folding_encoder.distance_smearing.centers",
}

CONFIDENCE_KEY_RENAMES = {
    "dist_bin_pairwise_embed": "distance_embedding",
    "s_to_z": "pair_left_projection",
    "s_to_z_transpose": "pair_right_projection",
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

    Resolved-class projection rows are exchanged within each atom slot from
    source ``[resolved, unresolved]`` to canonical ``[unresolved, resolved]``
    order.
    """

    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"confidence_head.pairformer_stack.blocks.{index}.{new_key}", value)
            for new_key, value in convert_rule_tensor(remainder, tensor, PAIRFORMER_LAYER_RULES, "pair_update.")
        ]
    if key.startswith("confidence_heads."):
        inner = key[len("confidence_heads.") :]
        head, separator, remainder = inner.partition(".")
        renamed = CONFIDENCE_OUTPUT_KEY_RENAMES.get(head)
        if renamed is None:
            fail("value_invalid", RequestRef("checkpoint", key))
        if renamed == "resolved_projection":
            tensor = tensor.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
        suffix = separator + remainder if separator else ""
        return [(f"confidence_head.output_head.{renamed}{suffix}", tensor)]
    if key.startswith(("rel_pos.", "token_bonds.", "token_bonds_type.", "contact_conditioning.")):
        return []
    head, separator, remainder = key.partition(".")
    renamed = CONFIDENCE_KEY_RENAMES.get(head)
    if renamed is None:
        fail("value_invalid", RequestRef("checkpoint", key))
    suffix = separator + remainder if separator else ""
    return [(f"confidence_head.{renamed}{suffix}", tensor)]


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
        (r"^s_to_z_1\.", "pair_left_projection."),
        (r"^s_to_z_2\.", "pair_right_projection."),
        (r"^s_post_norm\.", "single_post_norm."),
        *transition_rules("transition_s", "single_transition"),
        *transition_rules("transition_z", "pair_transition"),
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
        *transition_rules("transition_z", "pair_transition"),
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
        InvalidRequestError: ``value_invalid`` if the key does not match any known MSA sub-block.
    """

    if key.startswith("pair_weighted_averaging."):
        return [
            (
                cast(
                    str,
                    reprefix(
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
                    reprefix(key, "msa_transition.", "msa_transition.", MSA_TRANSITION_RULES),
                ),
                tensor,
            )
        ]
    if key.startswith("outer_product_mean."):
        return [
            (
                cast(
                    str,
                    reprefix(key, "outer_product_mean.", "outer_product_mean.", OUTER_PRODUCT_MEAN_RULES),
                ),
                tensor,
            )
        ]
    if key.startswith("pairformer_layer."):
        inner = convert_miniformer_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in inner]
    fail("value_invalid", RequestRef("checkpoint", key))


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
        inner = convert_msa_module(
            key[len("msa_module.") :],
            tensor,
            layer_converter=convert_miniformer_msa_layer if miniformer_msa else convert_standard_msa_layer,
        )
        return [(f"msa_trunk.{new_key}", new_value) for new_key, new_value in inner]
    if key.startswith(("pairformer_stack.layers.", "pairformer_module.layers.")):
        prefix = (
            "pairformer_stack.layers." if key.startswith("pairformer_stack.layers.") else "pairformer_module.layers."
        )
        index, _, remainder = key[len(prefix) :].partition(".")
        inner = (
            convert_miniformer_layer(remainder, tensor)
            if miniformer_pairformer
            else convert_rule_tensor(remainder, tensor, PAIRFORMER_LAYER_RULES, "pair_update.")
        )
        return [(f"trunk_stack.blocks.{index}.{new_key}", new_value) for new_key, new_value in inner]
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
    if key.startswith("bfactor_module."):
        new = convert_keys(key[len("bfactor_module.") :], B_FACTOR_RULES)
        return [("b_factor_head." + new, tensor)] if new is not None else []
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


def convert_input_embedder(key: str, tensor: Tensor, *, num_atom_encoder_blocks: int) -> list[tuple[str, Tensor]]:
    r"""
    Convert one ``input_embedder.*`` key onto its BoltzGen input-embedding parameters.

    The released input embedder normalizes the atom pairs once and projects them to every atom-encoder
    block's heads at once; OFoldX keeps one normalization and one head projection per block. The shared
    normalization is copied to every block, and the projection rows split block-major.

    Args:
        key: Checkpoint key with the ``input_embedder.`` prefix already stripped.
        tensor: The associated weight tensor.
        num_atom_encoder_blocks: Number of atom-encoder Transformer blocks of the checkpoint.

    Returns:
        ``(key, tensor)`` pairs under the OFoldX names; empty to drop the tied duplicate.
    """
    if key.startswith("atom_encoder."):
        return _convert_atom_encoder(key[len("atom_encoder.") :], tensor)
    if key.startswith("atom_attention_encoder."):
        inner = _convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return [] if inner is None else [("atom_encoder." + inner, tensor)]
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
    head = key.split(".")[0]
    if head in INPUT_EMBEDDER_CONDITIONING:
        return [(INPUT_EMBEDDER_CONDITIONING[head] + key[len(head) :], tensor)]
    return [(key, tensor)]


GNN_LAYER_RENAME = {
    "attn_weight_mlp": {
        "0": "attention.weight_feed_forward.input_projection",
        "2": "attention.weight_feed_forward.intermediate_projections.0",
        "4": "attention.weight_feed_forward.output_projection",
    },
    "attn_value_mlp": {
        "0": "attention.value_feed_forward.input_projection",
        "2": "attention.value_feed_forward.intermediate_projections.0",
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
    The released Gaussian-smearing centers are not returned: the embedder computes them, and
    [`convert_boltzgen_model`][ofoldx.models.boltzgen.convert.convert_boltzgen_model] checks the released copy.

    Args:
        state: Mapping of ``token_distance_module``-relative keys to tensors.
        miniformer_blocks: Whether the token-distance pairformer uses compact Miniformer pair updates.

    Returns:
        New mapping of converted keys to tensors.
    """
    out: dict = {}
    for key, value in state.items():
        if f"token_distance_module.{key}" in _COMPUTED_BUFFERS:
            continue
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            inner = (
                convert_miniformer_pair_only_layer(suffix, value)
                if miniformer_blocks
                else convert_rule_tensor(suffix, value, PAIR_ONLY_LAYER_RULES)
            )
            for new_suffix, new_value in inner:
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
            inner = (
                convert_miniformer_pair_only_layer(suffix, value)
                if miniformer_blocks
                else convert_rule_tensor(suffix, value, PAIR_ONLY_LAYER_RULES)
            )
            for new_suffix, new_value in inner:
                out[f"pair_stack.blocks.{index}.{new_suffix}"] = new_value
        else:
            head = key.split(".")[0]
            out[rename.get(head, head) + key[len(head) :]] = value
    return out


def convert_inverse_folding(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole inverse-folding checkpoint state dict onto a BoltzGenInverseFoldingModel state dict.

    Routes the three top-level prefixes (``input_embedder`` / ``inverse_folding_encoder`` / ``structure_module``)
    to their sub-converters and packs the adaptive-layer-norm projections into the shared parameter layout. A
    released copy of a tensor the model computes from its configuration is not returned;
    [`convert_boltzgen_inverse_folding_model`][ofoldx.models.boltzgen.convert.convert_boltzgen_inverse_folding_model]
    checks it instead.

    Args:
        state: Upstream inverse-folding checkpoint state dict (e.g. ``boltzgen1_ifold.ckpt``).

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.
    """
    out: dict[str, torch.Tensor] = {}
    num_atom_encoder_blocks = _count_layers(state, _INPUT_ATOM_LAYER_PREFIX)
    for key, value in state.items():
        if key in _COMPUTED_BUFFERS:
            continue
        if key.startswith("input_embedder."):
            entries = [
                ("input_embeddings." + inner_key, inner_value)
                for inner_key, inner_value in convert_input_embedder(
                    key[len("input_embedder.") :],
                    value,
                    num_atom_encoder_blocks=num_atom_encoder_blocks,
                )
            ]
        elif key.startswith("inverse_folding_encoder."):
            entries = [("inverse_folding_encoder." + convert_encoder(key[len("inverse_folding_encoder.") :]), value)]
        elif key.startswith("structure_module."):
            entries = [("inverse_folding_decoder." + convert_decoder(key[len("structure_module.") :]), value)]
        else:
            fail("value_invalid", RequestRef("checkpoint", key))
        if not entries:
            if is_tied_output_projection(key):
                continue
            fail("value_invalid", RequestRef("checkpoint", key))
        for new_key, new_value in entries:
            for unpacked_key, unpacked_value in unpack_gate_value_projection(new_key, new_value):
                out[unpacked_key] = unpacked_value
    pack_conditioned_adaptive_norms(out)
    return _convert_vocabulary_axes(out)


def convert_structure_design(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole BoltzGen structure-design checkpoint (e.g. ``boltzgen1_diverse``) onto OFoldX clean names.

    The shared backbone prefixes route through this file's ``convert_trunk_key``. The input embedder carries
    design conditioning, and ``token_distance_module`` plus the v1 template module use local sub-converters.
    Tied ``output_projection.0`` atom-layer duplicates are dropped. The Miniformer variants are auto-detected from
    the presence of ``triangular.*`` keys. A widened diffusion Transformer's single-conditioning width adapter is
    folded into the single projection it feeds. A released copy of a tensor the model computes from its
    configuration is not returned; [`convert_boltzgen_model`][ofoldx.models.boltzgen.convert.convert_boltzgen_model]
    checks it instead.

    Args:
        state: Upstream structure-design checkpoint state dict.

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.
    """

    state = _fold_single_width_adapter(state)
    num_atom_encoder_blocks = _count_layers(state, _INPUT_ATOM_LAYER_PREFIX)

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
        if key.startswith(("token_distance_module.", "template_module.")) or key in _COMPUTED_BUFFERS:
            continue
        if key.startswith("confidence_module."):
            for new_key, new_value in convert_confidence_key(
                key[len("confidence_module.") :],
                value,
            ):
                out[new_key] = new_value
            continue
        if key.startswith("input_embedder."):
            entries = convert_input_embedder(
                key[len("input_embedder.") :],
                value,
                num_atom_encoder_blocks=num_atom_encoder_blocks,
            )
            if not entries:
                if is_tied_output_projection(key):
                    continue
                fail("value_invalid", RequestRef("checkpoint", key))
            for inner_key, inner_value in entries:
                for new_key, new_value in unpack_gate_value_projection("input_embeddings." + inner_key, inner_value):
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
                fail("value_invalid", RequestRef("checkpoint", source_key))
            for packed_key, packed_value in converted:
                for new_key, new_value in unpack_gate_value_projection(packed_key, packed_value):
                    out[new_key] = new_value
    pack_conditioned_adaptive_norms(out)
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
        fail("value_invalid", RequestRef("checkpoint", "confidence"), topology="structure_model")
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
    config.num_recycles = training_args["recycling_steps"] + 1
    config.min_conditioning_cutoff = hyperparameters["conditioning_cutoff_min"]
    config.max_conditioning_cutoff = hyperparameters["conditioning_cutoff_max"]

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
    config.trunk.msa_triangle_attention_head_size = msa_args["pairwise_head_width"]
    config.trunk.num_msa_triangle_attention_heads = msa_args["pairwise_num_heads"]
    config.trunk.triangle_attention_head_size = pairformer_args.get("pairwise_head_width", 32)
    config.trunk.num_triangle_attention_heads = pairformer_args.get("pairwise_num_heads", 4)
    config.trunk.use_paired_feature = msa_args["use_paired_feature"]
    if bool(msa_args["miniformer_blocks"]) != miniformer:
        fail("value_invalid", RequestRef("checkpoint", "msa_args"), topology="miniformer")
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
        fail("value_invalid", RequestRef("checkpoint", "template_args"), topology="miniformer")
    config.design.use_token_distance = any(key.startswith("token_distance_module.") for key in state)
    if config.design.use_token_distance:
        token_distance_args: Mapping[str, Any] = hyperparameters["token_distance_args"]
        config.design.token_distance_size = token_distance_args["token_distance_dim"]
        config.design.num_token_distance_blocks = token_distance_args["token_distance_blocks"]
        config.design.token_distance_dropout = token_distance_args.get("dropout", 0.25)
        config.design.token_distance_post_layer_norm = token_distance_args.get("post_layer_norm", False)
        if bool(token_distance_args.get("miniformer_blocks", False)) != miniformer:
            fail("value_invalid", RequestRef("checkpoint", "token_distance_args"), topology="miniformer")
        config.design.num_distance_gaussians = token_distance_args["distance_gaussian_dim"]
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
    config.denoiser.num_coordinate_gaussians = score_model_args.get(
        "gaussian_random_3d_encoding_dim",
        0,
    )
    config.denoiser.sigma_data = score_model_args["sigma_data"]
    config.denoiser.alignment_reverse_diff = diffusion_process_args["alignment_reverse_diff"]
    sampler = config.sampler
    sampler.num_sampling_steps = diffusion_process_args.get("num_sampling_steps", sampler.num_sampling_steps)
    sampler.min_noise_level = diffusion_process_args.get("sigma_min", sampler.min_noise_level)
    sampler.max_noise_level = diffusion_process_args.get("sigma_max", sampler.max_noise_level)
    sampler.rho = diffusion_process_args.get("rho", sampler.rho)
    sampler.churn_rate = diffusion_process_args.get("gamma_0", sampler.churn_rate)
    sampler.min_churn_noise_level = diffusion_process_args.get("gamma_min", sampler.min_churn_noise_level)
    sampler.noise_scale = diffusion_process_args.get("noise_scale", sampler.noise_scale)
    sampler.step_scale = diffusion_process_args.get("step_scale", sampler.step_scale)
    sampler.sampling_schedule = diffusion_process_args.get("sampling_schedule", sampler.sampling_schedule)
    if sampler.sampling_schedule == "af3":
        sampler.sampling_schedule = "karras"
    sampler.time_dilation = diffusion_process_args.get("time_dilation", sampler.time_dilation)
    sampler.time_dilation_start = diffusion_process_args.get("time_dilation_start", sampler.time_dilation_start)
    sampler.time_dilation_end = diffusion_process_args.get("time_dilation_end", sampler.time_dilation_end)
    sampler.step_scale_function = diffusion_process_args.get("step_scale_function", "constant")
    sampler.noise_scale_function = diffusion_process_args.get("noise_scale_function", "constant")
    sampler.min_step_scale = diffusion_process_args.get("min_step_scale", 1.0)
    sampler.max_step_scale = diffusion_process_args.get("max_step_scale", 1.0)
    sampler.step_scale_alpha = diffusion_process_args.get("step_scale_alpha", 1.0)
    sampler.step_scale_beta = diffusion_process_args.get("step_scale_beta", 1.0)
    sampler.min_noise_scale = diffusion_process_args.get("min_noise_scale", 1.0)
    sampler.max_noise_scale = diffusion_process_args.get("max_noise_scale", 1.0)
    sampler.noise_scale_alpha = diffusion_process_args.get("noise_scale_alpha", 1.0)
    sampler.noise_scale_beta = diffusion_process_args.get("noise_scale_beta", 1.0)
    step_scale_random = diffusion_process_args.get("step_scale_random")
    sampler.step_scale_random = None if step_scale_random is None else tuple(step_scale_random)
    if has_confidence:
        confidence_config = cast(BoltzGenConfidenceConfig, config).confidence
        confidence_model_args = hyperparameters["confidence_model_args"]
        if not isinstance(confidence_model_args, Mapping):
            fail("value_invalid", RequestRef("checkpoint", "confidence_model_args"))
        confidence_pairformer_args: Mapping[str, Any] = confidence_model_args["pairformer_args"]
        confidence_output_args: Mapping[str, Any] = confidence_model_args.get("confidence_args", {})

        confidence_config.num_pairformer_blocks = confidence_pairformer_args["num_blocks"]
        confidence_config.num_pair_attention_heads = confidence_pairformer_args["num_heads"]
        confidence_config.triangle_attention_head_size = confidence_pairformer_args.get(
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
    converted = cast_floating_state(convert_structure_design(state), dtype)
    config = boltzgen_config_from_checkpoint(checkpoint) if config is None else config
    if has_confidence and type(config) is not BoltzGenConfidenceConfig:
        fail("value_invalid", RequestRef("config", type(config).__name__), expected="BoltzGenConfidenceConfig")
    if not has_confidence and type(config) is BoltzGenConfidenceConfig:
        fail("value_invalid", RequestRef("checkpoint", "without confidence_module"))
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
            fail("value_invalid", RequestRef("config", type(config).__name__))
    _check_computed_buffers(model, state)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
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

    model = convert_boltzgen_model(checkpoint_path, config=config, dtype=dtype)
    if type(model) is not BoltzGenMiniformerModel:
        fail("value_invalid", RequestRef("config", type(model.config).__name__), expected="miniformer")
    save_repository(
        BoltzGenMiniformerPipeline(
            model=cast(BoltzGenMiniformerModel, model),
            processor=BoltzGenProcessor(dtype=dtype),
        ),
        output_dir,
        sources={"checkpoint": checkpoint_path},
        converter="ofoldx.models.boltzgen.convert.convert_boltzgen_miniformer_checkpoint",
        options={"dtype": None if dtype is None else str(dtype)},
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
    converted = cast_floating_state(convert_inverse_folding(state), dtype)
    with torch.device("meta"):
        model = BoltzGenInverseFoldingModel(config)
    _check_computed_buffers(model, state)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def _check_computed_buffers(model: torch.nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    r"""
    Check a release's copies of the tensors the model computes from its configuration.

    The release serialized these tensors although its configuration determines them, so they are evidence
    about that configuration rather than state to load: a disagreement means the configuration does not
    describe the checkpoint.
    """

    for source_key, released in state_dict.items():
        buffer_name = _COMPUTED_BUFFERS.get(source_key)
        if buffer_name is not None:
            expected = model.get_buffer(buffer_name)
            tolerance = (
                8 * max(torch.finfo(released.dtype).eps, torch.finfo(expected.dtype).eps)
                if released.is_floating_point() and expected.is_floating_point()
                else 0.0
            )
            if not torch.allclose(released, expected.to(released), rtol=tolerance, atol=0.0):
                fail("value_invalid", RequestRef("checkpoint", source_key), disagrees_with=buffer_name)


def convert_boltzgen_checkpoint(
    diverse_checkpoint_path: str | PathLike[str],
    adherence_checkpoint_path: str | PathLike[str],
    inverse_folding_checkpoint_path: str | PathLike[str],
    soluble_checkpoint_path: str | PathLike[str],
    fold_checkpoint_path: str | PathLike[str],
    affinity_checkpoint_path: str | PathLike[str],
    esmfold2_repository: str | PathLike[str],
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
    Convert BoltzGen's checkpoint roles and assemble its complete design repository.

    The diverse structure designer occupies ``design_diverse/``. The adherence designer,
    inverse-folding models, Boltz-2 folding/affinity and the explicitly supplied
    native ESMFold2 pair occupy their fixed component subdirectories.
    """

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
    soluble_model = convert_proteinmpnn_model(soluble_checkpoint_path, dtype=dtype)
    esmfold2_source = classify_reference(esmfold2_repository)
    save_repository(
        BoltzGenPipeline(
            diverse_generator=BoltzGenBackbonePipeline(
                model=diverse_model,
                processor=BoltzGenProcessor(dtype=dtype),
            ),
            adherence_generator=BoltzGenBackbonePipeline(
                model=adherence_model,
                processor=BoltzGenProcessor(dtype=dtype),
            ),
            inverse_folder=BoltzGenInverseFoldingPipeline(
                model=inverse_folding_model,
                processor=BoltzGenInverseFoldingProcessor(dtype=dtype),
            ),
            soluble_designer=ProteinMpnnPipeline(
                model=soluble_model,
                processor=ProteinMpnnProcessor(
                    ProteinMpnnProcessorConfig(ca_only=soluble_model.config.ca_only), dtype=dtype
                ),
            ),
            folder=Boltz2Pipeline(
                model=fold_model,
                processor=Boltz2Processor(dtype=dtype),
            ),
            affinity_scorer=Boltz2AffinityScorePipeline(
                model=affinity_model,
                processor=Boltz2AffinityProcessor(dtype=dtype),
            ),
            esmfold2=EsmFold2Pipeline.from_pretrained(esmfold2_repository, dtype=dtype, token=token),
        ),
        output_dir,
        sources={
            "diverse": diverse_checkpoint_path,
            "adherence": adherence_checkpoint_path,
            "inverse_folding": inverse_folding_checkpoint_path,
            "soluble": soluble_checkpoint_path,
            "fold": fold_checkpoint_path,
            "affinity": affinity_checkpoint_path,
            "esmfold2": SourceCheckpoint(
                esmfold2_source if isinstance(esmfold2_source, Path) else None,
                reference=str(esmfold2_repository),
            ),
        },
        converter="ofoldx.models.boltzgen.convert.convert_boltzgen_checkpoint",
        options={"dtype": None if dtype is None else str(dtype)},
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
    parser.add_argument("soluble_checkpoint", type=Path, help="released SolubleMPNN .pt")
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
    parser.add_argument("esmfold2_repository", help="native ESMFold2 Model-Processor repository")
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
        args.soluble_checkpoint,
        args.fold_checkpoint,
        args.affinity_checkpoint,
        args.esmfold2_repository,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
