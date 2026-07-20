r"""
Boltz-1 checkpoint converter.

The public Boltz-1 checkpoints store the older monolithic score model. OFoldX splits the invariant
pair/atom conditioning into ``diffusion_conditioning`` and keeps the denoising layers in
``structure_module.score_model``. This converter performs that split while preserving the clean OFoldX
parameter names.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .config import Boltz1Config
from .model import Boltz1Model

_TIED_OUTPUT_PROJECTION = re.compile(
    r"(?:\.diffusion_transformer\.layers\.\d+|\.token_transformer(?:_layers\.0)?\.layers\.\d+)"
    r"\.(?:transition\.)?output_projection\.0\.(?:weight|bias)$"
)


def is_tied_output_projection(key: str) -> bool:
    r"""Return whether ``key`` is a released Boltz checkpoint's tied projection duplicate."""

    return _TIED_OUTPUT_PROJECTION.search(key) is not None


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
        for target_key, target_value in items:
            target_value = cast(Tensor, target_value)
            if dtype is not None and target_value.is_floating_point():
                target_value = target_value.to(dtype=dtype)
            converted[target_key] = target_value
    return converted


def convert_boltz1_key(key: str, tensor: object) -> list[tuple[str, object]]:
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
        msa_items = convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_module.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("distogram_module."):
        new_key = convert_keys(key[len("distogram_module.") :], DISTOGRAM_RULES)
        return [] if new_key is None else [(f"distogram_head.{new_key}", tensor)]
    if key.startswith("structure_module.score_model."):
        return convert_score_model_key(key[len("structure_module.score_model.") :], tensor)
    if key.startswith("structure_module.out_token_feat_update."):
        return convert_out_token_feature_update_key(key[len("structure_module.out_token_feat_update.") :], tensor)
    if key.startswith("confidence_module."):
        confidence_items = convert_confidence_head_key(key[len("confidence_module.") :], tensor)
        return [(f"confidence_head.{new_key}", value) for new_key, value in confidence_items]
    if key in _TRUNK_FLAT:
        return [(_TRUNK_FLAT[key], tensor)]
    return []


def convert_input_embedder_key(key: str, tensor: object) -> tuple[str, object] | None:
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

    prefix = "input_embedder."
    if key.startswith("atom_attention_encoder."):
        inner = key[len("atom_attention_encoder.") :]
        atom_key = _convert_atom_encoder_base_key(inner)
        if atom_key is not None:
            return (prefix + atom_key, tensor)
        routed = _convert_transformer_layer_key(
            inner,
            old_prefix="atom_encoder.diffusion_transformer.layers.",
            layer_prefix="input_embedder.atom_attention_encoder.atom_transformer.transformer.layers.",
            pair_bias_prefix="input_embedder.atom_pair_bias_projection.layers.",
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


def convert_score_model_key(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Convert one ``structure_module.score_model.*`` key from the old Boltz-1 score model.

    This is where the OFoldX split happens: the invariant pairwise conditioner moves to
    ``diffusion_conditioning`` while the single conditioner, atom encoder/decoder, token transformer, and
    the single-to-token / token-norm layers stay under ``structure_module.score_model``.

    Args:
        key: The ``score_model``-relative key (leading ``structure_module.score_model.`` already removed).
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    score_prefix = "structure_module.score_model."
    if key.startswith("single_conditioner."):
        new = convert_keys(key[len("single_conditioner.") :], SINGLE_CONDITIONING_RULES)
        return [] if new is None else [(score_prefix + "single_conditioner." + new, tensor)]
    if key.startswith("pairwise_conditioner."):
        new = convert_keys(key[len("pairwise_conditioner.") :], PAIRWISE_CONDITIONING_RULES)
        return [] if new is None else [("diffusion_conditioning.pairwise_conditioner." + new, tensor)]
    if key.startswith("atom_attention_encoder."):
        return _convert_score_atom_encoder_key(key[len("atom_attention_encoder.") :], tensor)
    if key.startswith("token_transformer.layers."):
        routed = _convert_transformer_layer_key(
            key,
            old_prefix="token_transformer.layers.",
            layer_prefix=score_prefix + "token_transformer.layers.",
            pair_bias_prefix="diffusion_conditioning.token_trans_proj_pair.",
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


def convert_out_token_feature_update_key(key: str, tensor: object) -> list[tuple[str, object]]:
    r"""
    Convert one ``structure_module.out_token_feat_update.*`` key (the diffusion-token accumulator).

    Args:
        key: The ``out_token_feat_update``-relative key (its checkpoint prefix already removed).
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches this key.
    """

    prefix = "structure_module.out_token_feat_update."
    if key.startswith("norm_next."):
        return [(prefix + key.replace("norm_next.", "next_norm.", 1), tensor)]
    if key.startswith("fourier_embed.proj."):
        return [(prefix + key.replace("fourier_embed.proj.", "noise_fourier.projection.", 1), tensor)]
    if key.startswith("norm_fourier."):
        return [(prefix + key.replace("norm_fourier.", "fourier_norm.", 1), tensor)]
    if key.startswith("transition_block."):
        new = _convert_conditioned_transition_block_key(key[len("transition_block.") :])
        return [] if new is None else [(prefix + "transition_block." + new, tensor)]
    return []


def convert_confidence_head_key(key: str, tensor: object) -> list[tuple[str, object]]:
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
        return (
            []
            if input_inner is None
            else [(f"input_embedder.{input_inner[0].removeprefix('input_embedder.')}", input_inner[1])]
        )
    if key.startswith("msa_module."):
        msa_items = convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_module.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        pairformer_items = convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("confidence_heads."):
        confidence_key = _convert_boltz1_confidence_heads_key(key[len("confidence_heads.") :])
        return [] if confidence_key is None else [(f"confidence_heads.{confidence_key}", tensor)]
    if key.startswith("transition_block."):
        transition_key = _convert_conditioned_transition_block_key(key[len("transition_block.") :])
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
    resulting tensors into a native model. With the family's canonical Processor and ``FoldResult`` decoder now in
    place, persistence via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model] is unblocked; the converter
    itself saves nothing implicitly and declares no dtype policy beyond the optional cast.

    Args:
        checkpoint_path: Path to the original Boltz-1 ``.ckpt`` file.
        config: Optional Boltz-1 config (or mapping); defaults to [`Boltz1Config`][].
        dtype: Optional dtype to cast floating-point tensors to during conversion.

    Returns:
        The converted, strict-loaded model.

    Raises:
        KeyError: If a source key is unmapped.

    Examples:
        >>> model = convert_boltz1_checkpoint("boltz1_conf.ckpt")  # doctest: +SKIP
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


def _convert_score_atom_encoder_key(key: str, tensor: object) -> list[tuple[str, object]]:
    atom_key = _convert_atom_encoder_base_key(key)
    if atom_key is not None:
        if atom_key.startswith("atom_attention_encoder.noisy_position_projection."):
            if key == "r_to_q_trans.weight":
                tensor = cast(Tensor, tensor)[:, :3]
            return [("structure_module.score_model." + atom_key, tensor)]
        return [("diffusion_conditioning." + atom_key, tensor)]
    routed = _convert_transformer_layer_key(
        key,
        old_prefix="atom_encoder.diffusion_transformer.layers.",
        layer_prefix="structure_module.score_model.atom_attention_encoder.atom_transformer.transformer.layers.",
        pair_bias_prefix="diffusion_conditioning.atom_enc_proj_pair.",
        tensor=tensor,
    )
    if routed is not None:
        return [routed]
    if key.startswith("atom_to_token_trans."):
        return [
            (
                "structure_module.score_model."
                + key.replace("atom_to_token_trans.", "atom_attention_encoder.token_projection.", 1),
                tensor,
            )
        ]
    return []


def _convert_score_atom_decoder_key(key: str, tensor: object) -> list[tuple[str, object]]:
    routed = _convert_transformer_layer_key(
        key,
        old_prefix="atom_decoder.diffusion_transformer.layers.",
        layer_prefix="structure_module.score_model.atom_attention_decoder.atom_transformer.transformer.layers.",
        pair_bias_prefix="diffusion_conditioning.atom_dec_proj_pair.",
        tensor=tensor,
    )
    if routed is not None:
        return [routed]
    if key.startswith("a_to_q_trans."):
        return [
            (
                "structure_module.score_model."
                + key.replace("a_to_q_trans.", "atom_attention_decoder.token_projection.", 1),
                tensor,
            )
        ]
    if key.startswith("atom_feat_to_atom_pos_update."):
        return [
            (
                "structure_module.score_model."
                + key.replace("atom_feat_to_atom_pos_update.", "atom_attention_decoder.position_update.", 1),
                tensor,
            )
        ]
    return []


def _convert_atom_encoder_base_key(key: str) -> str | None:
    new = convert_keys(key, _ATOM_ENCODER_RULES)
    if new is not None and new != key:
        return "atom_encoder." + new
    if key.startswith("r_to_q_trans."):
        return key.replace("r_to_q_trans.", "atom_attention_encoder.noisy_position_projection.", 1)
    return None


def _convert_transformer_layer_key(
    key: str,
    *,
    old_prefix: str,
    layer_prefix: str,
    pair_bias_prefix: str,
    tensor: object,
) -> tuple[str, object] | None:
    if not key.startswith(old_prefix):
        return None
    index, _, remainder = key[len(old_prefix) :].partition(".")
    pair_bias = _convert_pair_bias_projection_suffix(remainder)
    if pair_bias is not None:
        return (f"{pair_bias_prefix}{index}.{pair_bias}", tensor)
    new = convert_keys(remainder, CONDITIONED_TRANSFORMER_LAYER_RULES)
    return None if new is None else (f"{layer_prefix}{index}.{new}", tensor)


def _convert_boltz1_confidence_heads_key(key: str) -> str | None:
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


def _convert_conditioned_transition_block_key(key: str) -> str | None:
    if key.startswith("adaln.a_norm."):
        return None
    replacements = (
        ("adaln.s_norm.", "adaptive_norm.conditioning_norm."),
        ("adaln.s_scale.", "adaptive_norm.gate_projection."),
        ("adaln.s_bias.", "adaptive_norm.shift_projection."),
        ("swish_gate.0.", "gate_value_projection."),
        ("a_to_b.", "expand_projection."),
        ("b_to_a.", "output_projection."),
        ("output_projection.0.", "output_gate_projection."),
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


def _prefix_index(index: str, items: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [(f"{index}.{key}", value) for key, value in items]


# ---------------------------------------------------------------------------
# Boltz-1 checkpoint-name conversion helpers.
#
# Relocated verbatim from the former shared models-root Boltz converter; Boltz-1 is now the sole
# owner (Boltz-2 / BoltzGen carry their own copies). Only the definitions this converter uses are
# kept; the same-named ``is_tied_output_projection`` / ``convert_input_embedder_key`` are NOT
# duplicated here -- Boltz-1's own versions above are the ones in use.
# ---------------------------------------------------------------------------

Rule = tuple[re.Pattern[str], str | None]
Converter = list[Rule] | Callable[[str], "str | None"]


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
