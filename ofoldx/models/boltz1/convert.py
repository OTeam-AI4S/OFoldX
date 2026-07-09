r"""
Boltz-1 checkpoint converter.

The public Boltz-1 checkpoints store the older monolithic score model. OFoldX splits the invariant
pair/atom conditioning into ``diffusion_conditioning`` and keeps the denoising layers in
``structure_module.score_model``. This converter performs that split while preserving the clean OFoldX
parameter names.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from ..boltz2 import convert as boltz2_convert
from .config import Boltz1Config
from .model import Boltz1Model

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def convert_boltz1_state_dict(
    state_dict: Mapping[str, object],
    *,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> dict[str, object]:
    r"""
    Convert a full Boltz-1 ``state_dict`` to [`Boltz1Model`][ofoldx.models.boltz1.model.Boltz1Model] keys.

    Every source key is stripped of a leading ``module.`` and routed through
    [`convert_boltz1_key`][ofoldx.models.boltz1.convert.convert_boltz1_key]; unrecognized, non-ignorable
    keys are collected and (under ``strict``) reported. Adaptive-layer-norm parameters are then packed and the
    zero atom-projection biases are synthesized so the OFoldX module tree loads without missing keys.

    Args:
        state_dict: The source Boltz-1 parameter mapping (checkpoint ``state_dict``).
        dtype: Optional dtype to cast floating-point tensors to; leaves them unchanged when ``None``.
        strict: When ``True``, raise if any non-ignorable source key fails to map.

    Returns:
        Mapping from OFoldX parameter name to tensor (plus any packed/synthesized entries).

    Raises:
        KeyError: If ``strict`` is set and one or more source keys are unmapped.
    """

    converted: dict[str, object] = {}
    unmapped: list[str] = []
    for source_key, value in state_dict.items():
        key = source_key.removeprefix("module.")
        items = convert_boltz1_key(key, value)
        if not items and not _is_ignorable_key(key):
            unmapped.append(key)
        for target_key, target_value in items:
            if dtype is not None and isinstance(target_value, Tensor) and target_value.is_floating_point():
                target_value = target_value.to(dtype=dtype)
            converted[target_key] = target_value
    if strict and unmapped:
        preview = ", ".join(unmapped[:20])
        suffix = "" if len(unmapped) <= 20 else f", ... (+{len(unmapped) - 20} more)"
        raise KeyError(f"unmapped Boltz-1 checkpoint keys: {preview}{suffix}")
    converted = boltz2_convert.pack_adaptive_layer_norm_state(converted)
    _add_zero_atom_projection_biases(converted)
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
        msa_items = boltz2_convert.convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_module.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = boltz2_convert.convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("distogram_module."):
        new_key = boltz2_convert.convert_keys(key[len("distogram_module.") :], boltz2_convert.DISTOGRAM_RULES)
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
        new = boltz2_convert.convert_keys(key[len("single_conditioner.") :], boltz2_convert.SINGLE_CONDITIONING_RULES)
        return [] if new is None else [(score_prefix + "single_conditioner." + new, tensor)]
    if key.startswith("pairwise_conditioner."):
        new = boltz2_convert.convert_keys(
            key[len("pairwise_conditioner.") :], boltz2_convert.PAIRWISE_CONDITIONING_RULES
        )
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
        msa_items = boltz2_convert.convert_msa_module(key[len("msa_module.") :], tensor)
        return [(f"msa_module.{new_key}", value) for new_key, value in msa_items]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        pairformer_items = boltz2_convert.convert_pairformer_layer(remainder, tensor)
        return [
            (f"pairformer_stack.blocks.{new_key}", value) for new_key, value in _prefix_index(index, pairformer_items)
        ]
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        pairformer_items = boltz2_convert.convert_pairformer_layer(remainder, tensor)
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
    output_dir: str | PathLike[str],
    *,
    config: Boltz1Config | Mapping[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> dict[str, object]:
    r"""
    Convert an original Boltz-1 ``.ckpt`` into an OFoldX pretrained directory.

    Loads the checkpoint (taking its ``state_dict`` entry when present), remaps the parameters via
    [`convert_boltz1_state_dict`][ofoldx.models.boltz1.convert.convert_boltz1_state_dict], and writes the
    OFoldX single-model-file artifacts: ``manifest.json``, ``config.json``, and ``model.safetensors``.

    Args:
        checkpoint_path: Path to the original Boltz-1 ``.ckpt`` file.
        output_dir: Directory to create and write the ``config.json`` / ``model.safetensors`` into.
        config: Optional Boltz-1 config (or mapping) to persist; defaults to [`Boltz1Config`][].
        dtype: Optional dtype to cast floating-point tensors to during conversion.
        strict: When ``True``, raise if any non-ignorable source key fails to map.

    Returns:
        A summary dict with the source ``checkpoint`` path, ``output_dir``, converted ``tensor_count``, and
        the ``dtype`` string (or ``None``).

    Raises:
        TypeError: If the loaded checkpoint does not contain a mapping state dict.
        KeyError: If ``strict`` is set and one or more source keys are unmapped.

    Examples:
        >>> convert_boltz1_checkpoint("boltz1_conf.ckpt", "boltz1")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("checkpoint does not contain a mapping state_dict")
    converted = convert_boltz1_state_dict(raw_state, dtype=dtype, strict=strict)

    model_config = Boltz1Config() if config is None else Boltz1Config(config)
    model = Boltz1Model(model_config)
    if dtype is not None:
        model.to(dtype=dtype)
    tensor_state: dict[str, Tensor] = {}
    for key, value in converted.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"converted Boltz-1 state contains a non-tensor value at {key!r}")
        tensor_state[key] = value
    model.load_state_dict(tensor_state, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "tensor_count": len(tensor_state),
        "dtype": str(dtype) if dtype is not None else None,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for the checkpoint-conversion entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an original Boltz-1 .ckpt file")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Allow non-ignorable source checkpoint keys to remain unmapped",
    )
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for Boltz-1 checkpoint conversion.

    Examples:
        >>> main(["boltz1_conf.ckpt", "out/boltz1"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_boltz1_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=dtype,
        strict=args.strict,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted Boltz-1 checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


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
    r"""
    Convert a score-model ``atom_attention_encoder.*`` key, splitting invariant atom-encoder
    parameters into ``diffusion_conditioning`` and keeping the noisy-position projection in the score model.

    Args:
        key: The ``atom_attention_encoder``-relative key.
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches.
    """

    atom_key = _convert_atom_encoder_base_key(key)
    if atom_key is not None:
        if atom_key.startswith("atom_attention_encoder.noisy_position_projection."):
            if key == "r_to_q_trans.weight":
                if not isinstance(tensor, Tensor):
                    raise TypeError("r_to_q_trans.weight must be a tensor")
                tensor = tensor[:, :3]
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
    r"""
    Convert a score-model ``atom_attention_decoder.*`` key (per-layer transformer, token projection,
    and the atom position-update projection), routing the pair bias into ``diffusion_conditioning``.

    Args:
        key: The ``atom_attention_decoder``-relative key.
        tensor: The associated parameter value.

    Returns:
        List of ``(target_key, value)`` pairs; empty when no rule matches.
    """

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
    r"""
    Map a base atom-encoder key via the shared Boltz-2 rules, plus the noisy-position projection rename.

    Args:
        key: An atom-encoder-relative key.

    Returns:
        The renamed key (prefixed with ``atom_encoder.``), or ``None`` when no rule matches.
    """

    new = boltz2_convert.convert_keys(key, boltz2_convert._ATOM_ENCODER_RULES)
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
    r"""
    Convert one per-layer conditioned-transformer key, splitting the pair-bias projection out to its own
    per-block namespace and remapping the remaining layer parameters via the shared Boltz-2 rules.

    Args:
        key: The full key to inspect (must start with ``old_prefix`` to match).
        old_prefix: The checkpoint layer prefix ending in ``layers.`` before the block index.
        layer_prefix: Target prefix for the transformer-layer parameters, indexed by block.
        pair_bias_prefix: Target prefix for the extracted pair-bias projection, indexed by block.
        tensor: The associated parameter value.

    Returns:
        A single ``(target_key, value)`` pair, or ``None`` when the key does not match this layer family.
    """

    if not key.startswith(old_prefix):
        return None
    index, _, remainder = key[len(old_prefix) :].partition(".")
    pair_bias = _convert_pair_bias_projection_suffix(remainder)
    if pair_bias is not None:
        return (f"{pair_bias_prefix}{index}.{pair_bias}", tensor)
    new = boltz2_convert.convert_keys(remainder, boltz2_convert.CONDITIONED_TRANSFORMER_LAYER_RULES)
    return None if new is None else (f"{layer_prefix}{index}.{new}", tensor)


def _convert_boltz1_confidence_heads_key(key: str) -> str | None:
    r"""
    Rename a ``confidence_heads.*`` projection key (pLDDT/PAE/PDE/resolved logits).

    Args:
        key: A ``confidence_heads``-relative key.

    Returns:
        The renamed key, or ``None`` when no rule matches.
    """

    rules = {
        "to_pde_logits.": "pde_projection.",
        "to_plddt_logits.": "plddt_projection.",
        "to_resolved_logits.": "resolved_projection.",
        "to_pae_logits.": "pae_projection.",
    }
    for old, new in rules.items():
        if key.startswith(old):
            return key.replace(old, new, 1)
    return None


def _convert_conditioned_transition_block_key(key: str) -> str | None:
    r"""
    Rename a conditioned-transition-block key (AdaLN scale/bias, swish gate, expand/output projections).

    The affine-free ``adaln.a_norm.*`` parameters carry no learnable weights and are intentionally dropped.

    Args:
        key: A ``transition_block``-relative key.

    Returns:
        The renamed key, or ``None`` when the key is dropped or unmatched.
    """

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
    r"""
    Extract the pair-bias projection suffix (``proj_z.0``/``proj_z.1``) from a pair-biased-attention key.

    Args:
        key: A per-layer transformer remainder key.

    Returns:
        The ``0.``/``1.``-prefixed projection suffix, or ``None`` when the key is not a pair-bias projection.
    """

    if key.startswith("pair_bias_attn.proj_z.0."):
        return "0." + key[len("pair_bias_attn.proj_z.0.") :]
    if key.startswith("pair_bias_attn.proj_z.1."):
        return "1." + key[len("pair_bias_attn.proj_z.1.") :]
    return None


def _prefix_index(index: str, items: list[tuple[str, object]]) -> list[tuple[str, object]]:
    r"""
    Prepend a block-index component to each key in ``items``.

    Args:
        index: The block index string to prepend.
        items: ``(key, value)`` pairs to re-key.

    Returns:
        The same values keyed as ``"{index}.{key}"``.
    """

    return [(f"{index}.{key}", value) for key, value in items]


def _add_zero_atom_projection_biases(state: dict[str, object]) -> None:
    r"""
    Synthesize zero biases for atom-feature projections that the checkpoint stored bias-free.

    OFoldX's atom-feature ``Linear`` carries a bias while the Boltz-1 checkpoint does not; this adds a
    matching zero bias in place so ``load_state_dict`` finds no missing key.

    Args:
        state: The (mutated) converted parameter mapping to backfill zero biases into.
    """

    for prefix in (
        "input_embedder.atom_encoder.atom_feature_projection",
        "diffusion_conditioning.atom_encoder.atom_feature_projection",
    ):
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        weight = state.get(weight_key)
        if bias_key not in state and isinstance(weight, Tensor):
            state[bias_key] = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)


def _is_ignorable_key(key: str) -> bool:
    r"""
    Report whether a source key is a non-parameter checkpoint artifact safe to drop.

    Args:
        key: A source checkpoint key.

    Returns:
        ``True`` for EMA/loss/metric buffers and container keys that carry no model parameters.
    """

    if key.startswith(("ema.", "loss.", "train_", "val_", "test_")):
        return True
    if key in {"state_dict", "hyper_parameters"}:
        return True
    # The per-layer ``output_projection.0`` is a tied duplicate of ``output_projection_linear``
    # (Boltz-2 drops it too); dropping it here is intentional, not an unmapped key.
    if re.search(r"\.layers\.\d+\.output_projection\.0\.(weight|bias)$", key):
        return True
    return False


if __name__ == "__main__":
    main()
