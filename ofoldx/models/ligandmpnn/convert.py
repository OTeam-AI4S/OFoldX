r"""
LigandMPNN checkpoint conversion helpers.

Translate the released LigandMPNN ``.pt`` checkpoints into OFoldX-native config objects and state dicts.
Per the naming guidelines, upstream tensor names live only here: the key maps rewrite the terse official
names (``W_e``, ``features.embeddings.linear``, ...) into the house vocabulary used by the model body, and the
sequence backbone (``ligandmpnn``) and the side-chain LigandMPNNPacker (``ligandmpnn_sc``) are kept strictly apart.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .config import LigandMPNNConfig, LigandMPNNPackerConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def config_from_ligandmpnn_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_variant: str | None = None,
) -> LigandMPNNConfig:
    r"""
    Build a native config from an official LigandMPNN sequence checkpoint.

    Reads the graph connectivity (``num_edges``), the ligand-context budget (``atom_context_num``), and the
    training noise level off the checkpoint metadata; unspecified fields fall back to the released defaults.

    Args:
        checkpoint: An official checkpoint dictionary (as saved by ``torch.save``).
        checkpoint_variant: Optional label recorded on the config to identify the source checkpoint.

    Returns:
        A [`LigandMPNNConfig`][ofoldx.models.ligandmpnn.config.LigandMPNNConfig] matching the checkpoint.

    Raises:
        ValueError: If the checkpoint is a ``ligandmpnn_sc`` side-chain LigandMPNNPacker rather than a sequence model.

    Examples:
        >>> from ofoldx.models.ligandmpnn.convert import config_from_ligandmpnn_checkpoint
        >>> config = config_from_ligandmpnn_checkpoint({"num_edges": 48})
        >>> config.k_neighbors
        48
        >>> config.architecture
        'ligandmpnn'
    """

    state_dict = checkpoint.get("model_state_dict", {})
    if isinstance(state_dict, Mapping) and _looks_like_side_chain_packer(state_dict):
        raise ValueError(
            "ligandmpnn_sc checkpoints are side-chain LigandMPNNPacker weights, not LigandMPNN sequence weights"
        )
    return LigandMPNNConfig(
        {
            "architecture": "ligandmpnn",
            "checkpoint_variant": checkpoint_variant,
            "k_neighbors": int(checkpoint.get("num_edges", 32)),
            "atom_context_num": int(checkpoint.get("atom_context_num", 25)),
            "noise_level": float(checkpoint["noise_level"]) if "noise_level" in checkpoint else None,
        }
    )


def config_from_packer_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_variant: str | None = None,
) -> LigandMPNNPackerConfig:
    r"""
    Build a native config from an official ``ligandmpnn_sc`` side-chain LigandMPNNPacker checkpoint.

    Args:
        checkpoint: An official checkpoint dictionary (as saved by ``torch.save``).
        checkpoint_variant: Optional label recorded on the config to identify the source checkpoint.

    Returns:
        A [`LigandMPNNPackerConfig`][ofoldx.models.ligandmpnn.config.LigandMPNNPackerConfig] matching the checkpoint.

    Raises:
        ValueError: If the checkpoint does not look like side-chain LigandMPNNPacker weights.

    Examples:
        >>> config = config_from_packer_checkpoint({"model_state_dict": {"W_torsions.weight": object()}})
        >>> config.architecture
        'ligandmpnn_packer'
    """

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping) or not _looks_like_side_chain_packer(state_dict):
        raise ValueError("checkpoint is not a ligandmpnn_sc side-chain LigandMPNNPacker checkpoint")
    return LigandMPNNPackerConfig(
        {
            "architecture": "ligandmpnn_packer",
            "checkpoint_variant": checkpoint_variant,
            "hidden_size": _tensor_dim(state_dict, "W_e.weight", 0, 128),
            "num_encoder_layers": _layer_count(state_dict, "encoder_layers", 3),
            "num_decoder_layers": _layer_count(state_dict, "decoder_layers", 3),
            "num_rbf": _tensor_dim(state_dict, "features.W_XY_project_down1.weight", 0, 16),
            "k_neighbors": int(checkpoint.get("num_edges", 32)),
            "atom_context_num": int(checkpoint.get("atom_context_num", 16)),
            "num_mix": _tensor_dim(state_dict, "W_torsions.weight", 0, 36) // 12,
            "num_positional_embeddings": _tensor_dim(state_dict, "features.positional_embeddings.linear.weight", 0, 16),
            "noise_level": float(checkpoint["noise_level"]) if "noise_level" in checkpoint else None,
        }
    )


def convert_ligandmpnn_state_dict(
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    dtype: torch.dtype | None = None,
    model: nn.Module | None = None,
) -> dict[str, Tensor]:
    r"""
    Convert an official LigandMPNN sequence checkpoint to native OFoldX tensor names.

    Rewrites every tensor key via [`convert_original_state_dict_key`]
    [ofoldx.models.ligandmpnn.convert.convert_original_state_dict_key] and optionally validates that the
    result loads strictly into a given model. Non-tensor entries in the source dict are dropped.

    Args:
        checkpoint_or_state_dict: A checkpoint dict (with ``model_state_dict``) or a raw state dict.
        dtype: Optional dtype to cast floating-point tensors to; integer tensors are left unchanged.
        model: Optional model instance; when given, the converted keys are checked strict-loadable.

    Returns:
        State dict keyed by OFoldX tensor names.

    Raises:
        TypeError: If the input is neither a state dict nor a mapping with ``model_state_dict``.
        ValueError: If the checkpoint is a ``ligandmpnn_sc`` side-chain LigandMPNNPacker rather than a sequence model.
        KeyError: If a checkpoint tensor name is not recognized.
        RuntimeError: If ``model`` is given and the converted keys are not strict-loadable.
    """

    state_dict = checkpoint_or_state_dict.get("model_state_dict", checkpoint_or_state_dict)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must be a state dict or contain a model_state_dict mapping")
    if _looks_like_side_chain_packer(state_dict):
        raise ValueError(
            "ligandmpnn_sc checkpoints are side-chain LigandMPNNPacker weights, not LigandMPNN sequence weights"
        )

    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        target_key = convert_original_state_dict_key(key)
        converted[target_key] = value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
    if model is not None:
        _validate_converted_state_dict(model, converted)
    return converted


def convert_original_state_dict_key(key: str) -> str:
    r"""
    Translate one official LigandMPNN sequence tensor name to its native OFoldX name.

    Handles the feature-encoder table, the top-level projection/embedding table, and the per-layer encoder /
    decoder / context / atom-context groups. This is the single place upstream key jargon is resolved.

    Args:
        key: An official checkpoint tensor name (e.g. ``"W_e.weight"`` or ``"encoder_layers.0.W1.weight"``).

    Returns:
        The corresponding OFoldX tensor name.

    Raises:
        KeyError: If the key does not match any known feature, top-level, or per-layer pattern.

    Examples:
        >>> from ofoldx.models.ligandmpnn.convert import convert_original_state_dict_key
        >>> convert_original_state_dict_key("W_e.weight")
        'backbone.edge_embedding.weight'
        >>> convert_original_state_dict_key("encoder_layers.0.W1.weight")
        'backbone.encoder_layers.0.node_message.input.weight'
    """

    feature_replacements = {
        "features.embeddings.linear.weight": "backbone.feature_encoder.relative_position.linear.weight",
        "features.embeddings.linear.bias": "backbone.feature_encoder.relative_position.linear.bias",
        "features.edge_embedding.weight": "backbone.feature_encoder.edge_projection.weight",
        "features.norm_edges.weight": "backbone.feature_encoder.edge_norm.weight",
        "features.norm_edges.bias": "backbone.feature_encoder.edge_norm.bias",
        "features.node_project_down.weight": "backbone.feature_encoder.atom_projection.weight",
        "features.node_project_down.bias": "backbone.feature_encoder.atom_projection.bias",
        "features.norm_nodes.weight": "backbone.feature_encoder.atom_norm.weight",
        "features.norm_nodes.bias": "backbone.feature_encoder.atom_norm.bias",
        "features.type_linear.weight": "backbone.feature_encoder.element_projection.weight",
        "features.type_linear.bias": "backbone.feature_encoder.element_projection.bias",
        "features.y_nodes.weight": "backbone.feature_encoder.context_node_projection.weight",
        "features.y_edges.weight": "backbone.feature_encoder.context_edge_projection.weight",
        "features.norm_y_edges.weight": "backbone.feature_encoder.context_edge_norm.weight",
        "features.norm_y_edges.bias": "backbone.feature_encoder.context_edge_norm.bias",
        "features.norm_y_nodes.weight": "backbone.feature_encoder.context_node_norm.weight",
        "features.norm_y_nodes.bias": "backbone.feature_encoder.context_node_norm.bias",
    }
    if key in feature_replacements:
        return feature_replacements[key]
    top_level_replacements = {
        "W_e.weight": "backbone.edge_embedding.weight",
        "W_e.bias": "backbone.edge_embedding.bias",
        "W_s.weight": "backbone.token_embedding.weight",
        "W_v.weight": "backbone.atom_context_embedding.weight",
        "W_v.bias": "backbone.atom_context_embedding.bias",
        "W_c.weight": "backbone.residue_context_input.weight",
        "W_c.bias": "backbone.residue_context_input.bias",
        "W_nodes_y.weight": "backbone.atom_node_embedding.weight",
        "W_nodes_y.bias": "backbone.atom_node_embedding.bias",
        "W_edges_y.weight": "backbone.atom_edge_embedding.weight",
        "W_edges_y.bias": "backbone.atom_edge_embedding.bias",
        "V_C.weight": "backbone.context_output.weight",
        "V_C_norm.weight": "backbone.context_norm.weight",
        "V_C_norm.bias": "backbone.context_norm.bias",
        "W_out.weight": "backbone.output_projection.weight",
        "W_out.bias": "backbone.output_projection.bias",
    }
    if key in top_level_replacements:
        return top_level_replacements[key]
    if key.startswith("encoder_layers."):
        return _convert_encoder_key(key)
    if key.startswith("decoder_layers."):
        return _convert_decoder_key(key, "backbone.decoder_layers")
    if key.startswith("context_encoder_layers."):
        return _convert_decoder_key(key, "backbone.context_layers")
    if key.startswith("y_context_encoder_layers."):
        return _convert_decoder_key(key, "backbone.atom_context_layers")
    raise KeyError(f"unrecognized LigandMPNN checkpoint key: {key}")


def _looks_like_side_chain_packer(state_dict: Mapping[str, Any]) -> bool:
    r"""Return whether a state dict carries the ``ligandmpnn_sc`` LigandMPNNPacker-only tensor names.

    Args:
        state_dict: A raw checkpoint state dict.

    Returns:
        ``True`` if any LigandMPNNPacker-exclusive key (torsion / side-chain / enc-edge) is present.
    """

    return any(key.startswith("features.enc_edge_embedding.") for key in state_dict) or any(
        key.startswith(("W_torsions.", "W_f.", "W_v_sc.", "linear_down.")) for key in state_dict
    )


def _tensor_dim(state_dict: Mapping[str, Any], key: str, dim: int, default: int) -> int:
    r"""Return a tensor dimension from ``state_dict`` if present, otherwise ``default``."""

    value = state_dict.get(key)
    if not isinstance(value, Tensor):
        return default
    return int(value.shape[dim])


def _layer_count(state_dict: Mapping[str, Any], prefix: str, default: int) -> int:
    r"""Infer the number of indexed layers in an official checkpoint group."""

    layer_indices: list[int] = []
    for key in state_dict:
        if not key.startswith(prefix + "."):
            continue
        parts = key.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            layer_indices.append(int(parts[1]))
    return max(layer_indices) + 1 if layer_indices else default


_PACKER_TOP_LEVEL = {
    "W_e": "edge_embedding",
    "W_v": "node_embedding",
    "W_f": "decode_edge_embedding",
    "W_v_sc": "side_chain_node_embedding",
    "linear_down": "linear_down",
    "W_torsions": "torsion_projection",
    "W_c": "residue_context_input",
    "W_e_context": "context_edge_embedding",
    "W_nodes_y": "atom_node_embedding",
    "W_edges_y": "atom_edge_embedding",
    "V_C": "context_output",
    "V_C_norm": "context_norm",
    "features.enc_edge_embedding": "features.edge_projection",
    "features.enc_norm_edges": "features.edge_norm",
    "features.enc_node_embedding": "features.node_projection",
    "features.enc_norm_nodes": "features.node_norm",
    "features.W_XY_project_down1": "features.context_down_projection",
    "features.dec_edge_embedding1": "features.decode_edge_projection",
    "features.dec_norm_edges1": "features.decode_edge_norm",
    "features.dec_node_embedding1": "features.decode_node_projection",
    "features.dec_norm_nodes1": "features.decode_node_norm",
    "features.node_project_down": "features.context_node_projection",
    "features.norm_nodes": "features.context_node_norm",
    "features.type_linear": "features.element_projection",
    "features.y_nodes": "features.atom_node_projection",
    "features.y_edges": "features.atom_edge_projection",
    "features.norm_y_edges": "features.atom_edge_norm",
    "features.norm_y_nodes": "features.atom_node_norm",
    "features.positional_embeddings.linear": "features.relative_position.linear",
}
_PACKER_ENCODER_SUBLAYER = {
    "norm1": "node_norm",
    "norm2": "feed_forward_norm",
    "norm3": "edge_norm",
    "W1": "node_message.input",
    "W2": "node_message.middle",
    "W3": "node_message.output",
    "W11": "edge_message.input",
    "W12": "edge_message.middle",
    "W13": "edge_message.output",
    "dense.W_in": "feed_forward.input",
    "dense.W_out": "feed_forward.output",
}
_PACKER_DECODER_SUBLAYER = {
    "norm1": "message_norm",
    "norm2": "feed_forward_norm",
    "W1": "message.input",
    "W2": "message.middle",
    "W3": "message.output",
    "dense.W_in": "feed_forward.input",
    "dense.W_out": "feed_forward.output",
}
_PACKER_LAYER_GROUPS = {
    "encoder_layers": ("encoder_layers", _PACKER_ENCODER_SUBLAYER),
    "decoder_layers": ("decoder_layers", _PACKER_DECODER_SUBLAYER),
    "context_encoder_layers": ("context_layers", _PACKER_DECODER_SUBLAYER),
    "y_context_encoder_layers": ("atom_context_layers", _PACKER_DECODER_SUBLAYER),
}


def convert_packer_state_dict(
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    dtype: torch.dtype | None = None,
    model: nn.Module | None = None,
) -> dict[str, Tensor]:
    r"""
    Convert an official LigandMPNN side-chain LigandMPNNPacker checkpoint to native OFoldX keys.

    The LigandMPNNPacker is the ``ligandmpnn_sc`` checkpoint family (Von Mises torsion weights), distinct from
    the sequence backbone. Load the result into
    [`LigandMPNNPacker`][ofoldx.models.ligandmpnn.model.LigandMPNNPacker].

    Args:
        checkpoint_or_state_dict: A checkpoint dict (with ``model_state_dict``) or a raw state dict.
        dtype: Optional dtype to cast floating-point tensors to.
        model: Optional LigandMPNNPacker instance; when given, the converted keys are validated strict-loadable.

    Returns:
        State dict with OFoldX LigandMPNNPacker tensor names.

    Raises:
        TypeError: If the input is neither a state dict nor a mapping with ``model_state_dict``.
        KeyError: If a checkpoint tensor name is not recognized.
        RuntimeError: If ``model`` is given and the converted keys are not strict-loadable.
    """

    state_dict = checkpoint_or_state_dict.get("model_state_dict", checkpoint_or_state_dict)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint must be a state dict or contain a model_state_dict mapping")
    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        target_key = _convert_packer_key(key)
        converted[target_key] = value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
    if model is not None:
        _validate_converted_state_dict(model, converted)
    return converted


def _convert_packer_key(key: str) -> str:
    r"""Translate one official LigandMPNNPacker tensor name to the native OFoldX name.

    Args:
        key: An official ``ligandmpnn_sc`` LigandMPNNPacker tensor name.

    Returns:
        The corresponding native LigandMPNNPacker tensor name.

    Raises:
        KeyError: If the key matches no known LigandMPNNPacker layer-group or top-level pattern.
    """

    for prefix, (target, sublayer) in _PACKER_LAYER_GROUPS.items():
        if key.startswith(prefix + "."):
            parts = key.split(".")
            remainder = ".".join(parts[2:])
            for original, replacement in sublayer.items():
                if remainder.startswith(original + "."):
                    return f"{target}.{parts[1]}.{replacement}{remainder[len(original):]}"
            raise KeyError(f"unrecognized LigandMPNNPacker sublayer key: {key}")
    for original, replacement in _PACKER_TOP_LEVEL.items():
        if key in (f"{original}.weight", f"{original}.bias"):
            return replacement + key[len(original) :]
    raise KeyError(f"unrecognized LigandMPNNPacker checkpoint key: {key}")


def _convert_encoder_key(key: str) -> str:
    r"""Translate one official ``encoder_layers.<i>.*`` tensor name to its native name.

    Args:
        key: An official encoder-layer tensor name.

    Returns:
        The corresponding ``backbone.encoder_layers.<i>.*`` name.

    Raises:
        KeyError: If the key is malformed or its sublayer suffix is unknown.
    """

    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"malformed encoder key: {key}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
        "norm1.weight": "node_norm.weight",
        "norm1.bias": "node_norm.bias",
        "norm2.weight": "feed_forward_norm.weight",
        "norm2.bias": "feed_forward_norm.bias",
        "norm3.weight": "edge_norm.weight",
        "norm3.bias": "edge_norm.bias",
        "W1.weight": "node_message.input.weight",
        "W1.bias": "node_message.input.bias",
        "W2.weight": "node_message.middle.weight",
        "W2.bias": "node_message.middle.bias",
        "W3.weight": "node_message.output.weight",
        "W3.bias": "node_message.output.bias",
        "W11.weight": "edge_message.input.weight",
        "W11.bias": "edge_message.input.bias",
        "W12.weight": "edge_message.middle.weight",
        "W12.bias": "edge_message.middle.bias",
        "W13.weight": "edge_message.output.weight",
        "W13.bias": "edge_message.output.bias",
        "dense.W_in.weight": "feed_forward.input.weight",
        "dense.W_in.bias": "feed_forward.input.bias",
        "dense.W_out.weight": "feed_forward.output.weight",
        "dense.W_out.bias": "feed_forward.output.bias",
    }
    if suffix not in replacements:
        raise KeyError(f"unrecognized encoder key: {key}")
    return f"backbone.encoder_layers.{layer}.{replacements[suffix]}"


def _convert_decoder_key(key: str, target_prefix: str) -> str:
    r"""Translate one official decoder-style ``*_layers.<i>.*`` tensor name to its native name.

    Shared by the decoder, context-encoder, and atom-context-encoder groups, which use the same sublayer
    layout under different native prefixes.

    Args:
        key: An official decoder-style layer tensor name.
        target_prefix: The native module prefix to emit (e.g. ``"backbone.decoder_layers"``).

    Returns:
        The corresponding ``<target_prefix>.<i>.*`` name.

    Raises:
        KeyError: If the key is malformed or its sublayer suffix is unknown.
    """

    parts = key.split(".")
    if len(parts) < 4:
        raise KeyError(f"malformed decoder key: {key}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    replacements = {
        "norm1.weight": "message_norm.weight",
        "norm1.bias": "message_norm.bias",
        "norm2.weight": "feed_forward_norm.weight",
        "norm2.bias": "feed_forward_norm.bias",
        "W1.weight": "message.input.weight",
        "W1.bias": "message.input.bias",
        "W2.weight": "message.middle.weight",
        "W2.bias": "message.middle.bias",
        "W3.weight": "message.output.weight",
        "W3.bias": "message.output.bias",
        "dense.W_in.weight": "feed_forward.input.weight",
        "dense.W_in.bias": "feed_forward.input.bias",
        "dense.W_out.weight": "feed_forward.output.weight",
        "dense.W_out.bias": "feed_forward.output.bias",
    }
    if suffix not in replacements:
        raise KeyError(f"unrecognized decoder key: {key}")
    return f"{target_prefix}.{layer}.{replacements[suffix]}"


def _validate_converted_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    r"""Assert that a converted state dict loads strictly into ``model``.

    Args:
        model: The target module whose ``state_dict`` keys/shapes define the contract.
        state_dict: The converted state dict to check.

    Raises:
        RuntimeError: If any key is missing, unexpected, or shape-mismatched, with a truncated report.
    """

    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    mismatched = [
        (key, tuple(model_state[key].shape), tuple(state_dict[key].shape))
        for key in sorted(set(model_state) & set(state_dict))
        if tuple(model_state[key].shape) != tuple(state_dict[key].shape)
    ]
    if not missing and not unexpected and not mismatched:
        return
    lines = [
        "LigandMPNN checkpoint is not strict-loadable after conversion",
        f"missing: {len(missing)}",
        f"unexpected: {len(unexpected)}",
        f"shape_mismatch: {len(mismatched)}",
    ]
    lines.extend(f"  missing {key}" for key in missing[:20])
    lines.extend(f"  unexpected {key}" for key in unexpected[:20])
    lines.extend(
        f"  mismatched {key}: model{model_shape} checkpoint{ckpt_shape}"
        for key, model_shape, ckpt_shape in mismatched[:20]
    )
    raise RuntimeError("\n".join(lines))


def load_official_ligandmpnn_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    checkpoint_variant: str | None = None,
) -> tuple[LigandMPNNConfig, dict[str, Tensor]]:
    r"""
    Load an official LigandMPNN sequence ``.pt`` checkpoint into an OFoldX config and weights.

    Reads the checkpoint from disk, derives the config from its metadata, and converts the tensor keys. The
    file stem is used as the default ``checkpoint_variant`` label. This is the convenience entry point for
    materializing a released checkpoint; the returned state dict loads into ``LigandMPNNDesigner``.

    Args:
        checkpoint_path: Path to the official ``.pt`` checkpoint.
        map_location: Device to map the loaded tensors onto.
        checkpoint_variant: Optional variant label; defaults to the checkpoint file stem.

    Returns:
        Tuple ``(config, state_dict)`` with an OFoldX config and native-keyed weights.

    Raises:
        ValueError: If the checkpoint is a ``ligandmpnn_sc`` side-chain LigandMPNNPacker rather than a sequence model.
        KeyError: If a checkpoint tensor name is not recognized.

    Examples:
        >>> from ofoldx.models.ligandmpnn.convert import load_official_ligandmpnn_checkpoint
        >>> config, state_dict = load_official_ligandmpnn_checkpoint("ligandmpnn_v_32_010_25.pt")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    variant = checkpoint_variant if checkpoint_variant is not None else checkpoint_path.stem
    return (
        config_from_ligandmpnn_checkpoint(checkpoint, checkpoint_variant=variant),
        convert_ligandmpnn_state_dict(checkpoint),
    )


def load_official_ligandmpnn_packer_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    checkpoint_variant: str | None = None,
) -> tuple[LigandMPNNPackerConfig, dict[str, Tensor]]:
    r"""
    Load an official ``ligandmpnn_sc`` side-chain LigandMPNNPacker checkpoint into an OFoldX config and weights.

    Args:
        checkpoint_path: Path to the official ``ligandmpnn_sc_v_*.pt`` checkpoint.
        map_location: Device to map the loaded tensors onto.
        checkpoint_variant: Optional variant label; defaults to the checkpoint file stem.

    Returns:
        Tuple ``(config, state_dict)`` with a LigandMPNNPacker config and native-keyed weights.

    Raises:
        ValueError: If the checkpoint is not a side-chain LigandMPNNPacker checkpoint.

    Examples:
        >>> load_official_ligandmpnn_packer_checkpoint("ligandmpnn_sc_v_32_002_16.pt")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    variant = checkpoint_variant if checkpoint_variant is not None else checkpoint_path.stem
    return (
        config_from_packer_checkpoint(checkpoint, checkpoint_variant=variant),
        convert_packer_state_dict(checkpoint),
    )


def convert_ligandmpnn_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    dtype: torch.dtype | None = None,
    checkpoint_variant: str | None = None,
    checkpoint_kind: str = "auto",
) -> dict[str, Any]:
    r"""
    Convert an official LigandMPNN checkpoint into a pretrained OFoldX artifact directory.

    ``checkpoint_kind="auto"`` routes sequence checkpoints to ``ligandmpnn`` and side-chain packer
    checkpoints to ``ligandmpnn_packer``.

    Args:
        checkpoint_path: Path to an official ``ligandmpnn_v_*.pt`` or ``ligandmpnn_sc_v_*.pt`` checkpoint.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        dtype: Optional floating-point dtype for persisted weights.
        checkpoint_variant: Optional label to store in the config; defaults to the checkpoint stem.
        checkpoint_kind: ``"auto"``, ``"sequence"``, or ``"packer"``.

    Returns:
        JSON-serializable conversion report.

    Raises:
        ValueError: If ``checkpoint_kind`` disagrees with the checkpoint tensor family.

    Examples:
        >>> convert_ligandmpnn_checkpoint("ligandmpnn_v_32_010_25.pt", "ligandmpnn")  # doctest: +SKIP
        >>> convert_ligandmpnn_checkpoint("ligandmpnn_sc_v_32_002_16.pt", "ligandmpnn-packer")  # doctest: +SKIP
    """

    from .model import LigandMPNNDesigner

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint_path.stem if checkpoint_variant is None else checkpoint_variant
    kind = _resolve_checkpoint_kind(checkpoint, checkpoint_kind)
    if kind == "packer":
        return convert_ligandmpnn_packer_checkpoint(
            checkpoint_path,
            output_dir,
            dtype=dtype,
            checkpoint_variant=checkpoint_variant,
        )
    config = config_from_ligandmpnn_checkpoint(checkpoint, checkpoint_variant=variant)
    model = LigandMPNNDesigner(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = convert_ligandmpnn_state_dict(checkpoint, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": config.architecture,
        "checkpoint_variant": config.checkpoint_variant,
        "tensor_count": len(state_dict),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(state_dict),
    }


def convert_ligandmpnn_packer_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    dtype: torch.dtype | None = None,
    checkpoint_variant: str | None = None,
) -> dict[str, Any]:
    r"""
    Convert an official ``ligandmpnn_sc`` side-chain LigandMPNNPacker checkpoint into a pretrained artifact.

    Args:
        checkpoint_path: Path to an official ``ligandmpnn_sc_v_*.pt`` checkpoint.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        dtype: Optional floating-point dtype for persisted weights.
        checkpoint_variant: Optional label to store in the config; defaults to the checkpoint stem.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_ligandmpnn_packer_checkpoint("ligandmpnn_sc_v_32_002_16.pt", "ligandmpnn-packer")  # doctest: +SKIP
    """

    from .model import LigandMPNNPacker

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = checkpoint_path.stem if checkpoint_variant is None else checkpoint_variant
    config = config_from_packer_checkpoint(checkpoint, checkpoint_variant=variant)
    model = LigandMPNNPacker(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = convert_packer_state_dict(checkpoint, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": config.architecture,
        "checkpoint_variant": config.checkpoint_variant,
        "tensor_count": len(state_dict),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(state_dict),
    }


def _resolve_checkpoint_kind(checkpoint: Mapping[str, Any], requested: str) -> str:
    r"""Resolve ``auto`` / explicit checkpoint family selection."""

    if requested not in {"auto", "sequence", "packer"}:
        raise ValueError("checkpoint_kind must be 'auto', 'sequence', or 'packer'")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    is_packer = isinstance(state_dict, Mapping) and _looks_like_side_chain_packer(state_dict)
    if requested == "auto":
        return "packer" if is_packer else "sequence"
    if requested == "packer" and not is_packer:
        raise ValueError("checkpoint_kind='packer' requires ligandmpnn_sc side-chain LigandMPNNPacker weights")
    if requested == "sequence" and is_packer:
        raise ValueError("checkpoint_kind='sequence' cannot convert ligandmpnn_sc side-chain LigandMPNNPacker weights")
    return requested


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""Return a sorted count of tensors per dtype name in ``state_dict``."""

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for LigandMPNN checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an official LigandMPNN .pt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--checkpoint-variant", default=None, help="Variant label to store in config.json")
    parser.add_argument(
        "--checkpoint-kind",
        choices=("auto", "sequence", "packer"),
        default="auto",
        help="Checkpoint family to convert; auto detects ligandmpnn_sc packer weights",
    )
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for LigandMPNN checkpoint conversion.

    Examples:
        >>> main(["ligandmpnn_v_32_010_25.pt", "out/ligandmpnn"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_ligandmpnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=dtype,
        checkpoint_variant=args.checkpoint_variant,
        checkpoint_kind=args.checkpoint_kind,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted LigandMPNN checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
