r"""
Checkpoint helpers for ProteinMPNN.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .config import ProteinMPNNConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_proteinmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    r"""
    Load an official ProteinMPNN checkpoint from disk.

    The released checkpoints are pickled dicts carrying a ``"model_state_dict"`` entry alongside training
    metadata (``num_edges``, ``noise_level``). This loader returns that dict unchanged; use
    [`convert_proteinmpnn_state_dict`][ofoldx.models.proteinmpnn.convert.convert_proteinmpnn_state_dict]
    to translate the weights and
    [`proteinmpnn_config_from_checkpoint`][ofoldx.models.proteinmpnn.convert.proteinmpnn_config_from_checkpoint]
    to recover the config.

    Args:
        checkpoint_path: Path to a released ``proteinmpnn_v_*.pt`` file.
        map_location: Device or device string that ``torch.load`` maps the stored tensors onto.

    Returns:
        The full checkpoint mapping, including ``"model_state_dict"`` and any training metadata.

    Raises:
        ValueError: If the file does not contain an official checkpoint with a ``"model_state_dict"`` entry.

    Examples:
        >>> checkpoint = load_proteinmpnn_checkpoint("proteinmpnn_v_48_020.pt")  # doctest: +SKIP
        >>> state_dict = convert_proteinmpnn_state_dict(checkpoint)  # doctest: +SKIP
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("expected an official checkpoint with a 'model_state_dict' entry")
    return dict(checkpoint)


def proteinmpnn_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> ProteinMPNNConfig:
    r"""
    Build a ProteinMPNN config from official checkpoint metadata.

    Starts from the default [`ProteinMPNNConfig`][ofoldx.models.proteinmpnn.config.ProteinMPNNConfig] and
    overrides only what the checkpoint records: ``num_edges`` sets ``k_neighbors``, and ``noise_level`` sets both
    ``noise_level`` and the derived ``checkpoint_variant`` tag (e.g. ``"proteinmpnn_v_48_020"``). Missing metadata
    leaves the corresponding fields at their defaults.

    Args:
        checkpoint: A loaded official checkpoint mapping (as returned by
            [`load_proteinmpnn_checkpoint`][ofoldx.models.proteinmpnn.convert.load_proteinmpnn_checkpoint]).

    Returns:
        A config whose dimensions and ``checkpoint_variant`` reflect the checkpoint metadata.

    Examples:
        >>> config = proteinmpnn_config_from_checkpoint({"num_edges": 48, "noise_level": 0.2})
        >>> config.k_neighbors
        48
        >>> config.checkpoint_variant
        'proteinmpnn_v_48_020'
    """

    config = ProteinMPNNConfig()
    num_edges = checkpoint.get("num_edges")
    if isinstance(num_edges, int):
        config.k_neighbors = num_edges
    noise_level = checkpoint.get("noise_level")
    if isinstance(noise_level, int | float):
        config.noise_level = float(noise_level)
        config.checkpoint_variant = f"proteinmpnn_v_{config.k_neighbors}_{int(round(float(noise_level) * 100)):03d}"
    state = checkpoint.get("model_state_dict")
    if isinstance(state, Mapping):
        node_embedding = state.get("features.node_embedding.weight")
        if isinstance(node_embedding, Tensor) and node_embedding.ndim == 2:
            config.node_feature_size = int(node_embedding.shape[1])
    return config


def convert_proteinmpnn_state_dict(
    checkpoint_or_state: Mapping[str, Any] | Mapping[str, Tensor],
    model: nn.Module | None = None,
) -> dict[str, Tensor]:
    r"""
    Return a strict-loadable ProteinMPNN state dict from an official checkpoint.

    Official ProteinMPNN checkpoints use the original repository module names. OFoldX keeps a native module tree,
    so conversion translates every tensor key to its native name (per
    [`convert_original_state_dict_key`][ofoldx.models.proteinmpnn.convert.convert_original_state_dict_key]).
    When ``model`` is given the result is checked for strict loadability against that model's ``state_dict``.

    Args:
        checkpoint_or_state: Either a full checkpoint mapping (with a ``"model_state_dict"`` entry) or a bare
            ``name -> tensor`` state dict.
        model: Optional target module; when provided, the converted keys and shapes are validated against it.

    Returns:
        A ``native_name -> tensor`` mapping ready for ``model.load_state_dict(..., strict=True)``.

    Raises:
        TypeError: If ``checkpoint_or_state`` is neither a mapping nor contains a mapping ``"model_state_dict"``.
        KeyError: If a checkpoint key is not a recognized official ProteinMPNN tensor name.
        RuntimeError: If ``model`` is given and the converted state dict is not strict-loadable.

    Examples:
        >>> state_dict = convert_proteinmpnn_state_dict(checkpoint, model)  # doctest: +SKIP
        >>> model.load_state_dict(state_dict, strict=True)  # doctest: +SKIP
    """

    state = checkpoint_or_state.get("model_state_dict", checkpoint_or_state)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint_or_state must be a mapping or contain a mapping 'model_state_dict'")

    converted = {
        convert_original_state_dict_key(key): value
        for key, value in state.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if model is not None:
        _validate_converted_state_dict(model, converted)
    return converted


def convert_proteinmpnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    checkpoint_variant: str | None = None,
) -> dict[str, Any]:
    r"""
    Convert an official ProteinMPNN checkpoint into a pretrained OFoldX artifact directory.

    Args:
        checkpoint_path: Path to a released ``proteinmpnn_v_*.pt`` checkpoint.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        dtype: Optional floating-point dtype for persisted weights.
        checkpoint_variant: Optional label recorded in ``config.json``; defaults to the checkpoint stem.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_proteinmpnn_checkpoint("proteinmpnn_v_48_020.pt", "proteinmpnn")  # doctest: +SKIP
    """

    from .model import ProteinMPNNDesigner

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = load_proteinmpnn_checkpoint(checkpoint_path)
    config = proteinmpnn_config_from_checkpoint(checkpoint)
    config.checkpoint_variant = checkpoint_path.stem if checkpoint_variant is None else checkpoint_variant
    model = ProteinMPNNDesigner(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = _cast_floating_tensors(convert_proteinmpnn_state_dict(checkpoint), dtype)
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


def _cast_floating_tensors(state_dict: Mapping[str, Tensor], dtype: torch.dtype | None) -> dict[str, Tensor]:
    r"""Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""Return a sorted count of tensors per dtype name in ``state_dict``."""

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for ProteinMPNN checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an official ProteinMPNN .pt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--checkpoint-variant", default=None, help="Variant label to store in config.json")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for ProteinMPNN checkpoint conversion.

    Examples:
        >>> main(["proteinmpnn_v_48_020.pt", "out/proteinmpnn"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_proteinmpnn_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=dtype,
        checkpoint_variant=args.checkpoint_variant,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted ProteinMPNN checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


def convert_original_state_dict_key(key: str) -> str:
    r"""
    Translate one official ProteinMPNN tensor name to the native OFoldX name.

    Top-level feature/embedding/output tensors are mapped by an explicit table; per-layer encoder and decoder
    tensors dispatch to
    [`_convert_encoder_key`][ofoldx.models.proteinmpnn.convert._convert_encoder_key] and
    [`_convert_decoder_key`][ofoldx.models.proteinmpnn.convert._convert_decoder_key]. This is the single place
    upstream repository jargon (``W_e``, ``W1``, ``norm1``, ...) is confined; the rest of the model speaks the
    house vocabulary.

    Args:
        key: An official ProteinMPNN state-dict key.

    Returns:
        The corresponding native OFoldX parameter name (rooted at ``backbone.``).

    Raises:
        KeyError: If ``key`` does not correspond to any known ProteinMPNN tensor.

    Examples:
        >>> convert_original_state_dict_key("W_s.weight")
        'backbone.token_embedding.weight'
        >>> convert_original_state_dict_key("encoder_layers.0.norm1.weight")
        'backbone.encoder_layers.0.node_norm.weight'
    """

    if key == "features.embeddings.linear.weight":
        return "backbone.feature_encoder.relative_position.linear.weight"
    if key == "features.embeddings.linear.bias":
        return "backbone.feature_encoder.relative_position.linear.bias"
    if key == "features.edge_embedding.weight":
        return "backbone.feature_encoder.edge_projection.weight"
    if key == "features.node_embedding.weight":
        return "backbone.feature_encoder.node_projection.weight"
    if key == "features.norm_nodes.weight":
        return "backbone.feature_encoder.node_norm.weight"
    if key == "features.norm_nodes.bias":
        return "backbone.feature_encoder.node_norm.bias"
    if key == "features.norm_edges.weight":
        return "backbone.feature_encoder.edge_norm.weight"
    if key == "features.norm_edges.bias":
        return "backbone.feature_encoder.edge_norm.bias"
    if key == "W_e.weight":
        return "backbone.edge_embedding.weight"
    if key == "W_e.bias":
        return "backbone.edge_embedding.bias"
    if key == "W_v.weight":
        return "backbone.node_embedding.weight"
    if key == "W_v.bias":
        return "backbone.node_embedding.bias"
    if key == "W_s.weight":
        return "backbone.token_embedding.weight"
    if key == "W_out.weight":
        return "backbone.output_projection.weight"
    if key == "W_out.bias":
        return "backbone.output_projection.bias"
    if key.startswith("encoder_layers."):
        return _convert_encoder_key(key)
    if key.startswith("decoder_layers."):
        return _convert_decoder_key(key)
    raise KeyError(f"unrecognized ProteinMPNN checkpoint key: {key}")


def _convert_encoder_key(key: str) -> str:
    r"""
    Map one ``encoder_layers.<i>.<tensor>`` official key to its native encoder-layer name.

    Args:
        key: An official encoder-layer state-dict key.

    Returns:
        The native ``backbone.encoder_layers.<i>.<tensor>`` name.

    Raises:
        KeyError: If the key is malformed or its per-layer tensor suffix is unrecognized.
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


def _convert_decoder_key(key: str) -> str:
    r"""
    Map one ``decoder_layers.<i>.<tensor>`` official key to its native decoder-layer name.

    Args:
        key: An official decoder-layer state-dict key.

    Returns:
        The native ``backbone.decoder_layers.<i>.<tensor>`` name.

    Raises:
        KeyError: If the key is malformed or its per-layer tensor suffix is unrecognized.
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
    return f"backbone.decoder_layers.{layer}.{replacements[suffix]}"


def _validate_converted_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    r"""
    Assert that a converted state dict loads strictly into ``model``.

    Compares the converted keys and tensor shapes against ``model.state_dict()`` and, on any mismatch, raises a
    report listing (up to 20 each of) the missing, unexpected, and shape-mismatched keys.

    Args:
        model: Target module the converted weights should load into.
        state_dict: The converted ``native_name -> tensor`` mapping.

    Raises:
        RuntimeError: If any key is missing, unexpected, or shape-mismatched.
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
        "ProteinMPNN checkpoint is not strict-loadable after conversion",
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


if __name__ == "__main__":
    main()
