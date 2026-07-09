r"""
Checkpoint helpers for NA-MPNN.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .config import NAMPNNConfig

OFFICIAL_CHECKPOINTS = {
    "design": {
        "path": Path("models/design_model/s_19137.pt"),
        "url": "https://github.com/baker-laboratory/NA-MPNN/raw/main/models/design_model/s_19137.pt",
        "checkpoint_variant": "nampnn-design",
    },
    "specificity": {
        "path": Path("models/specificity_model/s_70114.pt"),
        "url": "https://github.com/baker-laboratory/NA-MPNN/raw/main/models/specificity_model/s_70114.pt",
        "checkpoint_variant": "nampnn-specificity",
    },
}

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_official_nampnn_checkpoint(
    variant: str,
    *,
    checkpoint_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
) -> Path:
    r"""
    Resolve an official NA-MPNN checkpoint path, downloading it when necessary.

    Args:
        variant: Official checkpoint variant, either ``"design"`` or ``"specificity"``.
        checkpoint_path: Explicit checkpoint path. When set, no default lookup or download is attempted.
        cache_dir: Optional cache directory used for GitHub downloads.

    Returns:
        Local checkpoint path.

    Raises:
        ValueError: If ``variant`` is unknown.

    Examples:
        >>> path = resolve_official_nampnn_checkpoint("design")  # doctest: +SKIP
    """

    if variant not in OFFICIAL_CHECKPOINTS:
        raise ValueError(f"unknown NA-MPNN checkpoint variant {variant!r}")
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    metadata = OFFICIAL_CHECKPOINTS[variant]
    cache_root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "ofoldx" / "nampnn"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / cast(Path, metadata["path"]).name
    if cached.exists():
        return cached
    urllib.request.urlretrieve(str(metadata["url"]), cached)
    return cached


def load_nampnn_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    r"""
    Load an official NA-MPNN checkpoint from disk.

    Args:
        checkpoint_path: Path to an official ``s_*.pt`` checkpoint.
        map_location: Device or device string used by ``torch.load``.

    Returns:
        The full checkpoint mapping.

    Raises:
        ValueError: If the checkpoint does not carry ``"model_state_dict"``.

    Examples:
        >>> checkpoint = load_nampnn_checkpoint("s_19137.pt")  # doctest: +SKIP
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("expected an official NA-MPNN checkpoint with a 'model_state_dict' entry")
    return dict(checkpoint)


def nampnn_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_variant: str | None = None,
) -> NAMPNNConfig:
    r"""
    Build an NA-MPNN config from official checkpoint tensor shapes.

    Args:
        checkpoint: Loaded official checkpoint.
        checkpoint_variant: Optional variant label recorded in the config.

    Returns:
        Config matching the checkpoint architecture.

    Examples:
        >>> config = nampnn_config_from_checkpoint({"model_state_dict": {}})
        >>> config.architecture
        'nampnn'
    """

    config = NAMPNNConfig()
    config.checkpoint_variant = checkpoint_variant
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        return config
    token_embedding = state.get("W_s.weight")
    if isinstance(token_embedding, Tensor) and token_embedding.ndim == 2:
        config.vocab_size = int(token_embedding.shape[0])
        config.hidden_size = int(token_embedding.shape[1])
    node_projection = state.get("features.node_embedding.weight")
    if isinstance(node_projection, Tensor) and node_projection.ndim == 2:
        config.node_feature_size = int(node_projection.shape[0])
        config.num_polymer_types = int(node_projection.shape[1])
    edge_projection = state.get("features.edge_embedding.weight")
    if isinstance(edge_projection, Tensor) and edge_projection.ndim == 2:
        config.edge_feature_size = int(edge_projection.shape[0])
    num_encoder_layers = _count_layers(state, "encoder_layers")
    num_decoder_layers = _count_layers(state, "decoder_layers")
    if num_encoder_layers:
        config.num_encoder_layers = num_encoder_layers
    if num_decoder_layers:
        config.num_decoder_layers = num_decoder_layers
    return config


def convert_nampnn_state_dict(
    checkpoint_or_state: Mapping[str, Any] | Mapping[str, Tensor],
    model: nn.Module | None = None,
) -> dict[str, Tensor]:
    r"""
    Convert official NA-MPNN checkpoint keys into native OFoldX keys.

    Args:
        checkpoint_or_state: Full checkpoint mapping or bare tensor state dict.
        model: Optional target module used for strict-load validation.

    Returns:
        Strict-loadable native state dict.

    Raises:
        TypeError: If no tensor state dict can be found.
        RuntimeError: If ``model`` is provided and keys or shapes do not match.

    Examples:
        >>> state = convert_nampnn_state_dict({"model_state_dict": {}})
        >>> state
        {}
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


def convert_nampnn_checkpoint(
    output_dir: str | PathLike[str],
    *,
    checkpoint_path: str | PathLike[str] | None = None,
    variant: str = "design",
    cache_dir: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    checkpoint_variant: str | None = None,
) -> dict[str, Any]:
    r"""
    Convert an official NA-MPNN checkpoint into an OFoldX pretrained artifact.

    Args:
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        checkpoint_path: Optional path to an official checkpoint.
        variant: Official checkpoint variant used when ``checkpoint_path`` is omitted.
        cache_dir: Optional cache directory for downloads.
        dtype: Optional floating-point dtype for persisted weights.
        checkpoint_variant: Optional variant label recorded in config.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_nampnn_checkpoint("nampnn-design", variant="design")  # doctest: +SKIP
    """

    from .model import NAMPNNDesigner

    source = resolve_official_nampnn_checkpoint(variant, checkpoint_path=checkpoint_path, cache_dir=cache_dir)
    checkpoint = load_nampnn_checkpoint(source)
    if checkpoint_variant is None:
        checkpoint_variant = str(OFFICIAL_CHECKPOINTS[variant]["checkpoint_variant"])
    config = nampnn_config_from_checkpoint(checkpoint, checkpoint_variant=checkpoint_variant)
    model = NAMPNNDesigner(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = _cast_floating_tensors(convert_nampnn_state_dict(checkpoint), dtype)
    model.load_state_dict(state_dict, strict=True)
    output_dir = Path(output_dir)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(source),
        "output_dir": str(output_dir),
        "architecture": config.architecture,
        "checkpoint_variant": config.checkpoint_variant,
        "tensor_count": len(state_dict),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(state_dict),
    }


def convert_original_state_dict_key(key: str) -> str:
    r"""
    Translate one official NA-MPNN tensor name to the native OFoldX name.

    Args:
        key: Official NA-MPNN state-dict key.

    Returns:
        Native OFoldX parameter name rooted at ``backbone.``.

    Raises:
        KeyError: If ``key`` is not an official NA-MPNN tensor.

    Examples:
        >>> convert_original_state_dict_key("W_s.weight")
        'backbone.token_embedding.weight'
        >>> convert_original_state_dict_key("features.node_embedding.weight")
        'backbone.feature_encoder.node_projection.weight'
    """

    top_level = {
        "features.embeddings.linear.weight": "backbone.feature_encoder.relative_position.linear.weight",
        "features.embeddings.linear.bias": "backbone.feature_encoder.relative_position.linear.bias",
        "features.node_embedding.weight": "backbone.feature_encoder.node_projection.weight",
        "features.norm_nodes.weight": "backbone.feature_encoder.node_norm.weight",
        "features.norm_nodes.bias": "backbone.feature_encoder.node_norm.bias",
        "features.edge_embedding.weight": "backbone.feature_encoder.edge_projection.weight",
        "features.norm_edges.weight": "backbone.feature_encoder.edge_norm.weight",
        "features.norm_edges.bias": "backbone.feature_encoder.edge_norm.bias",
        "W_e.weight": "backbone.edge_embedding.weight",
        "W_e.bias": "backbone.edge_embedding.bias",
        "W_v.weight": "backbone.node_embedding.weight",
        "W_v.bias": "backbone.node_embedding.bias",
        "W_s.weight": "backbone.token_embedding.weight",
        "W_out.weight": "backbone.output_projection.weight",
        "W_out.bias": "backbone.output_projection.bias",
    }
    if key in top_level:
        return top_level[key]
    if key.startswith("encoder_layers."):
        return _convert_encoder_key(key)
    if key.startswith("decoder_layers."):
        return _convert_decoder_key(key)
    raise KeyError(f"unrecognized NA-MPNN checkpoint key: {key}")


def _convert_encoder_key(key: str) -> str:
    r"""
    Map one official encoder-layer key to its native name.
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
    Map one official decoder-layer key to its native name.
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


def _count_layers(state: Mapping[str, Any], prefix: str) -> int:
    r"""
    Count layer indices in an official state dict.
    """

    layers = set()
    for key in state:
        if isinstance(key, str) and key.startswith(f"{prefix}."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                layers.add(int(parts[1]))
    return max(layers) + 1 if layers else 0


def _cast_floating_tensors(state_dict: Mapping[str, Tensor], dtype: torch.dtype | None) -> dict[str, Tensor]:
    r"""
    Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested.
    """

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""
    Count tensors by dtype name.
    """

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _validate_converted_state_dict(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    r"""
    Assert that a converted state dict strictly matches ``model``.
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
        "NA-MPNN checkpoint is not strict-loadable after conversion",
        f"missing: {len(missing)}",
        f"unexpected: {len(unexpected)}",
        f"mismatched: {len(mismatched)}",
    ]
    if missing:
        lines.append(f"missing keys: {missing[:20]}")
    if unexpected:
        lines.append(f"unexpected keys: {unexpected[:20]}")
    if mismatched:
        lines.append(f"shape mismatches: {mismatched[:20]}")
    raise RuntimeError("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""
    Parse command-line arguments for NA-MPNN conversion.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to an official NA-MPNN .pt checkpoint")
    parser.add_argument("--variant", choices=tuple(OFFICIAL_CHECKPOINTS), default="design")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for downloaded checkpoints")
    parser.add_argument("--checkpoint-variant", default=None, help="Variant label to store in config.json")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for NA-MPNN checkpoint conversion.

    Examples:
        >>> main(["out/nampnn-design", "--variant", "design"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_nampnn_checkpoint(
        args.output_dir,
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        cache_dir=args.cache_dir,
        dtype=dtype,
        checkpoint_variant=args.checkpoint_variant,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted NA-MPNN checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
