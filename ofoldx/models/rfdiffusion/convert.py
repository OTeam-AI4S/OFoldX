r"""
RfDiffusion checkpoint conversion.

The released RosettaCommons checkpoints store two pieces OFoldX must preserve: ``config_dict`` and
``model_state_dict``. Conversion preserves checkpoint tensors under the ``network`` submodule key space, without
vendoring the RosettaCommons runtime.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from chanfig import Config
from safetensors.torch import save_file
from torch import Tensor

from ...artifacts import ARTIFACT_MANIFEST_NAME, build_artifact_manifest
from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from ..mixin import ModelMixin
from .config import RfDiffusionConfig

_STATE_KEYS = ("model_state_dict", "model", "state_dict")
_CONFIG_GROUPS = (
    "model",
    "diffuser",
    "preprocess",
    "denoiser",
    "inference",
    "contigmap",
    "ppi",
    "potentials",
    "scaffoldguided",
    "logging",
)
# Upstream RFdiffusion "model" config keys -> OFoldX house vocabulary, so a released checkpoint's
# config_dict populates the renamed RfDiffusionModelConfig fields. The SE3_param_* inner keys
# (l0_in_features / l0_out_features / num_edge_features) are SE3-standard and are left unchanged.
_MODEL_CONFIG_RENAMES = {
    "n_extra_block": "num_extra_blocks",
    "n_main_block": "num_main_blocks",
    "n_ref_block": "num_refinement_blocks",
    "d_msa": "msa_size",
    "d_msa_full": "full_msa_size",
    "d_pair": "pair_size",
    "d_templ": "template_size",
    "n_head_msa": "num_msa_attention_heads",
    "n_head_pair": "num_pair_attention_heads",
    "n_head_templ": "num_template_attention_heads",
    "d_hidden": "hidden_size",
    "d_hidden_templ": "template_hidden_size",
    "p_drop": "dropout",
    "d_time_emb": "time_embedding_size",
    "d_time_emb_proj": "time_embedding_projection_size",
    "SE3_param_full": "se3_param_full",
    "SE3_param_topk": "se3_param_topk",
}
_SE3_CONFIG_RENAMES = {
    "n_heads": "num_attention_heads",
}
# Upstream "preprocess" feature-dimension keys -> house vocabulary.
_PREPROCESS_CONFIG_RENAMES = {
    "d_t1d": "template_1d_size",
    "d_t2d": "template_2d_size",
}
# Upstream "diffuser" schedule keys -> house vocabulary. The timestep-count key ``T`` becomes ``num_timesteps``.
_DIFFUSER_CONFIG_RENAMES = {
    "T": "num_timesteps",
    "partial_T": "partial_timesteps",
}
_GROUP_CONFIG_RENAMES = {
    "model": _MODEL_CONFIG_RENAMES,
    "preprocess": _PREPROCESS_CONFIG_RENAMES,
    "diffuser": _DIFFUSER_CONFIG_RENAMES,
}
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_rfdiffusion_checkpoint(checkpoint_path: str | Path, *, map_location: Any = "cpu") -> Mapping[str, Any]:
    r"""
    Load a released RfDiffusion ``.pt`` checkpoint into a mapping.

    Args:
        checkpoint_path: Path to an official RosettaCommons RfDiffusion checkpoint.
        map_location: Device mapping forwarded to [torch.load][]; defaults to CPU.

    Returns:
        The unpickled checkpoint mapping, typically holding ``config_dict`` and ``model_state_dict``.

    Raises:
        TypeError: If the loaded object is not a mapping.

    Examples:
        >>> from ofoldx.models.rfdiffusion.convert import load_rfdiffusion_checkpoint
        >>> checkpoint = load_rfdiffusion_checkpoint("Base_ckpt.pt")  # doctest: +SKIP
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"RfDiffusion checkpoint must be a mapping, got {type(checkpoint)!r}")
    return checkpoint


def convert_rfdiffusion_state_dict(
    state_dict: Mapping[str, Any],
    *,
    prefix: str = "network.",
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Rekey an official RfDiffusion tensor state dict into the OFoldX native model key space.

    Strips any leading ``module.`` (DataParallel) prefix, re-roots each tensor under ``prefix`` (the
    ``network`` submodule the model builds), and optionally down/up-casts floating-point tensors. Non-tensor
    entries are skipped.

    Args:
        state_dict: Mapping of upstream parameter names to tensors.
        prefix: Key prefix for the converted namespace; the model nests its network under `"network."`.
        dtype: Optional dtype to cast floating-point tensors to; `None` preserves the stored dtype.

    Returns:
        A dict mapping OFoldX parameter names to tensors.

    Raises:
        ValueError: If two source keys collide after conversion, or no tensors are present.

    Examples:
        >>> from ofoldx.models.rfdiffusion.convert import convert_rfdiffusion_state_dict
        >>> import torch
        >>> converted = convert_rfdiffusion_state_dict({"module.emb.weight": torch.zeros(2, 2)})
        >>> sorted(converted)
        ['network.emb.weight']
        >>> converted = convert_rfdiffusion_state_dict({"templ_emb.emb_t1d.weight": torch.zeros(2, 2)})
        >>> sorted(converted)
        ['network.templ_emb.emb_template_1d.weight']
    """

    converted: dict[str, Tensor] = {}
    for key, value in sorted(state_dict.items()):
        if not torch.is_tensor(value):
            continue
        tensor = value
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        converted_key = f"{prefix}{convert_rfdiffusion_state_dict_key(key)}"
        if converted_key in converted:
            raise ValueError(f"duplicate RfDiffusion tensor key after conversion: {converted_key}")
        converted[converted_key] = tensor
    if not converted:
        raise ValueError("RfDiffusion state dict did not contain any tensors to convert")
    return converted


def convert_rfdiffusion_state_dict_key(key: str) -> str:
    r"""
    Translate one released RfDiffusion tensor key to the OFoldX native module key.

    Args:
        key: Tensor key from the released checkpoint, with or without a leading ``module.`` prefix.

    Returns:
        The key relative to the OFoldX RfDiffusion network module.

    Examples:
        >>> convert_rfdiffusion_state_dict_key("module.c6d_pred.proj_symm.weight")
        'coords_6d_pred.proj_symm.weight'
        >>> convert_rfdiffusion_state_dict_key("simulator.main_block.0.msa2msa.ff.norm.weight")
        'simulator.main_block.0.msa2msa.feed_forward.norm.weight'
    """

    key = key.removeprefix("module.")
    if key.startswith("c6d_pred."):
        key = "coords_6d_pred." + key[len("c6d_pred.") :]
    key = key.replace("templ_emb.emb_t1d.", "templ_emb.emb_template_1d.")
    key = key.replace("templ_emb.proj_t1d.", "templ_emb.proj_template_1d.")
    key = key.replace(".ff.", ".feed_forward.")
    return key


def config_from_rfdiffusion_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_variant: str = "base",
    require_config: bool = False,
) -> RfDiffusionConfig:
    r"""
    Build an [`RfDiffusionConfig`][ofoldx.models.rfdiffusion.config.RfDiffusionConfig] from a checkpoint.

    Copies each recognized ``config_dict`` group into the matching config field, applying the upstream-to-house
    key renames (see the module-level rename maps). When the checkpoint carries no ``config_dict``, a default
    config is returned unless ``require_config`` forces an error.

    Args:
        checkpoint: A loaded RfDiffusion checkpoint mapping.
        checkpoint_variant: Variant label stored on the resulting config.
        require_config: If `True`, raise when the checkpoint has no ``config_dict``.

    Returns:
        A populated config reflecting the checkpoint's training-time settings.

    Raises:
        KeyError: If ``require_config`` is set but the checkpoint lacks ``config_dict``.
        TypeError: If ``config_dict`` is present but not a mapping.

    Examples:
        >>> from ofoldx.models.rfdiffusion.convert import config_from_rfdiffusion_checkpoint
        >>> config = config_from_rfdiffusion_checkpoint({}, checkpoint_variant="base")
        >>> config.checkpoint_variant
        'base'
        >>> config = config_from_rfdiffusion_checkpoint({
        ...     "config_dict": {"model": {"SE3_param_full": {"n_heads": 4}}}
        ... })
        >>> config.model.se3_param_full.num_attention_heads
        4
    """

    config = RfDiffusionConfig()
    config.checkpoint_variant = checkpoint_variant
    source_config = checkpoint.get("config_dict")
    if source_config is None:
        if require_config:
            raise KeyError("official RfDiffusion checkpoint is missing config_dict")
        return config
    if not isinstance(source_config, Mapping):
        raise TypeError(f"checkpoint config_dict must be a mapping, got {type(source_config)!r}")
    for group in _CONFIG_GROUPS:
        value = source_config.get(group)
        if isinstance(value, Mapping):
            plain = _to_plain_container(value)
            renames = _GROUP_CONFIG_RENAMES.get(group)
            if renames is not None and isinstance(plain, dict):
                plain = {renames.get(key, key): item for key, item in plain.items()}
            if group == "model" and isinstance(plain, dict):
                plain = _normalize_model_config(plain)
            config[group] = Config(plain)
    return config


def convert_rfdiffusion_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_variant: str | None = None,
    state_key: str = "auto",
    dtype: str = "keep",
    strict_load_check: bool = True,
) -> dict[str, Any]:
    r"""
    Convert a released RfDiffusion checkpoint into an OFoldX pretrained directory.

    Writes ``config.json``, ``model.safetensors``, and a ``conversion_report.json`` describing the run. When
    ``strict_load_check`` is set, the freshly-written artifact is strict-loaded through
    [`RfDiffusionDesigner`][ofoldx.models.rfdiffusion.model.RfDiffusionDesigner] to prove the
    key space matches the model before returning.

    Args:
        checkpoint_path: Path to the official ``.pt`` checkpoint.
        output_dir: Destination directory for the converted artifact (created if missing).
        checkpoint_variant: Variant label; defaults to the checkpoint file stem.
        state_key: Which checkpoint key holds the tensor state dict (`"auto"` searches known keys).
        dtype: Target tensor dtype (`"keep"`, `"float32"`, `"float16"`, or `"bfloat16"`).
        strict_load_check: Whether to strict-load the written artifact as a final validation.

    Returns:
        A report dict recording input/output paths, tensor and dtype counts, the selected state key, and the
        load-check result.

    Raises:
        RuntimeError: If ``strict_load_check`` is set and the written artifact fails to strict-load.

    Examples:
        >>> from ofoldx.models.rfdiffusion.convert import convert_rfdiffusion_checkpoint
        >>> report = convert_rfdiffusion_checkpoint("Base_ckpt.pt", "out/rfdiffusion")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = load_rfdiffusion_checkpoint(checkpoint_path)
    state_dict, selected_state_key = _select_state_dict(checkpoint, state_key=state_key)
    target_dtype = _resolve_dtype(dtype)
    variant = checkpoint_path.stem if checkpoint_variant is None else checkpoint_variant
    config = config_from_rfdiffusion_checkpoint(checkpoint, checkpoint_variant=variant, require_config=True)
    converted = convert_rfdiffusion_state_dict(state_dict, dtype=target_dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    _artifact_manifest().json(output_dir / ARTIFACT_MANIFEST_NAME)
    _config_for_artifact(config).json(config_path)
    save_file(converted, str(weights_path))

    load_check = {"strict": strict_load_check, "ok": None, "error": None}
    if strict_load_check:
        load_check = _strict_load_check(output_dir)

    report = {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "state_key": selected_state_key,
        "tensor_count": len(converted),
        "dtype": dtype,
        "dtype_counts": _dtype_counts(converted),
        "load_check": load_check,
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if strict_load_check and not load_check["ok"]:
        raise RuntimeError(f"converted RfDiffusion checkpoint is not strict-loadable: {load_check['error']}")
    return report


def _select_state_dict(checkpoint: Mapping[str, Any], *, state_key: str) -> tuple[Mapping[str, Any], str]:
    r"""
    Locate the tensor state dict within a checkpoint.

    Args:
        checkpoint: The loaded checkpoint mapping.
        state_key: Explicit key, `"auto"` to search known keys, or `"root"` to treat the checkpoint itself.

    Returns:
        A ``(state_dict, selected_key)`` pair naming which key was used.

    Raises:
        TypeError: If no tensor state dict can be found under ``state_key``.
    """

    if state_key == "auto":
        for key in _STATE_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value, key
        if _looks_like_state_dict(checkpoint):
            return checkpoint, "root"
    if state_key == "root" and _looks_like_state_dict(checkpoint):
        return checkpoint, "root"
    value = checkpoint.get(state_key)
    if isinstance(value, Mapping):
        return value, state_key
    raise TypeError(f"checkpoint does not contain tensor state dict {state_key!r}")


def _looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    r"""Return whether ``value`` looks like a raw tensor state dict (has string keys mapping to tensors)."""

    return bool(value) and any(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items())


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    r"""
    Resolve a dtype name to a torch dtype.

    Args:
        dtype: One of `"keep"`, `"float32"`, `"float16"`, or `"bfloat16"`.

    Returns:
        The corresponding [torch.dtype][], or `None` for `"keep"`.

    Raises:
        ValueError: If ``dtype`` is not a supported name.
    """

    if dtype == "keep":
        return None
    try:
        return _DTYPES[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {dtype!r}; expected keep, float32, float16, or bfloat16") from error


def _config_for_artifact(config: RfDiffusionConfig) -> Config:
    r"""Return a copy of ``config`` containing only runtime model settings."""

    artifact = config.copy()
    artifact.pop("architecture", None)
    return artifact


def _artifact_manifest() -> Config:
    r"""Return the root manifest for a converted RFdiffusion model artifact."""

    return build_artifact_manifest(
        kind="model",
        entrypoint="ofoldx.models.rfdiffusion.model.RfDiffusionDesigner",
        architecture="rfdiffusion",
        task="design",
        files={"config": ModelMixin.CONFIG_NAME},
        weights={"default": ModelMixin.WEIGHTS_NAME},
    )


def _strict_load_check(output_dir: Path) -> dict[str, Any]:
    r"""
    Strict-load the written artifact to confirm its keys match the model.

    Args:
        output_dir: Directory holding the converted ``config.json`` and weights.

    Returns:
        A dict with ``strict``, ``ok``, and ``error`` describing the outcome.
    """

    from .model import RfDiffusionDesigner

    try:
        RfDiffusionDesigner.from_pretrained(output_dir, strict=True)
    except Exception as error:
        return {"strict": True, "ok": False, "error": f"{type(error).__name__}: {error}"}
    return {"strict": True, "ok": True, "error": None}


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""Return a sorted count of tensors per dtype name in ``state_dict`` (for the conversion report)."""

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _to_plain_container(value: Any) -> Any:
    r"""Recursively convert chanfig configs and nested mappings/sequences into plain Python containers."""

    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_plain_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_container(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_container(item) for item in value]
    return value


def _normalize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    r"""Normalize nested RfDiffusion model config dictionaries after top-level key renaming."""

    for key in ("se3_param_full", "se3_param_topk"):
        value = config.get(key)
        if isinstance(value, dict):
            config[key] = {_SE3_CONFIG_RENAMES.get(item_key, item_key): item for item_key, item in value.items()}
    return config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for the checkpoint-conversion entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an official RfDiffusion .pt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output OFoldX pretrained directory")
    parser.add_argument("--checkpoint-variant", default=None, help="Variant label to store in config.json")
    parser.add_argument(
        "--state-key", default="auto", help="Checkpoint state key: auto, model_state_dict, state_dict, or root"
    )
    parser.add_argument("--dtype", default="keep", choices=("keep", "float32", "float16", "bfloat16"))
    parser.add_argument(
        "--strict-load-check",
        dest="strict_load_check",
        action="store_true",
        default=True,
        help="Strict-load the written artifact (default)",
    )
    parser.add_argument(
        "--no-strict-load-check",
        dest="strict_load_check",
        action="store_false",
        help="Skip strict-load validation after writing the artifact",
    )
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main() -> None:
    r"""
    Command-line entry point: convert a checkpoint and print the JSON conversion report.

    Reads ``checkpoint`` and ``output_dir`` (plus optional flags) from ``sys.argv`` via
    [`_parse_args`][ofoldx.models.rfdiffusion.convert._parse_args], runs
    [`convert_rfdiffusion_checkpoint`][ofoldx.models.rfdiffusion.convert.convert_rfdiffusion_checkpoint],
    and writes the report to stdout.

    Examples:
        >>> from ofoldx.models.rfdiffusion.convert import main
        >>> main()  # doctest: +SKIP
    """

    args = _parse_args()
    report = convert_rfdiffusion_checkpoint(
        args.checkpoint,
        args.output_dir,
        checkpoint_variant=args.checkpoint_variant,
        state_key=args.state_key,
        dtype=args.dtype,
        strict_load_check=args.strict_load_check,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted RfDiffusion checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
