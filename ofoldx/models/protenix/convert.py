r"""
Protenix checkpoint converter.
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
from ..alphafold3.converters.protenix import (
    ReferenceKey,
    _alphafold3_reference_key,
    _cast_floating_tensors,
    _dtype_counts,
)
from ..alphafold3.converters.protenix import load_reference_state as load_protenix_checkpoint
from ..alphafold3.converters.protenix import translate_reference_state

_CONSTRAINT_PREFIXES = ("constraint_embedder.",)
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_reference_state(checkpoint_path: str | Any, *, map_location: str | Any = "cpu") -> dict[str, Tensor]:
    r"""
    Load a Protenix checkpoint into a flat tensor state dict with distributed prefixes removed.

    Delegates to the shared AlphaFold 3 loader, which strips distributed-training wrapper prefixes
    (e.g. ``module.``) so the keys line up with the reference Protenix parameter names.

    Args:
        checkpoint_path: Path (or file-like object) of the released Protenix checkpoint.
        map_location: Device passed through to the underlying loader; defaults to CPU.

    Returns:
        Mapping from reference parameter name to tensor.

    Examples:
        >>> state = load_reference_state("protenix_base_default_v0.5.0.pt")  # doctest: +SKIP
    """

    return load_protenix_checkpoint(checkpoint_path, map_location=map_location)


def translate_protenix_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Translate a complete Protenix checkpoint into OFoldX key names.

    This strict path expects every OFoldX parameter to be sourced from the reference state, including
    ``constraint_embedder.*`` tensors. The public v0.5 base checkpoint does not carry those tensors; use
    [`translate_protenix_from_base_state`][ofoldx.models.protenix.convert.translate_protenix_from_base_state]
    for that release.

    Args:
        model: Target OFoldX model whose parameter names drive the key mapping.
        reference_state: Reference Protenix tensor state (see `load_reference_state`).

    Returns:
        OFoldX-keyed state dict ready to load into ``model``.

    Examples:
        >>> state = load_reference_state("protenix_full.pt")  # doctest: +SKIP
        >>> translated = translate_protenix_state(model, state)  # doctest: +SKIP
        >>> model.load_state_dict(translated)  # doctest: +SKIP
    """

    return translate_reference_state(model, reference_state, _protenix_reference_key)


def convert_protenix_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    r"""
    Convert a Protenix base checkpoint into a pretrained OFoldX artifact directory.

    Args:
        checkpoint_path: Path to the released Protenix base checkpoint.
        output_dir: Directory to write ``manifest.json``, ``config.json``, and ``model.safetensors``.
        dtype: Optional floating-point dtype for persisted weights.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_protenix_checkpoint("protenix_base_default_v0.5.0.pt", "protenix")  # doctest: +SKIP
    """

    from .config import ProtenixConfig
    from .model import ProtenixModel

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    model = ProtenixModel(ProtenixConfig())
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = _cast_floating_tensors(
        translate_protenix_from_base_state(model, load_reference_state(checkpoint_path)),
        dtype,
    )
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": model.config.architecture,
        "checkpoint_source": "protenix",
        "checkpoint_format": "torch_state_dict",
        "tensor_count": len(state_dict),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(state_dict),
    }


def _with_initialized_prefixes(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, Tensor]:
    r"""
    Fill tolerated missing keys from the model's configured initialization before strict loading.
    """

    completed = dict(state_dict)
    for key, value in model.state_dict().items():
        if key.startswith(prefixes) and key not in completed:
            completed[key] = value.detach().clone()
    return completed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for Protenix checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a Protenix base checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for Protenix checkpoint conversion.

    Examples:
        >>> main(["protenix_base_default_v0.5.0.pt", "out/protenix"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_protenix_checkpoint(args.checkpoint, args.output_dir, dtype=dtype)
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted Protenix checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


def translate_protenix_from_base_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""
    Initialize the shared AF3/Protenix trunk from a non-constraint Protenix checkpoint.

    Constraint-embedder tensors are absent in the public v0.5 base checkpoint, so
    ``constraint_embedder.*`` keys are filled from ``model``'s configured initialization.

    Args:
        model: Target OFoldX model whose parameter names drive the key mapping.
        reference_state: Reference non-constraint Protenix tensor state.

    Returns:
        OFoldX-keyed state dict that can be loaded with ``strict=True``.

    Examples:
        >>> state = load_reference_state("protenix_base_default_v0.5.0.pt")  # doctest: +SKIP
        >>> translated = translate_protenix_from_base_state(model, state)  # doctest: +SKIP
        >>> model.load_state_dict(translated, strict=True)  # doctest: +SKIP
    """

    return _with_initialized_prefixes(
        model,
        translate_reference_state(
            model,
            reference_state,
            _protenix_reference_key,
            allow_missing_prefixes=_CONSTRAINT_PREFIXES,
        ),
        _CONSTRAINT_PREFIXES,
    )


def _protenix_reference_key(model_key: str) -> ReferenceKey:
    r"""
    Map an OFoldX parameter name to its reference Protenix checkpoint key.

    Constraint-embedder parameters share their names with the reference and pass through unchanged;
    all other keys defer to the shared AlphaFold 3 key mapping.

    Args:
        model_key: OFoldX parameter name.

    Returns:
        The corresponding reference checkpoint key.
    """

    if model_key.startswith("constraint_embedder."):
        return model_key
    return _alphafold3_reference_key(model_key)


if __name__ == "__main__":
    main()
