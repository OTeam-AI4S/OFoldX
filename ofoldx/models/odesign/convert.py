r"""
ODesign checkpoint converter.

Translates a released ODesign checkpoint into OFoldX key names. ODesign extends the shared AlphaFold 3 /
Protenix trunk with two design-specific heads -- the token bond-type predictor and the constraint-distogram
embedder -- so the key mapping defers to the shared
[`_alphafold3_reference_key`][ofoldx.models.alphafold3.converters.protenix] and only injects the handful of
ODesign-specific submodule renames through its ``model_rules`` hook.
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
    _replace_many,
)
from ..alphafold3.converters.protenix import load_reference_state as load_protenix_checkpoint
from ..alphafold3.converters.protenix import translate_reference_state

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_reference_state(checkpoint_path: str | Any, *, map_location: str | Any = "cpu") -> dict[str, Tensor]:
    r"""
    Load an ODesign checkpoint into a flat tensor state dict with wrapper prefixes removed.

    Delegates to the shared AlphaFold 3 loader, which unwraps a ``{"model": ...}`` container and strips the
    distributed-training ``module.`` prefix so the keys line up with the reference ODesign parameter names.

    Args:
        checkpoint_path: Path (or file-like object) of the released ODesign checkpoint.
        map_location: Device passed through to the underlying loader; defaults to CPU.

    Returns:
        Mapping from reference parameter name to tensor.

    Examples:
        >>> state = load_reference_state("odesign_base_prot_flex.pt")  # doctest: +SKIP
    """

    return load_protenix_checkpoint(checkpoint_path, map_location=map_location)


def translate_odesign_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Translate a complete ODesign checkpoint into OFoldX key names.

    Every OFoldX parameter -- including the ODesign-specific ``bond_type_predictor`` and
    ``constraint_distogram_embedder`` tensors -- is expected to be sourced from the reference state, so this
    is a strict translation suitable for ``load_state_dict(strict=True)``.

    Args:
        model: Target OFoldX model whose parameter names drive the key mapping.
        reference_state: Reference ODesign tensor state (see `load_reference_state`).

    Returns:
        OFoldX-keyed state dict ready to load into ``model``.

    Examples:
        >>> state = load_reference_state("odesign_base_prot_flex.pt")  # doctest: +SKIP
        >>> translated = translate_odesign_state(model, state)  # doctest: +SKIP
        >>> model.load_state_dict(translated, strict=True)  # doctest: +SKIP
    """

    return translate_reference_state(model, reference_state, _odesign_reference_key)


def convert_odesign_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    r"""
    Convert an ODesign checkpoint into a pretrained OFoldX artifact directory.

    Builds an [`ODesignConfig`][ofoldx.models.odesign.config.ODesignConfig] with the constraint-distogram
    embedder enabled (matching the released flex checkpoint), translates the reference tensors, loads them
    strictly, and writes ``manifest.json``, ``config.json``, and ``model.safetensors``.

    Args:
        checkpoint_path: Path to the released ODesign checkpoint.
        output_dir: Directory to write the pretrained artifact into.
        dtype: Optional floating-point dtype for persisted weights.

    Returns:
        JSON-serializable conversion report.

    Examples:
        >>> convert_odesign_checkpoint("odesign_base_prot_flex.pt", "odesign")  # doctest: +SKIP
    """

    from .config import ODesignConfig
    from .model import ODesignModel

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    config = ODesignConfig()
    config.constraint.enabled = True
    model = ODesignModel(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = _cast_floating_tensors(
        translate_odesign_state(model, load_reference_state(checkpoint_path)),
        dtype,
    )
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "architecture": model.config.architecture,
        "checkpoint_source": "odesign",
        "checkpoint_format": "torch_state_dict",
        "tensor_count": len(state_dict),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(state_dict),
    }


def _bond_type_head_key(key: str) -> str:
    r"""Rename the OFoldX token bond-type head onto the reference ``bond_type_predictor``."""

    prefix = "token_bond_type_head"
    if key == prefix or key.startswith(prefix + "."):
        return "bond_type_predictor" + key[len(prefix) :]
    return key


def _constraint_embedder_key(key: str) -> str:
    r"""
    Rewrite constraint-distogram-embedder submodule names to the reference convention.

    Mirrors the shared template-embedder mapping (Algorithm 16 template path): only the embedder's own
    projections and norms are renamed. Keys under the inner ``pairformer_stack`` (which reuse the shared
    Pairformer submodule names, including their own ``output_projection``) are left untouched so the shared
    Pairformer rules translate them.
    """

    if not key.startswith("constraint_distogram_embedder."):
        return key
    if key.startswith("constraint_distogram_embedder.pairformer_stack."):
        return key
    return _replace_many(
        key,
        {
            ".pair_norm.": ".layernorm_z.",
            ".pair_projection.": ".linear_no_bias_z.",
            ".distogram_projection.": ".linear_no_bias_a.",
            ".hidden_norm.": ".layernorm_v.",
            ".output_projection.": ".linear_no_bias_u.",
        },
    )


_MODEL_RULES = (_bond_type_head_key, _constraint_embedder_key)


def _odesign_reference_key(model_key: str) -> ReferenceKey:
    r"""
    Map an OFoldX parameter name to its reference ODesign checkpoint key.

    The ODesign-specific bond-type and constraint-embedder renames are injected via ``model_rules`` right
    after the top-level rename; every other key defers to the shared AlphaFold 3 mapping.

    Args:
        model_key: OFoldX parameter name.

    Returns:
        The corresponding reference checkpoint key.
    """

    return _alphafold3_reference_key(model_key, model_rules=_MODEL_RULES)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    r"""Parse command-line arguments for ODesign checkpoint conversion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to an ODesign checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for ODesign checkpoint conversion.

    Examples:
        >>> main(["odesign_base_prot_flex.pt", "out/odesign"])  # doctest: +SKIP
    """

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_odesign_checkpoint(args.checkpoint, args.output_dir, dtype=dtype)
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted ODesign checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
