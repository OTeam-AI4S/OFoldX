r"""
Official AlphaFold 3 checkpoint conversion entry point.

This module owns the DeepMind AlphaFold 3 parameter source. Protenix checkpoints
are converted by ``ofoldx.models.protenix.convert`` so the generated artifact
keeps the ``protenix`` architecture identity.
"""

from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .converters import deepmind
from .converters.deepmind import (
    OFFICIAL_AF3_CODE_LICENSE,
    OFFICIAL_AF3_PARAMS_TERMS,
    OFFICIAL_AF3_REPOSITORY_URL,
    OFFICIAL_AF3_WEIGHTS_TERMS_URL,
    convert_official_alphafold3_checkpoint,
    inspect_official_alphafold3_checkpoint,
    load_official_alphafold3_params,
    read_official_alphafold3_records,
    select_official_model_files,
    translate_official_alphafold3_state,
)
from .converters.protenix import (
    ReferenceKey,
    ReferenceValue,
    _alphafold3_reference_key,
    _cast_floating_tensors,
    _dtype_counts,
    load_reference_state,
    translate_alphafold3_state,
    translate_reference_state,
)

CHECKPOINT_SOURCES = ("deepmind",)
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

__all__ = [
    "CHECKPOINT_SOURCES",
    "OFFICIAL_AF3_CODE_LICENSE",
    "OFFICIAL_AF3_PARAMS_TERMS",
    "OFFICIAL_AF3_REPOSITORY_URL",
    "OFFICIAL_AF3_WEIGHTS_TERMS_URL",
    "ReferenceKey",
    "ReferenceValue",
    "_alphafold3_reference_key",
    "_cast_floating_tensors",
    "_dtype_counts",
    "convert_alphafold3_checkpoint",
    "convert_official_alphafold3_checkpoint",
    "inspect_alphafold3_checkpoint",
    "inspect_official_alphafold3_checkpoint",
    "load_official_alphafold3_params",
    "load_reference_state",
    "main",
    "read_official_alphafold3_records",
    "select_official_model_files",
    "translate_alphafold3_state",
    "translate_official_alphafold3_state",
    "translate_reference_state",
]


def inspect_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
) -> dict[str, Any]:
    r"""
    Inspect an AlphaFold 3 checkpoint without writing an artifact.
    """

    source = _normalize_source(source)
    if source == "deepmind":
        return deepmind.inspect_official_alphafold3_checkpoint(checkpoint_path, model_name=model_name)
    raise AssertionError("unreachable checkpoint source")


def convert_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    inspect_only: bool = False,
) -> dict[str, Any]:
    r"""
    Convert an AlphaFold 3 checkpoint into an OFoldX model artifact.
    """

    source = _normalize_source(source)
    if inspect_only:
        return inspect_alphafold3_checkpoint(checkpoint_path, source=source, model_name=model_name)
    if source == "deepmind":
        return deepmind.convert_official_alphafold3_checkpoint(
            checkpoint_path,
            output_dir,
            model_name=model_name,
            dtype=dtype,
        )
    raise AssertionError("unreachable checkpoint source")


def _normalize_source(source: str) -> str:
    normalized = source.lower()
    if normalized not in CHECKPOINT_SOURCES:
        choices = ", ".join(CHECKPOINT_SOURCES)
        raise ValueError(f"unsupported AlphaFold 3 checkpoint source {source!r}; expected one of: {choices}")
    return normalized


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=CHECKPOINT_SOURCES, default="deepmind", help="Checkpoint source format")
    parser.add_argument("checkpoint", type=Path, help="Source checkpoint path")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--model-name", default=None, help="Source-specific model id")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    parser.add_argument("--inspect-only", action="store_true", help="Only print a checkpoint summary")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for AlphaFold 3 checkpoint conversion.
    """

    args = _parse_args(argv)
    if args.inspect_only and args.push_to_hub:
        raise SystemExit("--inspect-only and --push-to-hub cannot be combined because inspection writes no artifact")
    if args.push_to_hub:
        raise SystemExit("official DeepMind AlphaFold 3 parameters must not be uploaded by this converter")
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_alphafold3_checkpoint(
        args.checkpoint,
        args.output_dir,
        source=args.source,
        model_name=args.model_name,
        dtype=dtype,
        inspect_only=args.inspect_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message=f"Upload converted AlphaFold 3 {args.source} checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
