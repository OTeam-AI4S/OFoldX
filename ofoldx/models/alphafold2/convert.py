r"""
Unified AlphaFold 2 checkpoint conversion entry point.

The public CLI stays source-oriented:

```
python -m ofoldx.models.alphafold2.convert --source deepmind CHECKPOINT OUTPUT_DIR
```

Source-specific details live in ``ofoldx.models.alphafold2.converters`` so that
DeepMind official NPZ/tar conversion, OpenFold checkpoints, and future community
formats can evolve independently while sharing one user-facing command.
"""

from __future__ import annotations

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import Any

import torch

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .converters import deepmind, openfold
from .converters.deepmind import (
    OFFICIAL_AF2_CODE_LICENSE,
    OFFICIAL_AF2_MODEL_NAMES,
    OFFICIAL_AF2_PARAMS_ARCHIVE,
    OFFICIAL_AF2_PARAMS_LICENSE,
    OFFICIAL_AF2_PARAMS_URL,
    convert_official_alphafold2_state_dict,
    convert_official_tensor,
    inspect_official_alphafold2_checkpoint,
    inspect_official_alphafold2_mapping,
    load_official_alphafold2_params,
    normalize_official_param_key,
    official_param_filename,
    unflatten_official_params,
)
from .converters.openfold import (
    OPENFOLD_MODEL_NAMES,
    OPENFOLD_PARAMS_HTTPS_URL,
    OPENFOLD_PARAMS_LICENSE,
    OPENFOLD_PARAMS_S3_URL,
    convert_openfold_alphafold2_state_dict,
    inspect_openfold_alphafold2_checkpoint,
    normalize_openfold_state_key,
)

CHECKPOINT_SOURCES = ("deepmind", "openfold")
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

__all__ = [
    "CHECKPOINT_SOURCES",
    "OFFICIAL_AF2_MODEL_NAMES",
    "OFFICIAL_AF2_PARAMS_ARCHIVE",
    "OFFICIAL_AF2_PARAMS_LICENSE",
    "OFFICIAL_AF2_PARAMS_URL",
    "OFFICIAL_AF2_CODE_LICENSE",
    "OPENFOLD_MODEL_NAMES",
    "OPENFOLD_PARAMS_HTTPS_URL",
    "OPENFOLD_PARAMS_LICENSE",
    "OPENFOLD_PARAMS_S3_URL",
    "convert_alphafold2_checkpoint",
    "convert_official_alphafold2_state_dict",
    "convert_official_tensor",
    "convert_openfold_alphafold2_state_dict",
    "inspect_alphafold2_checkpoint",
    "inspect_official_alphafold2_checkpoint",
    "inspect_official_alphafold2_mapping",
    "inspect_openfold_alphafold2_checkpoint",
    "load_official_alphafold2_params",
    "main",
    "normalize_official_param_key",
    "normalize_openfold_state_key",
    "official_param_filename",
    "unflatten_official_params",
]


def inspect_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
) -> dict[str, Any]:
    r"""
    Inspect an AlphaFold 2 checkpoint without writing an artifact.

    Args:
        checkpoint_path: Source checkpoint path.
        source: Checkpoint source format. Supported values are ``"deepmind"`` and ``"openfold"``.
        model_name: Optional source-specific model id.

    Returns:
        JSON-serializable checkpoint summary.
    """

    source = _normalize_source(source)
    if source == "deepmind":
        return deepmind.inspect_official_alphafold2_checkpoint(checkpoint_path, model_name=model_name)
    if source == "openfold":
        return openfold.inspect_openfold_alphafold2_checkpoint(checkpoint_path, model_name=model_name)
    raise AssertionError("unreachable checkpoint source")


def convert_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    inspect_only: bool = False,
    verify_artifact: bool = True,
) -> dict[str, Any]:
    r"""
    Convert an AlphaFold 2 checkpoint into an OFoldX model artifact.

    Args:
        checkpoint_path: Source checkpoint path.
        output_dir: Directory to write the converted model artifact.
        source: Checkpoint source format. Supported values are ``"deepmind"`` and ``"openfold"``.
        model_name: Optional source-specific model id.
        dtype: Optional dtype for floating-point persisted tensors.
        inspect_only: If ``True``, only return a checkpoint summary and write nothing.
        verify_artifact: If ``True``, reload the written artifact with strict model loading.

    Returns:
        JSON-serializable conversion report.
    """

    source = _normalize_source(source)
    if inspect_only:
        return inspect_alphafold2_checkpoint(checkpoint_path, source=source, model_name=model_name)
    if source == "deepmind":
        return deepmind.convert_deepmind_alphafold2_checkpoint(
            checkpoint_path,
            output_dir,
            model_name=model_name,
            dtype=dtype,
            verify_artifact=verify_artifact,
        )
    if source == "openfold":
        return openfold.convert_openfold_alphafold2_checkpoint(
            checkpoint_path,
            output_dir,
            model_name=model_name,
            dtype=dtype,
            verify_artifact=verify_artifact,
        )
    raise AssertionError("unreachable checkpoint source")


def _normalize_source(source: str) -> str:
    normalized = source.lower()
    if normalized not in CHECKPOINT_SOURCES:
        choices = ", ".join(CHECKPOINT_SOURCES)
        raise ValueError(f"unsupported AlphaFold 2 checkpoint source {source!r}; expected one of: {choices}")
    return normalized


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=CHECKPOINT_SOURCES, default="deepmind", help="Checkpoint source format")
    parser.add_argument("checkpoint", type=Path, help="Source checkpoint path")
    parser.add_argument("output_dir", type=Path, help="Output OFoldX model artifact directory")
    parser.add_argument("--model-name", default=None, help="Source-specific model id")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    parser.add_argument("--inspect-only", action="store_true", help="Only print a checkpoint summary")
    parser.add_argument("--no-verify", action="store_true", help="Do not strict-reload the written model artifact")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""
    Command-line entry point for AlphaFold 2 model conversion.
    """

    args = _parse_args(argv)
    if args.inspect_only and args.push_to_hub:
        raise SystemExit("--inspect-only and --push-to-hub cannot be combined because inspection writes no artifact")
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_alphafold2_checkpoint(
        args.checkpoint,
        args.output_dir,
        source=args.source,
        model_name=args.model_name,
        dtype=dtype,
        inspect_only=args.inspect_only,
        verify_artifact=not args.no_verify,
    )
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message=f"Convert AlphaFold 2 {args.source} {report['model_name']}",
    )
    if upload_result is not None:
        report["hub_upload"] = str(upload_result)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
