r"""
Official AlphaFold 3 checkpoint conversion entry point.

This module owns the DeepMind AlphaFold 3 parameter source. Protenix checkpoints
are converted by ``ofoldx.models.protenix.convert`` so the resulting model keeps
the ``protenix`` architecture identity.
"""

from __future__ import annotations

import argparse
from os import PathLike
from pathlib import Path
from typing import cast

import torch

from .config import AlphaFold3ProcessorConfig
from .converters import deepmind
from .model import AlphaFold3Model
from .pipeline import AlphaFold3Pipeline
from .processor import AlphaFold3Processor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def convert_alphafold3_model(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> AlphaFold3Model:
    r"""
    Strict-load official AF3 parameters into an in-memory native Model.
    """

    return cast(
        AlphaFold3Model,
        deepmind.convert_official_alphafold3_checkpoint(checkpoint_path, model_name=model_name, dtype=dtype),
    )


def convert_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Strict-load official AF3 parameters and write a standard OFoldX repository.

    Converts the DeepMind AlphaFold 3 parameter archive at ``checkpoint_path`` into a
    native model, pairs it with [`AlphaFold3Processor`][], and writes a
    routable ``model.json`` / ``processor.json`` / ``pipeline.json`` / safetensors
    repository to ``output_dir`` through [`AlphaFold3Pipeline`][].

    Saving is local-only when ``repo_id`` is omitted.
    """

    model = convert_alphafold3_model(
        checkpoint_path,
        model_name=model_name,
        dtype=dtype,
    )
    AlphaFold3Pipeline(
        model=model,
        processor=AlphaFold3Processor(AlphaFold3ProcessorConfig(), dtype=dtype),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run alphafold3 checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official AF3 parameter archive (e.g. af3.bin.zst)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-name", default=None, help="official model name when the archive holds several")
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    convert_alphafold3_checkpoint(
        args.checkpoint,
        args.output_dir,
        model_name=args.model_name,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
