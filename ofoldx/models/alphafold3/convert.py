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

import torch

from ...artifacts import save_model
from .config import AlphaFold3ProcessorConfig
from .converters import deepmind
from .processor import AlphaFold3Processor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def convert_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = True,
) -> None:
    r"""Strict-load official AF3 parameters and write a standard OFoldX repository.

    Converts the DeepMind AlphaFold 3 parameter archive at ``checkpoint_path`` into a
    native model, pairs it with the family :class:`AlphaFold3Processor`, and writes a
    self-contained ``model.json`` / ``processor.json`` / safetensors repository to
    ``output_dir`` via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model].

    Saving is local-only by default. Official DeepMind AlphaFold 3 parameters are **not
    redistributable (RED)**: any Hub upload defaults to ``private=True`` and a public
    upload is refused. When ``repo_id`` is given the fully written repository is uploaded
    privately; passing ``private=False`` with a ``repo_id`` raises rather than publishing
    the restricted weights. Omitting ``repo_id`` never touches the network.
    """

    if repo_id is not None and not private:
        raise ValueError(
            "refusing to publish AlphaFold 3 parameters to a public Hub repository: the official "
            "DeepMind AlphaFold 3 weights are NOT redistributable (RED). Upload privately "
            "(private=True, the default) or save locally by omitting repo_id."
        )

    model = deepmind.convert_official_alphafold3_checkpoint(
        checkpoint_path,
        model_name=model_name,
        dtype=dtype,
    )
    save_model(
        output_dir,
        model=model,
        processor=AlphaFold3Processor(AlphaFold3ProcessorConfig()),
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
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
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="upload the --repo-id repo as private; AF3 weights are RED, so --no-private is refused",
    )
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
