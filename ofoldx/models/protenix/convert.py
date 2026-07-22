r"""
Protenix checkpoint converter.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chanfig
import torch
from torch import Tensor, nn

from ...artifacts import save_model
from .. import _protenix_conversion

if TYPE_CHECKING:
    from .model import ProtenixModel


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_reference_state(checkpoint_path: str | Any, *, map_location: str | Any = "cpu") -> dict[str, Tensor]:
    r"""
    Load a reference checkpoint's tensor state, unwrapping the ``model`` container and ``module.`` prefixes.

    Reads the checkpoint, unwraps a ``{"model": ...}`` wrapper when present, drops the distributed-training
    ``module.`` key prefix, and keeps only tensor-valued entries.

    Args:
        checkpoint_path: Path (or file-like object) accepted by `torch.load`.
        map_location: Device mapping forwarded to `torch.load`; defaults to CPU.

    Returns:
        Mapping from clean checkpoint key to tensor.

    Examples:
        >>> state = load_reference_state("af3.pt")  # doctest: +SKIP
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    reference_state: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = source_key.removeprefix("module.")
        if key in reference_state:
            raise ValueError(f"multiple Protenix source tensors normalize to {key!r}")
        reference_state[key] = value
    return reference_state


def translate_reference_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], _protenix_conversion.ReferenceKey],
) -> dict[str, Tensor]:
    r"""Translate every target and require every source tensor to participate."""

    translated: dict[str, Tensor] = {}
    consumed: set[str] = set()
    for model_key in model.state_dict():
        reference_key = key_fn(model_key)
        reference_value = _protenix_conversion.lookup_reference_value(reference_state, reference_key)
        translated[model_key] = _protenix_conversion.transform_reference_value(model_key, reference_value)
        if isinstance(reference_key, tuple):
            consumed.update(reference_key)
        else:
            consumed.add(reference_key)

    unconsumed = sorted(set(reference_state).difference(consumed))
    if unconsumed:
        preview = ", ".join(repr(key) for key in unconsumed[:20])
        suffix = "" if len(unconsumed) <= 20 else f", ... ({len(unconsumed)} total)"
        raise ValueError(f"unconsumed Protenix source tensors: {preview}{suffix}")
    return translated


def translate_protenix_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every tensor in one released Protenix graph."""

    return translate_reference_state(model, reference_state, _protenix_reference_key)


def convert_protenix_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
) -> ProtenixModel:
    r"""
    Convert a Protenix base checkpoint into a strict-loaded native model.

    Args:
        checkpoint_path: Path to the released Protenix base checkpoint.
        dtype: Optional floating-point dtype for converted weights.

    Returns:
        The converted, strict-loaded `ProtenixModel`. Use `save_protenix_checkpoint` to persist the paired
        model/processor repository locally or on the Hub.

    Examples:
        >>> model = convert_protenix_checkpoint("protenix_base_default_v0.5.0.pt")  # doctest: +SKIP
    """

    from .config import ProtenixConfig, ProtenixConstraintConfig
    from .model import ProtenixConstraintModel, ProtenixModel

    reference_state = load_reference_state(Path(checkpoint_path))
    if "constraint_embedder.pocket_z_embedder.weight" in reference_state:
        model: ProtenixModel = ProtenixConstraintModel(ProtenixConstraintConfig())
    else:
        model = ProtenixModel(ProtenixConfig())
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_state_dict(translate_protenix_state(model, reference_state), strict=True)
    model.eval()
    return model


def save_protenix_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert a Protenix checkpoint and write a routable OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_protenix_checkpoint`][ofoldx.models.protenix.convert.convert_protenix_checkpoint] — which
    auto-selects the default :class:`ProtenixModel` or the constraint
    :class:`ProtenixConstraintModel` from the checkpoint tensors — pairs it with the matching processor,
    and writes a self-contained ``model.json`` / ``processor.json`` / safetensors repository
    via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model].

    The default pair routes through the generic standard ``fold`` route (no marker). The constraint
    pair additionally writes a ``pipeline.json = {"pipeline": "ofoldx.protenix_constraint"}`` marker so
    ``pipeline("fold", model=dir)`` routes to :class:`ProtenixConstraintPipeline`, whose ``fold``
    accepts and forwards the released ``atom_contacts`` / ``token_contacts`` / ``pocket`` channels that
    the generic ``FoldPipeline`` cannot.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from .model import ProtenixConstraintModel
    from .processor import ProtenixConstraintProcessor, ProtenixProcessor

    model = convert_protenix_checkpoint(Path(checkpoint_path).expanduser(), dtype=dtype)
    destination = Path(output_dir).expanduser()
    if isinstance(model, ProtenixConstraintModel):
        processor: object = ProtenixConstraintProcessor()
    else:
        processor = ProtenixProcessor()
    save_model(destination, model=model, processor=processor)
    if isinstance(model, ProtenixConstraintModel):
        from ...pipelines.registry import PIPELINES
        from .pipeline import ProtenixConstraintPipeline

        pipeline_id = PIPELINES.id_for(ProtenixConstraintPipeline)
        chanfig.NestedDict({"pipeline": pipeline_id}).dump(destination / "pipeline.json")
    if repo_id is not None:
        from ..._repository import upload_repository

        upload_repository(destination, repo_id, token=token, private=private)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="released Protenix base checkpoint (e.g. protenix_base_default_v0.5.0.pt)",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    save_protenix_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


def _protenix_reference_key(model_key: str) -> _protenix_conversion.ReferenceKey:
    if model_key.startswith("constraint_embedder."):
        return model_key
    return _protenix_conversion.reference_key(model_key)


if __name__ == "__main__":
    main()
