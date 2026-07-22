r"""Strict translation of the official PxDesign v0.1 checkpoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import chanfig
import torch
from torch import Tensor, nn

from ...artifacts import save_model
from .. import _protenix_conversion
from .config import PxDesignConfig
from .model import PxDesignModel

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def translate_pxdesign_state(
    model: nn.Module,
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""Translate every official tensor to one exact native model tensor."""

    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    reference_state = _extract_reference_state(checkpoint_or_state_dict)
    converted = _protenix_conversion.translate_reference_state(model, reference_state, _pxdesign_reference_key)

    expected_reference_keys: set[str] = set()
    for model_key in model.state_dict():
        reference_key = _pxdesign_reference_key(model_key)
        expected_reference_keys.update(reference_key if isinstance(reference_key, tuple) else (reference_key,))
    unexpected = sorted(set(reference_state) - expected_reference_keys)
    if unexpected:
        raise RuntimeError(
            "checkpoint translation failed\n"
            f"unexpected: {len(unexpected)}\n" + "\n".join(f"  unexpected {key}" for key in unexpected[:20])
        )

    if dtype is None:
        return converted
    return {key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in converted.items()}


def convert_pxdesign_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> PxDesignModel:
    r"""Build a PxDesign model and strict-load the complete official checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)
    config = PxDesignConfig()
    with torch.device("meta"):
        model = PxDesignModel(config)
    state_dict = translate_pxdesign_state(model, checkpoint, dtype=dtype)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    return model


def save_pxdesign_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert a PxDesign checkpoint and write a routable OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_pxdesign_checkpoint`][ofoldx.models.pxdesign.convert.convert_pxdesign_checkpoint], pairs
    it with :class:`PxDesignProcessor`, and writes ``model.json`` / ``processor.json`` /
    safetensors via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model]. The registered pair
    has no generic standard task, so a
    ``pipeline.json = {"pipeline": "ofoldx.pxdesign"}`` marker is written to route
    ``pipeline("design", model=dir)`` to :class:`PxDesignPipeline` — which returns the
    coordinate-producing ``PxDesignResult`` — rather than the generic standard ``design`` route.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from ...pipelines.registry import PIPELINES
    from .pipeline import PxDesignPipeline
    from .processor import PxDesignProcessor

    model = convert_pxdesign_checkpoint(checkpoint_path, map_location=map_location, dtype=dtype)
    destination = Path(output_dir).expanduser()
    save_model(destination, model=model, processor=PxDesignProcessor())
    chanfig.NestedDict({"pipeline": PIPELINES.id_for(PxDesignPipeline)}).dump(destination / "pipeline.json")
    if repo_id is not None:
        from ..._repository import upload_repository

        upload_repository(destination, repo_id, token=token, private=private)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="released PxDesign checkpoint (e.g. pxdesign_v0.1.0.pt)")
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
    save_pxdesign_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


def _extract_reference_state(checkpoint_or_state_dict: Mapping[str, Any]) -> dict[str, Tensor]:
    state = checkpoint_or_state_dict.get("model", checkpoint_or_state_dict)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a model mapping or be a state dict")
    normalized = {}
    for key, value in state.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized:
            raise ValueError(f"checkpoint keys collide after removing module. prefix: {normalized_key!r}")
        normalized[normalized_key] = value
    return normalized


def _pxdesign_reference_key(model_key: str) -> str | tuple[str, ...]:
    if model_key.startswith("condition_embedder.input_map."):
        return "design_condition_embedder.input_embedder.input_map." + model_key.removeprefix(
            "condition_embedder.input_map."
        )
    if model_key.startswith("condition_embedder.template_embedder.embedding."):
        return "design_condition_embedder.condition_template_embedder.embedder." + model_key.removeprefix(
            "condition_embedder.template_embedder.embedding."
        )
    if model_key.startswith("condition_embedder.atom_encoder."):
        reference_key = _protenix_conversion.reference_key(
            "input_embedder.atom_encoder." + model_key.removeprefix("condition_embedder.atom_encoder.")
        )
        source = "input_embedder.atom_attention_encoder."
        target = "design_condition_embedder.input_embedder.atom_attention_encoder."
        if isinstance(reference_key, tuple):
            return tuple(target + key[len(source) :] if key.startswith(source) else key for key in reference_key)
        return target + reference_key[len(source) :] if reference_key.startswith(source) else reference_key
    if model_key.startswith("coordinate_denoiser."):
        return _protenix_conversion.reference_key(model_key)
    return model_key


if __name__ == "__main__":
    main()
