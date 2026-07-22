r"""
Unified AlphaFold 2 checkpoint conversion entry point.

Source-specific details live in ``ofoldx.models.alphafold2.converters``. The public
entry point selects exactly the released DeepMind NPZ/tar or OpenFold source format.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from ...artifacts import save_model
from .config import AlphaFold2Config
from .converters import deepmind, openfold
from .model import AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import AlphaFold2Processor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_OPENFOLD_TEMPLATE_PREFIXES = (
    "template_angle_embedder.",
    "template_pair_embedder.",
    "template_pair_stack.",
    "template_pointwise_att.",
)


def _as_openfold_tensor_state(value: object) -> Mapping[str, Tensor] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    if all(isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()):
        return cast(Mapping[str, Tensor], value)
    return None


def _select_openfold_inference_state(checkpoint: object) -> Mapping[str, Tensor]:
    r"""Select the inference weights from one released or training OpenFold checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("OpenFold checkpoint is not a mapping")

    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        for key in ("params", "state_dict"):
            state = _as_openfold_tensor_state(ema.get(key))
            if state is not None:
                return state
        state = _as_openfold_tensor_state(ema)
        if state is not None:
            return state

    for key in ("state_dict", "model_state_dict", "model", "params"):
        state = _as_openfold_tensor_state(checkpoint.get(key))
        if state is not None:
            return state
    state = _as_openfold_tensor_state(checkpoint)
    if state is not None:
        return state
    raise TypeError("OpenFold checkpoint does not contain a recognizable tensor state_dict mapping")


def _convert_openfold_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    dtype: torch.dtype | None,
) -> AlphaFold2Model:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = _select_openfold_inference_state(checkpoint)
    converted = openfold.convert_openfold_alphafold2_state_dict(state, dtype=dtype)
    config = AlphaFold2Config(
        {
            "heads": {"predicted_aligned_error": {"enabled": any(key.startswith("aux_heads.tm.") for key in state)}},
            "template": {"enabled": any(key.startswith(_OPENFOLD_TEMPLATE_PREFIXES) for key in state)},
        }
    )
    with torch.device("meta"):
        model = AlphaFold2Model(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> AlphaFold2Model:
    r"""
    Strict-load an AlphaFold 2 checkpoint into its native in-memory Model.

    Args:
        checkpoint_path: Source checkpoint path.
        source: Checkpoint source format. Supported values are ``"deepmind"`` and ``"openfold"``.
        model_name: Optional source-specific model id.
        dtype: Optional dtype for floating-point converted tensors.

    Returns:
        A strict-loaded AlphaFold2Model.
    """

    source = source.lower()
    if source == "deepmind":
        return deepmind.convert_deepmind_alphafold2_checkpoint(
            checkpoint_path,
            model_name=model_name,
            dtype=dtype,
        )
    if source == "openfold":
        if model_name is not None:
            raise ValueError("model_name selects a DeepMind NPZ member and is not valid for OpenFold checkpoints")
        return _convert_openfold_checkpoint(checkpoint_path, dtype=dtype)
    raise ValueError(f"unsupported AlphaFold 2 checkpoint source {source!r}; expected 'deepmind' or 'openfold'")


def save_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert an AlphaFold 2 checkpoint and write a standard OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_alphafold2_checkpoint`][ofoldx.models.alphafold2.convert.convert_alphafold2_checkpoint],
    pairs the resulting model with the processor selected by the checkpoint variant (the monomer/monomer-pTM
    :class:`AlphaFold2Processor`, or the :class:`AlphaFold2MultimerProcessor` when the selected
    ``model_*_multimer`` archive yields an :class:`AlphaFold2MultimerModel`), and writes a
    self-contained ``model.json`` / ``processor.json`` / safetensors repository via
    [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model].

    Saving is local-only by default. When ``repo_id`` is given the fully written repository is also
    uploaded to that Hub repo; no organization or token is assumed and nothing is published without
    an explicit ``repo_id``.
    """

    model = convert_alphafold2_checkpoint(
        checkpoint_path,
        source=source,
        model_name=model_name,
        dtype=dtype,
    )
    # The source converters reshape/transpose released tensors into the native tree, so some
    # parameters are views that alias (or do not cover) their storage. Reassign each as its own
    # contiguous clone: safetensors refuses to serialize storage-aliasing tensors on save.
    owned = {
        name: tensor.detach().clone(memory_format=torch.contiguous_format)
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(owned, strict=True, assign=True)
    if isinstance(model, AlphaFold2MultimerModel):
        processor: object = AlphaFold2MultimerProcessor()
    else:
        processor = AlphaFold2Processor()
    save_model(
        output_dir,
        model=model,
        processor=processor,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="AlphaFold 2 checkpoint (DeepMind NPZ/tar or OpenFold)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", choices=("deepmind", "openfold"), default="deepmind")
    parser.add_argument(
        "--model-name",
        default=None,
        help="DeepMind NPZ member name (e.g. model_1, model_1_multimer_v3)",
    )
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    save_alphafold2_checkpoint(
        args.checkpoint,
        args.output_dir,
        source=args.source,
        model_name=args.model_name,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
