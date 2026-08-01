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

from .config import AlphaFold2Config, AlphaFold2InitialGuessConfig
from .converters import deepmind, openfold
from .model import AlphaFold2InitialGuessModel, AlphaFold2Model, AlphaFold2MultimerModel
from .multimer_processor import AlphaFold2MultimerProcessor
from .processor import AlphaFold2InitialGuessProcessor, AlphaFold2Processor

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
    r"""
    Select the inference weights from one released or training OpenFold checkpoint.
    """

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


def alphafold2_own_parameter_storage(model: AlphaFold2Model | AlphaFold2MultimerModel) -> None:
    r"""
    Give every converted parameter independent contiguous storage for safetensors.
    """

    owned = {
        name: tensor.detach().clone(memory_format=torch.contiguous_format)
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(owned, strict=True, assign=True)


def convert_alphafold2_model(
    checkpoint_path: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    casp14: bool = False,
) -> AlphaFold2Model | AlphaFold2MultimerModel:
    r"""
    Strict-load an AlphaFold 2 checkpoint into its native in-memory Model.

    Args:
        checkpoint_path: Source checkpoint path.
        source: Checkpoint source format. Supported values are ``"deepmind"`` and ``"openfold"``.
        model_name: Optional source-specific model id.
        dtype: Optional dtype for floating-point converted tensors.
        casp14: Whether to use the released CASP14 eight-ensemble monomer
            configuration.

    Returns:
        A strict-loaded monomer or multimer model matching the selected checkpoint.
    """

    source = source.lower()
    if source == "deepmind":
        model = deepmind.convert_deepmind_alphafold2_checkpoint(
            checkpoint_path,
            model_name=model_name,
            dtype=dtype,
        )
    elif source == "openfold":
        if model_name is not None:
            raise ValueError("model_name selects a DeepMind NPZ member and is not valid for OpenFold checkpoints")
        model = _convert_openfold_checkpoint(checkpoint_path, dtype=dtype)
    else:
        raise ValueError(f"unsupported AlphaFold 2 checkpoint source {source!r}; expected 'deepmind' or 'openfold'")
    if casp14:
        if isinstance(model, AlphaFold2MultimerModel):
            raise ValueError("the CASP14 eight-ensemble configuration applies only to AlphaFold 2 monomer models")
        model.config.num_ensembles = 8
        model.num_ensembles = 8
    return model


def convert_alphafold2_initial_guess_model(
    checkpoint_path: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
) -> AlphaFold2InitialGuessModel:
    r"""
    Strict-load the released ``model_1_ptm`` weights into the initial-guess graph.

    The target module has the ordinary pTM parameter topology. Its distinct
    Config, Input, Processor, and Pipeline encode the first-recycle coordinate
    state and target--binder template-pair equation without adding a mode to
    [`AlphaFold2Model`][].

    Args:
        checkpoint_path: Official DeepMind NPZ, params directory, or parameter tar.
        dtype: Optional dtype for floating-point converted tensors.

    Returns:
        A strict-loaded initial-guess model.
    """

    source_model = deepmind.convert_deepmind_alphafold2_checkpoint(
        checkpoint_path,
        model_name="model_1_ptm",
        dtype=dtype,
    )
    with torch.device("meta"):
        model = AlphaFold2InitialGuessModel(AlphaFold2InitialGuessConfig())
    model.load_state_dict(source_model.state_dict(), strict=True, assign=True)
    return model


def convert_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    casp14: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert an AlphaFold 2 checkpoint and write a standard OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_alphafold2_model`][ofoldx.models.alphafold2.convert.convert_alphafold2_model],
    pairs the resulting model with the processor selected by the checkpoint variant (the monomer/monomer-pTM
    [`AlphaFold2Processor`][], or the [`AlphaFold2MultimerProcessor`][] when the selected
    ``model_*_multimer`` archive yields an [`AlphaFold2MultimerModel`][]), and writes a
    self-contained ``model.json`` / ``processor.json`` / safetensors repository
    marked with the exact concrete folding Pipeline.

    Saving is local-only by default. When ``repo_id`` is given the fully written repository is also
    uploaded to that Hub repo; no organization or token is assumed and nothing is published without
    an explicit ``repo_id``. ``casp14=True`` persists eight input ensembles on a
    monomer configuration.
    """

    model = convert_alphafold2_model(
        checkpoint_path,
        source=source,
        model_name=model_name,
        dtype=dtype,
        casp14=casp14,
    )
    # The source converters reshape/transpose released tensors into views whose
    # storage safetensors cannot serialize directly.
    alphafold2_own_parameter_storage(model)
    from .pipeline import AlphaFold2MultimerPipeline, AlphaFold2Pipeline

    pipeline: AlphaFold2Pipeline | AlphaFold2MultimerPipeline
    if isinstance(model, AlphaFold2MultimerModel):
        pipeline = AlphaFold2MultimerPipeline(
            model=model,
            processor=AlphaFold2MultimerProcessor(dtype=dtype),
        )
    else:
        pipeline = AlphaFold2Pipeline(
            model=model,
            processor=AlphaFold2Processor(dtype=dtype),
        )
    pipeline.save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_alphafold2_casp14_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the fixed five official CASP14 checkpoints into one repository.

    ``checkpoint_path`` must contain the official DeepMind ``model_1`` through
    ``model_5`` parameter members. Each Model retains the ordinary
    ``alphafold2`` architecture and persists the released eight-input-ensemble
    configuration.

    Args:
        checkpoint_path: Official DeepMind parameter directory or tar containing
            ``model_1`` through ``model_5``.
        output_dir: Local composite repository.
        dtype: Optional dtype for floating-point converted tensors.
        repo_id: Optional Hub repository to upload after the local write.
        token: Optional Hugging Face token.
        private: Whether a newly created Hub repository is private.
    """

    from .pipeline import AlphaFold2Casp14Pipeline, AlphaFold2Pipeline

    models = []
    for model_name in ("model_1", "model_2", "model_3", "model_4", "model_5"):
        model = convert_alphafold2_model(
            checkpoint_path,
            source="deepmind",
            model_name=model_name,
            dtype=dtype,
            casp14=True,
        )
        assert isinstance(model, AlphaFold2Model)
        alphafold2_own_parameter_storage(model)
        models.append(model)
    model_1, model_2, model_3, model_4, model_5 = models

    AlphaFold2Casp14Pipeline(
        model_1=AlphaFold2Pipeline(
            model=model_1,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
        model_2=AlphaFold2Pipeline(
            model=model_2,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
        model_3=AlphaFold2Pipeline(
            model=model_3,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
        model_4=AlphaFold2Pipeline(
            model=model_4,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
        model_5=AlphaFold2Pipeline(
            model=model_5,
            processor=AlphaFold2Processor(dtype=dtype),
        ),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_alphafold2_initial_guess_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert the released AF2-IG pTM checkpoint into its distinct repository pair.

    Args:
        checkpoint_path: Official DeepMind NPZ, params directory, or parameter tar.
        output_dir: Local output repository.
        dtype: Optional floating-point checkpoint dtype.
        repo_id: Optional Hub repository to upload after the local write.
        token: Optional Hugging Face token.
        private: Whether a newly created Hub repository is private.
    """

    from .pipeline import AlphaFold2InitialGuessPipeline

    model = convert_alphafold2_initial_guess_model(
        checkpoint_path,
        dtype=dtype,
    )
    alphafold2_own_parameter_storage(model)
    AlphaFold2InitialGuessPipeline(
        model=model,
        processor=AlphaFold2InitialGuessProcessor(dtype=dtype),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run alphafold2 checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="AlphaFold 2 checkpoint (DeepMind NPZ/tar or OpenFold)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source", choices=("deepmind", "openfold"), default="deepmind")
    parser.add_argument(
        "--model-name",
        default=None,
        help="DeepMind NPZ member name (e.g. model_1, model_1_multimer_v3)",
    )
    route = parser.add_mutually_exclusive_group()
    route.add_argument(
        "--casp14",
        action="store_true",
        help="persist the released eight-input-ensemble monomer configuration",
    )
    route.add_argument(
        "--casp14-five-model",
        action="store_true",
        help="convert the fixed official model_1 through model_5 CASP14 composite",
    )
    route.add_argument(
        "--initial-guess",
        action="store_true",
        help="convert the fixed model_1_ptm initial-guess pair",
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
    if args.initial_guess:
        if args.source != "deepmind":
            parser.error("--initial-guess requires --source deepmind")
        if args.model_name is not None:
            parser.error("--model-name is not valid with --initial-guess")
        convert_alphafold2_initial_guess_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
        return
    if args.casp14_five_model:
        if args.source != "deepmind":
            parser.error("--casp14-five-model requires --source deepmind")
        if args.model_name is not None:
            parser.error("--model-name is not valid with --casp14-five-model")
        convert_alphafold2_casp14_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
        return
    convert_alphafold2_checkpoint(
        args.checkpoint,
        args.output_dir,
        source=args.source,
        model_name=args.model_name,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        casp14=args.casp14,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
