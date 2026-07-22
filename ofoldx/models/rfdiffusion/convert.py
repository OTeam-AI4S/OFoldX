r"""Convert official RFdiffusion checkpoints to native OFoldX Models."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import chanfig
import torch
from torch import Tensor

from ...artifacts import save_model
from .config import RFdiffusionConfig
from .model import RFdiffusionModel, rfdiffusion_schedule

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_NETWORK_RENAMES = {
    "n_extra_block": "num_extra_blocks",
    "n_main_block": "num_main_blocks",
    "n_ref_block": "num_refinement_blocks",
    "d_msa": "msa_size",
    "d_msa_full": "full_msa_size",
    "d_pair": "pair_size",
    "d_templ": "template_size",
    "n_head_msa": "num_msa_attention_heads",
    "n_head_pair": "num_pair_attention_heads",
    "n_head_templ": "num_template_attention_heads",
    "d_hidden": "hidden_size",
    "d_hidden_templ": "template_hidden_size",
    "p_drop": "dropout",
    "freeze_track_motif": "freeze_track_motif",
    "SE3_param_full": "se3_param_full",
    "SE3_param_topk": "se3_param_topk",
}
_DIFFUSER_RENAMES = {
    "T": "num_timesteps",
    "b_0": "beta_initial",
    "b_T": "beta_final",
    "schedule_type": "schedule_type",
    "so3_type": "rotation_diffuser",
    "crd_scale": "coordinate_scale",
    "so3_schedule_type": "rotation_schedule_type",
    "min_b": "min_rotation_beta",
    "max_b": "max_rotation_beta",
    "min_sigma": "min_rotation_sigma",
    "max_sigma": "max_rotation_sigma",
}
_PREPROCESS_RENAMES = {
    "sidechain_input": "sidechain_input",
    "motif_sidechain_input": "motif_sidechain_input",
    "d_t1d": "template_1d_size",
    "d_t2d": "template_2d_size",
    "str_self_cond": "structure_self_conditioning",
}


def load_rfdiffusion_checkpoint(
    checkpoint: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    r"""Load one official checkpoint with PyTorch's restricted loader."""

    value = torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=True,
        mmap=True,
    )
    return dict(value)


def rfdiffusion_config_from_checkpoint(checkpoint: Mapping[str, object]) -> RFdiffusionConfig:
    r"""Recover only persistent network/diffusion/preprocess architecture."""

    source = cast(Mapping[str, object], checkpoint["config_dict"])
    values = {}
    for source_name, target_name, renames in (
        ("model", "network", _NETWORK_RENAMES),
        ("diffuser", "diffuser", _DIFFUSER_RENAMES),
        ("preprocess", "preprocess", _PREPROCESS_RENAMES),
    ):
        group = cast(Mapping[str, object], source[source_name])
        normalized = {target: group[key] for key, target in renames.items()}
        if target_name == "network":
            for key in ("se3_param_full", "se3_param_topk"):
                nested = cast(Mapping[object, object], normalized[key])
                normalized[key] = {
                    ("num_attention_heads" if item_key == "n_heads" else str(item_key)): item
                    for item_key, item in nested.items()
                }
        values[target_name] = normalized
    return RFdiffusionConfig(values)


def convert_rfdiffusion_state_dict_key(key: str) -> str:
    r"""Translate one official tensor name to the public Model module tree."""

    key = key.removeprefix("module.")
    if key.startswith("c6d_pred."):
        key = "coords_6d_pred." + key[len("c6d_pred.") :]
    key = key.replace("templ_emb.emb_t1d.", "templ_emb.emb_template_1d.")
    key = key.replace("templ_emb.proj_t1d.", "templ_emb.proj_template_1d.")
    key = key.replace(".ff.", ".feed_forward.")
    return f"network.{key}"


def convert_rfdiffusion_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""Return a strict-loadable native state dict."""

    return {
        convert_rfdiffusion_state_dict_key(key): (
            value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        )
        for key, value in state_dict.items()
    }


def convert_rfdiffusion_checkpoint(
    checkpoint: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    schedule_source: str | PathLike[str] | None = None,
) -> RFdiffusionModel:
    r"""Return one native Model loaded strictly from an official checkpoint.

    The released IGSO(3) schedule table is copied verbatim from ``schedule_source`` (an ``.npz`` or
    original RFdiffusion ``.pkl``; the untracked ``rfdiffusion_schedule.npz`` beside the module by
    default) into the model's persistent ``igso3_*`` float64 buffers, so it ships baked into
    ``model.safetensors`` rather than as vendored package data.
    """

    loaded = load_rfdiffusion_checkpoint(checkpoint)
    config = rfdiffusion_config_from_checkpoint(loaded)
    with torch.device("meta"):
        model = RFdiffusionModel(config)
    state = convert_rfdiffusion_state_dict(
        cast(Mapping[str, Tensor], loaded["model_state_dict"]),
        dtype=dtype,
    )
    for name, array in rfdiffusion_schedule(schedule_source).items():
        state[f"igso3_{name}"] = torch.as_tensor(array, dtype=torch.float64)
    model.load_state_dict(state, strict=True, assign=True)
    return model


def save_rfdiffusion_checkpoint(
    checkpoint: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
    schedule_source: str | PathLike[str] | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert an RFdiffusion checkpoint and write a routable OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_rfdiffusion_checkpoint`][ofoldx.models.rfdiffusion.convert.convert_rfdiffusion_checkpoint],
    pairs it with :class:`RFdiffusionProcessor`, and writes ``model.json`` /
    ``processor.json`` / safetensors via
    [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model]. The registered pair has no
    generic standard task, so a ``pipeline.json = {"pipeline": "ofoldx.rfdiffusion"}`` marker
    is written to route ``pipeline("design", model=dir)`` to :class:`RFdiffusionPipeline` rather than
    the generic standard ``design`` route.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from ...pipelines.registry import PIPELINES
    from .pipeline import RFdiffusionPipeline
    from .processor import RFdiffusionProcessor

    model = convert_rfdiffusion_checkpoint(checkpoint, dtype=dtype, schedule_source=schedule_source)
    destination = Path(output_dir).expanduser()
    save_model(destination, model=model, processor=RFdiffusionProcessor())
    chanfig.NestedDict({"pipeline": PIPELINES.id_for(RFdiffusionPipeline)}).dump(destination / "pipeline.json")
    if repo_id is not None:
        from ..._repository import upload_repository

        upload_repository(destination, repo_id, token=token, private=private)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official RFdiffusion checkpoint (e.g. Base_ckpt.pt)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--schedule-source",
        type=Path,
        required=True,
        help="IGSO(3) schedule source (the original RFdiffusion .pkl cache, or an .npz archive of "
        "the same table); required, as the table is not shipped in the package -- it is baked into "
        "the converted checkpoint's igso3_* buffers",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    save_rfdiffusion_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        schedule_source=args.schedule_source,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
