r"""Convert official RFdiffusion checkpoints to native OFoldX Models."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike

import torch
from torch import Tensor

from .config import RFdiffusionConfig
from .model import RFdiffusionModel

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

    source = checkpoint["config_dict"]
    values = {}
    for source_name, target_name, renames in (
        ("model", "network", _NETWORK_RENAMES),
        ("diffuser", "diffuser", _DIFFUSER_RENAMES),
        ("preprocess", "preprocess", _PREPROCESS_RENAMES),
    ):
        group = source[source_name]
        normalized = {target: group[key] for key, target in renames.items()}
        if target_name == "network":
            for key in ("se3_param_full", "se3_param_topk"):
                nested = normalized[key]
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
) -> RFdiffusionModel:
    r"""Return one native Model loaded strictly from an official checkpoint."""

    loaded = load_rfdiffusion_checkpoint(checkpoint)
    config = rfdiffusion_config_from_checkpoint(loaded)
    with torch.device("meta"):
        model = RFdiffusionModel(config)
    state = convert_rfdiffusion_state_dict(loaded["model_state_dict"], dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model
