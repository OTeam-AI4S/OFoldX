r"""
Biohub ESMFold2 checkpoint conversion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from .config import ESMFold2Config

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DEFAULT_REPO = "biohub/ESMFold2"


def convert_esmfold2_checkpoint(
    checkpoint_or_repo: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    config_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    r"""
    Convert a Biohub ESMFold2 safetensors checkpoint into an OFoldX artifact.

    Args:
        checkpoint_or_repo: Local ``model.safetensors`` path, directory containing
            ``model.safetensors``, or Hugging Face repo id such as ``biohub/ESMFold2``.
        output_dir: Destination directory for ``manifest.json``, ``config.json``, and
            ``model.safetensors``.
        config_path: Optional local config JSON. When omitted, ``config.json`` is
            read beside the checkpoint or downloaded from the Hub repo.
        dtype: Optional floating-point dtype for saved weights.
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: Resolve Hub files from local cache only.

    Returns:
        JSON-serializable conversion summary.
    """

    from .model import ESMFold2Model

    checkpoint_path, resolved_config_path, source = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = _load_config(resolved_config_path) if resolved_config_path is not None else ESMFold2Config()
    config.architecture = "esmfold2"
    model = ESMFold2Model(config)
    if dtype is not None:
        model.to(dtype=dtype)
    state_dict = _cast_floating_tensors(load_esmfold2_state_dict(checkpoint_path), dtype)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)
    return {
        "architecture": config.architecture,
        "checkpoint": str(checkpoint_path),
        "config": None if resolved_config_path is None else str(resolved_config_path),
        "dtype": None if dtype is None else str(dtype).removeprefix("torch."),
        "dtype_counts": _dtype_counts(state_dict),
        "output_dir": str(output_dir),
        "source": source,
        "tensor_count": len(state_dict),
    }


def resolve_esmfold2_source(
    checkpoint_or_repo: str | PathLike[str],
    *,
    config_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
) -> tuple[Path, Path | None, str]:
    r"""Resolve a local or Hub ESMFold2 checkpoint source."""

    source = str(checkpoint_or_repo)
    path = Path(checkpoint_or_repo)
    if path.is_dir():
        checkpoint_path = path / "model.safetensors"
        resolved_config = Path(config_path) if config_path is not None else path / "config.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ESMFold2 checkpoint does not exist: {checkpoint_path}")
        if not resolved_config.is_file():
            raise FileNotFoundError(f"ESMFold2 config does not exist: {resolved_config}")
        return checkpoint_path, resolved_config, str(path)
    if path.is_file():
        resolved_config = Path(config_path) if config_path is not None else path.with_name("config.json")
        if not resolved_config.is_file():
            raise FileNotFoundError(f"ESMFold2 config does not exist: {resolved_config}")
        return path, resolved_config, str(path)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - dependency is present in normal Hub workflows.
        raise ImportError("huggingface_hub is required to download ESMFold2 checkpoints") from error

    hub_cache_dir = None if cache_dir is None else Path(cache_dir)
    checkpoint_path = Path(
        hf_hub_download(
            source,
            filename="model.safetensors",
            cache_dir=hub_cache_dir,
            local_files_only=local_files_only,
        )
    )
    if config_path is None:
        resolved_config = Path(
            hf_hub_download(
                source,
                filename="config.json",
                cache_dir=hub_cache_dir,
                local_files_only=local_files_only,
            )
        )
    else:
        resolved_config = Path(config_path)
    return checkpoint_path, resolved_config, source


def load_esmfold2_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, Tensor]:
    r"""Load Biohub ESMFold2 safetensors and drop non-parameter extra-state tensors."""

    state = load_file(str(checkpoint_path), device="cpu")
    return {key: value for key, value in state.items() if not key.endswith("._extra_state")}


def _load_config(config_path: Path) -> ESMFold2Config:
    raw = json.loads(config_path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError(f"ESMFold2 config must be a JSON object: {config_path}")
    return ESMFold2Config(_translate_biohub_config(raw))


def _translate_biohub_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(raw_config))
    config.pop("architectures", None)
    _rename(config, "d_pair", "pair_size")
    _rename(config, "d_single", "single_size")
    _rename(config, "n_relative_residx_bins", "num_relative_residue_bins")
    _rename(config, "n_relative_chain_bins", "num_relative_chain_bins")
    _rename(config, "lm_d_model", "language_model_hidden_size")
    _rename(config, "lm_num_layers", "num_language_model_layers")
    _rename(config, "lm_dropout", "language_model_dropout")
    _rename(config, "lm_mask_pct", "language_model_mask_probability")
    _rename(
        config,
        "force_lm_dropout_during_inference",
        "force_language_model_dropout_during_inference",
    )

    inputs = _mapping(config.get("inputs"))
    _rename(inputs, "d_inputs", "input_size")
    atom_encoder = _mapping(inputs.get("atom_encoder"))
    _rename(atom_encoder, "d_atom", "atom_size")
    _rename(atom_encoder, "d_token", "token_size")
    _rename(atom_encoder, "n_blocks", "num_blocks")
    _rename(atom_encoder, "n_heads", "num_attention_heads")
    _rename(atom_encoder, "swa_window_size", "sliding_window_size")
    _rename(atom_encoder, "n_spatial_rope_pairs_per_axis", "num_spatial_rope_pairs_per_axis")
    _rename(atom_encoder, "n_uid_rope_pairs", "num_uid_rope_pairs")

    _translate_folding_trunk(_mapping(config.get("folding_trunk")))

    confidence_head = _mapping(config.get("confidence_head"))
    _translate_folding_trunk(_mapping(confidence_head.get("folding_trunk")))

    msa_encoder = _mapping(config.get("msa_encoder"))
    _rename(msa_encoder, "d_msa", "msa_size")
    _rename(msa_encoder, "d_hidden", "hidden_size")
    _rename(msa_encoder, "n_layers", "num_blocks")
    _rename(msa_encoder, "n_heads_msa", "num_msa_attention_heads")
    _rename(msa_encoder, "msa_head_width", "msa_head_dim")

    parcae = _mapping(config.get("parcae"))
    _rename(parcae, "coda_n_layers", "num_coda_blocks")

    lm_encoder = _mapping(config.get("lm_encoder"))
    _rename(lm_encoder, "n_layers", "num_blocks")
    _rename(lm_encoder, "lm_dropout", "dropout")
    _rename(lm_encoder, "per_loop_lm_dropout", "per_loop_dropout")

    structure_head = _mapping(config.get("structure_head"))
    _rename(structure_head, "inference_num_steps", "num_sampling_steps")
    diffusion = _mapping(structure_head.get("diffusion_module"))
    _rename(diffusion, "c_atom", "atom_size")
    _rename(diffusion, "c_token", "token_size")
    _rename(diffusion, "c_z", "pair_size")
    _rename(diffusion, "c_s_inputs", "input_size")
    _rename(diffusion, "atom_num_blocks", "num_atom_blocks")
    _rename(diffusion, "atom_num_heads", "num_atom_attention_heads")
    _rename(diffusion, "token_num_blocks", "num_token_blocks")
    _rename(diffusion, "token_num_heads", "num_token_attention_heads")
    return config


def _translate_folding_trunk(config: dict[str, Any]) -> None:
    _rename(config, "n_layers", "num_blocks")
    _rename(config, "n_heads", "num_attention_heads")


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _rename(config: dict[str, Any], old: str, new: str) -> None:
    if old in config:
        config[new] = config.pop(old)


def _cast_floating_tensors(state_dict: Mapping[str, Tensor], dtype: torch.dtype | None) -> dict[str, Tensor]:
    r"""Return ``state_dict`` with floating tensors cast to ``dtype`` when requested."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""Return a sorted count of tensors per dtype name."""

    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_or_repo", help=f"Local checkpoint/directory or Hub repo id, e.g. {_DEFAULT_REPO}")
    parser.add_argument("output_dir", type=Path, help="Output pretrained directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional local ESMFold2 config.json")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Resolve Hub files from local cache only")
    parser.add_argument("--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep")
    add_hub_upload_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    r"""Command-line entry point."""

    args = _parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    report = convert_esmfold2_checkpoint(
        args.checkpoint_or_repo,
        args.output_dir,
        config_path=args.config,
        dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted ESMFold2 checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
