r"""Convert Biohub ESMFold2 and MultiMolecule ESMC into one OFoldX repository."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from multimolecule import EsmCModel
from safetensors.torch import load_file
from torch import Tensor

from ...artifacts import save_model
from ...references import classify_reference
from .config import ESMFold2Config, ESMFold2ESMCConfig, ESMFold2ExperimentalConfig, ESMFold2ProcessorConfig
from .model import ESMFold2ExperimentalModel, ESMFold2Model
from .processor import ESMFold2Processor

_DEFAULT_ESMC_REPOSITORY = "multimolecule/esmc-6b"
_BIOHUB_UNUSED_STATE_KEYS = (
    "confidence_head.s_input_to_s.weight",
    "confidence_head.s_inputs_to_single.weight",
    "confidence_head.s_norm.bias",
    "confidence_head.s_norm.weight",
)
_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def convert_esmfold2_checkpoint(
    checkpoint_or_repo: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    esmc: EsmCModel | None = None,
    esmc_source: str | PathLike[str] | None = None,
    config_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Write a self-contained ``model.json``/``processor.json``/safetensors repository.

    Local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model];
    no organization or token is assumed and nothing is published without an explicit ``repo_id``.
    """

    if esmc is not None and esmc_source is not None:
        raise ValueError("esmc and esmc_source are mutually exclusive")
    if esmc is None:
        esmc_source = _DEFAULT_ESMC_REPOSITORY if esmc_source is None else esmc_source
        esmc = EsmCModel.from_pretrained(
            esmc_source,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
        )

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    esmc_config = ESMFold2ESMCConfig({name: getattr(esmc.config, name) for name in ESMFold2ESMCConfig().dict()})
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if raw_config["model_type"] != "esmfold2" or raw_config["type"] != "release":
        raise ValueError("checkpoint config is not the released Biohub ESMFold2 architecture")
    translated = _translate_biohub_config(raw_config)
    translated["esmc"] = esmc_config.dict()
    config = ESMFold2Config(translated)
    model = ESMFold2Model(config, esmc=esmc)

    head_state = load_esmfold2_state_dict(checkpoint_path)
    for name in _BIOHUB_UNUSED_STATE_KEYS:
        head_state.pop(name, None)
    if dtype is not None:
        model.to(dtype=dtype)
    incompatible = model.load_state_dict(head_state, strict=False)
    expected_missing = {f"esmc.{name}" for name in model.esmc.state_dict()}
    missing = set(incompatible.missing_keys)
    if missing != expected_missing or incompatible.unexpected_keys:
        missing_head = sorted(missing - expected_missing)
        unexpected = sorted(incompatible.unexpected_keys)
        raise RuntimeError(
            f"ESMFold2 checkpoint does not match the released head; "
            f"missing_head={missing_head!r}, unexpected={unexpected!r}"
        )

    save_model(
        output_dir,
        model=model,
        processor=ESMFold2Processor(ESMFold2ProcessorConfig({"msa_conditioning": config.msa_encoder.enabled})),
        repo_id=repo_id,
        token=token,
        private=private,
    )


def resolve_esmfold2_source(
    checkpoint_or_repo: str | PathLike[str],
    *,
    config_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
) -> tuple[Path, Path]:
    r"""Resolve the upstream folding-head checkpoint and its source config."""

    classified = classify_reference(checkpoint_or_repo)
    if isinstance(classified, Path):
        if classified.is_dir():
            checkpoint = classified / "model.safetensors"
            config = Path(config_path).expanduser() if config_path is not None else classified / "config.json"
        elif classified.is_file():
            checkpoint = classified
            config = Path(config_path).expanduser() if config_path is not None else classified.with_name("config.json")
        else:
            raise FileNotFoundError(f"ESMFold2 checkpoint does not exist: {classified}")
    else:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                classified,
                filename="model.safetensors",
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
            )
        )
        config = (
            Path(config_path).expanduser()
            if config_path is not None
            else Path(
                hf_hub_download(
                    classified,
                    filename="config.json",
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    local_files_only=local_files_only,
                )
            )
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ESMFold2 checkpoint does not exist: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"ESMFold2 config does not exist: {config}")
    return checkpoint, config


def load_esmfold2_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, Tensor]:
    r"""Load the released folding head, excluding Transformers bookkeeping tensors."""

    state = load_file(str(checkpoint_path), device="cpu")
    return {name: tensor for name, tensor in state.items() if not name.endswith("._extra_state")}


def _translate_biohub_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    inputs = raw_config["inputs"]
    atom_encoder = inputs["atom_encoder"]
    folding_trunk = raw_config["folding_trunk"]
    structure = raw_config["structure_head"]
    diffusion = structure["diffusion_module"]
    confidence = raw_config["confidence_head"]
    confidence_trunk = confidence["folding_trunk"]
    msa = raw_config["msa_encoder"]
    parcae = raw_config["parcae"]
    lm_encoder = raw_config["lm_encoder"]

    # The published folding API uses 0.1 for ESMFold2-Fast and 0.0 for the
    # MSA-capable model, while both released JSON files omit this runtime
    # default. The release architecture identifies Fast by its disabled MSA
    # encoder; an explicit future config value takes precedence.
    language_model_mask_probability = raw_config.get(
        "lm_mask_pct",
        0.0 if raw_config["msa_encoder"]["enabled"] else 0.1,
    )
    return {
        "single_size": raw_config["d_single"],
        "pair_size": raw_config["d_pair"],
        "num_relative_residue_bins": raw_config["n_relative_residx_bins"],
        "num_relative_chain_bins": raw_config["n_relative_chain_bins"],
        "num_loops": raw_config["num_loops"],
        "language_model_mask_probability": language_model_mask_probability,
        "msa_encoder_overwrite": raw_config["msa_encoder_overwrite"],
        "inputs": {
            "input_size": inputs["d_inputs"],
            "atom_encoder": {
                "atom_size": atom_encoder["d_atom"],
                "token_size": atom_encoder["d_token"],
                "num_blocks": atom_encoder["n_blocks"],
                "num_attention_heads": atom_encoder["n_heads"],
                "sliding_window_size": atom_encoder["swa_window_size"],
                "expansion_ratio": atom_encoder["expansion_ratio"],
                "spatial_rope_base_frequency": atom_encoder["spatial_rope_base_frequency"],
                "num_spatial_rope_pairs_per_axis": atom_encoder["n_spatial_rope_pairs_per_axis"],
                "num_uid_rope_pairs": atom_encoder["n_uid_rope_pairs"],
                "uid_rope_base_frequency": atom_encoder["uid_rope_base_frequency"],
            },
        },
        # n_heads/dropout are inert upstream dataclass leftovers: the released
        # PairUpdateBlock has neither an attention nor a dropout operation.
        "folding_trunk": {"num_blocks": folding_trunk["n_layers"]},
        "structure_head": {
            "diffusion_module": {
                "sigma_data": diffusion["sigma_data"],
                "atom_size": diffusion["c_atom"],
                "token_size": diffusion["c_token"],
                "pair_size": diffusion["c_z"],
                "input_size": diffusion["c_s_inputs"],
                "fourier_dim": diffusion["fourier_dim"],
                "num_atom_blocks": diffusion["atom_num_blocks"],
                "num_atom_attention_heads": diffusion["atom_num_heads"],
                "num_token_blocks": diffusion["token_num_blocks"],
                "num_token_attention_heads": diffusion["token_num_heads"],
                "transition_multiplier": diffusion["transition_multiplier"],
            },
            "distogram_bins": structure["distogram_bins"],
            "gamma_0": structure["gamma_0"],
            "gamma_min": structure["gamma_min"],
            "noise_scale": structure["noise_scale"],
            "step_scale": structure["step_scale"],
            "inference_s_max": structure["inference_s_max"],
            "inference_s_min": structure["inference_s_min"],
            "inference_p": structure["inference_p"],
            "num_sampling_steps": structure["inference_num_steps"],
        },
        "confidence_head": {
            "num_plddt_bins": confidence["num_plddt_bins"],
            "num_pde_bins": confidence["num_pde_bins"],
            "num_pae_bins": confidence["num_pae_bins"],
            "min_dist": confidence["min_dist"],
            "max_dist": confidence["max_dist"],
            "distogram_bins": confidence["distogram_bins"],
            "folding_trunk": {"num_blocks": confidence_trunk["n_layers"]},
        },
        "msa_encoder": {
            "enabled": msa["enabled"],
            "msa_size": msa["d_msa"],
            "hidden_size": msa["d_hidden"],
            "num_blocks": msa["n_layers"],
            "num_msa_attention_heads": msa["n_heads_msa"],
            "msa_head_dim": msa["msa_head_width"],
        },
        # enabled/poisson_mean/min_steps/max_steps are not read by the released
        # Model; it always runs num_loops + 1 updates and always builds the coda.
        "parcae": {"num_coda_blocks": parcae["coda_n_layers"]},
        "lm_encoder": {
            "enabled": lm_encoder["enabled"],
            "num_blocks": lm_encoder["n_layers"],
            "dropout": lm_encoder["lm_dropout"],
            "per_loop_dropout": lm_encoder["per_loop_lm_dropout"],
        },
    }


def experimental_config_from_biohub(raw_config: Mapping[str, Any]) -> ESMFold2ExperimentalConfig:
    r"""Build an [`ESMFold2ExperimentalConfig`][ofoldx.models.esmfold2.ESMFold2ExperimentalConfig]
    from a ``type="experimental"`` recurrent-graph checkpoint ``config.json``.

    Every graph dimension is read from the checkpoint (the Fast inversion model ships 24
    trunk blocks with the MSA encoder disabled; the MSA-conditioned critic ships 48 blocks);
    nothing is guessed.
    """

    if raw_config.get("model_type") != "esmfold2" or raw_config.get("type") != "experimental":
        raise ValueError("checkpoint config is not the experimental Biohub ESMFold2 architecture")

    config = ESMFold2ExperimentalConfig()
    config.single_size = raw_config["d_single"]
    config.pair_size = raw_config["d_pair"]
    config.num_relative_residue_bins = raw_config["n_relative_residx_bins"]
    config.num_relative_chain_bins = raw_config["n_relative_chain_bins"]
    config.num_loops = raw_config["num_loops"]
    config.disable_msa_features = raw_config["disable_msa_features"]
    config.language_model_dropout = raw_config.get("lm_dropout", 0.0)
    config.force_language_model_dropout = raw_config.get("force_lm_dropout_during_inference", False)

    # ESMC trunk width/depth the folding head projects from (``LanguageModelPairEncoder``
    # base_z_linear/base_z_combine): read from the checkpoint so the smaller scaling-study
    # trunks strict-load. The 6B checkpoints ship 2560/80 (the defaults); the 300M/600M
    # scaling critics ship 960/30 and 1152/36.
    config.esmc.hidden_size = raw_config.get("lm_d_model", config.esmc.hidden_size)
    config.esmc.num_hidden_layers = raw_config.get("lm_num_layers", config.esmc.num_hidden_layers)

    inputs = raw_config["inputs"]
    atom_encoder = inputs["atom_encoder"]
    config.inputs.input_size = inputs["d_inputs"]
    ae = config.inputs.atom_encoder
    ae.atom_size = atom_encoder["d_atom"]
    ae.token_size = atom_encoder["d_token"]
    ae.num_blocks = atom_encoder["n_blocks"]
    ae.num_attention_heads = atom_encoder["n_heads"]
    ae.sliding_window_size = atom_encoder["swa_window_size"]
    ae.expansion_ratio = atom_encoder["expansion_ratio"]
    ae.spatial_rope_base_frequency = atom_encoder["spatial_rope_base_frequency"]
    ae.num_spatial_rope_pairs_per_axis = atom_encoder["n_spatial_rope_pairs_per_axis"]
    ae.num_uid_rope_pairs = atom_encoder["n_uid_rope_pairs"]
    ae.uid_rope_base_frequency = atom_encoder["uid_rope_base_frequency"]

    config.folding_trunk.num_blocks = raw_config["folding_trunk"]["n_layers"]

    structure = raw_config["structure_head"]
    diffusion = structure["diffusion_module"]
    sh = config.structure_head
    sh.distogram_bins = structure["distogram_bins"]
    sh.gamma_0 = structure["gamma_0"]
    sh.gamma_min = structure["gamma_min"]
    sh.noise_scale = structure["noise_scale"]
    sh.step_scale = structure["step_scale"]
    sh.inference_s_max = structure["inference_s_max"]
    sh.inference_s_min = structure["inference_s_min"]
    sh.inference_p = structure["inference_p"]
    sh.num_sampling_steps = structure["inference_num_steps"]
    dm = sh.diffusion_module
    dm.sigma_data = diffusion["sigma_data"]
    dm.atom_size = diffusion["c_atom"]
    dm.token_size = diffusion["c_token"]
    dm.pair_size = diffusion["c_z"]
    dm.input_size = diffusion["c_s_inputs"]
    dm.fourier_dim = diffusion["fourier_dim"]
    dm.num_atom_blocks = diffusion["atom_num_blocks"]
    dm.num_atom_attention_heads = diffusion["atom_num_heads"]
    dm.num_token_blocks = diffusion["token_num_blocks"]
    dm.num_token_attention_heads = diffusion["token_num_heads"]
    dm.transition_multiplier = diffusion["transition_multiplier"]

    confidence = raw_config["confidence_head"]
    ch = config.confidence_head
    ch.enabled = confidence.get("enabled", True)
    ch.distogram_bins = confidence["distogram_bins"]
    ch.min_dist = confidence["min_dist"]
    ch.max_dist = confidence["max_dist"]
    ch.num_plddt_bins = confidence["num_plddt_bins"]
    ch.num_pae_bins = confidence["num_pae_bins"]
    ch.folding_trunk.num_blocks = confidence["folding_trunk"]["n_layers"]

    msa = raw_config["msa_encoder"]
    mc = config.msa_encoder
    mc.enabled = msa["enabled"]
    mc.msa_size = msa["d_msa"]
    mc.hidden_size = msa["d_hidden"]
    mc.num_blocks = msa["n_layers"]
    mc.num_msa_attention_heads = msa["n_heads_msa"]
    mc.msa_head_dim = msa["msa_head_width"]

    return config


def load_experimental_esmfold2(
    checkpoint_or_repo: str | PathLike[str],
    *,
    esmc: EsmCModel | None = None,
    config_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
) -> ESMFold2ExperimentalModel:
    r"""Strict-load a ``type="experimental"`` recurrent ESMFold2 head into an OFoldX model.

    The MultiMolecule ESMC child is loaded separately (the head safetensors does not carry
    it): the head strict-loads first with ``esmc=None`` and the optional ``esmc`` is attached
    afterwards via [`load_esmc`][ofoldx.models.esmfold2.ESMFold2ExperimentalModel.load_esmc].
    """

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    config = experimental_config_from_biohub(raw_config)
    model = ESMFold2ExperimentalModel(config, esmc=None)
    head_state = load_esmfold2_state_dict(checkpoint_path)
    model.load_state_dict(head_state, strict=True)
    if esmc is not None:
        model.load_esmc(esmc)
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_or_repo", help="Head checkpoint or Hub repo, e.g. biohub/ESMFold2")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--esmc", default=_DEFAULT_ESMC_REPOSITORY, help="MultiMolecule ESMC repository")
    parser.add_argument("--config", type=Path, default=None, help="Optional upstream config.json")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    convert_esmfold2_checkpoint(
        args.checkpoint_or_repo,
        args.output_dir,
        esmc_source=args.esmc,
        config_path=args.config,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )
