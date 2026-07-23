r"""Convert Biohub ESMFold2 and MultiMolecule ESMC into one OFoldX repository."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from multimolecule import AutoModel, AutoTokenizer, EsmCForMaskedLM, EsmCModel, ProteinTokenizer
from safetensors.torch import load_file
from torch import Tensor
from transformers import AutoModelForMaskedLM

from ...artifacts import save_model
from ...references import classify_reference
from .config import (
    EsmFold2Config,
    EsmFold2EsmCConfig,
    EsmFold2ExperimentalConfig,
    EsmFold2ExperimentalProcessorConfig,
    EsmFold2ProcessorConfig,
)
from .model import EsmFold2ExperimentalModel, EsmFold2Model
from .processor import EsmFold2ExperimentalProcessor, EsmFold2Processor

_DEFAULT_ESMC_REPOSITORY = "multimolecule/esmc-6b"
_BIOHUB_STATE_PREFIXES = (
    ("inputs_embedder.", "input_embeddings."),
    ("rel_pos.", "relative_position_embedder."),
    ("language_model.", "language_model_pair_encoder."),
    ("lm_encoder.", "language_model_encoder."),
    ("structure_head.", "diffusion_structure_head."),
)
_BIOHUB_UNUSED_STATE_KEYS = (
    "confidence_head.input_skip_projection.weight",
    "confidence_head.input_to_single_projection.weight",
    "confidence_head.single_norm.bias",
    "confidence_head.single_norm.weight",
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
        esmc = cast(
            EsmCModel,
            AutoModel.from_pretrained(
                esmc_source,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                token=token,
            ),
        )

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    esmc_config = EsmFold2EsmCConfig({name: getattr(esmc.config, name) for name in EsmFold2EsmCConfig().dict()})
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if raw_config["model_type"] != "esmfold2" or raw_config["type"] != "release":
        raise ValueError("checkpoint config is not the released Biohub ESMFold2 architecture")
    translated = _translate_biohub_config(raw_config)
    translated["esmc"] = esmc_config.dict()
    config = EsmFold2Config(translated)
    model = EsmFold2Model(config, esmc=esmc)

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
        processor=EsmFold2Processor(EsmFold2ProcessorConfig({"msa_conditioning": config.msa_encoder.enabled})),
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
    token: str | None = None,
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
                token=token,
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
                    token=token,
                )
            )
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ESMFold2 checkpoint does not exist: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"ESMFold2 config does not exist: {config}")
    return checkpoint, config


def load_esmfold2_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, Tensor]:
    r"""Load and strictly translate one Biohub folding-head state dictionary."""

    source_state = load_file(str(checkpoint_path), device="cpu")
    translated: dict[str, Tensor] = {}

    for source_name, tensor in source_state.items():
        if source_name.endswith("._extra_state"):
            continue
        target_name = _biohub_transition_projection(source_name)
        if target_name is None:
            target_name = _translate_biohub_state_key(source_name)
        for unpacked_name, unpacked_tensor in _unpack_biohub_projection(target_name, tensor):
            _store_translated_tensor(
                translated,
                unpacked_name,
                unpacked_tensor,
                source_name=source_name,
            )

    return translated


def _store_translated_tensor(
    state: dict[str, Tensor],
    name: str,
    tensor: Tensor,
    *,
    source_name: str,
) -> None:
    if name in state:
        raise RuntimeError(f"Biohub state keys collide at {name!r}; latest source is {source_name!r}")
    state[name] = tensor


def _biohub_transition_projection(name: str) -> str | None:
    match = re.fullmatch(
        r"structure_head\.diffusion_module\.conditioning\.([zs])_transitions\.(\d+)\.([ab])_proj\.weight",
        name,
    )
    if match is None:
        return None
    representation, index, projection = match.groups()
    track = "pair" if representation == "z" else "single"
    branch = "gate" if projection == "a" else "value"
    return (
        "diffusion_structure_head.denoiser.conditioning."
        f"{track}_transitions.{index}.feed_forward.{branch}_projection.weight"
    )


def _unpack_biohub_projection(name: str, tensor: Tensor) -> tuple[tuple[str, Tensor], ...]:
    packed_projections = (
        ("query_key_value_projection", ("query_projection", "key_projection", "value_projection")),
        ("key_value_projection", ("key_projection", "value_projection")),
        ("gate_value_projection", ("gate_projection", "value_projection")),
        (
            "value_gate_projection",
            ("left_value_projection", "right_value_projection", "left_gate_projection", "right_gate_projection"),
        ),
        ("left_right_projection", ("left_projection", "right_projection")),
    )
    for packed_name, projection_names in packed_projections:
        marker = f".{packed_name}."
        if marker not in name:
            continue
        if tensor.shape[0] % len(projection_names):
            raise RuntimeError(f"Biohub projection {name!r} cannot be split into {len(projection_names)} equal tensors")
        prefix, suffix = name.rsplit(marker, 1)
        return tuple(
            (f"{prefix}.{projection_name}.{suffix}", chunk)
            for projection_name, chunk in zip(projection_names, tensor.chunk(len(projection_names), dim=0))
        )
    return ((name, tensor),)


def _translate_biohub_state_key(name: str) -> str:
    diffusion_transformer_prefix = "structure_head.diffusion_module.token_transformer."
    if name.startswith(f"{diffusion_transformer_prefix}attn_blocks."):
        block, suffix = name.removeprefix(f"{diffusion_transformer_prefix}attn_blocks.").split(".", 1)
        name = f"diffusion_structure_head.denoiser.token_transformer.blocks.{block}.attention.{suffix}"
    elif name.startswith(f"{diffusion_transformer_prefix}transition_blocks."):
        block, suffix = name.removeprefix(f"{diffusion_transformer_prefix}transition_blocks.").split(".", 1)
        name = f"diffusion_structure_head.denoiser.token_transformer.blocks.{block}.transition.{suffix}"

    for upstream_prefix, canonical_prefix in _BIOHUB_STATE_PREFIXES:
        if name.startswith(upstream_prefix):
            name = f"{canonical_prefix}{name.removeprefix(upstream_prefix)}"
            break

    root_replacements = (
        ("parcae_log_a", "parcae_log_decay"),
        ("parcae_b_cont", "parcae_input_matrix"),
        ("z_init_1.", "left_pair_projection."),
        ("z_init_2.", "right_pair_projection."),
        ("token_bonds.", "bond_projection."),
        ("pair_loop_proj.0.", "recycling_norm."),
        ("pair_loop_proj.1.", "recycling_projection."),
    )
    for source, target in root_replacements:
        if name.startswith(source):
            name = f"{target}{name.removeprefix(source)}"
            break

    language_model_replacements = (
        ("language_model_pair_encoder.base_z_combine", "language_model_pair_encoder.layer_weights"),
        ("language_model_pair_encoder.base_z_linear.0.", "language_model_pair_encoder.hidden_norm."),
        ("language_model_pair_encoder.base_z_linear.1.", "language_model_pair_encoder.hidden_projection."),
        (
            "language_model_pair_encoder.base_z_mlp.0.downproject.",
            "language_model_pair_encoder.single_to_pair.single_projection.",
        ),
        (
            "language_model_pair_encoder.base_z_mlp.0.output_mlp.0.",
            "language_model_pair_encoder.single_to_pair.pair_input_projection.",
        ),
        (
            "language_model_pair_encoder.base_z_mlp.0.output_mlp.2.",
            "language_model_pair_encoder.single_to_pair.pair_output_projection.",
        ),
        ("language_model_pair_encoder.base_z_mlp.1.", "language_model_pair_encoder.output_norm."),
        ("relative_position_embedder.embed.", "relative_position_embedder.projection."),
    )
    for source, target in language_model_replacements:
        name = name.replace(source, target)

    structural_replacements = (
        (".atom_attention_encoder.", ".atom_encoder."),
        (".atom_linear.", ".atom_projection."),
        (".atom_norm.", ".input_norm."),
        (".atom_to_token_linear.", ".token_projection."),
        (".token_to_atom_linear.", ".token_projection."),
        (".atom_transformer.", ".transformer."),
        (".coords_linear.", ".coordinate_projection."),
        (".atom_decoder.norm.", ".atom_decoder.output_norm."),
        (".atom_decoder.output_linear.", ".atom_decoder.coordinate_projection."),
        (".adaln_modulation.1.", ".adaptive_norm_projection."),
        (".attn.Wqkv.", ".attention.query_key_value_projection."),
        (".attn.gate_proj.", ".attention.gate_projection."),
        (".attn.out_proj.", ".attention.output_projection."),
        (".ffn.w_up.", ".transition.gate_value_projection."),
        (".ffn.w_down.", ".transition.output_projection."),
        (".tri_mul_out._engine.", ".triangle_multiplication_outgoing."),
        (".tri_mul_in._engine.", ".triangle_multiplication_incoming."),
        (".norm_start.", ".input_norm."),
        (".proj_bundle.", ".value_gate_projection."),
        (".norm_mix.", ".output_norm."),
        (".proj_emit.", ".output_projection."),
        (".proj_gate.", ".output_gate_projection."),
        (".outer_product_mean.norm.", ".outer_product_mean.msa_norm."),
        (".outer_product_mean.Wout.", ".outer_product_mean.output_projection."),
        (".outer_product_mean.W.", ".outer_product_mean.left_right_projection."),
        (".msa_pair_weighted_averaging.norm_single.", ".msa_pair_weighted_averaging.msa_norm."),
        (".msa_pair_weighted_averaging.compute_bias.0.", ".msa_pair_weighted_averaging.pair_norm."),
        (
            ".msa_pair_weighted_averaging.compute_bias.1.",
            ".msa_pair_weighted_averaging.pair_bias_projection.",
        ),
        (".msa_pair_weighted_averaging.Wv.", ".msa_pair_weighted_averaging.value_projection."),
        (".msa_pair_weighted_averaging.Wgate.", ".msa_pair_weighted_averaging.gate_projection."),
        (".msa_pair_weighted_averaging.Wout.", ".msa_pair_weighted_averaging.output_projection."),
    )
    for source, target in structural_replacements:
        name = name.replace(source, target)

    if name.startswith("msa_encoder.embed."):
        name = f"msa_encoder.msa_projection.{name.removeprefix('msa_encoder.embed.')}"
    elif name.startswith("msa_encoder.project_inputs."):
        name = f"msa_encoder.input_projection.{name.removeprefix('msa_encoder.project_inputs.')}"

    name = re.sub(
        r"^(folding_trunk|language_model_encoder|parcae_coda|confidence_head\.folding_trunk)"
        r"\.blocks\.(\d+)\.pair_transition\.",
        r"\1.blocks.\2.transition.",
        name,
    )
    name = re.sub(
        r"\.(transition|pair_transition|msa_transition)\.norm\.",
        r".\1.input_norm.",
        name,
    )
    name = re.sub(
        r"\.(transition|pair_transition|msa_transition)\.ffn\.w12\.",
        r".\1.feed_forward.gate_value_projection.",
        name,
    )
    name = re.sub(
        r"\.(transition|pair_transition|msa_transition)\.ffn\.w3\.",
        r".\1.feed_forward.output_projection.",
        name,
    )

    diffusion_replacements = (
        (".conditioning.z_input_norm.", ".conditioning.pair_input_norm."),
        (".conditioning.z_proj.", ".conditioning.pair_projection."),
        (".conditioning.z_transitions.", ".conditioning.pair_transitions."),
        (".conditioning.s_input_norm.", ".conditioning.single_input_norm."),
        (".conditioning.s_proj.", ".conditioning.single_projection."),
        (".conditioning.s_transitions.", ".conditioning.single_transitions."),
        (".conditioning.fourier.w", ".conditioning.noise_embedding.frequencies"),
        (".conditioning.fourier.b", ".conditioning.noise_embedding.phases"),
        (".conditioning.noise_proj.", ".conditioning.noise_projection."),
        (".diffusion_module.s_step_norm.", ".diffusion_module.single_norm."),
        (".diffusion_module.s_to_token.", ".diffusion_module.single_projection."),
        (".attention.adaln.", ".attention.adaptive_norm."),
        (".transition.adaln.", ".transition.adaptive_norm."),
        (".adaptive_norm.s_scale", ".adaptive_norm.conditioning_norm.weight"),
        (".adaptive_norm.s_gate.", ".adaptive_norm.gate_projection."),
        (".adaptive_norm.s_shift.", ".adaptive_norm.shift_projection."),
        (".attention.out_gate.", ".attention.output_gate_projection."),
        (".attention.q_proj.", ".attention.query_projection."),
        (".attention.kv_proj.", ".attention.key_value_projection."),
        (".attention.g_proj.", ".attention.gate_projection."),
        (".attention.out_proj.", ".attention.output_projection."),
        (".attention.pair_bias_proj.", ".attention.pair_bias_projection."),
        (".transition.lin_swish.", ".transition.feed_forward.gate_value_projection."),
        (".transition.lin_out.", ".transition.feed_forward.output_projection."),
        (".transition.output_gate.", ".transition.output_gate_projection."),
    )
    for source, target in diffusion_replacements:
        name = name.replace(source, target)
    name = re.sub(
        r"\.(pair_transitions|single_transitions)\.(\d+)\.norm\.",
        r".\1.\2.input_norm.",
        name,
    )
    name = re.sub(
        r"\.(pair_transitions|single_transitions)\.(\d+)\.out_proj\.",
        r".\1.\2.feed_forward.output_projection.",
        name,
    )

    confidence_replacements = (
        ("confidence_head.boundaries", "confidence_head.distance_boundaries"),
        ("confidence_head.dist_bin_pairwise_embed.", "confidence_head.distance_embedding."),
        ("confidence_head.s_inputs_norm.", "confidence_head.input_norm."),
        ("confidence_head.z_norm.", "confidence_head.pair_norm."),
        ("confidence_head.s_to_z_prod_in1.", "confidence_head.left_product_projection."),
        ("confidence_head.s_to_z_prod_in2.", "confidence_head.right_product_projection."),
        ("confidence_head.s_to_z_prod_out.", "confidence_head.product_output_projection."),
        ("confidence_head.s_to_z_transpose.", "confidence_head.right_pair_projection."),
        ("confidence_head.s_to_z.", "confidence_head.left_pair_projection."),
        ("confidence_head.s_norm.", "confidence_head.single_norm."),
        ("confidence_head.s_inputs_to_single.", "confidence_head.input_to_single_projection."),
        ("confidence_head.s_input_to_s.", "confidence_head.input_skip_projection."),
        ("confidence_head.plddt_ln.", "confidence_head.plddt_norm."),
        ("confidence_head.plddt_weight", "confidence_head.plddt_projection_weight"),
        ("confidence_head.pae_ln.", "confidence_head.pae_norm."),
        ("confidence_head.pae_head.", "confidence_head.pae_projection."),
        ("confidence_head.pde_ln.", "confidence_head.pde_norm."),
        ("confidence_head.pde_head.", "confidence_head.pde_projection."),
        ("confidence_head.resolved_ln.", "confidence_head.resolved_norm."),
        ("confidence_head.resolved_weight", "confidence_head.resolved_projection_weight"),
        (".row_attention_pooling.attn_proj.", ".row_attention_pooling.attention_projection."),
        (".row_attention_pooling.out_proj.", ".row_attention_pooling.output_projection."),
    )
    for source, target in confidence_replacements:
        name = name.replace(source, target)
    name = name.replace("diffusion_structure_head.diffusion_module.", "diffusion_structure_head.denoiser.", 1)
    return name


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
        "recycling_steps": int(raw_config["num_loops"]) + 1,
        "language_model_mask_probability": language_model_mask_probability,
        "msa_encoder_overwrite": raw_config["msa_encoder_overwrite"],
        "input_embeddings": {
            "input_size": inputs["d_inputs"],
            "atom_encoder": {
                "atom_hidden_size": atom_encoder["d_atom"],
                "token_hidden_size": atom_encoder["d_token"],
                "num_blocks": atom_encoder["n_blocks"],
                "num_attention_heads": atom_encoder["n_heads"],
                "sliding_window_size": atom_encoder["swa_window_size"],
                "expansion": atom_encoder["expansion_ratio"],
                "spatial_rope_base_frequency": atom_encoder["spatial_rope_base_frequency"],
                "num_spatial_rope_pairs_per_axis": atom_encoder["n_spatial_rope_pairs_per_axis"],
                "num_reference_space_rope_pairs": atom_encoder["n_uid_rope_pairs"],
                "reference_space_rope_base_frequency": atom_encoder["uid_rope_base_frequency"],
            },
        },
        # n_heads/dropout are inert upstream dataclass leftovers: the released
        # upstream PairUpdateBlock has neither an attention nor a dropout operation.
        "folding_trunk": {"num_blocks": folding_trunk["n_layers"]},
        "diffusion_structure_head": {
            "denoiser": {
                "sigma_data": diffusion["sigma_data"],
                "atom_hidden_size": diffusion["c_atom"],
                "token_hidden_size": diffusion["c_token"],
                "pair_size": diffusion["c_z"],
                "input_size": diffusion["c_s_inputs"],
                "fourier_embedding_size": diffusion["fourier_dim"],
                "num_atom_blocks": diffusion["atom_num_blocks"],
                "num_atom_attention_heads": diffusion["atom_num_heads"],
                "num_token_blocks": diffusion["token_num_blocks"],
                "num_token_attention_heads": diffusion["token_num_heads"],
                "expansion": diffusion["transition_multiplier"],
            },
            "num_distogram_bins": structure["distogram_bins"],
            "churn_rate": structure["gamma_0"],
            "min_churn_noise_level": structure["gamma_min"],
            "noise_scale": structure["noise_scale"],
            "step_scale": structure["step_scale"],
            "max_noise_level": structure["inference_s_max"],
            "min_noise_level": structure["inference_s_min"],
            "rho": structure["inference_p"],
            "num_sampling_steps": structure["inference_num_steps"],
        },
        "confidence": {
            "num_plddt_bins": confidence["num_plddt_bins"],
            "num_pde_bins": confidence["num_pde_bins"],
            "num_pae_bins": confidence["num_pae_bins"],
            "min_distance": confidence["min_dist"],
            "max_distance": confidence["max_dist"],
            "num_distogram_bins": confidence["distogram_bins"],
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
        # Model; recycling_steps stores the total updates and always builds the coda.
        "parcae": {"num_coda_blocks": parcae["coda_n_layers"]},
        "language_model_encoder": {
            "enabled": lm_encoder["enabled"],
            "num_blocks": lm_encoder["n_layers"],
            "dropout": lm_encoder["lm_dropout"],
            "per_loop_dropout": lm_encoder["per_loop_lm_dropout"],
        },
    }


def experimental_config_from_biohub(raw_config: Mapping[str, Any]) -> EsmFold2ExperimentalConfig:
    r"""Build an [`EsmFold2ExperimentalConfig`][ofoldx.models.esmfold2.EsmFold2ExperimentalConfig]
    from a ``type="experimental"`` recurrent-graph checkpoint ``config.json``.

    Every graph dimension is read from the checkpoint (the Fast inversion model ships 24
    trunk blocks with the MSA encoder disabled; the MSA-conditioned critic ships 48 blocks);
    nothing is guessed.
    """

    if raw_config.get("model_type") != "esmfold2" or raw_config.get("type") != "experimental":
        raise ValueError("checkpoint config is not the experimental Biohub ESMFold2 architecture")

    config = EsmFold2ExperimentalConfig()
    config.single_size = raw_config["d_single"]
    config.pair_size = raw_config["d_pair"]
    config.num_relative_residue_bins = raw_config["n_relative_residx_bins"]
    config.num_relative_chain_bins = raw_config["n_relative_chain_bins"]
    config.recycling_steps = int(raw_config["num_loops"]) + 1
    config.use_msa_features = not raw_config["disable_msa_features"]
    config.language_model_dropout = raw_config.get("lm_dropout", 0.0)
    config.force_language_model_dropout = raw_config.get("force_lm_dropout_during_inference", False)

    # ESMC trunk width/depth the folding head projects from
    # (``EsmFold2LanguageModelPairEncoder``
    # base_z_linear/base_z_combine): read from the checkpoint so the smaller scaling-study
    # trunks strict-load. The 6B checkpoints ship 2560/80 (the defaults); the 300M/600M
    # scaling critics ship 960/30 and 1152/36.
    config.esmc.hidden_size = raw_config.get("lm_d_model", config.esmc.hidden_size)
    config.esmc.num_hidden_layers = raw_config.get("lm_num_layers", config.esmc.num_hidden_layers)

    inputs = raw_config["inputs"]
    atom_encoder = inputs["atom_encoder"]
    config.input_embeddings.input_size = inputs["d_inputs"]
    atom_encoder_config = config.input_embeddings.atom_encoder
    atom_encoder_config.atom_hidden_size = atom_encoder["d_atom"]
    atom_encoder_config.token_hidden_size = atom_encoder["d_token"]
    atom_encoder_config.num_blocks = atom_encoder["n_blocks"]
    atom_encoder_config.num_attention_heads = atom_encoder["n_heads"]
    atom_encoder_config.sliding_window_size = atom_encoder["swa_window_size"]
    atom_encoder_config.expansion = atom_encoder["expansion_ratio"]
    atom_encoder_config.spatial_rope_base_frequency = atom_encoder["spatial_rope_base_frequency"]
    atom_encoder_config.num_spatial_rope_pairs_per_axis = atom_encoder["n_spatial_rope_pairs_per_axis"]
    atom_encoder_config.num_reference_space_rope_pairs = atom_encoder["n_uid_rope_pairs"]
    atom_encoder_config.reference_space_rope_base_frequency = atom_encoder["uid_rope_base_frequency"]

    config.folding_trunk.num_blocks = raw_config["folding_trunk"]["n_layers"]

    structure = raw_config["structure_head"]
    diffusion = structure["diffusion_module"]
    structure_head_config = config.diffusion_structure_head
    structure_head_config.num_distogram_bins = structure["distogram_bins"]
    structure_head_config.churn_rate = structure["gamma_0"]
    structure_head_config.min_churn_noise_level = structure["gamma_min"]
    structure_head_config.noise_scale = structure["noise_scale"]
    structure_head_config.step_scale = structure["step_scale"]
    structure_head_config.max_noise_level = structure["inference_s_max"]
    structure_head_config.min_noise_level = structure["inference_s_min"]
    structure_head_config.rho = structure["inference_p"]
    structure_head_config.num_sampling_steps = structure["inference_num_steps"]
    denoiser_config = structure_head_config.denoiser
    denoiser_config.sigma_data = diffusion["sigma_data"]
    denoiser_config.atom_hidden_size = diffusion["c_atom"]
    denoiser_config.token_hidden_size = diffusion["c_token"]
    denoiser_config.pair_size = diffusion["c_z"]
    denoiser_config.input_size = diffusion["c_s_inputs"]
    denoiser_config.fourier_embedding_size = diffusion["fourier_dim"]
    denoiser_config.num_atom_blocks = diffusion["atom_num_blocks"]
    denoiser_config.num_atom_attention_heads = diffusion["atom_num_heads"]
    denoiser_config.num_token_blocks = diffusion["token_num_blocks"]
    denoiser_config.num_token_attention_heads = diffusion["token_num_heads"]
    denoiser_config.expansion = diffusion["transition_multiplier"]

    confidence = raw_config["confidence_head"]
    confidence_config = config.confidence
    confidence_config.enabled = confidence.get("enabled", True)
    confidence_config.num_distogram_bins = confidence["distogram_bins"]
    confidence_config.min_distance = confidence["min_dist"]
    confidence_config.max_distance = confidence["max_dist"]
    confidence_config.num_plddt_bins = confidence["num_plddt_bins"]
    confidence_config.num_pae_bins = confidence["num_pae_bins"]
    confidence_config.folding_trunk.num_blocks = confidence["folding_trunk"]["n_layers"]

    msa = raw_config["msa_encoder"]
    msa_config = config.msa_encoder
    msa_config.enabled = msa["enabled"]
    msa_config.msa_size = msa["d_msa"]
    msa_config.hidden_size = msa["d_hidden"]
    msa_config.num_blocks = msa["n_layers"]
    msa_config.num_msa_attention_heads = msa["n_heads_msa"]
    msa_config.msa_head_dim = msa["msa_head_width"]

    return config


def load_experimental_esmfold2(
    checkpoint_or_repo: str | PathLike[str],
    *,
    esmc: EsmCModel | EsmCForMaskedLM | None = None,
    config_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> EsmFold2ExperimentalModel:
    r"""Strict-load a ``type="experimental"`` recurrent ESMFold2 head into an OFoldX model.

    The MultiMolecule ESMC child is loaded separately (the head safetensors does not carry
    it): the head strict-loads first with ``esmc=None`` and the optional ``esmc`` is attached
    afterwards via [`load_esmc`][ofoldx.models.esmfold2.EsmFold2ExperimentalModel.load_esmc].
    """

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    config = experimental_config_from_biohub(raw_config)
    model = EsmFold2ExperimentalModel(config, esmc=None)
    head_state = load_esmfold2_state_dict(checkpoint_path)
    model.load_state_dict(head_state, strict=True)
    if esmc is not None:
        model.load_esmc(esmc)
    return model


def save_experimental_esmfold2_checkpoint(
    checkpoint_or_repo: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    esmc_source: str | PathLike[str] = _DEFAULT_ESMC_REPOSITORY,
    config_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert an experimental (binder-inversion) ESMFold2 head and write a routable repository.

    Strict-loads the ``type="experimental"`` recurrent head via
    [`load_experimental_esmfold2`][ofoldx.models.esmfold2.convert.load_experimental_esmfold2] with
    ``esmc=None`` so the root state dict remains exactly the experimental head. The selected
    MultiMolecule ESMC masked-LM model and tokenizer are attached to the concrete
    [`EsmFold2DesignPipeline`][], whose save contract persists them under ``esmc/`` and
    writes the ``ofoldx.esmfold2_design`` marker. The resulting repository is self-contained
    and routes ``pipeline("design", model=dir)`` to [`EsmFold2DesignPipeline.design`][].

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit ``repo_id``.
    """

    from .pipeline import EsmFold2DesignPipeline

    esmc = cast(
        EsmCForMaskedLM,
        AutoModelForMaskedLM.from_pretrained(
            esmc_source,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            dtype=dtype,
            token=token,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        esmc_source,
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
        token=token,
    )
    model = load_experimental_esmfold2(
        checkpoint_or_repo,
        esmc=None,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    if dtype is not None:
        model.to(dtype=dtype)
    model.load_esmc(esmc)
    processor = EsmFold2ExperimentalProcessor(
        EsmFold2ExperimentalProcessorConfig({"msa_conditioning": model.config.use_msa_features})
    )
    processor.tokenizer = cast(ProteinTokenizer, tokenizer)
    EsmFold2DesignPipeline(model=model, processor=processor).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


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
    parser.add_argument("--token", default=None, help="optional Hugging Face token for source access and --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="convert the type='experimental' recurrent binder-inversion head (design pipeline); "
        "persists the selected ESMC model and tokenizer with the head",
    )
    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    if args.experimental:
        save_experimental_esmfold2_checkpoint(
            args.checkpoint_or_repo,
            args.output_dir,
            esmc_source=args.esmc,
            config_path=args.config,
            dtype=dtype,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
        return
    convert_esmfold2_checkpoint(
        args.checkpoint_or_repo,
        args.output_dir,
        esmc_source=args.esmc,
        config_path=args.config,
        dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
