r"""
Convert Biohub ESMFold2 and MultiMolecule ESMC into one OFoldX repository.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from multimolecule import AutoModel, EsmCConfig, EsmCForMaskedLM, EsmCModel
from safetensors.torch import load_file
from torch import Tensor

from ...data.evidence import MSA_ALPHABET, MSA_VOCAB_SIZE
from ...data.tokenization import TOKEN_ALPHABET, TOKEN_VOCAB_SIZE
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

if TYPE_CHECKING:
    from .pipeline import EsmFold2ExperimentalPipeline

_DEFAULT_ESMC_REPOSITORY = "multimolecule/esmc-6b"
_BIOHUB_ESMC_VOCAB: tuple[str | None, ...] = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",
    "-",
    "|",
    "<mask>",
    *([None] * 31),
)
_ESMC_VOCAB = (
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    *tuple("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?"),
)
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
_BIOHUB_TOKEN_ALPHABET = (
    "OTHER",
    "-",
    *tuple("ARNDCQEGHILKMFPSTWYV"),
    "X",
    *tuple(f"RNA_{symbol}" for symbol in "AGCUN"),
    *tuple(f"DNA_{symbol}" for symbol in "AGCTN"),
)
_TOKEN_FEATURE_VOCAB_EXPANSION = TOKEN_VOCAB_SIZE + MSA_VOCAB_SIZE - 2 * len(_BIOHUB_TOKEN_ALPHABET)


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
    r"""
    Write a self-contained ``model.json``/``processor.json``/safetensors repository.

    Local-only by default. When ``repo_id`` is given, the converted repository is also
    uploaded to that Hub repo. The repository is marked with the exact
    [`EsmFold2Pipeline`][]; no organization or token is assumed and nothing is
    published without an explicit ``repo_id``.
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

    from .pipeline import EsmFold2Pipeline

    EsmFold2Pipeline(
        model=model,
        processor=EsmFold2Processor(
            EsmFold2ProcessorConfig({"msa_conditioning": config.msa_encoder.enabled}),
            dtype=dtype,
        ),
    ).save(
        output_dir,
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
    r"""
    Resolve the upstream folding-head checkpoint and its source config.
    """

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


def resolve_biohub_esmc_source(
    source: str | PathLike[str],
    *,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    r"""
    Resolve one raw Biohub ESMC ``config.json`` and all safetensor shards.
    """

    classified = classify_reference(source)
    if isinstance(classified, Path):
        root = classified
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                classified,
                allow_patterns=["config.json", "model*.safetensors"],
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=local_files_only,
                token=token,
            )
        )
    config = root / "config.json"
    checkpoints = tuple(sorted(root.glob("model*.safetensors")))
    if not config.is_file():
        raise FileNotFoundError(f"Biohub ESMC config does not exist: {config}")
    if not checkpoints:
        raise FileNotFoundError(f"Biohub ESMC safetensors do not exist under {root}")
    return config, checkpoints


def load_esmfold2_state_dict(checkpoint_path: str | PathLike[str]) -> dict[str, Tensor]:
    r"""
    Load and strictly translate one Biohub folding-head state dictionary.
    """

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

    _move_final_msa_block(translated)
    _pack_adaptive_norms(translated)
    return _convert_esmfold2_vocabulary_state_dict(translated)


def _convert_biohub_token_axis(
    tensor: Tensor,
    axis: int,
    alphabet: tuple[str, ...],
) -> Tensor:
    axis %= tensor.ndim
    source = tensor.movedim(axis, 0)
    if source.shape[0] != len(_BIOHUB_TOKEN_ALPHABET):
        raise RuntimeError(
            f"Biohub ESMFold2 token axis has {source.shape[0]} entries; " f"expected {len(_BIOHUB_TOKEN_ALPHABET)}"
        )
    source_by_symbol = {symbol: source[index] for index, symbol in enumerate(_BIOHUB_TOKEN_ALPHABET)}
    converted: list[Tensor] = []
    for symbol in alphabet:
        if symbol in source_by_symbol:
            value = source_by_symbol[symbol]
        elif symbol == "B":
            value = (source_by_symbol["D"] + source_by_symbol["N"]) / 2
        elif symbol == "Z":
            value = (source_by_symbol["E"] + source_by_symbol["Q"]) / 2
        elif symbol == "J":
            value = (source_by_symbol["I"] + source_by_symbol["L"]) / 2
        elif symbol == "U":
            value = source_by_symbol["C"]
        elif symbol == "O":
            value = source_by_symbol["K"]
        else:
            raise RuntimeError(f"Biohub ESMFold2 MSA conversion has no rule for {symbol!r}")
        converted.append(value)
    return torch.stack(converted).movedim(0, axis).contiguous()


def _convert_token_feature_axis(tensor: Tensor, axis: int) -> Tensor:
    axis %= tensor.ndim
    source = tensor.movedim(axis, 0)
    source_vocab_size = len(_BIOHUB_TOKEN_ALPHABET)
    source_tail_size = 2 * source_vocab_size + 1
    if source.shape[0] < source_tail_size:
        raise RuntimeError(f"Biohub ESMFold2 token-feature axis is too short: {source.shape[0]}")
    prefix = source[:-source_tail_size]
    residue_type = source[-source_tail_size : -source_vocab_size - 1]
    msa_profile = source[-source_vocab_size - 1 : -1]
    deletion_mean = source[-1:]
    return (
        torch.cat(
            (
                prefix,
                _convert_biohub_token_axis(residue_type, 0, TOKEN_ALPHABET),
                _convert_biohub_token_axis(msa_profile, 0, MSA_ALPHABET),
                deletion_mean,
            ),
            dim=0,
        )
        .movedim(0, axis)
        .contiguous()
    )


def _convert_esmfold2_vocabulary_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    converted = dict(state_dict)
    token_weight_keys = (
        "left_pair_projection.weight",
        "right_pair_projection.weight",
        "msa_encoder.input_projection.weight",
        "diffusion_structure_head.denoiser.conditioning.single_projection.weight",
        "confidence_head.left_pair_projection.weight",
        "confidence_head.right_pair_projection.weight",
        "confidence_head.left_product_projection.weight",
        "confidence_head.right_product_projection.weight",
    )
    token_vector_keys = (
        "diffusion_structure_head.denoiser.conditioning.single_input_norm.weight",
        "diffusion_structure_head.denoiser.conditioning.single_input_norm.bias",
        "confidence_head.input_norm.weight",
        "confidence_head.input_norm.bias",
    )
    for key in token_weight_keys:
        if key in converted:
            converted[key] = _convert_token_feature_axis(converted[key], 1)
    for key in token_vector_keys:
        if key in converted:
            converted[key] = _convert_token_feature_axis(converted[key], 0)

    msa_projection_key = "msa_encoder.msa_projection.weight"
    if msa_projection_key in converted:
        source = converted[msa_projection_key]
        expected = len(_BIOHUB_TOKEN_ALPHABET) + 2
        if source.ndim != 2 or source.shape[1] != expected:
            raise RuntimeError(
                f"Biohub ESMFold2 MSA projection has shape {tuple(source.shape)}; expected (*, {expected})"
            )
        converted[msa_projection_key] = torch.cat(
            (
                _convert_biohub_token_axis(
                    source[:, : len(_BIOHUB_TOKEN_ALPHABET)],
                    1,
                    MSA_ALPHABET,
                ),
                source[:, -2:],
            ),
            dim=1,
        ).contiguous()
    return converted


def _move_final_msa_block(state: dict[str, Tensor]) -> None:
    prefix = "msa_encoder.blocks."
    block_indices = {int(key.removeprefix(prefix).split(".", 1)[0]) for key in state if key.startswith(prefix)}
    if not block_indices:
        return
    final_prefix = f"{prefix}{max(block_indices)}."
    replacements = (
        ("outer_product_mean.", "msa_encoder.final_outer_product_mean."),
        ("triangle_multiplication_outgoing.", "msa_encoder.final_pair_update.triangle_multiplication_outgoing."),
        ("triangle_multiplication_incoming.", "msa_encoder.final_pair_update.triangle_multiplication_incoming."),
        ("pair_transition.", "msa_encoder.final_pair_update.transition."),
    )
    for key in tuple(state):
        if not key.startswith(final_prefix):
            continue
        suffix = key.removeprefix(final_prefix)
        for source, target in replacements:
            if suffix.startswith(source):
                state[f"{target}{suffix.removeprefix(source)}"] = state.pop(key)
                break


def _pack_adaptive_norms(state: dict[str, Tensor]) -> None:
    scale_weight_suffix = ".gate_projection.weight"
    for scale_weight_key in tuple(
        key
        for key in state
        if key.endswith(scale_weight_suffix) and (".adaptive_norm." in key or ".conditioned_norm." in key)
    ):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state.pop(scale_weight_key)
        scale_bias = state.pop(prefix + "gate_projection.bias")
        shift_weight = state.pop(prefix + "shift_projection.weight")
        state[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state[prefix + "conditioning_scale_bias"] = scale_bias


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


def load_biohub_esmc(
    source: str | PathLike[str],
    *,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> EsmCForMaskedLM:
    r"""
    Strictly convert one raw Biohub ESMC masked-language model.
    """

    config_path, checkpoint_paths = resolve_biohub_esmc_source(
        source,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = _esmc_config_from_biohub(raw_config)
    with torch.device("meta"):
        model = EsmCForMaskedLM(config)

    state: dict[str, Tensor] = {}
    source_names: set[str] = set()
    for checkpoint_path in checkpoint_paths:
        for source_name, tensor in load_file(str(checkpoint_path), device="cpu").items():
            if source_name in source_names:
                raise RuntimeError(f"Biohub ESMC state key occurs in multiple shards: {source_name!r}")
            source_names.add(source_name)
            if source_name.endswith("._extra_state"):
                continue
            _translate_biohub_esmc_tensor(config, state, source_name, tensor)

    _convert_esmc_token_rows(config, state)
    model.load_state_dict(state, strict=True, assign=True)
    return model


def _esmc_config_from_biohub(raw_config: Mapping[str, Any]) -> EsmCConfig:
    if raw_config.get("model_type") != "esmc":
        raise ValueError("checkpoint config is not a raw Biohub ESMC architecture")
    if int(raw_config["vocab_size"]) != len(_BIOHUB_ESMC_VOCAB):
        raise ValueError(f"Biohub ESMC vocabulary must have {len(_BIOHUB_ESMC_VOCAB)} rows")
    if int(raw_config["pad_token_id"]) != _BIOHUB_ESMC_VOCAB.index("<pad>"):
        raise ValueError("Biohub ESMC pad_token_id does not match its released vocabulary")
    if int(raw_config["mask_token_id"]) != _BIOHUB_ESMC_VOCAB.index("<mask>"):
        raise ValueError("Biohub ESMC mask_token_id does not match its released vocabulary")
    if raw_config.get("tie_word_embeddings", False):
        raise ValueError("Biohub ESMC input and output embeddings must be independent")

    return EsmCConfig(
        vocab_size=len(_ESMC_VOCAB),
        hidden_size=int(raw_config.get("hidden_size", raw_config.get("d_model", 960))),
        num_hidden_layers=int(raw_config.get("num_hidden_layers", raw_config.get("n_layers", 30))),
        num_attention_heads=int(raw_config.get("num_attention_heads", raw_config.get("n_heads", 15))),
        intermediate_size=raw_config.get("intermediate_size"),
        hidden_act=str(raw_config.get("hidden_act", "swiglu")),
        hidden_dropout=float(raw_config.get("hidden_dropout", 0.0)),
        attention_dropout=float(raw_config.get("attention_dropout", 0.0)),
        max_position_embeddings=int(raw_config.get("max_position_embeddings", 2048)),
        initializer_range=float(raw_config.get("initializer_range", 0.02)),
        layer_norm_eps=float(raw_config.get("layer_norm_eps", 1e-5)),
        attention_bias=bool(raw_config.get("attention_bias", False)),
        feedforward_bias=bool(raw_config.get("feedforward_bias", False)),
        attention_layer_norm_bias=bool(raw_config.get("attention_layer_norm_bias", True)),
        qk_layer_norm=bool(raw_config.get("qk_layer_norm", True)),
        qk_layer_norm_bias=bool(raw_config.get("qk_layer_norm_bias", False)),
        final_layer_norm_bias=bool(raw_config.get("final_layer_norm_bias", False)),
        residue_scaling_factor=raw_config.get("residue_scaling_factor"),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        mask_token_id=4,
        null_token_id=5,
        tie_word_embeddings=False,
    )


def _translate_biohub_esmc_tensor(
    config: EsmCConfig,
    state: dict[str, Tensor],
    source_name: str,
    tensor: Tensor,
) -> None:
    if source_name == "esmc.embed.weight":
        _store_translated_tensor(
            state,
            "model.embeddings.word_embeddings.weight",
            tensor,
            source_name=source_name,
        )
        return
    if source_name == "esmc.transformer.norm.weight":
        _store_translated_tensor(
            state,
            "model.encoder.layer_norm.weight",
            tensor,
            source_name=source_name,
        )
        return
    if source_name.startswith("esmc.sequence_head.") or source_name.startswith("lm_head."):
        suffix = source_name.removeprefix("esmc.sequence_head.").removeprefix("lm_head.")
        head_names = {
            "0.weight": "lm_head.transform.dense.weight",
            "0.bias": "lm_head.transform.dense.bias",
            "2.weight": "lm_head.transform.layer_norm.weight",
            "2.bias": "lm_head.transform.layer_norm.bias",
            "3.weight": "lm_head.decoder.weight",
            "3.bias": "lm_head.decoder.bias",
        }
        if suffix not in head_names:
            raise KeyError(f"Unhandled Biohub ESMC language-model head key: {source_name}")
        _store_translated_tensor(state, head_names[suffix], tensor, source_name=source_name)
        return

    match = re.fullmatch(r"esmc\.transformer\.blocks\.(\d+)\.(.+)", source_name)
    if match is None:
        raise KeyError(f"Unhandled Biohub ESMC checkpoint key: {source_name}")
    layer_index, suffix = match.groups()
    prefix = f"model.encoder.layer.{layer_index}"
    names = {
        "attn.layernorm_qkv.layer_norm_weight": f"{prefix}.attention.layer_norm.weight",
        "attn.layernorm_qkv.layer_norm_bias": f"{prefix}.attention.layer_norm.bias",
        "attn.q_ln.weight": f"{prefix}.attention.self.query_layer_norm.weight",
        "attn.k_ln.weight": f"{prefix}.attention.self.key_layer_norm.weight",
        "attn.out_proj.weight": f"{prefix}.attention.output.dense.weight",
        "ffn.layer_norm_weight": f"{prefix}.layer_norm.weight",
        "ffn.layer_norm_bias": f"{prefix}.layer_norm.bias",
        "ffn.fc2_weight": f"{prefix}.output.dense.weight",
    }
    if suffix in names:
        _store_translated_tensor(state, names[suffix], tensor, source_name=source_name)
        return
    if suffix == "attn.layernorm_qkv.weight":
        if tensor.shape[0] != 3 * config.hidden_size:
            raise RuntimeError(
                f"Biohub ESMC QKV tensor {source_name!r} has shape {tuple(tensor.shape)}; "
                f"expected first dimension {3 * config.hidden_size}"
            )
        for projection, chunk in zip(
            ("query", "key", "value"),
            tensor.chunk(3, dim=0),
        ):
            _store_translated_tensor(
                state,
                f"{prefix}.attention.self.{projection}.weight",
                chunk.contiguous(),
                source_name=source_name,
            )
        return
    if suffix == "ffn.fc1_weight":
        if tensor.shape[0] != 2 * config.intermediate_size:
            raise RuntimeError(
                f"Biohub ESMC SwiGLU tensor {source_name!r} has shape {tuple(tensor.shape)}; "
                f"expected first dimension {2 * config.intermediate_size}"
            )
        for projection, chunk in zip(
            ("gate_proj", "up_proj"),
            tensor.chunk(2, dim=0),
        ):
            _store_translated_tensor(
                state,
                f"{prefix}.intermediate.{projection}.weight",
                chunk.contiguous(),
                source_name=source_name,
            )
        return
    raise KeyError(f"Unhandled Biohub ESMC encoder key: {source_name}")


def _convert_esmc_token_rows(config: EsmCConfig, state: dict[str, Tensor]) -> None:
    source_word_embeddings = state["model.embeddings.word_embeddings.weight"]
    source_decoder_weight = state["lm_head.decoder.weight"]
    source_decoder_bias = state["lm_head.decoder.bias"]
    for name, tensor in (
        ("embedding", source_word_embeddings),
        ("decoder weight", source_decoder_weight),
        ("decoder bias", source_decoder_bias),
    ):
        if tensor.shape[0] != len(_BIOHUB_ESMC_VOCAB):
            raise RuntimeError(f"Biohub ESMC {name} has {tensor.shape[0]} rows; expected {len(_BIOHUB_ESMC_VOCAB)}")

    generator = torch.Generator(device="cpu").manual_seed(1016)
    word_embeddings = torch.normal(
        mean=0.0,
        std=config.initializer_range,
        size=(len(_ESMC_VOCAB), *source_word_embeddings.shape[1:]),
        generator=generator,
        dtype=source_word_embeddings.dtype,
        device="cpu",
    )
    decoder_weight = torch.normal(
        mean=0.0,
        std=config.initializer_range,
        size=(len(_ESMC_VOCAB), *source_decoder_weight.shape[1:]),
        generator=generator,
        dtype=source_decoder_weight.dtype,
        device="cpu",
    )
    decoder_bias = torch.zeros(
        (len(_ESMC_VOCAB), *source_decoder_bias.shape[1:]),
        dtype=source_decoder_bias.dtype,
        device="cpu",
    )
    target_indices = {token: index for index, token in enumerate(_ESMC_VOCAB)}
    for source_index, token in enumerate(_BIOHUB_ESMC_VOCAB):
        if token is None:
            continue
        target_index = target_indices[token]
        word_embeddings[target_index] = source_word_embeddings[source_index]
        decoder_weight[target_index] = source_decoder_weight[source_index]
        decoder_bias[target_index] = source_decoder_bias[source_index]

    j_index = target_indices["J"]
    il_indices = [target_indices["I"], target_indices["L"]]
    word_embeddings[j_index] = word_embeddings[il_indices].mean(dim=0)
    decoder_weight[j_index] = decoder_weight[il_indices].mean(dim=0)
    decoder_bias[j_index] = decoder_bias[il_indices].mean(dim=0)
    state["model.embeddings.word_embeddings.weight"] = word_embeddings
    state["lm_head.decoder.weight"] = decoder_weight
    state["lm_head.decoder.bias"] = decoder_bias
    state["lm_head.bias"] = decoder_bias


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
        name = f"diffusion_structure_head.denoiser.token_encoder.blocks.{block}.attention.{suffix}"
    elif name.startswith(f"{diffusion_transformer_prefix}transition_blocks."):
        block, suffix = name.removeprefix(f"{diffusion_transformer_prefix}transition_blocks.").split(".", 1)
        name = f"diffusion_structure_head.denoiser.token_encoder.blocks.{block}.transition.{suffix}"

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
        (".atom_transformer.", ".encoder."),
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
            ".msa_pair_weighted_averaging.pair_projection.",
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

    if name.startswith("folding_trunk."):
        name = f"folding_stack.{name.removeprefix('folding_trunk.')}"
    elif name.startswith("confidence_head.folding_trunk."):
        name = f"confidence_head.folding_stack.{name.removeprefix('confidence_head.folding_trunk.')}"
    name = re.sub(
        r"^(folding_stack|language_model_encoder|parcae_coda|confidence_head\.folding_stack)"
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
        (".transition.adaln.", ".transition.conditioned_norm."),
        (".adaptive_norm.s_scale", ".adaptive_norm.conditioning_norm.weight"),
        (".adaptive_norm.s_gate.", ".adaptive_norm.gate_projection."),
        (".adaptive_norm.s_shift.", ".adaptive_norm.shift_projection."),
        (".conditioned_norm.s_scale", ".conditioned_norm.conditioning_norm.weight"),
        (".conditioned_norm.s_gate.", ".conditioned_norm.gate_projection."),
        (".conditioned_norm.s_shift.", ".conditioned_norm.shift_projection."),
        (".attention.out_gate.", ".attention.output_gate_projection."),
        (".attention.q_proj.", ".attention.query_projection."),
        (".attention.kv_proj.", ".attention.key_value_projection."),
        (".attention.g_proj.", ".attention.gate_projection."),
        (".attention.out_proj.", ".attention.output_projection."),
        (".attention.pair_bias_proj.", ".attention.pair_bias_projection."),
        (".transition.lin_swish.", ".transition.feed_forward.gate_value_projection."),
        (".transition.lin_out.", ".transition.feed_forward.output_projection."),
        (".transition.output_gate.", ".transition.conditioning_gate_projection."),
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
            "input_size": inputs["d_inputs"] + _TOKEN_FEATURE_VOCAB_EXPANSION,
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
        "folding_stack": {"num_blocks": folding_trunk["n_layers"]},
        "diffusion_structure_head": {
            "denoiser": {
                "sigma_data": diffusion["sigma_data"],
                "atom_hidden_size": diffusion["c_atom"],
                "token_hidden_size": diffusion["c_token"],
                "pair_size": diffusion["c_z"],
                "input_size": diffusion["c_s_inputs"] + _TOKEN_FEATURE_VOCAB_EXPANSION,
                "fourier_embedding_size": diffusion["fourier_dim"],
                "num_atom_blocks": diffusion["atom_num_blocks"],
                "num_atom_attention_heads": diffusion["atom_num_heads"],
                "num_token_encoder_blocks": diffusion["token_num_blocks"],
                "num_token_encoder_attention_heads": diffusion["token_num_heads"],
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
            "folding_stack": {"num_blocks": confidence_trunk["n_layers"]},
        },
        "msa_encoder": {
            "enabled": msa["enabled"],
            "msa_size": msa["d_msa"],
            "hidden_size": msa["d_hidden"],
            "num_stages": msa["n_layers"],
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
    r"""
    Build an [`EsmFold2ExperimentalConfig`][ofoldx.models.esmfold2.EsmFold2ExperimentalConfig]
    from a ``type="experimental"`` recurrent-graph checkpoint ``config.json``.

    Every graph dimension is read from the checkpoint; nothing is guessed from a
    repository name.
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

    # ESMC trunk width/depth consumed by the folding head's language-model
    # projection are checkpoint graph dimensions, not repository defaults.
    config.esmc.hidden_size = raw_config.get("lm_d_model", config.esmc.hidden_size)
    config.esmc.num_hidden_layers = raw_config.get("lm_num_layers", config.esmc.num_hidden_layers)

    inputs = raw_config["inputs"]
    atom_encoder = inputs["atom_encoder"]
    config.input_embeddings.input_size = inputs["d_inputs"] + _TOKEN_FEATURE_VOCAB_EXPANSION
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

    config.folding_stack.num_blocks = raw_config["folding_trunk"]["n_layers"]

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
    denoiser_config.input_size = diffusion["c_s_inputs"] + _TOKEN_FEATURE_VOCAB_EXPANSION
    denoiser_config.fourier_embedding_size = diffusion["fourier_dim"]
    denoiser_config.num_atom_blocks = diffusion["atom_num_blocks"]
    denoiser_config.num_atom_attention_heads = diffusion["atom_num_heads"]
    denoiser_config.num_token_encoder_blocks = diffusion["token_num_blocks"]
    denoiser_config.num_token_encoder_attention_heads = diffusion["token_num_heads"]
    denoiser_config.expansion = diffusion["transition_multiplier"]

    confidence = raw_config["confidence_head"]
    confidence_config = config.confidence
    confidence_config.enabled = confidence.get("enabled", True)
    confidence_config.num_distogram_bins = confidence["distogram_bins"]
    confidence_config.min_distance = confidence["min_dist"]
    confidence_config.max_distance = confidence["max_dist"]
    confidence_config.num_plddt_bins = confidence["num_plddt_bins"]
    confidence_config.num_pae_bins = confidence["num_pae_bins"]
    confidence_config.folding_stack.num_blocks = confidence["folding_trunk"]["n_layers"]

    msa = raw_config["msa_encoder"]
    msa_config = config.msa_encoder
    msa_config.enabled = msa["enabled"]
    msa_config.msa_size = msa["d_msa"]
    msa_config.hidden_size = msa["d_hidden"]
    msa_config.num_stages = msa["n_layers"]
    msa_config.num_msa_attention_heads = msa["n_heads_msa"]
    msa_config.msa_head_dim = msa["msa_head_width"]

    return config


def convert_esmfold2_experimental_model(
    checkpoint_or_repo: str | PathLike[str],
    *,
    esmc: EsmCForMaskedLM,
    config_path: str | PathLike[str] | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> EsmFold2ExperimentalModel:
    r"""
    Strict-load a ``type="experimental"`` recurrent ESMFold2 head into an OFoldX model.

    The Biohub head safetensors does not carry ESMC. This converter combines the
    supplied complete masked-language model with the released head into one
    directly runnable OFoldX Model.
    """

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if esmc.config.hidden_size != raw_config["lm_d_model"]:
        raise RuntimeError(
            f"experimental ESMFold2 head requires ESMC width {raw_config['lm_d_model']}; got {esmc.config.hidden_size}"
        )
    if esmc.config.num_hidden_layers != raw_config["lm_num_layers"]:
        raise RuntimeError(
            f"experimental ESMFold2 head requires {raw_config['lm_num_layers']} ESMC layers; "
            f"got {esmc.config.num_hidden_layers}"
        )
    config = experimental_config_from_biohub(raw_config)
    config.esmc = EsmFold2EsmCConfig({name: getattr(esmc.config, name) for name in EsmFold2EsmCConfig().dict()})
    model = EsmFold2ExperimentalModel(config, esmc=esmc)
    head_state = load_esmfold2_state_dict(checkpoint_path)
    for name in _BIOHUB_UNUSED_STATE_KEYS:
        head_state.pop(name, None)
    incompatible = model.load_state_dict(head_state, strict=False)
    expected_missing = {
        *(f"esmc.{name}" for name in model.esmc.state_dict()),
        *(f"esmc_lm_head.{name}" for name in model.esmc_lm_head.state_dict()),
    }
    missing = set(incompatible.missing_keys)
    if missing != expected_missing or incompatible.unexpected_keys:
        missing_head = sorted(missing - expected_missing)
        unexpected = sorted(incompatible.unexpected_keys)
        raise RuntimeError(
            "experimental ESMFold2 checkpoint does not match the released head; "
            f"missing_head={missing_head!r}, unexpected={unexpected!r}"
        )
    return model


def convert_esmfold2_experimental_checkpoint(
    checkpoint_or_repo: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    esmc_source: str | PathLike[str] | None = None,
    config_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert one experimental ESMFold2 head into an ordinary complete pair.

    Combines the ``type="experimental"`` recurrent head with the exact raw
    Biohub ESMC named by its ``config.json`` in one complete root Model. Passing
    ``esmc_source`` overrides only that source location.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the complete
    artifact. No organization or token is assumed and nothing is published without an
    explicit ``repo_id``.
    """

    pipeline = _convert_esmfold2_experimental_pair(
        checkpoint_or_repo,
        esmc_source=esmc_source,
        config_path=config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    pipeline.save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def convert_esmfold2_design_checkpoints(
    fast_checkpoint_or_repo: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    fast_cutoff2025_checkpoint_or_repo: str | PathLike[str],
    experimental_checkpoint_or_repo: str | PathLike[str],
    experimental_cutoff2025_checkpoint_or_repo: str | PathLike[str],
    base300m_step250k_checkpoint_or_repo: str | PathLike[str],
    base300m_step500k_checkpoint_or_repo: str | PathLike[str],
    base300m_step750k_checkpoint_or_repo: str | PathLike[str],
    base300m_step1000k_checkpoint_or_repo: str | PathLike[str],
    base300m_step1500k_checkpoint_or_repo: str | PathLike[str],
    base600m_step250k_checkpoint_or_repo: str | PathLike[str],
    base600m_step500k_checkpoint_or_repo: str | PathLike[str],
    base600m_step750k_checkpoint_or_repo: str | PathLike[str],
    base600m_step1000k_checkpoint_or_repo: str | PathLike[str],
    base600m_step1500k_checkpoint_or_repo: str | PathLike[str],
    base6b_step250k_checkpoint_or_repo: str | PathLike[str],
    base6b_step500k_checkpoint_or_repo: str | PathLike[str],
    base6b_step750k_checkpoint_or_repo: str | PathLike[str],
    base6b_step1000k_checkpoint_or_repo: str | PathLike[str],
    base6b_step1500k_checkpoint_or_repo: str | PathLike[str],
    esmc_sources: Mapping[str, str | PathLike[str]] | None = None,
    fast_config_path: str | PathLike[str] | None = None,
    fast_cutoff2025_config_path: str | PathLike[str] | None = None,
    experimental_config_path: str | PathLike[str] | None = None,
    experimental_cutoff2025_config_path: str | PathLike[str] | None = None,
    base300m_step250k_config_path: str | PathLike[str] | None = None,
    base300m_step500k_config_path: str | PathLike[str] | None = None,
    base300m_step750k_config_path: str | PathLike[str] | None = None,
    base300m_step1000k_config_path: str | PathLike[str] | None = None,
    base300m_step1500k_config_path: str | PathLike[str] | None = None,
    base600m_step250k_config_path: str | PathLike[str] | None = None,
    base600m_step500k_config_path: str | PathLike[str] | None = None,
    base600m_step750k_config_path: str | PathLike[str] | None = None,
    base600m_step1000k_config_path: str | PathLike[str] | None = None,
    base600m_step1500k_config_path: str | PathLike[str] | None = None,
    base6b_step250k_config_path: str | PathLike[str] | None = None,
    base6b_step500k_config_path: str | PathLike[str] | None = None,
    base6b_step750k_config_path: str | PathLike[str] | None = None,
    base6b_step1000k_config_path: str | PathLike[str] | None = None,
    base6b_step1500k_config_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    cache_dir: str | PathLike[str] | None = None,
    local_files_only: bool = False,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Write the released fixed nineteen-pair ESMFold2 binder-design repository.

    The two inversion roles are reused among four hero roles, alongside fifteen
    scaling roles. Every checkpoint role is an explicit argument. Each head's
    exact ``esmc_id`` selects its complete frozen raw Biohub ESMC; optional
    local overrides are keyed by that complete official ID.
    """

    from .pipeline import EsmFold2DesignPipeline

    esmc_cache: dict[str, EsmCForMaskedLM] = {}
    fast = _convert_esmfold2_experimental_pair(
        fast_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=fast_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    fast_cutoff2025 = _convert_esmfold2_experimental_pair(
        fast_cutoff2025_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=fast_cutoff2025_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    experimental = _convert_esmfold2_experimental_pair(
        experimental_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=experimental_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    experimental_cutoff2025 = _convert_esmfold2_experimental_pair(
        experimental_cutoff2025_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=experimental_cutoff2025_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base300m_step250k = _convert_esmfold2_experimental_pair(
        base300m_step250k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base300m_step250k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base300m_step500k = _convert_esmfold2_experimental_pair(
        base300m_step500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base300m_step500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base300m_step750k = _convert_esmfold2_experimental_pair(
        base300m_step750k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base300m_step750k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base300m_step1000k = _convert_esmfold2_experimental_pair(
        base300m_step1000k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base300m_step1000k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base300m_step1500k = _convert_esmfold2_experimental_pair(
        base300m_step1500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base300m_step1500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base600m_step250k = _convert_esmfold2_experimental_pair(
        base600m_step250k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base600m_step250k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base600m_step500k = _convert_esmfold2_experimental_pair(
        base600m_step500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base600m_step500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base600m_step750k = _convert_esmfold2_experimental_pair(
        base600m_step750k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base600m_step750k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base600m_step1000k = _convert_esmfold2_experimental_pair(
        base600m_step1000k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base600m_step1000k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base600m_step1500k = _convert_esmfold2_experimental_pair(
        base600m_step1500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base600m_step1500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base6b_step250k = _convert_esmfold2_experimental_pair(
        base6b_step250k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base6b_step250k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base6b_step500k = _convert_esmfold2_experimental_pair(
        base6b_step500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base6b_step500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base6b_step750k = _convert_esmfold2_experimental_pair(
        base6b_step750k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base6b_step750k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base6b_step1000k = _convert_esmfold2_experimental_pair(
        base6b_step1000k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base6b_step1000k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    base6b_step1500k = _convert_esmfold2_experimental_pair(
        base6b_step1500k_checkpoint_or_repo,
        esmc_sources=esmc_sources,
        esmc_cache=esmc_cache,
        config_path=base6b_step1500k_config_path,
        dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    EsmFold2DesignPipeline(
        fast=fast,
        fast_cutoff2025=fast_cutoff2025,
        experimental=experimental,
        experimental_cutoff2025=experimental_cutoff2025,
        base300m_step250k=base300m_step250k,
        base300m_step500k=base300m_step500k,
        base300m_step750k=base300m_step750k,
        base300m_step1000k=base300m_step1000k,
        base300m_step1500k=base300m_step1500k,
        base600m_step250k=base600m_step250k,
        base600m_step500k=base600m_step500k,
        base600m_step750k=base600m_step750k,
        base600m_step1000k=base600m_step1000k,
        base600m_step1500k=base600m_step1500k,
        base6b_step250k=base6b_step250k,
        base6b_step500k=base6b_step500k,
        base6b_step750k=base6b_step750k,
        base6b_step1000k=base6b_step1000k,
        base6b_step1500k=base6b_step1500k,
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def _convert_esmfold2_experimental_pair(
    checkpoint_or_repo: str | PathLike[str],
    *,
    esmc_source: str | PathLike[str] | None = None,
    esmc_sources: Mapping[str, str | PathLike[str]] | None = None,
    esmc_cache: dict[str, EsmCForMaskedLM] | None = None,
    config_path: str | PathLike[str] | None,
    dtype: torch.dtype | None,
    cache_dir: str | PathLike[str] | None,
    local_files_only: bool,
    token: str | None,
) -> EsmFold2ExperimentalPipeline:
    from .pipeline import EsmFold2ExperimentalPipeline

    checkpoint_path, resolved_config_path = resolve_esmfold2_source(
        checkpoint_or_repo,
        config_path=config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if raw_config.get("model_type") != "esmfold2" or raw_config.get("type") != "experimental":
        raise ValueError("checkpoint config is not the experimental Biohub ESMFold2 architecture")
    esmc_id = raw_config["esmc_id"]
    if not isinstance(esmc_id, str):
        raise TypeError("experimental ESMFold2 esmc_id must be a string")
    if esmc_cache is not None and esmc_id in esmc_cache:
        esmc = esmc_cache[esmc_id]
    else:
        source = esmc_source
        if source is None:
            source = esmc_id if esmc_sources is None else esmc_sources.get(esmc_id, esmc_id)
        esmc = load_biohub_esmc(
            source,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            token=token,
        )
        if esmc_cache is not None:
            esmc_cache[esmc_id] = esmc
    model = convert_esmfold2_experimental_model(
        checkpoint_path,
        esmc=esmc,
        config_path=resolved_config_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )
    if dtype is not None:
        model.to(dtype=dtype)
    processor = EsmFold2ExperimentalProcessor(
        EsmFold2ExperimentalProcessorConfig({"msa_conditioning": model.config.use_msa_features}),
        dtype=dtype,
    )
    return EsmFold2ExperimentalPipeline(model=model, processor=processor)


def main(argv: list[str] | None = None) -> None:
    r"""
    Run esmfold2 checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_or_repo", help="Head checkpoint or Hub repo, e.g. biohub/ESMFold2")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--esmc", default=None, help="optional ESMC source override for standard conversion")
    parser.add_argument("--config", type=Path, default=None, help="Optional upstream config.json")
    parser.add_argument(
        "--fast-cutoff2025",
        default=None,
        help="Fast-Cutoff2025 inversion head required by --experimental",
    )
    parser.add_argument(
        "--fast-cutoff2025-config",
        type=Path,
        default=None,
        help="Optional Fast-Cutoff2025 upstream config.json",
    )
    parser.add_argument(
        "--experimental-checkpoint",
        default=None,
        help="Experimental hero-critic head required by --experimental",
    )
    parser.add_argument(
        "--experimental-cutoff2025",
        default=None,
        help="Experimental-Cutoff2025 hero-critic head required by --experimental",
    )
    parser.add_argument("--base300m-step250k", default=None)
    parser.add_argument("--base300m-step500k", default=None)
    parser.add_argument("--base300m-step750k", default=None)
    parser.add_argument("--base300m-step1000k", default=None)
    parser.add_argument("--base300m-step1500k", default=None)
    parser.add_argument("--base600m-step250k", default=None)
    parser.add_argument("--base600m-step500k", default=None)
    parser.add_argument("--base600m-step750k", default=None)
    parser.add_argument("--base600m-step1000k", default=None)
    parser.add_argument("--base600m-step1500k", default=None)
    parser.add_argument("--base6b-step250k", default=None)
    parser.add_argument("--base6b-step500k", default=None)
    parser.add_argument("--base6b-step750k", default=None)
    parser.add_argument("--base6b-step1000k", default=None)
    parser.add_argument("--base6b-step1500k", default=None)
    parser.add_argument("--experimental-checkpoint-config", type=Path, default=None)
    parser.add_argument("--experimental-cutoff2025-config", type=Path, default=None)
    parser.add_argument("--base300m-step250k-config", type=Path, default=None)
    parser.add_argument("--base300m-step500k-config", type=Path, default=None)
    parser.add_argument("--base300m-step750k-config", type=Path, default=None)
    parser.add_argument("--base300m-step1000k-config", type=Path, default=None)
    parser.add_argument("--base300m-step1500k-config", type=Path, default=None)
    parser.add_argument("--base600m-step250k-config", type=Path, default=None)
    parser.add_argument("--base600m-step500k-config", type=Path, default=None)
    parser.add_argument("--base600m-step750k-config", type=Path, default=None)
    parser.add_argument("--base600m-step1000k-config", type=Path, default=None)
    parser.add_argument("--base600m-step1500k-config", type=Path, default=None)
    parser.add_argument("--base6b-step250k-config", type=Path, default=None)
    parser.add_argument("--base6b-step500k-config", type=Path, default=None)
    parser.add_argument("--base6b-step750k-config", type=Path, default=None)
    parser.add_argument("--base6b-step1000k-config", type=Path, default=None)
    parser.add_argument("--base6b-step1500k-config", type=Path, default=None)
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
        help="write the complete fixed nineteen-pair binder-design repository",
    )
    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    if args.experimental:
        required_roles = (
            ("--fast-cutoff2025", args.fast_cutoff2025),
            ("--experimental-checkpoint", args.experimental_checkpoint),
            ("--experimental-cutoff2025", args.experimental_cutoff2025),
            ("--base300m-step250k", args.base300m_step250k),
            ("--base300m-step500k", args.base300m_step500k),
            ("--base300m-step750k", args.base300m_step750k),
            ("--base300m-step1000k", args.base300m_step1000k),
            ("--base300m-step1500k", args.base300m_step1500k),
            ("--base600m-step250k", args.base600m_step250k),
            ("--base600m-step500k", args.base600m_step500k),
            ("--base600m-step750k", args.base600m_step750k),
            ("--base600m-step1000k", args.base600m_step1000k),
            ("--base600m-step1500k", args.base600m_step1500k),
            ("--base6b-step250k", args.base6b_step250k),
            ("--base6b-step500k", args.base6b_step500k),
            ("--base6b-step750k", args.base6b_step750k),
            ("--base6b-step1000k", args.base6b_step1000k),
            ("--base6b-step1500k", args.base6b_step1500k),
        )
        missing_roles = [flag for flag, value in required_roles if value is None]
        if missing_roles:
            parser.error("--experimental requires " + ", ".join(missing_roles))
        convert_esmfold2_design_checkpoints(
            args.checkpoint_or_repo,
            args.output_dir,
            fast_cutoff2025_checkpoint_or_repo=args.fast_cutoff2025,
            experimental_checkpoint_or_repo=args.experimental_checkpoint,
            experimental_cutoff2025_checkpoint_or_repo=args.experimental_cutoff2025,
            base300m_step250k_checkpoint_or_repo=args.base300m_step250k,
            base300m_step500k_checkpoint_or_repo=args.base300m_step500k,
            base300m_step750k_checkpoint_or_repo=args.base300m_step750k,
            base300m_step1000k_checkpoint_or_repo=args.base300m_step1000k,
            base300m_step1500k_checkpoint_or_repo=args.base300m_step1500k,
            base600m_step250k_checkpoint_or_repo=args.base600m_step250k,
            base600m_step500k_checkpoint_or_repo=args.base600m_step500k,
            base600m_step750k_checkpoint_or_repo=args.base600m_step750k,
            base600m_step1000k_checkpoint_or_repo=args.base600m_step1000k,
            base600m_step1500k_checkpoint_or_repo=args.base600m_step1500k,
            base6b_step250k_checkpoint_or_repo=args.base6b_step250k,
            base6b_step500k_checkpoint_or_repo=args.base6b_step500k,
            base6b_step750k_checkpoint_or_repo=args.base6b_step750k,
            base6b_step1000k_checkpoint_or_repo=args.base6b_step1000k,
            base6b_step1500k_checkpoint_or_repo=args.base6b_step1500k,
            fast_config_path=args.config,
            fast_cutoff2025_config_path=args.fast_cutoff2025_config,
            experimental_config_path=args.experimental_checkpoint_config,
            experimental_cutoff2025_config_path=args.experimental_cutoff2025_config,
            base300m_step250k_config_path=args.base300m_step250k_config,
            base300m_step500k_config_path=args.base300m_step500k_config,
            base300m_step750k_config_path=args.base300m_step750k_config,
            base300m_step1000k_config_path=args.base300m_step1000k_config,
            base300m_step1500k_config_path=args.base300m_step1500k_config,
            base600m_step250k_config_path=args.base600m_step250k_config,
            base600m_step500k_config_path=args.base600m_step500k_config,
            base600m_step750k_config_path=args.base600m_step750k_config,
            base600m_step1000k_config_path=args.base600m_step1000k_config,
            base600m_step1500k_config_path=args.base600m_step1500k_config,
            base6b_step250k_config_path=args.base6b_step250k_config,
            base6b_step500k_config_path=args.base6b_step500k_config,
            base6b_step750k_config_path=args.base6b_step750k_config,
            base6b_step1000k_config_path=args.base6b_step1000k_config,
            base6b_step1500k_config_path=args.base6b_step1500k_config,
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
