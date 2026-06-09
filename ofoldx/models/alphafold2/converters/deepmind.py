r"""
Official DeepMind AlphaFold 2 checkpoint conversion.

Official DeepMind AlphaFold 2 parameters are distributed as ``params_*.npz`` files in
``alphafold_params_2022-12-06.tar``. The archive keys follow the upstream Haiku
flat-parameter convention ``scope//name``. This module handles the official package
shape end to end: it finds the selected model, loads the NPZ without pickle,
normalizes the flat Haiku names into OFoldX-style tensor keys, converts common
Haiku dense/attention/outer-product layouts into PyTorch-style layouts, strict-loads
them into the PyTorch AF2 model, and writes an OFoldX model artifact.

The current OFoldX ``AlphaFold2Model`` can run through an AF2-shaped forward path,
and official monomer / pTM parameters strict-load into the PyTorch parameter tree.
AF-Multimer checkpoints use a distinct feature/model surface and are rejected until
that architecture is implemented explicitly.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import torch
from chanfig import Config
from torch import Tensor

from ....artifacts import ARTIFACT_MANIFEST_NAME
from ...mixin import ModelMixin, no_init_weights

OFFICIAL_AF2_PARAMS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
OFFICIAL_AF2_PARAMS_ARCHIVE = "alphafold_params_2022-12-06.tar"
OFFICIAL_AF2_CODE_LICENSE = "Apache-2.0"
OFFICIAL_AF2_PARAMS_LICENSE = "CC BY 4.0"
CONVERSION_CONFIG_NAME = "conversion_config.json"
CONVERTED_WEIGHTS_NAME = ModelMixin.WEIGHTS_NAME

_MODEL_NAME_RE = re.compile(r"^model_[1-5](?:_ptm|_multimer(?:_v[123])?)?$")
OFFICIAL_AF2_MODEL_NAMES = tuple(
    [f"model_{index}" for index in range(1, 6)]
    + [f"model_{index}_ptm" for index in range(1, 6)]
    + [f"model_{index}_multimer_v3" for index in range(1, 6)]
)
_STRICT_LOAD_READY = "strict_load_ready"
_STRICT_LOAD_BLOCKED = "strict_load_blocked"
_PARAMETER_NAME_MAP = {
    "weights": ("weight",),
    "scale": ("weight",),
    "offset": ("bias",),
    "bias": ("bias",),
    "query_w": ("query_projection", "weight"),
    "key_w": ("key_projection", "weight"),
    "value_w": ("value_projection", "weight"),
    "gating_w": ("gate_projection", "weight"),
    "gating_b": ("gate_projection", "bias"),
    "output_w": ("output_projection", "weight"),
    "output_b": ("output_projection", "bias"),
}
_HEAD_SCOPE_MAP = {
    "distogram_head": "distogram_head",
    "experimentally_resolved_head": "experimentally_resolved_head",
    "masked_msa_head": "masked_msa_head",
    "predicted_aligned_error_head": "predicted_aligned_error_head",
    "predicted_lddt_head": "predicted_lddt_head",
}
_INPUT_EMBEDDING_SCOPE_MAP = {
    "preprocess_1d": "preprocess_1d",
    "preprocess_msa": "preprocess_msa",
    "left_single": "left_single",
    "right_single": "right_single",
    "relpos_projection": "relpos_projection",
    # Official AF2 has this typo in the Haiku module name.
    "pair_activiations": "relpos_projection",
}
_RECYCLING_SCOPE_MAP = {
    "prev_pos_linear": "prev_pos_linear",
    "prev_msa_first_row_norm": "prev_msa_first_row_norm",
    "prev_pair_norm": "prev_pair_norm",
}
_EXTRA_MSA_EMBEDDING_SCOPE_MAP = {
    "extra_msa_activations": "extra_msa_projection",
    "extra_msa_projection": "extra_msa_projection",
}
_STACK_SCOPE_MAP = {
    "outer_product_mean": ("outer_product_mean",),
    "layer_norm_input": ("layer_norm_input",),
    "input_layer_norm": ("input_layer_norm",),
    "transition1": ("transition1",),
    "transition2": ("transition2",),
    "msa_row_attention_with_pair_bias": ("msa_stack", "msa_row_attention_with_pair_bias"),
    "msa_column_attention": ("msa_stack", "msa_column_attention"),
    "msa_column_global_attention": ("msa_stack", "msa_column_global_attention"),
    "msa_transition": ("msa_stack", "feed_forward"),
    "triangle_multiplication_outgoing": ("pair_stack", "triangle_multiplication_outgoing"),
    "triangle_multiplication_incoming": ("pair_stack", "triangle_multiplication_incoming"),
    "triangle_attention_starting_node": ("pair_stack", "triangle_attention_starting_node"),
    "triangle_attention_ending_node": ("pair_stack", "triangle_attention_ending_node"),
    "pair_transition": ("pair_stack", "feed_forward"),
    "query_norm": ("query_norm",),
    "feat_2d_norm": ("feat_2d_norm",),
    "layer_norm": ("layer_norm",),
    "center_layer_norm": ("center_layer_norm",),
    "left_gate": ("left_gate",),
    "right_gate": ("right_gate",),
    "gating_linear": ("gating_linear",),
    "attention": ("attention", "projection"),
}
_BLOCKED_PREFIX_REASONS: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class _NormalizedParamKey:
    target_key: str
    target_group: str
    strict_load_status: str
    strict_load_blocker: str | None


@dataclass(frozen=True, slots=True)
class _LoadedOfficialParams:
    tensors: dict[str, Tensor]
    model_name: str
    source_path: Path
    archive_member: str | None


def load_official_alphafold2_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Tensor]:
    r"""
    Load one official AlphaFold 2 parameter file as PyTorch tensors.

    ``params_path`` may be a direct ``.npz`` file, a DeepMind download directory
    containing ``params/params_*.npz``, or the downloaded
    ``alphafold_params_2022-12-06.tar`` archive. When the path contains more than
    one model file, ``model_name`` selects the desired one.

    Args:
        params_path: Direct NPZ file, params directory, download root, or tar archive.
        model_name: Official model id such as ``"model_1"`` or ``"model_1_ptm"``.

    Returns:
        Mapping from official flat Haiku key to tensor.

    Raises:
        ValueError: If the path/model selection does not identify exactly one official
            parameter file or if the NPZ does not look like official AF2 params.
    """

    return _load_official_alphafold2_params(params_path, model_name=model_name).tensors


def inspect_official_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    r"""
    Return a lightweight summary of an official AF2 checkpoint.

    Args:
        checkpoint_path: Official NPZ, params directory/download root, or tar archive.
        model_name: Official model id when the path contains multiple models.

    Returns:
        JSON-serializable checkpoint summary.
    """

    loaded = _load_official_alphafold2_params(checkpoint_path, model_name=model_name)
    scopes = Counter(key.split("//", 1)[0] for key in loaded.tensors)
    mapping_summary = inspect_official_alphafold2_mapping(loaded.tensors)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_source": "deepmind",
        "checkpoint_format": "official_alphafold2_npz",
        "model_name": loaded.model_name,
        "source_archive_member": loaded.archive_member,
        "num_tensors": len(loaded.tensors),
        "num_parameters": sum(tensor.numel() for tensor in loaded.tensors.values()),
        "top_scopes": dict(scopes.most_common(20)),
        "mapping_summary": mapping_summary,
        "dtypes": _dtype_counts(loaded.tensors),
        "source_release": OFFICIAL_AF2_PARAMS_URL,
        "official_code_license": OFFICIAL_AF2_CODE_LICENSE,
        "official_params_license": OFFICIAL_AF2_PARAMS_LICENSE,
    }


def convert_deepmind_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    inspect_only: bool = False,
    verify_artifact: bool = True,
) -> dict[str, Any]:
    r"""
    Convert official AlphaFold 2 parameters into an OFoldX model artifact.

    The artifact contains ``manifest.json``, ``config.json``, ``model.safetensors``,
    and ``conversion_config.json``. Official monomer and pTM checkpoints are
    strict-loaded into [`AlphaFold2Model`][] before the artifact is written.

    Args:
        checkpoint_path: Official AF2 NPZ, params directory/download root, or tar archive.
        output_dir: Directory to write the converted model artifact.
        model_name: Official model id when ``checkpoint_path`` contains multiple models.
        dtype: Optional dtype for floating-point persisted tensors.
        inspect_only: If ``True``, only return a checkpoint summary and write nothing.
        verify_artifact: If ``True``, reload the just-written artifact with strict
            model loading before returning.

    Returns:
        JSON-serializable conversion report.
    """

    if inspect_only:
        return inspect_official_alphafold2_checkpoint(checkpoint_path, model_name=model_name)

    loaded = _load_official_alphafold2_params(checkpoint_path, model_name=model_name)
    _validate_supported_model_package(loaded)
    converted = convert_official_alphafold2_state_dict(loaded.tensors, dtype=dtype)

    from ..config import AlphaFold2Config
    from ..model import AlphaFold2Model

    model_config = AlphaFold2Config(_model_config_overrides(loaded.model_name))
    model_config["checkpoint_variant"] = loaded.model_name
    model_config["model_family"] = _model_family(loaded.model_name)
    with no_init_weights():
        model = AlphaFold2Model(model_config)
    _validate_converted_state_dict_matches_model(converted, model.state_dict(), model_name=loaded.model_name)
    model.load_state_dict(converted, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)

    output_path = Path(output_dir)
    model.save_pretrained(output_path)
    _write_processor_component(output_path)

    persisted_state = dict(model.state_dict())
    config = _conversion_config(loaded, persisted_state, dtype=dtype)
    Config(config).json(output_path / CONVERSION_CONFIG_NAME)
    _update_manifest_metadata(output_path)

    if verify_artifact:
        AlphaFold2Model.from_pretrained(output_path, strict=True)

    return {
        "checkpoint": str(Path(checkpoint_path)),
        "output_dir": str(output_path),
        "architecture": "alphafold2",
        "artifact_kind": "model",
        "checkpoint_source": "deepmind",
        "checkpoint_format": "official_alphafold2_npz",
        "model_name": loaded.model_name,
        "source_archive_member": loaded.archive_member,
        "source_release": OFFICIAL_AF2_PARAMS_URL,
        "official_code_license": OFFICIAL_AF2_CODE_LICENSE,
        "official_params_license": OFFICIAL_AF2_PARAMS_LICENSE,
        "runnable_model": True,
        "tensor_count": len(persisted_state),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(persisted_state),
        "mapping_summary": _mapping_summary_from_keys(loaded.tensors),
        "processor_written": True,
        "verified": verify_artifact,
    }


def convert_official_alphafold2_state_dict(
    params: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Normalize official flat Haiku AF2 parameter names to OFoldX-style tensor keys.

    Haiku dense weights store input channels before output channels, and AF2's
    hand-written attention/outer-product modules use parameter names such as
    ``query_w`` and ``output_w``. The converter rewrites the known layouts into
    PyTorch-style ``out, in`` matrices where the target module shape is known.

    Args:
        params: Official ``scope//name`` tensor mapping.
        dtype: Optional dtype for floating-point tensors.

    Returns:
        Converted state dict preserving all tensors under normalized keys.

    Raises:
        ValueError: If a source key is malformed or two source keys collide after normalization.
    """

    _validate_official_param_keys(params)
    converted: dict[str, Tensor] = {}
    for source_key, tensor in params.items():
        target_key = normalize_official_param_key(source_key)
        target = convert_official_tensor(source_key, tensor)
        if dtype is not None and target.is_floating_point():
            target = target.to(dtype=dtype)
        expanded = _expand_stacked_block_tensor(source_key, target_key, tensor, target)
        for expanded_key, expanded_tensor in expanded.items():
            if expanded_key in converted:
                raise ValueError(f"multiple official AF2 tensors map to {expanded_key!r}")
            converted[expanded_key] = expanded_tensor
    return converted


def convert_official_tensor(key: str, tensor: Tensor) -> Tensor:
    r"""
    Convert one official tensor value to PyTorch layout where the mapping is known.
    """

    _, name = _split_official_param_key(key)
    is_global_column_attention = "msa_column_global_attention" in key
    if name in {"query_w", "gating_w"} and tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], tensor.shape[2] * tensor.shape[3], tensor.shape[1])
    if name in {"key_w", "value_w"} and tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], tensor.shape[2] * tensor.shape[3], tensor.shape[1])
    if is_global_column_attention and name in {"key_w", "value_w"} and tensor.ndim == 3:
        return tensor.transpose(-1, -2).contiguous()
    if name in {"query_w", "key_w", "value_w", "gating_w"} and tensor.ndim == 3:
        return tensor.permute(1, 2, 0).reshape(tensor.shape[1] * tensor.shape[2], tensor.shape[0]).contiguous()
    if name in {"query_w", "key_w", "value_w", "gating_w", "output_w"} and tensor.ndim == 2:
        return tensor.transpose(-1, -2).contiguous()
    if name == "output_w" and tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2).reshape(tensor.shape[0], tensor.shape[3], tensor.shape[1] * tensor.shape[2])
    if name == "output_w" and tensor.ndim == 3:
        return tensor.permute(2, 0, 1).reshape(tensor.shape[2], tensor.shape[0] * tensor.shape[1]).contiguous()
    if name in {"gating_b"} and tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0], -1).contiguous()
    if name in {"gating_b"} and tensor.ndim == 2:
        return tensor.reshape(-1).contiguous()
    if name in {"feat_2d_weights"} and tensor.ndim == 3:
        return tensor.transpose(-1, -2).contiguous()
    if name in {"feat_2d_weights"} and tensor.ndim == 2:
        return tensor.transpose(-1, -2).contiguous()
    if name == "weights" and tensor.ndim >= 2:
        return tensor.transpose(-1, -2).contiguous()
    return tensor.contiguous()


def normalize_official_param_key(key: str) -> str:
    r"""
    Return the normalized OFoldX-style key for one official AF2 flat Haiku key.

    Examples:
        >>> normalize_official_param_key("alphafold/alphafold_iteration/evoformer/preprocess_1d//weights")
        'input_embedding.preprocess_1d.weight'
        >>> normalize_official_param_key("alphafold/alphafold_iteration/structure_module/layer_norm//scale")
        'structure_module.layer_norm.weight'
    """

    return _normalize_official_param_key(key).target_key


def _expand_stacked_block_tensor(
    source_key: str,
    target_key: str,
    source_tensor: Tensor,
    converted_tensor: Tensor,
) -> dict[str, Tensor]:
    if not _is_stacked_block_tensor(source_key, target_key, source_tensor):
        return {target_key: converted_tensor}
    if ".blocks.stacked." in target_key:
        marker = ".blocks.stacked."
        replacement = ".blocks.{block_index}."
    elif ".template_pair_stack.stacked." in target_key:
        marker = ".template_pair_stack.stacked."
        replacement = ".template_pair_stack.blocks.{block_index}."
    else:
        return {target_key: converted_tensor}
    return {
        target_key.replace(marker, replacement.format(block_index=block_index)): converted_tensor[
            block_index
        ].contiguous()
        for block_index in range(converted_tensor.shape[0])
    }


def _is_stacked_block_tensor(source_key: str, target_key: str, tensor: Tensor) -> bool:
    if ".blocks.stacked." not in target_key and ".template_pair_stack.stacked." not in target_key:
        return False
    _, name = _split_official_param_key(source_key)
    if name in {"query_w", "gating_w", "output_w"}:
        return tensor.ndim == 4
    if name in {"key_w", "value_w"}:
        if "msa_column_global_attention" in source_key:
            return tensor.ndim == 3
        return tensor.ndim == 4
    if name in {"feat_2d_weights", "gating_b"}:
        return tensor.ndim == 3
    if name in {"output_b", "scale", "offset", "bias"}:
        return tensor.ndim == 2
    if name == "weights":
        return tensor.ndim >= 3
    return False


def inspect_official_alphafold2_mapping(params: Mapping[str, Tensor]) -> dict[str, Any]:
    r"""
    Summarize how official AF2 tensors normalize into the OFoldX parameter tree.

    The summary is intentionally about strict-load readiness, not just string
    rewriting. A tensor can have a clean OFoldX-style target key while the current
    module still cannot strict-load the official checkpoint because the native AF2
    module is not implemented yet.
    """

    _validate_official_param_keys(params)
    return _mapping_summary_from_keys(params)


def unflatten_official_params(params: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    r"""
    Rebuild the two-level Haiku ``scope -> name -> tensor`` structure.

    This mirrors the upstream ``flat_params_to_haiku`` helper while keeping values
    as PyTorch tensors for converter-side inspection.
    """

    nested: dict[str, dict[str, Tensor]] = {}
    for key, tensor in params.items():
        scope, name = _split_official_param_key(key)
        nested.setdefault(scope, {})[name] = tensor
    return nested


def official_param_filename(model_name: str) -> str:
    r"""
    Return the official NPZ filename for an AlphaFold 2 model id.
    """

    _validate_model_name(model_name)
    return f"params_{model_name}.npz"


def _normalize_official_param_key(key: str) -> _NormalizedParamKey:
    scope, name = _split_official_param_key(key)
    parts = [part for part in scope.split("/") if part]
    if parts[:2] == ["alphafold", "alphafold_iteration"]:
        parts = parts[2:]
    elif parts[:1] == ["alphafold"]:
        parts = parts[1:]

    normalized_parts = _normalize_scope_parts(parts)
    parameter_parts = _normalize_parameter_parts(name, normalized_parts)
    target_key = ".".join([*normalized_parts, *parameter_parts])
    target_group = _target_group(target_key)
    strict_load_blocker = _strict_load_blocker(target_key)
    return _NormalizedParamKey(
        target_key=target_key,
        target_group=target_group,
        strict_load_status=_STRICT_LOAD_READY if strict_load_blocker is None else _STRICT_LOAD_BLOCKED,
        strict_load_blocker=strict_load_blocker,
    )


def _mapping_summary_from_keys(params: Mapping[str, Tensor]) -> dict[str, Any]:
    mappings = [_normalize_official_param_key(key) for key in params]
    status_counts = Counter(mapping.strict_load_status for mapping in mappings)
    group_counts = Counter(mapping.target_group for mapping in mappings)
    blocker_counts = Counter(mapping.strict_load_blocker for mapping in mappings if mapping.strict_load_blocker)
    blocked_groups: dict[str, int] = {}
    for mapping in mappings:
        if mapping.strict_load_blocker is not None:
            blocked_groups[mapping.target_group] = blocked_groups.get(mapping.target_group, 0) + 1
    return {
        "strict_load_ready": not blocker_counts,
        "strict_load_status_counts": dict(sorted(status_counts.items())),
        "target_group_counts": dict(sorted(group_counts.items())),
        "blocked_group_counts": dict(sorted(blocked_groups.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
    }


def _normalize_scope_parts(parts: list[str]) -> list[str]:
    if not parts:
        raise ValueError("official AF2 parameter scope is empty after removing top-level wrapper")
    head_scope = _HEAD_SCOPE_MAP.get(parts[0])
    if head_scope is not None:
        return _normalize_head_scope([head_scope, *parts[1:]])
    if parts[0] == "evoformer":
        return _normalize_evoformer_scope(parts[1:])
    if parts[0] == "structure_module":
        return ["structure_module", *_normalize_structure_module_scope(parts[1:])]
    return parts


def _normalize_evoformer_scope(parts: list[str]) -> list[str]:
    if not parts:
        return ["evoformer"]
    input_embedding_scope = _INPUT_EMBEDDING_SCOPE_MAP.get(parts[0])
    if input_embedding_scope is not None:
        return ["input_embedding", input_embedding_scope, *_normalize_stack_scope(parts[1:])]
    recycling_scope = _RECYCLING_SCOPE_MAP.get(parts[0])
    if recycling_scope is not None:
        return ["recycling_embedding", recycling_scope, *_normalize_stack_scope(parts[1:])]
    extra_msa_embedding_scope = _EXTRA_MSA_EMBEDDING_SCOPE_MAP.get(parts[0])
    if extra_msa_embedding_scope is not None:
        return ["extra_msa_embedding", extra_msa_embedding_scope, *_normalize_stack_scope(parts[1:])]
    if parts[0] == "single_activations":
        return ["single_activations", *_normalize_stack_scope(parts[1:])]
    if parts[0] in {"template_single_embedding", "template_projection"}:
        return ["template_msa_embedding", *_normalize_stack_scope(parts)]
    if parts[0] == "template_embedding":
        return ["template_embedder", *_normalize_stack_scope(parts[1:])]
    if parts[0] == "evoformer_iteration":
        return ["evoformer", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    if parts[0] == "extra_msa_iteration":
        return ["extra_msa_stack", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    if parts[0] == "extra_msa_stack":
        return ["extra_msa_stack", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    return ["evoformer", *_normalize_stack_scope(parts)]


def _normalize_evoformer_block_scope(parts: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in parts:
        if part == "__layer_stack_no_state":
            continue
        mapped = _STACK_SCOPE_MAP.get(part)
        if mapped is None:
            normalized.append(part)
        else:
            normalized.extend(mapped)
    return normalized


def _normalize_stack_scope(parts: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in parts:
        if part == "__layer_stack_no_state":
            normalized.append("stacked")
            continue
        mapped = _STACK_SCOPE_MAP.get(part)
        if mapped is None:
            normalized.append(part)
        else:
            normalized.extend(mapped)
    return normalized


def _normalize_structure_module_scope(parts: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in parts:
        if part == "rigid_sidechain":
            normalized.append("sidechain")
            continue

        transition_index = _repeated_haiku_index(part, "transition")
        if transition_index is not None and "fold_iteration" in normalized and "sidechain" not in normalized:
            normalized.extend(("transition", "layers", str(transition_index)))
            continue

        input_projection_index = _repeated_haiku_index(part, "input_projection")
        if input_projection_index is not None and "sidechain" in normalized:
            normalized.extend(("input_projection", str(input_projection_index)))
            continue

        resblock1_index = _repeated_haiku_index(part, "resblock1")
        if resblock1_index is not None and "sidechain" in normalized:
            normalized.extend(("resblocks", str(resblock1_index), "resblock1"))
            continue

        resblock2_index = _repeated_haiku_index(part, "resblock2")
        if resblock2_index is not None and "sidechain" in normalized:
            normalized.extend(("resblocks", str(resblock2_index), "resblock2"))
            continue

        normalized.append(part)
    return normalized


def _repeated_haiku_index(name: str, base: str) -> int | None:
    if name == base:
        return 0
    prefix = f"{base}_"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.isdecimal():
            return int(suffix)
    return None


def _normalize_head_scope(parts: list[str]) -> list[str]:
    if parts[:2] == ["distogram_head", "half_logits"]:
        return ["distogram_head", "linear"]
    return parts


def _normalize_parameter_parts(name: str, normalized_scope_parts: list[str]) -> tuple[str, ...]:
    if name == "feat_2d_weights":
        return ("feat_2d_weights", "weight")
    parameter_parts = _PARAMETER_NAME_MAP.get(name)
    if parameter_parts is not None:
        return parameter_parts
    return (name,)


def _target_group(target_key: str) -> str:
    if target_key.startswith("input_embedding."):
        return "input_embedding"
    if target_key.startswith("recycling_embedding."):
        return "recycling"
    if target_key.startswith("extra_msa_embedding."):
        return "extra_msa_embedding"
    if target_key.startswith(("evoformer.", "extra_msa_stack.", "single_activations.")):
        return "evoformer"
    if target_key.startswith(("template_embedder.", "template_msa_embedding.")):
        return "template"
    if target_key.startswith("structure_module."):
        return "structure_module"
    if target_key.startswith(
        (
            "distogram_head.",
            "experimentally_resolved_head.",
            "masked_msa_head.",
            "predicted_aligned_error_head.",
            "predicted_lddt_head.",
        )
    ):
        return "heads"
    return "other"


def _strict_load_blocker(target_key: str) -> str | None:
    for prefix, reason in _BLOCKED_PREFIX_REASONS.items():
        if target_key.startswith(prefix):
            return reason
    return None


def _load_official_alphafold2_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None,
) -> _LoadedOfficialParams:
    source_path = Path(params_path)
    with (
        _open_official_npz(source_path, model_name=model_name) as (handle, resolved_model_name, archive_member),
        np.load(handle, allow_pickle=False) as npz,
    ):
        tensors = {_key: _tensor_from_numpy(npz[_key]) for _key in npz.files}
    _validate_official_param_keys(tensors)
    return _LoadedOfficialParams(
        tensors=tensors,
        model_name=resolved_model_name,
        source_path=source_path,
        archive_member=archive_member,
    )


@contextmanager
def _open_official_npz(
    source_path: Path,
    *,
    model_name: str | None,
) -> Iterator[tuple[BinaryIO, str, str | None]]:
    if model_name is not None:
        _validate_model_name(model_name)
    if source_path.is_dir():
        param_file, resolved_model_name = _select_directory_param_file(source_path, model_name=model_name)
        with param_file.open("rb") as handle:
            yield handle, resolved_model_name, None
        return
    if source_path.suffix == ".npz":
        resolved_model_name = model_name or _model_name_from_param_filename(source_path.name)
        _validate_model_name(resolved_model_name)
        with source_path.open("rb") as handle:
            yield handle, resolved_model_name, None
        return
    if source_path.is_file() and tarfile.is_tarfile(source_path):
        with tarfile.open(source_path) as archive:
            member = _select_archive_member(archive, model_name=model_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not extract {member.name!r} from {source_path}")
            payload = extracted.read()
        resolved_model_name = _model_name_from_param_filename(Path(member.name).name)
        yield io.BytesIO(payload), resolved_model_name, member.name
        return
    raise ValueError(
        "expected an official AlphaFold 2 params .npz file, params directory/download root, "
        f"or tar archive: {source_path}"
    )


def _select_directory_param_file(source_path: Path, *, model_name: str | None) -> tuple[Path, str]:
    search_roots = [source_path]
    params_dir = source_path / "params"
    if params_dir.is_dir():
        search_roots.append(params_dir)
    if model_name is not None:
        filename = official_param_filename(model_name)
        matches = [root / filename for root in search_roots if (root / filename).is_file()]
    else:
        matches = sorted({candidate for root in search_roots for candidate in root.glob("params_*.npz")})
    if len(matches) != 1:
        detail = "set model_name to one of the official AF2 model ids" if matches else "no params_*.npz files found"
        raise ValueError(f"expected exactly one official AF2 parameter file under {source_path}; {detail}")
    resolved_model_name = _model_name_from_param_filename(matches[0].name)
    _validate_model_name(resolved_model_name)
    return matches[0], resolved_model_name


def _select_archive_member(archive: tarfile.TarFile, *, model_name: str | None) -> tarfile.TarInfo:
    members = [
        member
        for member in archive.getmembers()
        if member.isfile() and Path(member.name).name.startswith("params_") and Path(member.name).suffix == ".npz"
    ]
    if model_name is not None:
        filename = official_param_filename(model_name)
        members = [member for member in members if Path(member.name).name == filename]
    if len(members) != 1:
        detail = "set model_name to select one NPZ" if members else "no params_*.npz members found"
        archive_name = archive.name.decode() if isinstance(archive.name, bytes) else str(archive.name)
        raise ValueError(f"expected exactly one official AF2 parameter file in {archive_name}; {detail}")
    return members[0]


def _model_name_from_param_filename(filename: str) -> str:
    if not filename.startswith("params_") or not filename.endswith(".npz"):
        raise ValueError(f"not an official AF2 parameter filename: {filename!r}")
    return filename[len("params_") : -len(".npz")]


def _validate_model_name(model_name: str) -> None:
    if not _MODEL_NAME_RE.fullmatch(model_name):
        known = ", ".join(OFFICIAL_AF2_MODEL_NAMES[:3])
        raise ValueError(f"unsupported AlphaFold 2 model name {model_name!r}; expected names like {known}, ...")


def _tensor_from_numpy(array: np.ndarray) -> Tensor:
    if array.dtype.hasobject:
        raise ValueError("official AlphaFold 2 parameter arrays must be numeric and pickle-free")
    writable = np.asarray(array)
    if not writable.flags.writeable:
        writable = writable.copy()
    return torch.from_numpy(writable)


def _split_official_param_key(key: str) -> tuple[str, str]:
    if key.count("//") != 1:
        raise ValueError(f"official AF2 parameter key must contain exactly one '//': {key!r}")
    scope, name = key.split("//", 1)
    if not scope or not name:
        raise ValueError(f"official AF2 parameter key must have non-empty scope and name: {key!r}")
    return scope, name


def _validate_official_param_keys(params: Mapping[str, Tensor]) -> None:
    if not params:
        raise ValueError("official AlphaFold 2 parameter mapping is empty")
    for key in params:
        _split_official_param_key(key)
    if not any(key.startswith("alphafold/alphafold_iteration/") for key in params):
        raise ValueError("expected official AlphaFold 2 params with 'alphafold/alphafold_iteration/' keys")


def _validate_supported_model_package(loaded: _LoadedOfficialParams) -> None:
    family = _model_family(loaded.model_name)
    if family == "multimer":
        raise NotImplementedError(
            "official AF-Multimer checkpoints use a distinct feature/model surface; "
            "convert monomer or monomer pTM AF2 checkpoints until AF-Multimer is implemented explicitly"
        )


def _validate_converted_state_dict_matches_model(
    converted: Mapping[str, Tensor],
    expected: Mapping[str, Tensor],
    *,
    model_name: str,
) -> None:
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    shape_mismatches = sorted(
        key for key in set(expected) & set(converted) if tuple(expected[key].shape) != tuple(converted[key].shape)
    )
    if not missing and not unexpected and not shape_mismatches:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing={len(missing)} examples={missing[:5]}")
    if unexpected:
        details.append(f"unexpected={len(unexpected)} examples={unexpected[:5]}")
    if shape_mismatches:
        examples = [
            f"{key}: expected {tuple(expected[key].shape)}, got {tuple(converted[key].shape)}"
            for key in shape_mismatches[:5]
        ]
        details.append(f"shape_mismatches={len(shape_mismatches)} examples={examples}")
    raise ValueError(
        f"official AF2 {model_name!r} converted tensors do not match AlphaFold2Model; "
        f"refusing to write a partial runnable model artifact ({'; '.join(details)})"
    )


def _conversion_config(
    loaded: _LoadedOfficialParams,
    persisted: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    return {
        "architecture": "alphafold2",
        "artifact_kind": "model",
        "model_name": loaded.model_name,
        "model_family": _model_family(loaded.model_name),
        "source_path": str(loaded.source_path),
        "source_archive_member": loaded.archive_member,
        "checkpoint_source": "deepmind",
        "checkpoint_format": "official_alphafold2_npz",
        "source_release": OFFICIAL_AF2_PARAMS_URL,
        "official_archive": OFFICIAL_AF2_PARAMS_ARCHIVE,
        "official_code_license": OFFICIAL_AF2_CODE_LICENSE,
        "official_params_license": OFFICIAL_AF2_PARAMS_LICENSE,
        "converted_weights": CONVERTED_WEIGHTS_NAME,
        "linear_weight_layout": "haiku_dense_attention_outer_product_layouts_to_pytorch",
        "tensor_count": len(persisted),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(persisted),
        "model_config": _model_config_overrides(loaded.model_name),
        "mapping_summary": _mapping_summary_from_keys(loaded.tensors),
        "runnable_model": True,
    }


def _write_processor_component(output_path: Path) -> None:
    from ..processor import AlphaFold2Processor

    AlphaFold2Processor().save_pretrained(output_path / "processor")


def _update_manifest_metadata(output_path: Path) -> None:
    manifest_path = output_path / ARTIFACT_MANIFEST_NAME
    manifest = Config.from_json(manifest_path)
    files = dict(manifest.get("files", {}))
    files["conversion_config"] = CONVERSION_CONFIG_NAME
    manifest["files"] = files
    components = dict(manifest.get("components", {}))
    components["processor"] = {"path": "processor"}
    manifest["components"] = components
    manifest.json(manifest_path)


def _model_family(model_name: str) -> str:
    if "_multimer" in model_name:
        return "multimer"
    if model_name.endswith("_ptm"):
        return "monomer_ptm"
    return "monomer"


def _model_config_overrides(model_name: str) -> dict[str, object]:
    pae_enabled = _model_family(model_name) != "monomer"
    template_enabled = _model_index(model_name) <= 2
    return {
        "heads": {"predicted_aligned_error": {"enabled": pae_enabled}},
        "template": {"enabled": template_enabled},
    }


def _model_index(model_name: str) -> int:
    parts = model_name.split("_")
    if len(parts) < 2 or not parts[1].isdecimal():
        raise ValueError(f"unsupported AlphaFold 2 model name {model_name!r}")
    return int(parts[1])


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
