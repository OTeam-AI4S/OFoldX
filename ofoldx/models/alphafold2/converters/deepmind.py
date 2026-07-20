r"""
Official DeepMind AlphaFold 2 checkpoint conversion.

Official DeepMind AlphaFold 2 parameters are distributed as ``params_*.npz`` files in
``alphafold_params_2022-12-06.tar``. The archive keys follow the upstream Haiku
flat-parameter convention ``scope//name``. This module handles the official package
shape end to end: it finds the selected model, loads the NPZ without pickle,
normalizes the flat Haiku names into OFoldX-style tensor keys, converts common
Haiku dense/attention/outer-product layouts into PyTorch-style layouts and strict-loads
them into the PyTorch AF2 model.

The current OFoldX ``AlphaFold2Model`` can run through an AF2-shaped forward path,
and official monomer / pTM parameters strict-load into the PyTorch parameter tree.
AF-Multimer checkpoints use a distinct feature/model surface and are rejected until
that architecture is implemented explicitly.
"""

from __future__ import annotations

import io
import re
import tarfile
from collections.abc import Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import torch
from torch import Tensor

from ..config import AlphaFold2Config
from ..model import AlphaFold2Model

_MODEL_NAME_RE = re.compile(r"^model_[1-5](?:_ptm|_multimer(?:_v[123])?)?$")
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

    params, _ = _load_official_alphafold2_params(params_path, model_name=model_name)
    return params


def convert_deepmind_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> AlphaFold2Model:
    r"""
    Convert official AlphaFold 2 parameters into a strict-loaded native model.

    Args:
        checkpoint_path: Official AF2 NPZ, params directory/download root, or tar archive.
        model_name: Official model id when ``checkpoint_path`` contains multiple models.
        dtype: Optional dtype for floating-point converted tensors.

    Returns:
        The converted, strict-loaded model.
    """

    params, resolved_model_name = _load_official_alphafold2_params(checkpoint_path, model_name=model_name)
    if _model_family(resolved_model_name) == "multimer":
        # AF-Multimer is a second registered pair in this family sharing the trunk;
        # its converter builds AlphaFold2MultimerConfig/AlphaFold2MultimerModel and
        # applies the multimer key normalization. Imported lazily to avoid a
        # converters<->model import cycle at module load.
        from . import deepmind_multimer

        return deepmind_multimer.convert_deepmind_multimer_checkpoint(
            params,
            model_name=resolved_model_name,
            dtype=dtype,
        )
    converted = convert_official_alphafold2_state_dict(params, dtype=dtype)

    model_config = AlphaFold2Config(_model_config_overrides(resolved_model_name))
    with torch.device("meta"):
        model = AlphaFold2Model(model_config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


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

    scope, name = _split_official_param_key(key)
    parts = [part for part in scope.split("/") if part]
    if parts[:2] == ["alphafold", "alphafold_iteration"]:
        parts = parts[2:]
    elif parts[:1] == ["alphafold"]:
        parts = parts[1:]
    normalized_parts = _normalize_scope_parts(parts)
    combined = _rename_triangle_attention_children([*normalized_parts, *_normalize_parameter_parts(name)])
    return ".".join(combined)


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


def official_param_filename(model_name: str) -> str:
    r"""
    Return the official NPZ filename for an AlphaFold 2 model id.
    """

    _validate_model_name(model_name)
    return f"params_{model_name}.npz"


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
    return _rename_outer_product_mean_norm(normalized)


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
    return _rename_outer_product_mean_norm(normalized)


def _rename_outer_product_mean_norm(normalized: list[str]) -> list[str]:
    # The outer-product mean now subclasses the shared ``OuterProductMean``, whose pre-LayerNorm is
    # registered as ``norm_msa``; the official Haiku scope names it ``layer_norm_input`` (the same
    # name the triangle-multiplication norm uses, which must keep its ``layer_norm_input`` target).
    for index in range(1, len(normalized)):
        if normalized[index] == "layer_norm_input" and normalized[index - 1] == "outer_product_mean":
            normalized[index] = "norm_msa"
    return _rename_triangle_multiplication_children(normalized)


# Child-leaf renames for the unified ``TriangleMultiplication``: the shared bias-free module names its
# submodules ``norm_input``/``norm_output``/``left_gate_projection``/``right_gate_projection``/
# ``gate_projection`` (the official Haiku scope uses ``layer_norm_input``/``center_layer_norm``/
# ``left_gate``/``right_gate``/``gating_linear``). For the *incoming* update the shared module fixes the
# canonical einsum ``"...kih,...kjh->...ijh"`` (left owns row ``i``, right owns column ``j``), so the
# left/right value and gate projections must swap to reproduce DeepMind's own ``"...kjc,...kic->..."``
# incoming contraction. Outgoing loads straight through. See TRUNK_EINSUM_LOCK.md.
_TRIANGLE_MULTIPLICATION_CHILD_MAP = {
    "layer_norm_input": "norm_input",
    "center_layer_norm": "norm_output",
    "left_projection": "left_projection",
    "right_projection": "right_projection",
    "left_gate": "left_gate_projection",
    "right_gate": "right_gate_projection",
    "output_projection": "output_projection",
    "gating_linear": "gate_projection",
}
_TRIANGLE_MULTIPLICATION_CHILD_MAP_SWAPPED = {
    **_TRIANGLE_MULTIPLICATION_CHILD_MAP,
    "left_projection": "right_projection",
    "right_projection": "left_projection",
    "left_gate": "right_gate_projection",
    "right_gate": "left_gate_projection",
}


# Child renames for the unified ``TriangleAttention`` (shared bias-free module + the AlphaFold 2
# subclass): its pre-attention LayerNorm is registered ``norm_pair`` and its per-head triangle-bias
# projection ``bias_projection`` (the official Haiku names are ``query_norm``/``feat_2d_weights``, which
# other modules — MSA row/column attention — keep). Scope the rename to a triangle-attention parent.
_TRIANGLE_ATTENTION_PARENTS = frozenset({"triangle_attention_starting_node", "triangle_attention_ending_node"})
_TRIANGLE_ATTENTION_CHILD_MAP = {"query_norm": "norm_pair", "feat_2d_weights": "bias_projection"}


def _rename_triangle_attention_children(parts: list[str]) -> list[str]:
    for index in range(1, len(parts)):
        if parts[index - 1] in _TRIANGLE_ATTENTION_PARENTS:
            parts[index] = _TRIANGLE_ATTENTION_CHILD_MAP.get(parts[index], parts[index])
    return parts


def _rename_triangle_multiplication_children(normalized: list[str]) -> list[str]:
    for index in range(len(normalized) - 1):
        parent = normalized[index]
        if parent == "triangle_multiplication_outgoing":
            child_map = _TRIANGLE_MULTIPLICATION_CHILD_MAP
        elif parent == "triangle_multiplication_incoming":
            child_map = _TRIANGLE_MULTIPLICATION_CHILD_MAP_SWAPPED
        else:
            continue
        normalized[index + 1] = child_map.get(normalized[index + 1], normalized[index + 1])
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


def _normalize_parameter_parts(name: str) -> tuple[str, ...]:
    if name == "feat_2d_weights":
        return ("feat_2d_weights", "weight")
    parameter_parts = _PARAMETER_NAME_MAP.get(name)
    if parameter_parts is not None:
        return parameter_parts
    return (name,)


def _load_official_alphafold2_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None,
) -> tuple[dict[str, Tensor], str]:
    source_path = Path(params_path)
    with (
        _open_official_npz(source_path, model_name=model_name) as (handle, resolved_model_name),
        np.load(handle, allow_pickle=False) as npz,
    ):
        tensors = {_key: torch.from_numpy(npz[_key]) for _key in npz.files}
    return tensors, resolved_model_name


@contextmanager
def _open_official_npz(
    source_path: Path,
    *,
    model_name: str | None,
) -> Iterator[tuple[BinaryIO, str]]:
    if model_name is not None:
        _validate_model_name(model_name)
    if source_path.is_dir():
        param_file, resolved_model_name = _select_directory_param_file(source_path, model_name=model_name)
        with param_file.open("rb") as handle:
            yield handle, resolved_model_name
        return
    if source_path.suffix == ".npz":
        resolved_model_name = model_name or _model_name_from_param_filename(source_path.name)
        _validate_model_name(resolved_model_name)
        with source_path.open("rb") as handle:
            yield handle, resolved_model_name
        return
    if source_path.is_file() and tarfile.is_tarfile(source_path):
        with tarfile.open(source_path) as archive:
            member = _select_archive_member(archive, model_name=model_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not extract {member.name!r} from {source_path}")
            payload = extracted.read()
        resolved_model_name = _model_name_from_param_filename(Path(member.name).name)
        yield io.BytesIO(payload), resolved_model_name
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
        raise ValueError(f"unsupported AlphaFold 2 model name {model_name!r}")


def _split_official_param_key(key: str) -> tuple[str, str]:
    if key.count("//") != 1:
        raise ValueError(f"official AF2 parameter key must contain exactly one '//': {key!r}")
    scope, name = key.split("//", 1)
    if not scope or not name:
        raise ValueError(f"official AF2 parameter key must have non-empty scope and name: {key!r}")
    return scope, name


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
