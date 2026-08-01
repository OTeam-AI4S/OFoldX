# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Official DeepMind AlphaFold 2 checkpoint conversion.

Official DeepMind AlphaFold 2 parameters are distributed as ``params_*.npz`` files in
``alphafold_params_2022-12-06.tar``. The archive keys follow the upstream Haiku
flat-parameter convention ``scope//name``. This module handles the official package
shape end to end: it finds the selected model, loads the NPZ without pickle,
normalizes the flat Haiku names into OFoldX-style tensor keys, converts common
Haiku dense/attention/outer-product layouts into PyTorch-style layouts and strict-loads
them into the PyTorch AF2 model.

Monomer and pTM parameters strict-load into ``AlphaFold2Model``. AF-Multimer
checkpoints use their distinct feature/model graph (below).
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

from ....errors import RequestRef, SourceRef, fail
from ...vocabulary import canonical_word_embeddings
from ..config import AlphaFold2Config, AlphaFold2MsaConfig, AlphaFold2MultimerConfig
from ..model import AlphaFold2Model, AlphaFold2MultimerModel
from ..vocabulary import SOURCE_MASKED_MSA_ALPHABET, SOURCE_MSA_ALPHABET

_SOURCE_RESIDUES = SOURCE_MSA_ALPHABET[:-1]  # the upstream target one-hot: 20 amino acids and X


def _canonical_columns(weight: Tensor, vocabulary: tuple[str, ...]) -> Tensor:
    return canonical_word_embeddings(weight, vocabulary=vocabulary, dim=1)[0]


def convert_alphafold2_msa_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Regenerate the released input columns on the 72 canonical classes: the clustered-MSA one-hot and
    profile (``ARNDCQEGHILKMFPSTWYVX-`` and the mask), the extra-MSA one-hot, and the target one-hot of the token and
    pair projections (after the monomer's chain-break column). The masked-MSA head keeps its upstream rows.
    """

    converted = dict(state_dict)
    msa_key = "input_embeddings.msa_projection.weight"
    extra_key = "extra_msa_projection.weight"
    msa, extra = converted[msa_key], converted[extra_key]
    width = len(SOURCE_MASKED_MSA_ALPHABET)
    converted[msa_key] = torch.cat(
        (
            _canonical_columns(msa[:, :width], SOURCE_MASKED_MSA_ALPHABET),
            msa[:, width : width + 2],
            _canonical_columns(msa[:, width + 2 : 2 * width + 2], SOURCE_MASKED_MSA_ALPHABET),
            msa[:, 2 * width + 2 :],
        ),
        dim=1,
    ).contiguous()
    converted[extra_key] = torch.cat(
        (_canonical_columns(extra[:, :width], SOURCE_MASKED_MSA_ALPHABET), extra[:, width:]), dim=1
    ).contiguous()
    for name in ("token_projection", "pair_left_projection", "pair_right_projection"):
        key = f"input_embeddings.{name}.weight"
        weight = converted[key]
        prefix = weight.shape[1] - len(_SOURCE_RESIDUES)  # the monomer's chain-break column; none in the multimer
        converted[key] = torch.cat(
            (weight[:, :prefix], _canonical_columns(weight[:, prefix:], _SOURCE_RESIDUES)), dim=1
        ).contiguous()
    return converted


_MODEL_NAME_RE = re.compile(r"^model_[1-5](?:_ptm|_multimer(?:_v[23])?)?$")
_MONOMER_MSA_LIMITS = {
    "model_1": (508, 5120),
    "model_2": (508, 1024),
    "model_3": (512, 5120),
    "model_4": (512, 5120),
    "model_5": (512, 1024),
    "model_1_ptm": (508, 5120),
    "model_2_ptm": (508, 1024),
    "model_3_ptm": (512, 5120),
    "model_4_ptm": (512, 5120),
    "model_5_ptm": (512, 1024),
}
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
    "preprocess_1d": "token_projection",
    "preprocess_msa": "msa_projection",
    "left_single": "pair_left_projection",
    "right_single": "pair_right_projection",
    "relpos_projection": "relative_position_projection",
    # Official AF2 has this typo in the Haiku module name.
    "pair_activiations": "relative_position_projection",
}
_TEMPLATE_PROJECTION_SCOPE_MAP = {
    "embedding2d": "template_pair_projection",
    "template_pair_embedding_0": "distogram_projection",
    "template_pair_embedding_1": "pseudo_beta_mask_projection",
    "template_pair_embedding_2": "column_residue_type_projection",
    "template_pair_embedding_3": "row_residue_type_projection",
    "template_pair_embedding_4": "unit_vector_x_projection",
    "template_pair_embedding_5": "unit_vector_y_projection",
    "template_pair_embedding_6": "unit_vector_z_projection",
    "template_pair_embedding_7": "backbone_pair_mask_projection",
    "template_pair_embedding_8": "query_pair_projection",
    "output_linear": "output_projection",
}
_STRUCTURE_PROJECTION_SCOPE_MAP = {
    "q_scalar": "query_scalar_projection",
    "kv_scalar": "key_value_scalar_projection",
    "q_point_local": "query_point_projection",
    "kv_point_local": "key_value_point_projection",
    "attention_2d": "pair_bias_projection",
    "q_scalar_projection": "query_scalar_projection",
    "k_scalar_projection": "key_scalar_projection",
    "v_scalar_projection": "value_scalar_projection",
    "q_point_projection": "query_point_projection",
    "k_point_projection": "key_point_projection",
    "v_point_projection": "value_point_projection",
}
_STRUCTURE_SCOPE_MAP = {
    "single_layer_norm": "single_norm",
    "pair_layer_norm": "pair_norm",
    "initial_projection": "input_projection",
    "attention_layer_norm": "attention_norm",
    "transition_layer_norm": "transition_norm",
}
_RECYCLING_SCOPE_MAP = {
    "prev_pos_linear": "previous_positions_projection",
    "prev_msa_first_row_norm": "previous_msa_first_row_norm",
    "prev_pair_norm": "previous_pair_norm",
}
_STACK_SCOPE_MAP = {
    "outer_product_mean": ("outer_product_mean",),
    "layer_norm_input": ("layer_norm_input",),
    "input_layer_norm": ("input_norm",),
    "output_layer_norm": ("output_norm",),
    "transition1": ("feed_forward", "input_projection"),
    "transition2": ("feed_forward", "output_projection"),
    "msa_row_attention_with_pair_bias": ("msa_stack", "msa_row_attention_with_pair_bias"),
    "msa_column_attention": ("msa_stack", "msa_column_attention"),
    "msa_column_global_attention": ("msa_stack", "msa_column_global_attention"),
    "msa_transition": ("msa_stack", "transition"),
    "triangle_multiplication_outgoing": ("pair_stack", "triangle_multiplication_outgoing"),
    "triangle_multiplication_incoming": ("pair_stack", "triangle_multiplication_incoming"),
    "triangle_attention_starting_node": ("pair_stack", "triangle_attention_starting_node"),
    "triangle_attention_ending_node": ("pair_stack", "triangle_attention_ending_node"),
    "pair_transition": ("pair_stack", "transition"),
    "query_norm": ("query_norm",),
    "feat_2d_norm": ("pair_norm",),
    "layer_norm": ("layer_norm",),
    "center_layer_norm": ("center_layer_norm",),
    "left_gate": ("left_gate",),
    "right_gate": ("right_gate",),
    "gating_linear": ("gating_linear",),
    "attention": ("attention", "projection"),
    "single_template_embedding": ("single_template_embedder",),
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
        InvalidRequestError: ``value_invalid`` if the path/model selection does not identify exactly one
            official parameter file or the path is not an official AF2 params file.
    """

    params, _ = _load_official_alphafold2_params(params_path, model_name=model_name)
    return params


def convert_deepmind_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    msa_config: AlphaFold2MsaConfig | None = None,
) -> AlphaFold2Model | AlphaFold2MultimerModel:
    r"""
    Convert official AlphaFold 2 parameters into a strict-loaded native model.

    Args:
        checkpoint_path: Official AF2 NPZ, params directory/download root, or tar archive.
        model_name: Official model id when ``checkpoint_path`` contains multiple models.
        dtype: Optional dtype for floating-point converted tensors.
        msa_config: Optional monomer MSA feature policy applied before Model construction.

    Returns:
        The converted, strict-loaded model.
    """

    params, resolved_model_name = _load_official_alphafold2_params(checkpoint_path, model_name=model_name)
    if _model_family(resolved_model_name) == "multimer":
        if msa_config is not None:
            fail("value_invalid", RequestRef("msa_config", resolved_model_name))
        # AF-Multimer uses a separate registered model/processor pair with the shared trunk;
        # its converter builds AlphaFold2MultimerConfig/AlphaFold2MultimerModel and
        # applies the multimer key normalization.
        return convert_deepmind_multimer_checkpoint(
            params,
            model_name=resolved_model_name,
            dtype=dtype,
        )
    converted = convert_official_alphafold2_state_dict(params, dtype=dtype)

    model_config = AlphaFold2Config(_model_config_overrides(resolved_model_name))
    if msa_config is not None:
        model_config.msa = msa_config
    with torch.device("meta"):
        model = AlphaFold2Model(model_config)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
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
        ContradictionError: ``checkpoint_invalid`` if a source key is malformed or two source keys collide after
            normalization.
    """

    converted: dict[str, Tensor] = {}
    for source_key, tensor in params.items():
        target_key = normalize_official_param_key(source_key)
        target = convert_official_tensor(source_key, tensor)
        if dtype is not None and target.is_floating_point():
            target = target.to(dtype=dtype)
        expanded = _expand_stacked_tensor(source_key, target_key, tensor, target)
        for expanded_key, expanded_tensor in expanded.items():
            if expanded_key in converted:
                fail(
                    "checkpoint_invalid", SourceRef("checkpoint", 0, item=expanded_key)
                )  # several official tensors map to it
            converted[expanded_key] = expanded_tensor
    return convert_alphafold2_msa_state_dict(converted)


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
    """

    scope, name = _split_official_param_key(key)
    parts = [part for part in scope.split("/") if part]
    if parts[:2] == ["alphafold", "alphafold_iteration"]:
        parts = parts[2:]
    elif parts[:1] == ["alphafold"]:
        parts = parts[1:]
    normalized_parts = _normalize_scope_parts(parts)
    combined = _rename_attention_children([*normalized_parts, *_normalize_parameter_parts(name)])
    return ".".join(combined)


def _expand_stacked_tensor(
    source_key: str,
    target_key: str,
    source_tensor: Tensor,
    converted_tensor: Tensor,
) -> dict[str, Tensor]:
    if not _is_stacked_tensor(source_key, target_key, source_tensor):
        return {target_key: converted_tensor}
    if ".template_pair_stack.stacked.pair_stack." in target_key:
        marker = ".template_pair_stack.stacked.pair_stack."
        replacement = ".template_pair_stack.blocks.{index}."
    elif ".template_embedding_iteration.blocks.stacked.pair_stack." in target_key:
        marker = ".template_embedding_iteration.blocks.stacked.pair_stack."
        replacement = ".template_pair_stack.blocks.{index}."
    elif ".blocks.stacked." in target_key:
        marker = ".blocks.stacked."
        replacement = ".blocks.{index}."
        target_key = target_key.replace(".msa_stack.", ".msa_block.").replace(".pair_stack.", ".pair_block.")
    elif ".layers.stacked." in target_key:
        marker = ".layers.stacked."
        replacement = ".layers.{index}."
    else:
        return {target_key: converted_tensor}
    return {
        target_key.replace(marker, replacement.format(index=index)): converted_tensor[index].contiguous()
        for index in range(converted_tensor.shape[0])
    }


def _is_stacked_tensor(source_key: str, target_key: str, tensor: Tensor) -> bool:
    if not any(
        marker in target_key
        for marker in (
            ".layers.stacked.",
            ".blocks.stacked.",
            ".template_pair_stack.stacked.pair_stack.",
            ".template_embedding_iteration.blocks.stacked.pair_stack.",
        )
    ):
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
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item="scope"))
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
        return ["evoformer_stack"]
    input_embeddings_scope = _INPUT_EMBEDDING_SCOPE_MAP.get(parts[0])
    if input_embeddings_scope is not None:
        return ["input_embeddings", input_embeddings_scope, *_normalize_stack_scope(parts[1:])]
    recycling_scope = _RECYCLING_SCOPE_MAP.get(parts[0])
    if recycling_scope is not None:
        return ["recycling_embeddings", recycling_scope, *_normalize_stack_scope(parts[1:])]
    if parts[0] in {"extra_msa_activations", "extra_msa_projection"}:
        return ["extra_msa_projection", *_normalize_stack_scope(parts[1:])]
    if parts[0] == "single_activations":
        return ["single_projection", *_normalize_stack_scope(parts[1:])]
    if parts[0] in {"template_single_embedding", "template_projection"}:
        projection = {
            "template_single_embedding": "input_projection",
            "template_projection": "output_projection",
        }[parts[0]]
        return ["template_msa_embedder", "feed_forward", projection, *_normalize_stack_scope(parts[1:])]
    if parts[0] == "template_embedding":
        return ["template_embedder", *_normalize_stack_scope(parts[1:])]
    if parts[0] == "evoformer_iteration":
        return ["evoformer_stack", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    if parts[0] == "extra_msa_iteration":
        return ["extra_msa_stack", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    if parts[0] == "extra_msa_stack":
        return ["extra_msa_stack", "blocks", "stacked", *_normalize_evoformer_block_scope(parts[1:])]
    return ["evoformer_stack", *_normalize_stack_scope(parts)]


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
    for index in range(len(normalized) - 2):
        if normalized[index : index + 3] == ["msa_column_global_attention", "attention", "projection"]:
            del normalized[index + 2]
            break
    return _rename_outer_product_mean_norm(normalized)


def _normalize_stack_scope(parts: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in parts:
        if part == "__layer_stack_no_state":
            normalized.append("stacked")
            continue
        template_projection = _TEMPLATE_PROJECTION_SCOPE_MAP.get(part)
        if template_projection is not None:
            normalized.append(template_projection)
            continue
        mapped = _STACK_SCOPE_MAP.get(part)
        if mapped is None:
            normalized.append(part)
        else:
            normalized.extend(mapped)
    return _rename_outer_product_mean_norm(normalized)


def _rename_outer_product_mean_norm(normalized: list[str]) -> list[str]:
    # The outer-product mean now subclasses the shared ``OuterProductMean``, whose pre-LayerNorm is
    # registered as ``msa_norm``; the official Haiku scope names it ``layer_norm_input`` (the same
    # name the triangle-multiplication norm uses, which must keep its ``layer_norm_input`` target).
    for index in range(1, len(normalized)):
        if normalized[index] == "layer_norm_input" and normalized[index - 1] == "outer_product_mean":
            normalized[index] = "msa_norm"
    return _rename_triangle_multiplication_children(normalized)


# Child-leaf renames for the unified ``TriangleMultiplication``: the shared bias-free module names its
# submodules ``input_norm``/``output_norm``/``left_gate_projection``/``right_gate_projection``/
# ``gate_projection`` (the official Haiku scope uses ``layer_norm_input``/``center_layer_norm``/
# ``left_gate``/``right_gate``/``gating_linear``). For the *incoming* update the shared module fixes the
# canonical einsum ``"...kih,...kjh->...ijh"`` (left owns row ``i``, right owns column ``j``), so the
# left/right value and gate projections must swap to reproduce DeepMind's own ``"...kjc,...kic->..."``
# incoming contraction. Outgoing loads straight through.
_TRIANGLE_MULTIPLICATION_CHILD_MAP = {
    "layer_norm_input": "input_norm",
    "center_layer_norm": "output_norm",
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


# Child renames for native attention state trees. Source modules reuse
# ``feat_2d_weights`` for different pair-bias projections, so parent scope owns the mapping.
_TRIANGLE_ATTENTION_PARENTS = frozenset({"triangle_attention_starting_node", "triangle_attention_ending_node"})
_TRIANGLE_ATTENTION_CHILD_MAP = {"query_norm": "pair_norm", "feat_2d_weights": "bias_projection"}
_MSA_ROW_ATTENTION_CHILD_MAP = {"feat_2d_weights": "pair_bias_projection"}


def _rename_attention_children(parts: list[str]) -> list[str]:
    for index in range(1, len(parts)):
        parent = parts[index - 1]
        if parent in _TRIANGLE_ATTENTION_PARENTS:
            parts[index] = _TRIANGLE_ATTENTION_CHILD_MAP.get(parts[index], parts[index])
        elif parent == "msa_row_attention_with_pair_bias":
            parts[index] = _MSA_ROW_ATTENTION_CHILD_MAP.get(parts[index], parts[index])
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
        structure_scope = _STRUCTURE_SCOPE_MAP.get(part)
        if structure_scope is not None:
            normalized.append(structure_scope)
            continue
        projection = _STRUCTURE_PROJECTION_SCOPE_MAP.get(part)
        if projection is not None:
            normalized.append(projection)
            continue
        if part == "fold_iteration":
            normalized.append("block")
            continue
        if part == "affine_update":
            normalized.append("backbone_update")
            continue
        if part == "quat_rigid":
            normalized.append("backbone_update")
            continue
        if part == "rigid" and normalized[-1:] == ["backbone_update"]:
            normalized.append("projection")
            continue
        if part == "rigid_sidechain":
            normalized.append("sidechain")
            continue

        if part == "unnormalized_angles" and "sidechain" in normalized:
            normalized.append("angle_projection")
            continue

        transition_index = _repeated_haiku_index(part, "transition")
        if transition_index is not None and "block" in normalized and "sidechain" not in normalized:
            transition_projection = (
                ("input_projection",)
                if transition_index == 0
                else (
                    ("output_projection",)
                    if transition_index == 2
                    else ("intermediate_projections", str(transition_index - 1))
                )
            )
            normalized.extend(("transition", *transition_projection))
            continue

        input_projection_index = _repeated_haiku_index(part, "input_projection")
        if input_projection_index is not None and "sidechain" in normalized:
            normalized.extend(("input_projections", str(input_projection_index)))
            continue

        resblock1_index = _repeated_haiku_index(part, "resblock1")
        if resblock1_index is not None and "sidechain" in normalized:
            normalized.extend(("blocks", str(resblock1_index), "feed_forward", "input_projection"))
            continue

        resblock2_index = _repeated_haiku_index(part, "resblock2")
        if resblock2_index is not None and "sidechain" in normalized:
            normalized.extend(("blocks", str(resblock2_index), "feed_forward", "output_projection"))
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
        return ["distogram_head", "projection"]
    if parts[0] in {"experimentally_resolved_head", "masked_msa_head", "predicted_aligned_error_head"} and parts[
        1:2
    ] == ["logits"]:
        return [parts[0], *parts[2:]]
    if parts[:2] == ["predicted_lddt_head", "act_0"]:
        return ["predicted_lddt_head", "feed_forward", "input_projection", *parts[2:]]
    if parts[:2] == ["predicted_lddt_head", "act_1"]:
        return ["predicted_lddt_head", "feed_forward", "intermediate_projections", "0", *parts[2:]]
    if parts[:2] == ["predicted_lddt_head", "input_layer_norm"]:
        return ["predicted_lddt_head", "input_norm", *parts[2:]]
    if parts[:2] == ["predicted_lddt_head", "logits"]:
        return ["predicted_lddt_head", "feed_forward", "output_projection", *parts[2:]]
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
                fail("checkpoint_invalid", SourceRef(str(source_path), 0, item=member.name))
            payload = extracted.read()
        resolved_model_name = _model_name_from_param_filename(Path(member.name).name)
        yield io.BytesIO(payload), resolved_model_name
        return
    fail("value_invalid", RequestRef("checkpoint_path", str(source_path)))


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
        fail("value_invalid", RequestRef("checkpoint_path", str(source_path)), parameter_files=len(matches))
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
        archive_name = archive.name.decode() if isinstance(archive.name, bytes) else str(archive.name)
        fail("value_invalid", RequestRef("checkpoint_path", archive_name), parameter_files=len(members))
    return members[0]


def _model_name_from_param_filename(filename: str) -> str:
    if not filename.startswith("params_") or not filename.endswith(".npz"):
        fail("value_invalid", RequestRef("checkpoint_path", filename))
    return filename[len("params_") : -len(".npz")]


def _validate_model_name(model_name: str) -> None:
    if not _MODEL_NAME_RE.fullmatch(model_name):
        fail("value_invalid", RequestRef("model_name", model_name))


def _split_official_param_key(key: str) -> tuple[str, str]:
    if key.count("//") != 1:
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))
    scope, name = key.split("//", 1)
    if not scope or not name:
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))
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
    max_msa_clusters, max_extra_msa = _MONOMER_MSA_LIMITS[model_name]
    return {
        "heads": {"predicted_aligned_error": {"enabled": pae_enabled}},
        "msa": {
            "max_msa_clusters": max_msa_clusters,
            "max_extra_msa": max_extra_msa,
        },
        "template": {"enabled": template_enabled},
    }


def _model_index(model_name: str) -> int:
    parts = model_name.split("_")
    if len(parts) < 2 or not parts[1].isdecimal():
        fail("value_invalid", RequestRef("model_name", model_name))
    return int(parts[1])


# Official DeepMind AlphaFold-Multimer checkpoint conversion.
#
# AF-Multimer parameters ship as ``params_model_{1..5}_multimer_v3.npz`` inside
# ``alphafold_params_2022-12-06.tar``. They share the flat Haiku ``scope//name``
# convention with the monomer parameters, so the multimer path reuses the monomer tensor
# layout conversion and stacked-tensor expansion and only overrides the scope
# normalization that differs on the multimer surface: the chain-relative input
# embedder (``position_activations``), the multimer template embedder, the source
# 22-class masked-MSA head, and the interface-TM head.
#
# VALIDATED: the official multimer Haiku module names for the chain-relative input
# embedder and the multimer template embedder are confirmed against the real
# ``params_model_1_multimer_v3`` release -- ``load_state_dict(strict=True, assign=True)``
# loads all 5062 native parameters from the 330 official tensors with zero missing,
# unexpected, or shape-mismatched keys, and the converted model reproduces the OpenFold
# >= 2.0 multimer forward to fp32 round-off (single / pair rel-L2 ~3e-7 at matched recycle
# count). ``load_state_dict(strict=True, assign=True)`` remains the sole key/shape authority.

_MULTIMER_INPUT_EMBEDDING_SCOPE_MAP = {
    "position_activations": "relative_position_projection",
    "preprocess_1d": "token_projection",
    "preprocess_msa": "msa_projection",
    "left_single": "pair_left_projection",
    "right_single": "pair_right_projection",
}
_MULTIMER_MSA_LIMITS = {
    "model_1_multimer": (252, 1152),
    "model_2_multimer": (252, 1152),
    "model_3_multimer": (252, 1152),
    "model_4_multimer": (252, 1152),
    "model_5_multimer": (252, 1152),
    "model_1_multimer_v2": (252, 1152),
    "model_2_multimer_v2": (252, 1152),
    "model_3_multimer_v2": (252, 1152),
    "model_4_multimer_v2": (252, 1152),
    "model_5_multimer_v2": (252, 1152),
    "model_1_multimer_v3": (508, 2048),
    "model_2_multimer_v3": (508, 2048),
    "model_3_multimer_v3": (508, 2048),
    "model_4_multimer_v3": (508, 1152),
    "model_5_multimer_v3": (508, 1152),
}


def convert_deepmind_multimer_checkpoint(
    params: Mapping[str, Tensor],
    *,
    model_name: str,
    dtype: torch.dtype | None = None,
) -> AlphaFold2MultimerModel:
    r"""
    Convert official AlphaFold-Multimer parameters into a strict-loaded native model.
    """

    converted = convert_official_multimer_state_dict(params, dtype=dtype)
    model_config = AlphaFold2MultimerConfig(_multimer_config_overrides(model_name))
    with torch.device("meta"):
        model = AlphaFold2MultimerModel(model_config)
    model.load_state_dict(converted, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def convert_official_multimer_state_dict(
    params: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Normalize official flat Haiku AF-Multimer parameter names to native tensor keys.
    """

    converted: dict[str, Tensor] = {}
    for source_key, tensor in params.items():
        target_key = normalize_multimer_param_key(source_key)
        target = _convert_multimer_tensor(source_key, tensor)
        if dtype is not None and target.is_floating_point():
            target = target.to(dtype=dtype)
        expanded = _expand_stacked_tensor(source_key, target_key, tensor, target)
        for expanded_key, expanded_tensor in expanded.items():
            for final_key, final_tensor in _split_multimer_tensor(expanded_key, expanded_tensor):
                if final_key in converted:
                    fail(
                        "checkpoint_invalid", SourceRef("checkpoint", 0, item=final_key)
                    )  # several official AF-Multimer tensors map to it
                converted[final_key] = final_tensor
    return convert_alphafold2_msa_state_dict(converted)


# AF-Multimer fuses the two operands of every ``TriangleMultiplication`` into a single
# ``projection`` (2c, in) and ``gate`` (2c, in) linear, whereas the native (monomer)
# ``TriangleMultiplication`` keeps unfused ``left_projection`` / ``right_projection`` and
# ``left_gate`` / ``right_gate`` (c, in). DeepMind's fused kernel splits the projected
# activation as ``proj[..., :c]`` (left/a operand) and ``proj[..., c:]`` (right/b operand)
# and applies the same layer norms and gating, so recovering the unfused native layout is a
# pure row split of the fused output dimension -- mathematically identical, no model change.
# The two LayerNorms are also renamed (``left_norm_input`` -> ``input_norm``, ``center_norm`` ->
# ``output_norm``, the shared module's names); the fused-operand ``gating_linear`` output gate was
# already renamed to the shared ``gate_projection`` by the monomer scope normalizer's
# triangle-multiplication post-pass, and ``output_projection`` shares its name. This applies uniformly
# to the Evoformer, extra-MSA, and multimer template pair stacks, all of which reuse the native module.
#
# The unified ``TriangleMultiplication`` fixes the operand order (left owns row ``i``, right owns
# column ``j``) with the canonical incoming einsum ``"...kih,...kjh->...ijh"``. DeepMind's fused
# ``projection``/``gate`` split ``proj[:c]`` = operand ``a`` and ``proj[c:]`` = operand ``b``; the
# outgoing update loads ``a``->left, ``b``->right, while the incoming update swaps ``a``->right,
# ``b``->left (value and gate together) to reproduce DeepMind's ``"...kjc,...kic->..."`` incoming
# contraction.
_TRIANGLE_MULTIPLICATION_SCOPES = (
    "triangle_multiplication_outgoing",
    "triangle_multiplication_incoming",
)
_TRIANGLE_RENAME = {"left_norm_input": "input_norm", "center_norm": "output_norm"}
_TRIANGLE_SPLIT_OUTGOING = {
    "projection": ("left_projection", "right_projection"),
    "gate": ("left_gate_projection", "right_gate_projection"),
}
_TRIANGLE_SPLIT_INCOMING = {
    "projection": ("right_projection", "left_projection"),
    "gate": ("right_gate_projection", "left_gate_projection"),
}


def _split_multimer_tensor(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Expand one fused AF-Multimer triangle-multiplication tensor into native keys.

    Non triangle-multiplication tensors and already-native sub-modules pass through
    unchanged. ``projection`` / ``gate`` weights ``(2c, in)`` and biases ``(2c,)`` split
    on the leading output dimension into ``a``/``b`` halves (assigned left/right per the update
    direction, swapped for the incoming update); the two LayerNorms rename.
    """

    parts = key.split(".")
    if len(parts) < 3 or parts[-3] not in _TRIANGLE_MULTIPLICATION_SCOPES:
        return [(key, tensor)]
    module, param = parts[-2], parts[-1]  # e.g. ("projection", "weight")
    prefix = ".".join(parts[:-2])
    rename = _TRIANGLE_RENAME.get(module)
    if rename is not None:
        return [(f"{prefix}.{rename}.{param}", tensor)]
    split_map = _TRIANGLE_SPLIT_INCOMING if parts[-3].endswith("_incoming") else _TRIANGLE_SPLIT_OUTGOING
    split = split_map.get(module)
    if split is None:
        return [(key, tensor)]
    if tensor.shape[0] % 2 != 0:
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))
    half = tensor.shape[0] // 2
    first, second = split
    return [
        (f"{prefix}.{first}.{param}", tensor[:half].contiguous()),
        (f"{prefix}.{second}.{param}", tensor[half:].contiguous()),
    ]


def _convert_multimer_tensor(source_key: str, tensor: Tensor) -> Tensor:
    r"""
    Convert one official multimer tensor, handling the multimer IPA projections.

    The AF-Multimer invariant-point-attention scalar/point projections store weights as
    ``(c_s, num_heads, out)`` Haiku tensors that flatten to ``(num_heads * out, c_s)`` in the
    native ``Linear`` layout, and their biases as ``(num_heads, out)`` flattening to
    ``(num_heads * out,)``. Every other tensor uses the shared monomer conversion.
    """

    _, name = _split_official_param_key(source_key)
    is_scalar_projection = source_key.endswith("_scalar_projection//weights")
    is_point_projection = "_point_projection/point_projection" in source_key
    if is_scalar_projection and tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0], -1).transpose(0, 1).contiguous()
    if is_point_projection and name == "weights" and tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0], -1).transpose(0, 1).contiguous()
    if is_point_projection and name == "bias" and tensor.ndim == 2:
        return tensor.reshape(-1).contiguous()
    # AF-Multimer's single-input template pair embedders (pseudo-beta mask, backbone unit
    # vector x/y/z, backbone-pair mask) store their weights with the singleton input axis
    # squeezed to shape ``(c_out,)``; restore the native ``Linear`` ``(c_out, 1)`` layout.
    if "template_pair_embedding_" in source_key and name == "weights" and tensor.ndim == 1:
        return tensor.reshape(-1, 1).contiguous()
    return convert_official_tensor(source_key, tensor)


def normalize_multimer_param_key(key: str) -> str:
    r"""
    Return the normalized native key for one official AF-Multimer flat Haiku key.

    Only the chain-relative input embedder needs multimer-specific handling here;
    every other scope reuses the monomer normalization, which already maps the
    template embedder, evoformer/extra-MSA stacks, structure module, and heads.
    """

    scope, name = _split_official_param_key(key)
    parts = [part for part in scope.split("/") if part]
    if parts[:2] == ["alphafold", "alphafold_iteration"]:
        parts = parts[2:]
    elif parts[:1] == ["alphafold"]:
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == "evoformer":
        # The chain-relative positional encoder ships under a Haiku ``~_relative_encoding``
        # sub-scope (``evoformer/~_relative_encoding/position_activations``); the other
        # input-embedder linears sit directly under ``evoformer``.
        embedder_part = parts[2] if parts[1] == "~_relative_encoding" and len(parts) >= 3 else parts[1]
        mapped = _MULTIMER_INPUT_EMBEDDING_SCOPE_MAP.get(embedder_part)
        if mapped is not None:
            return ".".join(["input_embeddings", mapped, *_normalize_parameter_parts(name)])
    normalized = normalize_official_param_key(key)
    # The multimer template embedder's per-iteration pair stack ships as a bare Haiku
    # ``hk.layer_stack`` (leading layer dimension, no ``__layer_stack_no_state`` scope marker), so
    # the shared normalizer maps it to ``template_embedding_iteration.pair_stack.*`` without a
    # ``blocks.stacked`` marker. Insert it so ``_expand_stacked_tensor`` unrolls
    # the two source blocks into native ``template_pair_stack.blocks.{0,1}.*`` keys.
    if ".template_embedding_iteration.pair_stack." in normalized:
        normalized = normalized.replace(
            ".template_embedding_iteration.pair_stack.",
            ".template_embedding_iteration.blocks.stacked.pair_stack.",
        )
    return normalized


def _multimer_config_overrides(model_name: str) -> dict[str, object]:
    r"""
    Return the released MSA limits, template path, and interface-TM head.
    """

    max_msa_clusters, max_extra_msa = _MULTIMER_MSA_LIMITS[model_name]
    return {
        "heads": {"predicted_aligned_error": {"enabled": True}},
        "msa": {
            "max_msa_clusters": max_msa_clusters,
            "max_extra_msa": max_extra_msa,
        },
        "template": {"enabled": True},
    }
