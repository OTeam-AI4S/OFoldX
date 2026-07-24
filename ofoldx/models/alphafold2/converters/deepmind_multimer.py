r"""
Official DeepMind AlphaFold-Multimer checkpoint conversion.

AF-Multimer parameters ship as ``params_model_{1..5}_multimer_v3.npz`` inside
``alphafold_params_2022-12-06.tar``. They share the flat Haiku ``scope//name``
convention with the monomer parameters, so this module reuses the monomer tensor
layout conversion and stacked-tensor expansion and only overrides the scope
normalization that differs on the multimer surface: the chain-relative input
embedder (``position_activations``), the multimer template embedder, the 22-class
masked-MSA head, and the interface-TM head.

VALIDATED: the official multimer Haiku module names for the chain-relative input
embedder and the multimer template embedder are confirmed against the real
``params_model_1_multimer_v3`` release -- ``load_state_dict(strict=True, assign=True)``
loads all 5062 native parameters from the 330 official tensors with zero missing,
unexpected, or shape-mismatched keys, and the converted model reproduces the OpenFold
>= 2.0 multimer forward to fp32 round-off (single / pair rel-L2 ~3e-7 at matched recycle
count). ``load_state_dict(strict=True, assign=True)`` remains the sole key/shape authority.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ..config import AlphaFold2MultimerConfig
from ..model import AlphaFold2Model, AlphaFold2MultimerModel
from . import deepmind

_MULTIMER_INPUT_EMBEDDING_SCOPE_MAP = {
    "position_activations": "relative_position_projection",
    "preprocess_1d": "token_projection",
    "preprocess_msa": "msa_projection",
    "left_single": "pair_left_projection",
    "right_single": "pair_right_projection",
}


def convert_deepmind_multimer_checkpoint(
    params: Mapping[str, Tensor],
    *,
    model_name: str,
    dtype: torch.dtype | None = None,
) -> AlphaFold2Model:
    r"""
    Convert official AlphaFold-Multimer parameters into a strict-loaded native model.
    """

    converted = convert_official_multimer_state_dict(params, dtype=dtype)
    model_config = AlphaFold2MultimerConfig(_multimer_config_overrides())
    with torch.device("meta"):
        model = AlphaFold2MultimerModel(model_config)
    model.load_state_dict(converted, strict=True, assign=True)
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
        expanded = deepmind._expand_stacked_tensor(source_key, target_key, tensor, target)
        for expanded_key, expanded_tensor in expanded.items():
            for final_key, final_tensor in _split_multimer_tensor(expanded_key, expanded_tensor):
                if final_key in converted:
                    raise ValueError(f"multiple official AF-Multimer tensors map to {final_key!r}")
                converted[final_key] = final_tensor
    return converted


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
# contraction. See TRUNK_EINSUM_LOCK.md.
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
        raise ValueError(f"fused AF-Multimer {module!r} tensor {key!r} has odd leading dim {tensor.shape[0]}")
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

    _, name = deepmind._split_official_param_key(source_key)
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
    return deepmind.convert_official_tensor(source_key, tensor)


def normalize_multimer_param_key(key: str) -> str:
    r"""
    Return the normalized native key for one official AF-Multimer flat Haiku key.

    Only the chain-relative input embedder needs multimer-specific handling here;
    every other scope reuses the monomer normalization, which already maps the
    template embedder, evoformer/extra-MSA stacks, structure module, and heads.
    """

    scope, name = deepmind._split_official_param_key(key)
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
            return ".".join(["input_embeddings", mapped, *deepmind._normalize_parameter_parts(name)])
    normalized = deepmind.normalize_official_param_key(key)
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


def _multimer_config_overrides() -> dict[str, object]:
    r"""
    AF-Multimer v3 always carries templates and the interface-TM head.
    """

    return {
        "heads": {"predicted_aligned_error": {"enabled": True}},
        "template": {"enabled": True},
    }
