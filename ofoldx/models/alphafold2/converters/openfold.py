r"""
OpenFold AlphaFold 2 checkpoint conversion hook.

OpenFold checkpoints are not a separate OFoldX architecture. They are another
source format for the AlphaFold 2 architecture, so this module owns only the
source-specific load/key-normalization path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from ..config import AlphaFold2Config
from ..model import AlphaFold2Model

_BLOCK_KEY_RE = re.compile(r"^(evoformer|extra_msa_stack)\.blocks\.(\d+)\.(.+)$")
_TEMPLATE_BLOCK_KEY_RE = re.compile(r"^template_pair_stack\.blocks\.(\d+)\.(.+)$")
_ATTENTION_PROJECTION_MAP = {
    "linear_q": "query_projection",
    "linear_k": "key_projection",
    "linear_v": "value_projection",
    "linear_o": "output_projection",
    "linear_g": "gate_projection",
}
_TOP_LEVEL_PREFIX_MAP = {
    "input_embedder.linear_tf_m.": "input_embeddings.token_projection.",
    "input_embedder.linear_msa_m.": "input_embeddings.msa_projection.",
    "input_embedder.linear_tf_z_i.": "input_embeddings.pair_left_projection.",
    "input_embedder.linear_tf_z_j.": "input_embeddings.pair_right_projection.",
    "input_embedder.linear_relpos.": "input_embeddings.relative_position_projection.",
    "recycling_embedder.linear.": "recycling_embeddings.previous_positions_projection.",
    "recycling_embedder.layer_norm_m.": "recycling_embeddings.previous_msa_first_row_norm.",
    "recycling_embedder.layer_norm_z.": "recycling_embeddings.previous_pair_norm.",
    "extra_msa_embedder.linear.": "extra_msa_embeddings.extra_msa_projection.",
    "evoformer.linear.": "single_projection.",
    "template_angle_embedder.linear_1.": "template_msa_embedder.input_projection.",
    "template_angle_embedder.linear_2.": "template_msa_embedder.output_projection.",
    "template_pair_embedder.linear.": "template_embedder.single_template_embedder.template_pair_projection.",
    "template_pair_stack.layer_norm.": "template_embedder.single_template_embedder.output_norm.",
    "aux_heads.distogram.linear.": "distogram_head.projection.",
    "aux_heads.masked_msa.linear.": "masked_msa_head.logits.",
    "aux_heads.experimentally_resolved.linear.": "experimentally_resolved_head.logits.",
    "aux_heads.tm.linear.": "predicted_aligned_error_head.logits.",
    "aux_heads.plddt.layer_norm.": "predicted_lddt_head.input_norm.",
    "aux_heads.plddt.linear_1.": "predicted_lddt_head.input_projection.",
    "aux_heads.plddt.linear_2.": "predicted_lddt_head.hidden_projection.",
    "aux_heads.plddt.linear_3.": "predicted_lddt_head.logits.",
}
_STRUCTURE_PREFIX_MAP = {
    "structure_module.layer_norm_s.": "structure_module.single_norm.",
    "structure_module.linear_in.": "structure_module.input_projection.",
    "structure_module.layer_norm_z.": "structure_module.pair_norm.",
    "structure_module.ipa.head_weights": ("structure_module.block.invariant_point_attention.trainable_point_weights"),
    "structure_module.ipa.linear_q.": "structure_module.block.invariant_point_attention.query_scalar_projection.",
    "structure_module.ipa.linear_kv.": "structure_module.block.invariant_point_attention.key_value_scalar_projection.",
    "structure_module.ipa.linear_q_points.": (
        "structure_module.block.invariant_point_attention.query_point_projection."
    ),
    "structure_module.ipa.linear_kv_points.": (
        "structure_module.block.invariant_point_attention.key_value_point_projection."
    ),
    "structure_module.ipa.linear_b.": "structure_module.block.invariant_point_attention.pair_bias_projection.",
    "structure_module.ipa.linear_out.": ("structure_module.block.invariant_point_attention.output_projection."),
    "structure_module.layer_norm_ipa.": "structure_module.block.attention_norm.",
    "structure_module.transition.layer_norm.": "structure_module.block.transition_norm.",
    "structure_module.bb_update.linear.": "structure_module.block.backbone_update.",
    "structure_module.angle_resnet.linear_in.": "structure_module.block.sidechain.input_projections.0.",
    "structure_module.angle_resnet.linear_initial.": "structure_module.block.sidechain.input_projections.1.",
    "structure_module.angle_resnet.linear_out.": "structure_module.block.sidechain.angle_projection.",
}


def convert_openfold_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    dtype: torch.dtype | None = None,
) -> AlphaFold2Model:
    r"""
    Convert an OpenFold AlphaFold 2 checkpoint into a strict-loaded native model.
    """

    state = _load_openfold_checkpoint(checkpoint_path)
    converted = convert_openfold_alphafold2_state_dict(state, dtype=dtype)

    model_config = AlphaFold2Config(
        {
            "heads": {"predicted_aligned_error": {"enabled": any(key.startswith("aux_heads.tm.") for key in state)}},
            "template": {"enabled": _has_template_parameters(state)},
        }
    )
    with torch.device("meta"):
        model = AlphaFold2Model(model_config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_openfold_alphafold2_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Translate OpenFold PyTorch state-dict keys to OFoldX AlphaFold2Model keys.

    OpenFold tensors are already stored in PyTorch layout, so conversion is a
    deterministic module-name aliasing pass rather than a layout transpose.
    """

    converted: dict[str, Tensor] = {}
    for source_key, tensor in state_dict.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"OpenFold state_dict value for {source_key!r} is not a tensor")
        target_key = normalize_openfold_state_key(source_key)
        target = tensor.detach().cpu().contiguous()
        if dtype is not None and target.is_floating_point():
            target = target.to(dtype=dtype)
        if target_key in converted:
            raise ValueError(f"multiple OpenFold tensors map to {target_key!r}")
        converted[target_key] = target
    return converted


def normalize_openfold_state_key(key: str) -> str:
    r"""
    Return the OFoldX state-dict key corresponding to one OpenFold tensor key.
    """

    mapped = _replace_prefix(key, _TOP_LEVEL_PREFIX_MAP)
    if mapped is not None:
        return mapped

    structure_key = _normalize_structure_module_key(key)
    if structure_key is not None:
        return structure_key

    if key.startswith("template_pointwise_att.mha."):
        suffix = key.removeprefix("template_pointwise_att.mha.")
        return "template_embedder.attention.projection." + _normalize_attention_projection_key(suffix)

    template_match = _TEMPLATE_BLOCK_KEY_RE.fullmatch(key)
    if template_match is not None:
        block_index, rest = template_match.groups()
        return (
            "template_embedder.single_template_embedder.template_pair_stack.blocks."
            f"{block_index}.{_normalize_pair_stack_key(rest, transition_name='transition')}"
        )

    block_match = _BLOCK_KEY_RE.fullmatch(key)
    if block_match is not None:
        stack_name, block_index, rest = block_match.groups()
        return _normalize_evoformer_block_key(stack_name, block_index, rest)

    raise ValueError(f"unsupported OpenFold AlphaFold 2 tensor key: {key!r}")


def _load_openfold_checkpoint(
    checkpoint_path: str | PathLike[str],
) -> Mapping[str, Tensor]:
    source_path = Path(checkpoint_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    return cast(Mapping[str, Tensor], _select_state_dict(checkpoint))


def _normalize_evoformer_block_key(stack_name: str, block_index: str, rest: str) -> str:
    stack_name = "evoformer_stack" if stack_name == "evoformer" else stack_name
    block_prefix = f"{stack_name}.blocks.{block_index}"
    if rest.startswith("msa_att_row."):
        return f"{block_prefix}.msa_block.msa_row_attention_with_pair_bias." + _normalize_msa_row_key(
            rest.removeprefix("msa_att_row.")
        )
    if rest.startswith("msa_att_col._msa_att."):
        return f"{block_prefix}.msa_block.msa_column_attention." + _normalize_msa_column_attention_key(
            rest.removeprefix("msa_att_col._msa_att.")
        )
    if rest.startswith("msa_att_col.layer_norm_m."):
        suffix = rest.removeprefix("msa_att_col.layer_norm_m.")
        return f"{block_prefix}.msa_block.msa_column_global_attention.query_norm.{suffix}"
    if rest.startswith("msa_att_col.global_attention."):
        suffix = rest.removeprefix("msa_att_col.global_attention.")
        return (
            f"{block_prefix}.msa_block.msa_column_global_attention.attention.projection."
            + _normalize_attention_projection_key(suffix)
        )
    if rest.startswith("core.outer_product_mean."):
        return f"{block_prefix}.outer_product_mean." + _normalize_outer_product_mean_key(
            rest.removeprefix("core.outer_product_mean.")
        )
    if rest.startswith("core.msa_transition."):
        return f"{block_prefix}.msa_block.transition." + _normalize_transition_key(
            rest.removeprefix("core.msa_transition.")
        )
    if rest.startswith("core."):
        return f"{block_prefix}.pair_block.{_normalize_pair_stack_key(rest.removeprefix('core.'))}"
    raise ValueError(f"unsupported OpenFold Evoformer block key tail: {rest!r}")


def _normalize_msa_row_key(rest: str) -> str:
    if rest.startswith("layer_norm_m."):
        return "query_norm." + rest.removeprefix("layer_norm_m.")
    if rest.startswith("layer_norm_z."):
        return "pair_norm." + rest.removeprefix("layer_norm_z.")
    if rest.startswith("linear_z."):
        return "pair_bias_projection." + rest.removeprefix("linear_z.")
    if rest.startswith("mha."):
        return "attention.projection." + _normalize_attention_projection_key(rest.removeprefix("mha."))
    raise ValueError(f"unsupported OpenFold MSA row-attention key tail: {rest!r}")


def _normalize_msa_column_attention_key(rest: str) -> str:
    if rest.startswith("layer_norm_m."):
        return "query_norm." + rest.removeprefix("layer_norm_m.")
    if rest.startswith("mha."):
        return "attention.projection." + _normalize_attention_projection_key(rest.removeprefix("mha."))
    raise ValueError(f"unsupported OpenFold MSA column-attention key tail: {rest!r}")


def _normalize_outer_product_mean_key(rest: str) -> str:
    return _replace_prefix_required(
        rest,
        {
            "layer_norm.": "msa_norm.",
            "linear_1.": "left_projection.",
            "linear_2.": "right_projection.",
            "linear_out.": "output_projection.",
        },
        source="OpenFold outer-product-mean key tail",
    )


def _normalize_pair_stack_key(rest: str, *, transition_name: str = "transition") -> str:
    if rest.startswith("tri_mul_out."):
        return "triangle_multiplication_outgoing." + _normalize_triangle_multiplication_key(
            rest.removeprefix("tri_mul_out."), outgoing=True
        )
    if rest.startswith("tri_mul_in."):
        return "triangle_multiplication_incoming." + _normalize_triangle_multiplication_key(
            rest.removeprefix("tri_mul_in."), outgoing=False
        )
    if rest.startswith("tri_att_start."):
        return "triangle_attention_starting_node." + _normalize_triangle_attention_key(
            rest.removeprefix("tri_att_start.")
        )
    if rest.startswith("tri_att_end."):
        return "triangle_attention_ending_node." + _normalize_triangle_attention_key(rest.removeprefix("tri_att_end."))
    if rest.startswith("pair_transition."):
        return transition_name + "." + _normalize_transition_key(rest.removeprefix("pair_transition."))
    raise ValueError(f"unsupported OpenFold pair-stack key tail: {rest!r}")


def _normalize_triangle_multiplication_key(rest: str, *, outgoing: bool) -> str:
    # The unified TriangleMultiplication fixes the operand order (left owns row ``i``, right owns
    # column ``j``) and uses the canonical incoming einsum ``"...kih,...kjh->...ijh"``. OpenFold's
    # ``a``/``b`` factors map straight through for the outgoing update, but the incoming update must
    # swap ``a`` (left) <-> ``b`` (right) — value and gate together — so the canonical einsum
    # reproduces DeepMind's ``"...kjc,...kic->..."`` incoming contraction. See TRUNK_EINSUM_LOCK.md.
    if outgoing:
        left_value, right_value = "left_projection.", "right_projection."
        left_gate, right_gate = "left_gate_projection.", "right_gate_projection."
    else:
        left_value, right_value = "right_projection.", "left_projection."
        left_gate, right_gate = "right_gate_projection.", "left_gate_projection."
    return _replace_prefix_required(
        rest,
        {
            "linear_a_p.": left_value,
            "linear_b_p.": right_value,
            "linear_a_g.": left_gate,
            "linear_b_g.": right_gate,
            "linear_z.": "output_projection.",
            "linear_g.": "gate_projection.",
            "layer_norm_in.": "input_norm.",
            "layer_norm_out.": "output_norm.",
        },
        source="OpenFold triangle-multiplication key tail",
    )


def _normalize_triangle_attention_key(rest: str) -> str:
    # The unified TriangleAttention (shared bias-free module + the AlphaFold 2 subclass) registers its
    # pre-attention LayerNorm as ``pair_norm`` and its per-head triangle-bias projection as
    # ``bias_projection`` (the OpenFold names are ``layer_norm``/``linear``). See TRUNK_EINSUM_LOCK.md.
    if rest.startswith("layer_norm."):
        return "pair_norm." + rest.removeprefix("layer_norm.")
    if rest.startswith("linear."):
        return "bias_projection." + rest.removeprefix("linear.")
    if rest.startswith("mha."):
        return "attention.projection." + _normalize_attention_projection_key(rest.removeprefix("mha."))
    raise ValueError(f"unsupported OpenFold triangle-attention key tail: {rest!r}")


def _normalize_transition_key(rest: str) -> str:
    return _replace_prefix_required(
        rest,
        {
            "layer_norm.": "input_norm.",
            "linear_1.": "input_projection.",
            "linear_2.": "output_projection.",
        },
        source="OpenFold transition key tail",
    )


def _normalize_attention_projection_key(rest: str) -> str:
    prefix, _, suffix = rest.partition(".")
    mapped = _ATTENTION_PROJECTION_MAP.get(prefix)
    if mapped is None or not suffix:
        raise ValueError(f"unsupported OpenFold attention projection key tail: {rest!r}")
    return f"{mapped}.{suffix}"


def _normalize_structure_module_key(key: str) -> str | None:
    mapped = _replace_prefix(key, _STRUCTURE_PREFIX_MAP)
    if mapped is not None:
        return mapped
    if key.startswith("structure_module.transition.layers.0.linear_"):
        rest = key.removeprefix("structure_module.transition.layers.0.linear_")
        linear_index, _, suffix = rest.partition(".")
        if linear_index in {"1", "2", "3"} and suffix:
            target_index = int(linear_index) - 1
            return f"structure_module.block.transition.projections.{target_index}.{suffix}"
    angle_match = re.fullmatch(r"structure_module\.angle_resnet\.layers\.(\d+)\.linear_([12])\.(.+)", key)
    if angle_match is not None:
        block_index, linear_index, suffix = angle_match.groups()
        target_name = "linear1" if linear_index == "1" else "linear2"
        return f"structure_module.block.sidechain.blocks.{block_index}.{target_name}.{suffix}"
    return None


def _replace_prefix(key: str, mapping: Mapping[str, str]) -> str | None:
    for source_prefix, target_prefix in mapping.items():
        if key == source_prefix:
            return target_prefix
        if key.startswith(source_prefix):
            return target_prefix + key.removeprefix(source_prefix)
    return None


def _replace_prefix_required(rest: str, mapping: Mapping[str, str], *, source: str) -> str:
    mapped = _replace_prefix(rest, mapping)
    if mapped is None:
        raise ValueError(f"unsupported {source}: {rest!r}")
    return mapped


def _has_template_parameters(state_dict: Mapping[str, Tensor]) -> bool:
    return any(
        key.startswith(
            (
                "template_angle_embedder.",
                "template_pair_embedder.",
                "template_pair_stack.",
                "template_pointwise_att.",
            )
        )
        for key in state_dict
    )


def _select_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "params"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        ema = checkpoint.get("ema")
        if isinstance(ema, Mapping):
            for key in ("params", "state_dict"):
                value = ema.get(key)
                if isinstance(value, Mapping):
                    return value
        if checkpoint and all(isinstance(key, str) and isinstance(value, Tensor) for key, value in checkpoint.items()):
            return checkpoint
    raise TypeError("OpenFold checkpoint does not contain a recognizable tensor state_dict mapping")
