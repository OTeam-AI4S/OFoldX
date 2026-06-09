r"""
OpenFold AlphaFold 2 checkpoint conversion hook.

OpenFold checkpoints are not a separate OFoldX architecture. They are another
source format for the AlphaFold 2 architecture, so this module owns only the
source-specific load/key-normalization path.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from chanfig import Config
from torch import Tensor

from ...mixin import no_init_weights
from .deepmind import (
    CONVERSION_CONFIG_NAME,
    CONVERTED_WEIGHTS_NAME,
    _dtype_counts,
    _update_manifest_metadata,
    _validate_converted_state_dict_matches_model,
    _write_processor_component,
)

OPENFOLD_PARAMS_S3_URL = "s3://openfold/openfold_params/"
OPENFOLD_PARAMS_HTTPS_URL = "https://openfold.s3.amazonaws.com/openfold_params/"
OPENFOLD_PARAMS_LICENSE = "CC BY 4.0"
OPENFOLD_CODE_LICENSE = "Apache-2.0"
OPENFOLD_MODEL_NAMES = (
    "initial_training",
    "finetuning_2",
    "finetuning_3",
    "finetuning_4",
    "finetuning_5",
    "finetuning_no_templ_1",
    "finetuning_no_templ_2",
    "finetuning_no_templ_ptm_1",
    "finetuning_ptm_1",
    "finetuning_ptm_2",
)

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
    "input_embedder.linear_tf_m.": "input_embedding.preprocess_1d.",
    "input_embedder.linear_msa_m.": "input_embedding.preprocess_msa.",
    "input_embedder.linear_tf_z_i.": "input_embedding.left_single.",
    "input_embedder.linear_tf_z_j.": "input_embedding.right_single.",
    "input_embedder.linear_relpos.": "input_embedding.relpos_projection.",
    "recycling_embedder.linear.": "recycling_embedding.prev_pos_linear.",
    "recycling_embedder.layer_norm_m.": "recycling_embedding.prev_msa_first_row_norm.",
    "recycling_embedder.layer_norm_z.": "recycling_embedding.prev_pair_norm.",
    "extra_msa_embedder.linear.": "extra_msa_embedding.extra_msa_projection.",
    "evoformer.linear.": "single_activations.",
    "template_angle_embedder.linear_1.": "template_msa_embedding.template_single_embedding.",
    "template_angle_embedder.linear_2.": "template_msa_embedding.template_projection.",
    "template_pair_embedder.linear.": "template_embedder.single_template_embedding.embedding2d.",
    "template_pair_stack.layer_norm.": "template_embedder.single_template_embedding.output_layer_norm.",
    "aux_heads.distogram.linear.": "distogram_head.linear.",
    "aux_heads.masked_msa.linear.": "masked_msa_head.logits.",
    "aux_heads.experimentally_resolved.linear.": "experimentally_resolved_head.logits.",
    "aux_heads.tm.linear.": "predicted_aligned_error_head.logits.",
    "aux_heads.plddt.layer_norm.": "predicted_lddt_head.input_layer_norm.",
    "aux_heads.plddt.linear_1.": "predicted_lddt_head.act_0.",
    "aux_heads.plddt.linear_2.": "predicted_lddt_head.act_1.",
    "aux_heads.plddt.linear_3.": "predicted_lddt_head.logits.",
}
_STRUCTURE_PREFIX_MAP = {
    "structure_module.layer_norm_s.": "structure_module.single_layer_norm.",
    "structure_module.linear_in.": "structure_module.initial_projection.",
    "structure_module.layer_norm_z.": "structure_module.pair_layer_norm.",
    "structure_module.ipa.head_weights": (
        "structure_module.fold_iteration.invariant_point_attention.trainable_point_weights"
    ),
    "structure_module.ipa.linear_q.": "structure_module.fold_iteration.invariant_point_attention.q_scalar.",
    "structure_module.ipa.linear_kv.": "structure_module.fold_iteration.invariant_point_attention.kv_scalar.",
    "structure_module.ipa.linear_q_points.": (
        "structure_module.fold_iteration.invariant_point_attention.q_point_local."
    ),
    "structure_module.ipa.linear_kv_points.": (
        "structure_module.fold_iteration.invariant_point_attention.kv_point_local."
    ),
    "structure_module.ipa.linear_b.": "structure_module.fold_iteration.invariant_point_attention.attention_2d.",
    "structure_module.ipa.linear_out.": (
        "structure_module.fold_iteration.invariant_point_attention.output_projection."
    ),
    "structure_module.layer_norm_ipa.": "structure_module.fold_iteration.attention_layer_norm.",
    "structure_module.transition.layer_norm.": "structure_module.fold_iteration.transition_layer_norm.",
    "structure_module.bb_update.linear.": "structure_module.fold_iteration.affine_update.",
    "structure_module.angle_resnet.linear_in.": "structure_module.fold_iteration.sidechain.input_projection.0.",
    "structure_module.angle_resnet.linear_initial.": "structure_module.fold_iteration.sidechain.input_projection.1.",
    "structure_module.angle_resnet.linear_out.": "structure_module.fold_iteration.sidechain.unnormalized_angles.",
}


@dataclass(frozen=True, slots=True)
class _LoadedOpenFoldCheckpoint:
    tensors: dict[str, Tensor]
    model_name: str
    source_path: Path
    selected_state_key: str
    template_enabled: bool
    ptm_head_enabled: bool


def inspect_openfold_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    r"""
    Inspect an OpenFold AlphaFold 2 checkpoint.
    """

    loaded = _load_openfold_checkpoint(checkpoint_path, model_name=model_name)
    top_prefixes = Counter(key.split(".", 1)[0] for key in loaded.tensors)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_source": "openfold",
        "checkpoint_format": "torch_state_dict",
        "model_name": loaded.model_name,
        "selected_state_key": loaded.selected_state_key,
        "num_tensors": len(loaded.tensors),
        "num_parameters": sum(tensor.numel() for tensor in loaded.tensors.values()),
        "non_tensor_entries": 0,
        "top_prefixes": dict(top_prefixes.most_common(20)),
        "dtypes": _dtype_counts(loaded.tensors),
        "template_enabled": loaded.template_enabled,
        "ptm_head_enabled": loaded.ptm_head_enabled,
        "source_release": OPENFOLD_PARAMS_S3_URL,
        "openfold_code_license": OPENFOLD_CODE_LICENSE,
        "openfold_params_license": OPENFOLD_PARAMS_LICENSE,
    }


def convert_openfold_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
    verify_artifact: bool = True,
) -> dict[str, Any]:
    r"""
    Convert an OpenFold AlphaFold 2 checkpoint into an OFoldX model artifact.
    """

    loaded = _load_openfold_checkpoint(checkpoint_path, model_name=model_name)
    converted = convert_openfold_alphafold2_state_dict(loaded.tensors, dtype=dtype)

    from ..config import AlphaFold2Config
    from ..model import AlphaFold2Model

    model_config = AlphaFold2Config(_model_config_overrides(loaded))
    model_config["checkpoint_variant"] = loaded.model_name
    model_config["model_family"] = _model_family(loaded)
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
    Config(_conversion_config(loaded, persisted_state, dtype=dtype)).json(output_path / CONVERSION_CONFIG_NAME)
    _update_manifest_metadata(output_path)

    if verify_artifact:
        AlphaFold2Model.from_pretrained(output_path, strict=True)

    return {
        "checkpoint": str(Path(checkpoint_path)),
        "output_dir": str(output_path),
        "architecture": "alphafold2",
        "artifact_kind": "model",
        "checkpoint_source": "openfold",
        "checkpoint_format": "torch_state_dict",
        "model_name": loaded.model_name,
        "source_release": OPENFOLD_PARAMS_S3_URL,
        "openfold_code_license": OPENFOLD_CODE_LICENSE,
        "openfold_params_license": OPENFOLD_PARAMS_LICENSE,
        "runnable_model": True,
        "tensor_count": len(persisted_state),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(persisted_state),
        "template_enabled": loaded.template_enabled,
        "ptm_head_enabled": loaded.ptm_head_enabled,
        "processor_written": True,
        "verified": verify_artifact,
    }


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
            "template_embedder.single_template_embedding.template_pair_stack.blocks."
            f"{block_index}.pair_stack.{_normalize_pair_stack_key(rest)}"
        )

    block_match = _BLOCK_KEY_RE.fullmatch(key)
    if block_match is not None:
        stack_name, block_index, rest = block_match.groups()
        return _normalize_evoformer_block_key(stack_name, block_index, rest)

    raise ValueError(f"unsupported OpenFold AlphaFold 2 tensor key: {key!r}")


def _load_openfold_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None,
) -> _LoadedOpenFoldCheckpoint:
    source_path = Path(checkpoint_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    state_dict, selected_state_key = _select_state_dict(checkpoint)
    tensor_state, _non_tensor_count = _tensor_state_dict(state_dict)
    resolved_model_name = model_name or source_path.stem
    return _LoadedOpenFoldCheckpoint(
        tensors=tensor_state,
        model_name=resolved_model_name,
        source_path=source_path,
        selected_state_key=selected_state_key,
        template_enabled=_has_template_parameters(tensor_state),
        ptm_head_enabled=any(key.startswith("aux_heads.tm.") for key in tensor_state),
    )


def _normalize_evoformer_block_key(stack_name: str, block_index: str, rest: str) -> str:
    block_prefix = f"{stack_name}.blocks.{block_index}"
    if rest.startswith("msa_att_row."):
        return f"{block_prefix}.msa_stack.msa_row_attention_with_pair_bias." + _normalize_msa_row_key(
            rest.removeprefix("msa_att_row.")
        )
    if rest.startswith("msa_att_col._msa_att."):
        return f"{block_prefix}.msa_stack.msa_column_attention." + _normalize_msa_column_attention_key(
            rest.removeprefix("msa_att_col._msa_att.")
        )
    if rest.startswith("msa_att_col.layer_norm_m."):
        suffix = rest.removeprefix("msa_att_col.layer_norm_m.")
        return f"{block_prefix}.msa_stack.msa_column_global_attention.query_norm.{suffix}"
    if rest.startswith("msa_att_col.global_attention."):
        suffix = rest.removeprefix("msa_att_col.global_attention.")
        return (
            f"{block_prefix}.msa_stack.msa_column_global_attention.attention.projection."
            + _normalize_attention_projection_key(suffix)
        )
    if rest.startswith("core.outer_product_mean."):
        return f"{block_prefix}.outer_product_mean." + _normalize_outer_product_mean_key(
            rest.removeprefix("core.outer_product_mean.")
        )
    if rest.startswith("core.msa_transition."):
        return f"{block_prefix}.msa_stack.feed_forward." + _normalize_transition_key(
            rest.removeprefix("core.msa_transition.")
        )
    if rest.startswith("core."):
        return f"{block_prefix}.pair_stack.{_normalize_pair_stack_key(rest.removeprefix('core.'))}"
    raise ValueError(f"unsupported OpenFold Evoformer block key tail: {rest!r}")


def _normalize_msa_row_key(rest: str) -> str:
    if rest.startswith("layer_norm_m."):
        return "query_norm." + rest.removeprefix("layer_norm_m.")
    if rest.startswith("layer_norm_z."):
        return "feat_2d_norm." + rest.removeprefix("layer_norm_z.")
    if rest.startswith("linear_z."):
        return "feat_2d_weights." + rest.removeprefix("linear_z.")
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
            "layer_norm.": "layer_norm_input.",
            "linear_1.": "left_projection.",
            "linear_2.": "right_projection.",
            "linear_out.": "output_projection.",
        },
        source="OpenFold outer-product-mean key tail",
    )


def _normalize_pair_stack_key(rest: str) -> str:
    if rest.startswith("tri_mul_out."):
        return "triangle_multiplication_outgoing." + _normalize_triangle_multiplication_key(
            rest.removeprefix("tri_mul_out.")
        )
    if rest.startswith("tri_mul_in."):
        return "triangle_multiplication_incoming." + _normalize_triangle_multiplication_key(
            rest.removeprefix("tri_mul_in.")
        )
    if rest.startswith("tri_att_start."):
        return "triangle_attention_starting_node." + _normalize_triangle_attention_key(
            rest.removeprefix("tri_att_start.")
        )
    if rest.startswith("tri_att_end."):
        return "triangle_attention_ending_node." + _normalize_triangle_attention_key(rest.removeprefix("tri_att_end."))
    if rest.startswith("pair_transition."):
        return "feed_forward." + _normalize_transition_key(rest.removeprefix("pair_transition."))
    raise ValueError(f"unsupported OpenFold pair-stack key tail: {rest!r}")


def _normalize_triangle_multiplication_key(rest: str) -> str:
    return _replace_prefix_required(
        rest,
        {
            "linear_a_p.": "left_projection.",
            "linear_b_p.": "right_projection.",
            "linear_a_g.": "left_gate.",
            "linear_b_g.": "right_gate.",
            "linear_z.": "output_projection.",
            "linear_g.": "gating_linear.",
            "layer_norm_in.": "layer_norm_input.",
            "layer_norm_out.": "center_layer_norm.",
        },
        source="OpenFold triangle-multiplication key tail",
    )


def _normalize_triangle_attention_key(rest: str) -> str:
    if rest.startswith("layer_norm."):
        return "query_norm." + rest.removeprefix("layer_norm.")
    if rest.startswith("linear."):
        return "feat_2d_weights." + rest.removeprefix("linear.")
    if rest.startswith("mha."):
        return "attention.projection." + _normalize_attention_projection_key(rest.removeprefix("mha."))
    raise ValueError(f"unsupported OpenFold triangle-attention key tail: {rest!r}")


def _normalize_transition_key(rest: str) -> str:
    return _replace_prefix_required(
        rest,
        {
            "layer_norm.": "input_layer_norm.",
            "linear_1.": "transition1.",
            "linear_2.": "transition2.",
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
            return f"structure_module.fold_iteration.transition.layers.{target_index}.{suffix}"
    angle_match = re.fullmatch(r"structure_module\.angle_resnet\.layers\.(\d+)\.linear_([12])\.(.+)", key)
    if angle_match is not None:
        block_index, linear_index, suffix = angle_match.groups()
        target_name = "resblock1" if linear_index == "1" else "resblock2"
        return f"structure_module.fold_iteration.sidechain.resblocks.{block_index}.{target_name}.{suffix}"
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


def _model_family(loaded: _LoadedOpenFoldCheckpoint) -> str:
    return "monomer_ptm" if loaded.ptm_head_enabled else "monomer"


def _model_config_overrides(loaded: _LoadedOpenFoldCheckpoint) -> dict[str, object]:
    return {
        "heads": {"predicted_aligned_error": {"enabled": loaded.ptm_head_enabled}},
        "template": {"enabled": loaded.template_enabled},
    }


def _conversion_config(
    loaded: _LoadedOpenFoldCheckpoint,
    persisted: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None,
) -> dict[str, object]:
    return {
        "architecture": "alphafold2",
        "artifact_kind": "model",
        "model_name": loaded.model_name,
        "model_family": _model_family(loaded),
        "source_path": str(loaded.source_path),
        "selected_state_key": loaded.selected_state_key,
        "checkpoint_source": "openfold",
        "checkpoint_format": "torch_state_dict",
        "source_release": OPENFOLD_PARAMS_S3_URL,
        "source_release_https": OPENFOLD_PARAMS_HTTPS_URL,
        "openfold_code_license": OPENFOLD_CODE_LICENSE,
        "openfold_params_license": OPENFOLD_PARAMS_LICENSE,
        "converted_weights": CONVERTED_WEIGHTS_NAME,
        "linear_weight_layout": "pytorch",
        "tensor_count": len(persisted),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(persisted),
        "model_config": _model_config_overrides(loaded),
        "template_enabled": loaded.template_enabled,
        "ptm_head_enabled": loaded.ptm_head_enabled,
        "runnable_model": True,
    }


def _select_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], str]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "params"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and _looks_like_state_dict(value):
                return value, key
        ema = checkpoint.get("ema")
        if isinstance(ema, Mapping):
            for key in ("params", "state_dict"):
                value = ema.get(key)
                if isinstance(value, Mapping) and _looks_like_state_dict(value):
                    return value, f"ema.{key}"
        if _looks_like_state_dict(checkpoint):
            return checkpoint, "root"
    raise TypeError("OpenFold checkpoint does not contain a recognizable tensor state_dict mapping")


def _looks_like_state_dict(value: Mapping[str, Any]) -> bool:
    return any(isinstance(key, str) for key in value)


def _tensor_state_dict(state_dict: Mapping[str, Any]) -> tuple[dict[str, Tensor], int]:
    tensors: dict[str, Tensor] = {}
    non_tensor_count = 0
    for key, value in state_dict.items():
        if not isinstance(key, str):
            non_tensor_count += 1
            continue
        if isinstance(value, Tensor):
            tensors[key] = value.detach().cpu()
        else:
            non_tensor_count += 1
    if not tensors:
        raise TypeError("OpenFold checkpoint state_dict does not contain tensor values")
    return tensors, non_tensor_count
