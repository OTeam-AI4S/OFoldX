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
RFdiffusion checkpoint conversion.

Converts official RFdiffusion checkpoints to native OFoldX Models.
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...conversion import save_repository
from ...errors import SourceRef, fail
from ..vocabulary import canonical_word_embeddings
from .config import RfDiffusionConfig
from .model import RFDIFFUSION_TEMPLATE_VOCABULARY, RFDIFFUSION_VOCABULARY, RfDiffusionModel
from .pipeline import RfDiffusionPipeline
from .processor import RfDiffusionProcessor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_TRANSLATION_SCHEDULE = "linear"
_ROTATION_DIFFUSER = "igso3"


def rfdiffusion_schedule(source: str | PathLike[str]) -> dict[str, np.ndarray]:
    r"""
    Load the released IGSO(3) lookup table from an original or safe converted source.
    """

    path = Path(source)
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            values = pickle.load(handle)
        return {name: np.asarray(array) for name, array in values.items()}
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name] for name in values.files}


_DENOISER_RENAMES = {
    "n_extra_block": "num_extra_blocks",
    "n_main_block": "num_main_blocks",
    "n_ref_block": "num_refinement_blocks",
    "d_msa": "msa_size",
    "d_msa_full": "full_msa_size",
    "d_pair": "pair_size",
    "d_templ": "template_size",
    "n_head_msa": "num_msa_attention_heads",
    "n_head_pair": "num_pair_attention_heads",
    "n_head_templ": "num_template_attention_heads",
    "d_hidden": "hidden_size",
    "d_hidden_templ": "template_hidden_size",
    "p_drop": "dropout",
    "freeze_track_motif": "freeze_track_motif",
    "SE3_param_full": "trunk_se3",
    "SE3_param_topk": "refinement_se3",
}
_SE3_RENAMES = {
    "num_layers": "num_blocks",
    "num_channels": "hidden_size",
    "n_heads": "num_attention_heads",
    "div": "channel_reduction_factor",
    "l0_in_features": "input_scalar_size",
    "l0_out_features": "output_scalar_size",
    "l1_in_features": "input_vector_size",
    "l1_out_features": "output_vector_size",
    "num_edge_features": "edge_size",
}
_DIFFUSER_RENAMES = {
    "T": "num_timesteps",
    "b_0": "beta_initial",
    "b_T": "beta_final",
    "crd_scale": "coordinate_scale",
    "so3_schedule_type": "rotation_schedule_type",
    "min_b": "min_rotation_beta",
    "max_b": "max_rotation_beta",
    "min_sigma": "min_rotation_sigma",
    "max_sigma": "max_rotation_sigma",
}
_PREPROCESS_RENAMES = {
    "sidechain_input": "sidechain_input",
    "motif_sidechain_input": "motif_sidechain_input",
    "d_t1d": "template_single_feature_size",
    "d_t2d": "template_pair_feature_size",
    "str_self_cond": "structure_self_conditioning",
}

_DENOISER_ROOT_RENAMES = {
    "latent_emb": "msa_embeddings",
    "full_emb": "extra_msa_embeddings",
    "templ_emb": "template_embedder",
    "recycle": "recycling_embeddings",
    "simulator": "trunk",
    "c6d_pred": "pairwise_geometry_head",
    "aa_pred": "masked_token_head",
    "lddt_pred": "predicted_lddt_head",
    "exp_pred": "experimentally_resolved_head",
}

_DENOISER_STRUCTURE_RENAMES = (
    ("msa_embeddings.pos.emb.", "msa_embeddings.relative_position_embedding.embedding."),
    ("msa_embeddings.emb_state.", "msa_embeddings.state_embedding."),
    ("msa_embeddings.emb_right.", "msa_embeddings.right_pair_embedding."),
    ("msa_embeddings.emb_left.", "msa_embeddings.left_pair_embedding."),
    ("msa_embeddings.emb_q.", "msa_embeddings.query_embedding."),
    ("msa_embeddings.emb.", "msa_embeddings.input_projection."),
    ("extra_msa_embeddings.emb_q.", "extra_msa_embeddings.query_embedding."),
    ("extra_msa_embeddings.emb.", "extra_msa_embeddings.input_projection."),
    ("template_embedder.templ_stack.block.", "template_embedder.template_pair_stack.blocks."),
    ("template_embedder.templ_stack.", "template_embedder.template_pair_stack."),
    ("template_embedder.emb_t1d.", "template_embedder.single_feed_forward.input_projection."),
    ("template_embedder.proj_t1d.", "template_embedder.single_feed_forward.output_projection."),
    ("template_embedder.attn_tor.", "template_embedder.state_attention."),
    ("template_embedder.attn.", "template_embedder.pair_attention."),
    ("template_embedder.emb.", "template_embedder.pair_projection."),
    ("recycling_embeddings.proj_dist.", "recycling_embeddings.pair_projection."),
    ("trunk.proj_state2.", "trunk.state_output_projection."),
    ("trunk.proj_state.", "trunk.state_input_projection."),
    ("trunk.extra_block.", "trunk.extra_blocks."),
    ("trunk.main_block.", "trunk.main_blocks."),
    ("trunk.str_refiner.", "trunk.refinement_update."),
    ("pairwise_geometry_head.proj_asymm.", "pairwise_geometry_head.asymmetric_projection."),
    ("pairwise_geometry_head.proj_symm.", "pairwise_geometry_head.symmetric_projection."),
    ("masked_token_head.proj.", "masked_token_head.projection."),
    ("predicted_lddt_head.proj.", "predicted_lddt_head.projection."),
    ("experimentally_resolved_head.proj.", "experimentally_resolved_head.projection."),
    (".msa2msa.", ".msa_block."),
    (".msa2pair.", ".outer_product_mean."),
    (".pair2pair.", ".pair_block."),
    (".str2str.", ".structure_update."),
    (".sc_predictor.", ".torsion_head."),
    (".se3.se3.", ".structure_module.transformer."),
)

_DENOISER_LEAF_RENAMES = (
    (".template_pair_stack.norm.", ".template_pair_norm."),
    (".norm_msa.", ".msa_norm."),
    (".norm_pair.", ".pair_norm."),
    (".norm_state.", ".state_norm."),
    (".norm_bias.", ".bias_norm."),
    (".proj_pair.", ".pair_projection."),
    (".proj_state.", ".state_projection."),
    (".emb_rbf.", ".radial_feed_forward.input_projection."),
    (".proj_rbf.", ".radial_feed_forward.output_projection."),
    (".row_attn.", ".row_attention."),
    (".col_attn.", ".column_attention."),
    (".msa_block.column_attention.msa_norm.", ".msa_block.column_attention.query_norm."),
    (".seq_weight.", ".sequence_weighting."),
    (".ff.", ".transition."),
    (".transition.norm.", ".transition.input_norm."),
    (".transition.linear1.", ".transition.feed_forward.input_projection."),
    (".transition.linear2.", ".transition.feed_forward.output_projection."),
    (".outer_product_mean.norm.", ".outer_product_mean.msa_norm."),
    (".outer_product_mean.proj_left.", ".outer_product_mean.left_projection."),
    (".outer_product_mean.proj_right.", ".outer_product_mean.right_projection."),
    (".outer_product_mean.proj_out.", ".outer_product_mean.output_projection."),
    (".structure_update.embed_x.", ".structure_update.node_projection."),
    (".structure_update.norm_node.", ".structure_update.node_norm."),
    (".structure_update.embed_e1.", ".structure_update.pair_projection."),
    (".structure_update.norm_edge1.", ".structure_update.pair_edge_norm."),
    (".structure_update.embed_e2.", ".structure_update.geometry_projection."),
    (".structure_update.norm_edge2.", ".structure_update.geometry_norm."),
    (".refinement_update.embed_x.", ".refinement_update.node_projection."),
    (".refinement_update.norm_node.", ".refinement_update.node_norm."),
    (".refinement_update.embed_e1.", ".refinement_update.pair_projection."),
    (".refinement_update.norm_edge1.", ".refinement_update.pair_edge_norm."),
    (".refinement_update.embed_e2.", ".refinement_update.geometry_projection."),
    (".refinement_update.norm_edge2.", ".refinement_update.geometry_norm."),
    (".torsion_head.norm_s0.", ".torsion_head.msa_norm."),
    (".torsion_head.linear_s0.", ".torsion_head.msa_projection."),
    (".torsion_head.norm_si.", ".torsion_head.state_norm."),
    (".torsion_head.linear_si.", ".torsion_head.state_projection."),
    (".torsion_head.linear_1.", ".torsion_head.blocks.0.feed_forward.input_projection."),
    (".torsion_head.linear_2.", ".torsion_head.blocks.0.feed_forward.output_projection."),
    (".torsion_head.linear_3.", ".torsion_head.blocks.1.feed_forward.input_projection."),
    (".torsion_head.linear_4.", ".torsion_head.blocks.1.feed_forward.output_projection."),
    (".torsion_head.linear_out.", ".torsion_head.output_projection."),
    (".to_key_value.", ".key_value_projection.convolution."),
    (".to_query.", ".query_projection."),
    (".to_key.", ".key_projection."),
    (".to_q.", ".query_projection."),
    (".to_k.", ".key_projection."),
    (".to_v.", ".value_projection."),
    (".to_b.", ".bias_projection."),
    (".to_g.", ".gate_projection."),
    (".to_out.", ".output_projection."),
    (".project.", ".output_projection."),
    (".state_attention.query_projection.", ".state_attention.projection.query_projection."),
    (".state_attention.key_projection.", ".state_attention.projection.key_projection."),
    (".state_attention.value_projection.", ".state_attention.projection.value_projection."),
    (".state_attention.output_projection.", ".state_attention.projection.output_projection."),
    (".pair_attention.query_projection.", ".pair_attention.projection.query_projection."),
    (".pair_attention.key_projection.", ".pair_attention.projection.key_projection."),
    (".pair_attention.value_projection.", ".pair_attention.projection.value_projection."),
    (".pair_attention.output_projection.", ".pair_attention.projection.output_projection."),
    (".conv_in.", ".input_convolutions."),
    (".conv_out.", ".output_convolutions."),
    (".radial_func.", ".radial_profile."),
    (".net.0.", ".input_projection."),
    (".net.1.", ".input_norm."),
    (".net.3.", ".intermediate_projections.0."),
    (".net.4.", ".intermediate_norms.0."),
    (".net.6.", ".output_projection."),
    (".to_kernel_self.", ".self_interaction_weights."),
)


#: Checkpoint tensors whose rows are the released residue vocabulary.
_VOCABULARY_ROW_KEYS = (
    "denoiser.msa_embeddings.query_embedding.weight",
    "denoiser.msa_embeddings.left_pair_embedding.weight",
    "denoiser.msa_embeddings.right_pair_embedding.weight",
    "denoiser.msa_embeddings.state_embedding.weight",
    "denoiser.extra_msa_embeddings.query_embedding.weight",
)


def _convert_feature_blocks(
    tensor: Tensor, *, vocabulary: Sequence[str], blocks: int, block_size: int, offset: int, dim: int = -1
) -> Tensor:
    r"""
    One weight whose features are ``offset`` leading channels, then ``blocks`` repetitions of a residue plane followed
    by that block's other channels, then any remaining features, with every plane regenerated onto the 72 canonical
    classes. Both ``offset`` and the trailing features keep their place.
    """

    plane = len(vocabulary)
    converted = [tensor.narrow(dim, 0, offset)]
    for block in range(blocks):
        start = offset + block * block_size
        (residues,) = canonical_word_embeddings(tensor.narrow(dim, start, plane), vocabulary=vocabulary, dim=dim)
        converted += [residues, tensor.narrow(dim, start + plane, block_size - plane)]
    end = offset + blocks * block_size
    converted.append(tensor.narrow(dim, end, tensor.shape[dim] - end))
    return torch.cat(converted, dim=dim)


def _convert_vocabulary_axes(
    state_dict: dict[str, Tensor], *, template_single_feature_size: int, template_pair_feature_size: int
) -> None:
    r"""
    Regenerate every released residue axis onto the 72 canonical classes: the MSA query, pair and state
    embeddings' rows, the two residue planes of the latent-MSA input projection and the one of the extra-MSA
    projection, and the residue plane of each template single-feature block. The template pair projection reads the
    pairwise features first, so its two single blocks start at ``template_pair_feature_size``. The auxiliary
    masked-token logits keep the released 21-class axis (no OFoldX consumer reads them). Every other channel keeps its
    place.
    """

    for key in _VOCABULARY_ROW_KEYS:
        if key in state_dict:
            (state_dict[key],) = canonical_word_embeddings(state_dict[key], vocabulary=RFDIFFUSION_VOCABULARY, dim=0)
    planes = len(RFDIFFUSION_VOCABULARY)
    for key, blocks in (
        ("denoiser.msa_embeddings.input_projection.weight", 2),
        ("denoiser.extra_msa_embeddings.input_projection.weight", 1),
    ):
        if key in state_dict:
            state_dict[key] = _convert_feature_blocks(
                state_dict[key], vocabulary=RFDIFFUSION_VOCABULARY, blocks=blocks, block_size=planes, offset=0
            )
    for key, blocks, offset in (
        ("denoiser.template_embedder.pair_projection.weight", 2, template_pair_feature_size),
        ("denoiser.template_embedder.single_feed_forward.input_projection.weight", 1, 0),
    ):
        if key in state_dict:
            state_dict[key] = _convert_feature_blocks(
                state_dict[key],
                vocabulary=RFDIFFUSION_TEMPLATE_VOCABULARY,
                blocks=blocks,
                block_size=template_single_feature_size,
                offset=offset,
            )


def load_rfdiffusion_checkpoint(
    checkpoint: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    r"""
    Load one official checkpoint with PyTorch's restricted loader.
    """

    value = torch.load(
        checkpoint,
        map_location=map_location,
        weights_only=True,
        mmap=True,
    )
    return dict(value)


def rfdiffusion_config_from_checkpoint(checkpoint: Mapping[str, object]) -> RfDiffusionConfig:
    r"""
    Recover only persistent denoiser/diffusion/preprocess architecture.
    """

    source = cast(Mapping[str, object], checkpoint["config_dict"])
    diffuser = cast(Mapping[str, object], source["diffuser"])
    if diffuser["schedule_type"] != _TRANSLATION_SCHEDULE:
        fail("model_capacity", SourceRef("checkpoint", 0, item="schedule_type"), schedule=diffuser["schedule_type"])
    if diffuser["so3_type"] != _ROTATION_DIFFUSER:
        fail("model_capacity", SourceRef("checkpoint", 0, item="so3_type"), diffuser=diffuser["so3_type"])
    values = {}
    for source_name, target_name, renames in (
        ("model", "denoiser", _DENOISER_RENAMES),
        ("diffuser", "diffuser", _DIFFUSER_RENAMES),
        ("preprocess", "preprocess", _PREPROCESS_RENAMES),
    ):
        group = cast(Mapping[str, object], source[source_name])
        normalized = {target: group[key] for key, target in renames.items()}
        if target_name == "denoiser":
            for key in ("trunk_se3", "refinement_se3"):
                nested = cast(Mapping[object, object], normalized[key])
                normalized[key] = {
                    _SE3_RENAMES.get(str(item_key), str(item_key)): item for item_key, item in nested.items()
                }
        values[target_name] = normalized
    return RfDiffusionConfig(values)


def convert_rfdiffusion_state_dict_key(key: str) -> str:
    r"""
    Translate one official tensor name to the public Model module tree.
    """

    key = key.removeprefix("module.")
    source_root, separator, remainder = key.partition(".")
    target_root = _DENOISER_ROOT_RENAMES.get(source_root, source_root)
    key = target_root + separator + remainder
    for source, target in _DENOISER_STRUCTURE_RENAMES:
        key = key.replace(source, target)
    key = _convert_rfdiffusion_se3_key(key)
    for source, target in _DENOISER_LEAF_RENAMES:
        key = key.replace(source, target)
    # Ordinary columns use gated MHA; extra-MSA columns use global-query attention.
    key = re.sub(
        r"^(trunk\.main_blocks\.\d+\.msa_block\.column_attention\.)"
        r"(query_projection|key_projection|value_projection|gate_projection|output_projection)\.",
        r"\1attention.projection.\2.",
        key,
    )
    key = re.sub(
        r"^(trunk\.extra_blocks\.\d+\.msa_block\.column_attention\.)"
        r"(query_projection|key_projection|value_projection|gate_projection|output_projection)\.",
        r"\1attention.\2.",
        key,
    )
    return f"denoiser.{key}"


def _convert_rfdiffusion_se3_key(key: str) -> str:
    r"""
    Translate the upstream interleaved SE(3) module list into named blocks and output.
    """

    marker = ".structure_module.transformer.graph_modules."
    prefix, separator, suffix = key.partition(marker)
    if not separator:
        return key
    module_index_text, _, remainder = suffix.partition(".")
    module_index = int(module_index_text)
    if remainder.startswith(("to_key_value.", "to_query.", "project.")):
        target = f".structure_module.transformer.blocks.{module_index // 2}."
    elif remainder.startswith("group_norm."):
        target = f".structure_module.transformer.blocks.{module_index // 2}.output_norm."
    else:
        target = ".structure_module.transformer.output_projection."
    return prefix + target + remainder


def convert_rfdiffusion_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    config: RfDiffusionConfig | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Return a strict-loadable native state dict, its residue axes regenerated onto the 72 canonical classes.

    Args:
        state_dict: The source RFdiffusion parameter mapping (checkpoint ``model_state_dict``).
        config: Source architecture recovered from the checkpoint; its template single- and pair-feature sizes locate
            the residue plane of each template block.
        dtype: Optional dtype to cast floating-point tensors to; leaves them unchanged when ``None``.
    """

    config = RfDiffusionConfig() if config is None else config
    converted = {
        convert_rfdiffusion_state_dict_key(key): (
            value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        )
        for key, value in state_dict.items()
    }
    _convert_vocabulary_axes(
        converted,
        template_single_feature_size=config.preprocess.template_single_feature_size,
        template_pair_feature_size=config.preprocess.template_pair_feature_size,
    )
    return converted


def convert_rfdiffusion_model(
    checkpoint: str | PathLike[str],
    *,
    schedule_source: str | PathLike[str],
    dtype: torch.dtype | None = None,
) -> RfDiffusionModel:
    r"""
    Return one native Model loaded strictly from an official checkpoint.

    The released IGSO(3) schedule table is copied verbatim from
    ``schedule_source`` (an ``.npz`` or original RFdiffusion ``.pkl``) into the
    model's persistent ``igso3_*`` float64 buffers. Repository saving then stores
    those buffers in ``model.safetensors``; OFoldX does not vendor the source
    table as package data.
    """

    loaded = load_rfdiffusion_checkpoint(checkpoint)
    config = rfdiffusion_config_from_checkpoint(loaded)
    with torch.device("meta"):
        model = RfDiffusionModel(config)
    state = convert_rfdiffusion_state_dict(
        cast(Mapping[str, Tensor], loaded["model_state_dict"]),
        config=config,
        dtype=dtype,
    )
    for name, array in rfdiffusion_schedule(schedule_source).items():
        state[f"igso3_{name}"] = torch.as_tensor(array, dtype=torch.float64)
    state["igso3_schedule_present"] = torch.tensor(True, dtype=torch.bool)
    model.load_state_dict(state, strict=True, assign=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def convert_rfdiffusion_checkpoint(
    checkpoint: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    schedule_source: str | PathLike[str],
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert an RFdiffusion checkpoint and write a routable OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_rfdiffusion_model`][ofoldx.models.rfdiffusion.convert.convert_rfdiffusion_model],
    pairs it with [`RfDiffusionProcessor`][], and saves the registered
    [`RfDiffusionPipeline`][]. Its one-field marker routes
    ``pipeline("design", model=dir)`` to that concrete Pipeline.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    model = convert_rfdiffusion_model(checkpoint, dtype=dtype, schedule_source=schedule_source)
    save_repository(
        RfDiffusionPipeline(model=model, processor=RfDiffusionProcessor(dtype=dtype)),
        output_dir,
        sources={"checkpoint": checkpoint, "igso3_schedule": schedule_source},
        converter="ofoldx.models.rfdiffusion.convert.convert_rfdiffusion_checkpoint",
        options={"dtype": None if dtype is None else str(dtype)},
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run rfdiffusion checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official RFdiffusion checkpoint (e.g. Base_ckpt.pt)")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--schedule-source",
        type=Path,
        required=True,
        help="IGSO(3) schedule source (the original RFdiffusion .pkl cache, or an .npz archive "
        "of the same arrays); conversion stores it in the model safetensors",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    convert_rfdiffusion_checkpoint(
        args.checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        schedule_source=args.schedule_source,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
