r"""
Convert official RFdiffusion checkpoints to native OFoldX Models.
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from .config import RfDiffusionConfig
from .model import RfDiffusionModel

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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
_DIFFUSER_RENAMES = {
    "T": "num_timesteps",
    "b_0": "beta_initial",
    "b_T": "beta_final",
    "schedule_type": "schedule_type",
    "so3_type": "rotation_diffuser",
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
    ("template_embedder.emb_t1d.", "template_embedder.single_input_projection."),
    ("template_embedder.proj_t1d.", "template_embedder.single_output_projection."),
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
    (".emb_rbf.", ".radial_embedding."),
    (".proj_rbf.", ".radial_projection."),
    (".row_attn.", ".row_attention."),
    (".col_attn.", ".column_attention."),
    (".seq_weight.", ".sequence_weighting."),
    (".ff.", ".transition."),
    (".transition.norm.", ".transition.input_norm."),
    (".transition.linear1.", ".transition.input_projection."),
    (".transition.linear2.", ".transition.output_projection."),
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
    (".torsion_head.linear_1.", ".torsion_head.blocks.0.input_projection."),
    (".torsion_head.linear_2.", ".torsion_head.blocks.0.output_projection."),
    (".torsion_head.linear_3.", ".torsion_head.blocks.1.input_projection."),
    (".torsion_head.linear_4.", ".torsion_head.blocks.1.output_projection."),
    (".torsion_head.linear_out.", ".torsion_head.output_projection."),
    (".to_key_value.", ".key_value_projection."),
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
    (".net.3.", ".hidden_projection."),
    (".net.4.", ".hidden_norm."),
    (".net.6.", ".output_projection."),
    (".to_kernel_self.", ".self_interaction_weights."),
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
                    {
                        "n_heads": "num_attention_heads",
                        "num_layers": "num_blocks",
                    }.get(str(item_key), str(item_key)): item
                    for item_key, item in nested.items()
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
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Return a strict-loadable native state dict.
    """

    return {
        convert_rfdiffusion_state_dict_key(key): (
            value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        )
        for key, value in state_dict.items()
    }


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
        dtype=dtype,
    )
    for name, array in rfdiffusion_schedule(schedule_source).items():
        state[f"igso3_{name}"] = torch.as_tensor(array, dtype=torch.float64)
    state["igso3_schedule_present"] = torch.tensor(True, dtype=torch.bool)
    model.load_state_dict(state, strict=True, assign=True)
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

    from .pipeline import RfDiffusionPipeline
    from .processor import RfDiffusionProcessor

    model = convert_rfdiffusion_model(checkpoint, dtype=dtype, schedule_source=schedule_source)
    RfDiffusionPipeline(model=model, processor=RfDiffusionProcessor(dtype=dtype)).save(
        output_dir,
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
