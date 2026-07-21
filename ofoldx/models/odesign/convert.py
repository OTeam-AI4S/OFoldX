r"""Convert released ODesign checkpoints into paired OFoldX repositories."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

from .config import ODesignConfig, ODesignProcessorConfig
from .model import ODesignModel
from .processor import ODesignProcessor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_UNUSED_REFERENCE_KEYS = frozenset(
    {
        "template_embedder.linear_no_bias_z.weight",
        "template_embedder.layernorm_z.weight",
        "template_embedder.layernorm_z.bias",
        "template_embedder.linear_no_bias_a.weight",
        "template_embedder.layernorm_v.weight",
        "template_embedder.layernorm_v.bias",
        "template_embedder.linear_no_bias_u.weight",
        "confidence_head.lower_bins",
        "confidence_head.upper_bins",
        "confidence_head.plddt_weight",
        "confidence_head.resolved_weight",
        "confidence_head.linear_no_bias_s1.weight",
        "confidence_head.linear_no_bias_s2.weight",
        "confidence_head.linear_no_bias_d.weight",
        "confidence_head.linear_no_bias_d_wo_onehot.weight",
        "confidence_head.linear_no_bias_pae.weight",
        "confidence_head.linear_no_bias_pde.weight",
        "confidence_head.input_strunk_ln.weight",
        "confidence_head.input_strunk_ln.bias",
        "confidence_head.pae_ln.weight",
        "confidence_head.pae_ln.bias",
        "confidence_head.pde_ln.weight",
        "confidence_head.pde_ln.bias",
        "confidence_head.plddt_ln.weight",
        "confidence_head.plddt_ln.bias",
        "confidence_head.resolved_ln.weight",
        "confidence_head.resolved_ln.bias",
    }
).union(
    f"confidence_head.pairformer_stack.blocks.{block}.{suffix}"
    for block in range(4)
    for suffix in (
        "tri_mul_out.linear_g.weight",
        "tri_mul_out.linear_z.weight",
        "tri_mul_out.layer_norm_in.weight",
        "tri_mul_out.layer_norm_in.bias",
        "tri_mul_out.layer_norm_out.weight",
        "tri_mul_out.layer_norm_out.bias",
        "tri_mul_out.linear_a_p.weight",
        "tri_mul_out.linear_a_g.weight",
        "tri_mul_out.linear_b_p.weight",
        "tri_mul_out.linear_b_g.weight",
        "tri_mul_in.linear_g.weight",
        "tri_mul_in.linear_z.weight",
        "tri_mul_in.layer_norm_in.weight",
        "tri_mul_in.layer_norm_in.bias",
        "tri_mul_in.layer_norm_out.weight",
        "tri_mul_in.layer_norm_out.bias",
        "tri_mul_in.linear_a_p.weight",
        "tri_mul_in.linear_a_g.weight",
        "tri_mul_in.linear_b_p.weight",
        "tri_mul_in.linear_b_g.weight",
        "tri_att_start.layer_norm.weight",
        "tri_att_start.layer_norm.bias",
        "tri_att_start.linear.weight",
        "tri_att_start.mha.linear_q.weight",
        "tri_att_start.mha.linear_k.weight",
        "tri_att_start.mha.linear_v.weight",
        "tri_att_start.mha.linear_o.weight",
        "tri_att_start.mha.linear_g.weight",
        "tri_att_end.layer_norm.weight",
        "tri_att_end.layer_norm.bias",
        "tri_att_end.linear.weight",
        "tri_att_end.mha.linear_q.weight",
        "tri_att_end.mha.linear_k.weight",
        "tri_att_end.mha.linear_v.weight",
        "tri_att_end.mha.linear_o.weight",
        "tri_att_end.mha.linear_g.weight",
        "pair_transition.layernorm1.weight",
        "pair_transition.layernorm1.bias",
        "pair_transition.linear_no_bias_a.weight",
        "pair_transition.linear_no_bias_b.weight",
        "pair_transition.linear_no_bias.weight",
        "attention_pair_bias.layernorm_a.weight",
        "attention_pair_bias.layernorm_a.bias",
        "attention_pair_bias.attention.linear_q.weight",
        "attention_pair_bias.attention.linear_q.bias",
        "attention_pair_bias.attention.linear_k.weight",
        "attention_pair_bias.attention.linear_v.weight",
        "attention_pair_bias.attention.linear_o.weight",
        "attention_pair_bias.attention.linear_g.weight",
        "attention_pair_bias.layernorm_z.weight",
        "attention_pair_bias.layernorm_z.bias",
        "attention_pair_bias.linear_nobias_z.weight",
        "single_transition.layernorm1.weight",
        "single_transition.layernorm1.bias",
        "single_transition.linear_no_bias_a.weight",
        "single_transition.linear_no_bias_b.weight",
        "single_transition.linear_no_bias.weight",
    )
)


def load_reference_state(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    r"""Load tensor weights from a released ODesign checkpoint without executing pickle code."""

    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def translate_odesign_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""Translate every parameter of one ODesign graph from released key names."""

    used_reference_keys = set()
    for model_key in model.state_dict():
        reference_key = odesign_reference_key(model_key)
        if isinstance(reference_key, tuple):
            used_reference_keys.update(reference_key)
        else:
            used_reference_keys.add(reference_key)
    unexpected = set(reference_state) - used_reference_keys - _UNUSED_REFERENCE_KEYS
    if unexpected:
        lines = ["checkpoint contains unexpected ODesign tensors"]
        lines.extend(f"  {key}" for key in sorted(unexpected)[:20])
        raise RuntimeError("\n".join(lines))
    return translate_reference_state(model, reference_state, odesign_reference_key)


def _infer_receptor_mode(checkpoint_path: str | Path) -> Literal["rigid", "flex"]:
    r"""Infer the released receptor mode from a ``*_flex`` / ``*_rigid`` checkpoint filename.

    The two released protein archives ``odesign_base_prot_flex.pt`` /
    ``odesign_base_prot_rigid.pt`` share one graph and state-dict contract (§4.11.1); they
    differ only in receptor conditioning, which is carried by the Processor's
    ``receptor_mode`` (§4.11.2). The filename names it unambiguously; a name carrying neither
    token is rejected so the mode is never silently guessed.
    """

    stem = Path(checkpoint_path).stem.lower()
    has_rigid = "rigid" in stem
    has_flex = "flex" in stem
    if has_rigid == has_flex:
        raise ValueError(
            "cannot infer receptor_mode from checkpoint name "
            f"{Path(checkpoint_path).name!r}; pass --receptor-mode {{rigid,flex}} explicitly"
        )
    return "rigid" if has_rigid else "flex"


def convert_odesign_model(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    config: ODesignConfig | None = None,
) -> ODesignModel:
    r"""Strict-load one released ODesign backbone checkpoint into a native :class:`ODesignModel`."""

    model = ODesignModel(ODesignConfig() if config is None else config)
    if dtype is not None:
        model.to(dtype=dtype)
    translated = translate_odesign_state(model, load_reference_state(checkpoint_path))
    model.load_state_dict(translated, strict=True)
    return model


def convert_odesign_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    protein_designer: str | PathLike[str],
    ligand_designer: str | PathLike[str],
    dtype: torch.dtype | None = None,
    config: ODesignConfig | None = None,
    receptor_mode: Literal["rigid", "flex"] | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert the ODesign backbone checkpoint and write the aggregate composite repository.

    The released ODesign protein deliverable is a DESIGN §7.4 multi-pair :class:`ODesignPipeline`
    (§4.11.6): the ODesign backbone pair owns the runnable root, and the OInvFold sequence step is the
    *separately registered, read-only* ProteinMPNN / LigandMPNN Pipelines bound at the fixed
    ``protein_designer/`` and ``ligand_designer/`` sub-directories. The backbone pair alone keeps
    ``standard_tasks=()`` and cannot route as a lone standard task, so this writes the AGGREGATE
    repository — every class-owned pair plus the one-field ``pipeline.json = {"pipeline": "ofoldx.odesign"}``
    marker — via ``ODesignPipeline.save`` so ``pipeline("design", model=dir)`` routes it back through the
    multi-pair loader.

    ``protein_designer`` / ``ligand_designer`` are references (local repositories or Hub ids) to the
    already-converted OInvFold sequence-designer repositories; each is loaded read-only through the
    public repository factory (``pipeline("design", model=ref)``) and re-vendored into the ODesign
    repository — the sequence network is never re-converted here.

    ``receptor_mode`` selects the released flex/rigid conditioning written into the root pair's
    ``processor.json`` (§4.11.2/§4.11.8); it is inferred from the checkpoint filename when not given.

    Saving is local-only by default. When ``repo_id`` is given the fully written composite repository is
    also uploaded to that Hub repo; no organization or token is assumed and nothing is published without
    an explicit ``repo_id``.
    """

    from ...api import pipeline as _load_pipeline
    from .pipeline import ODesignPipeline

    if receptor_mode is None:
        receptor_mode = _infer_receptor_mode(checkpoint_path)
    model = convert_odesign_model(checkpoint_path, dtype=dtype, config=config)
    ODesignPipeline(
        model=model,
        processor=ODesignProcessor(ODesignProcessorConfig(receptor_mode=receptor_mode)),
        protein_designer=_load_pipeline("design", model=protein_designer),
        ligand_designer=_load_pipeline("design", model=ligand_designer),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def _bond_type_head_key(key: str) -> str:
    prefix = "token_bond_type_head"
    if key == prefix or key.startswith(prefix + "."):
        return "bond_type_predictor" + key[len(prefix) :]
    return key


def _constraint_embedder_key(key: str) -> str:
    if not key.startswith("constraint_distogram_embedder."):
        return key
    if key.startswith("constraint_distogram_embedder.pairformer_stack."):
        return key
    for current, reference in {
        ".pair_norm.": ".layernorm_z.",
        ".pair_projection.": ".linear_no_bias_z.",
        ".distogram_projection.": ".linear_no_bias_a.",
        ".hidden_norm.": ".layernorm_v.",
        ".output_projection.": ".linear_no_bias_u.",
    }.items():
        key = key.replace(current, reference)
    return key


def odesign_reference_key(model_key: str) -> ReferenceKey:
    r"""Return the released-checkpoint key for one OFoldX ODesign state key."""

    return _alphafold3_reference_key(
        model_key,
        model_rules=(_bond_type_head_key, _constraint_embedder_key),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="released ODesign backbone checkpoint (e.g. odesign_base_prot_flex.pt)",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--protein-designer",
        required=True,
        help="converted ProteinMPNN sequence-designer repository (local dir or Hub id)",
    )
    parser.add_argument(
        "--ligand-designer",
        required=True,
        help="converted LigandMPNN sequence-designer repository (local dir or Hub id)",
    )
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--receptor-mode",
        choices=("rigid", "flex"),
        default=None,
        help="released receptor conditioning; inferred from the *_flex/*_rigid filename when omitted",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    convert_odesign_checkpoint(
        args.checkpoint,
        args.output_dir,
        protein_designer=args.protein_designer,
        ligand_designer=args.ligand_designer,
        dtype=dtype,
        receptor_mode=args.receptor_mode,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


# ---------------------------------------------------------------------------
# Shared Protenix-style all-atom checkpoint key mapping.
#
# These helpers were formerly imported from the models-root ``_protenix_convert`` module. They
# are duplicated here verbatim so the ODesign converter is family-self-contained and imports no
# cross-family / models-root code (see the family layering rule). ``load_reference_state`` is NOT
# duplicated: ODesign defines its own strict variant above.
# ---------------------------------------------------------------------------

ReferenceKey: TypeAlias = str | tuple[str, ...]
ReferenceValue: TypeAlias = Tensor | tuple[Tensor, ...]


def _alphafold3_reference_key(model_key: str, *, model_rules: tuple[Callable[[str], str], ...] = ()) -> ReferenceKey:
    r"""
    Return the reference checkpoint key corresponding to an OFoldX model key.

    ``model_rules`` lets model-specific converters inject their own key rewrites right after the top-level
    rename, before the shared template/stack rules.
    """

    key = _replace_top_level(model_key)
    for rule in model_rules:
        key = rule(key)
    key = _replace_template_key(key)
    key = _replace_confidence_key(key)
    key = _replace_token_stack_key(key)
    key = _replace_denoiser_key(key)
    key = _replace_atom_stack_key(key)
    key = _replace_atom_key(key)
    key = _replace_pairformer_key(key)
    key = _replace_attention_key(key)
    key = _replace_attention_projection_key(key)
    key = _replace_transition_key(key)
    return _replace_conditioned_norm_key(key)


def translate_reference_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], ReferenceKey],
) -> dict[str, Tensor]:
    r"""Translate source names and values; PyTorch owns target-key and shape validation."""

    translated = {}
    for model_key in model.state_dict():
        reference_key = key_fn(model_key)
        reference_value = _lookup_reference_value(reference_state, reference_key)
        translated[model_key] = _transform_reference_value(model_key, reference_value)
    return translated


def _lookup_reference_value(reference_state: Mapping[str, Tensor], reference_key: ReferenceKey) -> ReferenceValue:
    r"""Read the source tensor or tensors named by one target key."""

    if isinstance(reference_key, tuple):
        return tuple(reference_state[key] for key in reference_key)
    return reference_state[reference_key]


def _transform_reference_value(model_key: str, reference_value: ReferenceValue) -> Tensor:
    r"""
    Apply value transforms that absorb OFoldX<->Protenix convention differences.

    OFoldX's atom-pair encoder follows the AF3/Protenix ``d = query - key`` convention directly, so the
    distance-vector projection loads without a sign transform.
    """

    if isinstance(reference_value, tuple):
        if model_key.endswith(".conditioning_projection.weight"):
            return torch.cat(reference_value, dim=0)
        raise RuntimeError(f"unexpected multi-tensor reference mapping for {model_key}")

    reference_tensor = reference_value
    if model_key.endswith(".noise_embedding.projection.weight") and reference_tensor.dim() == 1:
        return reference_tensor.unsqueeze(-1)
    return reference_tensor


def _remap_token_transformer_blocks(key: str) -> str:
    r"""
    Map a consolidated token transformer's per-block params onto Protenix ``...blocks.{i}...`` names.
    """

    rules = [
        (r"\.pair_norms\.(\d+)\.", r".blocks.\1.attention_pair_bias.layernorm_z."),
        (r"\.bias_projections\.(\d+)\.", r".blocks.\1.attention_pair_bias.linear_nobias_z."),
        (r"\.transformer\.layers\.(\d+)\.transition\.", r".blocks.\1.conditioned_transition_block."),
        (r"\.transformer\.layers\.(\d+)\.conditioned_norm\.", r".blocks.\1.attention_pair_bias.layernorm_a."),
        (r"\.transformer\.layers\.(\d+)\.attention\.", r".blocks.\1.attention_pair_bias.attention."),
        (r"\.transformer\.layers\.(\d+)\.output_gate_projection\.", r".blocks.\1.attention_pair_bias.linear_a_last."),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_top_level(key: str) -> str:
    r"""
    Rewrite a top-level module name (or its prefix) to the reference checkpoint's name.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with its leading module renamed to the reference convention (unchanged if none matches).
    """

    replacements = {
        "relative_position_embedder": "relative_position_encoding",
        "single_init_projection": "linear_no_bias_sinit",
        "pair_left_projection": "linear_no_bias_zinit1",
        "pair_right_projection": "linear_no_bias_zinit2",
        "token_bond_projection": "linear_no_bias_token_bond",
        "pair_recycle_projection": "linear_no_bias_z_cycle",
        "single_recycle_projection": "linear_no_bias_s",
        "pair_recycle_norm": "layernorm_z_cycle",
        "single_recycle_norm": "layernorm_s",
        "input_embedder.atom_encoder": "input_embedder.atom_attention_encoder",
        "msa_trunk": "msa_module",
        "coordinate_denoiser": "diffusion_module",
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return key


def _replace_template_key(key: str) -> str:
    r"""
    Rewrite template-embedder submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten key, or the input unchanged when it is not a template-embedder key.
    """

    if not key.startswith("template_embedder."):
        return key
    return _replace_many(
        key,
        {
            ".norm_pair.": ".layernorm_z.",
            ".pair_projection.": ".linear_no_bias_z.",
            ".template_projection.": ".linear_no_bias_a.",
            ".hidden_norm.": ".layernorm_v.",
            ".output_projection.": ".linear_no_bias_u.",
        },
    )


def _replace_confidence_key(key: str) -> str:
    r"""
    Rewrite confidence-head submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten key, or the input unchanged when it is not a confidence-head key.
    """

    if not key.startswith("confidence_head."):
        return key
    replacements = {
        "confidence_head.pair_embedder.lower_bins": "confidence_head.lower_bins",
        "confidence_head.pair_embedder.upper_bins": "confidence_head.upper_bins",
        "confidence_head.pair_embedder.left_token_projection": "confidence_head.linear_no_bias_s2",
        "confidence_head.pair_embedder.right_token_projection": "confidence_head.linear_no_bias_s1",
        "confidence_head.pair_embedder.distance_one_hot_projection": "confidence_head.linear_no_bias_d",
        "confidence_head.pair_embedder.distance_projection": "confidence_head.linear_no_bias_d_wo_onehot",
        "confidence_head.single_norm": "confidence_head.input_strunk_ln",
        "confidence_head.atom_projector.plddt_weight": "confidence_head.plddt_weight",
        "confidence_head.atom_projector.resolved_weight": "confidence_head.resolved_weight",
        "confidence_head.atom_projector.plddt_norm": "confidence_head.plddt_ln",
        "confidence_head.atom_projector.resolved_norm": "confidence_head.resolved_ln",
        "confidence_head.pair_projector.pae_norm": "confidence_head.pae_ln",
        "confidence_head.pair_projector.pde_norm": "confidence_head.pde_ln",
        "confidence_head.pair_projector.pae_projection": "confidence_head.linear_no_bias_pae",
        "confidence_head.pair_projector.pde_projection": "confidence_head.linear_no_bias_pde",
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return key


def _replace_token_stack_key(key: str) -> str:
    r"""
    Rewrite the token-encoder stack onto the reference diffusion-transformer block names.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten key, or the input unchanged when it holds no token-encoder segment.
    """

    if ".token_encoder." not in key:
        return key
    key = key.replace(".token_refiner.token_encoder.", ".diffusion_transformer.")
    return _remap_token_transformer_blocks(key)


def _replace_denoiser_key(key: str) -> str:
    r"""
    Rewrite diffusion-module (denoiser) submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten key, or the input unchanged when it is not a diffusion-module key.
    """

    if not key.startswith("diffusion_module."):
        return key
    replacements = {
        ".conditioner.relative_position_embedder.": ".diffusion_conditioning.relpe.",
        ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
        ".conditioner.pair_input_norm.": ".diffusion_conditioning.layernorm_z.",
        ".conditioner.pair_projection.": ".diffusion_conditioning.linear_no_bias_z.",
        ".conditioner.pair_feed_forward_1.": ".diffusion_conditioning.transition_z1.",
        ".conditioner.pair_feed_forward_2.": ".diffusion_conditioning.transition_z2.",
        ".conditioner.single_conditioning_norm.": ".diffusion_conditioning.layernorm_s.",
        ".conditioner.single_conditioning_projection.": ".diffusion_conditioning.linear_no_bias_s.",
        ".conditioner.noise_embedding.projection.weight": ".diffusion_conditioning.fourier_embedding.w",
        ".conditioner.noise_embedding.projection.bias": ".diffusion_conditioning.fourier_embedding.b",
        ".conditioner.noise_norm.": ".diffusion_conditioning.layernorm_n.",
        ".conditioner.noise_projection.": ".diffusion_conditioning.linear_no_bias_n.",
        ".conditioner.single_feed_forward_1.": ".diffusion_conditioning.transition_s1.",
        ".conditioner.single_feed_forward_2.": ".diffusion_conditioning.transition_s2.",
        ".atom_encoder.": ".atom_attention_encoder.",
        ".token_refiner.single_norm.": ".layernorm_s.",
        ".token_refiner.single_to_token_projection.": ".linear_no_bias_s.",
        ".token_refiner.token_encoder.": ".diffusion_transformer.",
        ".token_refiner.token_norm.": ".layernorm_a.",
        ".atom_decoder.": ".atom_attention_decoder.",
    }
    return _replace_many(key, replacements)


def _replace_atom_stack_key(key: str) -> str:
    r"""
    Rewrite the atom-attention stack's per-layer params onto the reference block names.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten key, or the input unchanged when it holds no atom-attention-stack segment.
    """

    if ".atom_attention_stack." not in key:
        return key
    block = r".atom_transformer.diffusion_transformer.blocks.\1"
    layer = r"\.atom_attention_stack\.atom_attention\.transformer\.layers\.(\d+)\."
    rules = [
        (r"\.atom_attention_stack\.pair_norms\.(\d+)\.", block + r".attention_pair_bias.layernorm_z."),
        (r"\.atom_attention_stack\.bias_projections\.(\d+)\.", block + r".attention_pair_bias.linear_nobias_z."),
        (layer + r"transition\.", block + r".conditioned_transition_block."),
        (layer + r"conditioned_norm\.", block + r".attention_pair_bias.layernorm_a."),
        (layer + r"key_value_norm\.", block + r".attention_pair_bias.layernorm_kv."),
        (layer + r"attention\.", block + r".attention_pair_bias.attention."),
        (layer + r"output_gate_projection\.", block + r".attention_pair_bias.linear_a_last."),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_atom_key(key: str) -> str:
    r"""
    Rewrite atom-encoder / atom-pair-encoder submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with matching atom-path segments substituted (unchanged if none match).
    """

    replacements = {
        ".atom_conditioner.reference_position_projection.": ".linear_no_bias_ref_pos.",
        ".atom_conditioner.reference_charge_projection.": ".linear_no_bias_ref_charge.",
        ".atom_conditioner.atom_feature_projection.": ".linear_no_bias_f.",
        ".atom_pair_encoder.distance_vector_projection.": ".linear_no_bias_d.",
        ".atom_pair_encoder.inverse_distance_projection.": ".linear_no_bias_invd.",
        ".atom_pair_encoder.distance_mask_projection.": ".linear_no_bias_v.",
        ".atom_pair_encoder.left_atom_projection.": ".linear_no_bias_cl.",
        ".atom_pair_encoder.right_atom_projection.": ".linear_no_bias_cm.",
        ".atom_pair_encoder.feed_forward.1.": ".small_mlp.1.",
        ".atom_pair_encoder.feed_forward.3.": ".small_mlp.3.",
        ".atom_pair_encoder.feed_forward.5.": ".small_mlp.5.",
        ".atom_attention_stack.": ".atom_transformer.diffusion_transformer.",
        ".atom_attention_encoder.token_projection.": ".atom_attention_encoder.linear_no_bias_q.",
        ".atom_attention_decoder.token_projection.": ".atom_attention_decoder.linear_no_bias_a.",
        ".single_norm.": ".layernorm_s.",
        ".single_to_atom_projection.": ".linear_no_bias_s.",
        ".atom_attention_encoder.norm_pair.": ".atom_attention_encoder.layernorm_z.",
        ".pair_projection.": ".linear_no_bias_z.",
        ".noisy_position_projection.": ".linear_no_bias_r.",
        ".atom_attention_decoder.output_projection.": ".atom_attention_decoder.linear_no_bias_out.",
        ".atom_attention_decoder.query_norm.": ".atom_attention_decoder.layernorm_q.",
    }
    return _replace_many(key, replacements)


def _replace_pairformer_key(key: str) -> str:
    r"""
    Rewrite Pairformer submodule names (triangle ops, transitions, attention) to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with matching Pairformer-path segments substituted (unchanged if none match).
    """

    replacements = {
        ".triangle_multiplication_outgoing.": ".tri_mul_out.",
        ".triangle_multiplication_incoming.": ".tri_mul_in.",
        ".triangle_attention_starting_node.": ".tri_att_start.",
        ".triangle_attention_ending_node.": ".tri_att_end.",
        ".pair_feed_forward.": ".pair_transition.",
        ".pair_attention.": ".attention_pair_bias.",
        ".single_feed_forward.": ".single_transition.",
        ".tri_mul_out.norm_input.": ".tri_mul_out.layer_norm_in.",
        ".tri_mul_in.norm_input.": ".tri_mul_in.layer_norm_in.",
        ".tri_mul_out.norm_output.": ".tri_mul_out.layer_norm_out.",
        ".tri_mul_in.norm_output.": ".tri_mul_in.layer_norm_out.",
        ".tri_mul_out.left_projection.": ".tri_mul_out.linear_a_p.",
        ".tri_mul_in.left_projection.": ".tri_mul_in.linear_a_p.",
        ".tri_mul_out.left_gate_projection.": ".tri_mul_out.linear_a_g.",
        ".tri_mul_in.left_gate_projection.": ".tri_mul_in.linear_a_g.",
        ".tri_mul_out.right_projection.": ".tri_mul_out.linear_b_p.",
        ".tri_mul_in.right_projection.": ".tri_mul_in.linear_b_p.",
        ".tri_mul_out.right_gate_projection.": ".tri_mul_out.linear_b_g.",
        ".tri_mul_in.right_gate_projection.": ".tri_mul_in.linear_b_g.",
        ".tri_mul_out.output_projection.": ".tri_mul_out.linear_z.",
        ".tri_mul_in.output_projection.": ".tri_mul_in.linear_z.",
        ".tri_mul_out.gate_projection.": ".tri_mul_out.linear_g.",
        ".tri_mul_in.gate_projection.": ".tri_mul_in.linear_g.",
        ".tri_att_start.norm_pair.": ".tri_att_start.layer_norm.",
        ".tri_att_end.norm_pair.": ".tri_att_end.layer_norm.",
        ".tri_att_start.bias_projection.": ".tri_att_start.linear.",
        ".tri_att_end.bias_projection.": ".tri_att_end.linear.",
        ".tri_att_start.attention.projection.": ".tri_att_start.mha.",
        ".tri_att_end.attention.projection.": ".tri_att_end.mha.",
    }
    return _replace_many(key, replacements)


def _replace_attention_key(key: str) -> str:
    r"""
    Rewrite attention-block norm and bias submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with matching attention-path segments substituted (unchanged if none match).
    """

    replacements = {
        ".pair_attention.": ".attention_pair_bias.",
        ".hidden_norm.": ".layernorm_a.",
        ".atom_norm.": ".layernorm_a.",
        ".key_value_norm.": ".layernorm_kv.",
        ".attention_pair_bias.norm_pair.": ".attention_pair_bias.layernorm_z.",
        ".attention_pair_bias.bias_projection.": ".attention_pair_bias.linear_nobias_z.",
        ".attention_pair_bias.conditioning_gate_projection.": ".attention_pair_bias.linear_a_last.",
    }
    return _replace_many(key, replacements)


def _replace_attention_projection_key(key: str) -> str:
    r"""
    Rewrite the q/k/v/o/gate projection and MSA-module names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with matching projection-path segments substituted (unchanged if none match).
    """

    replacements = {
        "relative_position_encoding.output_projection.": "relative_position_encoding.linear_no_bias.",
        ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
        ".fourier_embedding.frequency": ".fourier_embedding.w",
        ".fourier_embedding.phase": ".fourier_embedding.b",
        "msa_module.msa_projection.": "msa_module.linear_no_bias_m.",
        "msa_module.token_projection.": "msa_module.linear_no_bias_s.",
        ".outer_product_mean.": ".outer_product_mean_msa.",
        ".outer_product_mean_msa.norm_msa.": ".outer_product_mean_msa.layer_norm.",
        ".outer_product_mean_msa.left_projection.": ".outer_product_mean_msa.linear_1.",
        ".outer_product_mean_msa.right_projection.": ".outer_product_mean_msa.linear_2.",
        ".outer_product_mean_msa.output_projection.": ".outer_product_mean_msa.linear_out.",
        ".msa_pair_weighted_averaging.norm_msa.": ".msa_pair_weighted_averaging.layernorm_m.",
        ".msa_pair_weighted_averaging.value_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mv.",
        ".msa_pair_weighted_averaging.norm_pair.": ".msa_pair_weighted_averaging.layernorm_z.",
        ".msa_pair_weighted_averaging.pair_projection.": ".msa_pair_weighted_averaging.linear_no_bias_z.",
        ".msa_pair_weighted_averaging.gate_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mg.",
        ".msa_pair_weighted_averaging.output_projection.": ".msa_pair_weighted_averaging.linear_no_bias_out.",
        ".msa_stack.feed_forward.": ".msa_stack.transition_m.",
        ".pair_stack.feed_forward.": ".pair_stack.pair_transition.",
        ".attention.projection.query_projection.": ".attention.linear_q.",
        ".attention.projection.key_projection.": ".attention.linear_k.",
        ".attention.projection.value_projection.": ".attention.linear_v.",
        ".attention.projection.output_projection.": ".attention.linear_o.",
        ".attention.projection.gate_projection.": ".attention.linear_g.",
        ".mha.query_projection.": ".mha.linear_q.",
        ".mha.key_projection.": ".mha.linear_k.",
        ".mha.value_projection.": ".mha.linear_v.",
        ".mha.output_projection.": ".mha.linear_o.",
        ".mha.gate_projection.": ".mha.linear_g.",
        ".relative_position_encoding.output_projection.": ".relative_position_encoding.linear_no_bias.",
    }
    return _replace_many(key, replacements)


def _replace_transition_key(key: str) -> str:
    r"""
    Rewrite transition / conditioned-transition submodule names to the reference convention.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The key with matching transition-path segments substituted (unchanged if none match).
    """

    replacements = {
        ".norm_input.": ".layernorm1.",
        ".conditioned_norm.": ".adaln.",
        ".feed_forward.": ".conditioned_transition_block.",
        ".conditioned_transition_block.gate_projection.": ".conditioned_transition_block.linear_nobias_a1.",
        ".conditioned_transition_block.value_projection.": ".conditioned_transition_block.linear_nobias_a2.",
        ".conditioned_transition_block.output_projection.": ".conditioned_transition_block.linear_nobias_b.",
        ".conditioned_transition_block.conditioning_gate_projection.": ".conditioned_transition_block.linear_s.",
        ".pair_transition.gate_projection.": ".pair_transition.linear_no_bias_a.",
        ".pair_transition.value_projection.": ".pair_transition.linear_no_bias_b.",
        ".pair_transition.output_projection.": ".pair_transition.linear_no_bias.",
        ".single_transition.gate_projection.": ".single_transition.linear_no_bias_a.",
        ".single_transition.value_projection.": ".single_transition.linear_no_bias_b.",
        ".single_transition.output_projection.": ".single_transition.linear_no_bias.",
        ".transition_m.gate_projection.": ".transition_m.linear_no_bias_a.",
        ".transition_m.value_projection.": ".transition_m.linear_no_bias_b.",
        ".transition_m.output_projection.": ".transition_m.linear_no_bias.",
        ".transition_z1.gate_projection.": ".transition_z1.linear_no_bias_a.",
        ".transition_z1.value_projection.": ".transition_z1.linear_no_bias_b.",
        ".transition_z1.output_projection.": ".transition_z1.linear_no_bias.",
        ".transition_z2.gate_projection.": ".transition_z2.linear_no_bias_a.",
        ".transition_z2.value_projection.": ".transition_z2.linear_no_bias_b.",
        ".transition_z2.output_projection.": ".transition_z2.linear_no_bias.",
        ".transition_s1.gate_projection.": ".transition_s1.linear_no_bias_a.",
        ".transition_s1.value_projection.": ".transition_s1.linear_no_bias_b.",
        ".transition_s1.output_projection.": ".transition_s1.linear_no_bias.",
        ".transition_s2.gate_projection.": ".transition_s2.linear_no_bias_a.",
        ".transition_s2.value_projection.": ".transition_s2.linear_no_bias_b.",
        ".transition_s2.output_projection.": ".transition_s2.linear_no_bias.",
    }
    return _replace_many(key, replacements)


def _replace_conditioned_norm_key(key: str) -> ReferenceKey:
    r"""
    Rewrite adaptive-LayerNorm conditioning params to the Protenix reference names.

    Args:
        key: An OFoldX model parameter key.

    Returns:
        The rewritten reference key, either a single key or a tuple of keys to concatenate.
    """

    if key.endswith(".conditioning_projection.weight"):
        base = key[: -len("conditioning_projection.weight")]
        return (base + "linear_s.weight", base + "linear_nobias_s.weight")
    if key.endswith(".conditioning_scale_bias"):
        base = key[: -len("conditioning_scale_bias")]
        return base + "linear_s.bias"
    replacements = {
        ".hidden_norm.": ".layernorm_a.",
        ".conditioning_norm.": ".layernorm_s.",
        ".gate_projection.": ".linear_s.",
        ".shift_projection.": ".linear_nobias_s.",
    }
    return _replace_many(key, replacements)


def _replace_many(key: str, replacements: Mapping[str, str]) -> str:
    r"""
    Apply each ``source -> target`` substring substitution to ``key`` in order.

    Args:
        key: The key to rewrite.
        replacements: Ordered mapping of substring to its replacement.

    Returns:
        The key with every matching substring substituted.
    """

    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


if __name__ == "__main__":
    main()
