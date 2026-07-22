r"""Convert released ODesign checkpoints into paired OFoldX repositories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn

from .. import _protenix_conversion
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

    used_reference_keys: set[str] = set()
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
    return _protenix_conversion.translate_reference_state(model, reference_state, odesign_reference_key)


def _infer_receptor_mode(checkpoint_path: str | Path) -> Literal["rigid", "flex"]:
    r"""Infer the released receptor mode from a ``*_flex`` / ``*_rigid`` checkpoint filename.

    The two protein archives ``odesign_base_prot_flex.pt`` and
    ``odesign_base_prot_rigid.pt`` share one graph and state-dict contract; they
    differ only in receptor conditioning, which is carried by the Processor's
    ``receptor_mode``. The filename names it unambiguously; a name carrying neither
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

    The ODesign protein repository contains a multi-pair :class:`ODesignPipeline`: the ODesign
    backbone pair owns the runnable root, and the OInvFold sequence step uses the
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

    ``receptor_mode`` selects the flex/rigid conditioning written into the root pair's
    ``processor.json``; it is inferred from the checkpoint filename when not given.

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


def odesign_reference_key(model_key: str) -> _protenix_conversion.ReferenceKey:
    r"""Return the released-checkpoint key for one OFoldX ODesign state key."""

    return _protenix_conversion.reference_key(
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


if __name__ == "__main__":
    main()
