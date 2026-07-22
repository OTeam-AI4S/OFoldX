r"""
Boltz-2 checkpoint → OFoldX key converter.

OFoldX modules use clean, idiomatic parameter names; the public ``boltz2_conf.ckpt`` uses Boltz's
own names. This module keeps the Boltz-2 target routes and strict checkpoint assembly, while the
common released Boltz source-format rules live in ``ofoldx.models._boltz_conversion``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor

from ...artifacts import save_model
from .. import _boltz_conversion
from .config import Boltz2AffinityConfig, Boltz2Config
from .model import Boltz2AffinityModel, Boltz2Model
from .processor import Boltz2AffinityProcessor, Boltz2Processor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Boltz TemplateV2Module -> OFoldX TemplateV2Embedder.
TEMPLATE_FLAT_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^z_norm\.", "norm_pair."),
        (r"^v_norm\.", "hidden_norm."),
        (r"^z_proj\.", "pair_projection."),
        (r"^a_proj\.", "template_projection."),
        (r"^u_proj\.", "output_projection."),
    ]
)


def convert_template_module(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for the whole ``template_module.*`` (prefix-stripped).

    Args:
        key: A ``template_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX template-embedder names (empty when dropped).
    """

    if key.startswith("pairformer.layers."):
        index, _, remainder = key[len("pairformer.layers.") :].partition(".")
        return [(f"blocks.{index}.{nk}", nv) for nk, nv in _boltz_conversion.convert_pair_only_layer(remainder, tensor)]
    new_key = _boltz_conversion.convert_keys(key, TEMPLATE_FLAT_RULES)
    return [] if new_key is None else [(new_key, tensor)]


# Token-level encodings + conditioning embeddings at the input-embedder top level.
INPUT_EMBEDDER_FLAT_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^res_type_encoding\.", "residue_type_projection."),
        (r"^msa_profile_encoding\.", "msa_profile_projection."),
        (r"^method_conditioning_init\.", "method_embedding."),
        (r"^modified_conditioning_init\.", "modified_embedding."),
        (r"^cyclic_conditioning_init\.", "cyclic_projection."),
        (r"^mol_type_conditioning_init\.", "molecule_type_embedding."),
        (r"^atom_enc_proj_z\.", "atom_pair_bias_projection."),
        (r"^atom_attention_encoder\.atom_to_token_trans\.", "atom_attention_encoder.token_projection."),
    ]
)

_ATTN_OLD = "atom_attention_encoder.atom_encoder.diffusion_transformer.layers."
_ATTN_NEW = "atom_attention_encoder.atom_transformer.transformer.layers."


def convert_input_embedder_key(key: str) -> str | None:
    r"""
    Route a Boltz ``input_embedder.*`` (prefix-stripped) key to its OFoldX name.

    Args:
        key: An ``input_embedder``-relative Boltz key.

    Returns:
        The OFoldX key, or ``None`` to drop it.
    """

    if key.startswith("atom_encoder."):
        return _boltz_conversion.reprefix(key, "atom_encoder.", "atom_encoder.", _boltz_conversion.ATOM_ENCODER_RULES)
    if key.startswith(_ATTN_OLD):
        return reprefix_indexed(key, _ATTN_OLD, _ATTN_NEW, _boltz_conversion.CONDITIONED_TRANSFORMER_LAYER_RULES)
    return _boltz_conversion.convert_keys(key, INPUT_EMBEDDER_FLAT_RULES)


def reprefix_indexed(key: str, old_prefix: str, new_prefix: str, sub: _boltz_conversion.Converter) -> str | None:
    r"""
    Like ``reprefix`` but preserves a ``<index>.`` segment that follows the prefix.

    Args:
        key: The full key beginning with ``old_prefix`` followed by an ``<index>.`` segment.
        old_prefix: The prefix to strip before the index.
        new_prefix: The prefix to prepend to the ``<index>.`` and converted remainder.
        sub: Converter applied to the remainder after the index.

    Returns:
        ``new_prefix<index>.`` + converted remainder, or ``None`` when ``sub`` drops the remainder.
    """

    index, _, remainder = key[len(old_prefix) :].partition(".")
    converted = _boltz_conversion.apply_converter(remainder, sub)
    return None if converted is None else f"{new_prefix}{index}.{converted}"


_AFFINITY_PREFIXES = ("affinity_module1.", "affinity_module2.")


CONFIDENCE_FLAT_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "distance_embedding."),
        (r"^s_inputs_norm\.", "input_single_norm."),
        (r"^s_norm\.", "single_norm."),
        (r"^z_norm\.", "norm_pair."),
        (r"^s_input_to_s\.", "input_single_projection."),
        (r"^s_to_z_transpose\.", "single_to_pair_col."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^s_to_z_prod_out\.", "single_to_pair_product_out."),
        (r"^s_to_z\.", "single_to_pair_row."),
        (r"^rel_pos\.linear_layer\.", "relative_position_embedder.output_projection."),
        (r"^token_bonds_type\.", "token_bond_type_embedding."),
        (r"^token_bonds\.", "token_bond_projection."),
        (r"^boundaries$", "boundaries"),
    ]
)
CONFIDENCE_HEADS_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^to_pae_intra_logits\.", "pae_intra_projection."),
        (r"^to_pae_inter_logits\.", "pae_inter_projection."),
        (r"^to_pde_intra_logits\.", "pde_intra_projection."),
        (r"^to_pde_inter_logits\.", "pde_inter_projection."),
        (r"^to_plddt_logits\.", "plddt_projection."),
        (r"^to_resolved_logits\.", "resolved_projection."),
    ]
)


def convert_confidence_head(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for source ``confidence_module.*`` (prefix-stripped).

    Args:
        key: A ``confidence_module``-relative Boltz key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX confidence-head names (empty when dropped).
    """

    if key.startswith("contact_conditioning."):
        new = _boltz_conversion.convert_keys(
            key[len("contact_conditioning.") :], _boltz_conversion.CONTACT_CONDITIONING_RULES
        )
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"pairformer_stack.blocks.{index}.{nk}", nv)
            for nk, nv in _boltz_conversion.convert_pairformer_layer(remainder, tensor)
        ]
    if key.startswith("confidence_heads."):
        new = _boltz_conversion.convert_keys(key[len("confidence_heads.") :], CONFIDENCE_HEADS_RULES)
        return [("confidence_heads." + new, tensor)] if new is not None else []
    new = _boltz_conversion.convert_keys(key, CONFIDENCE_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


AFFINITY_FLAT_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^dist_bin_pairwise_embed\.", "dist_bin_pairwise_embed."),
        (r"^s_to_z_prod_in1\.", "single_to_pair_product_left."),
        (r"^s_to_z_prod_in2\.", "single_to_pair_product_right."),
        (r"^z_norm\.", "pair_norm."),
        (r"^z_linear\.", "pair_linear."),
        (r"^boundaries$", "boundaries"),
    ]
)
AFFINITY_HEADS_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^affinity_out_mlp\.", "affinity_out_mlp."),
        (r"^to_affinity_pred_value\.", "to_affinity_pred_value."),
        (r"^to_affinity_pred_score\.", "to_affinity_pred_score."),
        (r"^to_affinity_logits_binary\.", "to_affinity_logits_binary."),
    ]
)
AFFINITY_PAIR_ONLY_LAYER_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        *_boltz_conversion.triangle_multiplication_rules("tri_mul_out", "triangle_multiplication_outgoing"),
        *_boltz_conversion.triangle_multiplication_rules("tri_mul_in", "triangle_multiplication_incoming"),
        *_boltz_conversion.triangle_attention_rules("tri_att_start", "triangle_attention_starting_node"),
        *_boltz_conversion.triangle_attention_rules("tri_att_end", "triangle_attention_ending_node"),
        *_boltz_conversion.transition_rules("transition_z", "pair_feed_forward"),
    ]
)


def convert_affinity_pairformer_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for an affinity-head pair-only Pairformer layer.

    The affinity head uses ``PairformerStack(single_size=0)``, whose pair transition remains named
    ``pair_feed_forward``. Template/MSA pair-only stacks use ``TrianglePairUpdateBlock.feed_forward`` instead.
    """

    split = _boltz_conversion.split_fused_projection(key, tensor)
    if split is not None:
        return split
    new_key = _boltz_conversion.convert_keys(key, AFFINITY_PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


def convert_affinity_head(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for a prefix-stripped upstream ``affinity_module*.*`` key.

    Args:
        key: Checkpoint key with the ``affinity_module*.`` prefix already stripped.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the OFoldX affinity-head names.
    """

    if key.startswith("pairwise_conditioner."):
        new = _boltz_conversion.convert_keys(
            key[len("pairwise_conditioner.") :], _boltz_conversion.PAIRWISE_CONDITIONING_RULES
        )
        return [("pairwise_conditioner." + new, tensor)] if new is not None else []
    if key.startswith("pairformer_stack.layers."):
        index, _, remainder = key[len("pairformer_stack.layers.") :].partition(".")
        return [
            (f"pairformer_stack.blocks.{index}.{nk}", nv)
            for nk, nv in convert_affinity_pairformer_layer(remainder, tensor)
        ]
    if key.startswith("affinity_heads."):
        new = _boltz_conversion.convert_keys(key[len("affinity_heads.") :], AFFINITY_HEADS_RULES)
        return [("affinity_heads." + new, tensor)] if new is not None else []
    new = _boltz_conversion.convert_keys(key, AFFINITY_FLAT_RULES)
    return [(new, tensor)] if new is not None else []


def convert_affinity_modules(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert upstream affinity head weights into ``Boltz2AffinityModel`` module names.

    Args:
        state_dict: A full Boltz-2 affinity checkpoint ``state_dict``.

    Returns:
        Converted affinity head tensors under ``affinity_module*``.
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for module in ("affinity_module1", "affinity_module2"):
            prefix = module + "."
            if key.startswith(prefix):
                entries = convert_affinity_head(key[len(prefix) :], value)
                if not entries:
                    raise KeyError(f"unmapped affinity checkpoint key: {key}")
                for new_key, new_value in entries:
                    converted[f"{module}.{new_key}"] = new_value
                break
    return converted


def convert_trunk_key(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Map one Boltz checkpoint key to its ``Boltz2Model`` trunk param(s); ``[]`` drops non-trunk keys.

    Routes the key to the per-subsystem converter matching its prefix (input embedder, template, MSA,
    Pairformer, contact conditioning, distogram, diffusion, score model, confidence, B-factor) or the top-level
    trunk rename table.

    Args:
        key: A full Boltz checkpoint key.
        tensor: The associated weight tensor.

    Returns:
        A list of ``(key, tensor)`` pairs under the ``Boltz2Model`` param names; empty for non-trunk keys.
    """

    if key.startswith("input_embedder."):
        new = convert_input_embedder_key(key[len("input_embedder.") :])
        return [("input_embedder." + new, tensor)] if new is not None else []
    if key.startswith("template_module."):
        inner = convert_template_module(key[len("template_module.") :], tensor)
        return [("template_embedder." + nk, nv) for nk, nv in inner]
    if key.startswith("msa_module."):
        inner = _boltz_conversion.convert_msa_module(key[len("msa_module.") :], tensor)
        return [("msa_module." + nk, nv) for nk, nv in inner]
    if key.startswith("pairformer_module.layers."):
        index, _, remainder = key[len("pairformer_module.layers.") :].partition(".")
        inner = _boltz_conversion.convert_pairformer_layer(remainder, tensor)
        return [(f"pairformer_stack.blocks.{index}.{nk}", nv) for nk, nv in inner]
    if key.startswith("contact_conditioning."):
        new = _boltz_conversion.convert_keys(
            key[len("contact_conditioning.") :], _boltz_conversion.CONTACT_CONDITIONING_RULES
        )
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("distogram_module."):
        new = _boltz_conversion.convert_keys(key[len("distogram_module.") :], _boltz_conversion.DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = _boltz_conversion.convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("diffusion_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = _boltz_conversion.convert_score_model(key[len("structure_module.score_model.") :])
        return [("structure_module.score_model." + new, tensor)] if new is not None else []
    if key.startswith("confidence_module."):
        inner = convert_confidence_head(key[len("confidence_module.") :], tensor)
        return [("confidence_head." + nk, nv) for nk, nv in inner]
    if key.startswith("bfactor_module."):
        new = _boltz_conversion.convert_keys(key[len("bfactor_module.") :], _boltz_conversion.BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in _boltz_conversion.TRUNK_FLAT:
        return [(_boltz_conversion.TRUNK_FLAT[key], tensor)]
    return []


def convert_trunk(state_dict: Mapping[str, torch.Tensor], *, allow_affinity: bool = False) -> dict[str, torch.Tensor]:
    r"""
    Convert a full Boltz-2 ``state_dict`` to the ``Boltz2Model`` parameter tree.

    Applies [`convert_trunk_key`][ofoldx.models.boltz2.convert.convert_trunk_key] to every entry, then packs
    the AdaLN scale/shift projections so the result ``strict``-loads into the OFoldX modules. Unknown keys fail;
    ``allow_affinity`` skips only the two exact affinity-module prefixes handled by the affinity converter.

    Args:
        state_dict: A full Boltz-2 checkpoint ``state_dict``.
        allow_affinity: Whether the released affinity-module prefixes may be routed by a separate converter.

    Returns:
        The converted trunk ``state_dict`` under the ``Boltz2Model`` param names.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> ckpt = torch.load("boltz2_conf.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_trunk(ckpt)  # doctest: +SKIP
        >>> model.load_state_dict(converted, strict=True)  # doctest: +SKIP
    """

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        entries = convert_trunk_key(key, value)
        if not entries:
            if _boltz_conversion.is_tied_output_projection(key):
                continue
            if allow_affinity and key.startswith(_AFFINITY_PREFIXES):
                continue
            raise KeyError(f"unmapped Boltz-2 checkpoint key: {key}")
        for new_key, new_value in entries:
            converted[new_key] = new_value
    _pack_conditioned_adaptive_norms(converted)
    return converted


def _pack_conditioned_adaptive_norms(state_dict: dict[str, torch.Tensor]) -> None:
    r"""Pack upstream Boltz AdaLN scale/shift projections into the shared fused layout."""

    # The fused module splits scale first and shift second; only upstream s_scale carries a bias.
    scale_weight_suffix = ".conditioned_norm.gate_projection.weight"
    for scale_weight_key in tuple(key for key in state_dict if key.endswith(scale_weight_suffix)):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state_dict.pop(scale_weight_key)
        scale_bias = state_dict.pop(prefix + "gate_projection.bias")
        shift_weight = state_dict.pop(prefix + "shift_projection.weight")
        state_dict[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state_dict[prefix + "conditioning_scale_bias"] = scale_bias


def convert_boltz2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2Model:
    r"""Convert a released Boltz-2 structure-prediction checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    converted = _cast_floating_tensors(convert_trunk(checkpoint["state_dict"]), dtype)
    with torch.device("meta"):
        model = Boltz2Model(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_boltz2_affinity_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: Boltz2AffinityConfig | None = None,
    dtype: torch.dtype | None = None,
) -> Boltz2AffinityModel:
    r"""Convert the released two-head Boltz-2 affinity checkpoint."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = convert_trunk(state, allow_affinity=True)
    converted.update(convert_affinity_modules(state))
    converted = _cast_floating_tensors(converted, dtype)
    with torch.device("meta"):
        model = Boltz2AffinityModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def _cast_floating_tensors(
    state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    r"""Return ``state_dict`` with floating-point tensors cast to ``dtype`` when requested."""

    return {
        key: value.to(dtype=dtype) if dtype is not None and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def save_boltz2_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert a Boltz-2 structure-prediction checkpoint and write a standard OFoldX repository.

    Strict-loads the checkpoint via
    [`convert_boltz2_checkpoint`][ofoldx.models.boltz2.convert.convert_boltz2_checkpoint], pairs the
    model with the config-less :class:`Boltz2Processor`, and writes a self-contained ``model.json`` /
    safetensors repository via [`ofoldx.artifacts.save_model`][ofoldx.artifacts.save_model]. This is
    the standard ``fold`` pair (``ofoldx.boltz2``); the Processor carries no config, so no
    ``processor.json`` is written.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository. No
    organization or token is assumed and nothing is published without an explicit ``repo_id``.
    """

    model = convert_boltz2_checkpoint(checkpoint_path, config=config, dtype=dtype)
    save_model(
        output_dir,
        model=model,
        processor=Boltz2Processor(),
        repo_id=repo_id,
        token=token,
        private=private,
    )


def save_boltz2_affinity_checkpoint(
    affinity_checkpoint_path: str | PathLike[str],
    structure_checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    affinity_config: Boltz2AffinityConfig | None = None,
    structure_config: Boltz2Config | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert the Boltz-2 affinity + structure checkpoints and write the composite repository.

    The affinity deliverable is a multi-pair :class:`Boltz2AffinityPipeline`: the affinity
    head (``ofoldx.boltz2_affinity``, ``standard_tasks=()``) runs only after a structure/confidence
    pass, so it cannot route as a lone standard pair. This writes the aggregate repository — the
    affinity pair at the root, the structure/confidence pair (:class:`Boltz2Model` /
    :class:`Boltz2Processor`) at the fixed ``structure/`` sub-directory, and the one-field
    ``pipeline.json`` marker — via ``Boltz2AffinityPipeline.save`` so ``pipeline(..., model=dir)``
    routes it back through the multi-pair loader.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the composite repository.
    """

    from .pipeline import Boltz2AffinityPipeline

    affinity_model = convert_boltz2_affinity_checkpoint(
        affinity_checkpoint_path,
        config=affinity_config,
        dtype=dtype,
    )
    structure_model = convert_boltz2_checkpoint(
        structure_checkpoint_path,
        config=structure_config,
        dtype=dtype,
    )
    Boltz2AffinityPipeline(
        model=affinity_model,
        processor=Boltz2AffinityProcessor(),
        structure_model=structure_model,
        structure_processor=Boltz2Processor(),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fold = subparsers.add_parser("fold", help="convert the Boltz-2 structure/confidence checkpoint")
    fold.add_argument("checkpoint", type=Path, help="released Boltz-2 fold .ckpt")
    fold.add_argument("output_dir", type=Path)
    fold.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    fold.add_argument("--repo-id", default=None, help="optional Hub repo id; local-only when omitted")
    fold.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    fold.add_argument("--private", action="store_true", help="create the --repo-id repo as private")

    affinity = subparsers.add_parser("affinity", help="convert the Boltz-2 affinity + structure checkpoints")
    affinity.add_argument("affinity_checkpoint", type=Path, help="released Boltz-2 affinity .ckpt")
    affinity.add_argument("structure_checkpoint", type=Path, help="released Boltz-2 structure/confidence .ckpt")
    affinity.add_argument("output_dir", type=Path)
    affinity.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    affinity.add_argument("--repo-id", default=None, help="optional Hub repo id; local-only when omitted")
    affinity.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    affinity.add_argument("--private", action="store_true", help="create the --repo-id repo as private")

    args = parser.parse_args(argv)
    dtype = None if args.dtype == "keep" else _DTYPES[args.dtype]
    if args.command == "fold":
        save_boltz2_checkpoint(
            args.checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )
    else:
        save_boltz2_affinity_checkpoint(
            args.affinity_checkpoint,
            args.structure_checkpoint,
            args.output_dir,
            dtype=dtype,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
        )


if __name__ == "__main__":
    main()
