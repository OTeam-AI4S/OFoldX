r"""
Checkpoint converter: BoltzGen inverse-folding checkpoint -> OFoldX clean names.

Routes the three top-level prefixes of ``boltzgen1_ifold.ckpt`` (input_embedder / inverse_folding_encoder /
structure_module) to the matching OFoldX modules. The input embedder reuses the Boltz-2 atom converters;
the encoder/decoder GNN layers use the clean MLP/transition names. Unmapped keys raise, and the resulting
state dictionary is suitable for strict loading.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .. import _boltz_conversion
from .config import BoltzGenConfig, BoltzGenIFoldConfig
from .model import BoltzGenIFoldModel, BoltzGenModel
from .processor import BoltzGenIFoldProcessor, BoltzGenProcessor

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

MINIFORMER_LAYER_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [
        (r"^pre_norm_s\.", "attention.hidden_norm."),
        (r"^attention\.proj_q\.", "attention.attention.projection.query_projection."),
        (r"^attention\.proj_k\.", "attention.attention.projection.key_projection."),
        (r"^attention\.proj_v\.", "attention.attention.projection.value_projection."),
        (r"^attention\.proj_g\.", "attention.attention.projection.gate_projection."),
        (r"^attention\.proj_o\.", "attention.attention.projection.output_projection."),
        (r"^attention\.proj_z\.0\.", "attention.norm_pair."),
        (r"^attention\.proj_z\.1\.", "attention.bias_projection."),
        *_boltz_conversion.transition_rules("transition_s", "transition_single"),
        *_boltz_conversion.transition_rules("transition_z", "transition_pair"),
    ]
)


def convert_miniformer_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one compact Miniformer layer.

    Passes ``triangular.*`` keys through unchanged (the compact triangular update keeps upstream names); otherwise
    applies the Miniformer rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("triangular."):
        return [(key, tensor)]
    new_key = _boltz_conversion.convert_keys(key, MINIFORMER_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


MINIFORMER_PAIR_ONLY_LAYER_RULES: list[_boltz_conversion.Rule] = _boltz_conversion.compile_rules(
    [*_boltz_conversion.transition_rules("transition_z", "transition_pair")]
)


def convert_miniformer_pair_only_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one compact pair-only Miniformer layer.

    Passes ``triangular.*`` keys through unchanged; otherwise applies the compact pair-only rename rules.

    Args:
        key: Checkpoint key relative to the layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is dropped).
    """

    if key.startswith("triangular."):
        return [(key, tensor)]
    new_key = _boltz_conversion.convert_keys(key, MINIFORMER_PAIR_ONLY_LAYER_RULES)
    return [] if new_key is None else [(new_key, tensor)]


def convert_miniformer_msa_layer(key: str, tensor: Tensor) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for one MSA layer whose pair update uses MiniformerNoSeq blocks.

    Same routing as [`convert_msa_layer`][ofoldx.models.boltzgen.convert.convert_msa_layer] but the pair-only
    sub-block uses the compact Miniformer converter.

    Args:
        key: Checkpoint key relative to the MSA-layer root.
        tensor: The associated weight tensor.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA sub-block.
    """

    if key.startswith("pair_weighted_averaging."):
        return [
            (
                cast(
                    str,
                    _boltz_conversion.reprefix(
                        key,
                        "pair_weighted_averaging.",
                        "pair_weighted_averaging.",
                        _boltz_conversion.MSA_PAIR_WEIGHTED_AVERAGING_RULES,
                    ),
                ),
                tensor,
            )
        ]
    if key.startswith("msa_transition."):
        return [
            (
                cast(
                    str,
                    _boltz_conversion.reprefix(
                        key, "msa_transition.", "msa_transition.", _boltz_conversion.MSA_TRANSITION_RULES
                    ),
                ),
                tensor,
            )
        ]
    if key.startswith("outer_product_mean."):
        return [
            (
                cast(
                    str,
                    _boltz_conversion.reprefix(
                        key, "outer_product_mean.", "outer_product_mean.", _boltz_conversion.OUTER_PRODUCT_MEAN_RULES
                    ),
                ),
                tensor,
            )
        ]
    if key.startswith("pairformer_layer."):
        inner = convert_miniformer_pair_only_layer(key[len("pairformer_layer.") :], tensor)
        return [(f"triangle_pair_update.{nk}", nv) for nk, nv in inner]
    raise KeyError(f"unmapped Miniformer MSA layer key: {key}")


def convert_msa_module(
    key: str,
    tensor: Tensor,
    *,
    miniformer_blocks: bool = False,
) -> list[tuple[str, Tensor]]:
    r"""
    Tensor-aware converter for the whole MSA module.

    Renames the token / MSA input projections and dispatches per-layer keys to the Pairformer or Miniformer MSA
    layer converter.

    Args:
        key: Checkpoint key with the ``msa_module.`` prefix already stripped.
        tensor: The associated weight tensor.
        miniformer_blocks: Whether the layers use MiniformerNoSeq pair updates.

    Returns:
        List of ``(new_key, tensor)`` pairs.

    Raises:
        KeyError: If the key does not match any known MSA-module sub-block.
    """

    if not miniformer_blocks:
        return _boltz_conversion.convert_msa_module(key, tensor)
    if key.startswith("s_proj."):
        return [(key.replace("s_proj.", "token_projection.", 1), tensor)]
    if key.startswith("msa_proj."):
        return [(key.replace("msa_proj.", "msa_projection.", 1), tensor)]
    if key.startswith("layers."):
        index, _, remainder = key[len("layers.") :].partition(".")
        return [(f"layers.{index}.{nk}", nv) for nk, nv in convert_miniformer_msa_layer(remainder, tensor)]
    raise KeyError(f"unmapped MSA module key: {key}")


def convert_trunk_key(
    key: str,
    tensor: Tensor,
    *,
    miniformer_pairformer: bool = False,
    miniformer_msa: bool = False,
) -> list[tuple[str, Tensor]]:
    r"""
    Convert one structure-design trunk checkpoint key onto the OFoldX trunk modules.

    Dispatches the shared backbone prefixes (contact conditioning, MSA module, Pairformer stack, distogram,
    diffusion conditioning, score model, B-factor head) to their sub-converters and applies the flat trunk renames
    to the top-level projections.

    Args:
        key: Checkpoint key relative to the trunk root.
        tensor: The associated weight tensor.
        miniformer_pairformer: Whether the Pairformer stack uses Miniformer layers.
        miniformer_msa: Whether the MSA module uses Miniformer pair updates.

    Returns:
        List of ``(new_key, tensor)`` pairs (empty when the key is not part of the trunk).
    """

    if key.startswith("contact_conditioning."):
        new = _boltz_conversion.convert_keys(
            key[len("contact_conditioning.") :], _boltz_conversion.CONTACT_CONDITIONING_RULES
        )
        return [("contact_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("msa_module."):
        inner = convert_msa_module(key[len("msa_module.") :], tensor, miniformer_blocks=miniformer_msa)
        return [(f"msa_module.{new_key}", new_value) for new_key, new_value in inner]
    if key.startswith(("pairformer_stack.layers.", "pairformer_module.layers.")):
        prefix = (
            "pairformer_stack.layers." if key.startswith("pairformer_stack.layers.") else "pairformer_module.layers."
        )
        index, _, remainder = key[len(prefix) :].partition(".")
        converter = convert_miniformer_layer if miniformer_pairformer else _boltz_conversion.convert_pairformer_layer
        return [
            (f"pairformer_stack.blocks.{index}.{new_key}", new_value)
            for new_key, new_value in converter(remainder, tensor)
        ]
    if key.startswith("distogram_module."):
        new = _boltz_conversion.convert_keys(key[len("distogram_module.") :], _boltz_conversion.DISTOGRAM_RULES)
        return [("distogram_head." + new, tensor)] if new is not None else []
    if key.startswith("diffusion_conditioning."):
        new = _boltz_conversion.convert_diffusion_conditioning(key[len("diffusion_conditioning.") :])
        return [("diffusion_conditioning." + new, tensor)] if new is not None else []
    if key.startswith("structure_module.score_model."):
        new = _boltz_conversion.convert_score_model(key[len("structure_module.score_model.") :])
        return [("structure_module.score_model." + new, tensor)] if new is not None else []
    if key.startswith("bfactor_module."):
        new = _boltz_conversion.convert_keys(key[len("bfactor_module.") :], _boltz_conversion.BFACTOR_RULES)
        return [("bfactor_head." + new, tensor)] if new is not None else []
    if key in _boltz_conversion.TRUNK_FLAT:
        return [(_boltz_conversion.TRUNK_FLAT[key], tensor)]
    return []


INPUT_EMBEDDER_CONDITIONING = {
    "atom_enc_proj_z": "atom_pair_bias_projection",
    "res_type_encoding": "residue_type_projection",
    "msa_profile_encoding": "msa_profile_projection",
    "method_conditioning_init": "method_embedding",
    "modified_conditioning_init": "modified_embedding",
    "cyclic_conditioning_init": "cyclic_projection",
    "mol_type_conditioning_init": "molecule_type_embedding",
    "binding_specification_conditioning_init": "binding_specification_embedding",
    "design_mask_conditioning_init": "design_mask_embedding",
    "ss_specification_init": "ss_specification_embedding",
}


def convert_input_embedder(key: str) -> str | None:
    r"""
    Convert one ``input_embedder.*`` key onto its AtomConditionedInputEmbedder name.

    Args:
        key: Checkpoint key with the ``input_embedder.`` prefix already stripped.

    Returns:
        The renamed key, or ``None`` to drop the tied duplicate.
    """
    if key.startswith("atom_encoder."):
        inner = _boltz_conversion.convert_keys(key[len("atom_encoder.") :], _boltz_conversion.ATOM_ENCODER_RULES)
        return None if inner is None else "atom_encoder." + inner
    if key.startswith("atom_attention_encoder."):
        inner = _boltz_conversion.convert_atom_attention_encoder(key[len("atom_attention_encoder.") :])
        return None if inner is None else "atom_attention_encoder." + inner
    head = key.split(".")[0]
    if head in INPUT_EMBEDDER_CONDITIONING:
        return INPUT_EMBEDDER_CONDITIONING[head] + key[len(head) :]
    return key


GNN_LAYER_RENAME = {
    "attn_weight_mlp": "attention_weight_mlp",
    "attn_value_mlp": "attention_value_mlp",
    "attn_output_linear": "attention_output",
    "attn_FFN": "node_transition",
    "edge_FFN": "edge_transition",
}


def _convert_gnn_layer(suffix: str) -> str:
    head, rest = suffix.split(".", 1)
    return f"{GNN_LAYER_RENAME.get(head, head)}.{rest}"


ENCODER_TOP = {
    "linear_token_to_node": "node_input_projection",
    "linear_token_to_pair": "edge_input_projection",
    "distance_gaussian_smearing": "distance_smearing",
}

DECODER_TOP = {
    "seq_to_s": "residue_type_projection",
    "predictor": "residue_type_head",
}


def convert_encoder(key: str) -> str:
    r"""
    Convert one ``inverse_folding_encoder.*`` key onto its InverseFoldingEncoder name.

    Args:
        key: Checkpoint key with the ``inverse_folding_encoder.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("encoder_layers."):
        index, suffix = key[len("encoder_layers.") :].split(".", 1)
        return f"layers.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return ENCODER_TOP.get(head, head) + key[len(head) :]


def convert_decoder(key: str) -> str:
    r"""
    Convert one ``structure_module.*`` key onto its InverseFoldingDecoder name.

    Args:
        key: Checkpoint key with the ``structure_module.`` prefix already stripped.

    Returns:
        The renamed key.
    """
    if key.startswith("decoder_layers."):
        index, suffix = key[len("decoder_layers.") :].split(".", 1)
        return f"layers.{index}.{_convert_gnn_layer(suffix)}"
    head = key.split(".")[0]
    return DECODER_TOP.get(head, head) + key[len(head) :]


TOKEN_DISTANCE_TOP = {
    "z_norm": "norm_pair",
    "v_norm": "hidden_norm",
    "z_proj": "pair_projection",
    "a_proj": "distance_feature_projection",
    "u_proj": "output_projection",
}

TOKEN_DISTANCE_ENCODER = {
    "distance_gaussian_smearing": "distance_smearing",
    "distance_token_bias_trans.fc1": "distance_token_bias_trans.gate_projection",
    "distance_token_bias_trans.fc2": "distance_token_bias_trans.value_projection",
    "distance_token_bias_trans.fc3": "distance_token_bias_trans.output_projection",
}


def convert_token_distance(state: dict) -> dict:
    r"""
    Convert a ``token_distance_module.*`` checkpoint subset onto the TokenDistanceModule (clean names).

    The pair-only pairformer layers are split tensor-aware; the distance encoder and top-level projections use the
    local rename maps.

    Args:
        state: Mapping of ``token_distance_module``-relative keys to tensors.

    Returns:
        New mapping of converted keys to tensors.
    """
    out: dict = {}
    for key, value in state.items():
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            for new_suffix, new_value in _boltz_conversion.convert_pair_only_layer(suffix, value):
                out[f"pairformer.{index}.{new_suffix}"] = new_value
        elif key.startswith("token_distance_encoder."):
            inner = key[len("token_distance_encoder.") :]
            for ref, clean in TOKEN_DISTANCE_ENCODER.items():
                if inner.startswith(ref):
                    inner = clean + inner[len(ref) :]
                    break
            out["distance_encoder." + inner] = value
        else:
            head = key.split(".")[0]
            out[TOKEN_DISTANCE_TOP.get(head, head) + key[len(head) :]] = value
    return out


def convert_template(state: dict, *, miniformer_blocks: bool = False) -> dict:
    r"""
    Convert a ``template_module.*`` (BoltzGen v1) checkpoint subset onto the TemplateModule (clean names).

    The pair-only pairformer layers are split tensor-aware (Miniformer variant when ``miniformer_blocks`` is set);
    the top-level projections use the local rename map.

    Args:
        state: Mapping of ``template_module``-relative keys to tensors.
        miniformer_blocks: Whether the template pairformer uses compact Miniformer pair updates.

    Returns:
        New mapping of converted keys to tensors.
    """
    rename = {**TOKEN_DISTANCE_TOP, "a_proj": "template_feature_projection"}
    out: dict = {}
    for key, value in state.items():
        if key.startswith("pairformer.layers."):
            index, suffix = key[len("pairformer.layers.") :].split(".", 1)
            converter = (
                convert_miniformer_pair_only_layer if miniformer_blocks else _boltz_conversion.convert_pair_only_layer
            )
            for new_suffix, new_value in converter(suffix, value):
                out[f"pairformer.{index}.{new_suffix}"] = new_value
        else:
            head = key.split(".")[0]
            out[rename.get(head, head) + key[len(head) :]] = value
    return out


def _pack_conditioned_adaptive_norms(state_dict: dict[str, torch.Tensor]) -> None:
    r"""Pack upstream BoltzGen AdaLN scale/shift projections into the shared fused layout."""

    scale_weight_suffix = ".conditioned_norm.gate_projection.weight"
    for scale_weight_key in tuple(key for key in state_dict if key.endswith(scale_weight_suffix)):
        prefix = scale_weight_key.removesuffix("gate_projection.weight")
        scale_weight = state_dict.pop(scale_weight_key)
        scale_bias = state_dict.pop(prefix + "gate_projection.bias")
        shift_weight = state_dict.pop(prefix + "shift_projection.weight")
        state_dict[prefix + "conditioning_projection.weight"] = torch.cat((scale_weight, shift_weight), dim=0)
        state_dict[prefix + "conditioning_scale_bias"] = scale_bias


def convert_inverse_folding(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole inverse-folding checkpoint state dict onto a BoltzGenIFoldModel state dict.

    Routes the three top-level prefixes (``input_embedder`` / ``inverse_folding_encoder`` / ``structure_module``)
    to their sub-converters and packs the adaptive-layer-norm projections into the fused OOps parameters.

    Args:
        state: Upstream inverse-folding checkpoint state dict (e.g. ``boltzgen1_ifold.ckpt``).

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> state = torch.load("boltzgen1_ifold.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_inverse_folding(state)  # doctest: +SKIP
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            new_key = None if inner is None else "input_embedder." + inner
        elif key.startswith("inverse_folding_encoder."):
            new_key = "encoder." + convert_encoder(key[len("inverse_folding_encoder.") :])
        elif key.startswith("structure_module."):
            new_key = "decoder." + convert_decoder(key[len("structure_module.") :])
        else:
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        if new_key is None:
            if _boltz_conversion.is_tied_output_projection(key):
                continue
            raise KeyError(f"unmapped inverse-folding checkpoint key: {key}")
        out[new_key] = value
    _pack_conditioned_adaptive_norms(out)
    return out


def convert_structure_design(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    r"""
    Convert a whole BoltzGen structure-design checkpoint (e.g. ``boltzgen1_diverse``) onto OFoldX clean names.

    The shared backbone prefixes route through this file's ``convert_trunk_key``. The input embedder carries
    design conditioning, and ``token_distance_module`` plus the v1 template module use local sub-converters.
    Tied ``output_projection.0`` atom-layer duplicates are dropped. The Miniformer variants are auto-detected from
    the presence of ``triangular.*`` keys.

    Args:
        state: Upstream structure-design checkpoint state dict.

    Returns:
        OFoldX state dict keyed by clean names, ready for strict loading.

    Examples:
        >>> import torch  # doctest: +SKIP
        >>> state = torch.load("boltzgen1_diverse.ckpt")["state_dict"]  # doctest: +SKIP
        >>> converted = convert_structure_design(state)  # doctest: +SKIP
    """

    def _sub(prefix: str) -> dict[str, torch.Tensor]:
        return {key[len(prefix) + 1 :]: value for key, value in state.items() if key.startswith(prefix + ".")}

    miniformer_pairformer = any(key.startswith("pairformer_module.layers.") and ".triangular." in key for key in state)
    miniformer_msa = any(
        key.startswith("msa_module.layers.") and ".pairformer_layer.triangular." in key for key in state
    )
    miniformer_template = any(
        key.startswith("template_module.pairformer.layers.") and ".triangular." in key for key in state
    )

    # Modules the boltz2 trunk converter doesn't cover (or covers as v2): route to the BoltzGen converters.
    out = {f"token_distance.{k}": v for k, v in convert_token_distance(_sub("token_distance_module")).items()}
    out.update(
        {
            f"template_module.{k}": v
            for k, v in convert_template(_sub("template_module"), miniformer_blocks=miniformer_template).items()
        }
    )
    for key, value in state.items():
        if key.startswith(("token_distance_module.", "template_module.")):
            continue
        if key.startswith("input_embedder."):
            inner = convert_input_embedder(key[len("input_embedder.") :])
            if inner is None:
                if _boltz_conversion.is_tied_output_projection(key):
                    continue
                raise KeyError(f"unmapped structure-design checkpoint key: {key}")
            out["input_embedder." + inner] = value
        else:
            # BoltzGen wraps the diffusion token transformer in an outer ModuleList (token_layers=1):
            # token_transformer_layers.0.layers.<i> == boltz2's flat token_transformer.layers.<i>.
            source_key = key
            key = key.replace(
                "structure_module.score_model.token_transformer_layers.0.layers.",
                "structure_module.score_model.token_transformer.layers.",
                1,
            )
            converted = convert_trunk_key(
                key,
                value,
                miniformer_pairformer=miniformer_pairformer,
                miniformer_msa=miniformer_msa,
            )
            if not converted:
                if _boltz_conversion.is_tied_output_projection(source_key):
                    continue
                raise KeyError(f"unmapped structure-design checkpoint key: {source_key}")
            for new_key, new_value in converted:
                out[new_key] = new_value
    _pack_conditioned_adaptive_norms(out)
    return out


def convert_boltzgen_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: BoltzGenConfig | None = None,
    dtype: torch.dtype | None = None,
) -> BoltzGenModel:
    r"""Convert a released BoltzGen structure-design checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_structure_design`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = _cast_floating_tensors(convert_structure_design(state), dtype)
    with torch.device("meta"):
        model = BoltzGenModel(config)
    model.load_state_dict(converted, strict=True, assign=True)
    return model


def convert_boltzgen_ifold_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    config: BoltzGenIFoldConfig | None = None,
    dtype: torch.dtype | None = None,
) -> BoltzGenIFoldModel:
    r"""Convert a released BoltzGen inverse-folding checkpoint.

    The released Lightning artifact pickles upstream BoltzGen objects, so this path loader requires the upstream
    ``boltzgen`` and ``pydssp`` deserialization environment. For a dependency-free conversion, export a plain
    state dict or safetensors upstream and call [`convert_inverse_folding`][] directly.
    """

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    converted = _cast_floating_tensors(convert_inverse_folding(state), dtype)
    with torch.device("meta"):
        model = BoltzGenIFoldModel(config)
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


def save_boltzgen_pipeline(
    design_checkpoint_path: str | PathLike[str],
    ifold_checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    design_config: BoltzGenConfig | None = None,
    ifold_config: BoltzGenIFoldConfig | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""Convert the BoltzGen design + inverse-fold checkpoints and write the composite repository.

    The released BoltzGen design deliverable is a multi-pair :class:`BoltzGenPipeline`:
    the structure-design pair (``ofoldx.boltzgen``, ``standard_tasks=()``) owns the runnable root and
    the inverse-folding pair lives at the fixed ``ifold/`` sub-directory. Neither registered pair
    routes as a lone standard task, so this writes the aggregate repository — both required pairs and
    the one-field ``pipeline.json`` marker — via ``BoltzGenPipeline.save`` so ``pipeline(..., model=dir)``
    routes it back through the multi-pair loader. The optional read-only boltz2 ``fold/`` / ``affinity/``
    stages are injected at load time and are not vendored here.

    Both designer Processors are config-less, so only ``model.json`` + safetensors are written per pair.
    Saving is local-only by default; passing ``repo_id`` additionally uploads the composite repository.
    """

    from .pipeline import BoltzGenPipeline

    design_model = convert_boltzgen_checkpoint(design_checkpoint_path, config=design_config, dtype=dtype)
    ifold_model = convert_boltzgen_ifold_checkpoint(ifold_checkpoint_path, config=ifold_config, dtype=dtype)
    BoltzGenPipeline(
        design_model=design_model,
        design_processor=BoltzGenProcessor(),
        ifold_model=ifold_model,
        ifold_processor=BoltzGenIFoldProcessor(),
    ).save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design_checkpoint", type=Path, help="released BoltzGen structure-design .ckpt")
    parser.add_argument("ifold_checkpoint", type=Path, help="released BoltzGen inverse-folding .ckpt")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    save_boltzgen_pipeline(
        args.design_checkpoint,
        args.ifold_checkpoint,
        args.output_dir,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
