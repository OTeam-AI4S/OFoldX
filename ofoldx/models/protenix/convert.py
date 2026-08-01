r"""
Protenix checkpoint converter.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, TypeAlias, cast, get_args

import torch
from torch import Tensor, nn

from ...data import MSA_ALPHABET, TOKEN_ALPHABET
from .config import ProtenixVariant

if TYPE_CHECKING:
    from .model import ProtenixModel, ProtenixStructureConditionModel


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_ReferenceKey: TypeAlias = str | tuple[str, ...]
_ReferenceValue: TypeAlias = Tensor | tuple[Tensor, ...]
_PROTENIX_VARIANTS = get_args(ProtenixVariant)
_ESM_VARIANTS = frozenset(
    {
        "protenix_mini_esm_v0.5.0",
        "protenix_mini_ism_v0.5.0",
    }
)
_TEMPLATE_VARIANTS = frozenset(
    {
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        "protenix-v2",
    }
)
_STRUCTURE_CONDITION_VARIANT = "protenix_mini_tmpl_v0.5.0"
_INERT_V05_TEMPLATE_KEYS = frozenset(
    {
        "template_embedder.linear_no_bias_z.weight",
        "template_embedder.layernorm_z.weight",
        "template_embedder.layernorm_z.bias",
        "template_embedder.linear_no_bias_a.weight",
        "template_embedder.layernorm_v.weight",
        "template_embedder.layernorm_v.bias",
        "template_embedder.linear_no_bias_u.weight",
    }
)


def load_reference_state(
    checkpoint_path: str | PathLike[str] | BinaryIO,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    r"""
    Load a reference checkpoint's tensor state, unwrapping the ``model`` container and ``module.`` prefixes.

    Reads the checkpoint, unwraps a ``{"model": ...}`` wrapper when present, drops the distributed-training
    ``module.`` key prefix, and keeps only tensor-valued entries.

    Args:
        checkpoint_path: Path (or file-like object) accepted by `torch.load`.
        map_location: Device mapping forwarded to `torch.load`; defaults to CPU.

    Returns:
        Mapping from clean checkpoint key to tensor.
    """

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = checkpoint["model"] if isinstance(checkpoint, Mapping) and "model" in checkpoint else checkpoint
    reference_state: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = source_key.removeprefix("module.")
        if key in reference_state:
            raise ValueError(f"multiple Protenix source tensors normalize to {key!r}")
        reference_state[key] = value
    return reference_state


def load_esm_reference_state(
    checkpoint_path: str | PathLike[str] | BinaryIO,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    r"""
    Load the tensor state from an official fair-esm companion checkpoint.

    The released companion stores an ``argparse.Namespace`` configuration
    beside the tensor state. That exact metadata type is allow-listed while the
    tensor-only unpickler remains enabled.
    """

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise TypeError("an ESM companion checkpoint must contain a model state mapping")
    reference_state: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = source_key.removeprefix("module.")
        if key in reference_state:
            raise ValueError(f"multiple ESM source tensors normalize to {key!r}")
        reference_state[key] = value
    return reference_state


def translate_reference_state(
    model: nn.Module,
    reference_state: Mapping[str, Tensor],
    key_fn: Callable[[str], _ReferenceKey],
    *,
    model_keys: Iterable[str] | None = None,
) -> dict[str, Tensor]:
    r"""
    Translate every target and require every source tensor to participate.
    """

    translated: dict[str, Tensor] = {}
    consumed: set[str] = set()
    target_keys = model.state_dict() if model_keys is None else model_keys
    for model_key in target_keys:
        reference_key = key_fn(model_key)
        reference_value = _lookup_reference_value(reference_state, reference_key)
        translated[model_key] = _transform_reference_value(model_key, reference_value)
        if isinstance(reference_key, tuple):
            consumed.update(reference_key)
        else:
            consumed.add(reference_key)

    unconsumed = sorted(set(reference_state).difference(consumed))
    if unconsumed:
        preview = ", ".join(repr(key) for key in unconsumed[:20])
        suffix = "" if len(unconsumed) <= 20 else f", ... ({len(unconsumed)} total)"
        raise ValueError(f"unconsumed source tensors: {preview}{suffix}")
    return translated


def translate_protenix_state(
    model: ProtenixModel | ProtenixStructureConditionModel,
    reference_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    r"""
    Translate every live tensor in one released Protenix graph.

    The pre-template source constructors serialize seven template
    projection/norm tensors even though their configured zero-block path is
    unreachable. Those exact tensors are omitted only when the target graph has
    no template path. ESM child tensors are translated separately from their
    companion checkpoint.
    """

    model_keys = tuple(key for key in model.state_dict() if not key.startswith("esm_encoder."))
    has_template = any(key.startswith("template_embedder.") for key in model_keys)
    live_reference_state = (
        dict(reference_state)
        if has_template
        else {key: value for key, value in reference_state.items() if key not in _INERT_V05_TEMPLATE_KEYS}
    )
    msa_final_block = len(model.msa_trunk.blocks)

    def reference_key(model_key: str) -> _ReferenceKey:
        return _protenix_reference_key(model_key, msa_final_block=msa_final_block)

    return translate_reference_state(
        model,
        live_reference_state,
        reference_key,
        model_keys=model_keys,
    )


def translate_esm_state(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Translate the exact fair-esm ESM2 t36 encoder into the packaged child.

    The masked-language-model head is not executed by Protenix, and the
    separately released contact-regression head is not part of the companion
    file or the target graph.
    """

    normalized: dict[str, Tensor] = {}
    for source_key, value in reference_state.items():
        key = source_key
        for prefix in ("encoder.sentence_encoder.", "encoder."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        if key.startswith(("lm_head.", "contact_head.")):
            continue
        if key in normalized:
            raise ValueError(f"multiple ESM source tensors normalize to {key!r}")
        normalized[key] = value

    model_keys = tuple(key for key in model.state_dict() if key.startswith("esm_encoder."))
    return translate_reference_state(
        model,
        normalized,
        _esm_reference_key,
        model_keys=model_keys,
    )


def convert_protenix_model(
    checkpoint_path: str | Path,
    *,
    variant: ProtenixVariant,
    esm_checkpoint_path: str | Path | None = None,
    dtype: torch.dtype | None = None,
) -> ProtenixModel | ProtenixStructureConditionModel:
    r"""
    Convert one explicitly selected released Protenix checkpoint.

    Args:
        checkpoint_path: Path to the released Protenix checkpoint.
        variant: Exact released variant name. It is never inferred from the
            filename or tensor graph.
        esm_checkpoint_path: Official ESM2 companion path, required only for
            mini ESM and mini ISM.
        dtype: Optional floating-point dtype for converted weights.

    Returns:
        The converted, strict-loaded model for the selected Protenix
        architecture. Use `convert_protenix_checkpoint` to persist the paired
        model/processor repository locally or on the Hub.
    """

    from .config import (
        ProtenixConfig,
        ProtenixConstraintConfig,
        ProtenixStructureConditionConfig,
        ProtenixTemplateModelConfig,
    )
    from .model import (
        ProtenixConstraintModel,
        ProtenixModel,
        ProtenixStructureConditionModel,
        ProtenixTemplateModel,
    )

    needs_esm = variant in _ESM_VARIANTS
    if needs_esm and esm_checkpoint_path is None:
        raise ValueError(f"{variant!r} requires its released ESM2 companion checkpoint")
    if not needs_esm and esm_checkpoint_path is not None:
        raise ValueError(f"{variant!r} has no ESM2 companion checkpoint")

    config = ProtenixConfig.from_variant(variant)
    model: ProtenixModel | ProtenixStructureConditionModel
    with torch.device("meta"):
        if variant == "protenix_base_constraint_v0.5.0":
            model = ProtenixConstraintModel(cast(ProtenixConstraintConfig, config))
        elif variant == _STRUCTURE_CONDITION_VARIANT:
            model = ProtenixStructureConditionModel(cast(ProtenixStructureConditionConfig, config))
        elif variant in _TEMPLATE_VARIANTS:
            model = ProtenixTemplateModel(cast(ProtenixTemplateModelConfig, config))
        else:
            model = ProtenixModel(config)
    reference_state = load_reference_state(Path(checkpoint_path))
    translated = translate_protenix_state(model, reference_state)
    if esm_checkpoint_path is not None:
        translated.update(
            translate_esm_state(
                model,
                load_esm_reference_state(Path(esm_checkpoint_path)),
            )
        )
    if dtype is not None:
        translated = {
            key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in translated.items()
        }
    model.load_state_dict(translated, strict=True, assign=True)
    model.eval()
    return model


def convert_protenix_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    variant: ProtenixVariant,
    esm_checkpoint_path: str | PathLike[str] | None = None,
    dtype: torch.dtype | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Convert an explicitly selected Protenix checkpoint and write a routable
    OFoldX repository. ESM and ISM companion weights are stored in the ordinary
    Model artifact.

    Saving is local-only by default; passing ``repo_id`` additionally uploads the repository — marker
    included. No organization or token is assumed and nothing is published without an explicit
    ``repo_id``.
    """

    from .model import (
        ProtenixConstraintModel,
        ProtenixModel,
        ProtenixStructureConditionModel,
        ProtenixTemplateModel,
    )
    from .pipeline import (
        ProtenixConstraintPipeline,
        ProtenixPipeline,
        ProtenixStructureConditionPipeline,
        ProtenixTemplatePipeline,
    )
    from .processor import (
        ProtenixConstraintProcessor,
        ProtenixProcessor,
        ProtenixStructureConditionProcessor,
        ProtenixTemplateProcessor,
    )

    model = convert_protenix_model(
        Path(checkpoint_path).expanduser(),
        variant=variant,
        esm_checkpoint_path=None if esm_checkpoint_path is None else Path(esm_checkpoint_path).expanduser(),
        dtype=dtype,
    )
    pipeline: (
        ProtenixPipeline | ProtenixTemplatePipeline | ProtenixStructureConditionPipeline | ProtenixConstraintPipeline
    )
    if variant == "protenix_base_constraint_v0.5.0":
        pipeline = ProtenixConstraintPipeline(
            model=cast(ProtenixConstraintModel, model),
            processor=ProtenixConstraintProcessor(dtype=dtype),
        )
    elif variant == _STRUCTURE_CONDITION_VARIANT:
        pipeline = ProtenixStructureConditionPipeline(
            model=cast(ProtenixStructureConditionModel, model),
            processor=ProtenixStructureConditionProcessor(dtype=dtype),
        )
    elif variant in _TEMPLATE_VARIANTS:
        pipeline = ProtenixTemplatePipeline(
            model=cast(ProtenixTemplateModel, model),
            processor=ProtenixTemplateProcessor(dtype=dtype),
        )
    else:
        pipeline = ProtenixPipeline(
            model=cast(ProtenixModel, model),
            processor=ProtenixProcessor(dtype=dtype),
        )
    pipeline.save(
        output_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )


def main(argv: list[str] | None = None) -> None:
    r"""
    Run protenix checkpoint conversion from the command line.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="released Protenix checkpoint",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--variant", choices=_PROTENIX_VARIANTS, required=True)
    parser.add_argument(
        "--esm-checkpoint",
        type=Path,
        default=None,
        help="official ESM2 companion; required only for mini ESM/ISM",
    )
    parser.add_argument("--dtype", choices=("keep", *_DTYPES), default="keep")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="optional Hub repo id (e.g. owner/name) to publish to; local-only when omitted",
    )
    parser.add_argument("--token", default=None, help="optional Hugging Face token for --repo-id")
    parser.add_argument("--private", action="store_true", help="create the --repo-id repo as private")
    args = parser.parse_args(argv)
    convert_protenix_checkpoint(
        args.checkpoint,
        args.output_dir,
        variant=args.variant,
        esm_checkpoint_path=args.esm_checkpoint,
        dtype=None if args.dtype == "keep" else _DTYPES[args.dtype],
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


def _esm_reference_key(model_key: str) -> _ReferenceKey:
    r"""
    Return the fair-esm ESM2 t36 key for one packaged encoder key.
    """

    key = model_key.removeprefix("esm_encoder.")
    if key == "embeddings.word_embeddings.weight":
        return "embed_tokens.weight"
    if key.startswith("encoder.emb_layer_norm_after."):
        return "emb_layer_norm_after." + key.removeprefix("encoder.emb_layer_norm_after.")
    if key == "rotary_embeddings.inv_freq":
        return tuple(f"layers.{index}.self_attn.rot_emb.inv_freq" for index in range(36))

    rotary_match = re.fullmatch(
        r"encoder\.layer\.(\d+)\.attention\.self\.rotary_embeddings\.inv_freq",
        key,
    )
    if rotary_match is not None:
        return f"layers.{rotary_match.group(1)}.self_attn.rot_emb.inv_freq"

    layer_match = re.fullmatch(r"encoder\.layer\.(\d+)\.(.+)", key)
    if layer_match is None:
        raise KeyError(f"unsupported packaged ESM2 key: {model_key!r}")
    layer_index, suffix = layer_match.groups()
    source_suffix = _replace_many(
        suffix,
        {
            "attention.self.query.": "self_attn.q_proj.",
            "attention.self.key.": "self_attn.k_proj.",
            "attention.self.value.": "self_attn.v_proj.",
            "attention.output.dense.": "self_attn.out_proj.",
            "attention.LayerNorm.": "self_attn_layer_norm.",
            "intermediate.dense.": "fc1.",
            "output.dense.": "fc2.",
            "LayerNorm.": "final_layer_norm.",
        },
    )
    return f"layers.{layer_index}.{source_suffix}"


def _protenix_reference_key(model_key: str, *, msa_final_block: int) -> _ReferenceKey:
    if model_key.startswith("noisy_structure_embedder."):
        return _replace_many(
            model_key,
            {
                ".lower_bins": ".bins",
                ".structure_projection.": ".linear_struct.",
                ".pair_norm.": ".layernorm_z.",
                ".pair_projection.": ".linear_z.",
                ".transition.input_norm.": ".transition_out.layernorm1.",
                ".transition.feed_forward.gate_projection.": ".transition_out.linear_no_bias_a.",
                ".transition.feed_forward.value_projection.": ".transition_out.linear_no_bias_b.",
                ".transition.feed_forward.output_projection.": ".transition_out.linear_no_bias.",
            },
        )
    if model_key.startswith("msa_trunk.final_pair_update."):
        model_key = f"msa_trunk.blocks.{msa_final_block}." + model_key.removeprefix("msa_trunk.final_pair_update.")
    if model_key.startswith("constraint_embeddings."):
        relative_key = model_key.removeprefix("constraint_embeddings.")
        encoder_match = re.fullmatch(
            r"substructure_embeddings\.block\.(.+)",
            relative_key,
        )
        if encoder_match is not None:
            (suffix,) = encoder_match.groups()
            source_suffix = {
                "attention.query_projection.weight": "self_attn.in_proj_weight",
                "attention.query_projection.bias": "self_attn.in_proj_bias",
                "attention.key_projection.weight": "self_attn.in_proj_weight",
                "attention.key_projection.bias": "self_attn.in_proj_bias",
                "attention.value_projection.weight": "self_attn.in_proj_weight",
                "attention.value_projection.bias": "self_attn.in_proj_bias",
                "attention.output_projection.weight": "self_attn.out_proj.weight",
                "attention.output_projection.bias": "self_attn.out_proj.bias",
                "attention_norm.weight": "norm1.weight",
                "attention_norm.bias": "norm1.bias",
                "transition.input_projection.weight": "linear1.weight",
                "transition.input_projection.bias": "linear1.bias",
                "transition.output_projection.weight": "linear2.weight",
                "transition.output_projection.bias": "linear2.bias",
                "transition_norm.weight": "norm2.weight",
                "transition_norm.bias": "norm2.bias",
            }[suffix]
            return f"constraint_embedder.substructure_z_embedder.transformer.layers.0.{source_suffix}"

        key = "constraint_embedder." + relative_key
        key = _replace_many(
            key,
            {
                ".pocket_projection.": ".pocket_z_embedder.",
                ".token_contact_projection.": ".contact_z_embedder.",
                ".atom_contact_projection.": ".contact_atom_z_embedder.",
                ".substructure_embeddings.": ".substructure_z_embedder.",
            },
        )
        return key.replace(".input_projection.", ".input_proj.").replace(".output_projection.", ".output_proj.")
    return _reference_key(model_key)


def _reference_key(model_key: str) -> _ReferenceKey:
    r"""
    Return the released Protenix checkpoint key for an OFoldX model key.
    """

    key = _replace_top_level(model_key)
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


def _lookup_reference_value(reference_state: Mapping[str, Tensor], reference_key: _ReferenceKey) -> _ReferenceValue:
    r"""
    Read the source tensor or tensors named by one target key.
    """

    if isinstance(reference_key, tuple):
        return tuple(reference_state[key] for key in reference_key)
    return reference_state[reference_key]


def _transform_reference_value(model_key: str, reference_value: _ReferenceValue) -> Tensor:
    r"""
    Apply tensor transforms required by the released Protenix format.
    """

    if isinstance(reference_value, tuple):
        if model_key == "esm_encoder.rotary_embeddings.inv_freq":
            first = reference_value[0]
            if any(not torch.equal(first, value) for value in reference_value[1:]):
                raise ValueError("fair-esm ESM2 layers do not share one rotary inverse-frequency buffer")
            return first
        if model_key.endswith(".conditioning_projection.weight"):
            return torch.cat(reference_value, dim=0)
        raise RuntimeError(f"unexpected multi-tensor reference mapping for {model_key}")

    if model_key.startswith("constraint_embeddings.substructure_embeddings.block."):
        projection_index = {
            ".attention.query_projection.": 0,
            ".attention.key_projection.": 1,
            ".attention.value_projection.": 2,
        }
        for projection, index in projection_index.items():
            if projection in model_key:
                return reference_value.chunk(3, dim=0)[index]

    if model_key == "msa_trunk.msa_projection.weight":
        residue, deletion = reference_value[:, :-2], reference_value[:, -2:]
        return torch.cat([_convert_msa_axis(residue), deletion], dim=1)
    if model_key == "template_embedder.template_projection.weight":
        return _convert_template_pair_axis(reference_value)
    if model_key == "esm_projection.weight":
        return _convert_token_feature_axis(reference_value, dim=0)
    if model_key in _TOKEN_FEATURE_INPUT_KEYS:
        return _convert_trailing_token_feature_axis(reference_value, dim=-1)
    if model_key == "coordinate_denoiser.conditioner.single_conditioning_norm.weight":
        return _convert_trailing_token_feature_axis(reference_value, dim=0)
    if model_key.endswith(".noise_embedding.projection.weight") and reference_value.dim() == 1:
        return reference_value.unsqueeze(-1)
    return reference_value


_TOKEN_FEATURE_INPUT_KEYS = {
    "single_init_projection.weight",
    "msa_trunk.token_projection.weight",
    "coordinate_denoiser.conditioner.single_conditioning_projection.weight",
    "confidence_head.pair_embedder.left_token_projection.weight",
    "confidence_head.pair_embedder.right_token_projection.weight",
}
_SOURCE_TOKEN_INDEX = {
    **{
        symbol: index
        for index, symbol in enumerate(
            ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
        )
    },
    "X": 20,
    "RNA_A": 21,
    "RNA_G": 22,
    "RNA_C": 23,
    "RNA_U": 24,
    "RNA_N": 25,
    "DNA_A": 26,
    "DNA_G": 27,
    "DNA_C": 28,
    "DNA_T": 29,
    "DNA_N": 30,
    "-": 31,
    "OTHER": 20,
}
_SOURCE_SINGLE_FEATURE_SIZE = 384
_SOURCE_STRUCTURAL_VOCAB_SIZE = 32
_SOURCE_MSA_VOCAB_SIZE = 32
_SOURCE_TOKEN_FEATURE_SIZE = _SOURCE_SINGLE_FEATURE_SIZE + _SOURCE_STRUCTURAL_VOCAB_SIZE + _SOURCE_MSA_VOCAB_SIZE + 1
_SOURCE_TEMPLATE_PAIR_PREFIX_SIZE = 40
_SOURCE_TEMPLATE_PAIR_SUFFIX_SIZE = 4


def _convert_trailing_token_feature_axis(source: Tensor, *, dim: int) -> Tensor:
    source = source.movedim(dim, -1)
    prefix, token_feature = (
        source[..., :-_SOURCE_TOKEN_FEATURE_SIZE],
        source[..., -_SOURCE_TOKEN_FEATURE_SIZE:],
    )
    converted = torch.cat([prefix, _convert_token_feature_axis(token_feature, dim=-1)], dim=-1)
    return converted.movedim(-1, dim).contiguous()


def _convert_token_feature_axis(source: Tensor, *, dim: int) -> Tensor:
    source = source.movedim(dim, -1)
    structural_start = _SOURCE_SINGLE_FEATURE_SIZE
    profile_start = structural_start + _SOURCE_STRUCTURAL_VOCAB_SIZE
    deletion_start = profile_start + _SOURCE_MSA_VOCAB_SIZE
    single = source[..., :structural_start]
    structural = source[..., structural_start:profile_start]
    profile = source[..., profile_start:deletion_start]
    deletion = source[..., deletion_start:]
    converted = torch.cat(
        [
            single,
            _convert_structural_axis(structural),
            _convert_msa_axis(profile),
            deletion,
        ],
        dim=-1,
    )
    return converted.movedim(-1, dim).contiguous()


def _convert_template_pair_axis(source: Tensor) -> Tensor:
    left_start = _SOURCE_TEMPLATE_PAIR_PREFIX_SIZE
    right_start = left_start + _SOURCE_STRUCTURAL_VOCAB_SIZE
    suffix_start = right_start + _SOURCE_STRUCTURAL_VOCAB_SIZE
    prefix = source[..., :left_start]
    left = source[..., left_start:right_start]
    right = source[..., right_start:suffix_start]
    suffix = source[..., -_SOURCE_TEMPLATE_PAIR_SUFFIX_SIZE:]
    return torch.cat(
        [
            prefix,
            _convert_structural_axis(left),
            _convert_structural_axis(right),
            suffix,
        ],
        dim=-1,
    )


def _convert_structural_axis(source: Tensor) -> Tensor:
    return _convert_source_token_axis(source, TOKEN_ALPHABET)


def _convert_msa_axis(source: Tensor) -> Tensor:
    return _convert_source_token_axis(source, MSA_ALPHABET)


def _convert_source_token_axis(source: Tensor, alphabet: Sequence[str]) -> Tensor:
    converted = []
    for symbol in alphabet:
        if symbol == "B":
            value = (source[..., _SOURCE_TOKEN_INDEX["D"]] + source[..., _SOURCE_TOKEN_INDEX["N"]]) / 2
        elif symbol == "Z":
            value = (source[..., _SOURCE_TOKEN_INDEX["E"]] + source[..., _SOURCE_TOKEN_INDEX["Q"]]) / 2
        elif symbol == "J":
            value = (source[..., _SOURCE_TOKEN_INDEX["I"]] + source[..., _SOURCE_TOKEN_INDEX["L"]]) / 2
        elif symbol == "U":
            value = source[..., _SOURCE_TOKEN_INDEX["C"]]
        elif symbol == "O":
            value = source[..., _SOURCE_TOKEN_INDEX["K"]]
        else:
            value = source[..., _SOURCE_TOKEN_INDEX[symbol]]
        converted.append(value)
    return torch.stack(converted, dim=-1)


def _remap_token_transformer_blocks(key: str) -> str:
    r"""
    Map a consolidated token transformer onto Protenix block names.
    """

    rules = [
        (r"\.pair_norms\.(\d+)\.", r".blocks.\1.attention_pair_bias.layernorm_z."),
        (r"\.bias_projections\.(\d+)\.", r".blocks.\1.attention_pair_bias.linear_nobias_z."),
        (r"\.transformer\.blocks\.(\d+)\.transition\.", r".blocks.\1.conditioned_transition_block."),
        (r"\.transformer\.blocks\.(\d+)\.conditioned_norm\.", r".blocks.\1.attention_pair_bias.layernorm_a."),
        (r"\.transformer\.blocks\.(\d+)\.attention\.", r".blocks.\1.attention_pair_bias.attention."),
        (
            r"\.transformer\.blocks\.(\d+)\.output_gate_projection\.",
            r".blocks.\1.attention_pair_bias.linear_a_last.",
        ),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_top_level(key: str) -> str:
    replacements = {
        "esm_projection": "input_embedder.linear_esm",
        "template_embedder.pair_norm": "template_embedder.layernorm_z",
        "template_embedder.pair_projection": "template_embedder.linear_no_bias_z",
        "template_embedder.template_projection": "template_embedder.linear_no_bias_a",
        "template_embedder.pair_update_stack": "template_embedder.pairformer_stack",
        "template_embedder.hidden_norm": "template_embedder.layernorm_v",
        "template_embedder.output_projection": "template_embedder.linear_no_bias_u",
        "relative_position_embedder": "relative_position_encoding",
        "single_init_projection": "linear_no_bias_sinit",
        "pair_left_projection": "linear_no_bias_zinit1",
        "pair_right_projection": "linear_no_bias_zinit2",
        "token_bond_projection": "linear_no_bias_token_bond",
        "pair_recycle_projection": "linear_no_bias_z_cycle",
        "single_recycle_projection": "linear_no_bias_s",
        "pair_recycle_norm": "layernorm_z_cycle",
        "single_recycle_norm": "layernorm_s",
        "distogram_head.projection": "distogram_head.linear",
        "input_embeddings.atom_encoder": "input_embedder.atom_attention_encoder",
        "msa_trunk": "msa_module",
        "coordinate_denoiser": "diffusion_module",
    }
    for source, target in replacements.items():
        if key == source or key.startswith(source + "."):
            return target + key[len(source) :]
    return key


def _replace_confidence_key(key: str) -> str:
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
    if ".token_encoder." not in key:
        return key
    key = key.replace(".token_refiner.token_encoder.", ".diffusion_transformer.")
    return _remap_token_transformer_blocks(key)


def _replace_denoiser_key(key: str) -> str:
    if not key.startswith("diffusion_module."):
        return key
    return _replace_many(
        key,
        {
            ".conditioner.relative_position_embedder.": ".diffusion_conditioning.relpe.",
            ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
            ".conditioner.pair_input_norm.": ".diffusion_conditioning.layernorm_z.",
            ".conditioner.pair_projection.": ".diffusion_conditioning.linear_no_bias_z.",
            ".conditioner.pair_transitions.0.": ".diffusion_conditioning.transition_z1.",
            ".conditioner.pair_transitions.1.": ".diffusion_conditioning.transition_z2.",
            ".conditioner.single_conditioning_norm.": ".diffusion_conditioning.layernorm_s.",
            ".conditioner.single_conditioning_projection.": ".diffusion_conditioning.linear_no_bias_s.",
            ".conditioner.noise_embedding.projection.weight": ".diffusion_conditioning.fourier_embedding.w",
            ".conditioner.noise_embedding.projection.bias": ".diffusion_conditioning.fourier_embedding.b",
            ".conditioner.noise_norm.": ".diffusion_conditioning.layernorm_n.",
            ".conditioner.noise_projection.": ".diffusion_conditioning.linear_no_bias_n.",
            ".conditioner.single_transitions.0.": ".diffusion_conditioning.transition_s1.",
            ".conditioner.single_transitions.1.": ".diffusion_conditioning.transition_s2.",
            ".atom_encoder.": ".atom_attention_encoder.",
            ".token_refiner.single_norm.": ".layernorm_s.",
            ".token_refiner.single_to_token_projection.": ".linear_no_bias_s.",
            ".token_refiner.token_encoder.": ".diffusion_transformer.",
            ".token_refiner.token_norm.": ".layernorm_a.",
            ".atom_decoder.": ".atom_attention_decoder.",
        },
    )


def _replace_atom_stack_key(key: str) -> str:
    if ".atom_self_attention." not in key:
        return key
    block = r".atom_transformer.diffusion_transformer.blocks.\1"
    target_block = r"\.atom_self_attention\.atom_attention\.transformer\.blocks\.(\d+)\."
    rules = [
        (r"\.atom_self_attention\.pair_norms\.(\d+)\.", block + r".attention_pair_bias.layernorm_z."),
        (
            r"\.atom_self_attention\.bias_projections\.(\d+)\.",
            block + r".attention_pair_bias.linear_nobias_z.",
        ),
        (target_block + r"transition\.", block + r".conditioned_transition_block."),
        (target_block + r"conditioned_norm\.", block + r".attention_pair_bias.layernorm_a."),
        (target_block + r"key_value_norm\.", block + r".attention_pair_bias.layernorm_kv."),
        (target_block + r"attention\.", block + r".attention_pair_bias.attention."),
        (target_block + r"output_gate_projection\.", block + r".attention_pair_bias.linear_a_last."),
    ]
    for pattern, replacement in rules:
        key = re.sub(pattern, replacement, key)
    return key


def _replace_atom_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".atom_conditioner.reference_position_projection.": ".linear_no_bias_ref_pos.",
            ".atom_conditioner.reference_charge_projection.": ".linear_no_bias_ref_charge.",
            ".atom_conditioner.atom_feature_projection.": ".linear_no_bias_f.",
            ".atom_pair_encoder.pair_encoding.distance_vector_projection.": ".linear_no_bias_d.",
            ".atom_pair_encoder.pair_encoding.inverse_distance_projection.": ".linear_no_bias_invd.",
            ".atom_pair_encoder.pair_encoding.distance_mask_projection.": ".linear_no_bias_v.",
            ".atom_pair_encoder.pair_encoding.left_atom_projection.": ".linear_no_bias_cl.",
            ".atom_pair_encoder.pair_encoding.right_atom_projection.": ".linear_no_bias_cm.",
            ".atom_pair_encoder.pair_encoding.feed_forward_input_projection.": ".small_mlp.1.",
            ".atom_pair_encoder.pair_encoding.feed_forward_hidden_projection.": ".small_mlp.3.",
            ".atom_pair_encoder.pair_encoding.feed_forward_output_projection.": ".small_mlp.5.",
            ".atom_self_attention.": ".atom_transformer.diffusion_transformer.",
            ".atom_attention_encoder.token_projection.": ".atom_attention_encoder.linear_no_bias_q.",
            ".atom_attention_decoder.token_projection.": ".atom_attention_decoder.linear_no_bias_a.",
            ".single_norm.": ".layernorm_s.",
            ".single_to_atom_projection.": ".linear_no_bias_s.",
            ".atom_attention_encoder.pair_norm.": ".atom_attention_encoder.layernorm_z.",
            ".pair_projection.": ".linear_no_bias_z.",
            ".noisy_position_projection.": ".linear_no_bias_r.",
            ".atom_attention_decoder.output_projection.": ".atom_attention_decoder.linear_no_bias_out.",
            ".atom_attention_decoder.query_norm.": ".atom_attention_decoder.layernorm_q.",
        },
    )


def _replace_pairformer_key(key: str) -> str:
    if key.startswith("template_embedder.pairformer_stack.blocks."):
        key = re.sub(
            r"(template_embedder\.pairformer_stack\.blocks\.\d+)\.transition\.",
            r"\1.pair_transition.",
            key,
        )
    return _replace_many(
        key,
        {
            ".pair_update.transition.": ".pair_transition.",
            ".pair_update.": ".",
            ".msa_update.": ".msa_stack.",
            ".triangle_pair_update.": ".pair_stack.",
            ".triangle_multiplication_outgoing.": ".tri_mul_out.",
            ".triangle_multiplication_incoming.": ".tri_mul_in.",
            ".triangle_attention_starting_node.": ".tri_att_start.",
            ".triangle_attention_ending_node.": ".tri_att_end.",
            ".pair_attention.": ".attention_pair_bias.",
            ".tri_mul_out.input_norm.": ".tri_mul_out.layer_norm_in.",
            ".tri_mul_in.input_norm.": ".tri_mul_in.layer_norm_in.",
            ".tri_mul_out.output_norm.": ".tri_mul_out.layer_norm_out.",
            ".tri_mul_in.output_norm.": ".tri_mul_in.layer_norm_out.",
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
            ".tri_att_start.pair_norm.": ".tri_att_start.layer_norm.",
            ".tri_att_end.pair_norm.": ".tri_att_end.layer_norm.",
            ".tri_att_start.bias_projection.": ".tri_att_start.linear.",
            ".tri_att_end.bias_projection.": ".tri_att_end.linear.",
            ".tri_att_start.attention.projection.": ".tri_att_start.mha.",
            ".tri_att_end.attention.projection.": ".tri_att_end.mha.",
        },
    )


def _replace_attention_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".pair_attention.": ".attention_pair_bias.",
            ".hidden_norm.": ".layernorm_a.",
            ".atom_norm.": ".layernorm_a.",
            ".key_value_norm.": ".layernorm_kv.",
            ".attention_pair_bias.pair_norm.": ".attention_pair_bias.layernorm_z.",
            ".attention_pair_bias.bias_projection.": ".attention_pair_bias.linear_nobias_z.",
            ".attention_pair_bias.conditioning_gate_projection.": ".attention_pair_bias.linear_a_last.",
        },
    )


def _replace_attention_projection_key(key: str) -> str:
    return _replace_many(
        key,
        {
            "relative_position_encoding.output_projection.": "relative_position_encoding.linear_no_bias.",
            ".diffusion_conditioning.relpe.output_projection.": ".diffusion_conditioning.relpe.linear_no_bias.",
            ".fourier_embedding.frequency": ".fourier_embedding.w",
            ".fourier_embedding.phase": ".fourier_embedding.b",
            "msa_module.msa_projection.": "msa_module.linear_no_bias_m.",
            "msa_module.token_projection.": "msa_module.linear_no_bias_s.",
            ".outer_product_mean.": ".outer_product_mean_msa.",
            ".outer_product_mean_msa.msa_norm.": ".outer_product_mean_msa.layer_norm.",
            ".outer_product_mean_msa.left_projection.": ".outer_product_mean_msa.linear_1.",
            ".outer_product_mean_msa.right_projection.": ".outer_product_mean_msa.linear_2.",
            ".outer_product_mean_msa.output_projection.": ".outer_product_mean_msa.linear_out.",
            ".msa_pair_weighted_averaging.msa_norm.": ".msa_pair_weighted_averaging.layernorm_m.",
            ".msa_pair_weighted_averaging.value_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mv.",
            ".msa_pair_weighted_averaging.pair_norm.": ".msa_pair_weighted_averaging.layernorm_z.",
            ".msa_pair_weighted_averaging.pair_projection.": ".msa_pair_weighted_averaging.linear_no_bias_z.",
            ".msa_pair_weighted_averaging.gate_projection.": ".msa_pair_weighted_averaging.linear_no_bias_mg.",
            ".msa_pair_weighted_averaging.output_projection.": ".msa_pair_weighted_averaging.linear_no_bias_out.",
            ".msa_stack.transition.": ".msa_stack.transition_m.",
            ".pair_stack.transition.": ".pair_stack.pair_transition.",
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
        },
    )


def _replace_transition_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".input_norm.": ".layernorm1.",
            ".conditioned_norm.": ".adaln.",
            (
                ".conditioned_transition_block.feed_forward.gate_projection."
            ): ".conditioned_transition_block.linear_nobias_a1.",
            (
                ".conditioned_transition_block.feed_forward.value_projection."
            ): ".conditioned_transition_block.linear_nobias_a2.",
            (
                ".conditioned_transition_block.feed_forward.output_projection."
            ): ".conditioned_transition_block.linear_nobias_b.",
            ".conditioned_transition_block.conditioning_gate_projection.": ".conditioned_transition_block.linear_s.",
            ".pair_transition.feed_forward.gate_projection.": ".pair_transition.linear_no_bias_a.",
            ".pair_transition.feed_forward.value_projection.": ".pair_transition.linear_no_bias_b.",
            ".pair_transition.feed_forward.output_projection.": ".pair_transition.linear_no_bias.",
            ".single_transition.feed_forward.gate_projection.": ".single_transition.linear_no_bias_a.",
            ".single_transition.feed_forward.value_projection.": ".single_transition.linear_no_bias_b.",
            ".single_transition.feed_forward.output_projection.": ".single_transition.linear_no_bias.",
            ".transition_m.feed_forward.gate_projection.": ".transition_m.linear_no_bias_a.",
            ".transition_m.feed_forward.value_projection.": ".transition_m.linear_no_bias_b.",
            ".transition_m.feed_forward.output_projection.": ".transition_m.linear_no_bias.",
            ".transition_z1.feed_forward.gate_projection.": ".transition_z1.linear_no_bias_a.",
            ".transition_z1.feed_forward.value_projection.": ".transition_z1.linear_no_bias_b.",
            ".transition_z1.feed_forward.output_projection.": ".transition_z1.linear_no_bias.",
            ".transition_z2.feed_forward.gate_projection.": ".transition_z2.linear_no_bias_a.",
            ".transition_z2.feed_forward.value_projection.": ".transition_z2.linear_no_bias_b.",
            ".transition_z2.feed_forward.output_projection.": ".transition_z2.linear_no_bias.",
            ".transition_s1.feed_forward.gate_projection.": ".transition_s1.linear_no_bias_a.",
            ".transition_s1.feed_forward.value_projection.": ".transition_s1.linear_no_bias_b.",
            ".transition_s1.feed_forward.output_projection.": ".transition_s1.linear_no_bias.",
            ".transition_s2.feed_forward.gate_projection.": ".transition_s2.linear_no_bias_a.",
            ".transition_s2.feed_forward.value_projection.": ".transition_s2.linear_no_bias_b.",
            ".transition_s2.feed_forward.output_projection.": ".transition_s2.linear_no_bias.",
        },
    )


def _replace_conditioned_norm_key(key: str) -> _ReferenceKey:
    if key.endswith(".conditioning_projection.weight"):
        base = key[: -len("conditioning_projection.weight")]
        return (base + "linear_s.weight", base + "linear_nobias_s.weight")
    if key.endswith(".conditioning_scale_bias"):
        base = key[: -len("conditioning_scale_bias")]
        return base + "linear_s.bias"
    return _replace_many(
        key,
        {
            ".hidden_norm.": ".layernorm_a.",
            ".conditioning_norm.": ".layernorm_s.",
            ".gate_projection.": ".linear_s.",
            ".shift_projection.": ".linear_nobias_s.",
        },
    )


def _replace_many(key: str, replacements: Mapping[str, str]) -> str:
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


if __name__ == "__main__":
    main()
