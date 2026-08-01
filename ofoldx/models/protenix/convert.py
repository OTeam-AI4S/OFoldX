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
Protenix checkpoint conversion.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import BinaryIO, TypeAlias, cast, get_args

import torch
from accelerate import init_empty_weights
from torch import Tensor, nn

from ...conversion import save_repository
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, SourceRef, fail
from ..convert import (
    atom_stack_reference_key,
    attention_projection_reference_key,
    attention_reference_key,
    conditioned_norm_reference_key,
    token_transformer_reference_key,
)
from ..vocabulary import canonical_word_embeddings
from .config import (
    ProtenixConfig,
    ProtenixConstraintConfig,
    ProtenixStructureConditionConfig,
    ProtenixTemplateModelConfig,
    ProtenixVariant,
)
from .model import (
    PROTENIX_SOURCE_TOKENS,
    ProtenixConstraintModel,
    ProtenixModel,
    ProtenixNoisyStructureEmbedder,
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
# Target module -> the released (lower, upper) distance-bin edge tensors it recomputes.
_RELEASED_BIN_EDGES = {
    "confidence_head.pair_embedder": ("confidence_head.lower_bins", "confidence_head.upper_bins"),
    "noisy_structure_embedder": ("noisy_structure_embedder.bins", "noisy_structure_embedder.upper_bins"),
}
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
            fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))  # several source tensors normalize to it
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
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item="model"))
    reference_state: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = source_key.removeprefix("module.")
        if key in reference_state:
            fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))  # several source tensors normalize to it
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
        fail(
            "checkpoint_invalid",
            *(SourceRef("checkpoint", 0, item=key) for key in unconsumed[:20]),
            unconsumed=len(unconsumed),
        )
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
    live_reference_state = _consume_bin_edges(model, live_reference_state)
    msa_final_block = len(model.msa_trunk.blocks)

    def reference_key(model_key: str) -> _ReferenceKey:
        return _protenix_reference_key(model_key, msa_final_block=msa_final_block)

    return translate_reference_state(
        model,
        live_reference_state,
        reference_key,
        model_keys=model_keys,
    )


def _consume_bin_edges(model: nn.Module, reference_state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    r"""
    Check the released distance-bin edges against the target graph's, then drop them.

    The target graph computes its bin edges from its configuration instead of storing them, so the
    copies a released checkpoint carries are evidence to check rather than tensors to load. Its open
    final bin ends at ``+inf`` where the release stores a finite sentinel, so only the finite edges
    are compared.
    """

    remaining = dict(reference_state)
    for module_name, (lower_key, upper_key) in _RELEASED_BIN_EDGES.items():
        if lower_key not in remaining:
            continue
        lower, upper = remaining.pop(lower_key), remaining.pop(upper_key)
        if module_name == "noisy_structure_embedder":
            embedder = cast(ProtenixNoisyStructureEmbedder, model.get_submodule(module_name))
            configured_lower = torch.linspace(
                embedder.min_distance,
                embedder.max_distance,
                embedder.num_bins,
                device=lower.device,
                dtype=lower.dtype,
            )
            configured_upper = torch.cat((configured_lower[1:], configured_lower.new_tensor([torch.inf])))
        else:
            configured_lower = model.get_buffer(f"{module_name}.lower_bins")
            configured_upper = model.get_buffer(f"{module_name}.upper_bins")
        if not (
            torch.allclose(lower, configured_lower.to(lower))
            and torch.allclose(upper[:-1], configured_upper[:-1].to(upper))
        ):
            fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=module_name))
    return remaining


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
            fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=key))  # several source tensors normalize to it
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

    needs_esm = variant in _ESM_VARIANTS
    if needs_esm and esm_checkpoint_path is None:
        fail("value_invalid", RequestRef("esm_checkpoint_path", "None"))
    if not needs_esm and esm_checkpoint_path is not None:
        fail("value_invalid", RequestRef("esm_checkpoint_path", str(esm_checkpoint_path)))

    config = ProtenixConfig.from_variant(variant)
    model: ProtenixModel | ProtenixStructureConditionModel
    with init_empty_weights(include_buffers=False):
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
    if dtype is not None:
        model.to(dtype=dtype)
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
    sources: dict[str, str | PathLike[str]] = {"checkpoint": checkpoint_path}
    if esm_checkpoint_path is not None:
        sources["esm_checkpoint"] = esm_checkpoint_path
    save_repository(
        pipeline,
        output_dir,
        sources=sources,
        converter="ofoldx.models.protenix.convert.convert_protenix_checkpoint",
        options={"variant": variant, "dtype": None if dtype is None else str(dtype)},
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
        fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=model_key))
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
                "attention.projection.query_projection.weight": "self_attn.in_proj_weight",
                "attention.projection.query_projection.bias": "self_attn.in_proj_bias",
                "attention.projection.key_projection.weight": "self_attn.in_proj_weight",
                "attention.projection.value_projection.weight": "self_attn.in_proj_weight",
                "attention.projection.value_projection.bias": "self_attn.in_proj_bias",
                "attention.projection.output_projection.weight": "self_attn.out_proj.weight",
                "attention.projection.output_projection.bias": "self_attn.out_proj.bias",
                "attention_norm.weight": "norm1.weight",
                "attention_norm.bias": "norm1.bias",
                "feed_forward.input_projection.weight": "linear1.weight",
                "feed_forward.input_projection.bias": "linear1.bias",
                "feed_forward.output_projection.weight": "linear2.weight",
                "feed_forward.output_projection.bias": "linear2.bias",
                "feed_forward_norm.weight": "norm2.weight",
                "feed_forward_norm.bias": "norm2.bias",
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

    key = _replace_atom_module_key(model_key)
    key = _replace_top_level(key)
    key = _replace_confidence_key(key)
    key = _replace_token_stack_key(key)
    key = _replace_denoiser_key(key)
    key = atom_stack_reference_key(key)
    key = _replace_atom_key(key)
    key = _replace_pairformer_key(key)
    key = attention_reference_key(key)
    key = attention_projection_reference_key(key)
    key = _replace_transition_key(key)
    return conditioned_norm_reference_key(key)


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
                fail("checkpoint_invalid", SourceRef("checkpoint", 0, item=model_key))
            return first
        if model_key.endswith(".conditioning_projection.weight"):
            return torch.cat(reference_value, dim=0)
        raise RuntimeError(f"unexpected multi-tensor reference mapping for {model_key}")

    if model_key.startswith("constraint_embeddings.substructure_embeddings.block."):
        # PyTorch packs Q/K/V along dim 0. The ordinary CSC attention has no
        # key-head normalization, so its K-bias contributes only one constant
        # per softmax row and is mathematically dead, including with dropout
        # after softmax. Consume the packed source via Q/V and omit that K
        # slice; the learned V-bias remains live under attention dropout.
        projection_index = {
            ".attention.projection.query_projection.": 0,
            ".attention.projection.key_projection.": 1,
            ".attention.projection.value_projection.": 2,
        }
        for projection, index in projection_index.items():
            if projection in model_key:
                return reference_value.chunk(3, dim=0)[index].clone()

    if model_key == "msa_trunk.msa_projection.weight":
        (residue,) = _class_axis(reference_value[:, :-2], dim=-1)
        return torch.cat([residue, reference_value[:, -2:]], dim=1)
    if model_key == "template_embedder.template_projection.weight":
        return _convert_template_pair_axis(reference_value)
    if model_key == "esm_projection.weight":
        return _convert_token_feature_axis(reference_value, dim=0, planes=_copied_rows)
    if model_key in _TOKEN_FEATURE_INPUT_KEYS:
        return _convert_trailing_token_feature_axis(reference_value, dim=-1)
    if model_key.endswith(".noise_embedding.projection.weight") and reference_value.dim() == 1:
        return reference_value.unsqueeze(-1)
    return reference_value


_TOKEN_FEATURE_INPUT_KEYS = {
    "single_init_projection.weight",
    "msa_trunk.token_projection.weight",
    "confidence_head.pair_embedder.pair_left_projection.weight",
    "confidence_head.pair_embedder.pair_right_projection.weight",
}
_SOURCE_SINGLE_FEATURE_SIZE = 384
_SOURCE_VOCAB_SIZE = len(PROTENIX_SOURCE_TOKENS)
_SOURCE_TOKEN_FEATURE_SIZE = _SOURCE_SINGLE_FEATURE_SIZE + 2 * _SOURCE_VOCAB_SIZE + 1
_SOURCE_TEMPLATE_PAIR_PREFIX_SIZE = 40
_SOURCE_TEMPLATE_PAIR_SUFFIX_SIZE = 4


def _class_axis(*sources: Tensor, dim: int) -> tuple[Tensor, ...]:
    r"""
    Regenerate tensors indexed by the upstream alphabet along ``dim`` onto the 72 canonical classes:
    MultiMolecule ``convert_word_embeddings`` with ``OTHER`` a copy of ``X`` (upstream gives ligand tokens UNK).
    """

    converted = canonical_word_embeddings(*sources, vocabulary=PROTENIX_SOURCE_TOKENS, dim=dim)
    for tensor in converted:
        tensor.select(dim, TOKEN_INDEX["OTHER"]).copy_(tensor.select(dim, TOKEN_INDEX["X"]))
    return converted


def _copied_rows(*sources: Tensor, dim: int) -> tuple[Tensor, ...]:
    r"""
    The canonical axis of an output added to the token features: classes of the upstream alphabet copy their row,
    every other class is zero, so the projections reading the regenerated planes see exactly the upstream sum.
    """

    rows = [PROTENIX_SOURCE_TOKENS.index(token) if token in PROTENIX_SOURCE_TOKENS else -1 for token in TOKENS]
    converted = []
    for source in sources:
        moved = source.movedim(dim, 0)
        target = moved.new_zeros((len(TOKENS), *moved.shape[1:]))
        for column, row in enumerate(rows):
            if row >= 0:
                target[column] = moved[row]
        converted.append(target.movedim(0, dim).contiguous())
    return tuple(converted)


def _convert_trailing_token_feature_axis(source: Tensor, *, dim: int) -> Tensor:
    source = source.movedim(dim, -1)
    prefix, token_feature = (
        source[..., :-_SOURCE_TOKEN_FEATURE_SIZE],
        source[..., -_SOURCE_TOKEN_FEATURE_SIZE:],
    )
    converted = torch.cat([prefix, _convert_token_feature_axis(token_feature, dim=-1)], dim=-1)
    return converted.movedim(-1, dim).contiguous()


def _convert_token_feature_axis(
    source: Tensor, *, dim: int, planes: Callable[..., tuple[Tensor, ...]] = _class_axis
) -> Tensor:
    r"""``[single | class | profile | deletion]`` along ``dim``: both class planes onto the canonical classes."""
    source = source.movedim(dim, -1)
    structural_start = _SOURCE_SINGLE_FEATURE_SIZE
    profile_start = structural_start + _SOURCE_VOCAB_SIZE
    deletion_start = profile_start + _SOURCE_VOCAB_SIZE
    structural, profile = planes(
        source[..., structural_start:profile_start], source[..., profile_start:deletion_start], dim=-1
    )
    converted = torch.cat([source[..., :structural_start], structural, profile, source[..., deletion_start:]], dim=-1)
    return converted.movedim(-1, dim).contiguous()


def _convert_template_pair_axis(source: Tensor) -> Tensor:
    token_j_start = _SOURCE_TEMPLATE_PAIR_PREFIX_SIZE
    token_i_start = token_j_start + _SOURCE_VOCAB_SIZE
    suffix_start = token_i_start + _SOURCE_VOCAB_SIZE
    token_j, token_i = _class_axis(
        source[..., token_j_start:token_i_start], source[..., token_i_start:suffix_start], dim=-1
    )
    return torch.cat([source[..., :token_j_start], token_i, token_j, source[..., suffix_start:]], dim=-1)


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
        "confidence_pairformer_stack": "confidence_head.pairformer_stack",
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
        "confidence_head.pair_embedder.pair_left_projection": "confidence_head.linear_no_bias_s2",
        "confidence_head.pair_embedder.pair_right_projection": "confidence_head.linear_no_bias_s1",
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
    return token_transformer_reference_key(key)


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
            ".conditioner.single_norm.": ".diffusion_conditioning.layernorm_s.",
            ".conditioner.single_projection.": ".diffusion_conditioning.linear_no_bias_s.",
            ".conditioner.noise_embedding.projection.weight": ".diffusion_conditioning.fourier_embedding.w",
            ".conditioner.noise_embedding.projection.bias": ".diffusion_conditioning.fourier_embedding.b",
            ".conditioner.noise_norm.": ".diffusion_conditioning.layernorm_n.",
            ".conditioner.noise_projection.": ".diffusion_conditioning.linear_no_bias_n.",
            ".conditioner.transitions.0.": ".diffusion_conditioning.transition_s1.",
            ".conditioner.transitions.1.": ".diffusion_conditioning.transition_s2.",
            ".token_refiner.single_norm.": ".layernorm_s.",
            ".token_refiner.single_to_token_projection.": ".linear_no_bias_s.",
            ".token_refiner.token_encoder.": ".diffusion_transformer.",
            ".token_refiner.token_norm.": ".layernorm_a.",
        },
    )


# The Protenix atom attention encoder or decoder that holds each local-atom child of the input embeddings
# and the coordinate denoiser. The children split one Protenix module by concern; the prefixes they keep
# resolve in ``atom_stack_reference_key`` and ``_replace_atom_key``.
_ATOM_MODULE_PREFIXES = (
    ("input_embeddings.atom_encoder.", "input_embedder.atom_attention_encoder."),
    ("input_embeddings.", "input_embedder.atom_attention_encoder."),
    ("coordinate_denoiser.atom_conditioner.", "diffusion_module.atom_attention_encoder.atom_conditioner."),
    ("coordinate_denoiser.trunk_conditioning.", "diffusion_module.atom_attention_encoder.trunk_conditioning."),
    ("coordinate_denoiser.atom_pair_encoder.", "diffusion_module.atom_attention_encoder.atom_pair_encoder."),
    (
        "coordinate_denoiser.atom_encoder_pair_bias_projection.",
        "diffusion_module.atom_attention_encoder.atom_pair_bias_projection.",
    ),
    ("coordinate_denoiser.atom_encoder.", "diffusion_module.atom_attention_encoder."),
    (
        "coordinate_denoiser.atom_decoder_pair_bias_projection.",
        "diffusion_module.atom_attention_decoder.atom_pair_bias_projection.",
    ),
    ("coordinate_denoiser.atom_decoder.", "diffusion_module.atom_attention_decoder."),
)


def _replace_atom_module_key(key: str) -> str:
    for source, target in _ATOM_MODULE_PREFIXES:
        if key.startswith(source):
            return target + key.removeprefix(source)
    return key


def _replace_atom_key(key: str) -> str:
    return _replace_many(
        key,
        {
            ".atom_conditioner.reference_position_projection.": ".linear_no_bias_ref_pos.",
            ".atom_conditioner.reference_charge_projection.": ".linear_no_bias_ref_charge.",
            ".atom_conditioner.atom_feature_projection.": ".linear_no_bias_f.",
            ".atom_pair_encoder.distance_vector_projection.": ".linear_no_bias_d.",
            ".atom_pair_encoder.inverse_distance_projection.": ".linear_no_bias_invd.",
            ".atom_pair_encoder.distance_mask_projection.": ".linear_no_bias_v.",
            ".atom_pair_encoder.left_atom_projection.": ".linear_no_bias_cl.",
            ".atom_pair_encoder.right_atom_projection.": ".linear_no_bias_cm.",
            ".atom_pair_encoder.feed_forward.input_projection.": ".small_mlp.1.",
            ".atom_pair_encoder.feed_forward.intermediate_projections.0.": ".small_mlp.3.",
            ".atom_pair_encoder.feed_forward.output_projection.": ".small_mlp.5.",
            ".trunk_conditioning.single_norm.": ".layernorm_s.",
            ".trunk_conditioning.single_projection.": ".linear_no_bias_s.",
            ".trunk_conditioning.pair_norm.": ".layernorm_z.",
            ".trunk_conditioning.pair_projection.": ".linear_no_bias_z.",
            ".atom_attention_encoder.noisy_position_projection.": ".atom_attention_encoder.linear_no_bias_r.",
            ".atom_attention_encoder.token_projection.": ".atom_attention_encoder.linear_no_bias_q.",
            ".atom_attention_decoder.token_projection.": ".atom_attention_decoder.linear_no_bias_a.",
            ".atom_attention_decoder.position_norm.": ".atom_attention_decoder.layernorm_q.",
            ".atom_attention_decoder.position_update_projection.": ".atom_attention_decoder.linear_no_bias_out.",
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


def _replace_many(key: str, replacements: Mapping[str, str]) -> str:
    for source, target in replacements.items():
        key = key.replace(source, target)
    return key


if __name__ == "__main__":
    main()
