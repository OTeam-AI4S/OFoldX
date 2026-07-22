r"""
Official DeepMind AlphaFold 3 parameter reader.

DeepMind distributes AlphaFold 3 parameters separately from the Apache-2.0
source code. This module loads and converts local official record-stream parameter
files, but does not bundle, download, or publish those weights.
"""

from __future__ import annotations

import contextlib
import io
import re
import struct
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ._schema import resolve

_RECORD_HEADER = struct.Struct("<5i")
_MODEL_FILE_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (re.compile(r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$"), True),
    (re.compile(r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$"), True),
    (re.compile(r"(?P<model_name>.*)\.[0-9]+\.bin$"), False),
    (re.compile(r"(?P<model_name>.*)\.bin\.[0-9]+$"), False),
    (re.compile(r"(?P<model_name>.*)\.bin\.zst$"), True),
    (re.compile(r"(?P<model_name>.*)\.bin$"), False),
)
_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_SIZES = {name: torch.empty((), dtype=dtype).element_size() for name, dtype in _TORCH_DTYPES.items()}
# The official graph calls ``_per_atom_conditioning`` for its single output while
# Haiku still materializes the discarded pair-output projections; these eight
# tensors never participate in inference. The ninth record is file metadata.
_UNUSED_OFFICIAL_KEYS = {
    "__meta__//__identifier__",
    "diffuser/evoformer_conditioning_embed_pair_distances//weights",
    "diffuser/evoformer_conditioning_embed_pair_offsets//weights",
    "diffuser/evoformer_conditioning_single_to_pair_cond_col//weights",
    "diffuser/evoformer_conditioning_single_to_pair_cond_row//weights",
    "diffuser/~/diffusion_head/diffusion_embed_pair_distances//weights",
    "diffuser/~/diffusion_head/diffusion_embed_pair_offsets//weights",
    "diffuser/~/diffusion_head/diffusion_single_to_pair_cond_col//weights",
    "diffuser/~/diffusion_head/diffusion_single_to_pair_cond_row//weights",
}


def convert_official_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    r"""
    Convert official DeepMind AlphaFold 3 parameters into a strict-loaded native model.

    This is the in-memory parameter reader. Persistence to a standard repository (and any
    private-safe Hub upload) is owned by
    [`convert_alphafold3_checkpoint`][ofoldx.models.alphafold3.convert.convert_alphafold3_checkpoint].
    """

    from ..config import AlphaFold3Config
    from ..model import AlphaFold3Model

    checkpoint_path = Path(checkpoint_path)
    official_state = load_official_alphafold3_params(checkpoint_path, model_name=model_name)
    with torch.device("meta"):
        model = AlphaFold3Model(AlphaFold3Config())
    state_dict = translate_official_alphafold3_state(model, official_state, dtype=dtype)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    return model


def translate_official_alphafold3_state(
    model: torch.nn.Module,
    official_state: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Translate official DeepMind AlphaFold 3 Haiku parameters into OFoldX keys.
    """

    translated = {}
    used_official_keys: set[str] = set()

    for model_key, model_tensor in model.state_dict().items():
        ref = _official_ref_for_model_key(model_key)
        if ref is None:
            continue
        if all(key.startswith("<") for key in ref.keys):
            source_tensors = ()
        else:
            used_official_keys.update(ref.keys)
            source_tensors = tuple(official_state[key] for key in ref.keys)
        converted = _apply_official_transform(ref, source_tensors, model_tensor)
        if ref.negate:
            converted = -converted
        if dtype is not None and converted.is_floating_point():
            converted = converted.to(dtype=dtype)
        # Per-block refs slice the official stacked buffers as views, so several converted
        # keys can alias one storage. Give every tensor its own contiguous storage; otherwise
        # ``safetensors`` refuses to persist the aliased state dict (no name covers the buffer).
        translated[model_key] = converted.detach().clone(memory_format=torch.contiguous_format)

    unexpected = sorted(set(official_state) - used_official_keys - _UNUSED_OFFICIAL_KEYS)
    if unexpected:
        raise RuntimeError(
            "checkpoint contains unexpected official AlphaFold 3 tensors\n"
            + "\n".join(f"  {key}" for key in unexpected[:40])
        )
    return translated


def select_official_model_files(
    model_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> tuple[tuple[Path, ...], bool, str]:
    r"""
    Select official AlphaFold 3 parameter files from a model directory.
    """

    root = Path(model_dir)
    if root.is_file():
        for pattern, compressed in _MODEL_FILE_PATTERNS:
            match = pattern.fullmatch(root.name)
            if match is None:
                continue
            resolved_name = match.group("model_name")
            if model_name is not None and model_name != resolved_name:
                raise FileNotFoundError(f"official AlphaFold 3 model {model_name!r} did not match file {root.name!r}")
            return (root,), compressed, resolved_name
        raise FileNotFoundError(f"no official AlphaFold 3 model parameter file pattern matched {root}")
    files = [path for path in root.iterdir() if path.is_file()] if root.is_dir() else []
    for pattern, compressed in _MODEL_FILE_PATTERNS:
        models = _match_model_files(files, pattern)
        if model_name is not None:
            selected = models.get(model_name)
            if selected:
                return tuple(selected), compressed, model_name
        elif models:
            if len(models) > 1:
                names = ", ".join(sorted(models))
                raise RuntimeError(f"multiple official AlphaFold 3 models matched in {root}: {names}")
            resolved_name, selected = next(iter(models.items()))
            return tuple(selected), compressed, resolved_name
    raise FileNotFoundError(f"no official AlphaFold 3 model parameter files matched in {root}")


def read_official_alphafold3_records(stream: Any) -> Iterator[tuple[str, str, Tensor]]:
    r"""
    Yield ``(scope, name, tensor)`` records from an official AF3 parameter stream.
    """

    while record := _read_record(stream):
        yield record


def load_official_alphafold3_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Tensor]:
    r"""Load official DeepMind AlphaFold 3 parameters as ``"scope//name"`` tensors."""

    model_files, compressed, resolved_model_name = select_official_model_files(params_path, model_name=model_name)
    tensors: dict[str, Tensor] = {}
    with _open_model_stream(model_files, compressed=compressed) as stream:
        for scope, name, tensor in read_official_alphafold3_records(stream):
            key = f"{scope}//{name}"
            if key in tensors:
                raise ValueError(f"multiple official AlphaFold 3 records map to {key!r}")
            tensors[key] = tensor
    if not tensors:
        raise FileNotFoundError(f"official AlphaFold 3 model {resolved_model_name!r} has no parameter records")
    return tensors


def _match_model_files(paths: Sequence[Path], pattern: re.Pattern[str]) -> dict[str, list[Path]]:
    models: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is not None:
            models[match.group("model_name")].append(path)
    return {name: sorted(files) for name, files in models.items()}


@contextlib.contextmanager
def _open_model_stream(model_files: Sequence[Path], *, compressed: bool) -> Iterator[Any]:
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(path.open("rb")) for path in model_files]
        stream = _ConcatenatedBinaryReader(handles)
        if compressed:
            import zstandard

            zstd_stream = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(stream))
            try:
                yield zstd_stream
            finally:
                zstd_stream.close()
        else:
            yield stream


class _ConcatenatedBinaryReader(io.RawIOBase):
    def __init__(self, handles: Sequence[Any]) -> None:
        self._handles = tuple(handles)
        self._index = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer)
        total = 0
        while view and self._index < len(self._handles):
            chunk = self._handles[self._index].read(len(view))
            if not chunk:
                self._index += 1
                continue
            view[: len(chunk)] = chunk
            view = view[len(chunk) :]
            total += len(chunk)
        return total


def _read_record(stream: Any) -> tuple[str, str, Tensor] | None:
    header = stream.read(_RECORD_HEADER.size)
    if not header:
        return None
    if len(header) < _RECORD_HEADER.size:
        raise ValueError(f"incomplete official AF3 record header: {len(header)} < {_RECORD_HEADER.size}")
    scope_len, name_len, dtype_len, shape_len, array_bytes = _RECORD_HEADER.unpack(header)
    payload_format = f"<{scope_len}s{name_len}s{dtype_len}s{shape_len}i"
    metadata_bytes = struct.calcsize(payload_format)
    payload = bytearray(stream.read(metadata_bytes + array_bytes))
    if len(payload) < metadata_bytes + array_bytes:
        raise ValueError(f"incomplete official AF3 record payload: {len(payload)} < {metadata_bytes + array_bytes}")
    scope_raw, name_raw, dtype_raw, *shape = struct.unpack_from(payload_format, payload)
    dtype = dtype_raw.decode("utf-8")
    torch_dtype = _TORCH_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(f"unsupported official AF3 tensor dtype: {dtype!r}")
    expected_bytes = _numel(shape) * _DTYPE_SIZES[dtype]
    if expected_bytes != array_bytes:
        raise ValueError(
            f"official AF3 tensor byte count mismatch for {scope_raw!r}//{name_raw!r}: "
            f"header has {array_bytes}, shape/dtype implies {expected_bytes}"
        )
    raw = memoryview(payload)[metadata_bytes:]
    if sys.byteorder == "big" and _DTYPE_SIZES[dtype] > 1:
        element_size = _DTYPE_SIZES[dtype]
        for index in range(0, len(raw), element_size):
            raw[index : index + element_size] = bytes(raw[index : index + element_size])[::-1]
    tensor = torch.frombuffer(raw, dtype=torch_dtype).reshape(shape)
    return scope_raw.decode("utf-8"), name_raw.decode("utf-8"), tensor


def _numel(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


@dataclass(frozen=True)
class _OfficialParamRef:
    keys: tuple[str, ...]
    transform: str = "identity"
    indices: tuple[int, ...] = ()
    negate: bool = False


def _ref(key: str, transform: str = "identity", *indices: int, negate: bool = False) -> _OfficialParamRef:
    return _OfficialParamRef((key,), transform=transform, indices=tuple(indices), negate=negate)


def _multi_ref(keys: Sequence[str], transform: str, *indices: int, negate: bool = False) -> _OfficialParamRef:
    return _OfficialParamRef(tuple(keys), transform=transform, indices=tuple(indices), negate=negate)


def _k(scope: str, name: str) -> str:
    return f"{scope}//{name}"


_DIFFUSION_HEAD = "diffuser/~/diffusion_head"
_MSA_STACK_SCOPE = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"
# Per-context Haiku scopes for the four Pairformer-block instances enumerated by the shared schema.
# The DeepMind layout math (splits, transposes, glu chunking) stays entirely inside the leaf helpers.
_PAIRFORMER_CONTEXTS = {
    "trunk": ("diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer", True, "pair_feed_forward"),
    "msa_pair": (_MSA_STACK_SCOPE, False, "feed_forward"),
    "template": (
        "diffuser/evoformer/template_embedding/single_template_embedding/"
        "__layer_stack_no_per_layer/template_embedding_iteration",
        False,
        "feed_forward",
    ),
    "confidence": (
        "diffuser/confidence_head/__layer_stack_no_per_layer/confidence_pairformer",
        True,
        "pair_feed_forward",
    ),
}
_ATOM_ENCODER_CONTEXTS = {
    "input": (
        "diffuser/evoformer_conditioning",
        "diffuser/evoformer_conditioning_atom_transformer_encoder/"
        "__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder",
        False,
    ),
    "diffusion": (
        f"{_DIFFUSION_HEAD}/diffusion",
        f"{_DIFFUSION_HEAD}/diffusion_atom_transformer_encoder/"
        "__layer_stack_with_per_layer/diffusion_atom_transformer_encoder",
        True,
    ),
}
_MSA_TOP_LEVEL = {
    "msa_projection.weight": ("diffuser/evoformer/msa_activations", "linear"),
    "token_projection.weight": ("diffuser/evoformer/extra_msa_target_feat", "target_feat_linear"),
}


def _official_ref_for_model_key(model_key: str) -> _OfficialParamRef | None:
    node = resolve(model_key)
    if node is None:
        return None
    if node.section == "top":
        return _top_level_ref(node.rest)
    if node.section == "atom_encoder":
        base, stack, include_trunk = _ATOM_ENCODER_CONTEXTS[node.context]
        return _atom_encoder_ref(node.rest, base=base, stack=stack, include_trunk=include_trunk)
    if node.section == "pairformer":
        assert node.block is not None
        scope, with_single, pair_transition_target = _PAIRFORMER_CONTEXTS[node.context]
        return _pairformer_block_ref(
            node.rest, node.block, scope, with_single=with_single, pair_transition_target=pair_transition_target
        )
    if node.section == "msa":
        entry = _MSA_TOP_LEVEL.get(node.rest)
        return None if entry is None else _ref(_k(entry[0], "weights"), entry[1])
    if node.section == "msa_outer_product_mean":
        assert node.block is not None
        return _outer_product_mean_ref(node.rest, node.block, _MSA_STACK_SCOPE)
    if node.section == "msa_attention":
        assert node.block is not None
        return _msa_attention_ref(node.rest, node.block, _MSA_STACK_SCOPE)
    if node.section == "msa_transition":
        return _transition_ref(node.rest, node.block, f"{_MSA_STACK_SCOPE}/msa_transition")
    if node.section == "template":
        return _template_ref(node.rest)
    if node.section == "confidence":
        return _confidence_ref(node.rest)
    if node.section == "denoiser_conditioner":
        return _denoiser_conditioner_ref(node.rest, _DIFFUSION_HEAD)
    if node.section == "token_refiner":
        return _token_refiner_ref(node.rest, _DIFFUSION_HEAD)
    if node.section == "atom_decoder":
        return _atom_decoder_ref(node.rest, _DIFFUSION_HEAD)
    return None


def _top_level_ref(model_key: str) -> _OfficialParamRef | None:
    mapping = {
        "relative_position_embedder.output_projection.weight": _ref(
            _k("diffuser/evoformer/~_relative_encoding/position_activations", "weights"), "linear"
        ),
        "single_init_projection.weight": _ref(
            _k("diffuser/evoformer/single_activations", "weights"), "target_feat_linear"
        ),
        "pair_left_projection.weight": _ref(_k("diffuser/evoformer/left_single", "weights"), "target_feat_linear"),
        "pair_right_projection.weight": _ref(_k("diffuser/evoformer/right_single", "weights"), "target_feat_linear"),
        "token_bond_projection.weight": _ref(_k("diffuser/evoformer/bond_embedding", "weights"), "linear"),
        "pair_recycle_projection.weight": _ref(_k("diffuser/evoformer/prev_embedding", "weights"), "linear"),
        "single_recycle_projection.weight": _ref(_k("diffuser/evoformer/prev_single_embedding", "weights"), "linear"),
        "pair_recycle_norm.weight": _ref(_k("diffuser/evoformer/prev_embedding_layer_norm", "scale")),
        "pair_recycle_norm.bias": _ref(_k("diffuser/evoformer/prev_embedding_layer_norm", "offset")),
        "single_recycle_norm.weight": _ref(_k("diffuser/evoformer/prev_single_embedding_layer_norm", "scale")),
        "single_recycle_norm.bias": _ref(_k("diffuser/evoformer/prev_single_embedding_layer_norm", "offset")),
        "distogram_head.linear.weight": _ref(_k("diffuser/distogram_head/half_logits", "weights"), "linear"),
    }
    return mapping.get(model_key)


def _outer_product_mean_ref(rest: str, block: int, scope: str) -> _OfficialParamRef | None:
    norm_scope = f"{scope}/outer_product_mean/layer_norm_input"
    opm_scope = f"{scope}/outer_product_mean"
    mapping = {
        "norm_msa.weight": _ref(_k(norm_scope, "scale"), "identity", block),
        "norm_msa.bias": _ref(_k(norm_scope, "offset"), "identity", block),
        "left_projection.weight": _ref(_k(f"{opm_scope}/left_projection", "weights"), "linear", block),
        "right_projection.weight": _ref(_k(f"{opm_scope}/right_projection", "weights"), "linear", block),
        "output_projection.weight": _ref(_k(opm_scope, "output_w"), "opm_output", block),
        "output_projection.bias": _ref(_k(opm_scope, "output_b"), "identity", block),
    }
    return mapping.get(rest)


def _msa_attention_ref(rest: str, block: int, scope: str) -> _OfficialParamRef | None:
    base = f"{scope}/msa_attention1"
    mapping = {
        "norm_msa.weight": _ref(_k(f"{base}/act_norm", "scale"), "identity", block),
        "norm_msa.bias": _ref(_k(f"{base}/act_norm", "offset"), "identity", block),
        "norm_pair.weight": _ref(_k(f"{base}/pair_norm", "scale"), "identity", block),
        "norm_pair.bias": _ref(_k(f"{base}/pair_norm", "offset"), "identity", block),
        "value_projection.weight": _ref(_k(f"{base}/v_projection", "weights"), "linear", block),
        "pair_projection.weight": _ref(_k(f"{base}/pair_logits", "weights"), "linear", block),
        "gate_projection.weight": _ref(_k(f"{base}/gating_query", "weights"), "linear", block),
        "output_projection.weight": _ref(_k(f"{base}/output_projection", "weights"), "linear", block),
    }
    return mapping.get(rest)


def _template_ref(rest: str) -> _OfficialParamRef | None:
    base = "diffuser/evoformer/template_embedding"
    single = f"{base}/single_template_embedding"
    if rest == "norm_pair.weight":
        return _ref(_k(f"{single}/query_embedding_norm", "scale"))
    if rest == "norm_pair.bias":
        return _ref(_k(f"{single}/query_embedding_norm", "offset"))
    if rest == "pair_projection.weight":
        return _ref(_k(f"{single}/template_pair_embedding_8", "weights"), "linear")
    if rest == "template_projection.weight":
        return _multi_ref(
            [_k(f"{single}/template_pair_embedding_{index}", "weights") for index in range(8)],
            "template_projection",
        )
    if rest == "hidden_norm.weight":
        return _ref(_k(f"{single}/output_layer_norm", "scale"))
    if rest == "hidden_norm.bias":
        return _ref(_k(f"{single}/output_layer_norm", "offset"))
    if rest == "output_projection.weight":
        return _ref(_k(f"{base}/output_linear", "weights"), "linear")
    return None


def _confidence_ref(rest: str) -> _OfficialParamRef | None:
    base = "diffuser/confidence_head"
    if rest == "pair_embedder.left_token_projection.weight":
        # Consumes ``target_feat``; rotate the input columns to the native [atom|tail] token-embedding
        # order (the trunk single_init/pair projections already do this via ``target_feat_linear``).
        return _ref(_k(f"{base}/~_embed_features/right_target_feat_project", "weights"), "target_feat_linear")
    if rest == "pair_embedder.right_token_projection.weight":
        return _ref(_k(f"{base}/~_embed_features/left_target_feat_project", "weights"), "target_feat_linear")
    if rest == "pair_embedder.distance_one_hot_projection.weight":
        return _ref(_k(f"{base}/~_embed_features/distogram_feat_project", "weights"), "linear")
    mapping = {
        "atom_projector.plddt_weight": _ref(_k(f"{base}/plddt_logits", "weights"), "atom_confidence_weight"),
        "atom_projector.resolved_weight": _ref(
            _k(f"{base}/experimentally_resolved_logits", "weights"), "atom_confidence_weight"
        ),
        "atom_projector.plddt_norm.weight": _ref(_k(f"{base}/plddt_logits_ln", "scale")),
        "atom_projector.plddt_norm.bias": _ref(_k(f"{base}/plddt_logits_ln", "offset")),
        "atom_projector.resolved_norm.weight": _ref(_k(f"{base}/experimentally_resolved_ln", "scale")),
        "atom_projector.resolved_norm.bias": _ref(_k(f"{base}/experimentally_resolved_ln", "offset")),
        "pair_projector.pae_norm.weight": _ref(_k(f"{base}/pae_logits_ln", "scale")),
        "pair_projector.pae_norm.bias": _ref(_k(f"{base}/pae_logits_ln", "offset")),
        "pair_projector.pae_projection.weight": _ref(_k(f"{base}/pae_logits", "weights"), "linear"),
        "pair_projector.pde_norm.weight": _ref(_k(f"{base}/logits_ln", "scale")),
        "pair_projector.pde_norm.bias": _ref(_k(f"{base}/logits_ln", "offset")),
        "pair_projector.pde_projection.weight": _ref(_k(f"{base}/left_half_distance_logits", "weights"), "linear"),
    }
    return mapping.get(rest)


def _denoiser_conditioner_ref(rest: str, base: str) -> _OfficialParamRef | None:
    mapping = {
        "pair_input_norm.weight": _ref(_k(f"{base}/pair_cond_initial_norm", "scale")),
        "pair_projection.weight": _ref(_k(f"{base}/pair_cond_initial_projection", "weights"), "linear"),
        # The norm runs over the concatenated [single (384), target_feat (447)] channels before the
        # projection, so its per-channel scale must be rotated into the native [atom|tail] order too.
        "single_conditioning_norm.weight": _ref(
            _k(f"{base}/single_cond_initial_norm", "scale"), "single_prefixed_target_feat_scale"
        ),
        # Input is concat([single (384), target_feat (447)]); rotate ONLY the trailing target_feat
        # columns into the native [atom|tail] order (the single prefix is untouched).
        "single_conditioning_projection.weight": _ref(
            _k(f"{base}/single_cond_initial_projection", "weights"), "single_prefixed_target_feat_linear"
        ),
        "noise_embedding.projection.weight": _ref("<official_noise_fourier_weight>", "official_noise_fourier_weight"),
        "noise_embedding.projection.bias": _ref("<official_noise_fourier_bias>", "official_noise_fourier_bias"),
        "noise_norm.weight": _ref(_k(f"{base}/noise_embedding_initial_norm", "scale")),
        "noise_projection.weight": _ref(_k(f"{base}/noise_embedding_initial_projection", "weights"), "linear"),
    }
    if rest in mapping:
        return mapping[rest]
    for index, target in enumerate(("pair_feed_forward_1", "pair_feed_forward_2")):
        if rest.startswith(target + "."):
            return _flat_diffusion_transition_ref(rest.removeprefix(target + "."), f"{base}/pair_transition_{index}ffw")
    for index, target in enumerate(("single_feed_forward_1", "single_feed_forward_2")):
        if rest.startswith(target + "."):
            return _flat_diffusion_transition_ref(
                rest.removeprefix(target + "."), f"{base}/single_transition_{index}ffw"
            )
    return None


def _token_refiner_ref(rest: str, base: str) -> _OfficialParamRef | None:
    mapping = {
        "single_norm.weight": _ref(_k(f"{base}/single_cond_embedding_norm", "scale")),
        "single_to_token_projection.weight": _ref(_k(f"{base}/single_cond_embedding_projection", "weights"), "linear"),
        "token_norm.weight": _ref(_k(f"{base}/output_norm", "scale")),
    }
    if rest in mapping:
        return mapping[rest]
    if match := re.match(r"token_encoder\.pair_norms\.(\d+)\.weight", rest):
        return _ref(_k(f"{base}/transformer/pair_input_layer_norm", "scale"))
    if match := re.match(r"token_encoder\.bias_projections\.(\d+)\.weight", rest):
        layer = int(match.group(1))
        return _ref(
            _k(
                f"{base}/transformer/__layer_stack_with_per_layer/pair_logits_projection",
                "weights",
            ),
            "token_pair_logits",
            layer // 4,
            layer % 4,
        )
    if match := re.match(r"token_encoder\.transformer\.layers\.(\d+)\.(.+)", rest):
        layer = int(match.group(1))
        return _conditioned_transformer_layer_ref(
            match.group(2),
            f"{base}/transformer/__layer_stack_with_per_layer/__layer_stack_with_per_layer/transformer",
            indices=(layer // 4, layer % 4),
            has_key_value_norm=False,
            conditioning_prefix="single_cond",
        )
    return None


def _atom_decoder_ref(rest: str, base: str) -> _OfficialParamRef | None:
    mapping = {
        "token_projection.weight": _ref(
            _k(f"{base}/diffusion_project_token_features_for_broadcast", "weights"), "linear"
        ),
        "query_norm.weight": _ref(_k(f"{base}/diffusion_atom_features_layer_norm", "scale")),
        "output_projection.weight": _ref(_k(f"{base}/diffusion_atom_features_to_position_update", "weights"), "linear"),
    }
    if rest in mapping:
        return mapping[rest]
    if rest.startswith("atom_attention_stack."):
        return _atom_attention_stack_ref(
            rest.removeprefix("atom_attention_stack."),
            f"{base}/diffusion_atom_transformer_decoder/"
            "__layer_stack_with_per_layer/diffusion_atom_transformer_decoder",
        )
    return None


def _atom_encoder_ref(rest: str, *, base: str, stack: str, include_trunk: bool) -> _OfficialParamRef | None:
    mapping = {
        "atom_conditioner.reference_position_projection.weight": _ref(_k(f"{base}_embed_ref_pos", "weights"), "linear"),
        "atom_conditioner.reference_charge_projection.weight": _ref(
            _k(f"{base}_embed_ref_charge", "weights"), "linear"
        ),
        "atom_conditioner.atom_feature_projection.weight": _multi_ref(
            (
                _k(f"{base}_embed_ref_mask", "weights"),
                _k(f"{base}_embed_ref_element", "weights"),
                _k(f"{base}_embed_ref_atom_name", "weights"),
            ),
            "atom_feature_projection",
        ),
        "atom_pair_encoder.distance_vector_projection.weight": _ref(
            _k(f"{base}_embed_pair_offsets_1", "weights"), "linear"
        ),
        "atom_pair_encoder.inverse_distance_projection.weight": _ref(
            _k(f"{base}_embed_pair_distances_1", "weights"), "linear"
        ),
        "atom_pair_encoder.distance_mask_projection.weight": _ref(
            _k(f"{base}_embed_pair_offsets_valid", "weights"), "linear"
        ),
        "atom_pair_encoder.left_atom_projection.weight": _ref(
            _k(f"{base}_single_to_pair_cond_row_1", "weights"), "linear"
        ),
        "atom_pair_encoder.right_atom_projection.weight": _ref(
            _k(f"{base}_single_to_pair_cond_col_1", "weights"), "linear"
        ),
        "atom_pair_encoder.feed_forward.1.weight": _ref(_k(f"{base}_pair_mlp_1", "weights"), "linear"),
        "atom_pair_encoder.feed_forward.3.weight": _ref(_k(f"{base}_pair_mlp_2", "weights"), "linear"),
        "atom_pair_encoder.feed_forward.5.weight": _ref(_k(f"{base}_pair_mlp_3", "weights"), "linear"),
        "token_projection.weight": _ref(_k(f"{base}_project_atom_features_for_aggr", "weights"), "linear"),
    }
    if include_trunk:
        mapping.update(
            {
                "single_norm.weight": _ref(_k(f"{base}_lnorm_trunk_single_cond", "scale")),
                "single_to_atom_projection.weight": _ref(_k(f"{base}_embed_trunk_single_cond", "weights"), "linear"),
                "norm_pair.weight": _ref(_k(f"{base}_lnorm_trunk_pair_cond", "scale")),
                "pair_projection.weight": _ref(_k(f"{base}_embed_trunk_pair_cond", "weights"), "linear"),
                "noisy_position_projection.weight": _ref(_k(f"{base}_atom_positions_to_features", "weights"), "linear"),
            }
        )
    if rest in mapping:
        return mapping[rest]
    if rest.startswith("atom_attention_stack."):
        return _atom_attention_stack_ref(rest.removeprefix("atom_attention_stack."), stack)
    return None


def _atom_attention_stack_ref(rest: str, stack: str) -> _OfficialParamRef | None:
    root = stack.split("/__layer_stack_with_per_layer", 1)[0]
    if match := re.match(r"pair_norms\.(\d+)\.weight", rest):
        return _ref(_k(f"{root}/pair_input_layer_norm", "scale"))
    if match := re.match(r"bias_projections\.(\d+)\.weight", rest):
        return _ref(_k(f"{root}/pair_logits_projection", "weights"), "atom_pair_logits", int(match.group(1)))
    if match := re.match(r"atom_attention\.transformer\.layers\.(\d+)\.(.+)", rest):
        return _conditioned_transformer_layer_ref(
            match.group(2),
            stack,
            indices=(int(match.group(1)),),
            has_key_value_norm=True,
            conditioning_prefix="qsingle_cond",
            key_conditioning_prefix="ksingle_cond",
        )
    return None


def _pairformer_block_ref(
    rest: str,
    block: int,
    scope: str,
    *,
    with_single: bool,
    pair_transition_target: str,
) -> _OfficialParamRef | None:
    for target, official in (
        ("triangle_multiplication_outgoing", "triangle_multiplication_outgoing"),
        ("triangle_multiplication_incoming", "triangle_multiplication_incoming"),
    ):
        if rest.startswith(target + "."):
            return _triangle_multiplication_ref(rest.removeprefix(target + "."), block, f"{scope}/{official}")
    for target, official in (
        ("triangle_attention_starting_node", "pair_attention1"),
        ("triangle_attention_ending_node", "pair_attention2"),
    ):
        if rest.startswith(target + "."):
            return _grid_attention_ref(rest.removeprefix(target + "."), block, f"{scope}/{official}")
    if rest.startswith(pair_transition_target + "."):
        return _transition_ref(rest.removeprefix(pair_transition_target + "."), block, f"{scope}/pair_transition")
    if with_single and rest.startswith("pair_attention."):
        return _single_attention_ref(rest.removeprefix("pair_attention."), block, scope)
    if with_single and rest.startswith("single_feed_forward."):
        return _transition_ref(rest.removeprefix("single_feed_forward."), block, f"{scope}/single_transition")
    return None


def _triangle_multiplication_ref(rest: str, block: int, scope: str) -> _OfficialParamRef | None:
    # DeepMind interleaves the 2*C-wide projection/gate output into the two triangle factors
    # (a = even output columns, b = odd). For the OUTGOING equation the native ``left``/``right``
    # factors map directly onto (a, b); the INCOMING native equation (``kih,kjh->ijh``) is the
    # transpose of DeepMind's incoming (``ckj,cki->cij``), so ``left``/``right`` must take the
    # swapped halves (left = odd, right = even) to reproduce it.
    incoming = scope.endswith("incoming")
    left_split = "split_right_linear" if incoming else "split_left_linear"
    right_split = "split_left_linear" if incoming else "split_right_linear"
    mapping = {
        "norm_input.weight": _ref(_k(f"{scope}/left_norm_input", "scale"), "identity", block),
        "norm_input.bias": _ref(_k(f"{scope}/left_norm_input", "offset"), "identity", block),
        "norm_output.weight": _ref(_k(f"{scope}/center_norm", "scale"), "identity", block),
        "norm_output.bias": _ref(_k(f"{scope}/center_norm", "offset"), "identity", block),
        "left_projection.weight": _ref(_k(f"{scope}/projection", "weights"), left_split, block),
        "right_projection.weight": _ref(_k(f"{scope}/projection", "weights"), right_split, block),
        "left_gate_projection.weight": _ref(_k(f"{scope}/gate", "weights"), left_split, block),
        "right_gate_projection.weight": _ref(_k(f"{scope}/gate", "weights"), right_split, block),
        "output_projection.weight": _ref(_k(f"{scope}/output_projection", "weights"), "linear", block),
        "gate_projection.weight": _ref(_k(f"{scope}/gating_linear", "weights"), "linear", block),
    }
    return mapping.get(rest)


def _grid_attention_ref(rest: str, block: int, scope: str) -> _OfficialParamRef | None:
    mapping = {
        "norm_pair.weight": _ref(_k(f"{scope}/act_norm", "scale"), "identity", block),
        "norm_pair.bias": _ref(_k(f"{scope}/act_norm", "offset"), "identity", block),
        "bias_projection.weight": _ref(_k(f"{scope}/pair_bias_projection", "weights"), "linear", block),
        "attention.projection.query_projection.weight": _ref(
            _k(f"{scope}/q_projection", "weights"), "linear_transposed", block
        ),
        "attention.projection.key_projection.weight": _ref(
            _k(f"{scope}/k_projection", "weights"), "linear_transposed", block
        ),
        "attention.projection.value_projection.weight": _ref(_k(f"{scope}/v_projection", "weights"), "linear", block),
        "attention.projection.output_projection.weight": _ref(
            _k(f"{scope}/output_projection", "weights"), "linear", block
        ),
        "attention.projection.gate_projection.weight": _ref(
            _k(f"{scope}/gating_query", "weights"), "linear_transposed", block
        ),
    }
    return mapping.get(rest)


def _single_attention_ref(rest: str, block: int, scope: str) -> _OfficialParamRef | None:
    mapping = {
        "hidden_norm.weight": _ref(_k(f"{scope}/single_attention_layer_norm", "scale"), "identity", block),
        "hidden_norm.bias": _ref(_k(f"{scope}/single_attention_layer_norm", "offset"), "identity", block),
        "norm_pair.weight": _ref(_k(f"{scope}/single_pair_logits_norm", "scale"), "identity", block),
        "norm_pair.bias": _ref(_k(f"{scope}/single_pair_logits_norm", "offset"), "identity", block),
        "bias_projection.weight": _ref(_k(f"{scope}/single_pair_logits_projection", "weights"), "linear", block),
        "attention.projection.query_projection.weight": _ref(
            _k(f"{scope}/single_attention_q_projection", "weights"), "linear", block
        ),
        "attention.projection.query_projection.bias": _ref(
            _k(f"{scope}/single_attention_q_projection", "bias"), "bias", block
        ),
        "attention.projection.key_projection.weight": _ref(
            _k(f"{scope}/single_attention_k_projection", "weights"), "linear", block
        ),
        "attention.projection.value_projection.weight": _ref(
            _k(f"{scope}/single_attention_v_projection", "weights"), "linear", block
        ),
        "attention.projection.output_projection.weight": _ref(
            _k(f"{scope}/single_attention_transition2", "weights"), "linear", block
        ),
        "attention.projection.gate_projection.weight": _ref(
            _k(f"{scope}/single_attention_gating_query", "weights"), "linear", block
        ),
    }
    return mapping.get(rest)


def _transition_ref(rest: str, block: int | None, scope: str) -> _OfficialParamRef | None:
    indices = () if block is None else (block,)
    mapping = {
        "norm_input.weight": _ref(_k(f"{scope}/input_layer_norm", "scale"), "identity", *indices),
        "norm_input.bias": _ref(_k(f"{scope}/input_layer_norm", "offset"), "identity", *indices),
        "gate_projection.weight": _ref(_k(f"{scope}/transition1", "weights"), "glu_gate", *indices),
        "value_projection.weight": _ref(_k(f"{scope}/transition1", "weights"), "glu_value", *indices),
        "output_projection.weight": _ref(_k(f"{scope}/transition2", "weights"), "linear", *indices),
    }
    return mapping.get(rest)


def _flat_diffusion_transition_ref(rest: str, scope: str) -> _OfficialParamRef | None:
    mapping = {
        "norm_input.weight": _ref(_k(f"{scope}_layer_norm", "scale")),
        "norm_input.bias": _ref(_k(f"{scope}_layer_norm", "offset")),
        "gate_projection.weight": _ref(_k(f"{scope}_transition1", "weights"), "glu_gate"),
        "value_projection.weight": _ref(_k(f"{scope}_transition1", "weights"), "glu_value"),
        "output_projection.weight": _ref(_k(f"{scope}_transition2", "weights"), "linear"),
    }
    return mapping.get(rest)


def _conditioned_transformer_layer_ref(
    rest: str,
    scope: str,
    *,
    indices: tuple[int, ...],
    has_key_value_norm: bool,
    conditioning_prefix: str,
    key_conditioning_prefix: str = "ksingle_cond",
) -> _OfficialParamRef | None:
    if rest.startswith("conditioned_norm."):
        return _adaln_ref(rest.removeprefix("conditioned_norm."), scope, indices, conditioning_prefix)
    if has_key_value_norm and rest.startswith("key_value_norm."):
        return _adaln_ref(rest.removeprefix("key_value_norm."), scope, indices, key_conditioning_prefix)
    if rest.startswith("attention.projection."):
        return _conditioned_attention_ref(rest.removeprefix("attention.projection."), scope, indices)
    if rest.startswith("output_gate_projection."):
        suffix = rest.removeprefix("output_gate_projection.")
        if suffix == "weight":
            return _ref(_k(f"{scope}adaptive_zero_cond", "weights"), "linear", *indices)
        if suffix == "bias":
            return _ref(_k(f"{scope}adaptive_zero_cond", "bias"), "identity", *indices)
    if rest.startswith("transition."):
        return _conditioned_transition_ref(rest.removeprefix("transition."), scope, indices)
    return None


def _adaln_ref(rest: str, scope: str, indices: tuple[int, ...], prefix: str) -> _OfficialParamRef | None:
    if rest == "conditioning_norm.weight":
        return _ref(_k(f"{scope}{prefix}_layer_norm", "scale"), "identity", *indices)
    if rest == "gate_projection.weight":
        return _ref(_k(f"{scope}{prefix}_scale", "weights"), "linear", *indices)
    if rest == "gate_projection.bias":
        return _ref(_k(f"{scope}{prefix}_scale", "bias"), "identity", *indices)
    if rest == "shift_projection.weight":
        return _ref(_k(f"{scope}{prefix}_bias", "weights"), "linear", *indices)
    if rest == "conditioning_projection.weight":
        return _multi_ref(
            (
                _k(f"{scope}{prefix}_scale", "weights"),
                _k(f"{scope}{prefix}_bias", "weights"),
            ),
            "adaln_projection",
            *indices,
        )
    if rest == "conditioning_scale_bias":
        return _ref(_k(f"{scope}{prefix}_scale", "bias"), "identity", *indices)
    return None


def _conditioned_attention_ref(rest: str, scope: str, indices: tuple[int, ...]) -> _OfficialParamRef | None:
    mapping = {
        "query_projection.weight": _ref(_k(f"{scope}q_projection", "weights"), "linear", *indices),
        "query_projection.bias": _ref(_k(f"{scope}q_projection", "bias"), "bias", *indices),
        "key_projection.weight": _ref(_k(f"{scope}k_projection", "weights"), "linear", *indices),
        "value_projection.weight": _ref(_k(f"{scope}v_projection", "weights"), "linear", *indices),
        "gate_projection.weight": _ref(_k(f"{scope}gating_query", "weights"), "linear", *indices),
        "output_projection.weight": _ref(_k(f"{scope}transition2", "weights"), "linear", *indices),
    }
    return mapping.get(rest)


def _conditioned_transition_ref(rest: str, scope: str, indices: tuple[int, ...]) -> _OfficialParamRef | None:
    if rest.startswith("conditioned_norm."):
        return _adaln_ref(rest.removeprefix("conditioned_norm."), scope, indices, "ffw_single_cond")
    mapping = {
        "gate_projection.weight": _ref(_k(f"{scope}ffw_transition1", "weights"), "glu_gate", *indices),
        "value_projection.weight": _ref(_k(f"{scope}ffw_transition1", "weights"), "glu_value", *indices),
        "output_projection.weight": _ref(_k(f"{scope}ffw_transition2", "weights"), "linear", *indices),
        "conditioning_gate_projection.weight": _ref(
            _k(f"{scope}ffw_adaptive_zero_cond", "weights"), "linear", *indices
        ),
        "conditioning_gate_projection.bias": _ref(_k(f"{scope}ffw_adaptive_zero_cond", "bias"), "identity", *indices),
    }
    return mapping.get(rest)


def _apply_official_transform(ref: _OfficialParamRef, tensors: tuple[Tensor, ...], target: Tensor) -> Tensor:
    if ref.transform == "official_noise_fourier_weight":
        return torch.tensor(_OFFICIAL_NOISE_FOURIER_WEIGHT, dtype=torch.float32).reshape(target.shape)
    if ref.transform == "official_noise_fourier_bias":
        return torch.tensor(_OFFICIAL_NOISE_FOURIER_BIAS, dtype=torch.float32).reshape(target.shape)
    selected = tuple(_select_indices(tensor, ref.indices) for tensor in tensors)
    if ref.transform == "identity":
        return selected[0].clone()
    if ref.transform == "bias":
        return selected[0].reshape(-1).clone()
    if ref.transform == "linear":
        return _linear_weight(selected[0], target.shape)
    if ref.transform == "target_feat_linear":
        # DeepMind's target_feat is ordered [restype/profile/deletion tail (63), atom-encoder single
        # (384)]; the native token embedding is [atom-encoder single (384), tail (63)]. Rotate the input
        # columns of any projection that consumes target_feat so the converted weight matches the native
        # channel order (native cols [0:384] <- DeepMind cols [63:447], native cols [384:447] <- [0:63]).
        weight = _linear_weight(selected[0], None)
        tail = weight.shape[1] - 384
        return torch.cat([weight[:, tail:], weight[:, :tail]], dim=1).contiguous()
    if ref.transform == "single_prefixed_target_feat_linear":
        # As ``target_feat_linear`` but the projection input is concat([single (384), target_feat (447)]);
        # keep the leading ``single`` columns and rotate only the trailing ``target_feat`` columns from the
        # DeepMind [tail (63), atom (384)] order to the native [atom (384), tail (63)] token-embedding order.
        weight = _linear_weight(selected[0], None)
        single_width = weight.shape[1] - 447
        prefix = weight[:, :single_width]
        target_feat = weight[:, single_width:]
        tail = target_feat.shape[1] - 384
        rotated = torch.cat([target_feat[:, tail:], target_feat[:, :tail]], dim=1)
        return torch.cat([prefix, rotated], dim=1).contiguous()
    if ref.transform == "single_prefixed_target_feat_scale":
        # 1-D LayerNorm scale over concat([single (384), target_feat (447)]); rotate only the trailing
        # target_feat channels from DeepMind [tail (63), atom (384)] to native [atom (384), tail (63)].
        scale = selected[0].reshape(-1)
        single_width = scale.shape[0] - 447
        prefix = scale[:single_width]
        target_feat = scale[single_width:]
        tail = target_feat.shape[0] - 384
        return torch.cat([prefix, target_feat[tail:], target_feat[:tail]]).contiguous()
    if ref.transform == "linear_transposed":
        return _linear_transposed_weight(selected[0], target.shape)
    if ref.transform == "glu_gate":
        return _linear_weight(selected[0].chunk(2, dim=-1)[0], target.shape)
    if ref.transform == "glu_value":
        return _linear_weight(selected[0].chunk(2, dim=-1)[1], target.shape)
    if ref.transform == "split_left_linear":
        # DeepMind's TriangleMultiplication reshapes its 2*C-wide projection/gate output as
        # (C, 2) with the size-2 axis INNER (``reshape(2*C) -> (C, 2)``), so the left/right
        # triangle factors are the EVEN / ODD output columns, not two contiguous blocks.
        return _linear_weight(selected[0][..., 0::2].contiguous(), target.shape)
    if ref.transform == "split_right_linear":
        return _linear_weight(selected[0][..., 1::2].contiguous(), target.shape)
    if ref.transform == "opm_output":
        source = selected[0]
        return source.reshape(-1, source.shape[-1]).t().contiguous()
    if ref.transform == "template_projection":
        # Official features project residue j before residue i; the native packed layout stores i then j.
        ordered = (*selected[:2], selected[3], selected[2], *selected[4:])
        return torch.cat([_linear_weight(tensor, None) for tensor in ordered], dim=1)
    if ref.transform == "atom_feature_projection":
        return torch.cat([_linear_weight(tensor, None) for tensor in selected], dim=1)
    if ref.transform == "atom_confidence_weight":
        return selected[0].permute(1, 0, 2).contiguous()
    if ref.transform == "adaln_projection":
        return torch.cat([_linear_weight(tensor, None) for tensor in selected], dim=0)
    if ref.transform == "token_pair_logits":
        super_block, inner = ref.indices
        source = tensors[0][super_block, :, inner, :]
        return source.t().contiguous()
    if ref.transform == "atom_pair_logits":
        (layer,) = ref.indices
        source = tensors[0][:, layer, :]
        return source.t().contiguous()
    raise RuntimeError(f"unknown official AF3 tensor transform: {ref.transform}")


def _select_indices(tensor: Tensor, indices: tuple[int, ...]) -> Tensor:
    selected = tensor
    for index in indices:
        selected = selected[index]
    return selected


def _linear_weight(tensor: Tensor, target_shape: torch.Size | tuple[int, ...] | None) -> Tensor:
    if tensor.ndim == 1:
        result = tensor.reshape(-1, 1)
    else:
        result = tensor.reshape(tensor.shape[0], -1).t().contiguous()
    if target_shape is not None and tuple(result.shape) != tuple(target_shape) and tensor.ndim == 1:
        result = tensor.reshape(-1).contiguous()
    return result


def _linear_transposed_weight(tensor: Tensor, target_shape: torch.Size | tuple[int, ...] | None) -> Tensor:
    result = tensor.reshape(-1, tensor.shape[-1]).contiguous()
    if target_shape is not None and tuple(result.shape) != tuple(target_shape):
        result = _linear_weight(tensor, target_shape)
    return result


# Fixed architecture data published in DeepMind's Apache-2.0 AlphaFold 3 source:
# https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/network/noise_level_embeddings.py
# DeepMind froze the PRNG(42) result explicitly so it remains stable across JAX RNG changes. Keeping the same
# explicit values here makes conversion deterministic without importing or inspecting an external source tree.
# fmt: off
_OFFICIAL_NOISE_FOURIER_WEIGHT = (
    0.45873642, 0.06516238, -0.07278306, -0.26992258, 0.64292115,
    -0.40763968, 3.60116863, 0.54461384, -0.32644904, 2.10888267,
    1.30805349, 1.19838560, -1.37745857, 1.99475312, -1.64120293,
    1.07823789, -0.02288206, 0.88305283, 0.48099944, 0.17655374,
    0.30281949, 0.80646873, 0.62605333, -0.23965347, -1.02609432,
    0.75006109, -0.19913037, 0.07466396, 0.66431236, -0.60990530,
    -0.69709194, -0.44453633, -1.77656078, 0.02299878, 0.04095552,
    0.35485864, -0.47602659, -0.98820388, -0.24106771, -1.07254291,
    -0.99741757, 0.22697604, 1.41390419, 1.54984057, -0.12237291,
    0.20156337, 0.61767143, 0.23959029, 0.92454034, 1.84082258,
    0.89030224, 0.39598912, -1.52224910, 0.29669049, 1.52356744,
    -0.33968377, 0.24155144, -0.52308381, -0.23622665, 0.92825454,
    -0.63864607, -0.62169307, 0.78623551, -0.80352145, -0.45496067,
    1.30877995, -0.06686528, 1.00248849, -0.63593471, 0.16372502,
    -1.46133232, 1.10562658, -0.01693927, 0.28684548, -0.72843230,
    0.66133535, -1.92225552, 0.70241231, -0.96868867, -0.47309339,
    -1.66894221, 0.46018723, -0.56806105, 0.32694784, -0.46529883,
    1.02299964, 0.84688205, 1.19581807, -1.82454145, 0.05999713,
    -0.59530073, 1.44862521, -0.34933713, -0.46564487, -0.55005538,
    -1.61170268, 0.17502306, 0.38670063, -1.12133658, -0.29343036,
    -0.52527446, -1.26285112, 1.07982683, 0.51215219, 1.48963666,
    1.09847653, -0.01563358, 0.32574457, 1.94779706, -1.29198587,
    1.06249654, -0.86965990, 0.22975266, -0.27182648, -0.21130897,
    -0.41773933, -0.02329035, 1.31049252, 0.05579265, -1.23127055,
    -0.99691105, 0.27058721, -0.72509319, -0.14421797, -1.48605061,
    1.35041201, 1.29619241, -1.01022530, -0.79787987, -0.16166858,
    0.87210685, 1.69248152, 1.42469788, -0.72325104, -1.24823737,
    0.07051118, 0.71332991, -0.07360429, -0.91955227, -2.68856549,
    -0.44033936, 0.35482934, -0.57933813, 0.97468042, -0.31050494,
    -0.88454425, -2.08785224, 0.47322822, -0.02400172, 0.26644820,
    -0.19147627, -2.10538960, -1.27962470, -1.35999286, 2.09867334,
    0.65099514, 0.21604492, -0.45951018, 0.15994427, -0.31420693,
    -0.65202618, -0.61077976, -1.06100249, -1.47254968, 1.18165290,
    -0.78656220, 1.28182006, 1.80323684, 1.09196901, 0.26118696,
    -0.30168581, 0.39749333, 0.26812574, -1.51995814, -0.46909946,
    0.03874255, -1.36774313, 2.30143976, 2.06959820, -0.41647521,
    1.85624206, 0.49019700, -0.06726539, 0.00457313, 0.23915423,
    -1.84971249, -0.20482327, -0.34097880, -0.57933033, -1.10541213,
    -0.30269983, -0.16430426, -0.82371718, 0.10345812, 1.78753936,
    0.04786763, 1.86778629, -0.65214992, 0.81544143, -0.28214937,
    0.31187257, 0.57661986, 1.21938801, -1.56046617, 0.38046429,
    -0.18235965, 0.81794524, -0.40474343, 0.46538028, -1.15558851,
    0.59625793, -1.07801270, 0.07310858, 0.61526084, 0.55518496,
    -0.49787554, 0.92703879, -1.27780271, -0.83373469, -0.43015575,
    0.41877759, -1.03987372, -1.46055734, 0.61282396, 0.15590595,
    -0.34269521, 0.56509072, -1.17904210, 0.11374855, -1.83310866,
    0.38734794, -0.58623004, 0.77931106, 1.53930688, -0.70299625,
    -0.11389336, -1.14818096, -0.44400632, 1.21887410, 0.64066756,
    -0.70249403, -0.27244881, 0.38586098, -1.07925785, 0.12448707,
    -1.28286278, 0.37827531, 0.68812364, 1.65695465, 0.12440517,
    -0.03689830, 1.10224664, -0.28323629, -0.47939169, 0.70120829,
    -0.67204583,
)

_OFFICIAL_NOISE_FOURIER_BIAS = (
    0.00465965, 0.21738243, 0.22277749, 0.68463874, 0.84596848, 0.17337036,
    0.39573753, 0.78153563, 0.86311185, 0.21782327, 0.24377882, 0.42310703,
    0.19887352, 0.10486019, 0.48707581, 0.22205460, 0.97263455, 0.29714966,
    0.11244559, 0.53020525, 0.36796236, 0.37294638, 0.80261672, 0.04669094,
    0.86319661, 0.75907171, 0.77297020, 0.01114798, 0.55850804, 0.91799915,
    0.23032320, 0.12154722, 0.26701927, 0.42934716, 0.47951782, 0.96782577,
    0.86785042, 0.61985648, 0.05743814, 0.41800117, 0.68881893, 0.60575199,
    0.21058667, 0.64412105, 0.63958526, 0.89390790, 0.69755554, 0.89345169,
    0.53330755, 0.56985939, 0.30724049, 0.00984561, 0.91407037, 0.92118979,
    0.94153070, 0.81097460, 0.70537627, 0.32810748, 0.47227263, 0.11821401,
    0.44983089, 0.30767226, 0.31756389, 0.62969446, 0.69892538, 0.16949117,
    0.06207097, 0.46717727, 0.95348179, 0.62363589, 0.49018729, 0.06920040,
    0.39333904, 0.41299903, 0.52514863, 0.61197245, 0.56871891, 0.65053988,
    0.22203422, 0.46748531, 0.86931503, 0.87050021, 0.40208721, 0.32084906,
    0.55084610, 0.94584596, 0.76279902, 0.36250532, 0.74272907, 0.66682065,
    0.96452832, 0.64768302, 0.88070846, 0.56995463, 0.06395614, 0.69499350,
    0.44494808, 0.39775658, 0.20280898, 0.33363521, 0.05999005, 0.44414878,
    0.65227020, 0.01199079, 0.71995056, 0.19045687, 0.48342144, 0.25127733,
    0.66515994, 0.22465158, 0.22313106, 0.06302810, 0.55783665, 0.93625581,
    0.58800840, 0.72525370, 0.52879298, 0.77195418, 0.15548682, 0.01028740,
    0.39325142, 0.45401239, 0.71494079, 0.33011997, 0.05050695, 0.26381660,
    0.63064706, 0.47604024, 0.08593416, 0.00383425, 0.06352687, 0.05510247,
    0.03552997, 0.35810637, 0.56094289, 0.60922170, 0.88599777, 0.45419788,
    0.40486634, 0.71297824, 0.34976673, 0.97825217, 0.12915993, 0.09566259,
    0.64318919, 0.16717327, 0.82308614, 0.32672071, 0.81688786, 0.84857118,
    0.99922776, 0.07551706, 0.18766022, 0.13051236, 0.39136350, 0.08768725,
    0.92048228, 0.87185788, 0.39158428, 0.79224777, 0.17492688, 0.68902445,
    0.81980729, 0.70458186, 0.59489477, 0.93324888, 0.49986637, 0.40705478,
    0.89202917, 0.20673239, 0.39339757, 0.20996964, 0.02923799, 0.53992438,
    0.40119815, 0.10366607, 0.08044600, 0.95551598, 0.20518017, 0.68826210,
    0.90159297, 0.69008791, 0.86880815, 0.16246438, 0.89628279, 0.11481643,
    0.61353648, 0.41545081, 0.92478311, 0.78212476, 0.48292696, 0.79621077,
    0.11947489, 0.01747024, 0.22928023, 0.87387264, 0.86349785, 0.89526737,
    0.58904779, 0.13896775, 0.68194926, 0.55824125, 0.44428205, 0.55422378,
    0.28189969, 0.27923775, 0.09979951, 0.66994715, 0.45943546, 0.71207762,
    0.17300689, 0.83434916, 0.02573085, 0.45858085, 0.55934799, 0.30676675,
    0.52219367, 0.34544575, 0.19280875, 0.26937950, 0.07147646, 0.06295013,
    0.76382887, 0.38737607, 0.58825982, 0.17423475, 0.05509448, 0.97228825,
    0.94380617, 0.91664016, 0.18800116, 0.41771865, 0.59420645, 0.77371931,
    0.64687788, 0.27284670, 0.22310913, 0.15663862, 0.45573199, 0.50386798,
    0.66712272, 0.71649647, 0.28475654, 0.83415413, 0.75261366, 0.61517799,
    0.93544555, 0.76141870, 0.85474241, 0.74766934, 0.33459592, 0.78477907,
    0.07250881, 0.10174239, 0.95332730, 0.80793905,
)
# fmt: on
