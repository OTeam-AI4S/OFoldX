r"""
Official DeepMind AlphaFold 3 parameter reader.

DeepMind distributes AlphaFold 3 parameters separately from the Apache-2.0
source code. This module supports local inspection of the official record-stream
parameter files, but does not bundle, download, or publish those weights.
"""

from __future__ import annotations

import contextlib
import io
import re
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

OFFICIAL_AF3_REPOSITORY_URL = "https://github.com/google-deepmind/alphafold3"
OFFICIAL_AF3_WEIGHTS_TERMS_URL = "https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md"
OFFICIAL_AF3_CODE_LICENSE = "Apache-2.0"
OFFICIAL_AF3_PARAMS_TERMS = "AlphaFold 3 Model Parameters Terms of Use"

_RECORD_HEADER = struct.Struct("<5i")
_MODEL_FILE_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (re.compile(r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$"), True),
    (re.compile(r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$"), True),
    (re.compile(r"(?P<model_name>.*)\.[0-9]+\.bin$"), False),
    (re.compile(r"(?P<model_name>.*)\.bin\.[0-9]+$"), False),
    # Keep the official typo-shaped pattern harmlessly supported for users with
    # files produced by scripts that copied it literally.
    (re.compile(r"(?P<model_name>.*)\.bin]\.[0-9]+$"), False),
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


class RecordError(ValueError):
    r"""
    Raised when an official AF3 parameter record is malformed or truncated.
    """


@dataclass(frozen=True, slots=True)
class OfficialAlphaFold3Params:
    r"""
    Loaded official AlphaFold 3 parameter tensors.
    """

    tensors: dict[str, Tensor]
    model_name: str
    model_files: tuple[Path, ...]
    compressed: bool


def inspect_official_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    r"""
    Return a lightweight summary of official DeepMind AlphaFold 3 parameters.
    """

    loaded = _load_official_alphafold3_params(checkpoint_path, model_name=model_name)
    scopes = Counter(key.split("//", 1)[0] for key in loaded.tensors)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_source": "deepmind",
        "checkpoint_format": "official_alphafold3_records",
        "model_name": loaded.model_name,
        "model_files": [str(path) for path in loaded.model_files],
        "compressed": loaded.compressed,
        "num_tensors": len(loaded.tensors),
        "num_parameters": sum(tensor.numel() for tensor in loaded.tensors.values()),
        "top_scopes": dict(scopes.most_common(20)),
        "dtypes": _dtype_counts(loaded.tensors),
        "source_repository": OFFICIAL_AF3_REPOSITORY_URL,
        "official_code_license": OFFICIAL_AF3_CODE_LICENSE,
        "official_params_terms": OFFICIAL_AF3_PARAMS_TERMS,
        "official_params_terms_url": OFFICIAL_AF3_WEIGHTS_TERMS_URL,
    }


def load_official_alphafold3_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None = None,
) -> dict[str, Tensor]:
    r"""
    Load official DeepMind AlphaFold 3 parameters as ``"scope//name"`` tensors.
    """

    return _load_official_alphafold3_params(params_path, model_name=model_name).tensors


def convert_official_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    r"""
    Convert official DeepMind AlphaFold 3 parameters into an OFoldX artifact.
    """

    from ...mixin import no_init_weights
    from ..config import AlphaFold3Config
    from ..model import AlphaFold3Model

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_dir)
    loaded = _load_official_alphafold3_params(checkpoint_path, model_name=model_name)
    with no_init_weights():
        model = AlphaFold3Model(AlphaFold3Config())
    state_dict = translate_official_alphafold3_state(model, loaded.tensors, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
    model.save_pretrained(output_path)
    persisted_state = dict(model.state_dict())
    return {
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_path),
        "architecture": model.config.architecture,
        "artifact_kind": "model",
        "checkpoint_source": "deepmind",
        "checkpoint_format": "official_alphafold3_records",
        "model_name": loaded.model_name,
        "model_files": [str(path) for path in loaded.model_files],
        "official_code_license": OFFICIAL_AF3_CODE_LICENSE,
        "official_params_terms": OFFICIAL_AF3_PARAMS_TERMS,
        "official_params_terms_url": OFFICIAL_AF3_WEIGHTS_TERMS_URL,
        "runnable_model": True,
        "tensor_count": len(persisted_state),
        "dtype": str(dtype) if dtype is not None else None,
        "dtype_counts": _dtype_counts(persisted_state),
    }


def translate_official_alphafold3_state(
    model: torch.nn.Module,
    official_state: Mapping[str, Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    r"""
    Translate official DeepMind AlphaFold 3 Haiku parameters into OFoldX keys.
    """

    translated = dict(model.state_dict())
    required = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing: list[tuple[str, str]] = []
    mismatched: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []

    for model_key, model_tensor in model.state_dict().items():
        ref = _official_ref_for_model_key(model_key)
        if ref is None:
            if model_key in required:
                missing.append((model_key, "<no official mapping>"))
            continue
        if all(key.startswith("<") for key in ref.keys):
            source_tensors = ()
        else:
            tensors = tuple(official_state.get(key) for key in ref.keys)
            if any(tensor is None for tensor in tensors):
                missing.append((model_key, " + ".join(ref.keys)))
                continue
            source_tensors = tuple(tensor for tensor in tensors if tensor is not None)
        converted = _apply_official_transform(ref, source_tensors, model_tensor)
        if ref.negate:
            converted = -converted
        if tuple(converted.shape) != tuple(model_tensor.shape):
            mismatched.append((model_key, " + ".join(ref.keys), tuple(model_tensor.shape), tuple(converted.shape)))
            continue
        if dtype is not None and converted.is_floating_point():
            converted = converted.to(dtype=dtype)
        translated[model_key] = converted

    if missing or mismatched:
        lines = [
            "official AlphaFold 3 checkpoint translation failed",
            f"missing: {len(missing)}",
            f"shape_mismatch: {len(mismatched)}",
        ]
        lines.extend(f"  missing {model_key} <- {source}" for model_key, source in missing[:40])
        lines.extend(
            f"  mismatched {model_key} <- {source}: model{model_shape} official{official_shape}"
            for model_key, source, model_shape, official_shape in mismatched[:40]
        )
        raise RuntimeError("\n".join(lines))
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


def _load_official_alphafold3_params(
    params_path: str | PathLike[str],
    *,
    model_name: str | None,
) -> OfficialAlphaFold3Params:
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
    return OfficialAlphaFold3Params(
        tensors=tensors,
        model_name=resolved_model_name,
        model_files=model_files,
        compressed=compressed,
    )


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
            try:
                import zstandard
            except ImportError as error:  # pragma: no cover - depends on optional local install state.
                raise ImportError(
                    "reading official AlphaFold 3 .zst parameter files requires the 'zstandard' package"
                ) from error
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
        raise RecordError(f"incomplete official AF3 record header: {len(header)} < {_RECORD_HEADER.size}")
    scope_len, name_len, dtype_len, shape_len, array_bytes = _RECORD_HEADER.unpack(header)
    payload_format = f"<{scope_len}s{name_len}s{dtype_len}s{shape_len}i"
    metadata_bytes = struct.calcsize(payload_format)
    payload = stream.read(metadata_bytes + array_bytes)
    if len(payload) < metadata_bytes + array_bytes:
        raise RecordError(f"incomplete official AF3 record payload: {len(payload)} < {metadata_bytes + array_bytes}")
    scope_raw, name_raw, dtype_raw, *shape = struct.unpack_from(payload_format, payload)
    dtype = dtype_raw.decode("utf-8")
    torch_dtype = _TORCH_DTYPES.get(dtype)
    if torch_dtype is None:
        raise RecordError(f"unsupported official AF3 tensor dtype: {dtype!r}")
    expected_bytes = _numel(shape) * _DTYPE_SIZES[dtype]
    if expected_bytes != array_bytes:
        raise RecordError(
            f"official AF3 tensor byte count mismatch for {scope_raw!r}//{name_raw!r}: "
            f"header has {array_bytes}, shape/dtype implies {expected_bytes}"
        )
    raw = payload[-array_bytes:]
    if sys.byteorder == "big" and _DTYPE_SIZES[dtype] > 1:
        element_size = _DTYPE_SIZES[dtype]
        raw = b"".join(raw[index : index + element_size][::-1] for index in range(0, len(raw), element_size))
    tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype).clone().reshape(shape)
    return scope_raw.decode("utf-8"), name_raw.decode("utf-8"), tensor


def _numel(shape: Sequence[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


@dataclass(frozen=True, slots=True)
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


def _official_ref_for_model_key(model_key: str) -> _OfficialParamRef | None:
    top_level = _top_level_ref(model_key)
    if top_level is not None:
        return top_level
    if model_key.startswith("input_embedder.atom_encoder."):
        return _atom_encoder_ref(
            model_key.removeprefix("input_embedder.atom_encoder."),
            base="diffuser/evoformer_conditioning",
            stack="diffuser/evoformer_conditioning_atom_transformer_encoder/"
            "__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder",
            include_trunk=False,
        )
    if match := re.match(r"pairformer_stack\.blocks\.(\d+)\.(.+)", model_key):
        return _pairformer_block_ref(
            match.group(2),
            int(match.group(1)),
            "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer",
            with_single=True,
            pair_transition_target="pair_feed_forward",
        )
    if model_key.startswith("msa_trunk."):
        return _msa_ref(model_key.removeprefix("msa_trunk."))
    if model_key.startswith("template_embedder."):
        return _template_ref(model_key.removeprefix("template_embedder."))
    if model_key.startswith("confidence_head."):
        return _confidence_ref(model_key.removeprefix("confidence_head."))
    if model_key.startswith("coordinate_denoiser."):
        return _denoiser_ref(model_key.removeprefix("coordinate_denoiser."))
    return None


def _top_level_ref(model_key: str) -> _OfficialParamRef | None:
    mapping = {
        "relative_position_embedder.output_projection.weight": _ref(
            _k("diffuser/evoformer/~_relative_encoding/position_activations", "weights"), "linear"
        ),
        "single_init_projection.weight": _ref(_k("diffuser/evoformer/single_activations", "weights"), "linear"),
        "pair_left_projection.weight": _ref(_k("diffuser/evoformer/left_single", "weights"), "linear"),
        "pair_right_projection.weight": _ref(_k("diffuser/evoformer/right_single", "weights"), "linear"),
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


def _msa_ref(rest: str) -> _OfficialParamRef | None:
    if rest == "msa_projection.weight":
        return _ref(_k("diffuser/evoformer/msa_activations", "weights"), "linear")
    if rest == "token_projection.weight":
        return _ref(_k("diffuser/evoformer/extra_msa_target_feat", "weights"), "linear")
    match = re.match(r"blocks\.(\d+)\.(.+)", rest)
    if match is None:
        return None
    block = int(match.group(1))
    inner = match.group(2)
    scope = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"
    if inner.startswith("outer_product_mean."):
        return _outer_product_mean_ref(inner.removeprefix("outer_product_mean."), block, scope)
    if inner.startswith("msa_stack.msa_pair_weighted_averaging."):
        return _msa_attention_ref(inner.removeprefix("msa_stack.msa_pair_weighted_averaging."), block, scope)
    if inner.startswith("msa_stack.feed_forward."):
        return _transition_ref(inner.removeprefix("msa_stack.feed_forward."), block, f"{scope}/msa_transition")
    if inner.startswith("pair_stack."):
        return _pairformer_block_ref(
            inner.removeprefix("pair_stack."),
            block,
            scope,
            with_single=False,
            pair_transition_target="feed_forward",
        )
    return None


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
    if match := re.match(r"blocks\.(\d+)\.(.+)", rest):
        return _pairformer_block_ref(
            match.group(2),
            int(match.group(1)),
            f"{single}/__layer_stack_no_per_layer/template_embedding_iteration",
            with_single=False,
            pair_transition_target="feed_forward",
        )
    return None


def _confidence_ref(rest: str) -> _OfficialParamRef | None:
    base = "diffuser/confidence_head"
    if rest == "pair_embedder.left_token_projection.weight":
        return _ref(_k(f"{base}/~_embed_features/right_target_feat_project", "weights"), "linear")
    if rest == "pair_embedder.right_token_projection.weight":
        return _ref(_k(f"{base}/~_embed_features/left_target_feat_project", "weights"), "linear")
    if rest == "pair_embedder.distance_one_hot_projection.weight":
        return _ref(_k(f"{base}/~_embed_features/distogram_feat_project", "weights"), "linear")
    if match := re.match(r"pairformer_stack\.blocks\.(\d+)\.(.+)", rest):
        return _pairformer_block_ref(
            match.group(2),
            int(match.group(1)),
            f"{base}/__layer_stack_no_per_layer/confidence_pairformer",
            with_single=True,
            pair_transition_target="pair_feed_forward",
        )
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


def _denoiser_ref(rest: str) -> _OfficialParamRef | None:
    base = "diffuser/~/diffusion_head"
    if rest.startswith("conditioner."):
        return _denoiser_conditioner_ref(rest.removeprefix("conditioner."), base)
    if rest.startswith("atom_encoder."):
        return _atom_encoder_ref(
            rest.removeprefix("atom_encoder."),
            base=f"{base}/diffusion",
            stack=f"{base}/diffusion_atom_transformer_encoder/"
            "__layer_stack_with_per_layer/diffusion_atom_transformer_encoder",
            include_trunk=True,
        )
    if rest.startswith("token_refiner."):
        return _token_refiner_ref(rest.removeprefix("token_refiner."), base)
    if rest.startswith("atom_decoder."):
        return _atom_decoder_ref(rest.removeprefix("atom_decoder."), base)
    return None


def _denoiser_conditioner_ref(rest: str, base: str) -> _OfficialParamRef | None:
    mapping = {
        "pair_input_norm.weight": _ref(_k(f"{base}/pair_cond_initial_norm", "scale")),
        "pair_projection.weight": _ref(_k(f"{base}/pair_cond_initial_projection", "weights"), "linear"),
        "single_conditioning_norm.weight": _ref(_k(f"{base}/single_cond_initial_norm", "scale")),
        "single_conditioning_projection.weight": _ref(
            _k(f"{base}/single_cond_initial_projection", "weights"), "linear"
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
            _k(f"{base}_embed_pair_offsets_1", "weights"), "linear", negate=True
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
    mapping = {
        "norm_input.weight": _ref(_k(f"{scope}/left_norm_input", "scale"), "identity", block),
        "norm_input.bias": _ref(_k(f"{scope}/left_norm_input", "offset"), "identity", block),
        "norm_output.weight": _ref(_k(f"{scope}/center_norm", "scale"), "identity", block),
        "norm_output.bias": _ref(_k(f"{scope}/center_norm", "offset"), "identity", block),
        "left_projection.weight": _ref(_k(f"{scope}/projection", "weights"), "split_left_linear", block),
        "right_projection.weight": _ref(_k(f"{scope}/projection", "weights"), "split_right_linear", block),
        "left_gate_projection.weight": _ref(_k(f"{scope}/gate", "weights"), "split_left_linear", block),
        "right_gate_projection.weight": _ref(_k(f"{scope}/gate", "weights"), "split_right_linear", block),
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
        weight, _ = _official_noise_fourier_constants()
        return weight.reshape(target.shape)
    if ref.transform == "official_noise_fourier_bias":
        _, bias = _official_noise_fourier_constants()
        return bias.reshape(target.shape)
    selected = tuple(_select_indices(tensor, ref.indices) for tensor in tensors)
    if ref.transform == "identity":
        return selected[0].clone()
    if ref.transform == "bias":
        return selected[0].reshape(-1).clone()
    if ref.transform == "linear":
        return _linear_weight(selected[0], target.shape)
    if ref.transform == "linear_transposed":
        return _linear_transposed_weight(selected[0], target.shape)
    if ref.transform == "glu_gate":
        return _linear_weight(selected[0].chunk(2, dim=-1)[0], target.shape)
    if ref.transform == "glu_value":
        return _linear_weight(selected[0].chunk(2, dim=-1)[1], target.shape)
    if ref.transform == "split_left_linear":
        return _linear_weight(selected[0].chunk(2, dim=-1)[0], target.shape)
    if ref.transform == "split_right_linear":
        return _linear_weight(selected[0].chunk(2, dim=-1)[1], target.shape)
    if ref.transform == "opm_output":
        source = selected[0]
        return source.reshape(-1, source.shape[-1]).t().contiguous()
    if ref.transform == "template_projection":
        return torch.cat([_linear_weight(tensor, None) for tensor in selected], dim=1)
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


_NOISE_FOURIER_CACHE: tuple[Tensor, Tensor] | None = None


def _official_noise_fourier_constants() -> tuple[Tensor, Tensor]:
    global _NOISE_FOURIER_CACHE
    if _NOISE_FOURIER_CACHE is not None:
        return _NOISE_FOURIER_CACHE
    source = _find_official_noise_embedding_source()
    if source is None:
        raise FileNotFoundError(
            "official AF3 Fourier noise constants were not found; set ALPHAFOLD3_SOURCE_DIR "
            "to the official alphafold3 source tree before converting"
        )
    import ast

    module = ast.parse(source.read_text())
    values: dict[str, list[float]] = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"_WEIGHT", "_BIAS"}:
                    values[target.id] = ast.literal_eval(node.value)
    if "_WEIGHT" not in values or "_BIAS" not in values:
        raise ValueError(f"could not parse official AF3 noise constants from {source}")
    _NOISE_FOURIER_CACHE = (
        torch.tensor(values["_WEIGHT"], dtype=torch.float32).reshape(256, 1),
        torch.tensor(values["_BIAS"], dtype=torch.float32),
    )
    return _NOISE_FOURIER_CACHE


def _find_official_noise_embedding_source() -> Path | None:
    candidates: list[Path] = []
    import os

    if source_dir := os.environ.get("ALPHAFOLD3_SOURCE_DIR"):
        candidates.append(Path(source_dir) / "alphafold3/model/network/noise_level_embeddings.py")
        candidates.append(Path(source_dir) / "src/alphafold3/model/network/noise_level_embeddings.py")
    candidates.append(
        Path("/data/node4/home/zhiyuan/reference/alphafold3/src/alphafold3/model/network/noise_level_embeddings.py")
    )
    try:
        import importlib.util

        spec = importlib.util.find_spec("alphafold3.model.network.noise_level_embeddings")
        if spec is not None and spec.origin is not None:
            candidates.append(Path(spec.origin))
    except (ImportError, ValueError):
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tensor in state_dict.values():
        name = str(tensor.dtype).removeprefix("torch.")
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))
