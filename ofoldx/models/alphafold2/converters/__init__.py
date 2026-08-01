r"""
AlphaFold 2 source-specific checkpoint converters.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ....data.evidence import MSA_ALPHABET

_SOURCE_MSA_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYVX-")


def _convert_source_msa_axis(tensor: Tensor, axis: int, *, include_mask: bool = False) -> Tensor:
    axis %= tensor.ndim
    source = tensor.movedim(axis, 0)
    expected = len(_SOURCE_MSA_ALPHABET) + int(include_mask)
    if source.shape[0] != expected:
        raise RuntimeError(f"AlphaFold 2 source MSA axis has {source.shape[0]} entries; expected {expected}")

    source_by_symbol = {symbol: source[index] for index, symbol in enumerate(_SOURCE_MSA_ALPHABET)}
    converted: list[Tensor] = []
    for symbol in MSA_ALPHABET:
        if symbol in source_by_symbol:
            value = source_by_symbol[symbol]
        elif symbol == "B":
            value = (source_by_symbol["D"] + source_by_symbol["N"]) / 2
        elif symbol == "Z":
            value = (source_by_symbol["E"] + source_by_symbol["Q"]) / 2
        elif symbol == "J":
            value = (source_by_symbol["I"] + source_by_symbol["L"]) / 2
        elif symbol == "U":
            value = source_by_symbol["C"]
        elif symbol == "O":
            value = source_by_symbol["K"]
        else:
            value = torch.zeros_like(source[0])
        converted.append(value)
    if include_mask:
        converted.append(source[-1])
    return torch.stack(converted).movedim(0, axis).contiguous()


def _convert_alphafold2_msa_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    masked_msa_has_mask: bool,
) -> dict[str, Tensor]:
    r"""Expand every released AF2 MSA parameter axis to the canonical vocabulary."""

    converted = dict(state_dict)
    input_key = "input_embeddings.msa_projection.weight"
    extra_key = "extra_msa_projection.weight"
    head_weight_key = "masked_msa_head.weight"
    head_bias_key = "masked_msa_head.bias"
    missing = {key for key in (input_key, extra_key, head_weight_key, head_bias_key) if key not in converted}
    if missing:
        raise RuntimeError(f"AlphaFold 2 source state is missing MSA tensors: {sorted(missing)!r}")

    input_weight = converted[input_key]
    if input_weight.ndim != 2 or input_weight.shape[1] != 49:
        raise RuntimeError(f"AlphaFold 2 source clustered-MSA projection has shape {tuple(input_weight.shape)}")
    converted[input_key] = torch.cat(
        (
            _convert_source_msa_axis(input_weight[:, :23], 1, include_mask=True),
            input_weight[:, 23:25],
            _convert_source_msa_axis(input_weight[:, 25:48], 1, include_mask=True),
            input_weight[:, 48:],
        ),
        dim=1,
    ).contiguous()

    extra_weight = converted[extra_key]
    if extra_weight.ndim != 2 or extra_weight.shape[1] != 25:
        raise RuntimeError(f"AlphaFold 2 source extra-MSA projection has shape {tuple(extra_weight.shape)}")
    converted[extra_key] = torch.cat(
        (
            _convert_source_msa_axis(extra_weight[:, :23], 1, include_mask=True),
            extra_weight[:, 23:],
        ),
        dim=1,
    ).contiguous()

    converted[head_weight_key] = _convert_source_msa_axis(
        converted[head_weight_key], 0, include_mask=masked_msa_has_mask
    )
    converted[head_bias_key] = _convert_source_msa_axis(converted[head_bias_key], 0, include_mask=masked_msa_has_mask)
    return converted


# These imports must follow the helpers above, which the converter modules import.
from . import deepmind, deepmind_multimer, openfold  # noqa: E402

__all__ = ["deepmind", "deepmind_multimer", "openfold"]
