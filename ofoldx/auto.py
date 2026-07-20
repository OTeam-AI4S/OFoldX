r"""Canonical overloaded AutoModel and AutoProcessor factories."""

from __future__ import annotations

from os import PathLike
from typing import Any

import torch

from .hub import (
    _MODEL_METADATA,
    _PROCESSOR_METADATA,
    _load_model,
    _load_processor,
    _resolve_repository,
)
from .models.registry import _registration_for_model_config


class AutoModel:
    r"""Construct a registered Model from a typed config or model repository."""

    def __new__(
        cls,
        pretrained_or_config: object,
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any:
        if not isinstance(pretrained_or_config, str | PathLike):
            if revision is not None:
                raise ValueError("revision is valid only when AutoModel loads a model repository")
            registration = _registration_for_model_config(pretrained_or_config)
            model = registration.model(pretrained_or_config)
            if device is not None or dtype is not None:
                model.to(device=device, dtype=dtype)
            return model
        repository = _resolve_repository(
            pretrained_or_config,
            revision=revision,
            allow_patterns=_MODEL_METADATA,
        )
        return _load_model(repository, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any:
        r"""Thin familiar spelling of ``AutoModel(pretrained, ...)``."""

        if not isinstance(pretrained, str | PathLike):
            raise TypeError("AutoModel.from_pretrained requires a model repository reference")
        return cls(pretrained, revision=revision, device=device, dtype=dtype)


class AutoProcessor:
    r"""Load a registered Processor from a model repository."""

    def __new__(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any:
        if not isinstance(pretrained, str | PathLike):
            raise TypeError("AutoProcessor requires a model repository reference")
        repository = _resolve_repository(
            pretrained,
            revision=revision,
            allow_patterns=_PROCESSOR_METADATA,
        )
        return _load_processor(repository, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Any:
        r"""Thin familiar spelling of ``AutoProcessor(pretrained, ...)``."""

        if not isinstance(pretrained, str | PathLike):
            raise TypeError("AutoProcessor.from_pretrained requires a model repository reference")
        return cls(pretrained, revision=revision, device=device, dtype=dtype)
