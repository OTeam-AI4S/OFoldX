r"""Canonical overloaded AutoModel and AutoProcessor factories."""

from __future__ import annotations

from os import PathLike
from typing import cast

import chanfig
import torch

from .hub import (
    MODEL_METADATA,
    PROCESSOR_METADATA,
    load_model,
    load_processor,
    resolve_repository,
)
from .model import Model
from .models.registry import MODELS
from .processor import Processor


class AutoModel:
    r"""Construct a registered Model from a typed config or model repository."""

    def __new__(
        cls,
        pretrained_or_config: chanfig.Config | str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Model:
        if not isinstance(pretrained_or_config, str | PathLike):
            if revision is not None:
                raise ValueError("revision is valid only when AutoModel loads a model repository")
            registration = MODELS.for_config(pretrained_or_config)
            if token is not None:
                raise ValueError("token is valid only when AutoModel loads a model repository")
            model = registration.model(pretrained_or_config)
            if device is not None or dtype is not None:
                model.to(device=device, dtype=dtype)
            return model
        repository = resolve_repository(
            pretrained_or_config,
            revision=revision,
            allow_patterns=MODEL_METADATA,
            token=token,
        )
        return load_model(repository, device=device, dtype=dtype, token=token)

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Model:
        r"""Thin familiar spelling of ``AutoModel(pretrained, ...)``."""

        return cls(pretrained, revision=revision, token=token, device=device, dtype=dtype)


class AutoProcessor:
    r"""Load a registered Processor from a model repository."""

    def __new__(  # type: ignore[misc]
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Processor:
        repository = resolve_repository(
            pretrained,
            revision=revision,
            token=token,
            allow_patterns=PROCESSOR_METADATA,
        )
        return load_processor(repository, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Processor:
        r"""Thin familiar spelling of ``AutoProcessor(pretrained, ...)``."""

        return cast(Processor, cls(pretrained, revision=revision, token=token, device=device, dtype=dtype))
