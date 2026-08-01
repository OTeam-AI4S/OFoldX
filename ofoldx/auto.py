r"""
Automatic Model and Processor factories.
"""

from __future__ import annotations

from os import PathLike

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
from .modules.backend import Backend
from .processor import Processor


class AutoModel:
    r"""
    Construct a registered Model from a config or repository.
    """

    def __init__(self) -> None:
        raise TypeError("use AutoModel.from_config() or AutoModel.from_pretrained()")

    @classmethod
    def from_config(
        cls,
        config: chanfig.Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Model:
        r"""
        Construct the Model registered for an exact config class.
        """

        model_class = MODELS.model_for_config(config)
        model = model_class(config)
        if device is not None or dtype is not None:
            model.to(device=device, dtype=dtype)
        if backend is not None:
            model.set_backend(backend)
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Model:
        r"""
        Load the Model declared by a local or Hugging Face repository.
        """

        repository = resolve_repository(
            pretrained,
            revision=revision,
            allow_patterns=MODEL_METADATA,
            token=token,
        )
        model = load_model(repository, device=device, dtype=dtype, token=token)
        if backend is not None:
            model.set_backend(backend)
        return model


class AutoProcessor:
    r"""
    Load a registered Processor from a repository.
    """

    def __init__(self) -> None:
        raise TypeError("use AutoProcessor.from_pretrained()")

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
        r"""
        Load the Processor declared by a local or Hugging Face repository.
        """

        repository = resolve_repository(
            pretrained,
            revision=revision,
            allow_patterns=PROCESSOR_METADATA,
            token=token,
        )
        return load_processor(repository, device=device, dtype=dtype)
