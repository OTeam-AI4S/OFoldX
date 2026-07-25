r"""
Automatic model and processor factories.
"""

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
    r"""
    Construct a registered model or load one from a model repository.
    """

    def __new__(
        cls,
        pretrained_or_config: chanfig.Config | str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Model:
        r"""
        Construct a model from a config or repository.

        Args:
            pretrained_or_config: A registered architecture config, local model
                repository, or Hugging Face Hub repository ID.
            revision: Hub revision to load. Valid only for a Hub repository.
            token: Hugging Face access token used for private repositories.
            device: Device onto which the model is loaded.
            dtype: Floating-point dtype applied after loading.

        Returns:
            The model registered for the supplied config or repository metadata.
        """

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
        r"""
        Load a registered model from a local or Hugging Face Hub repository.

        Args:
            pretrained: Local model repository or Hugging Face Hub repository ID.
            revision: Hub revision to load.
            token: Hugging Face access token used for private repositories.
            device: Device onto which the model is loaded.
            dtype: Floating-point dtype applied after loading.

        Returns:
            The model declared by the repository metadata.
        """

        if not isinstance(pretrained, str | PathLike):
            raise TypeError("pretrained must be a repository reference")
        return cls(pretrained, revision=revision, token=token, device=device, dtype=dtype)


class AutoProcessor:
    r"""
    Load a registered processor from a model repository.
    """

    def __new__(  # type: ignore[misc]
        cls,
        pretrained: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Processor:
        r"""
        Load a processor from a local or Hugging Face Hub repository.

        Args:
            pretrained: Local model repository or Hugging Face Hub repository ID.
            revision: Hub revision to load.
            token: Hugging Face access token used for private repositories.
            device: Device used for tensors produced by the processor.
            dtype: Floating-point dtype used for tensors produced by the processor.

        Returns:
            The processor declared by the repository metadata.
        """

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
        r"""
        Load a registered processor from a local or Hugging Face Hub repository.

        Args:
            pretrained: Local model repository or Hugging Face Hub repository ID.
            revision: Hub revision to load.
            token: Hugging Face access token used for private repositories.
            device: Device used for tensors produced by the processor.
            dtype: Floating-point dtype used for tensors produced by the processor.

        Returns:
            The processor declared by the repository metadata.
        """

        if not isinstance(pretrained, str | PathLike):
            raise TypeError("pretrained must be a repository reference")
        return cast(Processor, cls(pretrained, revision=revision, token=token, device=device, dtype=dtype))
