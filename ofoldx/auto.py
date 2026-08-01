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
Automatic Model and Processor factories.
"""

from __future__ import annotations

from os import PathLike
from typing import ClassVar, cast

import chanfig
import torch

from .hub import (
    MODEL_METADATA,
    PROCESSOR_METADATA,
    ResolvedRepository,
    load_model,
    load_processor,
    resolve_repository,
)
from .model import Model
from .models.registry import MODELS
from .modules.backend import Backend
from .pipelines import Pipeline, PipelineTask
from .pipelines.loading import load_pipeline
from .processor import Processor


class AutoModel:
    r"""
    Construct a registered Model from a config or repository.
    """

    def __new__(  # type: ignore[misc]
        cls,
        source: chanfig.Config | str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Model:
        r"""Construct from a configuration or load from a local or Hub repository."""
        if isinstance(source, chanfig.Config):
            if revision is not None or token is not None:
                raise ValueError("revision and token apply only to repositories")
            return cls.from_config(source, device=device, dtype=dtype, backend=backend)
        repository = resolve_repository(source, revision=revision, allow_patterns=MODEL_METADATA, token=token)
        model = load_model(repository, device=device, dtype=dtype, token=token)
        if backend is not None:
            model.set_backend(backend)
        return model

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
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
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

        if isinstance(pretrained_model_name_or_path, chanfig.Config):
            raise TypeError("from_pretrained requires a repository, not a model configuration")
        return cls(
            pretrained_model_name_or_path, revision=revision, token=token, device=device, dtype=dtype, backend=backend
        )


class AutoProcessor:
    r"""
    Load a registered Processor from a repository.
    """

    def __new__(  # type: ignore[misc]
        cls,
        repository: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Processor:
        r"""Load the registered Processor from a local or Hub repository."""
        resolved = resolve_repository(repository, revision=revision, allow_patterns=PROCESSOR_METADATA, token=token)
        return load_processor(resolved, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Processor:
        r"""
        Load the Processor declared by a local or Hugging Face repository.
        """

        return cast(
            Processor, cls(pretrained_model_name_or_path, revision=revision, token=token, device=device, dtype=dtype)
        )


class _AutoPipeline:
    task: ClassVar[PipelineTask]

    def __new__(  # type: ignore[misc]
        cls,
        repository: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Pipeline:
        r"""Load the registered task Pipeline from a local or Hub repository."""
        return load_pipeline(
            repository,
            task=cls.task,
            revision=revision,
            token=token,
            device=device,
            dtype=dtype,
            backend=backend,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | ResolvedRepository,
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        backend: Backend | None = None,
    ) -> Pipeline:
        r"""Load the concrete workflow declared for this biological task."""

        return cast(
            Pipeline,
            cls(
                pretrained_model_name_or_path,
                revision=revision,
                token=token,
                device=device,
                dtype=dtype,
                backend=backend,
            ),
        )


class AutoPipelineForFold(_AutoPipeline):
    r"""Load a structure-prediction Pipeline."""

    task = "fold"


class AutoPipelineForDesign(_AutoPipeline):
    r"""Load a molecular-design Pipeline."""

    task = "design"


class AutoPipelineForScore(_AutoPipeline):
    r"""Load a scientific-scoring Pipeline."""

    task = "score"


class AutoPipelineForPack(_AutoPipeline):
    r"""Load a side-chain-packing Pipeline."""

    task = "pack"
