from __future__ import annotations

import chanfig
import pytest
import torch
from torch import nn

from ofoldx.model import Model
from ofoldx.models.registry import MODELS
from ofoldx.processor import Processor


class RepositoryConfig(chanfig.Config):
    width: int = 3


class RepositoryProcessorConfig(chanfig.Config):
    scale: int = 1


class RepositoryModel(Model):
    config_class = RepositoryConfig

    def __init__(self, config: RepositoryConfig | None = None) -> None:
        config = RepositoryConfig() if config is None else config
        super().__init__(config)
        self.weight = nn.Parameter(torch.arange(config.width, dtype=torch.float32))
        self.bias = nn.Parameter(torch.arange(config.width, dtype=torch.float32) + config.width)


class RepositoryProcessor(Processor):
    config_class = RepositoryProcessorConfig
    config: RepositoryProcessorConfig

    def __init__(
        self,
        config: RepositoryProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = RepositoryProcessorConfig() if config is None else config


MODELS.register("test_repository", model=RepositoryModel, processor=RepositoryProcessor)


@pytest.fixture
def repository_model() -> RepositoryModel:
    return RepositoryModel()


@pytest.fixture
def repository_processor() -> RepositoryProcessor:
    return RepositoryProcessor(RepositoryProcessorConfig({"scale": 2}), dtype=torch.float16)
