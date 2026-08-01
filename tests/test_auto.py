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

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ofoldx import AutoModel, AutoProcessor
from ofoldx.artifacts import save_model
from ofoldx.hub import resolve_repository
from ofoldx.models.registry import MODELS


class TestAutoModel:
    def test_constructs_registered_builtin_from_config(self) -> None:
        from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnModel

        model = AutoModel(ProteinMpnnConfig())
        assert isinstance(model, ProteinMpnnModel)
        assert model.training

    def test_custom_model_reuses_config_without_changing_default(
        self, tmp_path: Path, repository_model, repository_processor
    ) -> None:
        default_model = type(repository_model)

        class CustomRepositoryModel(default_model):
            pass

        MODELS.register(
            "test_custom_repository",
            model=CustomRepositoryModel,
            processor=type(repository_processor),
        )
        custom = CustomRepositoryModel(repository_model.config)
        with torch.no_grad():
            custom.weight.add_(7)
            custom.bias.sub_(2)
        save_model(tmp_path, model=custom, processor=repository_processor)

        loaded = AutoModel(tmp_path)
        processor = AutoProcessor(tmp_path)
        assert type(loaded) is CustomRepositoryModel
        assert type(processor) is type(repository_processor)
        assert dict(processor.config) == dict(repository_processor.config)
        assert processor.dtype == repository_processor.dtype
        for name, expected in custom.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)
        assert type(AutoModel(repository_model.config)) is default_model

    @pytest.mark.parametrize("factory", [AutoModel, AutoModel.from_config])
    def test_from_config(self, repository_model, factory) -> None:
        config = type(repository_model.config)({"width": 2})

        model = factory(config, device="cpu", dtype=torch.float64, backend="eager")

        assert type(model) is type(repository_model)
        assert model.weight.shape == (2,)
        assert model.weight.dtype == torch.float64
        assert model.backend == "eager"
        assert model.training

    def test_from_pretrained_does_not_initialize_from_config(self, repository_model) -> None:
        with pytest.raises(TypeError, match="requires a repository"):
            AutoModel.from_pretrained(repository_model.config)

    @pytest.mark.parametrize("factory", [AutoModel, AutoModel.from_pretrained])
    def test_from_pretrained(self, tmp_path: Path, repository_model, repository_processor, factory) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        model = factory(tmp_path, device="cpu", dtype=torch.float64, backend="eager")

        assert type(model) is type(repository_model)
        assert model.weight.dtype == torch.float64
        assert model.backend == "eager"
        assert not model.training


class TestAutoProcessor:
    @pytest.mark.parametrize("factory", [AutoProcessor, AutoProcessor.from_pretrained])
    def test_from_pretrained(self, tmp_path: Path, repository_model, repository_processor, factory) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        processor = factory(tmp_path, device="cpu", dtype=torch.float32)

        assert type(processor) is type(repository_processor)
        assert processor.config.scale == 2
        assert processor.device == torch.device("cpu")
        assert processor.dtype == torch.float32


def test_hf_keyword_and_resolved_snapshot(tmp_path: Path, repository_model, repository_processor) -> None:
    save_model(tmp_path, model=repository_model, processor=repository_processor)
    snapshot = resolve_repository(tmp_path)
    for model in (AutoModel(snapshot), AutoModel.from_pretrained(pretrained_model_name_or_path=tmp_path)):
        assert type(model) is type(repository_model)
        assert not model.training
        for name, value in repository_model.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value)
    for processor in (AutoProcessor(snapshot), AutoProcessor.from_pretrained(pretrained_model_name_or_path=tmp_path)):
        assert type(processor) is type(repository_processor)
        assert dict(processor.config) == dict(repository_processor.config)
