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

from typing import ClassVar

import pytest

from ofoldx.pipelines.registry import PipelineRegistry


class Pipeline:
    task: ClassVar[str] = "fold"


class OtherPipeline:
    task: ClassVar[str] = "fold"


def _registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register("pipeline", pipeline=Pipeline)
    return registry


class TestPipelineRegistry:
    def test_lookup(self) -> None:
        registration = _registry().lookup_pipeline("pipeline")

        assert registration.id == "pipeline"
        assert registration.pipeline is Pipeline
        assert registration.task == "fold"

    def test_reverse_lookup(self) -> None:
        assert _registry().id_for(Pipeline) == "pipeline"

    def test_exact_id(self) -> None:
        with pytest.raises(KeyError):
            _registry().lookup_pipeline("PIPELINE")

    def test_exact_class(self) -> None:
        with pytest.raises(KeyError):
            _registry().id_for(OtherPipeline)

    @pytest.mark.parametrize(
        ("id", "pipeline"),
        [
            pytest.param("pipeline", OtherPipeline, id="id"),
            pytest.param("other", Pipeline, id="class"),
            pytest.param("", OtherPipeline, id="empty-id"),
        ],
    )
    def test_conflicts(self, id: str, pipeline: type) -> None:
        with pytest.raises(ValueError):
            _registry().register(id, pipeline=pipeline)
