from __future__ import annotations

import pytest

from ofoldx.pipelines.registry import PipelineRegistry


class Pipeline:
    pass


class OtherPipeline:
    pass


def _registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register("pipeline", pipeline=Pipeline)
    return registry


class TestPipelineRegistry:
    def test_lookup(self) -> None:
        assert _registry().lookup_pipeline("pipeline") is Pipeline

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
