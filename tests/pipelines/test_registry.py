from __future__ import annotations

import pytest

from ofoldx.pipelines.registry import PIPELINES, PipelineExecutionStep, PipelineRegistry


class Pipeline:
    def fold(self) -> None:
        pass

    def prepare_score(self) -> None:
        pass

    def decode_score(self) -> None:
        pass


class Model:
    def score(self) -> None:
        pass


class OtherPipeline:
    pass


def _registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register(
        "pipeline",
        pipeline=Pipeline,
        executions={
            "fold": (PipelineExecutionStep("execute", "pipeline", Pipeline.fold),),
            "score": (
                PipelineExecutionStep("prepare", "pipeline", Pipeline.prepare_score),
                PipelineExecutionStep("forward", "model", Model.score),
                PipelineExecutionStep("decode", "pipeline", Pipeline.decode_score),
            ),
        },
    )
    return registry


class TestPipelineRegistry:
    def test_lookup(self) -> None:
        registration = _registry().lookup_pipeline("pipeline")

        assert registration.id == "pipeline"
        assert registration.pipeline is Pipeline
        assert registration.tasks == {"fold", "score"}
        assert registration.executions["score"] == (
            PipelineExecutionStep("prepare", "pipeline", Pipeline.prepare_score),
            PipelineExecutionStep("forward", "model", Model.score),
            PipelineExecutionStep("decode", "pipeline", Pipeline.decode_score),
        )

    def test_reverse_lookup(self) -> None:
        assert _registry().id_for(Pipeline) == "pipeline"

    def test_executions_are_required(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PipelineRegistry().register("pipeline", pipeline=Pipeline, executions={})

        with pytest.raises(ValueError, match="at least one step"):
            PipelineRegistry().register("pipeline", pipeline=Pipeline, executions={"fold": ()})

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
            _registry().register(
                id,
                pipeline=pipeline,
                executions={"fold": (PipelineExecutionStep("execute", "pipeline", Pipeline.fold),)},
            )

    def test_nampnn_score_execution(self) -> None:
        from ofoldx.models.nampnn import NaMpnnModel, NaMpnnPipeline

        execution = PIPELINES.lookup_pipeline("nampnn").executions["score"]

        assert execution == (
            PipelineExecutionStep("prepare", "pipeline", NaMpnnPipeline.prepare_probabilities),
            PipelineExecutionStep("forward", "model", NaMpnnModel.score),
            PipelineExecutionStep("decode", "pipeline", NaMpnnPipeline.decode_score),
        )
