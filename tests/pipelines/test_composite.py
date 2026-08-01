from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from ofoldx.model import Model
from ofoldx.pipelines import PIPELINES, CompositePair, CompositePipeline, PairedComponent, PipelineExecutionStep
from ofoldx.processor import Processor


class CompositeChildComponent(PairedComponent[Model, Processor]):
    model_architecture = "test_repository"


class RepositoryCompositePipeline(CompositePipeline):
    pair_layout = (
        CompositePair("root", "", CompositeChildComponent),
        CompositePair("child", "child", CompositeChildComponent),
    )

    def __init__(
        self,
        *,
        root: CompositeChildComponent,
        child: CompositeChildComponent,
    ) -> None:
        super().__init__({"root": root, "child": child})

    @property
    def root(self) -> CompositeChildComponent:
        return cast(CompositeChildComponent, self.components["root"])

    @property
    def child(self) -> CompositeChildComponent:
        return cast(CompositeChildComponent, self.components["child"])

    def fold(self) -> None:
        pass


PIPELINES.register(
    "test_composite_pipeline",
    pipeline=RepositoryCompositePipeline,
    executions={"fold": (PipelineExecutionStep("execute", "pipeline", RepositoryCompositePipeline.fold),)},
)


class TestCompositePipeline:
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        model_type = type(repository_model)
        model_config_type = type(repository_model.config)
        processor_type = type(repository_processor)
        processor_config_type = type(repository_processor.config)
        child_model = model_type(model_config_type({"width": 2}))
        child_processor = processor_type(processor_config_type({"scale": 7}), dtype=torch.float64)
        pipeline = RepositoryCompositePipeline(
            root=CompositeChildComponent(model=repository_model, processor=repository_processor),
            child=CompositeChildComponent(model=child_model, processor=child_processor),
        )
        assert pipeline.source is None
        assert pipeline.components == {"root": pipeline.root, "child": pipeline.child}
        assert pipeline.root.source is None
        assert pipeline.child.source is None
        pipeline.save(tmp_path)
        loaded = RepositoryCompositePipeline.from_repository(tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_composite_pipeline"
        }
        assert not (tmp_path / "child" / "pipeline.json").exists()
        assert loaded.root.model.weight.shape == repository_model.weight.shape
        assert loaded.child.model.weight.shape == child_model.weight.shape
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.root.source is not None
        assert loaded.root.source.path == tmp_path
        assert loaded.root.source.subdirectory == ""
        assert loaded.child.source is not None
        assert loaded.child.source.path == tmp_path / "child"
        assert loaded.child.source.subdirectory == "child"
        assert loaded.child.processor.config is not None
        assert child_processor.config is not None
        assert dict(loaded.child.processor.config) == dict(child_processor.config)
        assert loaded.child.processor.dtype is torch.float64
        assert loaded.root.model.backend == "eager"
        assert loaded.child.model.backend == "eager"
        for key, expected in repository_model.state_dict().items():
            torch.testing.assert_close(loaded.root.model.state_dict()[key], expected)
        for key, expected in child_model.state_dict().items():
            torch.testing.assert_close(loaded.child.model.state_dict()[key], expected)

    def test_component(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        model_type = type(repository_model)
        model_config_type = type(repository_model.config)
        processor_type = type(repository_processor)
        processor_config_type = type(repository_processor.config)
        child_model = model_type(model_config_type({"width": 2}))
        child_processor = processor_type(processor_config_type({"scale": 7}))
        RepositoryCompositePipeline(
            root=CompositeChildComponent(model=repository_model, processor=repository_processor),
            child=CompositeChildComponent(model=child_model, processor=child_processor),
        ).save(tmp_path)

        child = RepositoryCompositePipeline.component_from_repository(
            tmp_path,
            role="child",
            backend="eager",
        )

        assert type(child) is CompositeChildComponent
        assert child.model.weight.shape == child_model.weight.shape
        assert child.source is not None
        assert child.source.path == tmp_path / "child"
        assert child.source.subdirectory == "child"

        root = RepositoryCompositePipeline.component_from_repository(
            tmp_path,
            role="root",
            backend="eager",
        )
        assert type(root) is CompositeChildComponent
        assert root.source is not None
        assert root.source.path == tmp_path
        assert root.source.subdirectory == ""

    def test_component_role_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="has no unique component role 'missing'"):
            RepositoryCompositePipeline.component_from_repository(tmp_path, role="missing")
