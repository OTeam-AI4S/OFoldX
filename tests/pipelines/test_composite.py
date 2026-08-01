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

import json
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import torch

from ofoldx.model import Model
from ofoldx.pipelines import PIPELINES, ComponentSpec, CompositePipeline, PairedComponent, PairedPipeline, Pipeline
from ofoldx.processor import Processor


class CompositeChildComponent(PairedPipeline[Model, Processor]):
    task: ClassVar[str] = "fold"

    model_architecture = "test_repository"

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.model.weight + self.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output

    def fold(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = fold


PIPELINES.register("test_composite_child", pipeline=CompositeChildComponent)


class ShiftedChildPipeline(CompositeChildComponent):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input) + 2.0


PIPELINES.register("test_shifted_child", pipeline=ShiftedChildPipeline)


class RepositoryCompositePipeline(CompositePipeline):
    task: ClassVar[str] = "fold"

    component_layout = (
        ComponentSpec("root", "root", CompositeChildComponent),
        ComponentSpec("child", "child", CompositeChildComponent),
    )

    def __init__(
        self,
        *,
        root: Pipeline,
        child: Pipeline,
    ) -> None:
        super().__init__({"root": root, "child": child})

    @property
    def root(self) -> CompositeChildComponent:
        return cast(CompositeChildComponent, self.components["root"])

    @property
    def child(self) -> CompositeChildComponent:
        return cast(CompositeChildComponent, self.components["child"])

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return self.root.prepare(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.root.postprocess(input, self.root.forward(input))
        return self.child(output)

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output

    def fold(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = fold


PIPELINES.register(
    "test_composite_pipeline",
    pipeline=RepositoryCompositePipeline,
)


class RepositoryPairComponent(PairedComponent[Model, Processor]):
    model_architecture = "test_repository"


class PairConditionedPipeline(CompositePipeline):
    task: ClassVar[str] = "calibrate"

    component_layout = (ComponentSpec("pair", "pair", RepositoryPairComponent),)

    def __init__(self, *, pair: RepositoryPairComponent) -> None:
        super().__init__({"pair": pair})

    @property
    def pair(self) -> RepositoryPairComponent:
        return cast(RepositoryPairComponent, self.components["pair"])

    def prepare(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.pair.processor.config.scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.pair.model.weight + self.pair.model.bias

    def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return output

    def calibrate(self, input: torch.Tensor) -> torch.Tensor:
        prepared = self.prepare(input)
        return self.postprocess(prepared, self.forward(prepared))

    __call__ = calibrate


PIPELINES.register("test_pair_conditioned_pipeline", pipeline=PairConditionedPipeline)


class TestCompositePipeline:
    def test_direct_pair_enters_eval_without_disabling_gradients(
        self, repository_model: Any, repository_processor: Any
    ) -> None:
        pair = RepositoryPairComponent(model=repository_model, processor=repository_processor)
        assert pair.model.training
        workflow = PairConditionedPipeline(pair=pair)
        assert not pair.model.training
        workflow(torch.ones(3)).sum().backward()
        assert pair.model.weight.grad is not None
        pair.model.train()
        assert pair.model.training

    def test_nested_pair_enters_eval_when_attached_to_outer_workflow(
        self, repository_model: Any, repository_processor: Any
    ) -> None:
        pair = RepositoryPairComponent(model=repository_model, processor=repository_processor)
        inner = PairConditionedPipeline(pair=pair)
        pair.model.train()
        assert pair.model.training
        outer = RepositoryCompositePipeline(root=inner, child=inner)
        assert outer.components["root"] is inner
        assert outer.components["child"] is inner
        assert not pair.model.training
        inner(torch.ones(3)).sum().backward()
        assert pair.model.weight.grad is not None
        pair.model.train()
        assert pair.model.training

    def test_custom_pipeline_child_remains_usable(self, repository_model: Any, repository_processor: Any) -> None:
        class CustomPipeline(Pipeline):
            task: ClassVar[str] = "fold"

            def prepare(self, input: torch.Tensor) -> torch.Tensor:
                return input

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return input + 1

            def postprocess(self, input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
                return output

            def __call__(self, input: torch.Tensor) -> torch.Tensor:
                return self.postprocess(input, self.forward(self.prepare(input)))

            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any):
                raise NotImplementedError

            def save(self, *args: Any, **kwargs: Any) -> None:
                raise NotImplementedError

            def to(self, *args: Any, **kwargs: Any):
                return self

            def set_backend(self, *args: Any, **kwargs: Any):
                return self

        child = CustomPipeline()
        root = CompositeChildComponent(model=repository_model, processor=repository_processor)
        workflow = RepositoryCompositePipeline(root=root, child=child)
        assert workflow.components["child"] is child
        torch.testing.assert_close(workflow(torch.ones(3)), root(torch.ones(3)) + 1)

    @pytest.mark.parametrize("invalid_directory", [False, True])
    def test_invalid_composition_preserves_existing_repository(
        self, tmp_path: Path, repository_model: Any, repository_processor: Any, invalid_directory: bool
    ) -> None:
        child = CompositeChildComponent(model=repository_model, processor=repository_processor)
        workflow = RepositoryCompositePipeline(root=child, child=child)
        workflow.save(tmp_path)
        before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

        if invalid_directory:

            class InvalidLayoutPipeline(RepositoryCompositePipeline):
                component_layout = (ComponentSpec("root", "", CompositeChildComponent),)

            PIPELINES.register("test_invalid_layout", pipeline=InvalidLayoutPipeline)
            invalid = InvalidLayoutPipeline(root=child, child=child)
            error = ValueError
        else:

            class UnregisteredChildPipeline(CompositeChildComponent):
                pass

            unregistered = UnregisteredChildPipeline(model=repository_model, processor=repository_processor)
            invalid = RepositoryCompositePipeline(root=child, child=unregistered)
            error = KeyError

        with pytest.raises(error):
            invalid.save(tmp_path)
        assert {
            path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
        } == before

    def test_non_executable_pair_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
    ) -> None:
        workflow = PairConditionedPipeline(
            pair=RepositoryPairComponent(model=repository_model, processor=repository_processor)
        )
        input = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([3.0, 8.0, 17.0])
        torch.testing.assert_close(workflow(input), expected)
        workflow.save(tmp_path)
        assert not (tmp_path / "pair" / "pipeline.json").exists()

        loaded = PairConditionedPipeline.from_pretrained(tmp_path)
        torch.testing.assert_close(loaded(input), expected)
        assert loaded.pair.source is not None
        assert loaded.pair.source.path == tmp_path / "pair"
        assert loaded.pair.source.subdirectory == "pair"
        assert loaded.pair.processor.dtype == repository_processor.dtype
        assert loaded.pair.processor.device == repository_processor.device
        assert dict(loaded.pair.processor.config) == dict(repository_processor.config)

        pair = PairConditionedPipeline.load_component(tmp_path, role="pair")
        assert isinstance(pair, RepositoryPairComponent)
        assert pair.source is not None
        assert pair.source.path == loaded.pair.source.path
        assert pair.source.subdirectory == loaded.pair.source.subdirectory
        isolated = PairConditionedPipeline(pair=pair)
        torch.testing.assert_close(isolated(input), expected)

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
        loaded = RepositoryCompositePipeline.from_pretrained(pretrained_model_name_or_path=tmp_path, backend="eager")

        assert json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8")) == {
            "pipeline": "test_composite_pipeline",
            "components": {
                "root": {"directory": "root", "pipeline": "test_composite_child"},
                "child": {"directory": "child", "pipeline": "test_composite_child"},
            },
        }
        assert (tmp_path / "child" / "pipeline.json").is_file()
        assert loaded.root.model.weight.shape == repository_model.weight.shape
        assert loaded.child.model.weight.shape == child_model.weight.shape
        assert loaded.source is not None
        assert loaded.source.path == tmp_path
        assert loaded.root.source is not None
        assert loaded.root.source.path == tmp_path / "root"
        assert loaded.root.source.subdirectory == "root"
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

        child = RepositoryCompositePipeline.load_component(
            tmp_path,
            role="child",
            backend="eager",
        )

        assert type(child) is CompositeChildComponent
        assert child.model.weight.shape == child_model.weight.shape
        assert child.source is not None
        assert child.source.path == tmp_path / "child"
        assert child.source.subdirectory == "child"

        root = RepositoryCompositePipeline.load_component(
            tmp_path,
            role="root",
            backend="eager",
        )
        assert type(root) is CompositeChildComponent
        assert root.source is not None
        assert root.source.path == tmp_path / "root"
        assert root.source.subdirectory == "root"

    def test_component_role_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="has no unique component role 'missing'"):
            RepositoryCompositePipeline.load_component(tmp_path, role="missing")

    @pytest.mark.parametrize("hub", [False, True])
    def test_nested_substitution_round_trip(
        self,
        tmp_path: Path,
        repository_model: Any,
        repository_processor: Any,
        monkeypatch: pytest.MonkeyPatch,
        hub: bool,
    ) -> None:
        from ofoldx import AutoPipelineForFold

        leaf = ShiftedChildPipeline(model=repository_model, processor=repository_processor)
        nested = RepositoryCompositePipeline(root=leaf, child=leaf)
        parent = RepositoryCompositePipeline(root=nested, child=leaf)
        input = torch.tensor([1.0, 2.0, 3.0])
        scale = repository_processor.config.scale * repository_model.weight
        shift = repository_model.bias + 2.0
        expected = scale**3 * input + shift * (scale**2 + scale + 1)
        parent.save(tmp_path)
        if hub:
            monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(tmp_path))
        repository = "owner/workflow" if hub else tmp_path
        loaded = AutoPipelineForFold(repository)
        torch.testing.assert_close(loaded(input), expected)
        prepared = loaded.prepare(input)
        torch.testing.assert_close(loaded.postprocess(prepared, loaded.forward(prepared)), expected)
        assert isinstance(loaded.components["root"], RepositoryCompositePipeline)
        child = RepositoryCompositePipeline.load_component(repository, role="root")
        torch.testing.assert_close(child(input), scale**2 * input + shift * (scale + 1))
        assert child.source.subdirectory == "root"
        assert child.root.source.subdirectory == "root/root"
        if hub:
            assert child.root.source.commit == loaded.source.commit == tmp_path.name
