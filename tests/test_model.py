from __future__ import annotations

import chanfig
import pytest
from torch import nn

from ofoldx.compilation import HomogeneousBlockRegions
from ofoldx.model import Model


class _Operator(nn.Module):
    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend


class _RepeatedModule(HomogeneousBlockRegions, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.blocks = nn.ModuleList((_Operator(), _Operator()))


class _NestedRepeatedModule(HomogeneousBlockRegions, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList((_RepeatedModule(), _Operator()))


class _UnownedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList((_Operator(),))


class _Model(Model):
    config_class = chanfig.Config

    def __init__(self) -> None:
        super().__init__(chanfig.Config())
        self.operator = _Operator("eager")
        self.repeated = _RepeatedModule()
        self.unowned = _UnownedModule()
        self.post_init()


class TestModel:
    def test_backend_default(self) -> None:
        model = _Model()

        assert model.backend == "auto"
        assert model.operator.backend == "eager"

    @pytest.mark.parametrize("backend", ["auto", "eager", "sdpa", "oops", "cueq", "apex", "deepspeed"])
    def test_backend(self, backend: str) -> None:
        model = _Model()

        assert model.set_backend(backend) is model
        assert model.backend == backend
        assert model.operator.backend == backend

    def test_gradient_checkpointing(self) -> None:
        model = _Model()

        assert not model.is_gradient_checkpointing
        assert model.gradient_checkpointing_enable() is model
        assert model.is_gradient_checkpointing
        assert model.gradient_checkpointing_disable() is model
        assert not model.is_gradient_checkpointing

    def test_compilation_regions(self) -> None:
        model = _Model()

        assert model.compilation_region_groups() == (tuple(model.repeated.blocks),)

        model.repeated = _NestedRepeatedModule()
        with pytest.raises(ValueError, match="contain one another"):
            model.compilation_region_groups()
