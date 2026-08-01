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
