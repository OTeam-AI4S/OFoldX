from __future__ import annotations

import chanfig
import pytest
from torch import nn

from ofoldx.model import Model
from ofoldx.modules.backend import LayerNorm


class _Operator(nn.Module):
    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
        self.backend = backend


class _RepeatedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gradient_checkpointing = False


class _Model(Model):
    config_class = chanfig.Config

    def __init__(self) -> None:
        super().__init__(chanfig.Config())
        self.operator = _Operator("eager")
        self.repeated = _RepeatedModule()
        self.norm = LayerNorm(4)
        self.post_init()


def test_post_init_preserves_child_backend() -> None:
    model = _Model()

    assert model.backend == "auto"
    assert model.operator.backend == "eager"


@pytest.mark.parametrize("backend", ["auto", "eager", "sdpa", "oops", "cueq", "apex", "deepspeed"])
def test_set_backend_updates_nested_backend_owners(backend: str) -> None:
    model = _Model()

    returned = model.set_backend(backend)

    assert returned is model
    assert model.backend == backend
    assert model.operator.backend == backend
    assert model.norm.backend == backend


def test_gradient_checkpointing_lifecycle() -> None:
    model = _Model()

    assert model.is_gradient_checkpointing is False
    assert model.gradient_checkpointing_enable() is model
    assert model.is_gradient_checkpointing is True
    assert model.repeated.gradient_checkpointing is True
    assert callable(model.repeated._gradient_checkpointing_func)
    assert model.gradient_checkpointing_disable() is model
    assert model.is_gradient_checkpointing is False
