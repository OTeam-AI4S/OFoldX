from __future__ import annotations

import chanfig
import pytest
from torch import nn

from ofoldx.model import Model
from ofoldx.modules.backend import LayerNorm


class _Operator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backend = "auto"


class _Model(Model):
    config_class = chanfig.Config

    def __init__(self) -> None:
        super().__init__(chanfig.Config())
        self.operator = _Operator()
        self.norm = LayerNorm(4)
        self.post_init()


@pytest.mark.parametrize("backend", ["auto", "eager", "oops"])
def test_set_backend_updates_nested_backend_owners(backend: str) -> None:
    model = _Model()

    returned = model.set_backend(backend)

    assert returned is model
    assert model.backend == backend
    assert model.operator.backend == backend
    assert model.norm.backend == backend
