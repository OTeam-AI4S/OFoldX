r"""Fixtures for the public task-API surface (registry, pipelines, repository round-trip).

The registrations these tests exercise live in the registry objects
(``ofoldx.models.registry.MODELS`` and ``ofoldx.pipelines.registry.PIPELINES``, both ``dict``
subclasses). A test that
registers a synthetic architecture would otherwise leak it into every later test in the process, so
``_isolate_registries`` snapshots and restores those dicts around each test -- exactly the pattern the
suite already uses for the global RNG in ``tests/conftest.py``.

The synthetic Model/Processor pair is deliberately tiny: one real ``nn.Parameter`` so the safetensors
round-trip actually moves weights, a config-less Processor, and four task methods that record the
keyword arguments the Pipeline forwarded. Recording the forwarded kwargs is what lets the generator
boundary tests prove that ``seed`` is resolved at the Pipeline and never reaches the model graph.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def _isolate_registries():
    r"""Snapshot and restore the global model/pipeline registries around each test."""

    from ofoldx.models import registry as model_registry
    from ofoldx.pipelines import registry as pipeline_registry

    tables = (
        model_registry.MODELS,
        model_registry.MODELS.by_config,
        model_registry.MODELS.by_pair,
        pipeline_registry.PIPELINES,
        pipeline_registry.PIPELINES.ids,
    )
    snapshots = [dict(table) for table in tables]
    try:
        yield
    finally:
        for table, snapshot in zip(tables, snapshots):
            table.clear()
            table.update(snapshot)


def _build_tiny_classes():
    r"""Create a fresh, uniquely-typed synthetic Model/Processor/Config triple."""

    class TinyConfig(dict):
        r"""A config that is just its own mapping of values."""

        def __init__(self, values=None):
            super().__init__(values or {})

    class TinyProcessor:
        r"""A config-less Processor that yields a one-tensor carrier."""

        config_class = None

        def __init__(self, *, device=None, dtype=None):
            self.device = device
            self.dtype = dtype

        def __call__(self, input, **kwargs):
            # A single tensor is enough for ``inputs_device`` to read placement off the carrier.
            return {"x": torch.zeros(1)}

        def to(self, device=None, dtype=None):
            self.device = device
            self.dtype = dtype
            return self

    class TinyModel(nn.Module):
        r"""A one-parameter Model whose task methods record what the Pipeline forwarded."""

        config_class = TinyConfig

        def __init__(self, config):
            super().__init__()
            self.config = TinyConfig(dict(config))
            self.weight = nn.Parameter(torch.zeros(int(self.config.get("dim", 3))))
            self.received = None

        def _run(self, method, **kwargs):
            # ``kwargs`` is exactly what the Pipeline chose to forward -- capturing all of it (rather
            # than naming ``generator``) is what makes a leaked ``seed`` observable.
            self.received = {"method": method, **kwargs}
            return SimpleNamespace(**self.received)

        def fold(self, inputs, **kwargs):
            return self._run("fold", **kwargs)

        def design(self, inputs, **kwargs):
            return self._run("design", **kwargs)

        def score(self, inputs, **kwargs):
            return self._run("score", **kwargs)

        def pack(self, inputs, **kwargs):
            return self._run("pack", **kwargs)

    return TinyConfig, TinyProcessor, TinyModel


@pytest.fixture
def tiny_classes():
    r"""Factory returning a fresh (TinyConfig, TinyProcessor, TinyModel) triple per call."""

    return _build_tiny_classes
