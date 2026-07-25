r"""
Registry isolation for public task-API tests.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_registries():
    r"""
    Snapshot and restore the global model/pipeline registries around each test.
    """

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
            dict.clear(table)
            dict.update(table, snapshot)
