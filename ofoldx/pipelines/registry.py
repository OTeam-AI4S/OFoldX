r"""Concrete Pipeline registration."""

from __future__ import annotations

from importlib import import_module

_PIPELINES: dict[str, type] = {}
_PIPELINE_IDS: dict[type, str] = {}

_BUILTIN_PIPELINE_MODULES = {
    "ofoldx.ligandmpnn": "ofoldx.models.ligandmpnn",
    "ofoldx.mpnn_packer": "ofoldx.models.mpnn_packer",
    "ofoldx.nampnn": "ofoldx.models.nampnn",
    "ofoldx.proteinmpnn": "ofoldx.models.proteinmpnn",
}

# Deferred-but-reserved built-in ids: their pipeline registration is not wired up yet, so they are
# intentionally absent from _BUILTIN_PIPELINE_MODULES (i.e. not auto-imported / not routable via
# _pipeline_class). They must nonetheless stay RESERVED for their owner package so a third party
# cannot squat the id before the official owner is ready to register it.
#
# TODO(DESIGN.md §6.2 / §5948): promote "ofoldx.boltz2_affinity" into _BUILTIN_PIPELINE_MODULES once
# Boltz2AffinityPipeline gains a multi-pair (two-checkpoint) loader/factory and a .save(); its
# register_pipeline is deferred in ofoldx/models/boltz2/__init__.py until then. The class stays
# importable for manual construction, but the routed factory (api/tasks.py) must not resolve a
# 4-arg ctor with no save() through the (model, processor) path.
_RESERVED_PIPELINE_IDS = {
    "ofoldx.boltz2_affinity": "ofoldx.models.boltz2",
}


def register_pipeline(id: str, *, pipeline: type) -> None:
    r"""Register one concrete Pipeline class."""

    if not isinstance(id, str):
        raise TypeError("pipeline id must be a string")
    if not id:
        raise ValueError("pipeline id must not be empty")
    if not isinstance(pipeline, type):
        raise TypeError("pipeline must be a class")

    # Reserve every built-in pipeline id for its owner module (see models/registry.register_model):
    # a registration owns a built-in id only when the pipeline class is defined inside the owner
    # package, so a pre-import squatter cannot claim it first. New third-party ids are unaffected.
    owner_module = _BUILTIN_PIPELINE_MODULES.get(id) or _RESERVED_PIPELINE_IDS.get(id)
    if owner_module is not None:
        origin = getattr(pipeline, "__module__", "") or ""
        if origin != owner_module and not origin.startswith(f"{owner_module}."):
            raise ValueError(
                f"pipeline {id!r} is a reserved built-in id owned by {owner_module!r}; "
                "third-party pipelines must register under a new id"
            )

    if id in _PIPELINES:
        raise ValueError(f"pipeline {id!r} is already registered")
    if pipeline in _PIPELINE_IDS:
        raise ValueError(f"pipeline class is already registered as {_PIPELINE_IDS[pipeline]!r}")

    _PIPELINES[id] = pipeline
    _PIPELINE_IDS[pipeline] = id


def _pipeline_class(id: str) -> type:
    r"""Resolve one explicitly registered concrete Pipeline class."""

    pipeline = _PIPELINES.get(id)
    module = _BUILTIN_PIPELINE_MODULES.get(id)
    if pipeline is None and module is not None:
        import_module(module)
        pipeline = _PIPELINES.get(id)
    if pipeline is not None:
        return pipeline
    raise ValueError(f"unknown pipeline: {id!r}")
