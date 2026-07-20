r"""Concrete Pipeline registration."""

from __future__ import annotations

from importlib import import_module

_PIPELINES: dict[str, type] = {}
_PIPELINE_IDS: dict[type, str] = {}

_BUILTIN_PIPELINE_MODULES = {
    "ofoldx.esmfold2": "ofoldx.models.esmfold2",
    "ofoldx.ligandmpnn": "ofoldx.models.ligandmpnn",
    "ofoldx.mpnn_packer": "ofoldx.models.mpnn_packer",
    "ofoldx.nampnn": "ofoldx.models.nampnn",
    "ofoldx.proteinmpnn": "ofoldx.models.proteinmpnn",
    "ofoldx.rfdiffusion": "ofoldx.models.rfdiffusion",
    # DESIGN §7.4 multi-pair concrete Pipelines. Each declares a class-level ``_PAIR_LAYOUT`` and a
    # ``from_repository`` loader (``ofoldx/pipelines/_composite.py``), so the repository factory
    # (``api/tasks.py``) routes their marked repositories to that multi-pair loader instead of the
    # one-pair ``(model, processor)`` call. They remain owner-reserved by their presence here: a
    # third-party ``register_pipeline`` under these ids is rejected unless the class is defined in
    # the owner module.
    #   * ofoldx.boltz2_affinity -> Boltz2AffinityPipeline (affinity root pair + ``structure/`` pair).
    #   * ofoldx.odesign -> ODesignPipeline (ODesign root pair + ``protein_designer/`` +
    #     ``ligand_designer/`` read-only sequence-designer pairs).
    #   * ofoldx.boltzgen -> BoltzGenPipeline (design root pair + ``ifold/`` pair + optional injected
    #     boltz2 ``fold/`` / ``affinity/`` stages).
    "ofoldx.boltz2_affinity": "ofoldx.models.boltz2",
    "ofoldx.odesign": "ofoldx.models.odesign",
    "ofoldx.boltzgen": "ofoldx.models.boltzgen",
}

# No pipeline id is currently deferred-but-reserved: every owner-reserved built-in id is a routable
# entry in _BUILTIN_PIPELINE_MODULES above. The map is retained so a future concrete Pipeline can be
# reserved for its owner before its loader/factory route lands (see register_pipeline).
_RESERVED_PIPELINE_IDS: dict[str, str] = {}


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
