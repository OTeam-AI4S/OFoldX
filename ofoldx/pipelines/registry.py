r"""Concrete Pipeline registration."""

from __future__ import annotations

from importlib import import_module

_BUILTIN_PIPELINE_MODULES = {
    "ofoldx.esmfold2": "ofoldx.models.esmfold2",
    "ofoldx.ligandmpnn": "ofoldx.models.ligandmpnn",
    "ofoldx.mpnn_packer": "ofoldx.models.mpnn_packer",
    "ofoldx.nampnn": "ofoldx.models.nampnn",
    "ofoldx.protenix_constraint": "ofoldx.models.protenix",
    "ofoldx.proteinmpnn": "ofoldx.models.proteinmpnn",
    "ofoldx.pxdesign": "ofoldx.models.pxdesign",
    "ofoldx.rfdiffusion": "ofoldx.models.rfdiffusion",
    # DESIGN §7.4 multi-pair concrete Pipelines. Each declares a class-level ``_PAIR_LAYOUT`` and a
    # ``from_repository`` loader (``ofoldx/pipelines/_composite.py``), so the repository factory
    # (``api/tasks.py``) routes their marked repositories to that multi-pair loader instead of the
    # one-pair ``(model, processor)`` call. They remain owner-reserved by their presence here: a
    # third-party ``PIPELINES.register`` under these ids is rejected unless the class is defined in
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


class PipelineRegistry(dict):
    r"""Registry of concrete Pipeline classes, keyed by pipeline id.

    ``PIPELINES.register(id, pipeline=...)`` registers one class; :meth:`lookup`
    resolves an id (lazy-importing the owner of a built-in id), and :meth:`id_for`
    returns (via the reverse class-to-id index) the id a registered class was
    published under, giving a concrete writer one unambiguous marker value.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ids: dict[type, str] = {}

    def register(self, id: str, *, pipeline: type) -> None:
        r"""Register one concrete Pipeline class."""

        if not isinstance(id, str):
            raise TypeError("pipeline id must be a string")
        if not id:
            raise ValueError("pipeline id must not be empty")
        if not isinstance(pipeline, type):
            raise TypeError("pipeline must be a class")

        # Reserve every built-in pipeline id for its owner module (see models.registry.ModelRegistry):
        # a registration owns a built-in id only when the pipeline class is defined inside the owner
        # package, so a pre-import squatter cannot claim it first. New third-party ids are unaffected.
        owner_module = _BUILTIN_PIPELINE_MODULES.get(id)
        if owner_module is not None:
            origin = getattr(pipeline, "__module__", "") or ""
            if origin != owner_module and not origin.startswith(f"{owner_module}."):
                raise ValueError(
                    f"pipeline {id!r} is a reserved built-in id owned by {owner_module!r}; "
                    "third-party pipelines must register under a new id"
                )

        if id in self:
            raise ValueError(f"pipeline {id!r} is already registered")
        if pipeline in self.ids:
            raise ValueError(f"pipeline class is already registered as {self.ids[pipeline]!r}")

        self[id] = pipeline
        self.ids[pipeline] = id

    def lookup(self, id: str) -> type:
        r"""Resolve one registered Pipeline class, importing a built-in owner on demand."""

        if id not in self and id in _BUILTIN_PIPELINE_MODULES:
            import_module(_BUILTIN_PIPELINE_MODULES[id])
        try:
            return self[id]
        except KeyError:
            raise ValueError(f"unknown pipeline: {id!r}") from None

    def id_for(self, pipeline: type) -> str | None:
        r"""Return the id a registered Pipeline class was published under, or ``None``."""

        return self.ids.get(pipeline)


PIPELINES = PipelineRegistry()
