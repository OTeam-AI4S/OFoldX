r"""Lock built-in id reservation for the model and pipeline registries.

DESIGN §9.2 requires that a built-in architecture/pipeline id can be claimed only by its owner
package: a pre-import ``register_*("ofoldx.<builtin>", Fake)`` must be rejected (otherwise a squatter
would win first-writer-wins before the official owner is ever imported), the official owner must still
resolve, and a genuinely new third-party id must still register freely. The ``boltz2_affinity``
pipeline is deliberately *not* routable yet -- its ``register_pipeline`` is deferred until it grows a
two-checkpoint loader -- so the pipeline registry must reject that id.
"""

from __future__ import annotations

import pytest

from ofoldx.models.registry import _registration, register_model
from ofoldx.pipelines.registry import _pipeline_class, register_pipeline


def test_model_registry_rejects_non_owner_under_builtin_id(tiny_classes):
    r"""A foreign class cannot claim a reserved built-in architecture id."""

    _, TinyProcessor, TinyModel = tiny_classes()

    with pytest.raises(ValueError, match="reserved built-in id"):
        register_model("ofoldx.alphafold3", model=TinyModel, processor=TinyProcessor)


def test_model_registry_resolves_official_owner():
    r"""The official owner package resolves its reserved built-in id."""

    registration = _registration("ofoldx.alphafold3")

    assert registration.id == "ofoldx.alphafold3"
    assert registration.model.__module__.startswith("ofoldx.models.alphafold3")


def test_model_registry_accepts_new_third_party_id(tiny_classes):
    r"""A new, unreserved id registers and resolves to the third-party pair."""

    _, TinyProcessor, TinyModel = tiny_classes()

    register_model("thirdparty.tiny", model=TinyModel, processor=TinyProcessor, standard_tasks={"fold"})

    resolved = _registration("thirdparty.tiny")
    assert resolved.model is TinyModel
    assert resolved.processor is TinyProcessor


def test_pipeline_registry_rejects_non_owner_under_builtin_id():
    r"""A foreign class cannot claim a reserved built-in pipeline id."""

    class ThirdPartyPipeline:
        pass

    with pytest.raises(ValueError, match="reserved built-in id"):
        register_pipeline("ofoldx.proteinmpnn", pipeline=ThirdPartyPipeline)


def test_pipeline_registry_resolves_official_owner():
    r"""The official owner package resolves its reserved built-in pipeline id."""

    pipeline = _pipeline_class("ofoldx.proteinmpnn")

    assert pipeline.__module__.startswith("ofoldx.models.proteinmpnn")


def test_pipeline_registry_accepts_new_third_party_id():
    r"""A new, unreserved pipeline id registers and resolves."""

    class ThirdPartyPipeline:
        pass

    register_pipeline("thirdparty.tiny", pipeline=ThirdPartyPipeline)

    assert _pipeline_class("thirdparty.tiny") is ThirdPartyPipeline


def test_boltz2_affinity_pipeline_is_not_routable():
    r"""The deferred boltz2_affinity id must not resolve through the pipeline registry."""

    with pytest.raises(ValueError, match="unknown pipeline"):
        _pipeline_class("ofoldx.boltz2_affinity")


def test_deferred_boltz2_affinity_id_rejects_non_owner_squatter():
    r"""A deferred-but-reserved built-in id stays protected: a non-owner class cannot squat it.

    ``ofoldx.boltz2_affinity`` is absent from ``_BUILTIN_PIPELINE_MODULES`` (its registration is
    deferred until it grows a two-checkpoint loader), but it must remain reserved for the owner
    package so a third party cannot claim the id before the official owner is ready.
    """

    class ThirdPartyPipeline:
        pass

    with pytest.raises(ValueError, match="reserved built-in id"):
        register_pipeline("ofoldx.boltz2_affinity", pipeline=ThirdPartyPipeline)


def test_deferred_boltz2_affinity_id_accepts_owner_registration():
    r"""The owner package may still register the deferred, reserved id under its own module."""

    class OwnerPipeline:
        pass

    OwnerPipeline.__module__ = "ofoldx.models.boltz2.pipeline"

    register_pipeline("ofoldx.boltz2_affinity", pipeline=OwnerPipeline)

    assert _pipeline_class("ofoldx.boltz2_affinity") is OwnerPipeline
