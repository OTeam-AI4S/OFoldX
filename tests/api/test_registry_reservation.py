r"""Lock built-in id reservation for the model and pipeline registries.

DESIGN §9.2 requires that a built-in architecture/pipeline id can be claimed only by its owner
package: a pre-import ``register_*("ofoldx.<builtin>", Fake)`` must be rejected (otherwise a squatter
would win first-writer-wins before the official owner is ever imported), the official owner must still
resolve, and a genuinely new third-party id must still register freely. The three DESIGN §7.4
multi-pair pipeline ids (``ofoldx.boltz2_affinity`` / ``ofoldx.odesign`` / ``ofoldx.boltzgen``) are now
routable built-ins that resolve to their owner concrete Pipeline while staying owner-reserved against a
squatter.
"""

from __future__ import annotations

import pytest

from ofoldx.models.registry import MODELS
from ofoldx.pipelines.registry import PIPELINES


def test_model_registry_rejects_non_owner_under_builtin_id(tiny_classes):
    r"""A foreign class cannot claim a reserved built-in architecture id."""

    _, TinyProcessor, TinyModel = tiny_classes()

    with pytest.raises(ValueError, match="reserved built-in id"):
        MODELS.register("ofoldx.alphafold3", model=TinyModel, processor=TinyProcessor)


def test_model_registry_resolves_official_owner():
    r"""The official owner package resolves its reserved built-in id."""

    registration = MODELS.lookup("ofoldx.alphafold3")

    assert registration.id == "ofoldx.alphafold3"
    assert registration.model.__module__.startswith("ofoldx.models.alphafold3")


def test_model_registry_accepts_new_third_party_id(tiny_classes):
    r"""A new, unreserved id registers and resolves to the third-party pair."""

    _, TinyProcessor, TinyModel = tiny_classes()

    MODELS.register("thirdparty.tiny", model=TinyModel, processor=TinyProcessor, standard_tasks={"fold"})

    resolved = MODELS.lookup("thirdparty.tiny")
    assert resolved.model is TinyModel
    assert resolved.processor is TinyProcessor


def test_pipeline_registry_rejects_non_owner_under_builtin_id():
    r"""A foreign class cannot claim a reserved built-in pipeline id."""

    class ThirdPartyPipeline:
        pass

    with pytest.raises(ValueError, match="reserved built-in id"):
        PIPELINES.register("ofoldx.proteinmpnn", pipeline=ThirdPartyPipeline)


def test_pipeline_registry_resolves_official_owner():
    r"""The official owner package resolves its reserved built-in pipeline id."""

    pipeline = PIPELINES.lookup("ofoldx.proteinmpnn")

    assert pipeline.__module__.startswith("ofoldx.models.proteinmpnn")


def test_pipeline_registry_accepts_new_third_party_id():
    r"""A new, unreserved pipeline id registers and resolves."""

    class ThirdPartyPipeline:
        pass

    PIPELINES.register("thirdparty.tiny", pipeline=ThirdPartyPipeline)

    assert PIPELINES.lookup("thirdparty.tiny") is ThirdPartyPipeline


@pytest.mark.parametrize(
    ("pipeline_id", "owner_module"),
    [
        ("ofoldx.boltz2_affinity", "ofoldx.models.boltz2"),
        ("ofoldx.odesign", "ofoldx.models.odesign"),
        ("ofoldx.boltzgen", "ofoldx.models.boltzgen"),
    ],
)
def test_multipair_pipeline_id_routes_to_owner(pipeline_id, owner_module):
    r"""Each DESIGN §7.4 multi-pair id resolves to its owner concrete Pipeline and declares a layout."""

    resolved = PIPELINES.lookup(pipeline_id)

    assert resolved.__module__.startswith(owner_module)
    # The class-level pair-layout is the signal the factory uses to route to from_repository.
    assert getattr(resolved, "_PAIR_LAYOUT", None) is not None
    assert callable(getattr(resolved, "from_repository", None))


@pytest.mark.parametrize(
    "pipeline_id",
    ["ofoldx.boltz2_affinity", "ofoldx.odesign", "ofoldx.boltzgen"],
)
def test_multipair_pipeline_id_rejects_non_owner_squatter(pipeline_id):
    r"""A routable multi-pair built-in id stays owner-reserved: a non-owner class cannot squat it."""

    class ThirdPartyPipeline:
        pass

    with pytest.raises(ValueError, match="reserved built-in id"):
        PIPELINES.register(pipeline_id, pipeline=ThirdPartyPipeline)
