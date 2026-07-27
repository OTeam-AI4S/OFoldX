r"""
Trainable-family pairings for the V1 Runner.

Each entry declares, for one architecture, how a single config produces a consistent
``(Model, Processor, objective, collate)`` quadruple. The Runner special-cases no family; it resolves
the architecture, looks up its pairing, and wires everything -- adding a trainable family later means
adding one entry here (plus its objective and collate), nothing in the Runner.

V1 wires exactly the two MPNN inverse-folding families that have a complete Dataset -> Target ->
Objective contract. Every other architecture is intentionally rejected as not-yet-trainable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..model import Model
from ..models.registry import ModelRegistration
from ..processor import Processor
from .collate import collate_nampnn, collate_proteinmpnn
from .objectives.mpnn import NaMpnnObjective, ProteinMpnnObjective

Objective = Callable[[Any, Any], tuple[Any, Any]]
Collate = Callable[[list[Any]], Any]


@dataclass(frozen=True)
class TrainingFamily:
    r"""
    A trainable family's config->pairing contract.

    Args:
        shared_config_keys: Model-config keys copied into the Processor config so the two agree
            (e.g. ProteinMPNN ``ca_only``).
        make_objective: Build the training objective from the (fresh or loaded) Processor, threading
            any processor-owned parameters the loss needs (e.g. NA-MPNN ``shared_nucleic_tokens``).
        collate: The family's batch collator.
    """

    shared_config_keys: tuple[str, ...]
    make_objective: Callable[[Processor], Objective]
    collate: Collate


def _proteinmpnn_objective(processor: Processor) -> Objective:
    return ProteinMpnnObjective()


def _nampnn_objective(processor: Processor) -> Objective:
    return NaMpnnObjective(shared_nucleic_tokens=processor.config.shared_nucleic_tokens)


#: Architecture id -> training pairing. V1 trains only these two families.
TRAINABLE: dict[str, TrainingFamily] = {
    "proteinmpnn": TrainingFamily(("ca_only",), _proteinmpnn_objective, collate_proteinmpnn),
    "nampnn": TrainingFamily((), _nampnn_objective, collate_nampnn),
}


def resolve_family(architecture: str) -> TrainingFamily:
    r"""
    Return the training pairing for ``architecture``, or raise if it has no V1 contract.
    """

    family = TRAINABLE.get(architecture)
    if family is None:
        raise ValueError(
            f"architecture {architecture!r} is not trainable in this Runner; a training contract "
            f"(Dataset -> Target -> Objective) is wired only for {', '.join(sorted(TRAINABLE))}"
        )
    return family


def build_fresh(
    registration: ModelRegistration,
    overrides: Mapping[str, Any],
) -> tuple[Model, Processor, Objective, Collate]:
    r"""
    Build a fresh Model and a config-consistent Processor + objective + collate for one architecture.
    """

    family = resolve_family(registration.id)
    model_config = registration.model_config(**overrides) if overrides else registration.model_config()
    model = registration.model(model_config)
    shared = {key: model_config[key] for key in family.shared_config_keys if key in model_config}
    processor = registration.processor(registration.processor.config_class(**shared))
    return model, processor, family.make_objective(processor), family.collate


def wire_loaded(registration: ModelRegistration, processor: Processor) -> tuple[Objective, Collate]:
    r"""
    Build the objective + collate for an already-loaded (repository) Model/Processor pair.
    """

    family = resolve_family(registration.id)
    return family.make_objective(processor), family.collate
