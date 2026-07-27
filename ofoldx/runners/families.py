r"""
Explicit per-family training builders for the V1 Runner.

Each trainable architecture has one explicit builder that constructs its Model, a matching Processor
(from the model config *and* the processor-overrides config), the objective, and the collate. A small
``architecture -> builder`` dispatch selects one; there is no reflective wiring layer. Adding a family
is one builder plus one dispatch entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..model import Model
from ..processor import Processor
from .collate import collate_nampnn, collate_proteinmpnn
from .objectives.mpnn import NaMpnnObjective, ProteinMpnnObjective

Objective = Callable[[Any, Any], tuple[Any, Any]]
Collate = Callable[[list[Any]], Any]
Built = tuple[Model, Processor, Objective, Collate]


def _model_overrides(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model_config.items() if key not in ("name", "pretrained")}


def _build_proteinmpnn(model_config: Mapping[str, Any], processor_overrides: Mapping[str, Any]) -> Built:
    from ..models.proteinmpnn.config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
    from ..models.proteinmpnn.model import ProteinMpnnModel
    from ..models.proteinmpnn.processor import ProteinMpnnProcessor

    model = ProteinMpnnModel(ProteinMpnnConfig(**_model_overrides(model_config)))
    processor_config = ProteinMpnnProcessorConfig(**dict(processor_overrides))
    processor_config.ca_only = model.config.ca_only  # keep the Processor's graph consistent with the Model
    processor = ProteinMpnnProcessor(processor_config)
    return model, processor, *_wire_proteinmpnn(processor)


def _wire_proteinmpnn(processor: Processor) -> tuple[Objective, Collate]:
    return ProteinMpnnObjective(), collate_proteinmpnn


def _build_nampnn(model_config: Mapping[str, Any], processor_overrides: Mapping[str, Any]) -> Built:
    from ..models.nampnn.config import NaMpnnConfig, NaMpnnProcessorConfig
    from ..models.nampnn.model import NaMpnnModel
    from ..models.nampnn.processor import NaMpnnProcessor

    model = NaMpnnModel(NaMpnnConfig(**_model_overrides(model_config)))
    processor = NaMpnnProcessor(NaMpnnProcessorConfig(**dict(processor_overrides)))
    return model, processor, *_wire_nampnn(processor)


def _wire_nampnn(processor: Processor) -> tuple[Objective, Collate]:
    # ``shared_nucleic_tokens`` is a Processor setting; the loss must read it back so RNA target spans
    # match how the Processor tokenized them (fresh runs set it via `processor.shared_nucleic_tokens`,
    # repository runs recover it from the loaded Processor).
    return NaMpnnObjective(shared_nucleic_tokens=processor.config.shared_nucleic_tokens), collate_nampnn


_FRESH_BUILDERS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Built]] = {
    "proteinmpnn": _build_proteinmpnn,
    "nampnn": _build_nampnn,
}
_LOADED_WIRING: dict[str, Callable[[Processor], tuple[Objective, Collate]]] = {
    "proteinmpnn": _wire_proteinmpnn,
    "nampnn": _wire_nampnn,
}


def require_trainable(architecture: str) -> None:
    if architecture not in _FRESH_BUILDERS:
        raise ValueError(
            f"architecture {architecture!r} is not trainable in this Runner; a training contract "
            f"(Dataset -> Target -> Objective) is wired only for {', '.join(sorted(_FRESH_BUILDERS))}"
        )


def build_fresh(architecture: str, model_config: Mapping[str, Any], processor_overrides: Mapping[str, Any]) -> Built:
    r"""
    Build a fresh Model, matching Processor, objective, and collate for ``architecture``.
    """

    require_trainable(architecture)
    return _FRESH_BUILDERS[architecture](model_config, processor_overrides)


def wire_loaded(architecture: str, processor: Processor) -> tuple[Objective, Collate]:
    r"""
    Build the objective + collate for an already-loaded (repository) Processor.
    """

    require_trainable(architecture)
    return _LOADED_WIRING[architecture](processor)
