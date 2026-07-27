r"""
Family-declared training contracts.

Trainability and every model-specific training decision belong to the model family, not the Runner. Each
groundable family declares a [`TrainingContract`][ofoldx.models.training.TrainingContract] in its own
``objectives.py`` and registers it here; the Runner reads the contract for the exact architecture it is
asked to train and supplies only the generic machinery (batching, the freeze/DDP mechanism, the train/eval/
infer loop). Because a contract lists the *exact* architecture ids it covers, same-package variants that do
not share the contract (e.g. an affinity or confidence head) are simply never registered and are not
trainable -- trainability is an explicit family declaration, never inferred from a directory.

A family declares:

- ``architectures``: the exact registered ids this contract trains.
- ``criterion``: a factory ``(model, processor) -> criterion(pred, target) -> loss``. A factory (not a bare
  function) so the loss closes over the Model config (``sigma_data``) and Processor config (NA-MPNN's RNA
  token scheme) instead of baking a default that silently diverges from how the data was produced.
- ``strategy``: which generic batching/lowering/inference strategy the Runner should apply -- ``"sequence"``
  (padded recovery), ``"sequence_single"`` (batch_size=1 recovery), or ``"denoising"`` (batch_size=1
  coordinate diffusion). This is the only coupling to Runner machinery, and it is a name, not an import.
- ``collate_options``: family-specific batching parameters (e.g. the sequence pad token id).
- ``freeze``: parameter-name prefixes for the modules this objective does not train. The Runner freezes
  them (``requires_grad=False``) so they are excluded from DDP's gradient reduction; otherwise DDP raises
  on the second step. DDP's ``find_unused_parameters=True`` cannot help here -- it can only trace unused
  parameters when a model's forward returns a list/dict/iterable, and these models return dataclasses --
  so the family must name the modules off the loss path. For a diffusion model these are the auxiliary
  heads (distogram, confidence, B-factor) and, because they are unused whenever a structure carries no
  template, the template embedder (so V1 does not fine-tune template conditioning).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: A criterion factory: ``(model, processor) -> (pred, target) -> loss``.
CriterionFactory = Callable[[Any, Any], Callable[[Any, Any], Any]]


@dataclass(frozen=True)
class TrainingContract:
    r"""One family's declaration of how its architecture(s) train. See the module docstring."""

    architectures: tuple[str, ...]
    criterion: CriterionFactory
    strategy: str
    collate_options: Mapping[str, Any] = field(default_factory=dict)
    freeze: tuple[str, ...] = ()


class TrainingContractRegistry(dict):
    r"""Maps each exact architecture id to its family-declared training contract."""

    def register(self, contract: TrainingContract) -> TrainingContract:
        r"""Register a contract under each of its exact architecture ids."""

        for architecture in contract.architectures:
            self[architecture] = contract
        return contract


#: Exact architecture id -> its family-declared training contract.
TRAINING_CONTRACTS = TrainingContractRegistry()
