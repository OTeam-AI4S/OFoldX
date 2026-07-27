r"""
Family-declared training contracts.

Trainability and every model-specific training decision belong to the model family, not the Runner. Each
groundable family declares a module-level ``TRAINING`` [`TrainingContract`][ofoldx.models.training.TrainingContract]
in its own ``objectives.py``; the Runner reads it (``objectives.TRAINING``) for the exact architecture it is
asked to train and supplies only the generic machinery (batching, DDP, the train/eval/infer loop). Because a
contract lists the *exact* architecture ids it covers, same-package variants that do not share it (e.g. an
affinity or confidence head) are simply not covered and are not trainable -- trainability is an explicit
family declaration, never inferred from a directory, and there is no second global registry: the contract is
a plain attribute the Runner looks up.

A family declares:

- ``architectures``: the exact registered ids this contract trains.
- ``criterion``: a factory ``(model, processor) -> criterion(pred, target) -> loss``. A factory (not a bare
  function) so the loss closes over the Model config (``sigma_data``) and Processor config (NA-MPNN's RNA
  token scheme) instead of baking a default that silently diverges from how the data was produced.
- ``strategy``: which generic batching/lowering/inference strategy the Runner applies -- ``"sequence"``
  (padded recovery), ``"sequence_single"`` (batch_size=1 recovery), or ``"denoising"`` (batch_size=1
  coordinate diffusion). This is the only coupling to Runner machinery, and it is a name, not an import.
- ``collate_options``: family-specific batching parameters (e.g. the sequence pad token id).

The objective trains only the parameters on the loss path; a model's auxiliary heads (distogram, confidence,
B-factor), sampling-only modules, and (input-conditionally) template embedders receive no gradient. The
Runner runs DDP with ``static_graph=True``, which learns the used/unused parameter set from the first
iteration's autograd hooks -- so unused modules are handled without the family enumerating them, and without
the fragility of a freeze list. This assumes the used/unused set is constant across steps (a homogeneous
training set), which holds once the non-groundable variants are excluded.
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
