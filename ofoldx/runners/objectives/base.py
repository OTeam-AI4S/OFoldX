r"""The training-objective seam.

Family Models are inference-and-forward-only: their ``forward`` returns typed
predictions (or diffusion loss-ready tensors), never a scalar loss, and their
config carries no training state (DESIGN §10.3, DESIGN:3613-3620, DESIGN:4943-4946).
An :class:`Objective` is the single place where a batch's model output is reduced
to one scalar training loss — masked-mean reduction of the MPNN per-token loss,
the EDM weighted denoising loss of the diffusion families, and so on. Objectives
live entirely on the Runner side; nothing here is imported by Model or Processor
code.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class Objective(Protocol):
    r"""Reduce one collated, device-placed batch to ``(model_output, loss)``.

    ``batch`` is a family Input dataclass that has already been collated and moved
    to the Runner device. Implementations run ``model`` (the family ``forward`` or
    a family-specific training method), compute a single reduced scalar ``loss``
    ready for ``.backward()``, and return the raw model output alongside it so the
    Runner can feed metrics. The scalar ``loss`` must require grad in ``train``
    mode; the same call is reused for evaluation (no backward).
    """

    def __call__(self, model: nn.Module, batch: Any) -> tuple[Any, Tensor]:
        ...
