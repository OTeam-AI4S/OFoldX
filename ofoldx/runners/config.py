r"""Config-only training, evaluation, and inference configuration for the Runner.

The :class:`RunnerConfig` extends danling's ``RunnerConfig`` (optimizer,
scheduler, dataloader, checkpoint, score sections) with the two things the
OFoldX Runner adds: which Model to build (:class:`ModelConfig`) and which raw
structures to train/evaluate/infer on (:class:`DataConfig`). Training provenance
lives here on the Runner config and never leaks into the frozen Model or
Processor config (DESIGN Principle 10; §10.3): a family Model config carries no
dataset, optimizer, schedule, or objective state.
"""

from __future__ import annotations

from typing import Any

import chanfig
import danling as dl


class ModelConfig(chanfig.Config):
    r"""Selects and parameterizes the Model the Runner builds.

    - ``name`` — a **required** registered architecture id (e.g.
      ``"ofoldx.proteinmpnn"``); it selects the family the Runner builds.
    - ``pretrained`` — an **optional** Hub or local repository reference. When
      set, the Runner loads that repository's weights (fine-tuning / continued
      training); when unset, the Model is **randomly initialized**.

    Extra keys (anything other than ``name`` / ``pretrained``) are forwarded as
    typed Model-config overrides on the random-init path; on the ``pretrained``
    path the architecture and config come from the repository.
    """

    name: str | None = None
    pretrained: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = None
        self.pretrained = None
        super().__init__(*args, **kwargs)


class DatasetConfig(chanfig.Config):
    r"""One split's raw-structure source plus per-call Processor arguments.

    ``source`` names the raw structures (a directory, glob, or manifest file of
    mmCIF/PDB paths). Any extra keys become keyword arguments to the family
    Processor's ``__call__`` (e.g. ``msa=``, ``templates=``) so the same lowering
    path inference uses drives training.
    """

    source: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.source = None
        super().__init__(*args, **kwargs)


class DataConfig(chanfig.Config):
    r"""Split-name -> :class:`DatasetConfig`.

    Split names match the danling ``train_splits`` / ``evaluate_splits`` and the
    ``infer`` split (default keys ``train`` / ``validation`` / ``test`` /
    ``infer``). Each value is coerced to a :class:`DatasetConfig`.
    """

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        for split, spec in self.items():
            if not isinstance(spec, DatasetConfig):
                self[split] = DatasetConfig(spec) if isinstance(spec, dict) else DatasetConfig(source=spec)


class RunnerConfig(dl.RunnerConfig):
    r"""Top-level OFoldX Runner configuration.

    Inherits danling's ``optim`` / ``sched`` / ``dataloader`` / ``ckpt`` / ``score``
    sections and adds :attr:`model`, :attr:`data`, and :attr:`objective`. Accepts a
    fully constructed instance or any mapping it can be built from (YAML / CLI via
    the inherited ``parse``).

    Args:
        runner: Registry key resolved through :data:`ofoldx.runners.registry.RUNNERS`.
        training: When ``False``, training splits are ignored (evaluate / infer only).
        model: The Model to build (see :class:`ModelConfig`).
        data: Split -> dataset spec (see :class:`DataConfig`).
        objective: Objective key. When ``None`` the Runner derives it from the
            Model architecture (e.g. the MPNN families -> ``"mpnn"``, the diffusion
            families -> ``"diffusion"``).
    """

    runner: str = "ofoldx"
    training: bool = True
    objective: str | None = None

    model: ModelConfig
    data: DataConfig

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "model" not in self:
            self.model = ModelConfig()
        elif not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(self.model)
        if "data" not in self:
            self.data = DataConfig()
        elif not isinstance(self.data, DataConfig):
            self.data = DataConfig(self.data)

    def post(self) -> None:
        super().post()
        if not self.model.name:
            raise ValueError("`model.name` (a registered architecture id) is required")
