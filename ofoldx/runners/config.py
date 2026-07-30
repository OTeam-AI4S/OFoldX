r"""
Configuration values for the OFoldX runner.
"""

from __future__ import annotations

from typing import Any

import chanfig
import danling as dl


class ModelConfig(chanfig.Config):
    r"""
    Select and configure the model built by a runner.

    Additional configuration entries override architecture settings when
    ``pretrained`` is absent.

    Attributes:
        name: Registered architecture ID.
        pretrained: Optional local model repository or Hugging Face Hub repository ID.
        revision: Optional Hub revision (branch, tag, or commit) to resolve ``pretrained`` at. On load the
            Runner records the resolved commit sha back here, so a mutable id (branch/tag) resumes from the
            exact snapshot it was trained on.
    """

    name: str = ""
    pretrained: str | None = None
    revision: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.name = ""
        self.pretrained = None
        self.revision = None
        super().__init__(*args, **kwargs)


class DatasetConfig(chanfig.Config):
    r"""
    Configure one structure-dataset split.

    Additional configuration entries are forwarded to the processor for each item.

    Attributes:
        source: Directory, glob, manifest, structure file, or ``None``.
    """

    source: str | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.source = None
        super().__init__(*args, **kwargs)


class DataConfig(chanfig.Config):
    r"""
    Map split names to [`DatasetConfig`][ofoldx.runners.config.DatasetConfig] values.
    """

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        for split, spec in self.items():
            if not isinstance(spec, DatasetConfig):
                self[split] = DatasetConfig(spec) if isinstance(spec, dict) else DatasetConfig(source=spec)


class RunnerConfig(dl.RunnerConfig):
    r"""
    Configure OFoldX training and evaluation.

    Danling optimizer, scheduler, dataloader, checkpoint, and scoring options are
    inherited from [`danling.RunnerConfig`][danling.RunnerConfig].

    Attributes:
        model: Model selection and architecture (model-config) overrides.
        processor: Processor-config overrides that are not model settings
            (e.g. NA-MPNN ``shared_nucleic_tokens``).
        data: Dataset configuration keyed by split name.
    """

    model: ModelConfig
    processor: chanfig.Config
    data: DataConfig

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "model" not in self:
            self.model = ModelConfig()
        elif not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(self.model)
        if "processor" not in self:
            self.processor = chanfig.Config()
        elif not isinstance(self.processor, chanfig.Config):
            self.processor = chanfig.Config(self.processor)
        if "data" not in self:
            self.data = DataConfig()
        elif not isinstance(self.data, DataConfig):
            self.data = DataConfig(self.data)

    def post(self) -> None:
        r"""
        Finalize and validate the runner configuration.

        The model is selected either by ``model.pretrained`` (a repository whose snapshot supplies both
        the model and its processor) or by ``model.name`` (a registered architecture built fresh). When
        a repository is given, ``model.name`` is optional and, if present, is asserted against the loaded
        architecture at build time.

        Raises:
            ValueError: If neither ``model.pretrained`` nor ``model.name`` is set.
        """

        super().post()
        if not self.model.pretrained and not self.model.name:
            raise ValueError(
                "select a model with `model.pretrained` (a repository) or `model.name` "
                "(a registered architecture id)"
            )
