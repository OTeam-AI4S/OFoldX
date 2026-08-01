r"""Configuration values added by the OFoldX Runner."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import chanfig
import danling as dl
from danling.runners.config import CompileConfig as DanlingCompileConfig
from danling.runners.config import DdpConfig as DanlingDdpConfig

from ..modules.backend.resolve import PUBLIC_BACKENDS


class ModelConfig(chanfig.Config):
    r"""Select a fresh registered architecture or a repository to fine-tune."""

    architecture: str = ""
    pretrained: str | None = None
    revision: str | None = None


class DdpConfig(DanlingDdpConfig):
    r"""DDP options whose unused-parameter default may follow the objective."""

    find_unused_parameters: bool | None = None  # type: ignore[assignment]


class CompileConfig(DanlingCompileConfig):
    r"""Select whole-model or OFoldX-defined regional compilation."""

    scope: str = "model"

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if self.scope not in {"model", "regions"}:
            raise ValueError("compile.scope must be either 'model' or 'regions'")
        if self.scope == "regions":
            if not self.enabled:
                raise ValueError("compile.scope='regions' requires compile.enabled=true")
            if self.fullgraph is not True:
                raise ValueError("compile.scope='regions' requires compile.fullgraph=true")
            if self.dynamic is not True:
                raise ValueError("compile.scope='regions' requires compile.dynamic=true")


class DatasetConfig(chanfig.Config):
    r"""Configure one structure-dataset split."""

    source: str | None = None
    sampling: chanfig.Config | None = None
    msa_index: str | None = None
    msa_root: str | None = None
    paired_msa_filename: str = "uniref100_hits.a3m"
    unpaired_msa_filename: str = "mmseqs_other_hits.a3m"
    processing: chanfig.Config = chanfig.Config()

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        if "structure" in self:
            raise ValueError(
                "data split `structure` options are not supported; apply deterministic structure selection "
                "and CCD completion during offline preparation"
            )
        if self.source is not None:
            self.source = os.fspath(self.source)
        if self.msa_index is not None:
            self.msa_index = os.fspath(self.msa_index)
        if self.msa_root is not None:
            self.msa_root = os.fspath(self.msa_root)
        if self.sampling is not None and not isinstance(self.sampling, chanfig.Config):
            self.sampling = chanfig.Config(self.sampling)
        if self.sampling is not None and self.sampling.get("index") is not None:
            self.sampling.index = os.fspath(self.sampling.index)
        if "processor" in self:
            raise ValueError("data split processing arguments belong under `processing`, not `processor`")
        if not isinstance(self.processing, chanfig.Config):
            self.processing = chanfig.Config(self.processing)


class DataConfig(chanfig.Config):
    r"""Map split names to [`DatasetConfig`][ofoldx.runners.DatasetConfig]."""

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        for split, value in tuple(self.items()):
            if isinstance(value, DatasetConfig):
                continue
            if isinstance(value, Mapping):
                self[split] = DatasetConfig(value)
            else:
                self[split] = DatasetConfig(source=os.fspath(value))


class RunnerConfig(dl.RunnerConfig):
    r"""Configure model construction, processing, structure splits, and static ``forward`` keyword arguments."""

    model: ModelConfig = ModelConfig()
    processor: chanfig.Config = chanfig.Config()
    forward: chanfig.Config = chanfig.Config()
    data: DataConfig = DataConfig()
    ddp: DdpConfig = DdpConfig()
    compile: CompileConfig = CompileConfig()
    gradient_checkpointing: bool = False
    backend: str = "auto"

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        self.data = DataConfig(self.data)
        self.compile = CompileConfig(self.compile)
        self._validate_runner_options()

    def post(self) -> None:
        r"""Finalize inherited settings and require one model source."""

        self.data = DataConfig(self.data)
        self.compile = CompileConfig(self.compile)
        super().post()
        self._validate_runner_options()
        if not self.model.get("pretrained") and not self.model.get("architecture"):
            raise ValueError("select a model with `model.pretrained` or a registered id in `model.architecture`")

    def _validate_runner_options(self) -> None:
        if self.backend not in PUBLIC_BACKENDS:
            raise ValueError(f"backend must be one of: {', '.join(sorted(PUBLIC_BACKENDS))}")
        if self.compile.scope == "regions" and not self.gradient_checkpointing:
            raise ValueError("compile.scope='regions' requires gradient_checkpointing=true")
