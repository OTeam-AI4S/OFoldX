r"""
Configuration values for the OFoldX runner.
"""

from __future__ import annotations

from typing import Any

import chanfig
import danling as dl
from danling.runners.config import DdpConfig as DanlingDdpConfig


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


class ClusterSamplingConfig(chanfig.Config):
    r"""Protenix-style cluster-weighted sampling for a training CIF split.

    Example:
        ``{"data": {"train": {"source": "train.jsonl", "cluster_sampling":``
        ``{"enabled": True, "cluster_file": "clusters.txt", "epoch_size": 10000}}}}``

    This asymmetric-unit-only mode caches weighted index rows and scans all CIFs
    only when the cache is absent or invalid. Rows map back to full CIF
    records/structures and carry chain/interface metadata that can produce
    distinct Protenix crops. Default structure ingestion removes hydrogens and
    water; configured ion handling and ``exclude_mol_types`` change sampling
    rows only. ``epoch_size`` is the global draw count and DDP pads it upward
    so every rank receives
    ``ceil(epoch_size / world_size)`` rows.
    """

    enabled: bool = False
    cluster_file: str | None = None
    indices_file: str | None = None
    epoch_size: int | None = None
    replacement: bool = True
    contact_cutoff: float = 5.0
    include_all_chain_pairs: bool = False
    exclude_mol_types: list[str]
    seed: int | None = None
    beta: chanfig.Config
    alpha: chanfig.Config

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.enabled = False
        self.cluster_file = None
        self.indices_file = None
        self.epoch_size = None
        self.replacement = True
        self.contact_cutoff = 5.0
        self.include_all_chain_pairs = False
        self.exclude_mol_types = ["ions"]
        self.seed = None
        self.beta = chanfig.Config({"chain": 0.5, "interface": 1.0})
        self.alpha = chanfig.Config({"prot": 3.0, "nuc": 3.0, "ligand": 1.0})
        super().__init__(*args, **kwargs)


class DatasetConfig(chanfig.Config):
    r"""
    Configure one structure-dataset split.

    Additional configuration entries are forwarded to the processor for each item.

    Attributes:
        source: Directory, glob, manifest, structure file, or ``None``.
        seq_to_pdb_index: Optional JSON mapping from exact protein sequences to
            indexed MSA directory names.
        msa_path: Optional root directory for indexed MSAs. Must be configured
            together with ``seq_to_pdb_index``.
    """

    source: str | None = None
    seq_to_pdb_index: str | None = None
    msa_path: str | None = None
    cluster_sampling: ClusterSamplingConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.source = None
        self.seq_to_pdb_index = None
        self.msa_path = None
        self.cluster_sampling = ClusterSamplingConfig()
        super().__init__(*args, **kwargs)
        if not isinstance(self.cluster_sampling, ClusterSamplingConfig):
            self.cluster_sampling = ClusterSamplingConfig(self.cluster_sampling)


class DataConfig(chanfig.Config):
    r"""
    Map split names to [`DatasetConfig`][ofoldx.runners.config.DatasetConfig] values.
    """

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        super().__post_init__(*args, **kwargs)
        for split, spec in self.items():
            if not isinstance(spec, DatasetConfig):
                self[split] = DatasetConfig(spec) if isinstance(spec, dict) else DatasetConfig(source=spec)


class DdpConfig(DanlingDdpConfig):
    r"""DDP settings whose unused-parameter policy defaults to the selected objective."""

    find_unused_parameters: bool | None = None  # type: ignore[assignment]


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
        gradient_checkpointing: Enable model-native activation checkpointing
            during training. This is independent of training-state checkpoint
            persistence.
    """

    model: ModelConfig
    processor: chanfig.Config
    data: DataConfig
    ddp: DdpConfig = DdpConfig()
    gradient_checkpointing: bool = False

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
