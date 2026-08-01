from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.runners.config import DatasetConfig, RunnerConfig

STRUCTURE = Path(__file__).resolve().parents[2] / "examples" / "data" / "tripeptide.pdb"


class TestDatasetConfig:
    def test_structure_options(self) -> None:
        with pytest.raises(ValueError, match="offline preparation"):
            DatasetConfig({"source": str(STRUCTURE), "structure": {"model": 1}})

    def test_processing(self) -> None:
        config = DatasetConfig({"processing": {"filter_by_occupancy": False}})

        assert not config.processing.filter_by_occupancy


class TestRunnerConfig:
    def test_merged_data(self) -> None:
        config = RunnerConfig({"model": {"architecture": "proteinmpnn"}})
        config.merge({"data": {"train": str(STRUCTURE)}})

        config.post()

        assert isinstance(config.data.train, DatasetConfig)
        assert config.model.architecture == "proteinmpnn"
        assert config.data.train.source == str(STRUCTURE)
        with pytest.raises(ValueError, match="backend"):
            RunnerConfig({"model": {"architecture": "proteinmpnn"}, "backend": "unknown"})

    def test_compile_scope(self) -> None:
        config = RunnerConfig(
            {
                "model": {"architecture": "proteinmpnn"},
                "gradient_checkpointing": True,
                "compile": {
                    "enabled": True,
                    "scope": "regions",
                    "fullgraph": True,
                    "dynamic": True,
                },
            }
        )

        assert config.compile.scope == "regions"
        with pytest.raises(ValueError, match="compile.scope"):
            RunnerConfig(
                {
                    "model": {"architecture": "proteinmpnn"},
                    "compile": {"scope": "blocks"},
                }
            )
        with pytest.raises(ValueError, match="compile.fullgraph"):
            RunnerConfig(
                {
                    "model": {"architecture": "proteinmpnn"},
                    "compile": {
                        "enabled": True,
                        "scope": "regions",
                        "dynamic": True,
                    },
                }
            )
        with pytest.raises(ValueError, match="gradient_checkpointing"):
            RunnerConfig(
                {
                    "model": {"architecture": "proteinmpnn"},
                    "compile": {
                        "enabled": True,
                        "scope": "regions",
                        "fullgraph": True,
                        "dynamic": True,
                    },
                }
            )
