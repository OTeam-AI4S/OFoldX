# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

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
