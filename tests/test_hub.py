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

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from ofoldx.artifacts import save_model
from ofoldx.chemistry import Molecule, Recipe, build_system, read_smiles
from ofoldx.data import Structure
from ofoldx.hub import ResolvedRepository, load_model, load_processor, resolve_hub_repository, resolve_repository
from ofoldx.models.alphafold3 import AlphaFold3Processor, AlphaFold3ProcessorConfig
from ofoldx.models.boltz1 import Boltz1Processor
from ofoldx.models.boltz2 import Boltz2AffinityProcessor, Boltz2Processor
from ofoldx.models.pxdesign import PxDesignProcessor
from ofoldx.models.registry import MODELS


def test_dataset_snapshot_identity_survives_resolution(tmp_path, monkeypatch):
    def snapshot(*, repo_id, repo_type, revision, cache_dir, allow_patterns, token, local_files_only):
        assert repo_id == "example/ccd" and repo_type == "dataset"
        assert cache_dir in {None, tmp_path}
        assert local_files_only
        assert revision in {"release", "immutable-commit"}
        return str(tmp_path / "immutable-commit")

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    original = resolve_hub_repository(
        "example/ccd", revision="release", cache_dir=tmp_path, repo_type="dataset", local_files_only=True
    )
    restored = resolve_repository(original, allow_patterns=["components.safetensors"], local_files_only=True)
    assert restored == original
    assert restored.repo_type == "dataset" and restored.commit == "immutable-commit"


def test_resolved_snapshot_cannot_change_revision(tmp_path, monkeypatch):
    def snapshot(**kwargs):
        assert kwargs["revision"] == "immutable-commit"
        return str(tmp_path / "immutable-commit")

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    remote = ResolvedRepository(tmp_path / "immutable-commit", "example/model", "immutable-commit")
    assert resolve_repository(remote, revision="immutable-commit").commit == "immutable-commit"
    with pytest.raises(ValueError, match="already resolved"):
        resolve_repository(remote, revision="other-commit")
    with pytest.raises(ValueError, match="already resolved"):
        resolve_repository(ResolvedRepository(tmp_path), revision="main")
    with pytest.raises(ValueError, match="immutable commit"):
        resolve_repository(ResolvedRepository(tmp_path, "example/model"))


class TestLoadModel:
    @pytest.mark.parametrize("max_shard_size", ["5GB", 24])
    def test_round_trip(
        self,
        tmp_path: Path,
        repository_model,
        repository_processor,
        max_shard_size: int | str,
    ) -> None:
        save_model(
            tmp_path,
            model=repository_model,
            processor=repository_processor,
            max_shard_size=max_shard_size,
        )

        model = load_model(ResolvedRepository(tmp_path))
        assert type(model) is type(repository_model)
        assert not model.training
        assert torch.equal(model.weight, repository_model.weight)
        assert torch.equal(model.bias, repository_model.bias)
        assert model.weight.dtype == repository_model.weight.dtype

    def test_strict_weights(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)
        path = tmp_path / "model.safetensors"
        state_dict = load_file(path)
        weight = state_dict["weight"].clone()
        save_file({"weight": weight}, path, metadata={"format": "pt"})

        with pytest.raises(RuntimeError, match="Missing key"):
            load_model(ResolvedRepository(tmp_path))


class TestLoadProcessor:
    def test_load(self, tmp_path: Path, repository_model, repository_processor) -> None:
        save_model(tmp_path, model=repository_model, processor=repository_processor)

        processor = load_processor(ResolvedRepository(tmp_path))

        assert type(processor) is type(repository_processor)
        assert processor.config.scale == 2
        assert processor.dtype == torch.float16


@pytest.mark.parametrize(
    "processor_class",
    [AlphaFold3Processor, Boltz1Processor, Boltz2Processor, Boltz2AffinityProcessor, PxDesignProcessor],
)
def test_reference_seed_configuration_is_independent(processor_class):
    config = processor_class.config_class(seed=7)
    configured = processor_class(config)
    assert configured.seed == 7
    processor = processor_class(config, seed=0)
    assert config.seed == 7
    assert processor.seed == 0

    config.seed = 13
    assert configured.seed == 7
    assert processor.seed == 0
    processor.config.seed = 17
    assert processor.seed == 17
    processor.seed = 19
    assert processor.config.seed == 19


@pytest.mark.parametrize(
    ("architecture", "processor_class", "config", "artifact", "check_features"),
    [("test_reference_seed_af3", AlphaFold3Processor, AlphaFold3ProcessorConfig(seed=7), None, True)],
    ids=("seed-only",),
)
def test_reference_seed_repository_round_trip(
    tmp_path, repository_model, library, architecture, processor_class, config, artifact, check_features
):
    processor = processor_class(config, dtype=torch.float64)
    MODELS.register(architecture, model=type(repository_model), processor=processor_class)
    save_model(tmp_path, model=repository_model, processor=processor)
    restored = load_processor(ResolvedRepository(tmp_path))
    assert restored.seed == 7
    assert restored.dtype == torch.float64
    if artifact is not None:
        assert restored.config.artifact == artifact

    if check_features:
        component = read_smiles("CCCOCC", component_id="LIG").component
        system, _, _ = build_system(
            Recipe({"ligand": Molecule(("LIG",))}, {"L": "ligand"}), library, components={"LIG": component}
        )
        target = Structure.unobserved(system)
        expected = processor.prepare_input(target).features.reference_positions
        actual = restored.prepare_input(target).features.reference_positions
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    # Native repositories published before seed persistence contain no seed field.
    metadata_path = tmp_path / "processor.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("seed")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    legacy = load_processor(ResolvedRepository(tmp_path))
    assert legacy.seed == 42
    assert legacy.dtype == torch.float64
    if artifact is not None:
        assert legacy.config.artifact == artifact
    if check_features:
        expected = AlphaFold3Processor(dtype=torch.float64).prepare_input(target).features.reference_positions
        actual = legacy.prepare_input(target).features.reference_positions
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
