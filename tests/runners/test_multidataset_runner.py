from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from torch.utils.data import Dataset

from ofoldx.runners.config import DatasetMixtureConfig
from ofoldx.runners.multidataset import MixtureStructureDataset, MixtureWeightedSampler
from ofoldx.runners.runner import Runner


class _SizedDataset(Dataset[int]):
    def __init__(self, size: int = 2) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _Config(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _fake_runner(mixture: DatasetMixtureConfig, *, split: str = "train") -> SimpleNamespace:
    return SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        _forward_cluster_indices=False,
        config=_Config(data={split: mixture}, seed=31, workspace={"root": "workspace"}),
    )


def test_mixture_builds_sorted_children_with_independent_routing_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    cache_calls: list[tuple[str, str]] = []

    def structure_dataset(*args: Any, **kwargs: Any) -> _SizedDataset:
        calls.append(("ordinary", args, kwargs))
        return _SizedDataset()

    def clustered_dataset(*args: Any, **kwargs: Any) -> _SizedDataset:
        calls.append(("clustered", args, kwargs))
        return _SizedDataset()

    def cache_path(_workspace: str, source: str, cluster_file: str, **_kwargs: Any) -> str:
        cache_calls.append((source, cluster_file))
        return f"cache-{source}"

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    monkeypatch.setattr("ofoldx.runners.dataset.ClusteredStructureDataset", clustered_dataset)
    monkeypatch.setattr("ofoldx.runners.cluster.default_cluster_indices_path", cache_path)
    mixture = DatasetMixtureConfig(
        {
            "datasets": {
                "zeta": {
                    "source": "zeta.cif",
                    "seq_to_pdb_index": "zeta-index.json",
                    "msa_path": "zeta-msa",
                    "sampling_ratio": 3,
                    "processing_timeout_seconds": 4.5,
                    "max_consecutive_processing_errors": 9,
                    "max_tokens": 96,
                    "cluster_sampling": {"enabled": True, "cluster_file": "zeta-clusters.txt"},
                },
                "alpha": {
                    "source": "alpha.cif",
                    "seq_to_pdb_index": "alpha-index.json",
                    "msa_path": "alpha-msa",
                    "sampling_ratio": 1,
                    "max_tokens": 48,
                    "cluster_sampling": {"enabled": True, "cluster_file": "alpha-clusters.txt"},
                },
            }
        }
    )

    datasets = Runner.build_datasets(_fake_runner(mixture))  # type: ignore[arg-type]

    composite = datasets["train"]
    assert isinstance(composite, MixtureStructureDataset)
    assert tuple(composite.datasets) == ("alpha", "zeta")
    assert [call[1][0] for call in calls] == ["alpha.cif", "zeta.cif"]
    assert [call[2]["seq_to_pdb_index"] for call in calls] == ["alpha-index.json", "zeta-index.json"]
    assert [call[2]["msa_path"] for call in calls] == ["alpha-msa", "zeta-msa"]
    assert [call[2]["processor_kwargs"] for call in calls] == [{"max_tokens": 48}, {"max_tokens": 96}]
    assert [call[2]["processing_timeout_seconds"] for call in calls] == [30.0, 4.5]
    assert [call[2]["max_consecutive_processing_errors"] for call in calls] == [5, 9]
    assert cache_calls == [("alpha.cif", "alpha-clusters.txt"), ("zeta.cif", "zeta-clusters.txt")]
    assert [call[2]["cluster_indices_file"] for call in calls] == ["cache-alpha.cif", "cache-zeta.cif"]


def test_mixture_component_explicit_nulls_override_training_processing_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def structure_dataset(source: str, *_args: Any, **kwargs: Any) -> _SizedDataset:
        calls.append((source, kwargs))
        return _SizedDataset()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", structure_dataset)
    mixture = DatasetMixtureConfig(
        {
            "datasets": {
                "disabled": {
                    "source": "disabled.cif",
                    "processing_timeout_seconds": None,
                    "max_consecutive_processing_errors": None,
                    "max_tokens": 48,
                },
                "bounded": {
                    "source": "bounded.cif",
                    "processing_timeout_seconds": 8.5,
                    "max_consecutive_processing_errors": 3,
                    "max_tokens": 96,
                },
            }
        }
    )

    Runner.build_datasets(_fake_runner(mixture))  # type: ignore[arg-type]

    assert [source for source, _kwargs in calls] == ["bounded.cif", "disabled.cif"]
    assert [kwargs["processing_timeout_seconds"] for _source, kwargs in calls] == [8.5, None]
    assert [kwargs["max_consecutive_processing_errors"] for _source, kwargs in calls] == [3, None]
    assert [kwargs["processor_kwargs"] for _source, kwargs in calls] == [
        {"max_tokens": 96},
        {"max_tokens": 48},
    ]


def test_mixture_build_routes_ordinary_and_cluster_children_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kinds: list[tuple[str, str]] = []

    def ordinary(source: str, *_args: Any, **_kwargs: Any) -> _SizedDataset:
        kinds.append((source, "ordinary"))
        return _SizedDataset()

    def clustered(source: str, *_args: Any, **_kwargs: Any) -> _SizedDataset:
        kinds.append((source, "clustered"))
        return _SizedDataset()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", ordinary)
    monkeypatch.setattr("ofoldx.runners.dataset.ClusteredStructureDataset", clustered)
    mixture = DatasetMixtureConfig(
        {
            "datasets": {
                "plain": {"source": "plain.cif"},
                "cluster": {
                    "source": "cluster.cif",
                    "cluster_sampling": {
                        "enabled": True,
                        "cluster_file": "clusters.txt",
                        "indices_file": "indices.jsonl",
                    },
                },
            }
        }
    )

    Runner.build_datasets(_fake_runner(mixture))  # type: ignore[arg-type]

    assert kinds == [("cluster.cif", "clustered"), ("plain.cif", "ordinary")]


def test_mixture_is_rejected_on_validation_before_building_children(monkeypatch: pytest.MonkeyPatch) -> None:
    mixture = DatasetMixtureConfig({"datasets": {"only": {"source": "val.cif"}}})
    monkeypatch.setattr(
        Runner,
        "_build_structure_dataset",
        lambda *_args, **_kwargs: pytest.fail("validation mixture child was built"),
    )

    with pytest.raises(ValueError, match="mixtures are only supported for training"):
        Runner.build_datasets(_fake_runner(mixture, split="val"))  # type: ignore[arg-type]


def test_training_mixture_does_not_change_ordinary_validation_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: list[str] = []
    processor_kwargs: list[dict[str, Any]] = []

    def ordinary(source: str, *_args: Any, **kwargs: Any) -> _SizedDataset:
        sources.append(source)
        processor_kwargs.append(kwargs["processor_kwargs"])
        return _SizedDataset()

    monkeypatch.setattr("ofoldx.runners.dataset.StructureDataset", ordinary)
    mixture = DatasetMixtureConfig({"datasets": {"train-child": {"source": "train.cif"}}})
    validation = SimpleNamespace(
        source="val.cif",
        cluster_sampling=SimpleNamespace(enabled=False),
        items=lambda: {"source": "val.cif", "processor_seed": 7}.items(),
        get=lambda key, default=None: {"source": "val.cif", "processor_seed": 7}.get(key, default),
    )
    runner = SimpleNamespace(
        processor=lambda structure, **kwargs: (structure, kwargs),
        _process_training=False,
        _forward_cluster_indices=False,
        config=_Config(data={"train": mixture, "val": validation}, seed=31, workspace={"root": "workspace"}),
    )

    datasets = Runner.build_datasets(runner)  # type: ignore[arg-type]

    assert isinstance(datasets["train"], MixtureStructureDataset)
    assert isinstance(datasets["val"], _SizedDataset)
    assert sources == ["train.cif", "val.cif"]
    assert processor_kwargs == [{}, {"processor_seed": 7}]


def test_mixture_sampler_defaults_to_composite_length_and_global_seed() -> None:
    dataset = MixtureStructureDataset({"a": _SizedDataset(2), "b": _SizedDataset(3)}, {"a": 1, "b": 1})
    mixture = DatasetMixtureConfig({"datasets": {"a": {"source": "a"}, "b": {"source": "b"}}})
    runner = SimpleNamespace(
        train_splits=("train",),
        config=_Config(data={"train": mixture}, seed=31),
        distributed=False,
        fault_tolerance=None,
    )

    sampler = Runner.build_datasampler(runner, dataset, split="train", shuffle=True)  # type: ignore[arg-type]

    assert isinstance(sampler, MixtureWeightedSampler)
    assert sampler.num_samples == len(dataset)
    assert sampler.seed == 31
    assert sampler.replacement is True
    assert sampler.num_replicas == 1
    assert sampler.rank == 0


def test_mixture_sampler_reuses_fault_tolerant_data_parallel_topology() -> None:
    dataset = MixtureStructureDataset({"a": _SizedDataset(6), "b": _SizedDataset(6)}, {"a": 1, "b": 1})
    mixture = DatasetMixtureConfig(
        {
            "datasets": {"a": {"source": "a"}, "b": {"source": "b"}},
            "epoch_size": 5,
            "replacement": False,
            "seed": 17,
        }
    )
    topology_calls: list[tuple[int, int]] = []

    def data_parallel_info(num_replicas: int, rank: int) -> tuple[int, int]:
        topology_calls.append((num_replicas, rank))
        return 3, 2

    runner = SimpleNamespace(
        train_splits=("train",),
        config=_Config(data={"train": mixture}, seed=31),
        distributed=True,
        world_size=8,
        rank=5,
        fault_tolerance=SimpleNamespace(data_parallel_info=data_parallel_info),
    )

    sampler = Runner.build_datasampler(runner, dataset, split="train", shuffle=False)  # type: ignore[arg-type]

    assert topology_calls == [(8, 5)]
    assert sampler.num_samples == 5
    assert sampler.seed == 17
    assert sampler.replacement is False
    assert sampler.num_replicas == 3
    assert sampler.rank == 2


def test_mixture_sampler_defensively_rejects_nontraining_split() -> None:
    dataset = MixtureStructureDataset({"only": _SizedDataset()}, {"only": 1})
    runner = SimpleNamespace(train_splits=("train",))

    with pytest.raises(ValueError, match="mixtures are only supported for training"):
        Runner.build_datasampler(runner, dataset, split="val", shuffle=False)  # type: ignore[arg-type]
