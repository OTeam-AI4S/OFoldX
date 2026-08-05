from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch.utils.data import Dataset

from ofoldx.processor import UnresolvedTrainingCropError
from ofoldx.runners import MixtureStructureDataset, MixtureWeightedSampler
from ofoldx.runners.dataset import (
    ClusteredStructureDataset,
    ConsecutiveDataLoadingErrors,
    ProcessingTimeoutExhaustionError,
    StructureDataset,
    StructureProcessingTimeoutError,
)
from ofoldx.runners.multidataset import MixtureProcessingTimeoutExhaustionError


class Values(Dataset[str]):
    def __init__(self, values: Sequence[str]) -> None:
        self.values = list(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> str:
        return self.values[index]


class TimeoutValues(Dataset[str]):
    def __init__(self, name: str, size: int = 1) -> None:
        self.name = name
        self.size = size
        self.calls: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        self.calls.append(index)
        cause = StructureProcessingTimeoutError(f"{self.name} timed out", stage="read_structure")
        error = ProcessingTimeoutExhaustionError(
            f"{self.name} exhausted",
            timeout_count=1,
            scanned_candidates=1,
            processed_attempts=1,
        )
        raise error from cause


def clustered(values: Sequence[str], weights: Sequence[float]) -> ClusteredStructureDataset:
    dataset = ClusteredStructureDataset.__new__(ClusteredStructureDataset)
    dataset.entries = list(values)
    dataset.weights = list(weights)
    dataset._process_logical_index = lambda index: values[index]  # type: ignore[method-assign]
    return dataset


def test_mixture_sorts_names_routes_indices_and_supports_negative_indices() -> None:
    mixture = MixtureStructureDataset(
        {"zeta": Values(["z0"]), "alpha": Values(["a0", "a1"])},
        {"zeta": 1.0, "alpha": 1.0},
    )
    assert mixture.dataset_names == ("alpha", "zeta")
    assert list(mixture.datasets) == ["alpha", "zeta"]
    assert mixture.offsets == (0, 2, 3)
    assert [mixture[index] for index in range(len(mixture))] == ["a0", "a1", "z0"]
    assert mixture[-1] == "z0"
    assert mixture[-3] == "a0"
    with pytest.raises(IndexError):
        _ = mixture[3]
    with pytest.raises(IndexError):
        _ = mixture[-4]


def test_mixture_preserves_selected_child_timeout_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    child = StructureDataset(None, lambda value: value, processing_timeout_seconds=1.0)
    child.records = ["slow.cif", "success.cif"]
    calls: list[int] = []

    def process(index: int) -> str:
        calls.append(index)
        if index == 0:
            raise StructureProcessingTimeoutError("slow child candidate")
        return "success"

    monkeypatch.setattr(child, "_process_logical_index", process)
    mixture = MixtureStructureDataset({"child": child}, {"child": 1.0})

    assert mixture[0] == "success"
    assert calls == [0, 1]


def test_mixture_timeout_exhaustion_falls_back_to_next_component() -> None:
    primary = TimeoutValues("a")
    secondary = Values(["b-success"])
    mixture = MixtureStructureDataset({"a": primary, "b": secondary}, {"a": 4.0, "b": 1.0})
    original_weights = mixture.sampling_weights.clone()

    assert mixture[0] == "b-success"
    assert primary.calls == [0]
    assert torch.equal(mixture.sampling_weights, original_weights)


def test_mixture_tolerance_enabled_timeout_exhaustion_still_falls_back_to_next_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = StructureDataset(None, lambda value: value, max_consecutive_processing_errors=5)
    primary.records = ["slow.cif"]
    calls: list[int] = []

    def timeout(index: int, **kwargs: object) -> None:
        calls.append(index)
        raise StructureProcessingTimeoutError("slow child candidate", stage="read_structure")

    monkeypatch.setattr(primary, "_process_record", timeout)
    mixture = MixtureStructureDataset(
        {"a": primary, "b": Values(["b-success"])},
        {"a": 1.0, "b": 1.0},
    )

    assert mixture[0] == "b-success"
    assert calls == [0]


def test_mixture_fifth_tolerance_enabled_timeout_is_terminal_without_cross_corpus_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = StructureDataset(None, lambda value: value, max_consecutive_processing_errors=5)
    primary.records = [f"slow-{index}.cif" for index in range(6)]
    calls: list[int] = []
    fallback_calls: list[int] = []

    def timeout(index: int, **kwargs: object) -> None:
        calls.append(index)
        raise StructureProcessingTimeoutError(f"slow child candidate {index}", stage="read_structure")

    class RejectFallback(Values):
        def __getitem__(self, index: int) -> str:
            fallback_calls.append(index)
            raise AssertionError("mixture must not cross corpora after the fifth timeout")

    monkeypatch.setattr(primary, "_process_record", timeout)
    mixture = MixtureStructureDataset(
        {"a": primary, "b": RejectFallback(["unused"])},
        {"a": 1.0, "b": 1.0},
    )

    with pytest.raises(ConsecutiveDataLoadingErrors) as exc_info:
        mixture[0]
    assert calls == [0, 1, 2, 3, 4]
    assert type(exc_info.value.__cause__) is StructureProcessingTimeoutError
    assert str(exc_info.value.__cause__) == "slow child candidate 4"
    assert fallback_calls == []


def test_mixture_timeout_fallback_is_sorted_cyclic_and_wraps_local_index() -> None:
    alpha = Values(["a0", "a1"])
    beta = TimeoutValues("beta", size=3)
    gamma = TimeoutValues("gamma", size=2)
    mixture = MixtureStructureDataset(
        {"gamma": gamma, "alpha": alpha, "beta": beta},
        {"gamma": 1.0, "alpha": 1.0, "beta": 1.0},
    )
    beta_local_index = 2
    beta_global_index = mixture.offsets[1] + beta_local_index

    assert mixture[beta_global_index] == "a0"
    assert beta.calls == [2]
    assert gamma.calls == [0]


def test_mixture_child_internal_timeout_success_does_not_cross_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = StructureDataset(None, lambda value: value)
    primary.records = ["slow.cif", "success.cif"]
    calls: list[int] = []

    def process(index: int) -> str:
        calls.append(index)
        if index == 0:
            raise StructureProcessingTimeoutError("row timeout", stage="processor")
        return "internal-success"

    monkeypatch.setattr(primary, "_process_logical_index", process)

    class RejectFallback(Values):
        def __getitem__(self, index: int) -> str:
            raise AssertionError("mixture fallback must not run after internal child success")

    mixture = MixtureStructureDataset(
        {"a": primary, "b": RejectFallback(["unused"])},
        {"a": 1.0, "b": 1.0},
    )
    assert mixture[0] == "internal-success"
    assert calls == [0, 1]


def test_mixture_all_timeout_exhaustion_is_bounded_and_preserves_components() -> None:
    alpha = TimeoutValues("alpha", size=2)
    beta = TimeoutValues("beta", size=3)
    gamma = TimeoutValues("gamma", size=1)
    mixture = MixtureStructureDataset(
        {"alpha": alpha, "beta": beta, "gamma": gamma},
        {"alpha": 1.0, "beta": 1.0, "gamma": 1.0},
    )
    primary_local_index = 2
    primary_global_index = mixture.offsets[1] + primary_local_index

    with pytest.raises(MixtureProcessingTimeoutExhaustionError) as exc_info:
        mixture[primary_global_index]

    assert beta.calls == [2]
    assert gamma.calls == [0]
    assert alpha.calls == [0]
    assert exc_info.value.component_names == ("beta", "gamma", "alpha")
    assert set(exc_info.value.component_errors) == {"alpha", "beta", "gamma"}
    assert exc_info.value.timeout_count == 3
    assert exc_info.value.scanned_candidates == 3
    assert exc_info.value.processed_attempts == 3
    assert exc_info.value.__cause__ is exc_info.value.component_errors["alpha"]


@pytest.mark.parametrize(
    "ordinary_error",
    [ValueError("ordinary failure"), UnresolvedTrainingCropError("ordinary crop exhaustion")],
)
def test_mixture_does_not_swallow_ordinary_child_errors(ordinary_error: Exception) -> None:
    calls: list[int] = []

    class OrdinaryFailure(Dataset[str]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> str:
            calls.append(index)
            raise ordinary_error

    fallback = TimeoutValues("fallback")
    mixture = MixtureStructureDataset(
        {"a": OrdinaryFailure(), "b": fallback},
        {"a": 1.0, "b": 1.0},
    )

    with pytest.raises(type(ordinary_error), match=str(ordinary_error)):
        mixture[0]
    assert calls == [0]
    assert fallback.calls == []


def test_mixture_does_not_swallow_consecutive_data_loading_terminal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = StructureDataset(None, lambda value: value, max_consecutive_processing_errors=5)
    primary.records = ["bad.cif"]
    monkeypatch.setattr(
        primary,
        "_process_record",
        lambda index, **kwargs: (_ for _ in ()).throw(UnresolvedTrainingCropError("bad row")),
    )
    fallback_calls: list[int] = []

    class RejectFallback(Values):
        def __getitem__(self, index: int) -> str:
            fallback_calls.append(index)
            raise AssertionError("mixture must not cross corpora after the child reaches its data-error terminal")

    mixture = MixtureStructureDataset(
        {"a": primary, "b": RejectFallback(["unused"])},
        {"a": 1.0, "b": 1.0},
    )

    with pytest.raises(ConsecutiveDataLoadingErrors) as exc_info:
        mixture[0]
    assert type(exc_info.value.__cause__) is UnresolvedTrainingCropError
    assert fallback_calls == []


def test_mixture_normalizes_each_corpus_and_cluster_rows() -> None:
    mixture = MixtureStructureDataset(
        {"plain": Values(["p0", "p1"]), "cluster": clustered(["c0", "c1"], [1.0, 3.0])},
        {"plain": 1.0, "cluster": 3.0},
    )
    assert mixture.sampling_weights.tolist() == pytest.approx([3 / 16, 9 / 16, 1 / 8, 1 / 8])
    assert mixture.sampling_weights[:2].sum().item() == pytest.approx(3 / 4)
    assert mixture.sampling_weights[2:].sum().item() == pytest.approx(1 / 4)


def test_ordinary_corpus_mass_is_independent_of_child_size() -> None:
    short = MixtureStructureDataset({"a": Values(["a"]), "b": Values(["b"])}, {"a": 2.0, "b": 1.0})
    long = MixtureStructureDataset(
        {"a": Values(["a0", "a1", "a2", "a3"]), "b": Values(["b"])},
        {"a": 2.0, "b": 1.0},
    )
    assert short.sampling_weights[:1].sum() == pytest.approx(2 / 3)
    assert long.sampling_weights[:4].sum() == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("datasets", "ratios", "match"),
    [
        ({}, {}, "non-empty"),
        ({"a": Values([])}, {"a": 1.0}, "non-empty"),
        ({"a": Values(["a"])}, {"b": 1.0}, "identical names"),
        ({"a": Values(["a"])}, {"a": 0.0}, "positive"),
        ({"a": Values(["a"])}, {"a": float("nan")}, "finite"),
    ],
)
def test_mixture_rejects_invalid_children_and_ratios(datasets, ratios, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MixtureStructureDataset(datasets, ratios)


@pytest.mark.parametrize("weights", ([0.0, 0.0], [1.0, float("inf")], [1.0]))
def test_mixture_rejects_invalid_cluster_local_mass(weights) -> None:
    values = [f"c{index}" for index in range(2)]
    if len(weights) == 1:
        values = ["c0", "c1"]
    with pytest.raises(ValueError, match="weights"):
        MixtureStructureDataset({"cluster": clustered(values, weights)}, {"cluster": 1.0})


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([], "non-empty"),
        ([[1.0]], "one-dimensional"),
        ([float("nan")], "finite"),
        ([-1.0, 2.0], "non-negative"),
        ([0.0, 0.0], "positive sum"),
    ],
)
def test_sampler_rejects_invalid_weights(weights, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MixtureWeightedSampler(weights, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": 1, "num_replicas": 0}, "num_replicas"),
        ({"num_samples": 1, "num_replicas": 2, "rank": 2}, "rank"),
        ({"num_samples": 1.5}, "integers"),
    ],
)
def test_sampler_rejects_invalid_sampling_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MixtureWeightedSampler([1.0], **kwargs)


@pytest.mark.parametrize("replacement", [0, 1, None, "yes"])
def test_sampler_requires_boolean_replacement(replacement) -> None:
    with pytest.raises(TypeError, match="boolean"):
        MixtureWeightedSampler([1.0], 1, replacement=replacement)


def test_sampler_seed_epoch_and_state_are_reproducible() -> None:
    first = MixtureWeightedSampler([1.0, 3.0, 2.0], 100, seed=41)
    second = MixtureWeightedSampler([1.0, 3.0, 2.0], 100, seed=41)
    assert list(first) == list(second)
    first.set_epoch(7)
    assert list(first) != list(second)
    restored = MixtureWeightedSampler([1.0, 3.0, 2.0], 100, seed=41)
    restored.load_state_dict(first.state_dict())
    assert list(restored) == list(first)


def test_sampler_iterator_state_restores_exact_local_remainder() -> None:
    sampler = MixtureWeightedSampler([1.0, 4.0, 2.0], 13, seed=17, num_replicas=3, rank=1)
    expected = list(sampler)
    iterator = iter(sampler)
    assert [next(iterator), next(iterator)] == expected[:2]
    state = iterator.state_dict()

    sampler.set_epoch(9)
    restored_iterator = iter(sampler)
    restored_iterator.load_state_dict(state)
    assert list(restored_iterator) == expected[2:]
    assert sampler.epoch == 0


@pytest.mark.parametrize("cursor", [-1, 6])
def test_sampler_iterator_rejects_invalid_cursor(cursor: int) -> None:
    iterator = iter(MixtureWeightedSampler([1.0], 5))
    with pytest.raises(ValueError, match="cursor"):
        iterator.load_state_dict({"epoch": 0, "cursor": cursor})


def test_stateful_dataloader_restores_exact_sampler_remainder() -> None:
    stateful_dataloader = pytest.importorskip("torchdata.stateful_dataloader")
    dataset = torch.arange(20)
    sampler = MixtureWeightedSampler(torch.arange(1, 21, dtype=torch.double), 11, seed=29)
    loader = stateful_dataloader.StatefulDataLoader(dataset, batch_size=3, sampler=sampler, num_workers=0)
    iterator = iter(loader)
    consumed = [next(iterator).tolist(), next(iterator).tolist()]
    state = loader.state_dict()
    expected_remainder = [batch.tolist() for batch in iterator]

    restored_sampler = MixtureWeightedSampler(torch.arange(1, 21, dtype=torch.double), 11, seed=29)
    restored_loader = stateful_dataloader.StatefulDataLoader(
        dataset, batch_size=3, sampler=restored_sampler, num_workers=0
    )
    restored_loader.load_state_dict(state)

    assert len(consumed) == 2
    assert [batch.tolist() for batch in restored_loader] == expected_remainder


def test_sampler_ddp_ranks_reconstruct_one_padded_global_draw() -> None:
    weights = torch.tensor([1.0, 4.0, 2.0], dtype=torch.double)
    samplers = [MixtureWeightedSampler(weights, 5, seed=13, num_replicas=3, rank=rank) for rank in range(3)]
    generator = torch.Generator().manual_seed(13)
    global_indices = torch.multinomial(weights, 6, True, generator=generator).tolist()
    assert [len(sampler) for sampler in samplers] == [2, 2, 2]
    assert [list(sampler) for sampler in samplers] == [global_indices[rank::3] for rank in range(3)]


def test_sampler_nonreplacement_checks_padded_capacity() -> None:
    with pytest.raises(ValueError, match="total_size"):
        MixtureWeightedSampler([1.0, 1.0, 0.0], 2, replacement=False, num_replicas=3)
    sampler = MixtureWeightedSampler([1.0, 1.0, 1.0], 2, replacement=False, num_replicas=3)
    assert len(sampler) == 1
