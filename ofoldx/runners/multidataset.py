r"""Weighted composition of independently indexed structure datasets."""

from __future__ import annotations

import math
import operator
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from .dataset import ClusteredStructureDataset, ProcessingTimeoutExhaustionError


class MixtureProcessingTimeoutExhaustionError(ProcessingTimeoutExhaustionError):
    r"""Raised after every mixture component in one bounded timeout fallback is exhausted."""

    def __init__(self, component_errors: Mapping[str, ProcessingTimeoutExhaustionError]) -> None:
        self.component_names = tuple(component_errors)
        self.component_errors = dict(component_errors)
        super().__init__(
            "all mixture components exhausted after processing timeouts during bounded fallback "
            f"(components={self.component_names!r})",
            timeout_count=sum(error.timeout_count for error in component_errors.values()),
            scanned_candidates=sum(error.scanned_candidates for error in component_errors.values()),
            processed_attempts=sum(error.processed_attempts for error in component_errors.values()),
        )


class _MixtureWeightedSamplerIterator(Iterator[int]):
    r"""Stateful iterator over one rank's deterministic draw for an epoch."""

    def __init__(self, sampler: MixtureWeightedSampler) -> None:
        self.sampler = sampler
        self.epoch = sampler.epoch
        self.cursor = 0
        self._indices = self._draw_indices()

    def _draw_indices(self) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.sampler.seed + self.epoch)
        global_indices = torch.multinomial(
            self.sampler.weights,
            self.sampler.total_size,
            self.sampler.replacement,
            generator=generator,
        ).tolist()
        return global_indices[self.sampler.rank :: self.sampler.num_replicas]

    def __iter__(self) -> _MixtureWeightedSamplerIterator:
        return self

    def __next__(self) -> int:
        if self.cursor >= len(self._indices):
            raise StopIteration
        index = self._indices[self.cursor]
        self.cursor += 1
        return index

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        try:
            epoch = operator.index(state_dict["epoch"])
            cursor = operator.index(state_dict["cursor"])
        except (KeyError, TypeError) as error:
            raise ValueError("invalid mixture sampler iterator state_dict") from error
        if cursor < 0 or cursor > len(self):
            raise ValueError("mixture sampler iterator cursor is out of range")
        self.epoch = epoch
        self.sampler.epoch = epoch
        self._indices = self._draw_indices()
        self.cursor = cursor

    def __len__(self) -> int:
        return self.sampler.num_samples_per_replica


class MixtureStructureDataset(Dataset[Any]):
    r"""Flatten named datasets while preserving a configured mass per corpus.

    Each corpus is normalized independently before its (also normalized) mixture
    ratio is applied. Consequently, adding records to an ordinary dataset does
    not change that dataset's total sampling probability.

    Args:
        datasets: Child datasets keyed by their stable corpus names.
        sampling_ratios: Positive mixture ratios keyed by the same corpus names.
    """

    def __init__(self, datasets: Mapping[str, Dataset[Any]], sampling_ratios: Mapping[str, float]) -> None:
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError("datasets must be a non-empty mapping")
        if not isinstance(sampling_ratios, Mapping):
            raise ValueError("sampling_ratios must be a mapping")
        if any(not isinstance(name, str) for name in datasets) or any(
            not isinstance(name, str) for name in sampling_ratios
        ):
            raise ValueError("dataset names must be strings")
        if set(datasets) != set(sampling_ratios):
            missing = sorted(set(datasets) - set(sampling_ratios))
            extra = sorted(set(sampling_ratios) - set(datasets))
            raise ValueError(f"datasets and ratios must have identical names (missing={missing}, extra={extra})")
        self.dataset_names = tuple(sorted(datasets))
        self.datasets = {name: datasets[name] for name in self.dataset_names}
        self._dataset_values = tuple(self.datasets.values())

        ratio_values: list[float] = []
        for name in self.dataset_names:
            if isinstance(sampling_ratios[name], bool):
                raise ValueError(f"ratio for dataset {name!r} must be finite and positive")
            try:
                ratio = float(sampling_ratios[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"ratio for dataset {name!r} must be numeric") from error
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"ratio for dataset {name!r} must be finite and positive")
            ratio_values.append(ratio)
        ratio_total = sum(ratio_values)
        if not math.isfinite(ratio_total) or ratio_total <= 0:
            raise ValueError("dataset ratios must have a finite positive sum")

        offsets = [0]
        sampling_weights: list[float] = []
        for name, dataset, ratio in zip(self.dataset_names, self._dataset_values, ratio_values):
            child_size = len(dataset)
            if child_size <= 0:
                raise ValueError(f"dataset {name!r} must be non-empty")

            if isinstance(dataset, ClusteredStructureDataset):
                try:
                    local_weights = torch.as_tensor(dataset.weights, dtype=torch.double)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"weights for dataset {name!r} must be a numeric sequence") from error
                if local_weights.ndim != 1 or len(local_weights) != child_size:
                    raise ValueError(f"weights for dataset {name!r} must be one-dimensional and match its length")
            else:
                local_weights = torch.ones(child_size, dtype=torch.double)

            if not torch.isfinite(local_weights).all():
                raise ValueError(f"weights for dataset {name!r} must be finite")
            if torch.any(local_weights < 0):
                raise ValueError(f"weights for dataset {name!r} must be non-negative")
            local_total = float(local_weights.sum())
            if not math.isfinite(local_total) or local_total <= 0:
                raise ValueError(f"weights for dataset {name!r} must have a positive finite sum")

            corpus_mass = ratio / ratio_total
            sampling_weights.extend((local_weights / local_total * corpus_mass).tolist())
            offsets.append(offsets[-1] + child_size)

        self.offsets = tuple(offsets)
        self.sampling_weights = torch.tensor(sampling_weights, dtype=torch.double)

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> Any:
        logical_index = operator.index(index)
        if logical_index < 0:
            logical_index += len(self)
        if logical_index < 0 or logical_index >= len(self):
            raise IndexError("dataset index out of range")
        dataset_index = bisect_right(self.offsets, logical_index) - 1
        local_index = logical_index - self.offsets[dataset_index]
        try:
            return self._dataset_values[dataset_index][local_index]
        except ProcessingTimeoutExhaustionError as primary_error:
            component_errors = {self.dataset_names[dataset_index]: primary_error}

        for offset in range(1, len(self._dataset_values)):
            fallback_dataset_index = (dataset_index + offset) % len(self._dataset_values)
            fallback_dataset = self._dataset_values[fallback_dataset_index]
            fallback_local_index = local_index % len(fallback_dataset)
            try:
                return fallback_dataset[fallback_local_index]
            except ProcessingTimeoutExhaustionError as error:
                component_errors[self.dataset_names[fallback_dataset_index]] = error

        mixture_error = MixtureProcessingTimeoutExhaustionError(component_errors)
        raise mixture_error from next(reversed(component_errors.values()))


class MixtureWeightedSampler(Sampler[int]):
    r"""Draw global mixture indices reproducibly and stride them across ranks."""

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        num_samples: int,
        *,
        replacement: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        try:
            normalized_num_samples = operator.index(num_samples)
            normalized_num_replicas = operator.index(num_replicas)
            normalized_rank = operator.index(rank)
        except TypeError as error:
            raise ValueError("num_samples, num_replicas, and rank must be integers") from error
        try:
            weight_tensor = torch.as_tensor(weights, dtype=torch.double)
        except (TypeError, ValueError) as error:
            raise ValueError("weights must be a one-dimensional numeric sequence") from error
        if weight_tensor.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if len(weight_tensor) == 0:
            raise ValueError("weights must be non-empty")
        if not torch.isfinite(weight_tensor).all():
            raise ValueError("weights must be finite")
        if torch.any(weight_tensor < 0):
            raise ValueError("weights must be non-negative")
        weight_total = float(weight_tensor.sum())
        if not math.isfinite(weight_total) or weight_total <= 0:
            raise ValueError("weights must have a finite positive sum")
        if normalized_num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if normalized_num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= normalized_rank < normalized_num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if not isinstance(replacement, bool):
            raise TypeError(f"replacement must be a boolean, got {replacement!r}")

        self.weights = weight_tensor
        self.num_samples = normalized_num_samples
        self.replacement = replacement
        self.seed = int(seed)
        self.num_replicas = normalized_num_replicas
        self.rank = normalized_rank
        self.epoch = 0
        self.num_samples_per_replica = math.ceil(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas
        if not self.replacement and int(torch.count_nonzero(self.weights > 0)) < self.total_size:
            raise ValueError("non-replacement sampling requires at least total_size positive-weight indices")

    def __iter__(self) -> Iterator[int]:
        return _MixtureWeightedSamplerIterator(self)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = int(state_dict["epoch"])
