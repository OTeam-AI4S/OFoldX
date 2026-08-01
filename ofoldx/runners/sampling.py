r"""Deterministic weighted sampling over an explicit dataset-entry axis."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import Sampler


class SamplingEntry(NamedTuple):
    r"""One weighted structural view identified by its source path."""

    structure: str
    sample_type: str
    instance_ids: tuple[str, ...]
    weight: float


class SampleKey(NamedTuple):
    r"""Identify both the selected entry and its global draw across epochs."""

    entry_index: int
    draw_index: int


def read_sampling_index(
    path: str | os.PathLike[str],
) -> tuple[SamplingEntry, ...]:
    r"""Read explicit structure, chain, or interface views from JSON lines.

    Relative structure paths are resolved from the index directory.
    """

    path = Path(path)
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
                structure = payload["structure"]
                sample_type = payload["sample_type"]
                instance_ids = payload.get("instance_ids", [])
                weight = payload["weight"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid sampling index row {line_number} in {path}") from error

            if not isinstance(structure, str):
                raise ValueError(f"sampling index row {line_number} has an invalid structure path")
            structure_path = Path(structure).expanduser()
            if not structure_path.is_absolute():
                structure_path = path.parent / structure_path
            if not isinstance(instance_ids, list):
                raise ValueError(f"sampling index row {line_number} instance_ids must be a list")
            _validate_sampling_view(sample_type, instance_ids, f"sampling index row {line_number}")
            if not isinstance(weight, int | float) or isinstance(weight, bool):
                raise ValueError(f"sampling index row {line_number} has an invalid weight")
            weight = float(weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"sampling index row {line_number} weight must be finite and non-negative")
            entries.append(SamplingEntry(str(structure_path), sample_type, tuple(instance_ids), weight))
    return tuple(entries)


def _validate_sampling_view(
    sample_type: str,
    instance_ids: Sequence[str],
    context: str,
) -> None:
    expected = {"structure": 0, "chain": 1, "interface": 2}
    if not isinstance(sample_type, str) or sample_type not in expected:
        raise ValueError(f"{context} sample_type must be structure, chain, or interface")
    if not all(isinstance(instance_id, str) for instance_id in instance_ids):
        raise ValueError(f"{context} instance_ids must be strings")
    count = expected[sample_type]
    if len(instance_ids) != count:
        raise ValueError(f"{context} {sample_type} view requires {count} instance_ids")
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError(f"{context} instance_ids must be distinct")


class WeightedEpochSampler(Sampler[SampleKey]):
    r"""Draw deterministic weighted entry ordinals for one distributed epoch.

    ``epoch_size`` is the requested global number of draws. Distributed sampling
    pads it to a multiple of ``num_replicas`` so that every rank takes the same
    number of steps, as :class:`torch.utils.data.DistributedSampler` does.
    """

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        epoch_size: int,
        *,
        replacement: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
        if self.weights.ndim != 1 or not self.weights.numel():
            raise ValueError("weights must be a non-empty one-dimensional sequence")
        if not bool(torch.isfinite(self.weights).all()) or bool((self.weights < 0).any()):
            raise ValueError("weights must be finite and non-negative")
        if float(self.weights.sum()) <= 0:
            raise ValueError("weights must have a positive sum")
        if epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("rank must identify one of num_replicas")

        self.epoch_size = int(epoch_size)
        self.replacement = replacement
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self._offset = 0
        self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if not replacement and int(torch.count_nonzero(self.weights)) < self.epoch_size:
            raise ValueError("sampling without replacement requires one positive-weight entry per draw")

    def __iter__(self) -> Iterator[SampleKey]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        entry_indices = torch.multinomial(
            self.weights,
            self.epoch_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        if len(entry_indices) < self.total_size:
            padding = self.total_size - len(entry_indices)
            entry_indices.extend((entry_indices * math.ceil(padding / len(entry_indices)))[:padding])
        draw_offset = self.epoch * self.total_size
        draw_indices = range(self.rank, self.total_size, self.num_replicas)
        for position, draw_index in enumerate(draw_indices):
            if position < self._offset:
                continue
            self._offset = position + 1
            yield SampleKey(entry_indices[draw_index], draw_offset + draw_index)
        self.epoch += 1
        self._offset = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Select the deterministic random stream for an epoch."""

        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self._offset = 0

    def state_dict(self) -> dict[str, int]:
        r"""Return the epoch and local draw offset needed to resume exactly."""

        return {"epoch": self.epoch, "offset": self._offset}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        r"""Restore an exact point in the deterministic local stream."""

        self.epoch = int(state_dict["epoch"])
        self._offset = int(state_dict["offset"])
