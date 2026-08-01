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

r"""Lazy structure datasets backed by one family Processor operation."""

from __future__ import annotations

import glob
import json
import operator
import os
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.crop import Cropper, instance_anchored_crop, interface_anchored_crop
from ..errors import RequestRef, fail
from ..io import load_structure
from ..msa import MSA, IndexedMsaLoader, read_msa
from ..pipelines.input import entity_pools
from .sampling import SampleKey, SamplingEntry, validate_sampling_view

_GLOB_MAGIC = ("*", "?", "[")
_NATIVE_SUFFIX = ".safetensors"
_PARSED_SUFFIXES = (".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
_RECORD_KEYS = frozenset({"structure", "target", "msa"})
_DatasetItem = tuple[Any, int, str]


def collate_with_context(
    batch: list[_DatasetItem],
    *,
    collate_fn: Callable[[list[Any]], Any],
    forward: Mapping[str, Any] | None = None,
) -> Any:
    r"""Collate values, add static model arguments, and retain the source of every failed sample."""

    try:
        collated = collate_fn([value for value, _index, _path in batch])
        if forward:
            input = collated["input"]
            if isinstance(input, Mapping):
                collated["input"] = {**input, **forward}
            else:
                collated["input"] = {"input": input, **forward}
        return collated
    except Exception as error:
        error.add_note(
            "dataset samples: " + "; ".join(f"index={index}, structure={path}" for _value, index, path in batch)
        )
        raise


class StructureDataset(Dataset):
    r"""
    Read and process structures lazily from one explicit source.

    A sample's MSA pools come from its manifest record (``msa``: ``path`` and ``entity_id``, bound to every instance
    of that entity) or else from the indexed layout under ``msa_root`` (``IndexedMsaLoader``). Structures are loaded
    from native artifacts (``.safetensors``, ``load_structure``) and mmCIF or PDB is never parsed: a directory or glob
    source collects the artifacts it holds, and a source or manifest path naming another file is refused
    (``dataset_not_native``).
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        processor: Callable[..., Any],
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
        msa_index: Mapping[str, str | Sequence[str]] | str | os.PathLike[str] | None = None,
        msa_root: str | os.PathLike[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.records = _resolve_records(source)
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        if (msa_index is None) != (msa_root is None):
            raise ValueError("msa_index and msa_root must be configured together")
        self._msa_loader = (
            None if msa_index is None else IndexedMsaLoader(msa_index, cast(str | os.PathLike[str], msa_root))
        )
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        r"""Select the deterministic per-sample random view for one epoch."""

        self._epoch.fill_(epoch)

    def state_dict(self) -> dict[str, int]:
        r"""Return the random-view epoch for dataloader checkpointing."""

        return {"epoch": int(self._epoch)}

    def load_state_dict(self, state_dict: Mapping[str, int]) -> None:
        r"""Restore the random-view epoch."""

        self.set_epoch(state_dict["epoch"])

    def __getitem__(self, index: int) -> _DatasetItem:
        random_index = int(self._epoch) * max(len(self), 1) + index
        return self._process_record(index, random_index=random_index)

    def _process_record(
        self,
        record_index: int,
        *,
        random_index: int,
        processor_kwargs: Mapping[str, Any] | None = None,
        context: str | None = None,
    ) -> _DatasetItem:
        record = self.records[record_index]
        structure_path = record if isinstance(record, str) else record["structure"]
        try:
            structure = load_structure(structure_path)
            msa: Mapping[str, MSA] | None = None
            if not isinstance(record, str) and "msa" in record:
                msa_spec = record["msa"]
                msa = entity_pools(read_msa(msa_spec["path"], entity_id=msa_spec["entity_id"]), structure.system)
            elif self._msa_loader is not None:
                msa = self._msa_loader(structure)
            target = None
            if not isinstance(record, str) and "target" in record:
                target_spec = record["target"]
                target = load_structure(target_spec["path"])
            call_kwargs = self.processor_kwargs
            if processor_kwargs is not None:
                call_kwargs = {**call_kwargs, **processor_kwargs}
            if self.seed is not None:
                call_kwargs = {**call_kwargs, "generator": np.random.default_rng(self.seed + random_index)}
            if target is not None:
                call_kwargs = {**call_kwargs, "target": target}
                if "instance_ids" in target_spec:
                    call_kwargs = {**call_kwargs, "generated_target_instance_ids": target_spec["instance_ids"]}
            if msa is not None:
                call_kwargs = {**call_kwargs, "msa": msa}
            value = self.processor(structure, **call_kwargs)
        except Exception as error:
            error.add_note(f"{context or f'dataset index={record_index}'}, structure={structure_path}")
            raise
        return value, record_index, structure_path


class SampledStructureDataset(StructureDataset):
    r"""Expose explicit weighted structural views on an entry axis."""

    def __init__(
        self,
        source: str | os.PathLike[str],
        processor: Callable[..., Any],
        entries: Sequence[SamplingEntry],
        *,
        num_epoch_samples: int | None = None,
        replacement: bool = True,
        sampling_seed: int = 0,
        processor_kwargs: Mapping[str, Any] | None = None,
        msa_index: Mapping[str, str | Sequence[str]] | str | os.PathLike[str] | None = None,
        msa_root: str | os.PathLike[str] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            source,
            processor,
            processor_kwargs=processor_kwargs,
            msa_index=msa_index,
            msa_root=msa_root,
            seed=seed,
        )
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("sampling entries must not be empty")
        for index, entry in enumerate(self.entries):
            validate_sampling_view(entry.sample_type, entry.instance_ids, f"sampling entry {index}")
        if (
            any(entry.sample_type != "structure" for entry in self.entries)
            and self.processor_kwargs.get("cropper") is None
        ):
            raise ValueError("chain and interface sampling entries require a cropper")
        records_by_path: dict[Path, list[int]] = {}
        for index, record in enumerate(self.records):
            structure_path = record if isinstance(record, str) else record["structure"]
            records_by_path.setdefault(Path(structure_path).resolve(), []).append(index)
        record_indices = []
        for entry in self.entries:
            matches = records_by_path.get(Path(entry.structure).resolve(), [])
            if not matches:
                raise ValueError(f"sampling entry structure is absent from the dataset: {entry.structure}")
            if len(matches) != 1:
                raise ValueError(f"sampling entry structure is ambiguous in the dataset: {entry.structure}")
            record_indices.append(matches[0])
        self.record_indices = tuple(record_indices)
        self.weights: Sequence[float] | torch.Tensor = tuple(entry.weight for entry in self.entries)
        self.num_epoch_samples = len(self.entries) if num_epoch_samples is None else operator.index(num_epoch_samples)
        self.replacement = replacement
        self.sampling_seed = sampling_seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: int | SampleKey) -> _DatasetItem:
        if isinstance(key, SampleKey):
            entry_index = key.entry_index
            random_index = key.draw_index
        else:
            entry_index = operator.index(key)
            if entry_index < 0:
                entry_index += len(self)
            random_index = int(self._epoch) * max(len(self), 1) + entry_index
        if not 0 <= entry_index < len(self):
            raise IndexError("sampling entry index out of range")

        entry = self.entries[entry_index]
        record_index = self.record_indices[entry_index]
        processor_kwargs: dict[str, Any] = {}
        if entry.sample_type != "structure":
            cropper = cast(Cropper, self.processor_kwargs["cropper"])
            if entry.sample_type == "chain":
                processor_kwargs["cropper"] = partial(
                    instance_anchored_crop,
                    cropper=cropper,
                    instances=entry.instance_ids,
                )
            else:
                processor_kwargs["cropper"] = partial(
                    interface_anchored_crop,
                    cropper=cropper,
                    instances=entry.instance_ids,
                )
        return self._process_record(
            record_index,
            random_index=random_index,
            processor_kwargs=processor_kwargs,
            context=f"sampling entry={entry_index}, draw={random_index}, record={record_index}",
        )


def _is_native(path: str | Path) -> bool:
    return str(path).lower().endswith(_NATIVE_SUFFIX)


def _native(path: str, argument: str) -> str:
    if not _is_native(path):
        fail("dataset_not_native", RequestRef(argument, path))
    return path


def _native_files(paths: Sequence[str], source: str) -> list[str]:
    r"""The native artifacts among a directory's or glob's files; structure files alone are refused, never skipped."""
    native = [path for path in paths if _is_native(path)]
    if not native and any(path.lower().endswith(_PARSED_SUFFIXES) for path in paths):
        fail("dataset_not_native", RequestRef("source", source))
    return native


def _resolve_records(
    source: str | os.PathLike[str],
) -> Sequence[str | dict[str, Any]]:
    text = os.fspath(source)
    if any(character in text for character in _GLOB_MAGIC):
        matches = sorted(glob.glob(os.path.expanduser(text), recursive=True))
        return _native_files([path for path in matches if os.path.isfile(path)], text)

    path = Path(text).expanduser()
    if path.is_dir():
        return _native_files([str(entry) for entry in sorted(path.rglob("*")) if entry.is_file()], text)
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if str(path).lower().endswith(_PARSED_SUFFIXES):
        fail("dataset_not_native", RequestRef("source", str(path)))
    if _is_native(path):
        return [str(path)]
    return _read_manifest(path)


def _read_manifest(path: Path) -> list[str | dict[str, Any]]:
    records: list[str | dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("{"):
            records.append(_native(str(_relative_path(line, path.parent)), "structure"))
            continue

        record = json.loads(line)
        unknown = sorted(set(record) - _RECORD_KEYS)
        if unknown:
            raise ValueError(
                f"unknown dataset record key(s) {', '.join(unknown)}; "
                f"a dataset record accepts {', '.join(sorted(_RECORD_KEYS))}"
            )
        record["structure"] = _native(str(_relative_path(os.fspath(record["structure"]), path.parent)), "structure")
        if "target" in record:
            target = record["target"]
            if not isinstance(target, dict) or "path" not in target or set(target) - {"path", "instance_ids"}:
                raise ValueError("a dataset record `target` requires `path` and accepts optional `instance_ids`")
            if "instance_ids" in target:
                instance_ids = target["instance_ids"]
                if not isinstance(instance_ids, list) or not all(
                    isinstance(instance_id, str) for instance_id in instance_ids
                ):
                    raise ValueError("a dataset record `target.instance_ids` must be a list of strings")
            target["path"] = _native(str(_relative_path(os.fspath(target["path"]), path.parent)), "target.path")
        if "msa" in record:
            if not isinstance(record["msa"], dict):
                raise ValueError("a dataset record `msa` must be a JSON object with `path` and `entity_id`")
            record["msa"]["path"] = str(_relative_path(os.fspath(record["msa"]["path"]), path.parent))
        records.append(record)
    return records


def _relative_path(value: str, parent: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path
