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

import torch
from torch.utils.data import Dataset

from ..data.crop import Cropper, instance_anchored_crop, interface_anchored_crop
from ..data.evidence import MSA, MsaCollection
from ..data.structure import STRUCTURE_ARTIFACT_SUFFIX, load_structure
from .sampling import SampleKey, SamplingEntry, _validate_sampling_view

_GLOB_MAGIC = ("*", "?", "[")
_RAW_STRUCTURE_SUFFIXES = (".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
_RECORD_KEYS = frozenset({"structure", "target", "msa", "msa_collection"})
_DatasetItem = tuple[Any, int, str]


def _collate_with_context(
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
    r"""Read and process structures lazily from one explicit source."""

    def __init__(
        self,
        source: str | os.PathLike[str],
        processor: Callable[..., Any],
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
        msa_index: Mapping[str, object] | str | os.PathLike[str] | None = None,
        msa_root: str | os.PathLike[str] | None = None,
        paired_msa_filename: str = "uniref100_hits.a3m",
        unpaired_msa_filename: str = "mmseqs_other_hits.a3m",
        seed: int | None = None,
    ) -> None:
        self.records = _resolve_records(source)
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        if (msa_index is None) != (msa_root is None):
            raise ValueError("msa_index and msa_root must be configured together")
        if msa_index is None:
            self._msa_loader = None
        else:
            from ..io import IndexedMsaLoader

            self._msa_loader = IndexedMsaLoader(
                msa_index,
                cast(str | os.PathLike[str], msa_root),
                paired_filename=paired_msa_filename,
                unpaired_filename=unpaired_msa_filename,
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
        from ..io import decode_msa_collection, read_msa, read_structure

        record = self.records[record_index]
        structure_path = record if isinstance(record, str) else record["structure"]
        try:
            msa: MSA | MsaCollection | None
            if isinstance(record, str):
                msa = None
            elif "msa" in record:
                msa_spec = record["msa"]
                msa = read_msa(msa_spec["path"], entity_id=msa_spec["entity_id"])
            elif "msa_collection" in record:
                payload = json.loads(Path(record["msa_collection"]).read_text(encoding="utf-8"))
                msa = decode_msa_collection(payload)
            else:
                msa = None

            if str(structure_path).endswith(STRUCTURE_ARTIFACT_SUFFIX):
                structure = load_structure(structure_path)
            else:
                structure = read_structure(structure_path)
            target = None
            if not isinstance(record, str) and "target" in record:
                target_spec = record["target"]
                target_path = target_spec["path"]
                if str(target_path).endswith(STRUCTURE_ARTIFACT_SUFFIX):
                    target = load_structure(target_path)
                else:
                    target = read_structure(target_path)
            if msa is None and self._msa_loader is not None:
                msa = self._msa_loader(structure)
            call_kwargs = self.processor_kwargs
            if processor_kwargs is not None:
                call_kwargs = {**call_kwargs, **processor_kwargs}
            if self.seed is not None:
                generator = torch.Generator()
                generator.manual_seed(self.seed + random_index)
                call_kwargs = {**call_kwargs, "generator": generator}
            if target is not None:
                call_kwargs = {
                    **call_kwargs,
                    "target": target,
                    "generated_target_instance_ids": target_spec["instance_ids"],
                }
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
        epoch_size: int | None = None,
        replacement: bool = True,
        sampling_seed: int = 0,
        processor_kwargs: Mapping[str, Any] | None = None,
        msa_index: Mapping[str, object] | str | os.PathLike[str] | None = None,
        msa_root: str | os.PathLike[str] | None = None,
        paired_msa_filename: str = "uniref100_hits.a3m",
        unpaired_msa_filename: str = "mmseqs_other_hits.a3m",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            source,
            processor,
            processor_kwargs=processor_kwargs,
            msa_index=msa_index,
            msa_root=msa_root,
            paired_msa_filename=paired_msa_filename,
            unpaired_msa_filename=unpaired_msa_filename,
            seed=seed,
        )
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("sampling entries must not be empty")
        for index, entry in enumerate(self.entries):
            _validate_sampling_view(entry.sample_type, entry.instance_ids, f"sampling entry {index}")
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
        self.weights = tuple(entry.weight for entry in self.entries)
        self.epoch_size = len(self.entries) if epoch_size is None else operator.index(epoch_size)
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
                    instance_ids=entry.instance_ids,
                )
            else:
                processor_kwargs["cropper"] = partial(
                    interface_anchored_crop,
                    cropper=cropper,
                    instance_ids=entry.instance_ids,
                )
        return self._process_record(
            record_index,
            random_index=random_index,
            processor_kwargs=processor_kwargs,
            context=f"sampling entry={entry_index}, draw={random_index}, record={record_index}",
        )


def _is_structure(path: str | Path) -> bool:
    value = str(path)
    return value.endswith(STRUCTURE_ARTIFACT_SUFFIX) or value.lower().endswith(_RAW_STRUCTURE_SUFFIXES)


def _resolve_records(
    source: str | os.PathLike[str],
) -> Sequence[str | dict[str, Any]]:
    text = os.fspath(source)
    if any(character in text for character in _GLOB_MAGIC):
        return [
            path
            for path in sorted(glob.glob(os.path.expanduser(text), recursive=True))
            if os.path.isfile(path) and _is_structure(path)
        ]

    path = Path(text).expanduser()
    if path.is_dir():
        return [str(entry) for entry in sorted(path.rglob("*")) if entry.is_file() and _is_structure(entry)]
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if _is_structure(path):
        return [str(path)]
    return _read_manifest(path)


def _read_manifest(path: Path) -> list[str | dict[str, Any]]:
    records: list[str | dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("{"):
            records.append(str(_relative_path(line, path.parent)))
            continue

        record = json.loads(line)
        unknown = sorted(set(record) - _RECORD_KEYS)
        if unknown:
            raise ValueError(
                f"unknown dataset record key(s) {', '.join(unknown)}; "
                f"a dataset record accepts {', '.join(sorted(_RECORD_KEYS))}"
            )
        if "msa" in record and "msa_collection" in record:
            raise ValueError("a dataset record cannot contain both `msa` and `msa_collection`")
        record["structure"] = str(_relative_path(os.fspath(record["structure"]), path.parent))
        if "target" in record:
            target = record["target"]
            if not isinstance(target, dict) or set(target) != {"path", "instance_ids"}:
                raise ValueError("a dataset record `target` must contain exactly `path` and `instance_ids`")
            instance_ids = target["instance_ids"]
            if not isinstance(instance_ids, list) or not all(
                isinstance(instance_id, str) for instance_id in instance_ids
            ):
                raise ValueError("a dataset record `target.instance_ids` must be a list of strings")
            target["path"] = str(_relative_path(os.fspath(target["path"]), path.parent))
        if "msa" in record:
            if not isinstance(record["msa"], dict):
                raise ValueError("a dataset record `msa` must be a JSON object with `path` and `entity_id`")
            record["msa"]["path"] = str(_relative_path(os.fspath(record["msa"]["path"]), path.parent))
        if "msa_collection" in record:
            record["msa_collection"] = str(_relative_path(os.fspath(record["msa_collection"]), path.parent))
        records.append(record)
    return records


def _relative_path(value: str, parent: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path
