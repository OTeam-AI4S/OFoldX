r"""
Lazy structure datasets backed by a processor.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import math
import operator
import os
import signal
import threading
import time
import traceback
import zlib
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from types import FrameType
from typing import Any, Iterator

from biotite import InvalidFileError
from biotite.structure import BadStructureError
from torch.utils.data import Dataset, get_worker_info

from ..io.structure import NoRetainedStructureError
from ..processor import UnresolvedTrainingCropError

_STRUCTURE_SUFFIXES: frozenset[str] = frozenset(
    {".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif", ".mmcif.gz"}
)
_GLOB_MAGIC: tuple[str, ...] = ("*", "?", "[")
_LOGGER = logging.getLogger(__name__)
_PROCESSING_STAGES: frozenset[str] = frozenset({"read_structure", "read_msa", "processor", "unknown"})
_PROCESSING_STAGE: ContextVar[str] = ContextVar("ofoldx_structure_processing_stage", default="unknown")
_PROCESSING_ERROR_STAGE_ATTRIBUTE = "_ofoldx_processing_stage"
_RECOVERABLE_DATA_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    KeyError,
    IndexError,
    FileNotFoundError,
    EOFError,
    gzip.BadGzipFile,
    zlib.error,
    InvalidFileError,
    BadStructureError,
)


class StructureProcessingTimeoutError(TimeoutError):
    r"""Raised when one structure-dataset candidate exceeds its processing deadline."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        if stage is None:
            stage = _PROCESSING_STAGE.get()
        if stage not in _PROCESSING_STAGES:
            raise ValueError(f"invalid structure processing stage: {stage!r}")
        super().__init__(message)
        self.stage = stage


class ProcessingTimeoutExhaustionError(UnresolvedTrainingCropError):
    r"""Raised when a bounded dataset scan containing processing timeouts is exhausted."""

    had_processing_timeout = True

    def __init__(
        self,
        message: str,
        *,
        timeout_count: int,
        scanned_candidates: int,
        processed_attempts: int,
    ) -> None:
        super().__init__(message)
        self.timeout_count = timeout_count
        self.scanned_candidates = scanned_candidates
        self.processed_attempts = processed_attempts


class ConsecutiveDataLoadingErrors(RuntimeError):
    r"""Raised when enabled data-error recovery cannot produce a replacement item."""

    def __init__(
        self,
        message: str,
        *,
        failure_count: int,
        threshold: int,
        scanned_candidates: int,
        processed_attempts: int,
        limit_reached: bool,
    ) -> None:
        super().__init__(message)
        self.failure_count = failure_count
        self.threshold = threshold
        self.scanned_candidates = scanned_candidates
        self.processed_attempts = processed_attempts
        self.limit_reached = limit_reached


@contextmanager
def _processing_stage(stage: str) -> Iterator[None]:
    r"""Expose the active candidate-processing stage to the timeout handler."""

    if stage not in _PROCESSING_STAGES or stage == "unknown":
        raise ValueError(f"invalid explicit structure processing stage: {stage!r}")
    token = _PROCESSING_STAGE.set(stage)
    try:
        yield
    except Exception as error:
        # The ContextVar is reset before __getitem__ handles the exception.
        # Retain the narrow stage on the exception so recovery never mistakes
        # an unrelated dataset/runtime failure for malformed sample data.
        try:
            setattr(error, _PROCESSING_ERROR_STAGE_ATTRIBUTE, stage)
        except (AttributeError, TypeError):
            pass
        raise
    finally:
        _PROCESSING_STAGE.reset(token)


def _normalize_processing_timeout(value: float | None) -> float | None:
    r"""Validate and normalize a per-candidate processing timeout."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("processing_timeout_seconds must be a finite positive number or None")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("processing_timeout_seconds must be a finite positive number or None") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("processing_timeout_seconds must be a finite positive number or None")
    return timeout


def _normalize_max_consecutive_processing_errors(value: int | None) -> int | None:
    r"""Validate the opt-in consecutive recoverable-data-error threshold."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_consecutive_processing_errors must be a positive integer or None")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError("max_consecutive_processing_errors must be a positive integer or None") from error
    if normalized <= 0:
        raise ValueError("max_consecutive_processing_errors must be a positive integer or None")
    return normalized


@contextmanager
def _processing_deadline(seconds: float | None) -> Iterator[None]:
    r"""Apply one best-effort POSIX real-time deadline without disturbing another timer."""

    if seconds is None:
        yield
        return
    required_apis = ("SIGALRM", "ITIMER_REAL", "getitimer", "setitimer")
    if os.name != "posix" or any(not hasattr(signal, name) for name in required_apis):
        raise RuntimeError("structure processing timeouts require POSIX signal.setitimer(ITIMER_REAL) support")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("structure processing timeouts require the main thread of the DataLoader process")

    timer_kind = signal.ITIMER_REAL
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(timer_kind)
    if any(value > 0 for value in previous_timer):
        raise RuntimeError(
            "cannot start a structure processing timeout while an ITIMER_REAL timer is already active "
            f"(remaining={previous_timer[0]:g}, interval={previous_timer[1]:g})"
        )

    def handle_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise StructureProcessingTimeoutError(
            f"structure processing exceeded the configured {seconds:g}-second timeout",
            stage=_PROCESSING_STAGE.get(),
        )

    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        signal.setitimer(timer_kind, seconds)
        yield
    finally:
        # Disarm before restoring the previous handler so this deadline cannot
        # be delivered to an unrelated handler after the candidate returns.
        try:
            signal.setitimer(timer_kind, 0.0)
        finally:
            try:
                signal.signal(signal.SIGALRM, previous_handler)
            finally:
                signal.setitimer(timer_kind, *previous_timer)


class StructureDataset(Dataset):
    r"""
    Process structure files lazily into typed model inputs.

    Args:
        source: Directory, glob, manifest, structure file, or ``None`` for an empty
            dataset.
        processor: Callable that converts one structure into the value consumed by
            the model strategy.
        seq_to_pdb_index: Optional JSON sequence-to-directory mapping for indexed
            per-structure MSA loading.
        msa_path: Optional root directory for indexed MSAs.
        processor_kwargs: Arguments forwarded to every processor call.
        processing_timeout_seconds: Optional finite positive deadline for each
            candidate's structure, MSA, and processor work. ``None`` disables it.
        max_consecutive_processing_errors: Optional positive limit enabling
            replacement of samples whose data processing raises a recoverable
            malformed-input error. ``None`` preserves the legacy behavior.
        training_log_file: Internal runner-provided training log path used to
            place per-rank/per-worker structured data-error sidecars.
    """

    processing_timeout_seconds: float | None = None

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        seq_to_pdb_index: str | os.PathLike[str] | None = None,
        msa_path: str | os.PathLike[str] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        processing_timeout_seconds: float | None = None,
        max_consecutive_processing_errors: int | None = None,
        training_log_file: str | os.PathLike[str] | None = None,
        ccd_revision: str | None = None,
        ccd_snapshot_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if (seq_to_pdb_index is None) != (msa_path is None):
            raise ValueError("seq_to_pdb_index and msa_path must be provided together")
        self.processor = processor
        self.processor_kwargs = dict(processor_kwargs or {})
        self.processing_timeout_seconds = _normalize_processing_timeout(processing_timeout_seconds)
        self.max_consecutive_processing_errors = _normalize_max_consecutive_processing_errors(
            max_consecutive_processing_errors
        )
        self.training_log_file = None if training_log_file is None else Path(training_log_file).expanduser()
        if ccd_revision is not None and ccd_snapshot_path is not None:
            raise ValueError("configure at most one of ccd_revision or ccd_snapshot_path")
        self.ccd_revision = ccd_revision
        self.ccd_snapshot_path = None if ccd_snapshot_path is None else os.fspath(ccd_snapshot_path)
        self.seq_to_pdb_index: Mapping[str, object] | None = None
        self.msa_path: Path | None = None
        if seq_to_pdb_index is not None and msa_path is not None:
            index_path = Path(seq_to_pdb_index).expanduser()
            if not index_path.is_file():
                raise FileNotFoundError(f"sequence-to-PDB index is not a file: {index_path}")
            resolved_msa_path = Path(msa_path).expanduser()
            if not resolved_msa_path.is_dir():
                raise FileNotFoundError(f"MSA path is not a directory: {resolved_msa_path}")
            self.seq_to_pdb_index = _load_seq_to_pdb_index(index_path.resolve())
            self.msa_path = resolved_msa_path.resolve()
        # Records stay as paths and ordinary mappings; generic I/O reads their
        # canonical structure/evidence values per access.
        self.records: list[str | dict[str, Any]] = _resolve_records(source)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        length = len(self)
        logical_index = operator.index(index)
        if logical_index < 0:
            logical_index += length
        if logical_index < 0 or logical_index >= length:
            raise IndexError("dataset index out of range")

        if self.max_consecutive_processing_errors is not None:
            return self._getitem_with_processing_error_tolerance(logical_index)

        last_error: (
            UnresolvedTrainingCropError | NoRetainedStructureError | StructureProcessingTimeoutError | None
        ) = None
        timed_out_records: set[int] = set()
        scanned_candidates = 0
        processed_attempts = 0
        timeout_count = 0
        last_timeout_error: StructureProcessingTimeoutError | None = None
        for attempt in range(length):
            scanned_candidates += 1
            candidate = (logical_index + attempt) % length
            backing_record_index = self._backing_record_index(candidate)
            if backing_record_index in timed_out_records:
                continue
            processed_attempts += 1
            started_at = time.monotonic()
            try:
                with _processing_deadline(self.processing_timeout_seconds):
                    return self._process_logical_index(candidate)
            except StructureProcessingTimeoutError as error:
                last_error = error
                last_timeout_error = error
                timeout_count += 1
                if error.stage in {"read_structure", "read_msa"}:
                    timed_out_records.add(backing_record_index)
                self._warn_processing_timeout(candidate, error, elapsed_seconds=time.monotonic() - started_at)
            except (UnresolvedTrainingCropError, NoRetainedStructureError) as error:
                last_error = error
        message = (
            "no processable dataset item found after scanning "
            f"{scanned_candidates} logical candidates and processing {processed_attempts} attempts "
            f"starting at logical index {logical_index}"
        )
        if timeout_count:
            raise ProcessingTimeoutExhaustionError(
                message,
                timeout_count=timeout_count,
                scanned_candidates=scanned_candidates,
                processed_attempts=processed_attempts,
            ) from last_timeout_error
        raise UnresolvedTrainingCropError(message) from last_error

    def _getitem_with_processing_error_tolerance(self, logical_index: int) -> Any:
        r"""Replace malformed candidates until success, exhaustion, or the configured limit."""

        length = len(self)
        threshold = self.max_consecutive_processing_errors
        if threshold is None:  # pragma: no cover - guarded by __getitem__
            raise AssertionError("processing-error tolerance called while disabled")

        failed_backing_records: set[int] = set()
        last_error: Exception | None = None
        last_timeout_error: StructureProcessingTimeoutError | None = None
        scanned_candidates = 0
        processed_attempts = 0
        streak = 0
        timeout_count = 0
        for attempt in range(length):
            scanned_candidates += 1
            candidate = (logical_index + attempt) % length
            backing_record_index = self._backing_record_index(candidate)
            if backing_record_index in failed_backing_records:
                continue

            processed_attempts += 1
            started_at = time.monotonic()
            try:
                with _processing_deadline(self.processing_timeout_seconds):
                    return self._process_logical_index(candidate)
            except Exception as error:
                stage = self._processing_error_stage(error)
                if not self._is_recoverable_data_error(error, stage=stage):
                    raise

                last_error = error
                streak += 1
                if isinstance(error, StructureProcessingTimeoutError):
                    last_timeout_error = error
                    timeout_count += 1
                if stage in {"read_structure", "read_msa"}:
                    failed_backing_records.add(backing_record_index)
                limit_reached = streak >= threshold
                has_attemptable_candidate = any(
                    self._backing_record_index((logical_index + future_attempt) % length)
                    not in failed_backing_records
                    for future_attempt in range(attempt + 1, length)
                )
                if not has_attemptable_candidate:
                    # The look-ahead above inspected every remaining logical
                    # row. Count those de-duplicated rows as scanned without
                    # processing their already-failed backing record again.
                    scanned_candidates = length
                if limit_reached:
                    action = "raise_consecutive_error_limit"
                elif not has_attemptable_candidate and timeout_count:
                    action = "raise_processing_timeout_exhaustion"
                else:
                    action = "skip_to_next_candidate" if has_attemptable_candidate else "raise_candidate_exhaustion"
                self._log_recoverable_data_error(
                    candidate,
                    error,
                    stage=stage,
                    elapsed_seconds=time.monotonic() - started_at,
                    streak=streak,
                    threshold=threshold,
                    action=action,
                )
                if limit_reached:
                    message = (
                        f"aborting after {streak} consecutive recoverable data-loading errors "
                        f"(configured limit {threshold}) while scanning from logical index {logical_index}"
                    )
                    raise ConsecutiveDataLoadingErrors(
                        message,
                        failure_count=streak,
                        threshold=threshold,
                        scanned_candidates=scanned_candidates,
                        processed_attempts=processed_attempts,
                        limit_reached=True,
                    ) from error
                if not has_attemptable_candidate:
                    message = (
                        "no processable dataset item found before exhausting the unique candidates after "
                        f"{streak} consecutive recoverable data-loading errors, scanning {scanned_candidates} "
                        f"logical candidates and processing {processed_attempts} attempts from logical index "
                        f"{logical_index} (configured limit {threshold})"
                    )
                    if timeout_count:
                        raise ProcessingTimeoutExhaustionError(
                            message,
                            timeout_count=timeout_count,
                            scanned_candidates=scanned_candidates,
                            processed_attempts=processed_attempts,
                        ) from last_timeout_error
                    raise ConsecutiveDataLoadingErrors(
                        message,
                        failure_count=streak,
                        threshold=threshold,
                        scanned_candidates=scanned_candidates,
                        processed_attempts=processed_attempts,
                        limit_reached=False,
                    ) from error

        raise AssertionError(
            "processing-error recovery exhausted without recording a terminal candidate error"
        ) from last_error

    @staticmethod
    def _processing_error_stage(error: Exception) -> str:
        r"""Return the narrow processing stage retained on a candidate exception."""

        if isinstance(error, StructureProcessingTimeoutError):
            return error.stage
        stage = getattr(error, _PROCESSING_ERROR_STAGE_ATTRIBUTE, "unknown")
        return stage if stage in _PROCESSING_STAGES else "unknown"

    @staticmethod
    def _is_recoverable_data_error(error: Exception, *, stage: str) -> bool:
        r"""Classify malformed-sample failures without swallowing runtime/programming errors."""

        if isinstance(
            error,
            (StructureProcessingTimeoutError, UnresolvedTrainingCropError, NoRetainedStructureError),
        ):
            return True
        return stage in {"read_structure", "read_msa", "processor"} and isinstance(error, _RECOVERABLE_DATA_ERRORS)

    def _log_recoverable_data_error(
        self,
        logical_index: int,
        error: Exception,
        *,
        stage: str,
        elapsed_seconds: float,
        streak: int,
        threshold: int,
        action: str,
    ) -> None:
        r"""Record one handled data failure in logging and an optional worker-safe JSONL sidecar."""

        worker = get_worker_info()
        worker_id: int | str = "main" if worker is None else worker.id
        rank, rank_component = self._rank_provenance()
        backing_record_index = self._backing_record_index(logical_index)
        structure_path = self._safe_backing_record_path(logical_index)
        formatted_traceback = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        try:
            log_data_error = _LOGGER.error if action.startswith("raise_") else _LOGGER.warning
            log_data_error(
                "Handled recoverable data-loading error at stage %s for logical index %d, backing record "
                "index %d/path %r, rank %s, DataLoader worker ID %s, streak %d/%d, action %s: %s: %s",
                stage,
                logical_index,
                backing_record_index,
                structure_path,
                rank,
                worker_id,
                streak,
                threshold,
                action,
                type(error).__name__,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
        except Exception:
            # A broken/inherited logging handler must never replace the sample
            # error or disable the configured recovery path.
            pass
        if self.training_log_file is None:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": formatted_traceback,
            "logical_index": logical_index,
            "backing_record_index": backing_record_index,
            "structure_path": structure_path,
            "rank": rank,
            "worker_id": worker_id,
            "pid": os.getpid(),
            "elapsed_seconds": elapsed_seconds,
            "streak": streak,
            "threshold": threshold,
            "action": action,
        }
        try:
            self._append_data_error_event(event, rank_component=rank_component, worker_id=worker_id)
        except Exception as logging_error:
            try:
                _LOGGER.warning(
                    "Could not append recoverable data-loading error to the training-log sidecar %r: %s: %s",
                    self.training_log_file,
                    type(logging_error).__name__,
                    logging_error,
                )
            except Exception:
                pass

    def _append_data_error_event(
        self,
        event: Mapping[str, Any],
        *,
        rank_component: str,
        worker_id: int | str,
    ) -> None:
        r"""Append one complete JSON event to this rank/worker's single-writer sidecar."""

        if self.training_log_file is None:  # pragma: no cover - guarded by caller
            return
        worker_component = f"{worker_id:02d}" if isinstance(worker_id, int) else "main"
        log_path = self.training_log_file
        sidecar = log_path.with_name(
            f"{log_path.stem}.data-errors.rank-{rank_component}.worker-{worker_component}.jsonl"
        )
        payload = (json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = os.open(sidecar, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            remaining = memoryview(payload)
            interrupted_writes = 0
            while remaining:
                try:
                    written = os.write(descriptor, remaining)
                except InterruptedError:
                    interrupted_writes += 1
                    if interrupted_writes > 100:
                        raise OSError("JSONL sidecar write was interrupted more than 100 times") from None
                    continue
                if written <= 0:
                    raise OSError(f"JSONL sidecar write made no progress with {len(remaining)} bytes remaining")
                remaining = remaining[written:]
        finally:
            os.close(descriptor)

    @classmethod
    def _rank_provenance(cls) -> tuple[int | str, str]:
        r"""Return bounded log provenance and a safe filename component for the current rank."""

        raw_rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "unavailable"))
        if raw_rank.isascii() and raw_rank.isdigit() and len(raw_rank) <= 10:
            numeric_rank = int(raw_rank)
            return numeric_rank, f"{numeric_rank:05d}"
        bounded_rank = raw_rank if len(raw_rank) <= 128 else f"{raw_rank[:125]}..."
        bounded_rank = bounded_rank or "unavailable"
        return bounded_rank, cls._filename_component(bounded_rank)

    @staticmethod
    def _filename_component(value: str) -> str:
        r"""Make an environment-derived identifier safe as one filename component."""

        component = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_" for character in value
        )
        return component[:64] or "unavailable"

    def _safe_backing_record_path(self, logical_index: int) -> str:
        r"""Resolve provenance without allowing a secondary logging failure to hide the sample error."""

        try:
            return self._backing_record_path(logical_index)
        except Exception as error:
            return f"<unavailable: {type(error).__name__}: {error}>"

    def _process_logical_index(self, index: int) -> Any:
        r"""Process the record represented by one normalized logical index."""

        return self._process_record(index)

    def _backing_record_index(self, logical_index: int) -> int:
        r"""Map one normalized logical index to its backing record ordinal."""

        return logical_index

    def _backing_record_path(self, logical_index: int) -> str:
        r"""Return the structure path backing one normalized logical index."""

        record = self.records[self._backing_record_index(logical_index)]
        return record if isinstance(record, str) else os.fspath(record["structure"])

    def _warn_processing_timeout(
        self,
        logical_index: int,
        error: StructureProcessingTimeoutError,
        *,
        elapsed_seconds: float,
    ) -> None:
        r"""Log one handled timeout with enough provenance to locate the candidate."""

        worker = get_worker_info()
        worker_id: int | str = "main process" if worker is None else worker.id
        # DataLoader worker processes inherit the launch environment. Reading
        # it avoids calling an inherited distributed process group from a fork.
        rank = os.environ.get("RANK", "unavailable")
        backing_record_index = self._backing_record_index(logical_index)
        _LOGGER.warning(
            "Handled %s after %.3f actual seconds (configured deadline %s seconds) for logical index %d, "
            "backing record index %d/path %r, stage %s, rank %s, DataLoader worker ID %s",
            type(error).__name__,
            elapsed_seconds,
            self.processing_timeout_seconds,
            logical_index,
            backing_record_index,
            self._backing_record_path(logical_index),
            error.stage,
            rank,
            worker_id,
        )

    def _process_record(self, index: int, **extra_processor_kwargs: Any) -> Any:
        from ..data import ENTITY_PROTEIN, MSA, MSACollection, decode_msa_collection
        from ..io import read_msa, read_structure

        record = self.records[index]
        msa: MSA | MSACollection | None
        if isinstance(record, str):
            structure_path = record
        else:
            structure_path = record["structure"]
        with _processing_stage("read_structure"):
            structure = read_structure(
                structure_path,
                ccd_revision=self.ccd_revision,
                ccd_snapshot_path=self.ccd_snapshot_path,
            )
        with _processing_stage("read_msa"):
            if isinstance(record, str):
                msa = None
            else:
                if "msa" in record:
                    msa_spec = record["msa"]
                    msa = read_msa(msa_spec["path"], entity_id=msa_spec["entity_id"])
                elif "msa_collection" in record:
                    payload = json.loads(Path(record["msa_collection"]).read_text(encoding="utf-8"))
                    msa = decode_msa_collection(payload)
                else:
                    msa = None
            if msa is None and self.seq_to_pdb_index is not None and self.msa_path is not None:
                has_protein = any(int(entity_type) == ENTITY_PROTEIN for entity_type in structure.system.entity_type)
                if has_protein:
                    msa = read_msa(structure, self.seq_to_pdb_index, self.msa_path)
        processor_kwargs = {**self.processor_kwargs, **extra_processor_kwargs}
        with _processing_stage("processor"):
            if msa is not None:
                return self.processor(structure, msa=msa, **processor_kwargs)
            return self.processor(structure, **processor_kwargs)


class ClusteredStructureDataset(StructureDataset):
    r"""An eagerly indexed, asymmetric-unit CIF dataset for weighted sampling.

    A row ordinal is mapped to its complete backing CIF record. When
    ``forward_indices`` is true, the selected row is also passed to the
    processor so chain and interface rows can choose distinct spatial crop
    centres while retaining full-structure coordinate truth. Default structure
    ingestion removes hydrogens and water; configured ion handling and
    ``exclude_mol_types`` affect sampling rows only.
    """

    def __init__(
        self,
        source: Any,
        processor: Any,
        *,
        cluster_file: str,
        cluster_indices_file: str | os.PathLike[str] | None = None,
        contact_cutoff: float = 5.0,
        include_all_chain_pairs: bool = False,
        exclude_mol_types: tuple[str, ...] = ("ions",),
        beta: Mapping[str, float] | None = None,
        alpha: Mapping[str, float] | None = None,
        seq_to_pdb_index: str | os.PathLike[str] | None = None,
        msa_path: str | os.PathLike[str] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        processing_timeout_seconds: float | None = None,
        max_consecutive_processing_errors: int | None = None,
        training_log_file: str | os.PathLike[str] | None = None,
        forward_indices: bool = False,
        ccd_revision: str | None = None,
        ccd_snapshot_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__(
            source,
            processor,
            seq_to_pdb_index=seq_to_pdb_index,
            msa_path=msa_path,
            processor_kwargs=processor_kwargs,
            processing_timeout_seconds=processing_timeout_seconds,
            max_consecutive_processing_errors=max_consecutive_processing_errors,
            training_log_file=training_log_file,
            ccd_revision=ccd_revision,
            ccd_snapshot_path=ccd_snapshot_path,
        )
        self.forward_indices = bool(forward_indices)
        invalid = [
            Path(record if isinstance(record, str) else record["structure"])
            for record in self.records
            if _structure_suffix(record if isinstance(record, str) else record["structure"])
            not in {".cif", ".mmcif", ".cif.gz", ".mmcif.gz"}
        ]
        if invalid:
            raise ValueError(f"cluster sampling only supports CIF/mmCIF records: {invalid[0]}")
        from .cluster import calculate_cluster_weights, load_or_build_cluster_indices, make_cluster_indices

        self.cluster_indices_file = (
            Path(cluster_indices_file).expanduser().resolve() if cluster_indices_file is not None else None
        )
        if self.cluster_indices_file is None:
            raw_entries = make_cluster_indices(
                self.records,
                cluster_file,
                contact_cutoff=contact_cutoff,
                include_all_chain_pairs=include_all_chain_pairs,
                exclude_mol_types=exclude_mol_types,
                ccd_revision=self.ccd_revision,
                ccd_snapshot_path=self.ccd_snapshot_path,
            )
            self.entries, self.weights = calculate_cluster_weights(raw_entries, beta=beta, alpha=alpha)
            self.cluster_indices_status = None
        else:
            self.entries, self.weights, self.cluster_indices_status = load_or_build_cluster_indices(
                self.records,
                cluster_file,
                self.cluster_indices_file,
                contact_cutoff=contact_cutoff,
                include_all_chain_pairs=include_all_chain_pairs,
                exclude_mol_types=exclude_mol_types,
                beta=beta,
                alpha=alpha,
                ccd_revision=self.ccd_revision,
                ccd_snapshot_path=self.ccd_snapshot_path,
            )
        if not self.entries:
            raise ValueError("cluster sampling produced no valid chain or interface index rows")

    def __len__(self) -> int:
        return len(self.entries)

    def _process_logical_index(self, index: int) -> Any:
        r"""Process the backing record and metadata for one cluster row."""

        entry = self.entries[index]
        if self.forward_indices:
            return self._process_record(entry.record_index, indices=entry)
        return self._process_record(entry.record_index)

    def _backing_record_index(self, logical_index: int) -> int:
        r"""Map a cluster row ordinal to the CIF record it indexes."""

        return self.entries[logical_index].record_index


@cache
def _load_seq_to_pdb_index(path: Path) -> Mapping[str, object]:
    r"""Load and cache one canonical sequence-to-MSA-directory JSON mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"sequence-to-PDB index JSON must contain an object at the top level: {path}")
    return payload


def _resolve_records(source: Any) -> list[str | dict[str, Any]]:
    r"""
    Resolve a source into a concrete, ordered list of structure-file paths.

    ``None`` is an empty dataset. A glob pattern expands (sorted) to its structure
    matches; a directory yields the structure files within its tree (sorted); a
    manifest file yields its listed paths in order; a lone structure file is a
    single record.
    """

    if source is None:
        return []
    text = os.fspath(source)
    if any(character in text for character in _GLOB_MAGIC):
        matches = sorted(glob.glob(os.path.expanduser(text), recursive=True))
        return [m for m in matches if os.path.isfile(m) and _is_structure_path(m)]
    path = Path(text).expanduser()
    if path.is_dir():
        records: list[str | dict[str, Any]] = []
        records.extend(sorted(str(entry) for entry in path.rglob("*") if _is_structure_path(entry)))
        return records
    if not path.exists():
        raise FileNotFoundError(f"structure source does not exist: {path}")
    if _is_structure_path(path):
        return [str(path)]
    if path.suffix.lower() == ".gz":
        raise ValueError("gzip-compressed dataset source must have a .pdb.gz, .cif.gz, or .mmcif.gz suffix")
    return _read_manifest(path)


def _is_structure_path(path: str | os.PathLike[str]) -> bool:
    r"""Return whether a path has a supported simple or gzip-compound suffix."""

    return _structure_suffix(path) in _STRUCTURE_SUFFIXES


def _structure_suffix(path: str | os.PathLike[str]) -> str:
    r"""Return a normalized simple or gzip-compound structure suffix."""

    suffixes = Path(path).suffixes
    if not suffixes:
        return ""
    return "".join(suffixes[-2:]).lower() if suffixes[-1].lower() == ".gz" else suffixes[-1].lower()


def _read_manifest(path: Path) -> list[str | dict[str, Any]]:
    r"""
    Read structure paths or JSON records, one per line, resolved against its dir.

    Blank lines and ``#`` comments are skipped; relative entries resolve against the
    manifest's own directory so a manifest and its explicit MSA evidence are portable
    together.
    """

    records: list[str | dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            record = json.loads(line)
            if "msa" in record and "msa_collection" in record:
                raise ValueError("a dataset record cannot contain both `msa` and `msa_collection`")
            structure_path = Path(os.fspath(record["structure"])).expanduser()
            if not structure_path.is_absolute():
                structure_path = path.parent / structure_path
            record["structure"] = str(structure_path)
            if "msa" in record:
                msa_path = Path(os.fspath(record["msa"]["path"])).expanduser()
                if not msa_path.is_absolute():
                    msa_path = path.parent / msa_path
                record["msa"]["path"] = str(msa_path)
            if "msa_collection" in record:
                collection_path = Path(os.fspath(record["msa_collection"])).expanduser()
                if not collection_path.is_absolute():
                    collection_path = path.parent / collection_path
                record["msa_collection"] = str(collection_path)
            records.append(record)
            continue
        entry = Path(line).expanduser()
        if not entry.is_absolute():
            entry = path.parent / entry
        records.append(str(entry))
    return records
