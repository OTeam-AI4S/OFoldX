r"""Persistence for pipelines owning multiple model-processor pairs.

``_PAIR_LAYOUT`` assigns one pair to the repository root and additional pairs to fixed
subdirectories. Only the root carries ``pipeline.json``.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple, cast

import chanfig
import torch
from torch import nn

from .._repository import ResolvedRepository, load_model, load_processor, resolve_repository
from ..artifacts import save_model
from .registry import PIPELINES


class CompositePair(NamedTuple):
    r"""One pair in a composite pipeline's repository layout.

    Attributes:
        subdirectory: Repository-relative location; ``""`` selects the root.
        model_attribute: Model or wrapped pipeline constructor attribute.
        processor_attribute: Processor attribute, or ``None`` for a wrapped pair.
        wrap: Optional pipeline class used to wrap a loaded pair.
        required: Whether the pair must be present.
    """

    subdirectory: str
    model_attribute: str
    processor_attribute: str | None
    wrap: type | None = None
    required: bool = True


def _pair_directory(root: Path, subdirectory: str) -> Path:
    posix = PurePosixPath(subdirectory)
    windows = PureWindowsPath(subdirectory)
    if posix.anchor or windows.anchor or ".." in posix.parts or ".." in windows.parts:
        raise ValueError(f"composite pair path must stay within the repository: {subdirectory!r}")
    return root / subdirectory


class CompositePipeline:
    r"""Base class for pipelines with a fixed multi-pair repository layout."""

    _PAIR_LAYOUT: tuple[CompositePair, ...]

    def _composite_pair_source(self, pair: CompositePair) -> tuple[nn.Module, object] | None:
        r"""Return the ``(model, processor)`` a layout slot serializes, or ``None`` when absent."""

        bound = getattr(self, pair.model_attribute)
        if pair.wrap is not None:
            if bound is None:
                return None
            return bound.model, bound.processor
        processor = getattr(self, cast(str, pair.processor_attribute))
        if (bound is None) != (processor is None):
            location = pair.subdirectory or "root"
            raise ValueError(f"{location} model and processor must either both be present or both be absent")
        if bound is None:
            return None
        return bound, processor

    def save(
        self,
        directory: str | PathLike[str],
        *,
        max_shard_size: int | str = "5GB",
        repo_id: str | None = None,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> None:
        r"""Save every pair, write the root marker, and optionally upload the repository."""

        pipeline_id = PIPELINES.id_for(type(self))
        if pipeline_id is None:
            raise ValueError(f"{type(self).__name__} is not a registered concrete Pipeline; it cannot be saved")

        destination = Path(directory).expanduser()

        # Resolve every slot's source first so a missing required pair aborts the save
        # before the destination is mutated.
        plan: list[tuple[CompositePair, Path, tuple[nn.Module, object] | None]] = []
        for pair in self._PAIR_LAYOUT:
            pair_directory = _pair_directory(destination, pair.subdirectory)
            source = self._composite_pair_source(pair)
            if source is None and pair.required:
                location = pair.subdirectory or "root"
                raise ValueError(f"cannot save {type(self).__name__}: required {location} pair is missing")
            plan.append((pair, pair_directory, source))

        for pair, pair_directory, source in plan:
            if source is None:
                if pair.subdirectory:
                    stale = pair_directory
                    for filename in (
                        "model.json",
                        "processor.json",
                        "pipeline.json",
                        "model.safetensors",
                        "model.safetensors.index.json",
                    ):
                        (stale / filename).unlink(missing_ok=True)
                    for shard in stale.glob("model-*-of-*.safetensors"):
                        shard.unlink()
                continue
            model, processor = source
            save_model(
                pair_directory,
                model=model,
                processor=processor,
                max_shard_size=max_shard_size,
            )

        chanfig.NestedDict({"pipeline": pipeline_id}).dump(destination / "pipeline.json")

        if repo_id is not None:
            from .._repository import upload_repository

            upload_repository(
                destination,
                repo_id,
                token=token,
                private=private,
                commit_message=commit_message,
            )

    @classmethod
    def from_repository(
        cls,
        directory: str | PathLike[str],
        *,
        revision: str | None = None,
        token: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""Load every pair from one resolved repository snapshot.

        Missing optional pairs are skipped; missing required pairs raise ``ValueError``.
        """

        resolved = resolve_repository(directory, revision=revision, token=token)
        arguments: dict[str, object] = {}
        for pair in cls._PAIR_LAYOUT:
            subrepository = ResolvedRepository(_pair_directory(resolved.path, pair.subdirectory))
            if not (subrepository.path / "model.json").exists():
                if pair.required:
                    location = pair.subdirectory or "root"
                    raise ValueError(f"repository is missing the {location} pair required by {cls.__name__}")
                continue
            processor = load_processor(subrepository, device=device, dtype=dtype)
            model = load_model(subrepository, device=device, dtype=dtype)
            if pair.wrap is not None:
                arguments[pair.model_attribute] = pair.wrap(model=model, processor=processor)
            else:
                arguments[pair.model_attribute] = model
                arguments[cast(str, pair.processor_attribute)] = processor
        return cls(**arguments)
