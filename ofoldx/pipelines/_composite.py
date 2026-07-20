r"""Shared save/load for concrete multi-pair (composition) Pipelines (DESIGN §7.4).

A one-pair concrete Pipeline receives the loaded root pair directly through
``pipeline_class(model=..., processor=...)`` (frozen §7.4). A multi-pair concrete
Pipeline instead binds several class-owned Model/Processor pairs, so its live
constructor cannot be driven by that one call. This module freezes the multi-pair
half of §7.4: each such Pipeline declares a class-level ``_PAIR_LAYOUT`` — an ordered
tuple of named checkpoint pairs, one at the repository root and the rest at fixed
class-owned sub-directories — and the shared driver here writes and reloads every
pair through the ordinary single-pair repository writer/loader, then reconstructs the
Pipeline through its own multi-pair constructor.

The ``pipeline.json`` marker stays the frozen one-field object (§6.4/§7.3): the
pair-layout is fixed in Python by the class, never serialized. Sub-pairs use the same
ordinary pair format and carry no marker of their own (§7.3).
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from ..artifacts import save_model
from ..hub import _ResolvedRepository, _load_model, _load_processor, _resolve_repository
from .registry import _PIPELINE_IDS


class _CompositePair(NamedTuple):
    r"""One class-owned Model/Processor pair in a multi-pair Pipeline's fixed layout.

    Attributes:
        subdirectory: The fixed repository-relative directory holding the pair; the
            empty string is the independently runnable root pair, any other value is a
            fixed class-owned sub-directory (``structure``, ``protein_designer``, ...).
        model_attribute: The constructor keyword / instance attribute carrying the
            Model, or — when ``wrap`` is set — the child single-pair Pipeline.
        processor_attribute: The constructor keyword / instance attribute carrying the
            Processor. It is ``None`` exactly when ``wrap`` supplies a child Pipeline
            that owns both halves of the pair.
        wrap: When set, the reloaded pair is wrapped as ``wrap(model=..., processor=...)``
            and bound to ``model_attribute`` (used for the ODesign sequence designers,
            which are read-only single-pair child Pipelines rather than raw pairs).
        required: An optional pair is written and reloaded only when it is present.
    """

    subdirectory: str
    model_attribute: str
    processor_attribute: str | None
    wrap: type | None = None
    required: bool = True


class _CompositePipeline:
    r"""Mixin: DESIGN §7.4 multi-pair repository ``save`` / ``from_repository``.

    A concrete multi-pair Pipeline subclasses this and declares ``_PAIR_LAYOUT``. The
    presence of ``_PAIR_LAYOUT`` (equivalently, ``from_repository``) is the signal the
    repository factory uses to route to :meth:`from_repository` instead of the frozen
    one-pair ``(model, processor)`` call.
    """

    _PAIR_LAYOUT: tuple[_CompositePair, ...]

    def _composite_pair_source(self, pair: _CompositePair) -> tuple[nn.Module, object] | None:
        r"""Return the ``(model, processor)`` a layout slot serializes, or ``None`` when absent."""

        bound = getattr(self, pair.model_attribute, None)
        if pair.wrap is not None:
            if bound is None:
                return None
            return bound.model, bound.processor
        processor = getattr(self, pair.processor_attribute, None) if pair.processor_attribute else None
        if bound is None or processor is None:
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
        r"""Write every class-owned pair, then the one-field marker last, per DESIGN §7.4/§7.5.

        Each pair is written by the ordinary single-pair repository writer
        (:func:`ofoldx.artifacts.save_model`) — the root pair at the repository root and
        every named pair in its fixed sub-directory, without its own marker. The exact
        registered one-field ``pipeline.json`` marker is written only after all pairs
        succeed. Saving is local-only unless ``repo_id`` is given, in which case the
        fully written repository is uploaded through the same [`ofoldx.hub`][ofoldx.hub]
        uploader the single-pair writer uses.
        """

        pipeline_id = _PIPELINE_IDS.get(type(self))
        if pipeline_id is None:
            raise ValueError(f"{type(self).__name__} is not a registered concrete Pipeline; it cannot be saved")

        destination = Path(directory).expanduser()
        for pair in self._PAIR_LAYOUT:
            source = self._composite_pair_source(pair)
            if source is None:
                if pair.required:
                    location = pair.subdirectory or "root"
                    raise ValueError(f"cannot save {type(self).__name__}: required {location} pair is missing")
                continue
            model, processor = source
            save_model(
                destination / pair.subdirectory,
                model=model,
                processor=processor,
                max_shard_size=max_shard_size,
            )

        marker = json.dumps({"pipeline": pipeline_id}, indent=2, sort_keys=True) + "\n"
        (destination / "pipeline.json").write_text(marker, encoding="utf-8")

        if repo_id is not None:
            from ..hub import _upload_repository

            _upload_repository(
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
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""Reconstruct a multi-pair Pipeline from one immutable repository snapshot (DESIGN §7.4).

        The reference is resolved once so every class-owned pair loads from the same
        commit. Each pair is loaded through the ordinary single-pair loaders and mapped
        to its declared constructor keyword; an absent optional pair is skipped. A
        missing required pair fails without a standard-root fallback.
        """

        resolved = _resolve_repository(directory, revision=revision)
        arguments: dict[str, object] = {}
        for pair in cls._PAIR_LAYOUT:
            subrepository = _ResolvedRepository(resolved.path / pair.subdirectory)
            if not (subrepository.path / "model.json").exists():
                if pair.required:
                    location = pair.subdirectory or "root"
                    raise ValueError(
                        f"repository is missing the {location} pair required by {cls.__name__}"
                    )
                continue
            processor = _load_processor(subrepository, device=device, dtype=dtype)
            model = _load_model(subrepository, device=device, dtype=dtype)
            if pair.wrap is not None:
                arguments[pair.model_attribute] = pair.wrap(model=model, processor=processor)
            else:
                arguments[pair.model_attribute] = model
                arguments[pair.processor_attribute] = processor
        return cls(**arguments)
