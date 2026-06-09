r"""
Typed model input and output contracts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self


class BaseInput:
    r"""
    Base class for typed model inputs.

    Model-local input dataclasses inherit this class so pipelines and runners can identify already-processed
    batches without maintaining a closed list of concrete input types. Concrete subclasses are themselves
    ``@dataclass`` types whose tensor fields [`to`][ofoldx.models.contract.BaseInput.to] moves as a group.
    """

    __slots__ = ()

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self:
        r"""
        Return a copy with every tensor field moved via ``Tensor.to``.

        Every dataclass field is moved recursively: bare tensors are cast, and nested objects exposing a
        compatible ``to`` (lists and dicts of tensors included) are moved in turn. Non-tensor fields pass
        through unchanged. Only floating-point tensors follow ``dtype``; integer tensors keep their dtype.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            Input object with recursively moved tensor fields.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
            >>> batch = processor("complex.cif").to("cuda", dtype=torch.bfloat16)  # doctest: +SKIP
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@dataclass(slots=True)
class StructurePredictionOutput:
    r"""
    Collects the coordinate, distogram, and optional confidence tensors a structure-prediction model emits.

    Coordinate and distogram outputs are required; confidence/design heads are optional capabilities of specific
    architectures. Fields that are not produced by a concrete model remain ``None``.

    Attributes:
        pred_coords: ``[..., num_samples, N_atom, 3]`` predicted atom coords.
        distogram_logits: Pairwise distance-bin logits.
        plddt_logits: Optional per-token or per-atom pLDDT logits.
        pae_logits: Optional predicted aligned-error logits.
        pde_logits: Optional predicted distance-error logits.
        resolved_logits: Optional resolved-atom logits.
        bfactor_logits: Optional B-factor logits.
        token_bond_type_logits: Optional token bond-type reconstruction logits.
        denoised_coords: Optional single-step diffusion denoising prediction used during training.
        augmented_target: Optional augmented diffusion target used during training.
        noise_level: Optional diffusion noise level used during training.

    Examples:
        >>> import torch
        >>> output = StructurePredictionOutput(
        ...     pred_coords=torch.zeros(1, 5, 3),
        ...     distogram_logits=torch.zeros(1, 5, 5, 64),
        ... )
        >>> output.pred_coords.shape
        torch.Size([1, 5, 3])
        >>> output.plddt_logits is None
        True
    """

    pred_coords: Tensor
    distogram_logits: Tensor
    plddt_logits: Tensor | None = None
    pae_logits: Tensor | None = None
    pde_logits: Tensor | None = None
    resolved_logits: Tensor | None = None
    bfactor_logits: Tensor | None = None  # Optional B-factor head output.
    token_bond_type_logits: Tensor | None = None  # Optional token bond-type reconstruction output.
    # Single-step diffusion-training tensors (present only in training mode). The diffusion
    # loss compares ``denoised_coords`` against ``augmented_target`` at ``noise_level``;
    # ``pred_coords`` then carries the no-grad mini-rollout coords for confidence.
    denoised_coords: Tensor | None = None
    augmented_target: Tensor | None = None
    noise_level: Tensor | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> StructurePredictionOutput:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.

        ``None`` fields are preserved; only floating-point tensors follow ``dtype``.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            Output object with recursively moved tensor fields.

        Examples:
            >>> import torch
            >>> output = StructurePredictionOutput(
            ...     pred_coords=torch.zeros(1, 5, 3),
            ...     distogram_logits=torch.zeros(1, 5, 5, 64),
            ... )
            >>> output.to(dtype=torch.float64).pred_coords.dtype
            torch.float64
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@dataclass(slots=True)
class BindingAffinityPredictionOutput:
    r"""
    Collects scalar protein-ligand binding-affinity predictions.

    Attributes:
        affinity_pred_value: Primary predicted affinity value tensor.
        scores: Named score tensors emitted by the affinity head.
        structure_prediction: Optional nested structure prediction used to score the complex.
        metadata: Optional non-tensor metadata.
    """

    affinity_pred_value: Tensor
    scores: dict[str, Tensor]
    structure_prediction: StructurePredictionOutput | None = None
    metadata: dict[str, Any] | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> BindingAffinityPredictionOutput:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@dataclass(slots=True)
class DesignGeneratorOutput:
    r"""
    Single generator output for a biomolecular design model.

    This is the raw per-call output of a design generator; the pipeline later repackages accepted
    generations into [`DesignCandidate`][ofoldx.models.contract.DesignCandidate] entries.

    Attributes:
        sequence_logits: Optional sequence-token logits.
        sequence_tokens: Optional sampled or decoded sequence tokens.
        structure_prediction: Optional nested structure-prediction output.
        scores: Optional named score tensors.
        metadata: Optional non-tensor metadata.

    Examples:
        >>> import torch
        >>> output = DesignGeneratorOutput(sequence_tokens=torch.zeros(1, 8, dtype=torch.long))
        >>> output.sequence_tokens.shape
        torch.Size([1, 8])
        >>> output.structure_prediction is None
        True
    """

    sequence_logits: Tensor | None = None
    sequence_tokens: Tensor | None = None
    structure_prediction: StructurePredictionOutput | None = None
    scores: dict[str, Tensor] | None = None
    metadata: dict[str, Any] | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> DesignGeneratorOutput:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.

        The nested ``structure_prediction`` output and any tensors inside ``scores`` are moved recursively;
        ``metadata`` and other non-tensor fields pass through unchanged.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            Output object with recursively moved tensor fields.

        Examples:
            >>> import torch
            >>> output = DesignGeneratorOutput(scores={"plddt": torch.zeros(1)})
            >>> output.to(dtype=torch.float64).scores["plddt"].dtype
            torch.float64
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@dataclass(slots=True)
class DesignCandidate:
    r"""
    One candidate emitted by a design pipeline.

    Candidates are collected into a [`DesignOutput`][ofoldx.models.contract.DesignOutput]; each one
    bundles the designed sequence, an optional predicted structure, and per-candidate scores.

    Attributes:
        sequence_logits: Optional sequence-token logits.
        sequence_tokens: Optional sampled or decoded sequence tokens.
        structure_prediction: Optional nested structure-prediction output.
        scores: Optional named score tensors.
        metadata: Optional non-tensor metadata.

    Examples:
        >>> import torch
        >>> candidate = DesignCandidate(sequence_tokens=torch.zeros(6, dtype=torch.long))
        >>> candidate.sequence_tokens.shape
        torch.Size([6])
    """

    sequence_logits: Tensor | None = None
    sequence_tokens: Tensor | None = None
    structure_prediction: StructurePredictionOutput | None = None
    scores: dict[str, Tensor] | None = None
    metadata: dict[str, Any] | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> DesignCandidate:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.

        The nested ``structure_prediction`` output and any tensors inside ``scores`` are moved recursively;
        ``metadata`` and other non-tensor fields pass through unchanged.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            Candidate object with recursively moved tensor fields.

        Examples:
            >>> import torch
            >>> candidate = DesignCandidate(scores={"score": torch.zeros(1)})
            >>> candidate.to(dtype=torch.float64).scores["score"].dtype
            torch.float64
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@dataclass(slots=True)
class DesignOutput:
    r"""
    Bundles the ranked candidates a design workflow returns into one top-level result.

    The top-level return of a design run: a flat list of ranked or unranked
    [`DesignCandidate`][ofoldx.models.contract.DesignCandidate] entries.

    Attributes:
        candidates: Candidate list emitted by a design generator or normalized pipeline.

    Examples:
        >>> import torch
        >>> candidate = DesignCandidate(sequence_tokens=torch.zeros(4, dtype=torch.long))
        >>> output = DesignOutput(candidates=[candidate])
        >>> len(output.candidates)
        1
    """

    candidates: list[DesignCandidate]

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> DesignOutput:
        r"""
        Return a copy with every nested task output moved.

        Each candidate in ``candidates`` is moved recursively via its own ``to``.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            Design output with recursively moved candidate fields.

        Examples:
            >>> import torch
            >>> candidate = DesignCandidate(scores={"score": torch.zeros(1)})
            >>> moved = DesignOutput(candidates=[candidate]).to(dtype=torch.float64)
            >>> moved.candidates[0].scores["score"].dtype
            torch.float64
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


def move_dataclass_fields(
    instance: object,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
    copy: bool,
) -> dict[str, Any]:
    r"""
    Move every dataclass field recursively.

    Shared helper behind every ``to`` method on this module's typed outputs: it walks the dataclass
    fields of ``instance`` and moves each value, descending into nested tensors, lists, dicts, and objects
    exposing a compatible ``to``. Only floating-point tensors follow ``dtype``.

    Args:
        instance: Dataclass instance whose fields should be moved.
        device: Optional destination device.
        dtype: Optional destination dtype for floating-point tensors.
        non_blocking: Whether tensor copies may be asynchronous.
        copy: Whether to force tensor copies.

    Returns:
        Mapping from dataclass field name to moved value, ready to splat into the dataclass constructor.

    Examples:
        >>> import torch
        >>> output = StructurePredictionOutput(
        ...     pred_coords=torch.zeros(1, 3, 3),
        ...     distogram_logits=torch.zeros(1, 3, 3, 64),
        ... )
        >>> moved = move_dataclass_fields(
        ...     output, device=None, dtype=torch.float64, non_blocking=False, copy=False
        ... )
        >>> moved["pred_coords"].dtype
        torch.float64
    """

    moved: dict[str, Any] = {}
    for field in dataclasses.fields(instance):  # type: ignore[arg-type]
        moved[field.name] = _move_value(
            getattr(instance, field.name),
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
        )
    return moved


def _move_value(
    value: Any,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    non_blocking: bool,
    copy: bool,
) -> Any:
    r"""
    Move a single field value, recursing into containers and ``to``-capable objects.

    Args:
        value: Field value to move; ``None``, tensors, lists, dicts, and ``to``-capable objects are handled.
        device: Optional destination device.
        dtype: Optional destination dtype; applied only to floating-point tensors.
        non_blocking: Whether tensor copies may be asynchronous.
        copy: Whether to force tensor copies.

    Returns:
        The moved value, of the same structural type as ``value``.
    """

    if value is None:
        return None
    if isinstance(value, Tensor):
        cast = dtype if value.is_floating_point() else None
        return value.to(device=device, dtype=cast, non_blocking=non_blocking, copy=copy)
    if isinstance(value, list):
        return [_move_value(item, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_value(item, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            for key, item in value.items()
        }
    move = getattr(value, "to", None)
    if callable(move):
        return move(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
    return value
