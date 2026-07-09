r"""
Typed model input and output contracts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import Tensor
from torch.utils._pytree import register_pytree_node
from typing_extensions import Self

if TYPE_CHECKING:
    from ..modules.representation import Coords

_DataclassT = TypeVar("_DataclassT")


def register_pytree_dataclass(cls: _DataclassT) -> _DataclassT:
    r"""
    Register a ``@dataclass`` as a ``torch.utils._pytree`` node so ``torch.compile`` traces through it.

    Apply it *above* ``@dataclass`` on any model-boundary Input/Output dataclass. Flattening yields every
    field value as a pytree *child* (nested tensors, ``Coords``/``Single``/``Pair`` carriers, and container
    ``dict``/``list`` fields recurse automatically because they are themselves registered pytree nodes) and
    the field-name tuple as the static *context*. Unflattening rebuilds ``cls(**{name: child})`` by keyword,
    so it works for ``@dataclass(kw_only=True)`` inputs and frozen outputs alike. Python scalars (``int``,
    ``float``, ``bool``, ``str``) and ``None`` ride along as constant leaves, which dynamo handles without a
    graph break; construction of the dataclass happens outside ``forward`` at the flatten/unflatten boundary.

    Args:
        cls: The dataclass type to register. Returned unchanged so it can be used as a decorator.

    Returns:
        The same ``cls``, enabling ``@register_pytree_dataclass`` above ``@dataclass``.

    Examples:
        >>> import torch
        >>> from dataclasses import dataclass
        >>> from torch.utils._pytree import tree_flatten, tree_unflatten
        >>> @register_pytree_dataclass
        ... @dataclass(slots=True, kw_only=True)
        ... class DemoInput(BaseInput):
        ...     coords: torch.Tensor
        ...     temperature: float = 0.1
        >>> leaves, spec = tree_flatten(DemoInput(coords=torch.zeros(3)))
        >>> [type(leaf).__name__ for leaf in leaves]  # scalar fields ride along as constant leaves
        ['Tensor', 'float']
        >>> rebuilt = tree_unflatten(leaves, spec)
        >>> rebuilt.coords.shape, rebuilt.temperature
        (torch.Size([3]), 0.1)
    """

    field_names = tuple(field.name for field in dataclasses.fields(cls))  # type: ignore[arg-type]

    def _flatten(instance: Any) -> tuple[list[Any], tuple[str, ...]]:
        return [getattr(instance, name) for name in field_names], field_names

    def _unflatten(children: Any, context: tuple[str, ...]) -> Any:
        return cls(**dict(zip(context, children)))  # type: ignore[operator]

    register_pytree_node(cls, _flatten, _unflatten)  # type: ignore[arg-type]
    return cls


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


@register_pytree_dataclass
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

    @property
    def coords(self) -> Coords:
        r"""
        The predicted coordinates as a typed [`Coords`][ofoldx.modules.representation.Coords] carrier.

        The carrier checks the ``xyz`` axis and exposes named-axis operations (centering, frame application)
        so downstream code never hand-rolls axis surgery on the raw ``[..., num_samples, N_atom, 3]`` tensor.

        Examples:
            >>> import torch
            >>> output = StructurePredictionOutput(
            ...     pred_coords=torch.zeros(1, 5, 3), distogram_logits=torch.zeros(1, 5, 5, 64)
            ... )
            >>> output.coords.num_points
            5
        """

        from ..modules.representation import Coords

        return Coords(self.pred_coords)

    @property
    def plddt(self) -> Tensor | None:
        r"""
        Decode ``plddt_logits`` into per-token (or per-atom) pLDDT confidence in ``[0, 100]``.

        The standard AlphaFold pLDDT expectation: a softmax over the bin axis weighted by evenly spaced
        bin centers. This is the friendly accessor a caller wants instead of raw pre-softmax logits;
        returns ``None`` when the model has no pLDDT head.

        Returns:
            pLDDT values with the bin axis of ``plddt_logits`` reduced away, or ``None``.

        Examples:
            >>> import torch
            >>> output = StructurePredictionOutput(
            ...     pred_coords=torch.zeros(1, 4, 3),
            ...     distogram_logits=torch.zeros(1, 4, 4, 64),
            ...     plddt_logits=torch.zeros(1, 4, 50),  # uniform over 50 bins -> midpoint 50.0
            ... )
            >>> output.plddt.shape
            torch.Size([1, 4])
            >>> bool(torch.allclose(output.plddt, torch.full((1, 4), 50.0)))
            True
            >>> StructurePredictionOutput(
            ...     pred_coords=torch.zeros(1, 4, 3), distogram_logits=torch.zeros(1, 4, 4, 64)
            ... ).plddt is None
            True
        """

        if self.plddt_logits is None:
            return None
        num_bins = self.plddt_logits.shape[-1]
        bin_centers = torch.arange(num_bins, device=self.plddt_logits.device, dtype=self.plddt_logits.dtype) + 0.5
        bin_centers = bin_centers * (100.0 / num_bins)
        return (self.plddt_logits.softmax(dim=-1) * bin_centers).sum(dim=-1)

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


@register_pytree_dataclass
@dataclass(slots=True)
class BindingAffinityOutput:
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
    ) -> BindingAffinityOutput:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@register_pytree_dataclass
@dataclass(slots=True)
class DesignCandidate:
    r"""
    Base type for one candidate a design model emits.

    A design run returns a [`DesignOutput`][ofoldx.models.contract.DesignOutput] bundling these candidates. The
    base carries only the fields *every* design mode shares; the mode-specific payload lives on a subclass:

    - [`SequenceDesignCandidate`][ofoldx.models.contract.SequenceDesignCandidate] -- inverse folding: a designed
      sequence (proteinmpnn / ligandmpnn / nampnn / boltzgen inverse folding).
    - [`StructureDesignCandidate`][ofoldx.models.contract.StructureDesignCandidate] -- structure / backbone
      generation: a predicted structure, no designed sequence (odesign / pxdesign / boltzgen).
    - [`AllAtomDesignCandidate`][ofoldx.models.contract.AllAtomDesignCandidate] -- both a sequence and a
      structure (rfdiffusion backbones with placeholder tokens; the pxdesign composite that adds a sequence to a
      generated backbone).

    The candidate list is typed at the base, and the base exposes every mode-specific field as a read-only
    property defaulting to ``None``; a subclass that populates a field declares it as a real dataclass field,
    which shadows the property. So a consumer reads ``candidate.sequence_tokens`` / ``candidate.structure_prediction``
    uniformly across a heterogeneous list -- no ``getattr`` or ``isinstance`` -- and only the subclasses that fill a
    field carry its storage.

    Attributes:
        scores: Optional named score tensors.
        metadata: Optional non-tensor metadata.

    Examples:
        >>> import torch
        >>> candidate = DesignCandidate(scores={"score": torch.zeros(1)})
        >>> candidate.sequence_tokens is None and candidate.structure_prediction is None
        True
    """

    scores: dict[str, Tensor] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def sequence_logits(self) -> Tensor | None:
        r"""Sequence-token logits, or ``None`` unless a sequence-designing subclass populates it."""

        return None

    @property
    def sequence_tokens(self) -> Tensor | None:
        r"""Designed sequence tokens, or ``None`` unless a sequence-designing subclass populates it."""

        return None

    @property
    def structure_prediction(self) -> StructurePredictionOutput | None:
        r"""Nested predicted structure, or ``None`` unless a structure-designing subclass populates it."""

        return None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> DesignCandidate:
        r"""
        Return a copy with every tensor moved via ``Tensor.to``.

        Every dataclass field of the concrete subclass is moved recursively (nested ``structure_prediction`` and
        the tensors inside ``scores``); ``metadata`` and other non-tensor fields pass through unchanged.

        Args:
            device: Optional destination device.
            dtype: Optional destination dtype for floating-point tensors.
            non_blocking: Whether tensor copies may be asynchronous.
            copy: Whether to force tensor copies.

        Returns:
            A candidate of the same concrete type with recursively moved tensor fields.

        Examples:
            >>> import torch
            >>> candidate = DesignCandidate(scores={"score": torch.zeros(1)})
            >>> candidate.to(dtype=torch.float64).scores["score"].dtype
            torch.float64
        """

        return type(self)(
            **move_dataclass_fields(self, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        )


@register_pytree_dataclass
@dataclass(slots=True)
class SequenceDesignCandidate(DesignCandidate):
    r"""
    Inverse-folding candidate: a designed sequence for a fixed backbone, no predicted structure.

    Emitted by the sequence designers (proteinmpnn / ligandmpnn / nampnn and boltzgen inverse folding).

    Attributes:
        sequence_logits: Optional sequence-token logits.
        sequence_tokens: Optional sampled or decoded sequence tokens.

    Examples:
        >>> import torch
        >>> candidate = SequenceDesignCandidate(sequence_tokens=torch.zeros(6, dtype=torch.long))
        >>> candidate.sequence_tokens.shape
        torch.Size([6])
    """

    sequence_logits: Tensor | None = None
    sequence_tokens: Tensor | None = None


@register_pytree_dataclass
@dataclass(slots=True)
class StructureDesignCandidate(DesignCandidate):
    r"""
    Structure/backbone-generation candidate: a predicted structure, no designed sequence.

    Emitted by the structure generators (odesign / pxdesign / boltzgen). The coordinates live in the nested
    [`StructurePredictionOutput`][ofoldx.models.contract.StructurePredictionOutput].

    Attributes:
        structure_prediction: Optional nested structure-prediction output.

    Examples:
        >>> candidate = StructureDesignCandidate()
        >>> candidate.structure_prediction is None
        True
    """

    structure_prediction: StructurePredictionOutput | None = None


@register_pytree_dataclass
@dataclass(slots=True)
class AllAtomDesignCandidate(DesignCandidate):
    r"""
    Joint candidate carrying both a sequence and a structure.

    Emitted when a candidate has both payloads: rfdiffusion backbones (structure plus placeholder tokens) and the
    pxdesign composite, which assigns a sequence to a generated backbone. The forward-looking name for true
    sequence + all-atom-structure co-design.

    Attributes:
        sequence_logits: Optional sequence-token logits.
        sequence_tokens: Optional sampled or decoded sequence tokens.
        structure_prediction: Optional nested structure-prediction output.

    Examples:
        >>> import torch
        >>> candidate = AllAtomDesignCandidate(sequence_tokens=torch.zeros(4, dtype=torch.long))
        >>> candidate.sequence_tokens.shape
        torch.Size([4])
    """

    sequence_logits: Tensor | None = None
    sequence_tokens: Tensor | None = None
    structure_prediction: StructurePredictionOutput | None = None


@register_pytree_dataclass
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
        >>> candidate = SequenceDesignCandidate(sequence_tokens=torch.zeros(4, dtype=torch.long))
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
