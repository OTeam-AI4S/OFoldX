r"""
Biomolecular design pipeline.

Wraps a design model and its structure processor and normalizes whatever the model returns (a
[`DesignOutput`][ofoldx.models.DesignOutput], a single [`DesignCandidate`][ofoldx.models.DesignCandidate]
subclass, or a bare [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput]) into a single
[`DesignOutput`][ofoldx.models.DesignOutput] of candidates, so callers see one output type regardless of model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, TypeAlias, cast

import torch
from chanfig import Config
from typing_extensions import Self

from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, resolve_pretrained_path
from ..models import (
    AutoModel,
    BaseInput,
    DesignCandidate,
    DesignOutput,
    StructureDesignCandidate,
    StructurePredictionOutput,
)
from ..processors import AutoProcessor
from .auto import (
    PIPELINE_MANIFEST_NAME,
    build_pipeline_manifest,
    component_architecture,
    component_path,
    has_pipeline_manifest,
    model_weights_name,
    read_pipeline_manifest,
    save_component,
    validate_pipeline_class,
    validate_pipeline_task,
)
from .base import Pipeline, StructureProcessor, is_structure_input

_ModelBatch: TypeAlias = BaseInput
DesignInput: TypeAlias = _ModelBatch | Structure | str | PathLike[str]


class DesignPipeline(Pipeline[_ModelBatch, DesignOutput]):
    r"""
    Generate biomolecular design candidates from a typed batch, parsed structure, or structure file path.

    Calling the pipeline runs its model's ``design`` method when available (otherwise a plain forward) under
    inference mode and coerces the result into a [`DesignOutput`][ofoldx.models.DesignOutput].

    Args:
        model: Design-generator model used for inference; placed in ``eval`` mode on construction.
        processor: Optional processor for structure objects or paths. Already-typed model inputs do not require one.

    Examples:
        >>> from ofoldx.pipelines import DesignPipeline
        >>> pipeline = DesignPipeline.from_pretrained("oteam/design-model")  # doctest: +SKIP
        >>> output = pipeline("complex.cif")  # doctest: +SKIP
        >>> output.candidates[0].sequence_tokens  # doctest: +SKIP
    """

    task: ClassVar[str] = "design"

    @classmethod
    def from_config(
        cls,
        config: Config | dict[str, Any],
        *,
        processor: StructureProcessor[_ModelBatch] | None = None,
    ) -> Self:
        r"""
        Build the registered design generator for ``config`` and wrap it in a pipeline.

        Args:
            config: Raw or typed model config.
            processor: Optional processor override.

        Returns:
            Pipeline wrapping the registered design generator and matching processor.

        Examples:
            >>> from ofoldx.pipelines import DesignPipeline
            >>> pipeline = DesignPipeline.from_config({"architecture": "proteinmpnn"})  # doctest: +SKIP
        """

        selected_processor = (
            cast(StructureProcessor[_ModelBatch], AutoProcessor.from_config(config)) if processor is None else processor
        )
        return cls(
            AutoModel.from_config(config),
            processor=selected_processor,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_pipeline_name_or_path: str | PathLike[str],
        *,
        model: torch.nn.Module | None = None,
        processor: StructureProcessor[_ModelBatch] | None = None,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        model_config_overrides: dict[str, object] | None = None,
        processor_config_overrides: dict[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Self:
        r"""
        Load a design pipeline package from a local directory or Hub repo.

        Args:
            pretrained_pipeline_name_or_path: Local pipeline package directory or Hugging Face Hub repository id.
            model: Optional preloaded model override.
            processor: Optional preloaded processor override.
            strict: Whether checkpoint tensors must exactly match model parameters.
            weights_device: Device used while materializing safetensors weights.
            variant: Optional model weight variant suffix.
            model_config_overrides: Optional model config values applied during model load.
            processor_config_overrides: Optional processor config values applied during processor load.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.
            **kwargs: Rejected reserved extension point for future loader arguments.

        Returns:
            Loaded design pipeline.

        Raises:
            TypeError: If unexpected loader arguments are supplied or the saved package targets another pipeline.

        Examples:
            >>> from ofoldx.pipelines import DesignPipeline
            >>> pipeline = DesignPipeline.from_pretrained("oteam/design-pipeline")  # doctest: +SKIP
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected design pipeline load argument(s): {unexpected}")
        package_path = resolve_pretrained_path(
            pretrained_pipeline_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                PIPELINE_MANIFEST_NAME,
                "generator/config.json",
                f"generator/{PIPELINE_MANIFEST_NAME}",
                f"generator/{model_weights_name(variant)}",
                "processor/processor_config.json",
                f"processor/{PIPELINE_MANIFEST_NAME}",
                "config.json",
                model_weights_name(variant),
                "processor_config.json",
                trust_remote_code=trust_remote_code,
            ),
        )
        if has_pipeline_manifest(package_path):
            manifest = read_pipeline_manifest(package_path)
            validate_pipeline_class(manifest, cls)
            validate_pipeline_task(manifest, cls.task)
            model_path = component_path(package_path, manifest, "generator")
            processor_path = component_path(package_path, manifest, "processor")
        else:
            model_path = package_path
            processor_path = package_path
        if model is None:
            model = AutoModel.from_pretrained(
                model_path,
                strict=strict,
                weights_device=weights_device,
                variant=variant,
                config_overrides=model_config_overrides,
                trust_remote_code=trust_remote_code,
            )
        if processor is None:
            processor = cast(
                StructureProcessor[_ModelBatch],
                AutoProcessor.from_pretrained(
                    processor_path,
                    config_overrides=processor_config_overrides,
                    trust_remote_code=trust_remote_code,
                ),
            )
        return cls(model, processor=processor)

    def save_pretrained(self, save_directory: str | PathLike[str], *, variant: str | None = None) -> None:
        r"""
        Save a full design pipeline package.

        Writes the generator under ``generator/``, the processor under ``processor/``, and a root ``manifest.json``
        that lets [`from_pretrained`][ofoldx.pipelines.design.DesignPipeline.from_pretrained] reload the package.

        Args:
            save_directory: Destination directory.
            variant: Optional model weight variant suffix.

        Raises:
            ValueError: If the pipeline has no processor to save.

        Examples:
            >>> pipeline.save_pretrained("pretrained/my-design-pipeline")  # doctest: +SKIP
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_component(self.model, save_directory / "generator", "generator", variant=variant)
        if self.processor is None:
            raise ValueError("a processor is required to save a pipeline package")
        save_component(self.processor, save_directory / "processor", "processor")
        build_pipeline_manifest(
            f"{type(self).__module__}.{type(self).__qualname__}",
            {"generator": "generator", "processor": "processor"},
            task=type(self).task,
            architecture=component_architecture(self.model),
        ).json(save_directory / PIPELINE_MANIFEST_NAME)

    @torch.inference_mode()
    def __call__(self, inputs: DesignInput, **model_kwargs: Any) -> DesignOutput:
        r"""
        Run the design generator on one input and normalize its output under inference mode.

        Prepares the input (processing a structure or path when needed), moves the batch to the model device, calls the
        model's ``design`` method when present (otherwise a plain forward), and coerces the result into a
        [`DesignOutput`][ofoldx.models.DesignOutput]. This is the single-model design boundary; multi-model design
        workflows (e.g. backbone generation → sequence assignment → structure-predictor filtering) are the composite
        [`AllAtomDesignPipeline`][ofoldx.pipelines.design.AllAtomDesignPipeline], which injects the sub-providers as
        capability-typed components rather than importing model families.

        Args:
            inputs: Typed model input, parsed structure, or structure file path.
            **model_kwargs: Extra keyword arguments forwarded to the model or its ``design`` method.

        Returns:
            Normalized design output holding one or more candidates.

        Raises:
            ValueError: If ``inputs`` is a structure or path but the pipeline has no processor.
            TypeError: If ``inputs`` is neither a typed model input nor a structure input, or the model returns an
                unrecognized output type.

        Examples:
            >>> pipeline("complex.cif")  # doctest: +SKIP
        """

        batch = self._prepare(inputs)
        model_input = self._move_to_model(batch)
        designer = getattr(self.model, "design", None)
        if callable(designer):
            raw_output = designer(model_input, **model_kwargs)
        else:
            raw_output = self.model(model_input, **model_kwargs)
        return _to_design_output(raw_output)

    @torch.inference_mode()
    def design(self, inputs: DesignInput, **model_kwargs: Any) -> DesignOutput:
        r"""
        Design sequences or backbones.

        The house verb for design; equivalent to calling the pipeline; provided so callers can spell the intent
        explicitly.

        Args:
            inputs: Typed model input, parsed structure, or structure file path.
            **model_kwargs: Extra keyword arguments forwarded to the model or its ``design`` method.

        Returns:
            Normalized design output.

        Examples:
            >>> pipeline.design("complex.cif")  # doctest: +SKIP
        """

        return self(inputs, **model_kwargs)

    def _prepare(self, inputs: DesignInput) -> _ModelBatch:
        r"""
        Coerce a pipeline input into a typed model batch, running the processor for structure inputs.

        Args:
            inputs: Typed model input, parsed structure, or structure file path.

        Returns:
            Typed model batch ready to move to the model device.

        Raises:
            ValueError: If ``inputs`` needs a processor but the pipeline has none.
            TypeError: If ``inputs`` is neither a typed model input nor a structure input.
        """

        if is_structure_input(inputs):
            if self.processor is None:
                raise ValueError("a structure processor is required for Structure or path inputs")
            return self.processor(inputs)
        if not isinstance(inputs, BaseInput):
            raise TypeError(f"design pipeline inputs must be BaseInput or structure input, got {type(inputs).__name__}")
        return inputs


def _to_design_output(output: object) -> DesignOutput:
    r"""
    Normalize a raw design-generator return value into a [`DesignOutput`][ofoldx.models.DesignOutput].

    A [`DesignOutput`][ofoldx.models.DesignOutput] passes through unchanged; a single
    [`DesignCandidate`][ofoldx.models.DesignCandidate] (any mode subclass) becomes a one-candidate output; a bare
    [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] is wrapped as a structure-only candidate.

    Args:
        output: Value returned by a design model's ``design`` or forward call.

    Returns:
        Normalized design output holding one or more candidates.

    Raises:
        TypeError: If ``output`` is not one of the three recognized design return types.
    """

    if isinstance(output, DesignOutput):
        return output
    if isinstance(output, DesignCandidate):
        return DesignOutput([output])
    if isinstance(output, StructurePredictionOutput):
        return DesignOutput([StructureDesignCandidate(structure_prediction=output)])
    raise TypeError(
        "design models must return DesignOutput, a DesignCandidate, or StructurePredictionOutput, "
        f"got {type(output).__name__}"
    )


class SequenceDesignPipeline(DesignPipeline):
    r"""
    Design sequences for a fixed backbone (inverse folding): a structure in, sequence candidates out.

    The mode pipeline for sequence design -- ProteinMPNN, LigandMPNN, NAMPNN, and other inverse-folding models.
    Mechanically identical to the base [`DesignPipeline`][ofoldx.pipelines.design.DesignPipeline] (run the model's
    ``design`` verb, normalize to a [`DesignOutput`][ofoldx.models.DesignOutput]); the mode lives in the returned
    [`SequenceDesignCandidate`][ofoldx.models.SequenceDesignCandidate] and in the structure-in / sequence-out
    contract, not in the control flow.

    Examples:
        >>> from ofoldx.pipelines import SequenceDesignPipeline
        >>> pipeline = SequenceDesignPipeline.from_pretrained("oteam/proteinmpnn")  # doctest: +SKIP
        >>> output = pipeline("backbone.cif")  # doctest: +SKIP
        >>> output.candidates[0].sequence_tokens  # doctest: +SKIP
    """

    task: ClassVar[str] = "sequence_design"


class BackboneDesignPipeline(DesignPipeline):
    r"""
    Generate novel backbones de novo from a design specification: a spec in, backbone candidates out.

    The mode pipeline for backbone (de novo structure) design -- RFdiffusion, PxDesign, ODesign, and other
    generators conditioned on a specification (contigs, hotspots, length, symmetry) rather than an input
    structure. Mechanically identical to the base [`DesignPipeline`][ofoldx.pipelines.design.DesignPipeline]; the
    mode lives in the returned [`StructureDesignCandidate`][ofoldx.models.StructureDesignCandidate] and in the
    spec-in / backbone-out contract.

    Examples:
        >>> from ofoldx.pipelines import BackboneDesignPipeline
        >>> pipeline = BackboneDesignPipeline.from_pretrained("oteam/rfdiffusion")  # doctest: +SKIP
        >>> output = pipeline(spec)  # doctest: +SKIP
        >>> output.candidates[0].structure_prediction.pred_coords  # doctest: +SKIP
    """

    task: ClassVar[str] = "backbone_design"


class AllAtomDesignPipeline(BackboneDesignPipeline):
    r"""
    Design an all-atom complex: generate a backbone, assign its sequence, then filter by structure prediction.

    The composite mode pipeline -- backbone generation (its ``model``) followed by two optional, *injected* stages:
    sequence assignment through a [`Designer`][ofoldx.models.capability.Designer] ``sequence_designer`` (e.g.
    ProteinMPNN) and filtering through a [`Folder`][ofoldx.models.capability.Folder] ``structure_filter`` (e.g.
    AlphaFold3 / Protenix). It imports no model family: the sub-providers and their family-specific ``assign_stage``
    / ``filter_stage`` glue are injected at construction, so composing sibling models introduces no cross-family
    coupling. Returns [`AllAtomDesignCandidate`][ofoldx.models.AllAtomDesignCandidate]s carrying both sequence and
    structure.

    Args:
        model: The backbone design generator (e.g. a PxDesign model).
        processor: Optional structure processor for the backbone generator.
        sequence_designer: Optional injected `Designer` used by ``assign_stage`` for sequence assignment.
        structure_filter: Optional injected `Folder` used by ``filter_stage`` for candidate filtering.
        assign_stage: Optional family glue ``(candidates, *, batch, model, **cfg) -> DesignCandidate`` that assigns
            sequences with ``sequence_designer`` (e.g.
            [`ofoldx.models.pxdesign.assign_sequences`][ofoldx.models.pxdesign.stages.assign_sequences]).
        filter_stage: Optional family glue ``(candidates, *, batch, predictor, **cfg) -> DesignCandidate`` that
            scores/filters with ``structure_filter`` (e.g.
            [`ofoldx.models.pxdesign.filter_candidates`][ofoldx.models.pxdesign.stages.filter_candidates]).

    Examples:
        >>> from ofoldx.pipelines import AllAtomDesignPipeline
        >>> from ofoldx.models.pxdesign import assign_sequences, filter_candidates
        >>> pipeline = AllAtomDesignPipeline(  # doctest: +SKIP
        ...     backbone_model,
        ...     processor=backbone_processor,
        ...     sequence_designer=proteinmpnn,
        ...     structure_filter=alphafold3,
        ...     assign_stage=assign_sequences,
        ...     filter_stage=filter_candidates,
        ... )
        >>> output = pipeline(spec, sequence_assignment=True, filtering={"thresholds": {}})  # doctest: +SKIP
    """

    task: ClassVar[str] = "all_atom_design"

    def __init__(
        self,
        model: torch.nn.Module,
        processor: StructureProcessor[_ModelBatch] | None = None,
        *,
        sequence_designer: Any | None = None,
        structure_filter: Any | None = None,
        assign_stage: Callable[..., DesignCandidate] | None = None,
        filter_stage: Callable[..., DesignCandidate] | None = None,
    ) -> None:
        super().__init__(model, processor)
        self.sequence_designer = sequence_designer
        self.structure_filter = structure_filter
        self.assign_stage = assign_stage
        self.filter_stage = filter_stage

    @torch.inference_mode()
    def __call__(self, inputs: DesignInput, **model_kwargs: Any) -> DesignOutput:
        r"""Run backbone generation plus the injected sequence-assignment and filtering stages; see ``design``."""

        return self.design(inputs, **model_kwargs)

    @torch.inference_mode()
    def design(
        self,
        inputs: DesignInput,
        *,
        sequence_assignment: Mapping[str, Any] | bool | None = None,
        filtering: Mapping[str, Any] | bool | None = None,
        **model_kwargs: Any,
    ) -> DesignOutput:
        r"""
        Generate backbones, then optionally assign sequences and filter, using the injected components.

        Args:
            inputs: Typed model input, parsed structure, or structure file path (the design specification).
            sequence_assignment: Enable/configure the sequence-assignment stage (uses ``sequence_designer`` +
                ``assign_stage``). ``None``/``False`` skip it; ``True`` runs it with defaults; a mapping passes
                keyword options to ``assign_stage``.
            filtering: Enable/configure the filtering stage (uses ``structure_filter`` + ``filter_stage``), with the
                same convention as ``sequence_assignment``.
            **model_kwargs: Forwarded to the backbone generator's ``design`` verb.

        Returns:
            A normalized [`DesignOutput`][ofoldx.models.DesignOutput] of the (optionally sequenced and filtered)
            candidates.
        """

        batch = self._prepare(inputs)
        model_input = self._move_to_model(batch)
        designer = getattr(self.model, "design", None)
        raw = designer(model_input, **model_kwargs) if callable(designer) else self.model(model_input, **model_kwargs)

        assignment = _stage_config("sequence_assignment", sequence_assignment)
        if assignment is not None and self.sequence_designer is not None and self.assign_stage is not None:
            assignment.setdefault("model", self.sequence_designer)
            assignment.setdefault("generator", model_kwargs.get("generator"))
            raw = self.assign_stage(raw, batch=model_input, **assignment)
        filtering_config = _stage_config("filtering", filtering)
        if filtering_config is not None and self.structure_filter is not None and self.filter_stage is not None:
            filtering_config.setdefault("predictor", self.structure_filter)
            raw = self.filter_stage(raw, batch=model_input, **filtering_config)
        return _to_design_output(raw)


def _stage_config(name: str, config: Mapping[str, Any] | bool | None) -> dict[str, Any] | None:
    r"""
    Normalize a design-stage config into an explicit kwargs dict or ``None`` (skip).

    ``None`` and ``False`` disable the stage; ``True`` enables it with defaults; a mapping is copied to a plain dict
    of keyword arguments.

    Args:
        name: Stage name, used to phrase the error on an unsupported type.
        config: The raw stage config: mapping, bool, or ``None``.

    Returns:
        A kwargs dict when the stage is enabled, or ``None`` to skip it.

    Raises:
        TypeError: If ``config`` is an unsupported type for a known stage name.
        ValueError: If ``name`` is not a recognized stage.

    Examples:
        >>> _stage_config("filtering", None) is None
        True
        >>> _stage_config("filtering", True)
        {}
        >>> _stage_config("filtering", False) is None
        True
        >>> _stage_config("sequence_assignment", {"num_samples": 2})
        {'num_samples': 2}
    """

    if config is None:
        return None
    if isinstance(config, bool):
        return {} if config else None
    if isinstance(config, Mapping):
        return dict(config)
    if name in {"sequence_assignment", "filtering"}:
        raise TypeError(f"{name} must be a mapping, bool, or None")
    raise ValueError(f"unknown design stage: {name!r}")
