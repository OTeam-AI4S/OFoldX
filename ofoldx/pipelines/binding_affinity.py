r"""
Binding-affinity prediction task pipeline.

The Boltz-2 affinity checkpoint returns the same tensor container used by design generators, but its public task is
property prediction rather than sequence or backbone design. This pipeline keeps that task boundary explicit while
reusing the design-output normalization machinery.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, cast

import torch
from chanfig import Config
from typing_extensions import Self

from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, resolve_pretrained_path
from ..models import (
    AutoModel,
    BaseInput,
    BindingAffinityOutput,
    DesignCandidate,
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

_ModelBatch = BaseInput
BindingAffinityInput = _ModelBatch | Structure | str | PathLike[str]


class BindingAffinityPipeline(Pipeline[_ModelBatch, BindingAffinityOutput]):
    r"""
    Run a protein-ligand binding-affinity scorer behind a task-specific pipeline boundary.
    """

    REQUIRED_SCORE = "affinity_pred_value"

    @classmethod
    def from_config(
        cls,
        config: Config | dict[str, Any],
        *,
        processor: StructureProcessor[_ModelBatch] | None = None,
    ) -> Self:
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
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected binding-affinity pipeline load argument(s): {unexpected}")
        package_path = resolve_pretrained_path(
            pretrained_pipeline_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                PIPELINE_MANIFEST_NAME,
                "scorer/config.json",
                f"scorer/{PIPELINE_MANIFEST_NAME}",
                f"scorer/{model_weights_name(variant)}",
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
            validate_pipeline_task(manifest, "binding_affinity")
            model_path = component_path(package_path, manifest, "scorer")
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
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_component(self.model, save_directory / "scorer", "scorer", variant=variant)
        if self.processor is None:
            raise ValueError("a processor is required to save a pipeline package")
        save_component(self.processor, save_directory / "processor", "processor")
        build_pipeline_manifest(
            f"{type(self).__module__}.{type(self).__qualname__}",
            {"scorer": "scorer", "processor": "processor"},
            task="binding_affinity",
            architecture=component_architecture(self.model),
        ).json(save_directory / PIPELINE_MANIFEST_NAME)

    @torch.inference_mode()
    def __call__(self, inputs: BindingAffinityInput, **model_kwargs: Any) -> BindingAffinityOutput:
        r"""
        Score one protein-ligand complex for binding affinity.
        """

        batch = self._prepare(inputs)
        model_input = self._move_to_model(batch)
        scorer = getattr(self.model, "score", None)
        if callable(scorer):
            raw_output = scorer(model_input, **model_kwargs)
        else:
            raw_output = self.model(model_input, **model_kwargs)
        return _to_binding_affinity_output(raw_output, required_score=self.REQUIRED_SCORE)

    def score(self, inputs: BindingAffinityInput, **model_kwargs: Any) -> BindingAffinityOutput:
        r"""
        Score a protein-ligand complex -- the house verb for affinity, equivalent to calling the pipeline.

        Examples:
            >>> pipeline.score("complex.cif")  # doctest: +SKIP
        """

        return self(inputs, **model_kwargs)

    def _prepare(self, inputs: BindingAffinityInput) -> _ModelBatch:
        if is_structure_input(inputs):
            if self.processor is None:
                raise ValueError("a structure processor is required for Structure or path inputs")
            return self.processor(inputs)
        if not isinstance(inputs, BaseInput):
            raise TypeError(
                f"binding-affinity pipeline inputs must be BaseInput or structure input, got {type(inputs).__name__}"
            )
        return inputs


def _to_binding_affinity_output(output: object, *, required_score: str) -> BindingAffinityOutput:
    if isinstance(output, BindingAffinityOutput):
        if required_score not in output.scores:
            raise ValueError(f"binding-affinity output scores must contain {required_score!r}")
        return output
    if isinstance(output, DesignCandidate):
        scores = dict(output.scores or {})
        if required_score not in scores:
            raise ValueError(f"binding-affinity models must return a {required_score!r} score")
        return BindingAffinityOutput(
            affinity_pred_value=scores[required_score],
            scores=scores,
            structure_prediction=output.structure_prediction,
            metadata=output.metadata,
        )
    if isinstance(output, Mapping):
        scores = {key: value for key, value in output.items() if isinstance(value, torch.Tensor)}
        if required_score not in scores:
            raise ValueError(f"binding-affinity mapping outputs must contain {required_score!r}")
        return BindingAffinityOutput(affinity_pred_value=scores[required_score], scores=scores)
    raise TypeError(
        "binding-affinity models must return BindingAffinityOutput, a DesignCandidate, or a tensor "
        f"mapping, got {type(output).__name__}"
    )
