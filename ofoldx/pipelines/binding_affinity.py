r"""
Binding-affinity prediction task pipeline.

The Boltz-2 affinity checkpoint returns the same tensor container used by design generators, but its public task is
property prediction rather than sequence or backbone design. This pipeline keeps that task boundary explicit while
reusing the design-output normalization machinery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from chanfig import Config
from typing_extensions import Self

from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, publish_artifact_to_hub, resolve_pretrained_path
from ..models import (
    AutoModelForBindingAffinityPrediction,
    BaseInput,
    BindingAffinityPredictionOutput,
    DesignGeneratorOutput,
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


class BindingAffinityPipeline(Pipeline[_ModelBatch, BindingAffinityPredictionOutput]):
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
            AutoModelForBindingAffinityPrediction.from_config(config),
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
            validate_pipeline_task(manifest, "binding_affinity_prediction")
            model_path = component_path(package_path, manifest, "scorer")
            processor_path = component_path(package_path, manifest, "processor")
        else:
            model_path = package_path
            processor_path = package_path
        if model is None:
            model = AutoModelForBindingAffinityPrediction.from_pretrained(
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
            task="binding_affinity_prediction",
            architecture=component_architecture(self.model),
        ).json(save_directory / PIPELINE_MANIFEST_NAME)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        repo_type: str = "model",
        private: bool | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        create_pr: bool = False,
        delete_existing: bool = False,
        tag: str | None = None,
        model_card: bool = True,
        model_card_metadata: Mapping[str, object] | None = None,
        widget_examples: Sequence[Mapping[str, object]] | None = None,
        card_body: str | None = None,
        variant: str | None = None,
    ) -> Any:
        r"""
        Save this pipeline package and upload it to the Hugging Face Hub.
        """

        with TemporaryDirectory() as tmpdir:
            self.save_pretrained(tmpdir, variant=variant)
            return publish_artifact_to_hub(
                tmpdir,
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                token=token,
                revision=revision,
                commit_message=commit_message,
                create_pr=create_pr,
                delete_existing=delete_existing,
                tag=tag,
                model_card=model_card,
                model_card_metadata=model_card_metadata,
                widget_examples=widget_examples,
                card_body=card_body,
            )

    @torch.inference_mode()
    def __call__(self, inputs: BindingAffinityInput, **model_kwargs: Any) -> BindingAffinityPredictionOutput:
        r"""
        Score one protein-ligand complex for binding affinity.
        """

        batch = self._prepare(inputs)
        model_input = self._move_to_model(batch)
        generate = getattr(self.model, "generate", None)
        if callable(generate):
            raw_output = generate(model_input, **model_kwargs)
        else:
            raw_output = self.model(model_input, **model_kwargs)
        return _to_binding_affinity_output(raw_output, required_score=self.REQUIRED_SCORE)

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


def _to_binding_affinity_output(output: object, *, required_score: str) -> BindingAffinityPredictionOutput:
    if isinstance(output, BindingAffinityPredictionOutput):
        if required_score not in output.scores:
            raise ValueError(f"binding-affinity output scores must contain {required_score!r}")
        return output
    if isinstance(output, DesignGeneratorOutput):
        scores = dict(output.scores or {})
        if required_score not in scores:
            raise ValueError(f"binding-affinity models must return a {required_score!r} score")
        return BindingAffinityPredictionOutput(
            affinity_pred_value=scores[required_score],
            scores=scores,
            structure_prediction=output.structure_prediction,
            metadata=output.metadata,
        )
    if isinstance(output, Mapping):
        scores = {key: value for key, value in output.items() if isinstance(value, torch.Tensor)}
        if required_score not in scores:
            raise ValueError(f"binding-affinity mapping outputs must contain {required_score!r}")
        return BindingAffinityPredictionOutput(affinity_pred_value=scores[required_score], scores=scores)
    raise TypeError(
        "binding-affinity models must return BindingAffinityPredictionOutput, DesignGeneratorOutput, or a tensor "
        f"mapping, got {type(output).__name__}"
    )
