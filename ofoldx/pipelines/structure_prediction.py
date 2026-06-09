r"""
Structure-prediction task pipeline.

Wraps a structure-prediction model and its structure processor, running a forward pass under inference mode and
returning the model's [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] after verifying its
type. Unlike the design pipeline there is no output coercion: the model contract is a single output type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeAlias, cast

import torch
from chanfig import Config
from typing_extensions import Self

from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, publish_artifact_to_hub, resolve_pretrained_path
from ..models import (
    AutoModelForStructurePrediction,
    BaseInput,
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
StructurePredictionInput: TypeAlias = _ModelBatch | Structure | str | PathLike[str]


class StructurePredictionPipeline(Pipeline[_ModelBatch, StructurePredictionOutput]):
    r"""
    Run a structure-prediction model from a typed batch, parsed structure, or structure file path.

    Calling the pipeline prepares the input (processing a structure or path when needed), moves the batch to the model
    device, runs a forward pass under inference mode, and returns the resulting
    [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput].

    Args:
        model: Structure-prediction model used for inference; placed in ``eval`` mode on construction.
        processor: Optional processor for structure objects or paths. Already-typed model inputs do not require one.

    Examples:
        >>> from ofoldx.pipelines import StructurePredictionPipeline
        >>> pipeline = StructurePredictionPipeline.from_pretrained("oteam/structure-model")  # doctest: +SKIP
        >>> output = pipeline("complex.cif")  # doctest: +SKIP
        >>> output.pred_coords.shape  # doctest: +SKIP
    """

    @classmethod
    def from_config(
        cls,
        config: Config | dict[str, Any],
        *,
        processor: StructureProcessor[_ModelBatch] | None = None,
    ) -> Self:
        r"""
        Build the registered structure-prediction model for ``config`` and wrap it in a pipeline.

        Args:
            config: Raw or typed model config.
            processor: Optional processor override.

        Returns:
            Pipeline wrapping the registered model and matching processor.

        Examples:
            >>> from ofoldx.pipelines import StructurePredictionPipeline
            >>> pipeline = StructurePredictionPipeline.from_config({"architecture": "alphafold3"})  # doctest: +SKIP
        """

        selected_processor = (
            cast(StructureProcessor[_ModelBatch], AutoProcessor.from_config(config)) if processor is None else processor
        )
        return cls(
            AutoModelForStructurePrediction.from_config(config),
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
        Load a structure-prediction pipeline package from a local directory or Hub repo.

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
            Loaded structure-prediction pipeline.

        Raises:
            TypeError: If unexpected loader arguments are supplied or the saved package targets another pipeline.

        Examples:
            >>> from ofoldx.pipelines import StructurePredictionPipeline
            >>> pipeline = StructurePredictionPipeline.from_pretrained("oteam/structure-pipeline")  # doctest: +SKIP
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected structure-prediction pipeline load argument(s): {unexpected}")
        package_path = resolve_pretrained_path(
            pretrained_pipeline_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                PIPELINE_MANIFEST_NAME,
                "predictor/config.json",
                f"predictor/{PIPELINE_MANIFEST_NAME}",
                f"predictor/{model_weights_name(variant)}",
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
            validate_pipeline_task(manifest, "structure_prediction")
            model_path = component_path(package_path, manifest, "predictor")
            processor_path = component_path(package_path, manifest, "processor")
        else:
            model_path = package_path
            processor_path = package_path
        if model is None:
            model = AutoModelForStructurePrediction.from_pretrained(
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
        Save a full structure-prediction pipeline package.

        Writes the predictor under ``predictor/``, the processor under ``processor/``, and a root ``manifest.json``
        that lets
        [`from_pretrained`][ofoldx.pipelines.structure_prediction.StructurePredictionPipeline.from_pretrained]
        reload the package.

        Args:
            save_directory: Destination directory.
            variant: Optional model weight variant suffix.

        Raises:
            ValueError: If the pipeline has no processor to save.

        Examples:
            >>> pipeline.save_pretrained("pretrained/my-structure-pipeline")  # doctest: +SKIP
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_component(self.model, save_directory / "predictor", "predictor", variant=variant)
        if self.processor is None:
            raise ValueError("a processor is required to save a pipeline package")
        save_component(self.processor, save_directory / "processor", "processor")
        build_pipeline_manifest(
            f"{type(self).__module__}.{type(self).__qualname__}",
            {"predictor": "predictor", "processor": "processor"},
            task="structure_prediction",
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

        Args:
            repo_id: Destination Hugging Face Hub repository id.
            repo_type: Hub repository type.
            private: Optional privacy setting used when creating the repository.
            token: Optional Hugging Face authentication token.
            revision: Optional target branch for the upload.
            commit_message: Optional commit message for the upload.
            create_pr: If ``True``, upload changes through a Hub pull request.
            delete_existing: If ``True``, delete an existing destination repo before recreating it.
            tag: Optional Hub tag to create after uploading.
            model_card: If ``True``, write/update ``README.md`` with Hub card metadata.
            model_card_metadata: Extra model-card front matter values.
            widget_examples: Optional widget examples. ``None`` uses task defaults.
            card_body: Optional Markdown body for the generated model card.
            variant: Optional model weight variant suffix.

        Returns:
            The object returned by ``huggingface_hub.HfApi.upload_folder``.

        Examples:
            >>> pipeline.push_to_hub("oteam/my-structure-pipeline")  # doctest: +SKIP
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
    def __call__(self, inputs: StructurePredictionInput, **model_kwargs: Any) -> StructurePredictionOutput:
        r"""
        Predict a structure for one input under inference mode.

        Prepares the input (processing a structure or path when needed), moves the batch to the model device, runs the
        forward pass, and returns the [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput] after
        checking its type.

        Args:
            inputs: Typed model input, parsed structure, or structure file path.
            **model_kwargs: Extra keyword arguments forwarded to the model forward pass.

        Returns:
            Predicted structure output for the input.

        Raises:
            ValueError: If ``inputs`` is a structure or path but the pipeline has no processor.
            TypeError: If ``inputs`` is neither a typed model input nor a structure input, or the model does not return
                a [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput].

        Examples:
            >>> pipeline("complex.cif")  # doctest: +SKIP
        """

        batch = self._prepare(inputs)
        output = self.model(self._move_to_model(batch), **model_kwargs)
        if not isinstance(output, StructurePredictionOutput):
            raise TypeError(
                f"structure-prediction models must return StructurePredictionOutput, got {type(output).__name__}"
            )
        return output

    def _prepare(self, inputs: StructurePredictionInput) -> _ModelBatch:
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
            raise TypeError(
                "structure-prediction pipeline inputs must be BaseInput or structure input, "
                f"got {type(inputs).__name__}"
            )
        return inputs
