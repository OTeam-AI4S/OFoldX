r"""
Side-chain packing task pipeline.

Wraps the LigandMPNN LigandMPNNPacker behind the same save/load/push boundary as the other OFoldX pipelines.
The pipeline accepts a parsed structure, a structure path, or a precomputed feature mapping and returns the
feature mapping updated with packed atom14 coordinates, masks, torsion-mixture parameters, and side-chain
confidence features.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self

from ..data.structure import Structure
from ..hub import pretrained_allow_patterns, resolve_pretrained_path
from ..models import AutoModel
from ..models.ligandmpnn import LigandMPNNPacker, pack_side_chains
from ..models.ligandmpnn.processor import structure_to_packer_features
from ..processors.utils import structure_from_input
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
from .base import Pipeline, is_structure_input

SideChainPackingInput = Mapping[str, Any] | Structure | str | PathLike[str]


class SideChainPackingPipeline(Pipeline[Any, dict[str, Any]]):
    r"""
    Pack side chains from a structure or LigandMPNN LigandMPNNPacker feature mapping.

    Args:
        model: LigandMPNN [`LigandMPNNPacker`][ofoldx.models.ligandmpnn.model.LigandMPNNPacker] module.

    Examples:
        >>> from ofoldx.pipelines import SideChainPackingPipeline
        >>> pipeline = SideChainPackingPipeline.from_pretrained("oteam/ligandmpnn-packer-noise002")  # doctest: +SKIP
        >>> packed = pipeline("backbone.cif")  # doctest: +SKIP
        >>> packed["X"].shape  # doctest: +SKIP
    """

    def __init__(self, model: LigandMPNNPacker) -> None:
        super().__init__(model)
        self.model: LigandMPNNPacker

    @classmethod
    def from_pretrained(
        cls,
        pretrained_pipeline_name_or_path: str | PathLike[str],
        *,
        model: LigandMPNNPacker | None = None,
        strict: bool = True,
        weights_device: str = "cpu",
        variant: str | None = None,
        model_config_overrides: dict[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Self:
        r"""
        Load a side-chain packing pipeline package or a bare LigandMPNNPacker model artifact.

        Args:
            pretrained_pipeline_name_or_path: Local pipeline package/model artifact or Hub repo id.
            model: Optional preloaded LigandMPNNPacker override.
            strict: Whether checkpoint tensors must exactly match model parameters.
            weights_device: Device used while materializing safetensors weights.
            variant: Optional model weight variant suffix.
            model_config_overrides: Optional config values applied during model load.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.
            **kwargs: Rejected reserved extension point for future loader arguments.

        Returns:
            Loaded side-chain packing pipeline.
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected side-chain packing pipeline load argument(s): {unexpected}")
        package_path = resolve_pretrained_path(
            pretrained_pipeline_name_or_path,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
            allow_patterns=pretrained_allow_patterns(
                PIPELINE_MANIFEST_NAME,
                "packer/config.json",
                f"packer/{PIPELINE_MANIFEST_NAME}",
                f"packer/{model_weights_name(variant)}",
                "config.json",
                model_weights_name(variant),
                trust_remote_code=trust_remote_code,
            ),
        )
        if has_pipeline_manifest(package_path):
            manifest = read_pipeline_manifest(package_path)
            validate_pipeline_class(manifest, cls)
            validate_pipeline_task(manifest, "side_chain_packing")
            model_path = component_path(package_path, manifest, "packer")
        else:
            model_path = package_path
        if model is None:
            loaded_model = AutoModel.from_pretrained(
                model_path,
                strict=strict,
                weights_device=weights_device,
                variant=variant,
                config_overrides=model_config_overrides,
                trust_remote_code=trust_remote_code,
            )
            if not isinstance(loaded_model, LigandMPNNPacker):
                raise TypeError(
                    f"side-chain packing pipeline requires LigandMPNNPacker, got {type(loaded_model).__name__}"
                )
            model = loaded_model
        return cls(model)

    def save_pretrained(self, save_directory: str | PathLike[str], *, variant: str | None = None) -> None:
        r"""
        Save a full side-chain packing pipeline package.

        The package stores the LigandMPNNPacker under ``packer/`` and writes a root pipeline manifest.
        """

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        save_component(self.model, save_directory / "packer", "packer", variant=variant)
        build_pipeline_manifest(
            f"{type(self).__module__}.{type(self).__qualname__}",
            {"packer": "packer"},
            task="side_chain_packing",
            architecture=component_architecture(self.model),
        ).json(save_directory / PIPELINE_MANIFEST_NAME)

    @torch.inference_mode()
    def __call__(self, inputs: SideChainPackingInput, **packer_kwargs: Any) -> dict[str, Any]:
        r"""
        Pack side chains for one structure or feature mapping.
        """

        features = self._prepare(inputs)
        packed = pack_side_chains(_move_feature_mapping(features, self.device), self.model, **packer_kwargs)
        return packed

    def pack(self, inputs: SideChainPackingInput, **packer_kwargs: Any) -> dict[str, Any]:
        r"""
        Pack side chains -- the house verb for side-chain packing, equivalent to calling the pipeline.

        Examples:
            >>> pipeline.pack("backbone.cif")  # doctest: +SKIP
        """

        return self(inputs, **packer_kwargs)

    def _prepare(self, inputs: SideChainPackingInput) -> dict[str, Any]:
        if is_structure_input(inputs):
            return structure_to_packer_features(structure_from_input(inputs))
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "side-chain packing pipeline inputs must be a feature mapping or structure input, "
                f"got {type(inputs).__name__}"
            )
        return dict(inputs)


def _move_feature_mapping(features: MutableMapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved
