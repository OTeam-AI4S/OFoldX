r"""
Model and Processor registration.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, NamedTuple, cast

import chanfig

from ..model import Model
from ..processor import Processor

#: A criterion factory: ``(model, processor) -> (pred, target) -> loss``. A factory (not a bare loss) so the
#: loss can close over the Model config (e.g. ``sigma_data``) and Processor config (e.g. NA-MPNN's RNA token
#: scheme) instead of a default that silently diverges from how the data was produced.
CriterionFactory = Callable[[Model, Processor], Callable[[Any, Any], Any]]


class ModelRegistration(NamedTuple):
    r"""
    Classes registered under one architecture ID.

    Attributes:
        id: Architecture ID stored in model metadata.
        model: Registered model class.
        processor: Registered processor class.
        model_config: Model configuration class.
        criterion: Training criterion factory, or ``None`` for an inference-only architecture. Its presence
            is what makes an architecture trainable.
        strategy: Which generic Runner batching/lowering/inference strategy trains it (``"sequence"``,
            ``"sequence_single"``, ``"denoising"``); ``None`` when not trainable.
    """

    id: str
    model: type[Model]
    processor: type[Processor]
    model_config: type[chanfig.Config]
    criterion: CriterionFactory | None = None
    strategy: str | None = None


BUILTIN_MODULES = {
    "alphafold2": "ofoldx.models.alphafold2",
    "alphafold2_casp14": "ofoldx.models.alphafold2",
    "alphafold2_initial_guess": "ofoldx.models.alphafold2",
    "alphafold2_multimer": "ofoldx.models.alphafold2",
    "alphafold3": "ofoldx.models.alphafold3",
    "boltz1": "ofoldx.models.boltz1",
    "boltz2": "ofoldx.models.boltz2",
    "boltz2_affinity": "ofoldx.models.boltz2",
    "boltzgen": "ofoldx.models.boltzgen",
    "boltzgen_confidence": "ofoldx.models.boltzgen",
    "boltzgen_ifold": "ofoldx.models.boltzgen",
    "esmfold2": "ofoldx.models.esmfold2",
    "esmfold2_design": "ofoldx.models.esmfold2",
    "esmfold2_experimental": "ofoldx.models.esmfold2",
    "grnade": "ofoldx.models.grnade",
    "ligandmpnn": "ofoldx.models.ligandmpnn",
    "mpnn_packer": "ofoldx.models.mpnn_packer",
    "nampnn": "ofoldx.models.nampnn",
    "odesign": "ofoldx.models.odesign",
    "odesign_ligand": "ofoldx.models.odesign",
    "odesign_nucleic_acid": "ofoldx.models.odesign",
    "odesign_oinvfold_protein": "ofoldx.models.odesign",
    "odesign_oinvfold_rna": "ofoldx.models.odesign",
    "odesign_protein": "ofoldx.models.odesign",
    "oinvfold": "ofoldx.models.oinvfold",
    "oinvfold_ligand": "ofoldx.models.oinvfold",
    "proteinmpnn": "ofoldx.models.proteinmpnn",
    "protenix": "ofoldx.models.protenix",
    "protenix_constraint": "ofoldx.models.protenix",
    "protenix_structure_condition": "ofoldx.models.protenix",
    "protenix_template": "ofoldx.models.protenix",
    "pxdesign": "ofoldx.models.pxdesign",
    "pxdesign_backbone": "ofoldx.models.pxdesign",
    "rfdiffusion": "ofoldx.models.rfdiffusion",
}


class ModelRegistry(chanfig.Registry):
    r"""
    Registry of model architectures and their processors.
    """

    by_config: dict[type[chanfig.Config], ModelRegistration]
    by_pair: dict[tuple[type[Model], type[Processor]], ModelRegistration]

    def __init__(self) -> None:
        super().__init__(case_sensitive=True)
        self.setattr("separator", "\0")
        self.setattr("by_config", {})
        self.setattr("by_pair", {})

    def register(
        self,
        id: str,
        *,
        model: type[Model],
        processor: type[Processor],
        criterion: CriterionFactory | None = None,
        strategy: str | None = None,
    ) -> None:
        r"""
        Register one architecture.

        Args:
            id: Architecture ID stored in model metadata.
            model: Model class implementing the architecture.
            processor: Processor class producing the model's input type.
            criterion: Training criterion factory; pass it to make the architecture trainable.
            strategy: Generic Runner training strategy (required with ``criterion``).

        Raises:
            ValueError: If the model, processor, or config is already registered
                incompatibly.
        """

        if id == "":
            raise ValueError("model architecture id must be a nonempty string")

        self._import_builtin(id)
        model_config = model.config_class

        pair = (model, processor)
        if pair in self.by_pair:
            raise ValueError("Model/Processor pair is already registered")
        config_owner = self.by_config.get(model_config)
        if config_owner is not None and config_owner.model is not model:
            raise ValueError(f"Model config class is already registered as {config_owner.id!r}")

        registration = ModelRegistration(
            id=id,
            model=model,
            processor=processor,
            model_config=model_config,
            criterion=criterion,
            strategy=strategy,
        )
        super().register(registration, id)
        self.by_pair[pair] = registration
        self.by_config.setdefault(model_config, registration)

    def _import_builtin(self, id: str) -> None:
        module = BUILTIN_MODULES.get(id)
        if module is not None and id not in self:
            import_module(module)

    def lookup_model(self, id: str) -> ModelRegistration:
        r"""
        Return the registration for an architecture ID.

        Args:
            id: Registered architecture ID.

        Returns:
            The matching model registration.
        """

        self._import_builtin(id)
        return cast(ModelRegistration, super().lookup(id))

    def for_config(self, config: chanfig.Config) -> ModelRegistration:
        r"""
        Return the registration for a model configuration.

        Args:
            config: Model configuration whose exact class selects the registration.

        Returns:
            The matching model registration.
        """

        return self.by_config[cast(type[chanfig.Config], type(config))]

    def for_pair(self, model: Model, processor: Processor) -> ModelRegistration:
        r"""
        Return the registration for a model and processor pair.

        Args:
            model: Registered model instance.
            processor: Registered processor instance.

        Returns:
            The registration matching both exact classes.
        """

        return self.by_pair[type(model), type(processor)]


#: Global registry of OFoldX model architectures.
MODELS = ModelRegistry()
