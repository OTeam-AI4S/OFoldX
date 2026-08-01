r"""
Model and Processor registration.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import NamedTuple

import chanfig

from ..model import Model
from ..processor import Processor


class ModelRegistration(NamedTuple):
    r"""
    Classes registered under one architecture ID.

    Attributes:
        id: Architecture ID stored in model metadata.
        model: Registered model class.
        processor: Registered processor class.
    """

    id: str
    model: type[Model]
    processor: type[Processor]


def import_builtin(id: str) -> None:
    r"""
    Import the built-in family that owns an architecture or Pipeline ID, if one does.

    A family registers its IDs when its package is imported, and names each of them
    after itself, so the owner is the longest underscore-delimited prefix of ``id``
    that names a package here. An ID belonging to anyone else names none and imports
    nothing, leaving the caller to fail on the ID it was actually given.

    Args:
        id: Architecture or Pipeline identifier.
    """

    name = id
    while name:
        # A prefix that is not one identifier cannot name a family package, and asking
        # find_spec for it would look for the dotted parents it reads into instead.
        if name.isidentifier() and find_spec(f"{__package__}.{name}") is not None:
            import_module(f"{__package__}.{name}")
            return
        name = name.rpartition("_")[0]


class ModelRegistry:
    r"""
    Registry of exact Model, Processor, and configuration identities.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, ModelRegistration] = {}
        self._pairs: dict[tuple[type[Model], type[Processor]], ModelRegistration] = {}
        self._config_models: dict[type[chanfig.Config], type[Model]] = {}

    def register(
        self,
        id: str,
        *,
        model: type[Model],
        processor: type[Processor],
    ) -> None:
        r"""
        Register one architecture.

        Args:
            id: Architecture ID stored in model metadata.
            model: Model class implementing the architecture.
            processor: Processor class producing the model's inputs.
        """

        if id == "":
            raise ValueError("model architecture id must be nonempty")

        self._import_builtin(id)
        if id in self._registrations:
            raise ValueError(f"model architecture id is already registered: {id!r}")

        pair = (model, processor)
        if pair in self._pairs:
            raise ValueError("Model/Processor pair is already registered")

        config_class = model.config_class
        config_model = self._config_models.get(config_class)
        if config_model is not None and config_model is not model:
            raise ValueError("Model config class is already registered to another Model")

        registration = ModelRegistration(id, model, processor)
        self._registrations[id] = registration
        self._pairs[pair] = registration
        self._config_models.setdefault(config_class, model)

    def _import_builtin(self, id: str) -> None:
        if id not in self._registrations:
            import_builtin(id)

    def lookup_model(self, id: str) -> ModelRegistration:
        r"""
        Return the registration for an exact architecture ID.
        """

        self._import_builtin(id)
        return self._registrations[id]

    def model_for_config(self, config: chanfig.Config) -> type[Model]:
        r"""
        Return the Model class registered for a configuration's exact class.
        """

        return self._config_models[type(config)]

    def for_pair(self, model: Model, processor: Processor) -> ModelRegistration:
        r"""
        Return the registration matching both objects' exact classes.
        """

        registration = self._pairs[type(model), type(processor)]
        processor.validate_model_input_layout(model)
        return registration


#: Global registry of OFoldX model architectures.
MODELS = ModelRegistry()
