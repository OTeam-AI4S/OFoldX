r"""
Model registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chanfig import Config, ConfigRegistry

from ..processors.registry import architecture_from_config


def set_architecture(config: Config, architecture: str) -> Config:
    r"""
    Set the canonical ``architecture`` key on a config instance and return it.

    The ``architecture`` key is the registry lookup key that ties a saved artifact to its config and model
    classes, so every config the registries hand out carries it explicitly.

    Args:
        config: Config instance to stamp.
        architecture: Canonical architecture key (e.g. ``"alphafold3"``).

    Returns:
        The same ``config`` instance, with its ``architecture`` key set (returned for chaining).

    Examples:
        >>> from chanfig import Config
        >>> config = set_architecture(Config(), "alphafold3")
        >>> config["architecture"]
        'alphafold3'
    """

    config["architecture"] = architecture
    return config


def set_config_class_architecture(config_class: type[Config], architecture: str) -> None:
    r"""
    Set the canonical ``architecture`` class attribute on a config class.

    Stamps the key on the class itself (not an instance) so [`AutoConfig`][ofoldx.models.auto.AutoConfig]
    and the registries can recover an architecture from a config class without instantiating it.

    Args:
        config_class: Config class to stamp.
        architecture: Canonical architecture key (e.g. ``"alphafold3"``).

    Examples:
        >>> from chanfig import Config
        >>> class DemoConfig(Config):
        ...     pass
        >>> set_config_class_architecture(DemoConfig, "demo")
        >>> DemoConfig.architecture
        'demo'
    """

    config_class.architecture = architecture  # type: ignore[attr-defined]


class Registry(ConfigRegistry):
    r"""
    Registry that builds models from their typed config.

    A [`chanfig.ConfigRegistry`][] keyed on the config ``architecture`` field: registering a model class
    under an architecture lets ``build(config)`` instantiate the right model straight from a config. Backs
    the [`AutoModel`][ofoldx.models.auto.AutoModel] task-specific registries.
    """

    key = "architecture"


class ModelConfigRegistry(ConfigRegistry):
    r"""
    Registry that maps architecture values to config classes.

    Resolves the typed config class for a raw config mapping by its ``architecture`` field; used by
    [`AutoConfig`][ofoldx.models.auto.AutoConfig] to coerce artifact configs into their concrete type.
    """

    key = "architecture"

    def coerce(self, config: Config | Mapping[str, Any]) -> Config:
        r"""
        Return the typed config instance for a raw model config mapping.

        Looks up the registered config class for the mapping's ``architecture`` field and returns a typed
        instance stamped with that architecture. An input that is already the right typed class is returned
        as-is (only re-stamped), avoiding a redundant copy.

        Args:
            config: Raw config mapping or already-typed config object.

        Returns:
            Typed config instance registered for the config ``architecture``.

        Raises:
            ValueError: If the config has no ``architecture`` field or none is registered for it.

        Examples:
            >>> from ofoldx import AutoConfig  # doctest: +SKIP
            >>> config = AutoConfig.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        """

        architecture = architecture_from_config(config)
        config_class = self.lookup(architecture)
        if isinstance(config, config_class):
            return set_architecture(config, architecture)
        return set_architecture(config_class(config), architecture)


MODEL_CONFIGS = ModelConfigRegistry()
MODELS = Registry()
STRUCTURE_PREDICTION_MODELS = Registry()
DESIGN_GENERATOR_MODELS = Registry()
BINDING_AFFINITY_MODELS = Registry()
