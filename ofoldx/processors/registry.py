r"""
Processor registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar, overload

from chanfig import Config

ProcessorClassT = TypeVar("ProcessorClassT", bound=type[Any])


def architecture_from_config(config: Config | Mapping[str, Any]) -> str:
    r"""
    Return the canonical architecture key from a config.

    Args:
        config: Raw or typed model config.

    Returns:
        Lowercase architecture string.

    Raises:
        ValueError: If the config does not define a non-empty ``architecture``.

    Examples:
        >>> architecture_from_config({"architecture": "Boltz1"})
        'boltz1'
    """

    value = config.get("architecture") if isinstance(config, Mapping) else getattr(config, "architecture", None)
    if isinstance(value, str) and value:
        return value.lower()
    raise ValueError("model config must define architecture")


def architecture_from_config_class(config_class: type[Config]) -> str:
    r"""
    Return the canonical architecture key declared by a config class.

    Args:
        config_class: Model config class whose ``architecture`` class attribute is read.

    Returns:
        Lowercase architecture string.

    Raises:
        ValueError: If the class does not declare a non-empty ``architecture``.

    Examples:
        >>> from chanfig import Config
        >>> class DemoConfig(Config):
        ...     architecture = "AlphaFold3"
        >>> architecture_from_config_class(DemoConfig)
        'alphafold3'
    """

    value = getattr(config_class, "architecture", None)
    if isinstance(value, str) and value:
        return value.lower()
    raise ValueError(f"{config_class.__name__} must define architecture")


class ProcessorRegistry:
    r"""
    Registry mapping a lowercase model ``architecture`` key to its processor class.

    Model families self-register their processor here (directly or via the
    [`register`][ofoldx.processors.registry.ProcessorRegistry.register] decorator); the shared
    [`AutoProcessor`][ofoldx.processors.auto.AutoProcessor] resolves a config to its processor
    through the module-level [`PROCESSORS`][ofoldx.processors.registry.PROCESSORS] instance.

    Examples:
        >>> registry = ProcessorRegistry()
        >>> class DemoProcessor:
        ...     pass
        >>> registry.register("demo", DemoProcessor) is DemoProcessor
        True
        >>> "Demo" in registry
        True
        >>> registry.lookup("DEMO") is DemoProcessor
        True
    """

    def __init__(self) -> None:
        self._items: dict[str, type[Any]] = {}

    @overload
    def register(
        self,
        name: str,
        processor_class: None = None,
        *,
        exist_ok: bool = False,
    ) -> Callable[[ProcessorClassT], ProcessorClassT]: ...

    @overload
    def register(
        self,
        name: str,
        processor_class: ProcessorClassT,
        *,
        exist_ok: bool = False,
    ) -> ProcessorClassT: ...

    def register(
        self,
        name: str,
        processor_class: ProcessorClassT | None = None,
        *,
        exist_ok: bool = False,
    ) -> Callable[[ProcessorClassT], ProcessorClassT] | ProcessorClassT:
        r"""
        Register a processor class under a config ``architecture`` name.

        Usable directly (``register(name, cls)``) or as a class decorator (``@register(name)``); the key
        is lowercased so lookups are case-insensitive.

        Args:
            name: Processor registry key (case-insensitive).
            processor_class: Processor class to register, or ``None`` to return a decorator.
            exist_ok: If ``True``, allow re-registering an already-registered key.

        Returns:
            The registered class when ``processor_class`` is given, otherwise a class decorator that
            stores and returns the processor class.

        Raises:
            ValueError: If the key is already registered and ``exist_ok`` is ``False``.

        Examples:
            >>> registry = ProcessorRegistry()
            >>> @registry.register("demo")
            ... class DemoProcessor:
            ...     pass
            >>> "demo" in registry
            True
        """

        key = name.lower()

        def decorator(processor_class: ProcessorClassT) -> ProcessorClassT:
            r"""
            Store and return one processor class.

            Args:
                processor_class: Processor class to register.

            Returns:
                The same processor class, unchanged.
            """
            if key in self._items and not exist_ok:
                raise ValueError(f"processor architecture {name!r} is already registered")
            self._items[key] = processor_class
            return processor_class

        if processor_class is not None:
            return decorator(processor_class)
        return decorator

    def lookup(self, name: str) -> type[Any]:
        r"""
        Return the processor class registered under ``name``.

        Args:
            name: Processor registry key.

        Returns:
            Registered processor class.

        Raises:
            KeyError: If no processor is registered under ``name``.

        Examples:
            >>> registry = ProcessorRegistry()
            >>> class DemoProcessor:
            ...     pass
            >>> registry.register("demo", DemoProcessor)  # doctest: +ELLIPSIS
            <class '...DemoProcessor'>
            >>> registry.lookup("demo") is DemoProcessor
            True
            >>> registry.lookup("missing")
            Traceback (most recent call last):
                ...
            KeyError: "unknown processor architecture: 'missing'"
        """

        key = name.lower()
        try:
            return self._items[key]
        except KeyError as error:
            raise KeyError(f"unknown processor architecture: {name!r}") from error

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.lower() in self._items


PROCESSORS = ProcessorRegistry()
