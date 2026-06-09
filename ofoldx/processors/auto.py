r"""
Auto processor factories.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from os import PathLike
from typing import Any, cast

from chanfig import Config

from .mixin import ProcessorMixin
from .registry import PROCESSORS, architecture_from_config, architecture_from_config_class
from .utils import ProcessorOutput, StructureInput


class AutoProcessor:
    r"""
    Factory that resolves the structure processor for a model from its config or a pretrained artifact.

    Never instantiated: it dispatches through the
    [`PROCESSORS`][ofoldx.processors.registry.PROCESSORS] registry, keyed by a config's
    ``architecture``, mirroring the Hugging Face ``AutoProcessor`` ergonomics.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
        >>> model_input = processor("complex.cif")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        raise TypeError("AutoProcessor cannot be instantiated")

    @classmethod
    def register(
        cls,
        config_or_architecture: type[Config] | str,
        processor_class: type[ProcessorMixin],
        *,
        exist_ok: bool = False,
    ) -> None:
        r"""
        Register a processor class for an architecture name or a config class.

        Args:
            config_or_architecture: Architecture string, or a config class whose ``architecture`` is read.
            processor_class: Processor class to register under that architecture.
            exist_ok: If ``True``, allow overwriting an existing registration.

        Raises:
            ValueError: If the architecture is already registered and ``exist_ok`` is ``False``.

        Examples:
            >>> from ofoldx.processors import AutoProcessor  # doctest: +SKIP
            >>> AutoProcessor.register("boltz1", Boltz1Processor)  # doctest: +SKIP
        """

        architecture = (
            config_or_architecture.lower()
            if isinstance(config_or_architecture, str)
            else architecture_from_config_class(config_or_architecture)
        )
        PROCESSORS.register(architecture, processor_class, exist_ok=exist_ok)

    @classmethod
    def from_config(
        cls,
        config: Config | Mapping[str, Any],
        **processor_kwargs: Any,
    ) -> Callable[[StructureInput], ProcessorOutput]:
        r"""
        Build the registered processor for a raw or typed model config.

        Args:
            config: Raw model config mapping or typed config object.
            **processor_kwargs: Extra keyword arguments forwarded to the processor constructor or ``from_config``.

        Returns:
            Processor callable that converts a structure object or path into the config's model input.

        Raises:
            ValueError: If the config does not define a registered ``architecture``.

        Examples:
            >>> from ofoldx import AutoConfig, AutoProcessor  # doctest: +SKIP
            >>> config = AutoConfig.from_pretrained("oteam/model")  # doctest: +SKIP
            >>> processor = AutoProcessor.from_config(config)  # doctest: +SKIP
            >>> model_input = processor("complex.cif")  # doctest: +SKIP
        """

        typed_config = config if isinstance(config, Config) else Config(config)
        processor_class = PROCESSORS.lookup(architecture_from_config(typed_config))
        from_config = getattr(processor_class, "from_config", None)
        if callable(from_config):
            return cast(Callable[[StructureInput], ProcessorOutput], from_config(typed_config, **processor_kwargs))
        return cast(Callable[[StructureInput], ProcessorOutput], processor_class(**processor_kwargs))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_processor_name_or_path: str | PathLike[str],
        *,
        config_overrides: Mapping[str, object] | None = None,
        repo_type: str = "model",
        revision: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
    ) -> Callable[[StructureInput], ProcessorOutput]:
        r"""
        Load any OFoldX processor from a local processor directory.

        Args:
            pretrained_processor_name_or_path: Local processor directory or Hugging Face Hub repository id.
            config_overrides: Optional config values applied after loading ``processor_config.json``.
            repo_type: Hub repository type.
            revision: Optional Hub branch, tag, or commit id.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: If ``True``, load only from local files or cache.
            token: Optional Hugging Face authentication token.
            trust_remote_code: If ``True``, allow Python modules from the artifact directory or Hub snapshot.

        Returns:
            Processor callable that converts a structure object or path into a typed model input.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/model")  # doctest: +SKIP
            >>> model_input = processor("complex.cif")  # doctest: +SKIP
        """

        return cast(
            Callable[[StructureInput], ProcessorOutput],
            ProcessorMixin.from_pretrained(
                pretrained_processor_name_or_path,
                config_overrides=config_overrides,
                repo_type=repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                trust_remote_code=trust_remote_code,
            ),
        )
