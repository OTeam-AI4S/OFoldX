r"""
Translate official PxDesign checkpoints into loadable native OFoldX models.

Maps the official condition-embedder, atom-encoder, denoiser, and distogram keys onto the native module names,
loads and strictly inspects the result, and exposes the download URLs plus the CLI entry point that writes a
converted pretrained directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from ...hub import add_hub_upload_arguments, upload_artifact_from_args
from ..alphafold3.convert import _alphafold3_reference_key, translate_reference_state
from ..mixin import no_init_weights
from .config import PxDesignConfig
from .model import PxDesignDesignGenerator

OFFICIAL_PXDESIGN_CHECKPOINT_URLS = {
    "pxdesign_v0.1.0": "https://pxdesign.tos-cn-beijing.volces.com/release_model/pxdesign_v0.1.0.pt",
}


@dataclass(slots=True, frozen=True)
class PxDesignConversionReport:
    r"""
    Strict conversion-status report for an official PxDesign checkpoint.

    Summarizes how completely an official checkpoint mapped onto the native model:
    counts of translated/native/reference keys, the keys that stayed unmatched on
    either side, and the freshly-initialized keys (e.g. the optional distogram head).

    Attributes:
        converted_keys: Number of reference keys translated to native keys.
        native_keys: Number of tensors in the native model state dict.
        reference_keys: Number of tensors in the reference checkpoint state.
        missing_keys: Native keys the checkpoint did not supply (excluding allowed
            fresh prefixes).
        unexpected_keys: Converted keys the native model did not accept.
        unexpected_reference_keys: Reference keys with no native counterpart.
        fresh_keys: Native keys left freshly initialized (allowed fresh prefixes).
        dtype_counts: Mapping from dtype name to the count of converted tensors.
    """

    converted_keys: int
    native_keys: int
    reference_keys: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    unexpected_reference_keys: tuple[str, ...]
    fresh_keys: tuple[str, ...]
    dtype_counts: dict[str, int]

    @property
    def loadable(self) -> bool:
        r"""
        Whether the checkpoint loads with no missing or unexpected keys.

        Returns:
            ``True`` when there are no missing, unexpected, or unexpected-reference
            keys (freshly-initialized keys are still permitted).
        """

        return not self.missing_keys and not self.unexpected_keys and not self.unexpected_reference_keys

    @property
    def strict_loadable(self) -> bool:
        r"""
        Whether the checkpoint loads with full parity and no fresh keys.

        Returns:
            ``True`` when [`loadable`][ofoldx.models.pxdesign.convert.PxDesignConversionReport.loadable]
            holds and no native key was left freshly initialized.
        """

        return self.loadable and not self.fresh_keys

    def raise_for_errors(self, *, require_strict: bool = False) -> None:
        r"""
        Raise if conversion did not meet the requested parity level.

        Args:
            require_strict: When ``True``, treat freshly-initialized keys as a failure
                (require strict parity); otherwise only missing/unexpected keys fail.

        Raises:
            RuntimeError: If the report is not loadable, or ``require_strict`` is set
                and fresh keys remain. The message lists up to 20 keys per category.
        """

        if self.loadable and (not require_strict or not self.fresh_keys):
            return
        lines = [
            "PxDesign checkpoint is not strict-loadable after conversion",
            f"missing: {len(self.missing_keys)}",
            f"unexpected: {len(self.unexpected_keys)}",
            f"unexpected_reference: {len(self.unexpected_reference_keys)}",
            f"fresh: {len(self.fresh_keys)}",
        ]
        lines.extend(f"  missing {key}" for key in self.missing_keys[:20])
        lines.extend(f"  unexpected {key}" for key in self.unexpected_keys[:20])
        lines.extend(f"  unexpected_reference {key}" for key in self.unexpected_reference_keys[:20])
        lines.extend(f"  fresh {key}" for key in self.fresh_keys[:20])
        raise RuntimeError("\n".join(lines))


def load_pxdesign_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    r"""
    Load an official PxDesign checkpoint from disk.

    Reads ``.safetensors`` files into a ``{"model": ...}`` mapping, or loads a ``.pt``
    checkpoint mapping directly.

    Args:
        checkpoint_path: Path to the official checkpoint (``.pt`` or ``.safetensors``).
        map_location: Device or device string to map tensors onto while loading.

    Returns:
        The checkpoint contents as a plain dict.

    Raises:
        TypeError: If a ``.pt`` checkpoint does not deserialize to a mapping.

    Examples:
        >>> checkpoint = load_pxdesign_checkpoint("pxdesign_v0.1.0.pt")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return {"model": load_file(str(checkpoint_path), device=str(map_location))}
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, Mapping):
        return dict(checkpoint)
    raise TypeError(f"expected a checkpoint mapping, got {type(checkpoint).__name__}")


def pxdesign_config_from_checkpoint(
    checkpoint: Mapping[str, Any] | None = None,
    *,
    checkpoint_variant: str = "pxdesign_v0.1.0",
) -> PxDesignConfig:
    r"""
    Build a native PxDesign config for a converted artifact.

    Starts from the default [`PxDesignConfig`][ofoldx.models.pxdesign.config.PxDesignConfig],
    stamps the checkpoint variant, and adopts the checkpoint's ``model_name`` when present.

    Args:
        checkpoint: Optional loaded checkpoint mapping; only ``model_name`` is read.
        checkpoint_variant: Variant tag recorded on the returned config.

    Returns:
        A native PxDesign config describing the converted artifact.

    Examples:
        >>> config = pxdesign_config_from_checkpoint()
        >>> config.checkpoint_variant
        'pxdesign_v0.1.0'
    """

    config = PxDesignConfig()
    config.checkpoint_variant = checkpoint_variant
    model_name = None if checkpoint is None else checkpoint.get("model_name")
    if isinstance(model_name, str) and model_name:
        config.model_name = model_name
    return config


def convert_pxdesign_state_dict(
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    config: PxDesignConfig | None = None,
    model: nn.Module | None = None,
    dtype: torch.dtype | None = None,
    allow_fresh_distogram: bool = True,
) -> dict[str, Tensor]:
    r"""
    Convert official PxDesign checkpoint keys to native OFoldX keys.

    The PxDesign condition embedder and AF3-style diffusion module are translated to
    OFoldX names. The public PxDesign repository leaves the optional distogram heads
    commented out; by default that head is allowed to remain freshly initialized.

    Args:
        checkpoint_or_state_dict: Loaded checkpoint mapping or a bare state dict.
        config: Optional native config used to build a model when ``model`` is ``None``.
        model: Optional pre-built native model providing the target key set.
        dtype: Optional dtype to cast the converted floating-point tensors to.
        allow_fresh_distogram: When ``True``, permit the distogram head to remain
            unmapped (freshly initialized).

    Returns:
        A state dict keyed by native OFoldX names, optionally cast to ``dtype``.

    Examples:
        >>> state_dict = convert_pxdesign_state_dict(load_pxdesign_checkpoint("pxdesign_v0.1.0.pt"))  # doctest: +SKIP
    """

    reference_state = _extract_reference_state(checkpoint_or_state_dict)
    if model is None:
        with no_init_weights():
            model = PxDesignDesignGenerator(PxDesignConfig() if config is None else config)
    converted = translate_reference_state(
        model,
        reference_state,
        _pxdesign_reference_key,
        allow_missing_prefixes=("distogram_head.",) if allow_fresh_distogram else (),
    )
    if dtype is None:
        return converted
    return {key: value.to(dtype=dtype) if value.is_floating_point() else value for key, value in converted.items()}


def inspect_pxdesign_conversion(
    checkpoint_or_state_dict: Mapping[str, Any],
    *,
    config: PxDesignConfig | None = None,
    model: nn.Module | None = None,
    dtype: torch.dtype | None = None,
    allow_fresh_distogram: bool = True,
) -> PxDesignConversionReport:
    r"""
    Convert and strictly inspect an official PxDesign checkpoint without saving.

    Runs the conversion and loads it into a native model to produce a parity report,
    without writing any artifact to disk.

    Args:
        checkpoint_or_state_dict: Loaded checkpoint mapping or a bare state dict.
        config: Optional native config used to build a model when ``model`` is ``None``.
        model: Optional pre-built native model providing the target key set.
        dtype: Optional dtype to cast both the converted tensors and the model to.
        allow_fresh_distogram: When ``True``, permit the distogram head to remain fresh.

    Returns:
        A [`PxDesignConversionReport`][ofoldx.models.pxdesign.convert.PxDesignConversionReport]
        describing conversion completeness and parity.

    Examples:
        >>> report = inspect_pxdesign_conversion(load_pxdesign_checkpoint("pxdesign_v0.1.0.pt"))  # doctest: +SKIP
        >>> report.strict_loadable  # doctest: +SKIP
        True
    """

    reference_state = _extract_reference_state(checkpoint_or_state_dict)
    if model is None:
        with no_init_weights():
            model = PxDesignDesignGenerator(PxDesignConfig() if config is None else config)
    converted = convert_pxdesign_state_dict(
        checkpoint_or_state_dict,
        model=model,
        dtype=dtype,
        allow_fresh_distogram=allow_fresh_distogram,
    )
    if dtype is not None:
        model.to(dtype=dtype)
    return _load_conversion_report(
        model,
        converted,
        reference_state,
        allow_fresh_prefixes=("distogram_head.",) if allow_fresh_distogram else (),
    )


def convert_pxdesign_checkpoint(
    checkpoint_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = False,
    checkpoint_variant: str | None = None,
) -> PxDesignDesignGenerator:
    r"""
    Convert an official PxDesign checkpoint into a single-file OFoldX pretrained directory.

    Loads the checkpoint, builds a native model, converts and loads the weights,
    validates parity at the requested strictness, and saves the pretrained model.

    Args:
        checkpoint_path: Path to the official PxDesign checkpoint.
        output_dir: Directory to write the OFoldX pretrained model into.
        map_location: Device or device string to map tensors onto while loading.
        dtype: Optional dtype for the converted weights and saved model.
        strict: When ``True``, require full parity with no freshly-initialized keys.
        checkpoint_variant: Variant tag; defaults to the checkpoint file stem.

    Returns:
        The converted [`PxDesignDesignGenerator`][ofoldx.models.pxdesign.model.PxDesignDesignGenerator],
        already saved to ``output_dir``.

    Raises:
        RuntimeError: If conversion does not meet the requested parity level.

    Examples:
        >>> model = convert_pxdesign_checkpoint("pxdesign_v0.1.0.pt", "out/pxdesign")  # doctest: +SKIP
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_pxdesign_checkpoint(checkpoint_path, map_location=map_location)
    config = pxdesign_config_from_checkpoint(
        checkpoint,
        checkpoint_variant=checkpoint_variant if checkpoint_variant is not None else checkpoint_path.stem,
    )
    model = PxDesignDesignGenerator(config)
    state_dict = convert_pxdesign_state_dict(
        checkpoint,
        model=model,
        dtype=dtype,
        allow_fresh_distogram=not strict,
    )
    if dtype is not None:
        model.to(dtype=dtype)
    reference_state = _extract_reference_state(checkpoint)
    report = _load_conversion_report(
        model,
        state_dict,
        reference_state,
        allow_fresh_prefixes=() if strict else ("distogram_head.",),
    )
    report.raise_for_errors(require_strict=strict)
    model.save_pretrained(output_dir)
    return model


def _extract_reference_state(checkpoint_or_state_dict: Mapping[str, Any]) -> dict[str, Tensor]:
    r"""
    Extract the tensor state dict from a checkpoint mapping.

    Unwraps a ``model``/``state_dict`` sub-mapping when present, strips the DDP
    ``module.`` prefix, and keeps only tensor-valued entries.

    Args:
        checkpoint_or_state_dict: Loaded checkpoint mapping or a bare state dict.

    Returns:
        A dict of reference tensor keys to tensors.

    Raises:
        TypeError: If no model/state-dict mapping can be found.
    """

    state = checkpoint_or_state_dict.get("model", checkpoint_or_state_dict.get("state_dict", checkpoint_or_state_dict))
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a model/state_dict mapping or be a state dict")
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }


def _load_conversion_report(
    model: nn.Module,
    converted: Mapping[str, Tensor],
    reference_state: Mapping[str, Tensor],
    *,
    allow_fresh_prefixes: tuple[str, ...],
) -> PxDesignConversionReport:
    r"""
    Load converted weights into the model and build the parity report.

    Args:
        model: Native model to load the converted state into.
        converted: Converted state dict keyed by native names.
        reference_state: Original reference tensor state, for unexpected-key detection.
        allow_fresh_prefixes: Key prefixes permitted to remain freshly initialized.

    Returns:
        The populated [`PxDesignConversionReport`][ofoldx.models.pxdesign.convert.PxDesignConversionReport].
    """

    model_state = model.state_dict()
    fresh = tuple(key for key in model_state if key.startswith(allow_fresh_prefixes) and key not in converted)
    missing = tuple(sorted(set(model_state) - set(converted) - set(fresh)))
    unexpected = tuple(sorted(set(converted) - set(model_state)))
    if not missing and not unexpected:
        completed = dict(converted)
        for key in fresh:
            completed[key] = model_state[key].detach().clone()
        model.load_state_dict(completed, strict=True)
    expected_reference_keys = _expected_reference_keys(model)
    unexpected_reference = tuple(sorted(set(reference_state) - expected_reference_keys))
    return PxDesignConversionReport(
        converted_keys=len(converted),
        native_keys=len(model.state_dict()),
        reference_keys=len(reference_state),
        missing_keys=missing,
        unexpected_keys=unexpected,
        unexpected_reference_keys=unexpected_reference,
        fresh_keys=fresh,
        dtype_counts=_dtype_counts(converted),
    )


def _expected_reference_keys(model: nn.Module) -> set[str]:
    r"""
    Return the set of reference keys the model's native keys map back to.

    Args:
        model: Native model whose state-dict keys are translated to reference keys.

    Returns:
        The set of reference keys expected in a matching checkpoint.
    """

    expected: set[str] = set()
    for model_key in model.state_dict():
        reference_key = _pxdesign_reference_key(model_key)
        if isinstance(reference_key, tuple):
            expected.update(reference_key)
        else:
            expected.add(reference_key)
    return expected


def _dtype_counts(state_dict: Mapping[str, Tensor]) -> dict[str, int]:
    r"""
    Count tensors per dtype in a state dict.

    Args:
        state_dict: State dict whose tensor dtypes are tallied.

    Returns:
        A mapping from dtype name (without the ``torch.`` prefix) to tensor count.
    """

    counts: dict[str, int] = {}
    for value in state_dict.values():
        dtype_name = str(value.dtype).removeprefix("torch.")
        counts[dtype_name] = counts.get(dtype_name, 0) + 1
    return counts


def _pxdesign_reference_key(model_key: str) -> str | tuple[str, ...]:
    r"""
    Map a native model key to its official PxDesign reference key(s).

    Rewrites the condition-embedder, atom-encoder (via the AF3 key map), denoiser, and
    distogram-head prefixes to the names used in the official checkpoint.

    Args:
        model_key: Native model state-dict key.

    Returns:
        The reference key, or a tuple of reference keys when one native key maps to
        several official tensors.
    """

    if model_key.startswith("condition_embedder.input_map."):
        return "design_condition_embedder.input_embedder.input_map." + model_key.removeprefix(
            "condition_embedder.input_map."
        )
    if model_key.startswith("condition_embedder.template_embedder.embedding."):
        return "design_condition_embedder.condition_template_embedder.embedder." + model_key.removeprefix(
            "condition_embedder.template_embedder.embedding."
        )
    if model_key.startswith("condition_embedder.atom_encoder."):
        synthetic_key = "input_embedder.atom_encoder." + model_key.removeprefix("condition_embedder.atom_encoder.")
        return _replace_reference_prefix(
            _alphafold3_reference_key(synthetic_key),
            "input_embedder.atom_attention_encoder.",
            "design_condition_embedder.input_embedder.atom_attention_encoder.",
        )
    if model_key.startswith("coordinate_denoiser."):
        return _alphafold3_reference_key(model_key)
    if model_key.startswith("distogram_head.linear."):
        return "design_distogram_head.linear_no_bias." + model_key.removeprefix("distogram_head.linear.")
    return model_key


def _replace_reference_prefix(
    reference_key: str | tuple[str, ...],
    source: str,
    target: str,
) -> str | tuple[str, ...]:
    r"""
    Replace a leading ``source`` prefix with ``target`` in one or more reference keys.

    Args:
        reference_key: A single reference key or a tuple of them.
        source: Prefix to strip when present.
        target: Prefix to prepend in its place.

    Returns:
        The reference key(s) with the prefix rewritten where it matched.
    """

    if isinstance(reference_key, tuple):
        return tuple(target + key[len(source) :] if key.startswith(source) else key for key in reference_key)
    if reference_key.startswith(source):
        return target + reference_key[len(source) :]
    return reference_key


def main(argv: list[str] | None = None) -> None:
    r"""
    Command line entry point for local checkpoint conversion.

    Parses ``checkpoint_path``, ``output_dir``, ``--strict``, and ``--dtype`` and
    invokes [`convert_pxdesign_checkpoint`][ofoldx.models.pxdesign.convert.convert_pxdesign_checkpoint].

    Examples:
        >>> main()  # doctest: +SKIP
    """

    import argparse

    parser = argparse.ArgumentParser(description="Convert a PxDesign checkpoint to OFoldX format.")
    parser.add_argument("checkpoint_path", type=Path, help="Official PxDesign .pt checkpoint")
    parser.add_argument("output_dir", type=Path, help="Output OFoldX pretrained directory")
    parser.add_argument("--strict", action="store_true", help="Require every native tensor to be mapped")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    add_hub_upload_arguments(parser)
    args = parser.parse_args(argv)
    dtype = None if args.dtype is None else getattr(torch, args.dtype)
    convert_pxdesign_checkpoint(args.checkpoint_path, args.output_dir, dtype=dtype, strict=args.strict)
    upload_result = upload_artifact_from_args(
        args.output_dir,
        args,
        commit_message="Upload converted PxDesign checkpoint",
    )
    if upload_result is not None:
        print(f"hub_upload: {upload_result}")


if __name__ == "__main__":
    main()
