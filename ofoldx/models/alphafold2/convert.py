r"""
Unified AlphaFold 2 checkpoint conversion entry point.

Source-specific details live in ``ofoldx.models.alphafold2.converters``. The public
entry point selects exactly the released DeepMind NPZ/tar or OpenFold source format.
"""

from __future__ import annotations

from os import PathLike

import torch

from .converters import deepmind, openfold
from .model import AlphaFold2Model


def convert_alphafold2_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    source: str = "deepmind",
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> AlphaFold2Model:
    r"""
    Strict-load an AlphaFold 2 checkpoint into its native in-memory Model.

    Args:
        checkpoint_path: Source checkpoint path.
        source: Checkpoint source format. Supported values are ``"deepmind"`` and ``"openfold"``.
        model_name: Optional source-specific model id.
        dtype: Optional dtype for floating-point converted tensors.

    Returns:
        A strict-loaded AlphaFold2Model.
    """

    source = source.lower()
    if source == "deepmind":
        return deepmind.convert_deepmind_alphafold2_checkpoint(
            checkpoint_path,
            model_name=model_name,
            dtype=dtype,
        )
    if source == "openfold":
        if model_name is not None:
            raise ValueError("model_name selects a DeepMind NPZ member and is not valid for OpenFold checkpoints")
        return openfold.convert_openfold_alphafold2_checkpoint(checkpoint_path, dtype=dtype)
    raise ValueError(f"unsupported AlphaFold 2 checkpoint source {source!r}; expected 'deepmind' or 'openfold'")
