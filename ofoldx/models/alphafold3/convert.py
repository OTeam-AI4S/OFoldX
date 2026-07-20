r"""
Official AlphaFold 3 checkpoint conversion entry point.

This module owns the DeepMind AlphaFold 3 parameter source. Protenix checkpoints
are converted by ``ofoldx.models.protenix.convert`` so the resulting model keeps
the ``protenix`` architecture identity.
"""

from __future__ import annotations

from os import PathLike

import torch

from .converters import deepmind


def convert_alphafold3_checkpoint(
    checkpoint_path: str | PathLike[str],
    *,
    model_name: str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    r"""
    Convert an AlphaFold 3 checkpoint into a strict-loaded native model.

    Persistence waits for the complete family Processor and standard writer.
    """

    return deepmind.convert_official_alphafold3_checkpoint(
        checkpoint_path,
        model_name=model_name,
        dtype=dtype,
    )
