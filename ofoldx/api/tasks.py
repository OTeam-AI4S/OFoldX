# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Repository-backed Pipeline selection.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import torch

from ..data.structure import Structure
from ..io import load_structure, read_structure
from ..models.design import PositionSelection
from ..modules.backend import Backend
from ..pipelines import Pipeline, PipelineTask
from ..pipelines.loading import load_pipeline


def read_input(input: Structure | str | PathLike[str]) -> Structure:
    r"""
    Resolve a structure value or file for a convenience task call.

    Args:
        input: A canonical Structure, native ``.safetensors`` artifact, or mmCIF/PDB path.

    Returns:
        The input Structure unchanged, or the structure read from the file. Source files use
            ``read_structure`` defaults, including raising on conflicting evidence. Read a file explicitly first
            when its selection options or ReadReport are needed.
    """

    if isinstance(input, Structure):
        return input
    path = Path(input).expanduser()
    return load_structure(path) if path.suffix == ".safetensors" else read_structure(path)[0]


def pipeline(
    task: PipelineTask,
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
) -> Pipeline:
    r"""
    Load the requested task Pipeline from a model repository.

    Args:
        task: Registered biological operation, such as ``fold``, ``design``, ``score``, or ``pack``.
        model: Local native model repository or Hugging Face Hub repository ID.
        revision: Hub branch, tag, or commit to resolve for this load.
        token: Optional Hub authentication token.
        device: Device for the loaded Model and Processor.
        dtype: Floating-point dtype for the loaded Model and Processor.
        backend: Model-wide operator backend request.

    Returns:
        The concrete Pipeline selected for the repository and task. Its public signature defines the scientific
            inputs, execution controls, and result type.
    """

    return load_pipeline(
        model,
        task=task,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )


def score(
    input: Structure | str | PathLike[str],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Load a scoring Pipeline and evaluate one supplied structure.

    Args:
        input: Structure value, mmCIF/PDB path, or native structure artifact.
        model: Local model repository or Hub repository ID supporting scoring.
        revision: Hub branch, tag, or commit to resolve for this load.
        token: Optional Hub authentication token.
        device: Device for the loaded Model and Processor.
        dtype: Floating-point dtype for the loaded Model and Processor.
        backend: Model-wide operator backend request.
        seed: Seed for the concrete workflow's random stream; use this or ``generator``.
        generator: Caller-owned generator on the workflow's sampling device.

    Returns:
        The concrete scoring Pipeline's scientific result. Load a Pipeline with ``pipeline`` to reuse its model or
            pass family-specific scoring controls.
    """

    selected = pipeline(
        "score",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )
    return selected(read_input(input), seed=seed, generator=generator)


def pack(
    input: Structure | str | PathLike[str],
    *,
    model: str | PathLike[str],
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
    positions: PositionSelection = PositionSelection(),
    num_samples: int = 1,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> Any:
    r"""
    Load a side-chain packing Pipeline and run its common call.

    Args:
        input: Structure value, mmCIF/PDB path, or native structure artifact.
        model: Local model repository or Hub repository ID supporting side-chain packing.
        revision: Hub branch, tag, or commit to resolve for this load.
        token: Optional Hub authentication token.
        device: Device for the loaded Model and Processor.
        dtype: Floating-point dtype for the loaded Model and Processor.
        backend: Model-wide operator backend request.
        positions: Positions to pack under the selected family's position-selection rules.
        num_samples: Number of packing samples.
        seed: Seed for the concrete workflow's random stream; use this or ``generator``.
        generator: Caller-owned generator on the workflow's sampling device.

    Returns:
        The concrete packing Pipeline's scientific result. Load a Pipeline with ``pipeline`` to reuse its model or
            pass family-specific packing controls.
    """

    selected = pipeline(
        "pack",
        model=model,
        revision=revision,
        token=token,
        device=device,
        dtype=dtype,
        backend=backend,
    )
    return selected(
        read_input(input),
        positions=positions,
        num_samples=num_samples,
        seed=seed,
        generator=generator,
    )
