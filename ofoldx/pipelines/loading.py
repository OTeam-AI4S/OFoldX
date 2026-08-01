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
Load the concrete workflow declared by a Pipeline repository.
"""

from __future__ import annotations

from os import PathLike
from typing import cast

import torch

from ..hub import ResolvedRepository, load_json_object, resolve_repository
from ..modules.backend import Backend
from .base import Pipeline
from .registry import PIPELINES, PipelineTask


def load_pipeline(
    repository: str | PathLike[str] | ResolvedRepository,
    *,
    task: PipelineTask | None = None,
    revision: str | None = None,
    token: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    backend: Backend | None = None,
) -> Pipeline:
    r"""Load a declared workflow, optionally requiring a particular task."""

    resolved = resolve_repository(
        repository,
        revision=revision,
        token=token,
        allow_patterns=("pipeline.json", "model.json"),
    )
    marker = resolved.path / "pipeline.json"
    if marker.exists():
        pipeline_id = load_json_object(marker)["pipeline"]
        if not isinstance(pipeline_id, str):
            raise TypeError("pipeline.json pipeline must be a string")
        registration = PIPELINES.lookup_pipeline(pipeline_id)
        if task is not None and task != registration.task:
            if registration.model_architecture is None:
                raise ValueError(f"pipeline {pipeline_id!r} does not support {task}")
            registration = PIPELINES.for_model(registration.model_architecture, task)
    else:
        if task is None:
            raise ValueError("a model-only repository requires an explicit task")
        architecture = load_json_object(resolved.path / "model.json")["architecture"]
        if not isinstance(architecture, str):
            raise TypeError("model.json architecture must be a string")
        registration = PIPELINES.for_model(architecture, task)
    return cast(
        Pipeline,
        registration.pipeline.from_pretrained(
            resolved,
            token=token,
            device=device,
            dtype=dtype,
            backend=backend,
        ),
    )
