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
gRNAde inverse-folding Pipelines.

A design samples canonical classes on the prepared rows; ``produce_design`` builds each candidate's final target once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.structure import Structure
from ...metrics.sequence import token_recovery
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import DesignCandidate, DesignResult, Score
from ..design import DesignRecord, PositionSelection, produce_design
from ..preparation import PreparedInput
from ..processing import model_log_probabilities
from .model import GRNADE_CLASSES, GrnadeInput, GrnadeModel, GrnadeSamplingOutput
from .processor import GRNADE_LAYOUT, GrnadeProcessor, GrnadeV032Processor


@dataclass(eq=False)
class GrnadeDesignResult(DesignResult):
    r"""gRNAde designs and each candidate's design record."""

    records: tuple[DesignRecord, ...] = ()


class GrnadePipeline(PairedPipeline[GrnadeModel, GrnadeProcessor]):
    r"""
    RNA inverse folding with the released gRNAde v1 graph.

    Design conditions on observed backbone coordinates, so it takes read structures.
    """

    task = "design"

    model_architecture = "grnade"

    def prepare(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        secondary_structure: str | None = None,
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[GrnadeInput]:
        r"""Prepare one gRNAde structural context."""
        return self.processor.prepare_input(
            structure,
            instance_id=instance_id,
            design=design,
            secondary_structure=secondary_structure,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: GrnadeInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        residue_logit_bias: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeSamplingOutput:
        r"""Sample RNA identities from one prepared structural context."""
        return self.model.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            residue_logit_bias=residue_logit_bias,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[GrnadeInput],
        output: GrnadeSamplingOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> GrnadeDesignResult:
        r"""Build each candidate's final target with ``produce_design`` and score it."""
        return _design_result(prepared, output, library)

    def design(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        secondary_structure: str | None = None,
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        residue_logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> GrnadeDesignResult:
        r"""
        Sample RNA sequences from one structural context.

        Args:
            structure: One RNA structure, or aligned conformations of the same RNA; the first is the target.
            instance_id: The RNA instance when several hold RNA rows.
            design: The positions allowed to change (default every row).
            secondary_structure: Dot-bracket over the instance's rows; ``None`` is all unpaired.
            use_coordinates: Whether to include three-dimensional geometry.
            num_candidates: Number of sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            residue_logit_bias: Optional per-row additive logits over the 72 canonical classes, ``(rows, 72)``.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            seed: Optional sampling seed; mutually exclusive with ``generator``.
            generator: Optional caller-owned random generator.
            library: The definitions replacements are built from (default the configured artifact).

        Returns:
            Candidates in sampling order, each the final target of its design, with their sequence scores and design
            records.
        """

        prepared = self.prepare(
            structure,
            instance_id=instance_id,
            design=design,
            secondary_structure=secondary_structure,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=prepared.features.node_scalar_features.device)
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            residue_logit_bias=residue_logit_bias,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)


class GrnadeV032Pipeline(PairedPipeline[GrnadeModel, GrnadeV032Processor]):
    r"""RNA inverse folding with the released gRNAde v0.3.2 graph."""

    task = "design"

    model_architecture = "grnade_v0_3_2"

    def prepare(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[GrnadeInput]:
        r"""Prepare one released gRNAde v0.3.2 structural context; ``generator`` draws the conformations."""
        return self.processor.prepare_input(
            structure,
            instance_id=instance_id,
            design=design,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )

    def forward(
        self,
        input: GrnadeInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        residue_logit_bias: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> GrnadeSamplingOutput:
        r"""Sample RNA identities from one prepared structural context."""
        return self.model.sample(
            input,
            num_samples=num_candidates,
            temperature=temperature,
            residue_logit_bias=residue_logit_bias,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[GrnadeInput],
        output: GrnadeSamplingOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> GrnadeDesignResult:
        r"""Build each candidate's final target with ``produce_design`` and score it."""
        return _design_result(prepared, output, library)

    def design(
        self,
        structure: Structure | Sequence[Structure],
        *,
        instance_id: str | None = None,
        design: PositionSelection = PositionSelection(),
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        residue_logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> GrnadeDesignResult:
        r"""
        Sample with one released v0.3.2 conformation-count variant; ``seed`` also draws the conformations (a fresh
        draw without it).
        """

        prepared = self.prepare(
            structure,
            instance_id=instance_id,
            design=design,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
            generator=np.random.default_rng(seed),
        )
        generator = resolve_generator(seed, generator, device=prepared.features.node_scalar_features.device)
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            residue_logit_bias=residue_logit_bias,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)


def _design_result(
    prepared: PreparedInput[GrnadeInput], output: GrnadeSamplingOutput, library: ComponentLibrary | None
) -> GrnadeDesignResult:
    r"""Per candidate: its final target and the log probability, perplexity and native recovery of the designed rows."""
    input = prepared.features
    design = input.design_mask
    tokens = output.sequence_tokens[:, design]
    log_probabilities = model_log_probabilities(output.logits[:, design], torch.ones_like(tokens, dtype=torch.bool))
    selected = log_probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    native = design & torch.isin(input.sequence_tokens, torch.tensor(GRNADE_CLASSES, device=design.device))
    recovery = token_recovery(
        output.sequence_tokens, input.sequence_tokens, native.unsqueeze(0).expand_as(output.sequence_tokens)
    )
    totals = selected.sum(dim=-1).detach().cpu().tolist()
    means = selected.mean(dim=-1).detach().cpu().tolist()
    recoveries = recovery.detach().cpu().tolist()
    refs = prepared.context.structure.system.component_refs(input.component_index.cpu().numpy())
    scope = frozenset(ref for ref, keep in zip(refs, design.cpu().tolist()) if keep)
    native_scope = frozenset(ref for ref, keep in zip(refs, native.cpu().tolist()) if keep)
    positions = input.component_index[design].cpu().numpy()
    classes = tokens.detach().cpu().numpy()
    library = configured_library() if library is None else library
    candidates, records = [], []
    for index in range(classes.shape[0]):
        target, record = produce_design(
            prepared, classes[index], positions=positions, table=GRNADE_LAYOUT, library=library
        )
        candidates.append(
            DesignCandidate(
                target=target,
                scores={
                    "total_log_probability": Score(totals[index], "nat", "higher", scope),
                    "mean_log_probability": Score(means[index], "nat/residue", "higher", scope),
                    "perplexity": Score(math.exp(-means[index]), None, "lower", scope),
                    "sequence_recovery": Score(recoveries[index], None, "higher", native_scope),
                },
            )
        )
        records.append(record)
    return GrnadeDesignResult(candidates=tuple(candidates), records=tuple(records))
