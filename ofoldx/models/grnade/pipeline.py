r"""
Decoded gRNAde inverse folding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from typing import cast

import torch

from ...data.components import get_component
from ...data.molecular import (
    ComponentReference,
    repartition_entities_by_instance_sequence,
    replace_components,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_structure_input
from ...results import DesignCandidate, DesignResult, Score
from .model import GRNADE_COMPONENT_IDS, GrnadeInput, GrnadeModel, GrnadeSamplingOutput
from .processor import GrnadeProcessor, GrnadeV032Processor


class GrnadePipeline(PairedPipeline[GrnadeModel, GrnadeProcessor]):
    r"""
    RNA inverse folding with the released gRNAde v1 graph.

    Load a published pair with [`ofoldx.pipeline`][]. Design conditions on observed
    backbone coordinates, so it takes a structure read from a file rather than a topology
    built from a sequence.
    """

    model_architecture = "grnade"

    def prepare_design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        base_pairs: Sequence[tuple[ComponentReference, ComponentReference]] = (),
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
    ) -> GrnadeInput:
        r"""Prepare one gRNAde structural context."""

        structure = cast(Structure, resolve_structure_input(input)) if isinstance(input, str | PathLike) else input
        return self.processor(
            structure,
            instance_id=instance_id,
            design_positions=design_positions,
            base_pairs=base_pairs,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
        )

    def design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        base_pairs: Sequence[tuple[ComponentReference, ComponentReference]] = (),
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Sample RNA sequences from one structural context.

        Args:
            input: One RNA ``Structure``, aligned conformations, or one local
                structure path.
            instance_id: Target RNA instance when the input is ambiguous.
            design_positions: RNA component occurrences allowed to change.
            base_pairs: Undirected secondary-structure pairs.
            use_coordinates: Whether to include three-dimensional geometry.
            num_candidates: Number of sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            logit_bias: Optional additive logits in gRNAde vocabulary order.
            filter_by_occupancy: Whether zero-occupancy coordinates are unobserved.
            seed: Optional sampling seed.
            generator: Optional caller-owned random generator.

        Returns:
            Sampled systems and their sequence scores.
        """

        model_input = self.prepare_design(
            input,
            instance_id=instance_id,
            design_positions=design_positions,
            base_pairs=base_pairs,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(
            seed,
            generator,
            device=model_input.node_scalar_features.device,
        )
        sampled = self.model.sample(
            model_input,
            num_samples=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
            generator=generator,
        )
        return self.decode_design(model_input, sampled)

    def decode_design(self, input: GrnadeInput, output: GrnadeSamplingOutput) -> DesignResult:
        r"""Decode raw gRNAde samples without executing the Model."""

        return _decode_design(input, output)


class GrnadeV032Pipeline(PairedPipeline[GrnadeModel, GrnadeV032Processor]):
    r"""RNA inverse folding with the released gRNAde v0.3.2 graph."""

    model_architecture = "grnade_v0_3_2"

    def prepare_design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        use_coordinates: bool = True,
        filter_by_occupancy: bool = True,
        generator: torch.Generator | None = None,
    ) -> GrnadeInput:
        r"""Prepare one released gRNAde v0.3.2 structural context."""

        structure = cast(Structure, resolve_structure_input(input)) if isinstance(input, str | PathLike) else input
        return self.processor(
            structure,
            instance_id=instance_id,
            design_positions=design_positions,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )

    def design(
        self,
        input: Structure | Sequence[Structure] | str | PathLike[str],
        *,
        instance_id: str | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        use_coordinates: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        logit_bias: torch.Tensor | None = None,
        filter_by_occupancy: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""Sample with one released v0.3.2 conformation-count variant."""

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.prepare_design(
            input,
            instance_id=instance_id,
            design_positions=design_positions,
            use_coordinates=use_coordinates,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )
        sampled = self.model.sample(
            model_input,
            num_samples=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
            generator=generator,
        )
        return self.decode_design(model_input, sampled)

    def decode_design(self, input: GrnadeInput, output: GrnadeSamplingOutput) -> DesignResult:
        r"""Decode raw gRNAde v0.3.2 samples without executing the Model."""

        return _decode_design(input, output)


def _decode_design(
    input: GrnadeInput,
    output: GrnadeSamplingOutput,
) -> DesignResult:
    design = input.design_mask.unsqueeze(0).expand_as(output.sequence_tokens)
    count = design.sum(dim=-1)
    known_design = design & (input.sequence_tokens < len(GRNADE_COMPONENT_IDS))
    known_count = known_design.sum(dim=-1)

    selected_log_probability = output.log_probabilities.gather(
        -1,
        output.sequence_tokens.unsqueeze(-1),
    ).squeeze(-1)
    total_log_probability = (selected_log_probability * design).sum(dim=-1)
    mean_log_probability = total_log_probability / count
    native_recovery = ((output.sequence_tokens == input.sequence_tokens) & known_design).sum(dim=-1) / known_count

    component_index = input.component_index.detach().cpu().tolist()
    component_references = tuple(input.structure.system.component_reference(index) for index in component_index)
    host_design = input.design_mask.detach().cpu().tolist()
    scope = frozenset(
        reference for reference, selected in zip(component_references, host_design, strict=True) if selected
    )
    native_scope = frozenset(
        reference
        for reference, selected, token in zip(
            component_references,
            host_design,
            input.sequence_tokens.detach().cpu().tolist(),
            strict=True,
        )
        if selected and token < len(GRNADE_COMPONENT_IDS)
    )
    selected_component_index = input.component_index[input.design_mask].to(
        device=input.structure.system.component_symbol.device
    )
    host_tokens = output.sequence_tokens.detach().cpu().tolist()
    totals = total_log_probability.detach().cpu().tolist()
    means = mean_log_probability.detach().cpu().tolist()
    recoveries = native_recovery.detach().cpu().tolist()

    candidates = []
    for token_row, total, mean, recovery in zip(
        host_tokens,
        totals,
        means,
        recoveries,
        strict=True,
    ):
        replacements = tuple(
            get_component(GRNADE_COMPONENT_IDS[token])
            for token, selected in zip(token_row, host_design, strict=True)
            if selected
        )
        original_system = input.structure.system
        system, _, _ = replace_components(
            original_system,
            selected_component_index,
            replacements,
        )
        system = repartition_entities_by_instance_sequence(original_system, system)
        candidates.append(
            DesignCandidate(
                system=system,
                scores={
                    "total_log_probability": Score(
                        total,
                        unit="nat",
                        preferred="higher",
                        scope=scope,
                    ),
                    "mean_log_probability": Score(
                        mean,
                        unit="nat/residue",
                        preferred="higher",
                        scope=scope,
                    ),
                    "perplexity": Score(
                        math.exp(-mean),
                        preferred="lower",
                        scope=scope,
                    ),
                    "native_recovery": Score(
                        recovery,
                        preferred="higher",
                        scope=native_scope,
                    ),
                },
            )
        )
    return DesignResult(candidates=tuple(candidates))
