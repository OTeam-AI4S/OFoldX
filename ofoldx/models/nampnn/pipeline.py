r"""
NA-MPNN sequence design, scoring, and probability Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...data.components import get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    ComponentReference,
    MolecularSystem,
    repartition_entities_by_instance_sequence,
    replace_components,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_structure_input
from ...results import AutoregressiveProbabilityResult, DesignCandidate, DesignResult, Metric, Score, ScoreResult
from .model import (
    NA_MPNN_COMPONENT_IDS,
    NA_MPNN_DNA_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SHARED_INPUT_TOKENS,
    NaMpnnInput,
    NaMpnnModel,
    NaMpnnOutput,
    NaMpnnSamplingOutput,
)
from .processor import NaMpnnProcessor


class NaMpnnPipeline(PairedPipeline[NaMpnnModel, NaMpnnProcessor]):
    r"""
    Protein and nucleic-acid sequence design pipeline for NA-MPNN.

    Load a published pair with [`ofoldx.pipeline`][]. Every verb conditions on observed
    backbone coordinates, so it takes a structure read from a file rather than a topology
    built from a sequence.
    """

    model_architecture = "nampnn"

    def prepare_design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
    ) -> NaMpnnInput:
        r"""Prepare one fixed-backbone NA-MPNN design input."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(
            structure,
            positions=positions,
            nucleic_acid_only=nucleic_acid_only,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        num_candidates: int = 1,
        temperature: float = 0.1,
        logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Design mixed-polymer sequences on one fixed backbone.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences allowed to change.
            nucleic_acid_only: Whether to design only DNA and RNA while retaining protein context.
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the NA-MPNN vocabulary.
            pair_logit_bias: Optional adjacent-token bias matrix with shape ``(33, 33)``.
            global_omit_tokens: One-letter residue tokens excluded from generation.
            num_candidates: Number of complete sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            logit_bias: Optional vocabulary-wide sampling bias.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.

        Returns:
            A ``DesignResult`` containing candidates in sampling order.
        """

        model_input = self.prepare_design(
            input,
            positions=positions,
            nucleic_acid_only=nucleic_acid_only,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        output = self.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            logit_bias=logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )
        return self.decode_design(model_input, output)

    def decode_design(
        self,
        input: NaMpnnInput,
        output: NaMpnnSamplingOutput,
    ) -> DesignResult:
        r"""Decode raw NA-MPNN samples into mixed-polymer molecular systems."""

        num_candidates = output.sequence_tokens.shape[0]
        losses = -output.log_probabilities.gather(-1, output.sequence_tokens.unsqueeze(-1)).squeeze(-1)
        designed = (
            (input.residue_mask & input.design_positions).repeat_interleave(num_candidates, dim=0).to(losses.dtype)
        )
        design_count = designed.sum(dim=-1)
        confidence = torch.exp(-(losses * designed).sum(dim=-1) / design_count)
        matches = (output.sequence_tokens == input.sequence_tokens.repeat_interleave(num_candidates, dim=0)).to(
            losses.dtype
        )
        recovery = (matches * designed).sum(dim=-1) / design_count

        references = _component_references(input.structure.system, input.component_index[0])
        design_rows = (input.design_positions[0] & input.residue_mask[0]).tolist()
        designed_scope = frozenset(
            reference for reference, selected in zip(references, design_rows) if reference is not None and selected
        )
        host_sequence_tokens = output.sequence_tokens.detach().cpu().tolist()
        host_confidence = confidence.detach().cpu().tolist()
        host_recovery = recovery.detach().cpu().tolist()
        candidates = tuple(
            DesignCandidate(
                system=_realize_sequence(
                    input.structure.system,
                    input.component_index[0],
                    input.polymer_type_index[0],
                    host_sequence_tokens[index],
                    design_rows,
                ),
                scores={
                    "designed_sequence_confidence": Score(
                        float(host_confidence[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                    "sequence_recovery": Score(
                        float(host_recovery[index]),
                        preferred="higher",
                        scope=designed_scope,
                    ),
                },
            )
            for index in range(num_candidates)
        )
        return DesignResult(candidates=candidates)

    def prepare_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
    ) -> NaMpnnInput:
        r"""Prepare one observed or explicit mixed-polymer sequence for evaluation."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(structure, sequence=sequence, positions=positions)

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score a mixed-polymer sequence with autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            sequence: Optional replacement sequence to evaluate.
            positions: Residue occurrences evaluated after fixed sequence context.
            num_decoding_orders: Number of independently sampled decoding orders.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.

        Returns:
            A ``ScoreResult`` containing designed and global sequence NLL scores.
        """

        model_input = self.prepare_probabilities(input, sequence=sequence, positions=positions)
        if not bool((model_input.residue_mask & model_input.design_positions).any()):
            raise ValueError("score requires at least one valid scoring position")
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        output = self.model.score(
            model_input,
            num_decoding_orders=num_decoding_orders,
            num_neighbors=num_neighbors,
            generator=generator,
        )
        return self.decode_score(model_input, output)

    def decode_score(
        self,
        input: NaMpnnInput,
        output: NaMpnnOutput,
    ) -> ScoreResult:
        r"""Decode raw NA-MPNN logits into designed and global sequence NLL scores."""

        if not bool((input.residue_mask & input.design_positions).any()):
            raise ValueError("score requires at least one valid scoring position")
        evaluations = output.logits.shape[0]
        sequence_tokens = input.sequence_tokens.repeat_interleave(evaluations, dim=0)
        losses = -output.log_probabilities.gather(-1, sequence_tokens.unsqueeze(-1)).squeeze(-1)
        valid = input.residue_mask.repeat_interleave(evaluations, dim=0).to(losses.dtype)
        designed = (input.residue_mask & input.design_positions).repeat_interleave(evaluations, dim=0).to(losses.dtype)
        designed_values = (losses * designed).sum(dim=-1) / designed.sum(dim=-1)
        global_values = (losses * valid).sum(dim=-1) / valid.sum(dim=-1)
        host_designed_values = designed_values.detach().cpu()
        host_global_values = global_values.detach().cpu()
        if host_designed_values.dtype == torch.bfloat16:
            host_designed_values = host_designed_values.float()
            host_global_values = host_global_values.float()
        designed_evaluations = host_designed_values.numpy()
        global_evaluations = host_global_values.numpy()
        references = _component_references(input.structure.system, input.component_index[0])
        design_positions = input.design_positions[0].detach().cpu().tolist()
        residue_mask = input.residue_mask[0].detach().cpu().tolist()
        designed_scope = frozenset(
            reference
            for reference, selected, present in zip(references, design_positions, residue_mask, strict=True)
            if reference is not None and selected and present
        )
        global_scope = frozenset(
            reference
            for reference, present in zip(references, residue_mask, strict=True)
            if reference is not None and present
        )
        return ScoreResult(
            target=input.structure,
            scores={
                "designed_sequence_nll": Score(
                    value=float(designed_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=designed_scope,
                    evaluations=designed_evaluations,
                ),
                "global_sequence_nll": Score(
                    value=float(global_evaluations.mean()),
                    unit="nat/residue",
                    preferred="lower",
                    scope=global_scope,
                    evaluations=global_evaluations,
                ),
            },
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Compute sequence-conditioned probabilities over decoding orders.

        Args:
            input: A ``Structure`` or local structure path.
            sequence: Optional replacement sequence to evaluate.
            positions: Residue occurrences treated as designable.
            num_decoding_orders: Number of independently sampled decoding orders.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.

        Returns:
            An ``AutoregressiveProbabilityResult`` containing logits and decoding ranks.
        """

        model_input = self.prepare_probabilities(input, sequence=sequence, positions=positions)
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        output = self.model.autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            num_neighbors=num_neighbors,
            generator=generator,
        )
        return self.decode_probabilities(model_input, output)

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Return sequence-free all-equal-rank probabilities for every row.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Residue occurrences included in the probability evaluation.
            num_decoding_orders: Number of repeated evaluations to retain.
            num_neighbors: Optional nearest-neighbor count for this operation.

        Returns:
            An ``AutoregressiveProbabilityResult`` with one probability tensor per evaluation.
        """

        model_input = self.prepare_probabilities(input, positions=positions)
        output = self.model.unconditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            num_neighbors=num_neighbors,
        )
        return self.decode_probabilities(model_input, output)

    def specificity_profile(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        num_samples: int = 30,
        temperature: float = 0.6,
        logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Average nucleic-acid base probabilities with protein context fixed.

        Args:
            input: A ``Structure`` or local structure path.
            positions: Nucleic-acid residue occurrences included in the profile.
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the NA-MPNN vocabulary.
            pair_logit_bias: Optional adjacent-token bias matrix with shape ``(33, 33)``.
            global_omit_tokens: One-letter residue tokens excluded from generation.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive autoregressive sampling temperature.
            logit_bias: Optional vocabulary-wide sampling bias.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.

        Returns:
            A ``ScoreResult`` containing per-residue base-probability metrics.
        """

        model_input = self.prepare_design(
            input,
            positions=positions,
            nucleic_acid_only=True,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        output = self.model.specificity_profile(
            model_input,
            num_samples=num_samples,
            temperature=temperature,
            logit_bias=logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )
        return self.decode_specificity_profile(model_input, output)

    def decode_probabilities(
        self,
        input: NaMpnnInput,
        output: NaMpnnOutput,
    ) -> AutoregressiveProbabilityResult:
        r"""Bind raw NA-MPNN probability tensors to their canonical residue axis."""

        return AutoregressiveProbabilityResult(
            target=input.structure,
            component_references=_component_references(input.structure.system, input.component_index[0]),
            sequence_tokens=input.sequence_tokens[0],
            residue_mask=input.residue_mask[0],
            design_positions=input.design_positions[0],
            logits=output.logits,
            decoding_ranks=output.decoding_ranks,
        )

    def decode_specificity_profile(
        self,
        input: NaMpnnInput,
        output: NaMpnnSamplingOutput,
    ) -> ScoreResult:
        r"""Decode sampled NA-MPNN contexts into per-residue base probabilities."""

        mean_probabilities = output.sampling_probabilities.mean(dim=0).cpu()
        if mean_probabilities.dtype == torch.bfloat16:
            mean_probabilities = mean_probabilities.float()
        host_probabilities = mean_probabilities.numpy()
        references = _component_references(input.structure.system, input.component_index[0])
        polymer_types = input.polymer_type_index[0].detach().cpu().tolist()
        selected = (input.design_positions[0] & (input.polymer_type_index[0] != 0) & input.residue_mask[0]).tolist()
        na_rows = [
            (index, reference)
            for index, (reference, polymer, keep) in enumerate(zip(references, polymer_types, selected))
            if reference is not None and keep and polymer in (1, 2)
        ]
        axis = tuple(reference for _index, reference in na_rows)
        token_columns = {
            row: (
                NA_MPNN_DNA_GENERATABLE_TOKEN_IDS
                if polymer_types[row] == 1
                else (
                    NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS
                    if int(input.sequence_tokens[0, row]) in NA_MPNN_RNA_SHARED_INPUT_TOKENS
                    else NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS
                )
            )
            for row, _reference in na_rows
        }
        base_names = ("adenine", "cytosine", "guanine", "thymine_uracil")
        metrics: dict[str, Metric] = {}
        for base_index, base in enumerate(base_names):
            values = np.array(
                [host_probabilities[row, token_columns[row][base_index]] for row, _reference in na_rows],
                dtype=host_probabilities.dtype,
            )
            metrics[f"{base}_probability"] = Metric(values=values, axis=axis)
        return ScoreResult(target=input.structure, scores={}, metrics=metrics)


def _component_references(
    system: MolecularSystem,
    component_index: Tensor,
) -> tuple[ComponentReference | None, ...]:
    return tuple(
        None if component < 0 else system.component_reference(component)
        for component in component_index.detach().cpu().tolist()
    )


def _realize_sequence(
    system: MolecularSystem,
    component_index: Tensor,
    polymer_type_index: Tensor,
    sampled_tokens: list[int],
    design_positions: list[bool],
) -> MolecularSystem:
    r"""Realize one candidate while preserving entity and bond identity."""

    selected_components = []
    replacements = []
    for component, polymer_type, token, selected in zip(
        component_index.detach().cpu().tolist(),
        polymer_type_index.detach().cpu().tolist(),
        sampled_tokens,
        design_positions,
        strict=True,
    ):
        if component < 0 or not selected:
            continue
        component_id = _component_from_token(int(token), int(polymer_type))
        if system.symbols[int(system.component_symbol[component])] == component_id:
            continue
        selected_components.append(component)
        replacements.append(get_component(component_id))
    if not selected_components:
        return system
    realized, _, _ = replace_components(
        system,
        torch.tensor(
            selected_components,
            dtype=torch.int64,
            device=system.component_symbol.device,
        ),
        replacements,
    )
    return repartition_entities_by_instance_sequence(system, realized)


def _component_from_token(token: int, polymer_type: int) -> str:
    r"""Realize one sampled token as a canonical component under its polymer type."""

    if polymer_type == 0:
        if token >= 20:
            raise RuntimeError(f"NA-MPNN sampled non-protein token {token} for a protein residue")
        return NA_MPNN_COMPONENT_IDS[token]
    if polymer_type == 1:
        if not 21 <= token <= 24:
            raise RuntimeError(f"NA-MPNN sampled non-DNA token {token} for a DNA residue")
        return ("DA", "DC", "DG", "DT")[token - 21]
    if polymer_type == 2 and 21 <= token <= 24:
        return ("A", "C", "G", "U")[token - 21]
    if polymer_type == 2 and 26 <= token <= 29:
        return ("A", "C", "G", "U")[token - 26]
    raise RuntimeError(f"NA-MPNN sampled non-RNA token {token} for an RNA residue")
