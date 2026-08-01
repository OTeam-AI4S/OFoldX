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
NA-MPNN sequence design, scoring, and probability Pipelines.

A design samples canonical classes on the prepared rows; ``produce_design`` builds each candidate's final target once.
Under shared nucleic tokens an RNA row is sampled as the DNA class of its base and designed as the RNA one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...errors import RequestRef, fail
from ...metrics.sequence import known_recovery_mask, sequence_nll, token_recovery
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import AutoregressiveProbabilityResult, DesignCandidate, DesignResult, Metric, Score, ScoreResult
from ...results.probabilities import bind_autoregressive_probabilities
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import DesignRecord, PositionSelection, produce_design
from ..preparation import PreparedInput
from ..processing import model_log_probabilities
from .model import NaMpnnInput, NaMpnnModel, NaMpnnOutput, NaMpnnSamplingOutput
from .processor import DNA_AS_RNA, NAMPNN_LAYOUT, NaMpnnProcessor


@dataclass(eq=False)
class NaMpnnDesignResult(DesignResult):
    r"""NA-MPNN designs and each candidate's design record."""

    records: tuple[DesignRecord, ...] = ()


def _scopes(prepared: PreparedInput[NaMpnnInput]) -> tuple[frozenset[ComponentRef], frozenset[ComponentRef]]:
    r"""The designed and the global residue scopes of a prepared input."""
    input = prepared.features
    refs = prepared.context.structure.system.component_refs(input.component_index[0].cpu().numpy())
    valid = input.residue_mask[0].cpu().tolist()
    designed = (input.residue_mask[0] & input.design_mask[0]).cpu().tolist()
    return (
        frozenset(ref for ref, keep in zip(refs, designed) if keep),
        frozenset(ref for ref, keep in zip(refs, valid) if keep),
    )


class _NaMpnnProbabilityOperations(PairedPipeline[NaMpnnModel, NaMpnnProcessor]):
    r"""Shared fixed-backbone probability preparation and evaluation."""

    model_architecture = "nampnn"

    def prepare_probabilities(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
    ) -> PreparedInput[NaMpnnInput]:
        r"""Prepare the target's classes, or an explicit one-letter sequence, for evaluation."""
        return self.processor.prepare_input(structure, sequence=sequence, design=design)

    def autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""
        Compute sequence-conditioned probabilities over decoding orders.

        Args:
            structure: A read structure.
            sequence: Optional one-letter sequence evaluated instead of the target's classes.
            design: Residues treated as designable (default every valid row).
            num_decoding_orders: Number of independently sampled decoding orders.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.
        """

        prepared = self.prepare_probabilities(structure, sequence=sequence, design=design)
        generator = resolve_generator(seed, generator, device=prepared.features.atom_coordinates.device)
        output = self.model.autoregressive_probabilities(
            prepared.features, num_decoding_orders=num_decoding_orders, num_neighbors=num_neighbors, generator=generator
        )
        return self.decode_probabilities(prepared, output)

    def unconditional_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-free all-equal-rank probabilities for every row."""
        prepared = self.prepare_probabilities(structure, design=design)
        output = self.model.unconditional_probabilities(
            prepared.features, num_decoding_orders=num_decoding_orders, num_neighbors=num_neighbors
        )
        return self.decode_probabilities(prepared, output)

    def decode_probabilities(
        self, prepared: PreparedInput[NaMpnnInput], output: NaMpnnOutput
    ) -> AutoregressiveProbabilityResult:
        r"""Bind raw probability logits and ranks to their canonical residue axis."""
        input = prepared.features
        return bind_autoregressive_probabilities(
            prepared.context.structure,
            input.component_index,
            input.sequence_tokens,
            input.residue_mask,
            input.design_mask,
            output.logits,
            output.decoding_ranks,
        )


class NaMpnnPipeline(_NaMpnnProbabilityOperations):
    r"""
    Protein and nucleic-acid sequence design pipeline for NA-MPNN.

    Every verb conditions on observed backbone coordinates, so it takes a read structure.
    """

    task = "design"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentRef]] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
    ) -> PreparedInput[NaMpnnInput]:
        r"""Prepare one fixed-backbone NA-MPNN design input."""
        return self.processor.prepare_input(
            structure,
            design=design,
            nucleic_acid_only=nucleic_acid_only,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )

    def forward(
        self,
        input: NaMpnnInput,
        *,
        num_candidates: int = 1,
        temperature: float = 0.1,
        global_logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnSamplingOutput:
        r"""Execute the prepared design operation in its native draw order."""
        generator = resolve_generator(seed, generator, device=input.atom_coordinates.device)
        return self.model.design(
            input,
            num_candidates=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[NaMpnnInput],
        output: NaMpnnSamplingOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> NaMpnnDesignResult:
        r"""Build each candidate's final target with ``produce_design`` (definitions from ``library``, default the
        configured one) and score it."""
        input = prepared.features
        designed = input.residue_mask & input.design_mask
        confidence = torch.exp(-sequence_nll(output.log_probabilities, output.sequence_tokens, designed))
        native = known_recovery_mask(input.sequence_tokens, designed)
        recovery = token_recovery(output.sequence_tokens, input.sequence_tokens, native)
        refs = prepared.context.structure.system.component_refs(input.component_index[0].cpu().numpy())
        native_scope = frozenset(ref for ref, keep in zip(refs, native[0].cpu().tolist()) if keep)
        scores = torch.stack((confidence, recovery), dim=-1).detach().cpu().double().numpy()
        rows = designed[0].cpu().numpy()
        positions = input.component_index[0].cpu().numpy()[rows]
        rna = input.polymer_type_index[0].cpu().numpy()[rows] == 2
        sampled = output.sequence_tokens.detach().cpu().numpy()[:, rows]
        classes = np.where(rna, DNA_AS_RNA[sampled], sampled)
        designed_scope, _ = _scopes(prepared)
        library = configured_library() if library is None else library
        candidates, records = [], []
        for index in range(classes.shape[0]):
            target, record = produce_design(
                prepared, classes[index], positions=positions, table=NAMPNN_LAYOUT, library=library
            )
            candidates.append(
                DesignCandidate(
                    target=target,
                    scores={
                        "designed_sequence_confidence": Score(float(scores[index, 0]), None, "higher", designed_scope),
                        "sequence_recovery": Score(float(scores[index, 1]), None, "higher", native_scope),
                    },
                )
            )
            records.append(record)
        return NaMpnnDesignResult(candidates=tuple(candidates), records=tuple(records))

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentRef]] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        num_candidates: int = 1,
        temperature: float = 0.1,
        global_logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> NaMpnnDesignResult:
        r"""
        Design mixed-polymer sequences on one fixed backbone.

        Args:
            structure: A read structure.
            design: The positions to design (default every valid row).
            nucleic_acid_only: Whether only DNA and RNA rows may be designed (protein stays context).
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            pair_logit_bias: Optional adjacent-class bias matrix with shape ``(72, 72)``.
            global_omit_tokens: Canonical class names excluded from generation.
            num_candidates: Number of complete sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            global_logit_bias: Optional bias vector over the 72 canonical classes.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.
            library: The definitions replacements are built from (default the configured artifact).

        Returns:
            Candidates in sampling order, each the final target of its design, and each candidate's design record.
        """

        prepared = self.prepare(
            structure,
            design=design,
            nucleic_acid_only=nucleic_acid_only,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            num_neighbors=num_neighbors,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)

    def specificity_profile(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentRef]] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        num_samples: int = 30,
        temperature: float = 0.6,
        global_logit_bias: Tensor | None = None,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Average nucleic-acid base probabilities with protein context fixed.

        Args:
            structure: A read structure.
            design: The nucleic-acid positions included in the profile (default every valid DNA and RNA row).
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            pair_logit_bias: Optional adjacent-class bias matrix with shape ``(72, 72)``.
            global_omit_tokens: Canonical class names excluded from generation.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive autoregressive sampling temperature.
            global_logit_bias: Optional bias vector over the 72 canonical classes.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.

        Returns:
            Per-residue base-probability metrics (adenine, cytosine, guanine, thymine or uracil).
        """

        prepared = self.prepare(
            structure,
            design=design,
            nucleic_acid_only=True,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=prepared.features.atom_coordinates.device)
        output = self.model.specificity_profile(
            prepared.features,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            num_neighbors=num_neighbors,
            generator=generator,
        )
        return self.decode_specificity_profile(prepared, output)

    def decode_specificity_profile(
        self, prepared: PreparedInput[NaMpnnInput], output: NaMpnnSamplingOutput
    ) -> ScoreResult:
        r"""Decode sampled NA-MPNN contexts into per-residue base probabilities."""
        input = prepared.features
        probabilities = output.sampling_probabilities.mean(dim=0).detach().cpu().double().numpy()
        kind = input.polymer_type_index[0].cpu().numpy()
        rows = np.flatnonzero((input.design_mask[0] & input.residue_mask[0]).cpu().numpy() & (kind != 0))
        axis = prepared.context.structure.system.component_refs(input.component_index[0].cpu().numpy()[rows])
        columns = np.array([self.processor.generatable(int(kind[at])) for at in rows]).reshape(len(rows), 4)
        values = probabilities[rows[:, None], columns]
        metrics = {
            f"{base}_probability": Metric(values=values[:, index], axis=axis)
            for index, base in enumerate(("adenine", "cytosine", "guanine", "thymine_uracil"))
        }
        return ScoreResult(target=prepared.context.structure, scores={}, metrics=metrics)


class NaMpnnScorePipeline(_NaMpnnProbabilityOperations):
    r"""Score fixed-backbone sequences with NA-MPNN."""

    task = "score"

    def prepare(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
    ) -> PreparedInput[NaMpnnInput]:
        r"""Prepare the target's classes, or an explicit one-letter sequence, for scoring."""
        return self.processor.prepare_input(structure, sequence=sequence, design=design)

    def forward(
        self,
        input: NaMpnnInput,
        *,
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> NaMpnnOutput:
        r"""Execute the prepared score operation in its native draw order."""
        if not bool((input.residue_mask & input.design_mask).any()):
            fail("value_invalid", RequestRef("design", "no valid scoring position"))
        generator = resolve_generator(seed, generator, device=input.atom_coordinates.device)
        return self.model.score(
            input, num_decoding_orders=num_decoding_orders, num_neighbors=num_neighbors, generator=generator
        )

    def postprocess(self, prepared: PreparedInput[NaMpnnInput], output: NaMpnnOutput) -> ScoreResult:
        r"""Designed and global sequence NLL of the input's classes over the evaluated decoding orders."""
        input = prepared.features
        designed = input.residue_mask & input.design_mask
        tokens = input.sequence_tokens.repeat_interleave(output.logits.shape[0], dim=0)
        log_probabilities = model_log_probabilities(output.logits, input.residue_mask)
        designed_values = sequence_nll(log_probabilities, tokens, designed).detach().cpu().double().numpy()
        global_values = sequence_nll(log_probabilities, tokens, input.residue_mask).detach().cpu().double().numpy()
        designed_scope, global_scope = _scopes(prepared)
        return ScoreResult(
            target=prepared.context.structure,
            scores={
                "designed_sequence_nll": Score(
                    float(designed_values.mean()), "nat/residue", "lower", designed_scope, designed_values
                ),
                "global_sequence_nll": Score(
                    float(global_values.mean()), "nat/residue", "lower", global_scope, global_values
                ),
            },
        )

    def score(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        num_decoding_orders: int = 1,
        num_neighbors: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score a mixed-polymer sequence with autoregressive probabilities.

        Args:
            structure: A read structure.
            sequence: Optional one-letter sequence (a string for one polymer instance, or by instance); ``None``
                scores the structure's classes.
            design: Residues included in the designed-sequence score (default every valid row).
            num_decoding_orders: Number of independently sampled decoding orders.
            num_neighbors: Optional nearest-neighbor count for this operation.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional random generator; mutually exclusive with ``seed``.
        """

        prepared = self.prepare(structure, sequence=sequence, design=design)
        output = self.forward(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            num_neighbors=num_neighbors,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(score)
