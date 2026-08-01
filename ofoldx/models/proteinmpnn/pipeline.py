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
ProteinMPNN sequence design, scoring, and probability Pipelines.

A design samples canonical classes on the prepared rows; ``produce_design`` builds each candidate's final target once,
and the design-and-pack composite packs exactly the replaced positions of each candidate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.molecular import ComponentRef
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKENS
from ...errors import RequestRef, fail
from ...metrics.sequence import known_recovery_mask, sequence_nll, token_recovery
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import (
    AutoregressiveProbabilityResult,
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    PackResult,
    Score,
    ScoreResult,
)
from ...results.probabilities import bind_autoregressive_probabilities
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import DesignRecord, PositionSelection, produce_design
from ..mpnn_packer import MpnnPackerPipeline
from ..preparation import PreparedInput
from ..processing import model_log_probabilities
from .model import (
    PROTEINMPNN_VOCABULARY,
    ProteinMpnnDesignOutput,
    ProteinMpnnInput,
    ProteinMpnnModel,
    ProteinMpnnOutput,
)
from .processor import ProteinMpnnProcessor


@dataclass(eq=False)
class ProteinMpnnDesignResult(DesignResult):
    r"""ProteinMPNN designs, the source-sequence score and each candidate's design record.

    The input score evaluates ``X`` as the model's ``X`` class.
    """

    input_score: ScoreResult
    records: tuple[DesignRecord, ...] = ()


def residue_scopes(
    prepared: PreparedInput[ProteinMpnnInput],
) -> tuple[frozenset[ComponentRef], frozenset[ComponentRef]]:
    r"""The designed and the global residue scopes of a prepared input."""
    input = prepared.features
    system = prepared.context.structure.system
    refs = system.component_refs(input.component_index[0].cpu().numpy())
    valid = input.residue_mask[0].cpu().tolist()
    designed = (input.residue_mask[0] & input.design_mask[0]).cpu().tolist()
    return (
        frozenset(ref for ref, keep in zip(refs, designed) if keep),
        frozenset(ref for ref, keep in zip(refs, valid) if keep),
    )


class _ProteinMpnnProbabilityOperations(PairedPipeline[ProteinMpnnModel, ProteinMpnnProcessor]):
    r"""Shared fixed-backbone probability preparation and evaluation."""

    model_architecture = "proteinmpnn"

    def prepare_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""Prepare one ProteinMPNN probability-evaluation input."""
        return self.processor.prepare_input(
            structure, design=design, conditions=conditions, filter_by_occupancy=filter_by_occupancy
        )

    def _probabilities(
        self,
        operation: str,
        structure: Structure,
        *,
        design: PositionSelection,
        conditions: Sequence[TiedResidueIdentities],
        filter_by_occupancy: bool,
        num_decoding_orders: int,
        coordinate_noise: float,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> AutoregressiveProbabilityResult:
        prepared = self.prepare_probabilities(
            structure, design=design, conditions=conditions, filter_by_occupancy=filter_by_occupancy
        )
        generator = resolve_generator(seed, generator, device=prepared.features.backbone_coordinates.device)
        output = getattr(self.model, operation)(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )
        return self.decode_probabilities(prepared, output)

    def autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-conditioned autoregressive probabilities over independently drawn decoding orders."""
        return self._probabilities(
            "autoregressive_probabilities",
            structure,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[TiedResidueIdentities] = (),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Sequence-free autoregressive probabilities over independently drawn decoding orders."""
        return self._probabilities(
            "sequence_free_autoregressive_probabilities",
            structure,
            design=design,
            conditions=conditions,
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-last conditional probabilities for every row (each target decoded after all others)."""
        return self._probabilities(
            "conditional_probabilities",
            structure,
            design=design,
            conditions=(),
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> AutoregressiveProbabilityResult:
        r"""Target-first unconditional probabilities for every row (each target decoded before all others)."""
        return self._probabilities(
            "unconditional_probabilities",
            structure,
            design=design,
            conditions=(),
            filter_by_occupancy=filter_by_occupancy,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )

    def decode_probabilities(
        self, prepared: PreparedInput[ProteinMpnnInput], output: ProteinMpnnOutput
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


def score_result(prepared: PreparedInput[ProteinMpnnInput], output: ProteinMpnnOutput) -> ScoreResult:
    r"""Designed and global sequence NLL of the input's classes over the evaluated decoding orders."""
    input = prepared.features
    designed = input.residue_mask & input.design_mask
    if not bool(designed.any()):
        fail("value_invalid", RequestRef("design", "no valid scoring position"))
    tokens = input.sequence_tokens.repeat_interleave(output.logits.shape[0], dim=0)
    log_probabilities = model_log_probabilities(output.logits, input.residue_mask)
    designed_values = sequence_nll(log_probabilities, tokens, designed).detach().cpu().float().numpy()
    global_values = sequence_nll(log_probabilities, tokens, input.residue_mask).detach().cpu().float().numpy()
    designed_scope, global_scope = residue_scopes(prepared)
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


class ProteinMpnnPipeline(_ProteinMpnnProbabilityOperations):
    r"""
    Protein sequence design pipeline for ProteinMPNN.

    Every verb conditions on observed backbone coordinates, so it takes a read structure.
    """

    task = "design"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentRef, float] | None = None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""
        Prepare a fixed backbone and residue constraints for sequence design.

        Args:
            structure: Target structure with observed protein backbone coordinates.
            design: Positions to redesign; defaults to every valid protein residue.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue additive bias over the 72 canonical classes.
            pssm_coefficients: Per-residue PSSM mixing coefficients, used with ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the canonical classes.
            pssm_log_odds: Per-residue log-odds vectors over the canonical classes.
            filter_by_occupancy: Treat atoms with known occupancy at or below zero as unobserved.

        Returns:
            The model input, canonical target, atom correspondence, design scope and preparation records.
        """
        return self.processor.prepare_input(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignOutput:
        r"""
        Sample sequence candidates and evaluate the input sequence on a prepared backbone.

        Args:
            input: Prepared backbone features and residue constraints.
            num_candidates: Number of complete sequence candidates to sample.
            num_decoding_orders: Number of decoding orders used to score the input sequence.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            global_logit_bias: Additive bias over the 72 canonical classes.
            global_omit_tokens: One-letter classes excluded from sampling.
            pssm_mixture: Global PSSM mixture weight between zero and one.
            pssm_log_odds_threshold: Minimum PSSM log odds retained during sampling, when specified.
            seed: Optional integer seed, mutually exclusive with ``generator``.
            generator: Caller-owned PyTorch random generator on the sampling device.

        Returns:
            Raw candidate sampling tensors and input-sequence score tensors. ``postprocess`` binds them to
                molecular targets and named scores.
        """
        generator = resolve_generator(seed, generator, device=input.backbone_coordinates.device)
        return self.model.design(
            input,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=omit_token_mask(global_omit_tokens, device=input.residue_mask.device),
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )

    def postprocess(
        self,
        prepared: PreparedInput[ProteinMpnnInput],
        output: ProteinMpnnDesignOutput,
        *,
        library: ComponentLibrary | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""
        Construct the designed molecular targets and their sequence scores.

        Args:
            prepared: Backbone preparation, including its target and residue correspondence.
            output: Raw sequence sampling and input-scoring tensors from ``forward``.
            library: Chemical definitions used to construct replacement residues; defaults to the configured CCD.

        Returns:
            Candidates in sampling order, their design records, and the input-sequence score. Each candidate
                carries designed/global sequence NLL and sequence recovery.

        Notes:
            The shared design construction retains a source occurrence when its biological class is selected,
            such as MSE predicted as M. A different class creates its standard component while retaining
            backbone guide coordinates; new side-chain coordinates are unavailable.
        """
        input = prepared.features
        sampled = output.sample
        designed = input.residue_mask & input.design_mask
        designed_nll = sequence_nll(sampled.log_probabilities, sampled.sequence_tokens, designed)
        global_nll = sequence_nll(sampled.log_probabilities, sampled.sequence_tokens, input.residue_mask)
        native = known_recovery_mask(input.sequence_tokens, designed)
        recovery = token_recovery(sampled.sequence_tokens, input.sequence_tokens, native)
        refs = prepared.context.structure.system.component_refs(input.component_index[0].cpu().numpy())
        native_scope = frozenset(ref for ref, keep in zip(refs, native[0].cpu().tolist()) if keep)
        scores = torch.stack((designed_nll, global_nll, recovery), dim=-1).detach().cpu().double().numpy()
        rows = designed[0].cpu().numpy()
        positions = input.component_index[0].cpu().numpy()[rows]
        classes = sampled.sequence_tokens.detach().cpu().numpy()[:, rows]
        designed_scope, global_scope = residue_scopes(prepared)
        library = configured_library() if library is None else library
        candidates, records = [], []
        for index in range(classes.shape[0]):
            target, record = produce_design(
                prepared, classes[index], positions=positions, table=self.processor.layout, library=library
            )
            candidates.append(
                DesignCandidate(
                    target=target,
                    scores={
                        "designed_sequence_nll": Score(float(scores[index, 0]), "nat/residue", "lower", designed_scope),
                        "global_sequence_nll": Score(float(scores[index, 1]), "nat/residue", "lower", global_scope),
                        "sequence_recovery": Score(float(scores[index, 2]), None, "higher", native_scope),
                    },
                )
            )
            records.append(record)
        return ProteinMpnnDesignResult(
            candidates=tuple(candidates),
            input_score=score_result(prepared, output.input_score),
            records=tuple(records),
        )

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentRef, float] | None = None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""
        Design ProteinMPNN sequences on a fixed backbone.

        Args:
            structure: A read structure.
            design: The positions to design (default every valid protein residue).
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 72 canonical classes.
            pssm_log_odds: Per-residue log-odds vectors over the 72 canonical classes.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            num_decoding_orders: Number of decoding orders used to score the input sequence.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            global_logit_bias: Global bias vector over the 72 canonical classes.
            global_omit_tokens: One-letter classes excluded from sampling.
            pssm_mixture: Global PSSM mixture weight between zero and one.
            pssm_log_odds_threshold: Optional minimum PSSM log odds retained during sampling.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
            library: The definitions replacements are built from (default the configured artifact).

        Returns:
            Candidates in sampling order, each the final target of its design (MSE sampled as ``M`` stays MSE), the
            input-sequence score and each candidate's design record.
        """

        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared.features,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output, library=library)

    __call__ = task_call(design)


class ProteinMpnnScorePipeline(_ProteinMpnnProbabilityOperations):
    r"""Score fixed-backbone sequences with ProteinMPNN."""

    task = "score"

    def prepare(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""
        Prepare a fixed backbone and the sequence identities to evaluate.

        Args:
            structure: Target structure with observed protein backbone coordinates.
            sequence: Optional one-letter sequence, or sequences by instance; defaults to the target's classes.
            design: Positions included in the designed-sequence score.
            filter_by_occupancy: Treat atoms with known occupancy at or below zero as unobserved.

        Returns:
            The model features and canonical correspondence needed to score the supplied sequence.
        """
        return self.processor.prepare_input(
            structure, design=design, sequence=sequence, filter_by_occupancy=filter_by_occupancy
        )

    def forward(
        self,
        input: ProteinMpnnInput,
        *,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnOutput:
        r"""
        Evaluate a prepared sequence under sampled autoregressive orders.

        Args:
            input: Prepared backbone and sequence features.
            num_decoding_orders: Number of autoregressive decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed, mutually exclusive with ``generator``.
            generator: Caller-owned PyTorch random generator on the evaluation device.

        Returns:
            Raw logits and decoding ranks for each evaluated order.

        Raises:
            InvalidRequestError: No valid position is included in the designed-sequence score.
        """
        if not bool((input.residue_mask & input.design_mask).any()):
            fail("value_invalid", RequestRef("design", "no valid scoring position"))
        generator = resolve_generator(seed, generator, device=input.backbone_coordinates.device)
        return self.model.score(
            input, num_decoding_orders=num_decoding_orders, coordinate_noise=coordinate_noise, generator=generator
        )

    def postprocess(self, prepared: PreparedInput[ProteinMpnnInput], output: ProteinMpnnOutput) -> ScoreResult:
        r"""
        Bind evaluation logits to designed and global sequence NLL scores.

        Args:
            prepared: Sequence preparation with canonical residue correspondence and scoring scope.
            output: Logits and decoding ranks returned by ``forward``.

        Returns:
            The scored target and named NLL values in nats per residue, with molecular scopes and
                per-order evaluations retained.
        """
        return score_result(prepared, output)

    def score(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str] | None = None,
        design: PositionSelection = PositionSelection(),
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score a sequence over independently drawn decoding orders.

        ``X`` is scored as the model's ``X`` class.

        Args:
            structure: A read structure.
            sequence: Optional one-letter sequence (a string for one protein instance, or by instance); ``None``
                scores the structure's classes.
            design: Residues included in the designed-sequence score (default every valid protein residue).
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator.
        """

        prepared = self.prepare(structure, sequence=sequence, design=design, filter_by_occupancy=filter_by_occupancy)
        output = self.forward(
            prepared.features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(score)


@dataclass(eq=False)
class ProteinMpnnDesignAndPackPipelineInput:
    r"""Protein design input and the observation policy shared with packing."""

    design: PreparedInput[ProteinMpnnInput]
    filter_by_occupancy: bool


class ProteinMpnnDesignAndPackPipeline(CompositePipeline):
    r"""
    Design protein sequences with ProteinMPNN and pack the replaced positions of every candidate.

    Args:
        designer: ProteinMPNN sequence-design pipeline.
        packer: Side-chain packing pipeline.

    Notes:
        Repositories store the designer under ``design/`` and the packer under ``packer/``. Preparation retains the
        source structure, the exact design scope, and one ``filter_by_occupancy`` policy. Setting that policy to
        ``False`` keeps observed zero-occupancy atoms in both design and packing. One caller generator advances
        through sequence design and every candidate's packing.
    """

    task = "design"

    component_layout = (
        ComponentSpec("designer", "design", ProteinMpnnPipeline),
        ComponentSpec("packer", "packer", MpnnPackerPipeline),
    )

    def __init__(self, *, designer: ProteinMpnnPipeline, packer: MpnnPackerPipeline) -> None:
        super().__init__({"designer": designer, "packer": packer})

    @property
    def designer(self) -> ProteinMpnnPipeline:
        return cast(ProteinMpnnPipeline, self.components["designer"])

    @property
    def packer(self) -> MpnnPackerPipeline:
        return cast(MpnnPackerPipeline, self.components["packer"])

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentRef, float] | None = None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> ProteinMpnnDesignAndPackPipelineInput:
        r"""Prepare the source structure and the exact sequence-design scope."""
        prepared = self.designer.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )
        return ProteinMpnnDesignAndPackPipelineInput(design=prepared, filter_by_occupancy=filter_by_occupancy)

    def forward(
        self,
        input: ProteinMpnnDesignAndPackPipelineInput,
        *,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> tuple[ProteinMpnnDesignResult, tuple[PackResult, ...]]:
        r"""Design identities, then pack each candidate's replaced positions (or every packable residue)."""
        prepared = input.design
        generator = resolve_generator(seed, generator, device=prepared.features.residue_mask.device)
        output = self.designer.forward(
            prepared.features,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )
        designs = self.designer.postprocess(prepared, output, library=library)
        packs = pack_protein_designs(
            designs,
            self.packer,
            repack_everything=repack_everything,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=input.filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return designs, packs

    def postprocess(
        self,
        input: ProteinMpnnDesignAndPackPipelineInput,
        output: tuple[ProteinMpnnDesignResult, tuple[PackResult, ...]],
    ) -> DesignAndPackResult:
        r"""Assemble candidate-aligned sequence designs and side-chain predictions."""
        designs, packs = output
        return DesignAndPackResult(designs=designs, packs=packs)

    def design(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentRef, float] | None = None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        library: ComponentLibrary | None = None,
    ) -> DesignAndPackResult:
        r"""
        Design ProteinMPNN sequences and pack every resulting candidate.

        Only the positions a candidate replaced are packed (a position retained as its source component, such as MSE
        sampled as ``M``, is not), unless ``repack_everything``. The ``PackResult`` tuple follows candidate order.
        ``seed`` and ``generator`` are mutually exclusive.
        """

        prepared = self.prepare(
            structure,
            design=design,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=global_logit_bias,
            global_omit_tokens=global_omit_tokens,
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            use_packing_atom_context=use_packing_atom_context,
            repack_everything=repack_everything,
            num_packs_per_candidate=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            seed=seed,
            generator=generator,
            library=library,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)


def pack_protein_designs(
    designs: ProteinMpnnDesignResult,
    packer: MpnnPackerPipeline,
    *,
    repack_everything: bool,
    use_atom_context: bool,
    filter_by_occupancy: bool,
    num_samples: int,
    num_denoising_steps: int,
    num_torsion_samples: int,
    generator: torch.Generator | None,
) -> tuple[PackResult, ...]:
    r"""Pack each candidate's replaced positions (``DesignRecord.replaced``), or every packable residue."""
    return tuple(
        packer.pack(
            candidate.target,
            positions=PositionSelection() if repack_everything else PositionSelection(indices=record.replaced()),
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_samples,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        for candidate, record in zip(designs.candidates, designs.records, strict=True)
    )


def omit_token_mask(tokens: Sequence[str], *, device: torch.device) -> Tensor | None:
    r"""The canonical classes excluded from sampling, from one-letter tokens of ProteinMPNN's vocabulary; an empty
    selection omits nothing (``X`` becomes reachable)."""
    if not tokens:
        return None
    mask = torch.zeros(len(TOKENS), dtype=torch.bool, device=device)
    for token in tokens:
        if token not in PROTEINMPNN_VOCABULARY:
            fail("value_invalid", RequestRef("global_omit_tokens", token))
        mask[TOKEN_INDEX[token]] = True
    return mask
