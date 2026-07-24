r"""
NA-MPNN sequence design, scoring, and probability Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch
from torch import Tensor

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import DesignResult, MpnnProbabilityResult, ScoreResult
from .model import NaMpnnModel
from .processor import NaMpnnProcessor


class NaMpnnPipeline(PairedPipeline[NaMpnnModel, NaMpnnProcessor]):
    r"""
    Protein and nucleic-acid sequence design pipeline for NA-MPNN.
    """

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        num_candidates: int = 1,
        temperature: float = 0.1,
        global_logit_bias: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Design mixed-polymer sequences on one fixed backbone.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences allowed to change.
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the NA-MPNN vocabulary.
            pair_bias: Optional adjacent-token bias matrix with shape ``(33, 33)``.
            global_omit_tokens: One-letter residue tokens excluded from generation.
            num_candidates: Number of complete sequences to sample.
            temperature: Positive autoregressive sampling temperature.
            global_logit_bias: Optional vocabulary-wide sampling bias.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            A ``DesignResult`` containing candidates in sampling order.
        """

        model_input = self.processor(
            input,
            design_positions=design_positions,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_bias=pair_bias,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Return sequence-conditioned autoregressive probabilities over decoding orders.

        Args:
            input: A ``Structure`` or local structure path.
            sequence: Optional replacement sequence to evaluate.
            design_positions: Residue occurrences included in the probability evaluation.
            num_decoding_orders: Number of independently sampled decoding orders.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            An ``MpnnProbabilityResult`` whose leading axis follows decoding-order draw order.
        """

        model_input = self.processor(input, sequence=sequence, design_positions=design_positions)
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        return self.model.score(
            model_input,
            num_decoding_orders=num_decoding_orders,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
    ) -> MpnnProbabilityResult:
        r"""
        Return sequence-free all-equal-rank probabilities for every row.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences included in the probability evaluation.
            num_decoding_orders: Number of repeated evaluations to retain.

        Returns:
            An ``MpnnProbabilityResult`` with one probability tensor per evaluation.
        """

        model_input = self.processor(input, design_positions=design_positions)
        return self.model.unconditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
        )

    def specificity_profile(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        global_omit_tokens: Sequence[str] = (),
        num_samples: int = 30,
        temperature: float = 0.6,
        global_logit_bias: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Average nucleic-acid base probabilities with protein context fixed.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Nucleic-acid residue occurrences included in the profile.
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            global_omit_tokens: One-letter residue tokens excluded from generation.
            num_samples: Number of sampled sequence contexts to average.
            temperature: Positive autoregressive sampling temperature.
            global_logit_bias: Optional vocabulary-wide sampling bias.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            A ``ScoreResult`` containing per-residue base-probability metrics.
        """

        model_input = self.processor(
            input,
            design_positions=design_positions,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=model_input.atom_coordinates.device)
        return self.model.specificity_profile(
            model_input,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )
