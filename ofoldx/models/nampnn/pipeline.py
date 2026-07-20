r"""OFoldX's biological operation boundary for released NA-MPNN computations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch
from torch import Tensor

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...pipelines._generator import resolve_generator
from ...pipelines._paired import _PairedPipeline
from ...results import DesignResult, MpnnProbabilityResult, ScoreResult
from .model import NaMpnnModel
from .processor import NaMpnnProcessor


class NaMpnnPipeline(_PairedPipeline):
    r"""Bind one NaMpnn Model-Processor pair for released biological operations."""

    model: NaMpnnModel
    processor: NaMpnnProcessor

    def __init__(self, *, model: NaMpnnModel, processor: NaMpnnProcessor) -> None:
        super().__init__(model=model, processor=processor)

    def design(
        self,
        input: Structure | MolecularSystem | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
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
        r"""Design mixed-polymer sequences on one fixed backbone."""

        batch = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_bias=pair_bias,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=batch.atom_coordinates.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )

    def score(
        self,
        input: Structure | MolecularSystem | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-conditioned autoregressive probabilities over orders."""

        batch = self.processor(input, sequence=sequence, positions=positions)
        generator = resolve_generator(seed, generator, device=batch.atom_coordinates.device)
        return self.model.score(
            batch,
            num_decoding_orders=num_decoding_orders,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | MolecularSystem | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-free all-equal-rank probabilities for every row."""

        batch = self.processor(input, positions=positions)
        generator = resolve_generator(seed, generator, device=batch.atom_coordinates.device)
        return self.model.unconditional_probabilities(
            batch,
            num_decoding_orders=num_decoding_orders,
            generator=generator,
        )

    def specificity_profile(
        self,
        input: Structure | MolecularSystem | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        global_omit_tokens: Sequence[str] = (),
        num_samples: int = 30,
        temperature: float = 0.6,
        global_logit_bias: Tensor | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""Average released nucleic-acid base probabilities with protein context fixed."""

        batch = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            global_omit_tokens=global_omit_tokens,
        )
        generator = resolve_generator(seed, generator, device=batch.atom_coordinates.device)
        return self.model.specificity_profile(
            batch,
            num_samples=num_samples,
            temperature=temperature,
            global_logit_bias=global_logit_bias,
            generator=generator,
        )
