r"""OFoldX's biological operation boundary for released ProteinMPNN computations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from typing_extensions import Self

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import read_pdb
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...pipelines._generator import resolve_generator
from ...pipelines._paired import _PairedPipeline
from ...results import MpnnProbabilityResult, PackResult, ScoreResult
from .model import ProteinMpnnDesignResult, ProteinMpnnModel
from .processor import ProteinMpnnProcessor

if TYPE_CHECKING:
    from ..mpnn_packer import MpnnPackerPipeline


_GENERATED_TOKEN_INDEX_BY_LETTER: dict[str, int] = {
    letter: AMINO_ACID_TOKEN_BY_COMPONENT[PolymerSequence.from_one_letter(letter).components[0].component_id]
    for letter in "ACDEFGHIKLMNPQRSTVWY"
}
_GENERATED_TOKEN_INDEX_BY_LETTER["X"] = AMINO_ACID_UNKNOWN_TOKEN


def _global_omit_token_mask(
    tokens: Sequence[str],
    *,
    device: torch.device,
) -> Tensor | None:
    r"""Lower released one-letter omit tokens to a generatable-class support mask.

    An empty selection carries no omission and enables ProteinMPNN's reachable
    twenty-first generated class. Every token is a generatable one-letter identity;
    an unrecognized token is rejected rather than silently ignored.
    """

    if not tokens:
        return None
    mask = torch.zeros(
        len(AMINO_ACID_COMPONENT_IDS) + 1,
        dtype=torch.bool,
        device=device,
    )
    for token in tokens:
        try:
            mask[_GENERATED_TOKEN_INDEX_BY_LETTER[token]] = True
        except KeyError:
            raise ValueError(f"unknown omit token {token!r}") from None
    return mask


def _candidate_protein_sequences(
    system: MolecularSystem,
) -> dict[str, tuple[str, ...]]:
    r"""Map each protein instance to its realized component-ID sequence."""

    sequences: dict[str, tuple[str, ...]] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            sequences[instance_id] = tuple(component.component_id for component in entity.components)
    return sequences


class ProteinMpnnPipeline(_PairedPipeline):
    r"""Bind one ProteinMpnn Model–Processor pair for biological operations."""

    model: ProteinMpnnModel
    processor: ProteinMpnnProcessor

    def __init__(
        self,
        *,
        model: ProteinMpnnModel,
        processor: ProteinMpnnProcessor,
    ) -> None:
        super().__init__(model=model, processor=processor)

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
        global_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""Design sequences for selected positions on a fixed backbone."""

        batch = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
        )
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.design(
            batch,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            backbone_noise=backbone_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=_global_omit_token_mask(global_omit_tokens, device=batch.valid_mask.device),
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""Score one sequence over independently drawn decoding orders."""

        batch = self.processor(input, sequence=sequence, positions=positions)
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.score(
            batch,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-conditioned autoregressive probabilities over orders."""

        batch = self.processor(input, positions=positions, conditions=conditions)
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.autoregressive_probabilities(
            batch,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return order-dependent sequence-free autoregressive probabilities."""

        batch = self.processor(input, positions=positions, conditions=conditions)
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.sequence_free_autoregressive_probabilities(
            batch,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-last conditional probabilities for every row."""

        batch = self.processor(input, positions=positions)
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.conditional_probabilities(
            batch,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        backbone_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-first sequence-free probabilities for every row."""

        batch = self.processor(input, positions=positions)
        generator = resolve_generator(seed, generator, device=batch.backbone_coordinates.device)
        return self.model.unconditional_probabilities(
            batch,
            num_decoding_orders=num_decoding_orders,
            backbone_noise=backbone_noise,
            generator=generator,
        )


class ProteinMpnnDesignAndPackPipeline:
    r"""Compose the ProteinMpnn designer with the MpnnPacker over one source.

    The released protein and soluble combined command designs sequences on a fixed
    backbone and then packs each design. Both stages read one canonical
    ``Structure`` parsed exactly once, and a supplied random stream is consumed by
    design first and advanced pack-major across every candidate.
    """

    def __init__(
        self,
        *,
        designer: ProteinMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None:
        self.designer = designer
        self.packer = packer

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Place both children on one device and dtype in place."""

        self.designer.to(device=device, dtype=dtype)
        self.packer.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        num_decoding_orders: int = 1,
        temperature: float = 0.1,
        backbone_noise: float = 0.0,
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
    ) -> tuple[ProteinMpnnDesignResult, tuple[PackResult, ...]]:
        r"""Design sequences on one canonical Structure and pack each candidate."""

        structure = read_pdb(input) if isinstance(input, (str, PathLike)) else input
        design_input = self.designer.processor.process_structure(
            structure,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )
        device = design_input.valid_mask.device
        generator = resolve_generator(seed, generator, device=device)
        design_result = self.designer.model.design(
            design_input,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            backbone_noise=backbone_noise,
            global_logit_bias=global_logit_bias,
            global_omit_token_mask=_global_omit_token_mask(global_omit_tokens, device=device),
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )
        if repack_everything:
            packing_positions: Sequence[ComponentReference] | None = None
        else:
            references = design_input.component_references[0]
            selected = (design_input.valid_mask[0] & design_input.design_mask[0]).tolist()
            packing_positions = [
                reference for reference, keep in zip(references, selected) if keep and reference is not None
            ]
        packs = self.packer.pack_sequences(
            structure,
            sequences=[_candidate_protein_sequences(candidate.system) for candidate in design_result.candidates],
            positions=packing_positions,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return design_result, packs
