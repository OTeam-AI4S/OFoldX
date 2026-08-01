r"""
ProteinMPNN sequence design, scoring, and probability Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch
from torch import Tensor

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    AMINO_ACID_ONE_TO_THREE,
    ENTITY_PROTEIN,
    ComponentReference,
    MolecularSystem,
)
from ...data.structure import Structure
from ...io import read_structure
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...pipelines import CompositePair, CompositePipeline, PairedPipeline, resolve_generator
from ...results import DesignAndPackResult, MpnnProbabilityResult, ScoreResult
from ..mpnn_packer import MpnnPackerPipeline
from .model import ProteinMpnnDesignResult, ProteinMpnnModel
from .processor import ProteinMpnnProcessor

_GENERATED_TOKEN_INDEX_BY_LETTER: dict[str, int] = {
    letter: AMINO_ACID_TOKEN_BY_COMPONENT[component_id] for letter, component_id in AMINO_ACID_ONE_TO_THREE.items()
}
_GENERATED_TOKEN_INDEX_BY_LETTER["X"] = AMINO_ACID_UNKNOWN_TOKEN


class ProteinMpnnPipeline(PairedPipeline[ProteinMpnnModel, ProteinMpnnProcessor]):
    r"""
    Protein sequence design pipeline for ProteinMPNN.

    Load a published pair with [`ofoldx.pipeline`][]. Every verb conditions on observed
    backbone coordinates, so it takes a structure read from a file rather than a topology
    built from a sequence; ``examples/runners/proteinmpnn.py`` runs one end to end.
    """

    model_architecture = "proteinmpnn"

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
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
        coordinate_noise: float = 0.0,
        logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = ("X",),
        pssm_mixture: float = 0.0,
        pssm_log_odds_threshold: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ProteinMpnnDesignResult:
        r"""
        Design ProteinMPNN sequences on a fixed backbone.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the design target.
            positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with the required backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 21-token vocabulary.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires matching
                ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 21-token vocabulary.
            pssm_log_odds: Per-residue log-odds vectors over the 21-token vocabulary.
            num_candidates: Number of complete sequence candidates to sample.
            num_decoding_orders: Number of decoding orders used to score the input sequence.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            logit_bias: Global bias vector over the 21-token vocabulary.
            global_omit_tokens: One-letter residue tokens excluded from sampling.
            pssm_mixture: Global PSSM mixture weight between zero and one.
            pssm_log_odds_threshold: Optional minimum PSSM log odds retained during sampling.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            A ``ProteinMpnnDesignResult`` containing sampled candidates and the input-sequence
            ``ScoreResult``.

        Notes:
            ``seed`` and ``generator`` are mutually exclusive. Candidates follow sampling order;
            input-score evaluations follow decoding-order draw order.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=logit_bias,
            global_omit_token_mask=_global_omit_token_mask(global_omit_tokens, device=model_input.residue_mask.device),
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )

    def score(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score a ProteinMPNN sequence over independently drawn decoding orders.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the scoring target.
            sequence: Optional target sequence as one-letter text or an instance-to-sequence
                mapping. ``None`` scores the structure's identities.
            positions: Residue occurrences included in the designed-sequence score.
                ``None`` selects every valid protein occurrence.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            A ``ScoreResult`` containing designed and global sequence NLL scores.

        Notes:
            Score ``evaluations`` follow the decoding-order axis. ``seed`` and ``generator`` are
            mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            sequence=sequence,
            positions=positions,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.score(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute sequence-conditioned autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the probability target.
            positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with sequence-conditioned logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
            conditions=conditions,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute sequence-free autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the probability target.
            positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with sequence-free logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
            conditions=conditions,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.sequence_free_autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute target-last conditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the probability target.
            positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with each target decoded after all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.conditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute target-first unconditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            protein_instances: Optional protein instance IDs retained as the probability target.
            positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with each target decoded before all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.unconditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            generator=generator,
        )


class ProteinMpnnDesignAndPackPipeline(CompositePipeline):
    r"""
    Design protein sequences with ProteinMPNN and pack their side chains.

    Args:
        designer: ProteinMPNN sequence-design pipeline.
        packer: Side-chain packing pipeline.
    """

    pair_layout = (
        CompositePair("", "proteinmpnn", "designer", None, wrap=ProteinMpnnPipeline),
        CompositePair("packer", "mpnn_packer", "packer", None, wrap=MpnnPackerPipeline),
    )

    def __init__(
        self,
        *,
        designer: ProteinMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None:
        self.designer = designer
        self.packer = packer
        self._validate_components()

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        protein_instances: Sequence[str] | None = None,
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
        coordinate_noise: float = 0.0,
        logit_bias: Tensor | None = None,
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
    ) -> DesignAndPackResult:
        r"""
        Design ProteinMPNN sequences and pack every resulting candidate.

        Args:
            input: A ``Structure`` or local structure path, parsed once for both stages.
            protein_instances: Optional protein instance IDs retained by both stages.
            positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with the required backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 21-token vocabulary.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires matching
                ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 21-token vocabulary.
            pssm_log_odds: Per-residue log-odds vectors over the 21-token vocabulary.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            num_decoding_orders: Number of decoding orders used to score the input sequence.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of coordinate noise applied before encoding.
            logit_bias: Global bias vector over the 21-token vocabulary.
            global_omit_tokens: One-letter residue tokens excluded from sampling.
            pssm_mixture: Global PSSM mixture weight between zero and one.
            pssm_log_odds_threshold: Optional minimum PSSM log odds retained during sampling.
            use_packing_atom_context: Whether the packing stage uses nonprotein atom context.
            repack_everything: Whether to pack every eligible residue instead of only designed
                positions.
            num_packs_per_candidate: Number of packed structures generated per sequence candidate.
            num_denoising_steps: Number of side-chain denoising steps per packed structure.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator advanced through design before packing.

        Returns:
            The sequence-design result and its aligned side-chain packing results.

        Notes:
            The ``PackResult`` tuple follows candidate order, and each result's predictions follow
            packing-sample order. ``seed`` and ``generator`` are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.designer.processor(
            structure,
            protein_instances=protein_instances,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=filter_by_occupancy,
        )
        device = model_input.residue_mask.device
        generator = resolve_generator(seed, generator, device=device)
        design_result = self.designer.model.design(
            model_input,
            num_candidates=num_candidates,
            num_decoding_orders=num_decoding_orders,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            global_logit_bias=logit_bias,
            global_omit_token_mask=_global_omit_token_mask(global_omit_tokens, device=device),
            pssm_mixture=pssm_mixture,
            pssm_log_odds_threshold=pssm_log_odds_threshold,
            generator=generator,
        )
        if repack_everything:
            pack_positions: Sequence[ComponentReference] | None = None
        else:
            references = tuple(
                None if component < 0 else model_input.structure.system.component_reference(component)
                for component in model_input.component_index[0].detach().cpu().tolist()
            )
            selected = (model_input.residue_mask[0] & model_input.design_positions[0]).tolist()
            pack_positions = [
                reference for reference, keep in zip(references, selected) if keep and reference is not None
            ]
        packs = self.packer.pack_sequences(
            model_input.structure,
            sequences=[_candidate_protein_sequences(candidate.system) for candidate in design_result.candidates],
            pack_positions=pack_positions,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return DesignAndPackResult(designs=design_result, packs=packs)


def _global_omit_token_mask(
    tokens: Sequence[str],
    *,
    device: torch.device,
) -> Tensor | None:
    r"""
    Lower one-letter omit tokens to a generatable-class support mask.

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
    r"""
    Map each protein instance to its realized component-ID sequence.
    """

    sequences: dict[str, tuple[str, ...]] = {}
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        if int(system.entity_type[entity]) != ENTITY_PROTEIN:
            continue
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        sequences[system.instance_id(instance)] = tuple(
            system.symbols[int(symbol)] for symbol in system.component_symbol[start:stop].tolist()
        )
    return sequences
