r"""ODesign design-and-assign composition.

The Pipeline generates backbone coordinates over seed and diffusion-sample axes, then assigns
ProteinMPNN or LigandMPNN sequence variants to each generated protein chain. Variants are ranked
per chain, merged by rank, and written into coordinate-bearing structures while fixed context
positions remain unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from os import PathLike
from typing import Literal

import numpy as np
import torch

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.molecular import ComponentReference, MolecularSystem, PolymerSequence, is_water_component
from ...data.structure import Structure
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...pipelines import CompositePair, CompositePipeline, resolve_generator
from ...results import Score
from ..ligandmpnn import LigandMpnnPipeline
from ..proteinmpnn import ProteinMpnnPipeline
from .model import (
    Hotspot,
    ODesignDesignCandidate,
    ODesignDesignResult,
    ODesignInput,
    ODesignModel,
    ODesignPlacements,
    ODesignReferenceCoordinates,
)
from .processor import ODesignProcessor


def _has_atom_context(system: MolecularSystem) -> bool:
    r"""Report whether the backbone system carries non-protein (ligand / nucleic) context.

    Protein atoms and water are excluded. Any other heavy atom selects LigandMPNN for every
    designed protein chain; its Processor applies the per-residue interface mask.
    """

    for entity in system.entities.values():
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            continue
        for component in entity.components:
            if is_water_component(component):
                continue
            if any(atom.element != "H" for atom in component.atoms):
                return True
    return False


def _odesign_scores(scores: dict[str, Score], *, used_ligand: bool) -> dict[str, Score]:
    r"""Map MPNN per-sequence scores to the ODesign score table.

    ProteinMPNN and LigandMPNN per-sequence scores preserve raw ``log_softmax`` semantics rather
    than the tempered sampling distribution. ProteinMPNN emits ``designed_sequence_nll``
    (nat/residue) while LigandMPNN emits ``designed_sequence_confidence`` = ``exp(-nll)``;
    each is the exact inverse of the other, so both branches populate the same four keys with
    identical units / ``preferred`` / scope. ``ligand_interface_confidence`` is present only
    when the LigandMPNN atom-context model was used (its scope is the LigandMPNN interface
    subset ``I``); empty-scope and NaN behavior is carried through unchanged from the
    input score (LigandMPNN emits ``scope=frozenset()`` + ``NaN`` for a chain with no
    ligand interface).
    """

    recovery = scores["sequence_recovery"]
    if "designed_sequence_nll" in scores:  # ProteinMPNN emits the nll directly.
        nll = scores["designed_sequence_nll"]
        nll_value = nll.value
        designed_scope = nll.scope
        confidence_value = math.exp(-nll_value) if math.isfinite(nll_value) else float("nan")
    else:  # LigandMPNN emits the confidence; recover the nll as its exact inverse.
        confidence = scores["designed_sequence_confidence"]
        designed_scope = confidence.scope
        confidence_value = confidence.value
        nll_value = -math.log(confidence_value) if confidence_value > 0 else float("nan")
    mapped: dict[str, Score] = {
        "designed_sequence_nll": Score(
            nll_value,
            unit="nat/residue",
            preferred="lower",
            scope=designed_scope,
        ),
        "designed_sequence_confidence": Score(
            confidence_value,
            preferred="higher",
            scope=designed_scope,
        ),
        "sequence_recovery": Score(
            recovery.value,
            preferred="higher",
            scope=recovery.scope,
        ),
    }
    if used_ligand:
        interface = scores["ligand_interface_confidence"]
        mapped["ligand_interface_confidence"] = Score(
            interface.value,
            preferred="higher",
            scope=interface.scope,
        )
    return mapped


def _generated_protein_chains(input: ODesignInput) -> tuple[tuple[ComponentReference, ...], ...]:
    r"""Return generated protein positions partitioned in realized chain order."""

    generated = {
        reference
        for placements in input.layout.placement_map.values()
        for occurrences in placements.values()
        for reference in occurrences
    }
    has_placements = input.layout.placement is not None
    chains = []
    for instance_id, entity_id in input.layout.system.instances.items():
        entity = input.layout.system.entities[entity_id]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
            continue
        positions = tuple(
            (instance_id, component_index)
            for component_index in range(len(entity))
            if not has_placements or (instance_id, component_index) in generated
        )
        if positions:
            chains.append(positions)
    return tuple(chains)


def _decoding_ranks(priorities: torch.Tensor, design_mask: torch.Tensor) -> torch.Tensor:
    r"""Lower ProteinMPNN random priorities to explicit causal ranks."""

    order = torch.argsort((design_mask.to(priorities.dtype) + 0.0001) * priorities.abs(), dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(order.shape[1], device=order.device).expand_as(order),
    )
    return ranks


def _protein_sequence_variants(
    designer: ProteinMpnnPipeline,
    structure: Structure,
    positions: tuple[ComponentReference, ...],
    *,
    num_sequences: int,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[tuple[dict[ComponentReference, int], dict[str, Score]], ...]:
    r"""Run ProteinMPNN sampling for one target chain."""

    model_input = designer.processor(structure, design_positions=positions)
    device = model_input.backbone_coordinates.device
    length = model_input.sequence_tokens.shape[1]
    design_mask = (model_input.residue_mask & model_input.design_positions).expand(num_sequences, -1)
    priorities = torch.randn(
        num_sequences,
        length,
        device=device,
        dtype=model_input.backbone_coordinates.dtype,
        generator=generator,
    )
    omit = torch.zeros(len(AMINO_ACID_COMPONENT_IDS) + 1, dtype=torch.bool, device=device)
    omit[AMINO_ACID_UNKNOWN_TOKEN] = True
    sampled = designer.model.sample(
        model_input,
        num_samples=num_sequences,
        temperature=temperature,
        decoding_ranks=_decoding_ranks(priorities, design_mask),
        global_omit_token_mask=omit,
        generator=generator,
    )

    repeated_native = model_input.sequence_tokens.expand(num_sequences, -1)
    losses = -torch.gather(sampled.log_probabilities, -1, sampled.sequences.unsqueeze(-1)).squeeze(-1)
    selected = design_mask.to(losses.dtype)
    selected_count = selected.sum(dim=-1)
    nll = (losses * selected).sum(dim=-1) / selected_count
    recovery = ((sampled.sequences == repeated_native).to(losses.dtype) * selected).sum(dim=-1) / selected_count
    log_probability_sum = -(losses * selected).sum(dim=-1)

    references = model_input.component_references[0]
    selected_host = design_mask[0].detach().cpu().tolist()
    scope = frozenset(
        reference
        for reference, selected_position in zip(references, selected_host)
        if reference is not None and selected_position
    )
    sequences = sampled.sequences.detach().cpu().tolist()
    nll_values = nll.detach().cpu().tolist()
    recovery_values = recovery.detach().cpu().tolist()
    ranking = torch.argsort(log_probability_sum, descending=True, stable=True).detach().cpu().tolist()

    variants = []
    for index in ranking:
        replacements = {
            reference: int(token)
            for reference, token, selected_position in zip(references, sequences[index], selected_host)
            if reference is not None and selected_position
        }
        variants.append(
            (
                replacements,
                _odesign_scores(
                    {
                        "designed_sequence_nll": Score(
                            float(nll_values[index]),
                            unit="nat/residue",
                            preferred="lower",
                            scope=scope,
                        ),
                        "sequence_recovery": Score(
                            float(recovery_values[index]),
                            preferred="higher",
                            scope=scope,
                        ),
                    },
                    used_ligand=False,
                ),
            )
        )
    return tuple(variants)


def _ligand_sequence_variants(
    designer: LigandMpnnPipeline,
    structure: Structure,
    positions: tuple[ComponentReference, ...],
    *,
    num_sequences: int,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[tuple[dict[ComponentReference, int], dict[str, Score]], ...]:
    r"""Run LigandMPNN for one target chain and sort variants by confidence."""

    result = designer.design(
        structure,
        design_positions=positions,
        num_candidates=num_sequences,
        temperature=temperature,
        generator=generator,
    )
    variants = []
    for candidate in result.candidates:
        scores = _odesign_scores(candidate.scores, used_ligand=True)
        interface = scores["ligand_interface_confidence"]
        rank_score = (
            interface if interface.scope and math.isfinite(interface.value) else scores["designed_sequence_confidence"]
        )
        variants.append(
            (
                {
                    reference: AMINO_ACID_TOKEN_BY_COMPONENT[candidate.system.component(reference).component_id]
                    for reference in positions
                },
                scores,
                rank_score.value,
            )
        )
    variants.sort(key=lambda value: value[2], reverse=True)
    return tuple((replacements, scores) for replacements, scores, _ in variants)


def _combine_chain_scores(chain_scores: Sequence[dict[str, Score]]) -> dict[str, Score]:
    r"""Combine disjoint per-chain averages into assembly-level per-residue Scores."""

    def combine(name: str) -> tuple[float, frozenset]:
        values = [scores[name] for scores in chain_scores]
        scope = frozenset(reference for value in values for reference in value.scope or ())
        count = sum(len(value.scope or ()) for value in values)
        if count == 0:
            return float("nan"), scope
        combined = sum(value.value * len(value.scope or ()) for value in values) / count
        return combined, scope

    nll, designed_scope = combine("designed_sequence_nll")
    recovery, recovery_scope = combine("sequence_recovery")
    result = {
        "designed_sequence_nll": Score(nll, unit="nat/residue", preferred="lower", scope=designed_scope),
        "designed_sequence_confidence": Score(
            math.exp(-nll) if math.isfinite(nll) else float("nan"),
            preferred="higher",
            scope=designed_scope,
        ),
        "sequence_recovery": Score(recovery, preferred="higher", scope=recovery_scope),
    }
    if any("ligand_interface_confidence" in scores for scores in chain_scores):
        interface_scores = [scores["ligand_interface_confidence"] for scores in chain_scores]
        interface_scope = frozenset(reference for value in interface_scores for reference in value.scope or ())
        count = sum(len(value.scope or ()) for value in interface_scores)
        interface_nll = (
            sum(-math.log(value.value) * len(value.scope or ()) for value in interface_scores if value.scope) / count
            if count
            and all(value.value > 0 and math.isfinite(value.value) for value in interface_scores if value.scope)
            else float("nan")
        )
        result["ligand_interface_confidence"] = Score(
            math.exp(-interface_nll) if math.isfinite(interface_nll) else float("nan"),
            preferred="higher",
            scope=interface_scope,
        )
    return result


def _structure_with_system(structure: Structure, system: MolecularSystem) -> Structure:
    r"""Retain the generated backbone coordinates while replacing residue identities."""

    retained = []
    for index, site in enumerate(structure.sites):
        try:
            system.atom(site)
        except KeyError:
            continue
        retained.append(index)
    indices = np.asarray(retained, dtype=np.int64)
    return Structure(
        system=system,
        sites=tuple(structure.sites[index] for index in retained),
        coordinates=structure.coordinates[indices],
        coordinate_mask=structure.coordinate_mask[indices],
        occupancy=None if structure.occupancy is None else structure.occupancy[indices],
        residue_numbers=structure.residue_numbers,
    )


class ODesignPipeline(CompositePipeline):
    r"""Compose the ODesign backbone designer with MPNN sequence assignment.

    ODesign generates backbones over S seeds and M diffusion samples. Each backbone receives K
    ranked sequence variants from the ProteinMPNN or LigandMPNN Pipeline.

    Args:
        model: ODesign backbone Model.
        processor: Processor paired with ``model``.
        protein_designer: ProteinMPNN Pipeline used without atom context.
        ligand_designer: LigandMPNN Pipeline used when atom context is present.
    """

    # The ODesign backbone pair owns the root, and the two sequence
    # designers live at fixed subdirectories. Each designer is a
    # single-pair child Pipeline, so its reloaded pair is wrapped back into its Pipeline class.
    pair_layout = (
        CompositePair("", "model", "processor"),
        CompositePair("protein_designer", "protein_designer", None, wrap=ProteinMpnnPipeline),
        CompositePair("ligand_designer", "ligand_designer", None, wrap=LigandMpnnPipeline),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        protein_designer: ProteinMpnnPipeline,
        ligand_designer: LigandMpnnPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.protein_designer = protein_designer
        self.ligand_designer = ligand_designer

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: ODesignPlacements | None = None,
        num_candidates: int | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 0.1,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        receptor_mode: Literal["rigid", "flex"] | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> ODesignDesignResult:
        r"""Generate the ODesign backbone-and-sequence grid.

        Args:
            input: Optional target or context system, structure, or structure path.
            placement: Optional topology of existing spans and generated polymer segments.
            num_candidates: Standard OFoldX design count, mapped to ``num_backbones`` when set.
            num_seeds: Number of generation seeds to draw when ``seeds`` is not supplied.
            num_backbones: Number of diffusion backbones generated for each seed.
            num_sequences: Number of ranked inverse-folding variants assigned to each backbone.
            temperature: ProteinMPNN or LigandMPNN sequence-sampling temperature.
            seeds: Optional explicit generation seeds; their count replaces ``num_seeds``.
            seed: Optional integer seed used to draw generation seeds.
            generator: Optional PyTorch random generator used to draw generation seeds.
            receptor_mode: Optional receptor treatment, ``"rigid"`` or ``"flex"``.
            conditions: Hotspot, covalent-bond, and reference-coordinate conditions.
            num_sampling_steps: Number of coordinate-diffusion sampling steps per backbone.
            partial_diffusion_snr: Optional signal-to-noise ratio for partial diffusion from
                observed coordinates.

        Returns:
            An ``ODesignDesignResult`` containing the coordinate-bearing design grid.

        Notes:
            The candidate axes are seed (S), backbone (M), then sequence (K), and candidates are
            ordered S -> M -> K. ``num_candidates`` replaces M when supplied; explicit ``seeds``
            replace S. ``seed`` and ``generator`` are mutually exclusive.
        """

        generator = resolve_generator(seed, generator)
        if num_candidates is not None:
            # Standard single-axis entry: map the OFoldX design count onto the primary
            # backbone-generation axis (see the method docstring).
            num_backbones = num_candidates
        if num_seeds <= 0 or num_backbones <= 0 or num_sequences <= 0:
            raise ValueError("num_seeds, num_backbones, and num_sequences must be positive")

        seed_list = self._resolve_seeds(num_seeds, seeds, generator)

        model_input = self._build_batch(input, placement, conditions, receptor_mode)
        candidates: list[ODesignDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(int(backbone_seed), None, device=model_input.reference_positions.device)
            realized_input = self.processor.realize_ranges(model_input, generator=stream)
            backbones = self.model.design(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            design_chains = _generated_protein_chains(realized_input)
            if not design_chains:
                raise ValueError(
                    "ODesign protein sequence assignment requires at least one designable protein position"
                )

            for backbone_index, backbone in enumerate(backbones.candidates):
                structure = backbone.structure
                used_ligand = _has_atom_context(structure.system)
                chain_variants = []
                for positions in design_chains:
                    variants = (
                        _ligand_sequence_variants(
                            self.ligand_designer,
                            structure,
                            positions,
                            num_sequences=num_sequences,
                            temperature=temperature,
                            generator=stream,
                        )
                        if used_ligand
                        else _protein_sequence_variants(
                            self.protein_designer,
                            structure,
                            positions,
                            num_sequences=num_sequences,
                            temperature=temperature,
                            generator=stream,
                        )
                    )
                    chain_variants.append(variants)

                for sequence_index in range(num_sequences):
                    replacements = {
                        reference: token
                        for variants in chain_variants
                        for reference, token in variants[sequence_index][0].items()
                    }
                    system = structure.system.replace_components(
                        {
                            reference: get_component(AMINO_ACID_COMPONENT_IDS[token])
                            for reference, token in replacements.items()
                        }
                    )
                    assigned_structure = _structure_with_system(structure, system)
                    candidates.append(
                        ODesignDesignCandidate(
                            structure=assigned_structure,
                            scores=_combine_chain_scores([variants[sequence_index][1] for variants in chain_variants]),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )
        return ODesignDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )

    def _resolve_seeds(
        self,
        num_seeds: int,
        seeds: Sequence[int] | None,
        generator: torch.Generator | None,
    ) -> tuple[int, ...]:
        r"""Fix the S seed list (explicit ``configs.seeds`` or drawn from one base stream)."""

        if seeds is not None:
            seed_list = tuple(int(value) for value in seeds)
            if not seed_list:
                raise ValueError("seeds must not be empty")
            return seed_list
        device = None if generator is None else generator.device
        drawn = torch.randint(
            1,
            torch.iinfo(torch.int32).max,
            (num_seeds,),
            generator=generator,
            device=device,
        )
        return tuple(int(value) for value in drawn.tolist())

    def _build_batch(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        placement: ODesignPlacements | None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates],
        receptor_mode: Literal["rigid", "flex"] | None,
    ) -> ODesignInput:
        r"""Lower the canonical input once, honoring an optional per-call receptor_mode override."""

        return self.processor(
            input,
            placement=placement,
            conditions=conditions,
            receptor_mode=receptor_mode,
        )
