r"""
Released ODesign backbone generation and inverse-folding compositions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MsaCollection
from ...data.molecular import (
    ENTITY_DNA,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    copy_components,
    replace_components,
    water_component_mask,
)
from ...data.placement import PlacementSpec
from ...data.structure import Structure
from ...pipelines import CompositePair, CompositePipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_structure_input
from ...results import (
    DesignCandidate,
    Metric,
    Score,
    best,
)
from ..grnade import GrnadePipeline
from ..ligandmpnn import LigandMpnnPipeline
from ..oinvfold import (
    OInvFoldLigandPipeline,
    OInvFoldPolymerPipeline,
)
from ..proteinmpnn import ProteinMpnnPipeline
from .model import ODesignInput, ODesignLigandInput, ODesignModel
from .processor import ODesignLigandProcessor, ODesignProcessor

_MSA_ENTITY_TYPES = {ENTITY_PROTEIN, ENTITY_DNA, ENTITY_RNA}


@dataclass(eq=False)
class ODesignPolymerCandidate:
    r"""One ODesign polymer sequence assigned to a generated backbone."""

    structure: Structure
    scores: dict[str, Score]
    metrics: dict[str, Metric]


@dataclass(eq=False)
class ODesignPolymerResult:
    r"""ODesign polymer candidates in seed-major, backbone-major, sequence-major order."""

    candidates: tuple[ODesignPolymerCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignPolymerCandidate:
        r"""Return the earliest best candidate under one explicit score."""

        return best(self.candidates, score)


@dataclass(eq=False)
class ODesignLigandCandidate:
    r"""One provisional ligand assignment whose bond topology remains unresolved."""

    context: Structure | None
    slot_ids: tuple[str, ...]
    coordinates: NDArray
    elements: tuple[str, ...]
    per_atom_probabilities: NDArray
    token_bond_logits: NDArray
    scores: dict[str, Score]
    bond_state: Literal["unresolved"] = "unresolved"


@dataclass(eq=False)
class ODesignLigandResult:
    r"""ODesign ligand candidates in seed-major, backbone-major, sequence-major order."""

    candidates: tuple[ODesignLigandCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignLigandCandidate:
        r"""Return the earliest best candidate under one explicit score."""

        return best(self.candidates, score)


class ODesignProteinPipeline(CompositePipeline):
    r"""Official ODesign protein generation with context-selected MPNN inverse folding."""

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "protein_designer",
            "proteinmpnn",
            "protein_designer",
            None,
            wrap=ProteinMpnnPipeline,
        ),
        CompositePair(
            "ligand_designer",
            "ligandmpnn",
            "ligand_designer",
            None,
            wrap=LigandMpnnPipeline,
        ),
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
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.artifact not in ("protein_flex", "protein_rigid"):
            raise ValueError(
                f"ODesignProteinPipeline requires a protein ODesign artifact; got {self.processor.config.artifact!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerResult:
        r"""Generate protein backbones and assign sequences with the released MPNN route."""

        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates: list[ODesignPolymerCandidate] = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                placement=placement,
                chain_lengths=chain_lengths,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            chains = _generated_polymer_chains(model_input, "protein")
            backbones = _sample_polymer_backbones(
                self.model,
                model_input,
                num_backbones=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone in backbones:
                positions = tuple(position for _instance_id, chain_positions in chains for position in chain_positions)
                target = _protein_backbone_for_inverse_folding(backbone, positions)
                use_ligand_mpnn = _has_ligand_atom_context(
                    target,
                    filter_by_occupancy=filter_by_occupancy,
                )
                selections: list[list[tuple[tuple[ComponentReference, ...], Structure]]] = [
                    [] for _ in range(num_sequences)
                ]
                totals: list[list[float]] = [[] for _ in range(num_sequences)]
                means: list[list[float]] = [[] for _ in range(num_sequences)]
                scopes: list[list[frozenset[ComponentReference]]] = [[] for _ in range(num_sequences)]
                for _instance_id, chain_positions in chains:
                    designer: Literal["proteinmpnn", "ligandmpnn"]
                    if use_ligand_mpnn:
                        result = self.ligand_designer.design(
                            target,
                            positions=chain_positions,
                            num_candidates=num_sequences,
                            temperature=temperature,
                            filter_by_occupancy=filter_by_occupancy,
                            generator=stream,
                        )
                        designer = "ligandmpnn"
                    else:
                        result = self.protein_designer.design(
                            target,
                            positions=chain_positions,
                            num_candidates=num_sequences,
                            temperature=temperature,
                            generator=stream,
                        )
                        designer = "proteinmpnn"
                    for sequence_index, candidate in enumerate(result.candidates):
                        total, mean, scope = _mpnn_log_probabilities(
                            candidate,
                            designer,
                            design_scope=frozenset(chain_positions),
                        )
                        selections[sequence_index].append(
                            (
                                chain_positions,
                                _structure_with_system(target, candidate.system),
                            )
                        )
                        totals[sequence_index].append(total)
                        means[sequence_index].append(mean)
                        scopes[sequence_index].append(scope)
                candidates.extend(
                    _polymer_candidate(
                        backbone,
                        tuple(selections[sequence_index]),
                        tuple(chain[0] for chain in chains),
                        tuple(totals[sequence_index]),
                        tuple(means[sequence_index]),
                        tuple(scopes[sequence_index]),
                    )
                    for sequence_index in range(num_sequences)
                )
        return ODesignPolymerResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignDnaPipeline(CompositePipeline):
    r"""Official ODesign DNA generation with the released DNA OInvFold child."""

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "sequence_designer",
            "oinvfold",
            "sequence_designer",
            None,
            wrap=OInvFoldPolymerPipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        sequence_designer: OInvFoldPolymerPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.sequence_designer = sequence_designer
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.artifact != "nucleic_acid_rigid":
            raise ValueError(
                f"ODesignDnaPipeline requires the nucleic-acid ODesign artifact; got {self.processor.config.artifact!r}"
            )
        if self.sequence_designer.processor.config.modality != "dna":
            raise ValueError(
                "ODesignDnaPipeline requires a DNA OInvFold child; "
                f"got {self.sequence_designer.processor.config.modality!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerResult:
        r"""Generate DNA backbones and assign nucleotide identities with OInvFold."""

        _require_sequence_count(num_sequences, use_beam)
        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates: list[ODesignPolymerCandidate] = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                placement=placement,
                chain_lengths=chain_lengths,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            chains = _generated_polymer_chains(model_input, "dna")
            backbones = _sample_polymer_backbones(
                self.model,
                model_input,
                num_backbones=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone in backbones:
                selections: list[list[tuple[tuple[ComponentReference, ...], Structure]]] = [
                    [] for _ in range(num_sequences)
                ]
                totals: list[list[float]] = [[] for _ in range(num_sequences)]
                means: list[list[float]] = [[] for _ in range(num_sequences)]
                scopes: list[list[frozenset[ComponentReference]]] = [[] for _ in range(num_sequences)]
                for instance_id, positions in chains:
                    target = _nucleic_acid_backbone_for_inverse_folding(
                        backbone,
                        positions,
                        "dna",
                    )
                    result = self.sequence_designer.design(
                        target,
                        instance_id=instance_id,
                        design_positions=positions,
                        topk=num_sequences,
                        use_beam=use_beam,
                        temperature=temperature,
                        filter_by_occupancy=filter_by_occupancy,
                    )
                    scope = frozenset(
                        reference
                        for reference, selected in zip(
                            result.component_references,
                            result.design_mask.tolist(),
                            strict=True,
                        )
                        if selected
                    )
                    for sequence_index, candidate in enumerate(result.candidates):
                        selections[sequence_index].append((positions, candidate.structure))
                        totals[sequence_index].append(candidate.total_log_probability)
                        means[sequence_index].append(candidate.mean_log_probability)
                        scopes[sequence_index].append(scope)
                candidates.extend(
                    _polymer_candidate(
                        backbone,
                        tuple(selections[sequence_index]),
                        tuple(chain[0] for chain in chains),
                        tuple(totals[sequence_index]),
                        tuple(means[sequence_index]),
                        tuple(scopes[sequence_index]),
                    )
                    for sequence_index in range(num_sequences)
                )
        return ODesignPolymerResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignRnaPipeline(CompositePipeline):
    r"""Official ODesign RNA generation with the released gRNAde v1 child."""

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "sequence_designer",
            "grnade",
            "sequence_designer",
            None,
            wrap=GrnadePipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        sequence_designer: GrnadePipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.sequence_designer = sequence_designer
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.artifact != "nucleic_acid_rigid":
            raise ValueError(
                f"ODesignRnaPipeline requires the nucleic-acid ODesign artifact; got {self.processor.config.artifact!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerResult:
        r"""Generate RNA backbones and sample nucleotide identities with gRNAde."""

        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates: list[ODesignPolymerCandidate] = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                placement=placement,
                chain_lengths=chain_lengths,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            chains = _generated_polymer_chains(model_input, "rna")
            backbones = _sample_polymer_backbones(
                self.model,
                model_input,
                num_backbones=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone in backbones:
                selections: list[list[tuple[tuple[ComponentReference, ...], Structure]]] = [
                    [] for _ in range(num_sequences)
                ]
                totals: list[list[float]] = [[] for _ in range(num_sequences)]
                means: list[list[float]] = [[] for _ in range(num_sequences)]
                scopes: list[list[frozenset[ComponentReference]]] = [[] for _ in range(num_sequences)]
                for instance_id, positions in chains:
                    target = _nucleic_acid_backbone_for_inverse_folding(
                        backbone,
                        positions,
                        "rna",
                    )
                    result = self.sequence_designer.design(
                        target,
                        instance_id=instance_id,
                        design_positions=positions,
                        num_candidates=num_sequences,
                        temperature=temperature,
                        filter_by_occupancy=filter_by_occupancy,
                        generator=stream,
                    )
                    for sequence_index, candidate in enumerate(result.candidates):
                        total = candidate.scores["total_log_probability"]
                        mean = candidate.scores["mean_log_probability"]
                        selections[sequence_index].append(
                            (
                                positions,
                                _rna_structure_with_system(
                                    target,
                                    candidate.system,
                                    positions,
                                ),
                            )
                        )
                        totals[sequence_index].append(total.value)
                        means[sequence_index].append(mean.value)
                        scopes[sequence_index].append(cast(frozenset[ComponentReference], total.scope))
                candidates.extend(
                    _polymer_candidate(
                        backbone,
                        tuple(selections[sequence_index]),
                        tuple(chain[0] for chain in chains),
                        tuple(totals[sequence_index]),
                        tuple(means[sequence_index]),
                        tuple(scopes[sequence_index]),
                    )
                    for sequence_index in range(num_sequences)
                )
        return ODesignPolymerResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignLigandPipeline(CompositePipeline):
    r"""Official ODesign provisional ligand generation with ligand OInvFold."""

    pair_layout = (
        CompositePair("", "odesign_ligand", "model", "processor"),
        CompositePair(
            "element_designer",
            "oinvfold_ligand",
            "element_designer",
            None,
            wrap=OInvFoldLigandPipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignLigandProcessor,
        element_designer: OInvFoldLigandPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.element_designer = element_designer
        self._validate_components()

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        generated_atoms: int | range,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignLigandResult:
        r"""Generate provisional ligand coordinates and assign element identities."""

        _require_sequence_count(num_sequences, use_beam)
        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                generated_atoms=generated_atoms,
                msa=msa_value,
                hotspots=hotspots,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            output = self.model.sample(
                model_input,
                num_samples=num_backbones,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                num_model_seeds=1,
                partial_diffusion_snr=partial_diffusion_snr,
                generator=stream,
            )
            generated_token_mask = model_input.token_mask[0].bool() & model_input.token_component_index[0].lt(0)
            bond_logits = output.token_bond_logits[0, 0]
            token_bond_logits = bond_logits[generated_token_mask][:, generated_token_mask].detach().cpu().numpy()
            for backbone_index in range(num_backbones):
                coordinates = output.coordinates[0, 0, backbone_index]
                slot_mask = model_input.atom_mask[0].bool() & (model_input.atom_index[0] < 0)
                slot_coordinates = coordinates[slot_mask]
                context = _structure_from_ligand_context(coordinates, model_input)
                slot_ids = tuple(str(index) for index in range(slot_coordinates.shape[0]))
                result = self.element_designer.design(
                    slot_coordinates,
                    slot_ids=slot_ids,
                    context=context,
                    topk=num_sequences,
                    use_beam=use_beam,
                    temperature=temperature,
                )
                for candidate in result.candidates:
                    candidates.append(
                        ODesignLigandCandidate(
                            context=candidate.context,
                            slot_ids=candidate.slot_ids,
                            coordinates=candidate.coordinates,
                            elements=candidate.elements,
                            per_atom_probabilities=candidate.per_atom_probabilities,
                            token_bond_logits=token_bond_logits,
                            scores={
                                "total_log_probability": Score(
                                    candidate.total_log_probability,
                                    unit="nat",
                                    preferred="higher",
                                ),
                                "mean_log_probability": Score(
                                    candidate.mean_log_probability,
                                    unit="nat/atom",
                                    preferred="higher",
                                ),
                            },
                            bond_state=candidate.bond_state,
                        )
                    )
        return ODesignLigandResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignOInvFoldProteinPipeline(CompositePipeline):
    r"""Explicit OInvFold alternative for ODesign protein backbones."""

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "sequence_designer",
            "oinvfold",
            "sequence_designer",
            None,
            wrap=OInvFoldPolymerPipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        sequence_designer: OInvFoldPolymerPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.sequence_designer = sequence_designer
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.artifact not in ("protein_flex", "protein_rigid"):
            raise ValueError(
                "ODesignOInvFoldProteinPipeline requires a protein ODesign artifact; "
                f"got {self.processor.config.artifact!r}"
            )
        if self.sequence_designer.processor.config.modality != "protein":
            raise ValueError(
                "ODesignOInvFoldProteinPipeline requires a protein OInvFold child; "
                f"got {self.sequence_designer.processor.config.modality!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerResult:
        r"""Generate protein backbones and decode them with protein OInvFold."""

        _require_sequence_count(num_sequences, use_beam)
        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates: list[ODesignPolymerCandidate] = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                placement=placement,
                chain_lengths=chain_lengths,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            chains = _generated_polymer_chains(model_input, "protein")
            backbones = _sample_polymer_backbones(
                self.model,
                model_input,
                num_backbones=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone in backbones:
                target = _protein_backbone_for_inverse_folding(
                    backbone,
                    tuple(position for _instance_id, positions in chains for position in positions),
                )
                selections: list[list[tuple[tuple[ComponentReference, ...], Structure]]] = [
                    [] for _ in range(num_sequences)
                ]
                totals: list[list[float]] = [[] for _ in range(num_sequences)]
                means: list[list[float]] = [[] for _ in range(num_sequences)]
                scopes: list[list[frozenset[ComponentReference]]] = [[] for _ in range(num_sequences)]
                for instance_id, positions in chains:
                    result = self.sequence_designer.design(
                        target,
                        instance_id=instance_id,
                        design_positions=positions,
                        topk=num_sequences,
                        use_beam=use_beam,
                        temperature=temperature,
                        filter_by_occupancy=filter_by_occupancy,
                    )
                    scope = frozenset(
                        reference
                        for reference, selected in zip(
                            result.component_references,
                            result.design_mask.tolist(),
                            strict=True,
                        )
                        if selected
                    )
                    for sequence_index, candidate in enumerate(result.candidates):
                        selections[sequence_index].append((positions, candidate.structure))
                        totals[sequence_index].append(candidate.total_log_probability)
                        means[sequence_index].append(candidate.mean_log_probability)
                        scopes[sequence_index].append(scope)
                candidates.extend(
                    _polymer_candidate(
                        backbone,
                        tuple(selections[sequence_index]),
                        tuple(chain[0] for chain in chains),
                        tuple(totals[sequence_index]),
                        tuple(means[sequence_index]),
                        tuple(scopes[sequence_index]),
                    )
                    for sequence_index in range(num_sequences)
                )
        return ODesignPolymerResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignOInvFoldRnaPipeline(CompositePipeline):
    r"""Explicit OInvFold alternative for ODesign RNA backbones."""

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "sequence_designer",
            "oinvfold",
            "sequence_designer",
            None,
            wrap=OInvFoldPolymerPipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        sequence_designer: OInvFoldPolymerPipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.sequence_designer = sequence_designer
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.artifact != "nucleic_acid_rigid":
            raise ValueError(
                "ODesignOInvFoldRnaPipeline requires the nucleic-acid ODesign artifact; "
                f"got {self.processor.config.artifact!r}"
            )
        if self.sequence_designer.processor.config.modality != "rna":
            raise ValueError(
                "ODesignOInvFoldRnaPipeline requires an RNA OInvFold child; "
                f"got {self.sequence_designer.processor.config.modality!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: PlacementSpec | None = None,
        chain_lengths: Sequence[int | None] | None = None,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerResult:
        r"""Generate RNA backbones and decode them with RNA OInvFold."""

        _require_sequence_count(num_sequences, use_beam)
        seed_values = _resolve_seeds(num_seeds, seeds, seed, generator)
        canonical_input = resolve_structure_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        msa_value = resolve_msa_input(msa, system, entity_types=_MSA_ENTITY_TYPES)
        candidates: list[ODesignPolymerCandidate] = []
        for backbone_seed in seed_values:
            stream = resolve_generator(
                backbone_seed,
                None,
                device=next(self.model.parameters()).device,
            )
            model_input = self.processor(
                canonical_input,
                placement=placement,
                chain_lengths=chain_lengths,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                filter_by_occupancy=filter_by_occupancy,
                generator=stream,
            )
            chains = _generated_polymer_chains(model_input, "rna")
            backbones = _sample_polymer_backbones(
                self.model,
                model_input,
                num_backbones=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for backbone in backbones:
                selections: list[list[tuple[tuple[ComponentReference, ...], Structure]]] = [
                    [] for _ in range(num_sequences)
                ]
                totals: list[list[float]] = [[] for _ in range(num_sequences)]
                means: list[list[float]] = [[] for _ in range(num_sequences)]
                scopes: list[list[frozenset[ComponentReference]]] = [[] for _ in range(num_sequences)]
                for instance_id, positions in chains:
                    target = _nucleic_acid_backbone_for_inverse_folding(
                        backbone,
                        positions,
                        "rna",
                    )
                    result = self.sequence_designer.design(
                        target,
                        instance_id=instance_id,
                        design_positions=positions,
                        topk=num_sequences,
                        use_beam=use_beam,
                        temperature=temperature,
                        filter_by_occupancy=filter_by_occupancy,
                    )
                    scope = frozenset(
                        reference
                        for reference, selected in zip(
                            result.component_references,
                            result.design_mask.tolist(),
                            strict=True,
                        )
                        if selected
                    )
                    for sequence_index, candidate in enumerate(result.candidates):
                        selections[sequence_index].append((positions, candidate.structure))
                        totals[sequence_index].append(candidate.total_log_probability)
                        means[sequence_index].append(candidate.mean_log_probability)
                        scopes[sequence_index].append(scope)
                candidates.extend(
                    _polymer_candidate(
                        backbone,
                        tuple(selections[sequence_index]),
                        tuple(chain[0] for chain in chains),
                        tuple(totals[sequence_index]),
                        tuple(means[sequence_index]),
                        tuple(scopes[sequence_index]),
                    )
                    for sequence_index in range(num_sequences)
                )
        return ODesignPolymerResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_values),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


def _resolve_seeds(
    num_seeds: int,
    seeds: Sequence[int] | None,
    seed: int | None,
    generator: torch.Generator | None,
) -> tuple[int, ...]:
    if seeds is not None:
        if seed is not None or generator is not None:
            raise ValueError("seeds cannot be combined with seed or generator")
        return tuple(int(value) for value in seeds)
    generator = resolve_generator(seed, generator)
    device = None if generator is None else generator.device
    values = torch.randint(
        1,
        torch.iinfo(torch.int32).max,
        (num_seeds,),
        generator=generator,
        device=device,
    )
    return tuple(int(value) for value in values.cpu().tolist())


def _require_sequence_count(num_sequences: int, use_beam: bool) -> None:
    if not use_beam and num_sequences != 1:
        raise ValueError("greedy OInvFold decoding returns one sequence; set use_beam=True for multiple sequences")


def _sample_polymer_backbones(
    model: ODesignModel,
    input: ODesignInput,
    *,
    num_backbones: int,
    generator: torch.Generator | None,
    recycling_steps: int | None,
    num_sampling_steps: int,
    partial_diffusion_snr: float | None,
) -> tuple[Structure, ...]:
    output = model.sample(
        input,
        num_samples=num_backbones,
        recycling_steps=recycling_steps,
        num_sampling_steps=num_sampling_steps,
        num_model_seeds=1,
        partial_diffusion_snr=partial_diffusion_snr,
        generator=generator,
    )
    return tuple(
        _structure_from_atom_coordinates(output.coordinates[0, 0, index], input) for index in range(num_backbones)
    )


def _structure_from_atom_coordinates(
    coordinates: torch.Tensor,
    input: ODesignInput,
) -> Structure:
    system = input.system.cpu()  # type: ignore[attr-defined]
    atom_index = input.atom_index[0].detach().cpu().long()
    valid = input.atom_mask[0].detach().cpu().bool() & (atom_index >= 0)
    canonical_atom_index = atom_index[valid]
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    complete = torch.zeros(atom_count, 3, dtype=coordinates.dtype)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    complete[canonical_atom_index] = coordinates[valid.to(coordinates.device)].detach().cpu()
    coordinate_mask[canonical_atom_index] = True
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=complete,
        coordinate_mask=coordinate_mask,
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _structure_from_ligand_context(
    coordinates: torch.Tensor,
    input: ODesignLigandInput,
) -> Structure | None:
    atom_index = input.atom_index[0].detach().cpu().long()
    valid = input.atom_mask[0].detach().cpu().bool() & (atom_index >= 0)
    if not bool(valid.any()):
        return None
    system = input.system.cpu()  # type: ignore[attr-defined]
    canonical_atom_index = atom_index[valid]
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    complete = torch.zeros(atom_count, 3, dtype=coordinates.dtype)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    complete[canonical_atom_index] = coordinates[valid.to(coordinates.device)].detach().cpu()
    coordinate_mask[canonical_atom_index] = True
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=complete,
        coordinate_mask=coordinate_mask,
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _generated_polymer_chains(
    input: ODesignInput,
    expected_modality: Literal["protein", "dna", "rna"],
) -> tuple[tuple[str, tuple[ComponentReference, ...]], ...]:
    chains = []
    system = input.system
    modality_by_type = {
        ENTITY_PROTEIN: "protein",
        ENTITY_DNA: "dna",
        ENTITY_RNA: "rna",
    }
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        generated_modality = modality_by_type.get(int(system.entity_type[entity]))
        if generated_modality is None:
            continue
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        positions = tuple(
            (system.instance_id(instance), component - start)
            for component in range(start, stop)
            if bool(input.design_component_mask[component])
        )
        if positions:
            if generated_modality != expected_modality:
                raise ValueError(
                    f"ODesign {expected_modality} design accepts only generated {expected_modality} chains"
                )
            chains.append((system.instance_id(instance), positions))
    if not chains:
        raise ValueError(f"ODesign {expected_modality} design requires generated {expected_modality} positions")
    return tuple(chains)


def _protein_backbone_for_inverse_folding(
    structure: Structure,
    positions: tuple[ComponentReference, ...],
) -> Structure:
    component_index = torch.tensor(
        [structure.system.component_index(reference) for reference in positions],
        dtype=torch.int64,
        device=structure.system.component_symbol.device,
    )
    system, old_to_new, _ = replace_components(
        structure.system,
        component_index,
        [get_component("GLY")] * len(positions),
    )
    return _remap_structure(structure, system, old_to_new)


_NUCLEIC_ACID_IDENTITIES = {
    ("dna", "A"): ("DA", "N9"),
    ("rna", "A"): ("A", "N9"),
    ("rna", "C"): ("C", "N1"),
    ("rna", "G"): ("G", "N9"),
    ("rna", "U"): ("U", "N1"),
}


def _nucleic_acid_backbone_for_inverse_folding(
    structure: Structure,
    positions: tuple[ComponentReference, ...],
    modality: Literal["dna", "rna"],
) -> Structure:
    component_id, base_atom_id = _NUCLEIC_ACID_IDENTITIES[modality, "A"]
    component_index = torch.tensor(
        [structure.system.component_index(reference) for reference in positions],
        dtype=torch.int64,
        device=structure.system.component_symbol.device,
    )
    system, old_to_new, _ = replace_components(
        structure.system,
        component_index,
        [get_component(component_id)] * len(positions),
    )
    output = _remap_structure(structure, system, old_to_new)
    for reference in positions:
        source_atom = structure.system.atom_index(AtomSite(reference[0], reference[1], "N"))
        target_atom = system.atom_index(AtomSite(reference[0], reference[1], base_atom_id))
        _copy_atom_fields(output, target_atom, structure, source_atom)
    return output


def _has_ligand_atom_context(
    structure: Structure,
    *,
    filter_by_occupancy: bool,
) -> bool:
    system = structure.system
    atom_component = torch.repeat_interleave(
        torch.arange(
            system.component_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    component_instance = torch.repeat_interleave(
        torch.arange(
            system.instance_symbol.numel(),
            dtype=torch.int64,
            device=system.atom_symbol.device,
        ),
        system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
    )
    entity_type = system.entity_type[system.instance_entity[component_instance[atom_component]]]
    observed = structure.coordinate_mask & torch.isfinite(structure.coordinates).all(-1)
    if filter_by_occupancy:
        observed &= ~structure.occupancy_mask | (structure.occupancy > 0)
    ligand = (
        observed
        & (entity_type != ENTITY_PROTEIN)
        & ~water_component_mask(system)[atom_component]
        & (system.atomic_number > 1)
    )
    return bool(ligand.any())


def _mpnn_log_probabilities(
    candidate: DesignCandidate,
    designer: Literal["proteinmpnn", "ligandmpnn"],
    *,
    design_scope: frozenset[ComponentReference],
) -> tuple[float, float, frozenset[ComponentReference]]:
    if designer == "proteinmpnn":
        score = candidate.scores["designed_sequence_nll"]
        mean = -score.value
        scope = cast(frozenset[ComponentReference], score.scope)
        return mean * len(scope), mean, scope
    # LigandMPNN reports an interface score only when the target actually has a ligand.
    score = candidate.scores.get("ligand_interface_confidence") or candidate.scores["designed_sequence_confidence"]
    if not score.scope:
        score = candidate.scores["designed_sequence_confidence"]
    scope = design_scope if score.scope is None else cast(frozenset[ComponentReference], score.scope)
    mean = float(np.log(score.value))
    return mean * len(scope), mean, scope


def _polymer_candidate(
    backbone: Structure,
    selections: tuple[tuple[tuple[ComponentReference, ...], Structure], ...],
    instance_ids: tuple[str, ...],
    totals: tuple[float, ...],
    means: tuple[float, ...],
    scopes: tuple[frozenset[ComponentReference], ...],
) -> ODesignPolymerCandidate:
    total = sum(totals)
    scope = frozenset(reference for chain_scope in scopes for reference in chain_scope)
    count = sum(len(chain_scope) for chain_scope in scopes)
    return ODesignPolymerCandidate(
        structure=_combine_polymer_structures(
            backbone,
            selections,
        ),
        scores={
            "total_log_probability": Score(
                total,
                unit="nat",
                preferred="higher",
                scope=scope,
            ),
            "mean_log_probability": Score(
                total / count,
                unit="nat/residue",
                preferred="higher",
                scope=scope,
            ),
        },
        metrics={
            "chain_total_log_probability": Metric(
                values=np.asarray(totals),
                axis=instance_ids,
                unit="nat",
            ),
            "chain_mean_log_probability": Metric(
                values=np.asarray(means),
                axis=instance_ids,
                unit="nat/residue",
            ),
        },
    )


def _structure_with_system(
    structure: Structure,
    system: MolecularSystem,
) -> Structure:
    output = _empty_structure(system, structure)
    for source_atom in range(structure.system.atom_symbol.numel()):
        try:
            target_atom = system.atom_index(structure.system.atom_site(source_atom))
        except KeyError:
            continue
        _copy_atom_fields(output, target_atom, structure, source_atom)
    return output


def _rna_structure_with_system(
    structure: Structure,
    system: MolecularSystem,
    positions: tuple[ComponentReference, ...],
) -> Structure:
    selected = set(positions)
    output = _empty_structure(system, structure)
    for source_atom in range(structure.system.atom_symbol.numel()):
        site = structure.system.atom_site(source_atom)
        reference = (site.instance_id, site.component_index)
        atom_id = site.atom_id
        if reference in selected and atom_id in ("N", "N1", "N9"):
            component = system.component_index(reference)
            symbol = system.symbols[int(system.component_symbol[component])]
            _, atom_id = _NUCLEIC_ACID_IDENTITIES["rna", symbol]
        try:
            target_atom = system.atom_index(AtomSite(site.instance_id, site.component_index, atom_id))
        except KeyError:
            continue
        _copy_atom_fields(output, target_atom, structure, source_atom)
    return output


def _combine_polymer_structures(
    backbone: Structure,
    selections: tuple[tuple[tuple[ComponentReference, ...], Structure], ...],
) -> Structure:
    output = backbone
    for positions, candidate in selections:
        target_component = torch.tensor(
            [output.system.component_index(reference) for reference in positions],
            dtype=torch.int64,
            device=output.system.component_symbol.device,
        )
        source_component = torch.tensor(
            [candidate.system.component_index(reference) for reference in positions],
            dtype=torch.int64,
            device=candidate.system.component_symbol.device,
        )
        system, old_to_new, copied_atom_index = copy_components(
            output.system,
            target_component,
            candidate.system,
            source_component,
        )
        combined = _remap_structure(output, system, old_to_new)
        for source_atom, target_atom in zip(
            copied_atom_index[0].detach().cpu().tolist(),
            copied_atom_index[1].detach().cpu().tolist(),
            strict=True,
        ):
            _copy_atom_fields(combined, target_atom, candidate, source_atom)
        output = combined
    return output


def _empty_structure(
    system: MolecularSystem,
    source: Structure,
) -> Structure:
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    standard_component_id = source.standard_component_id[:component_count].clone()
    for component in range(component_count):
        source_id = source.system.symbols[int(source.system.component_symbol[component])]
        result_id = system.symbols[int(system.component_symbol[component])]
        if source_id != result_id:
            standard_component_id[component] = -1
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=torch.zeros(
            atom_count,
            3,
            dtype=source.coordinates.dtype,
            device=system.atom_symbol.device,
        ),
        coordinate_mask=torch.zeros(
            atom_count,
            dtype=torch.bool,
            device=system.atom_symbol.device,
        ),
        occupancy=torch.zeros(
            atom_count,
            dtype=source.occupancy.dtype,
            device=system.atom_symbol.device,
        ),
        occupancy_mask=torch.zeros(
            atom_count,
            dtype=torch.bool,
            device=system.atom_symbol.device,
        ),
        b_factor=torch.zeros(
            atom_count,
            dtype=source.b_factor.dtype,
            device=system.atom_symbol.device,
        ),
        b_factor_mask=torch.zeros(
            atom_count,
            dtype=torch.bool,
            device=system.atom_symbol.device,
        ),
        source_symbols=source.source_symbols,
        residue_id=source.residue_id[:component_count].clone(),
        insertion_code=source.insertion_code[:component_count].clone(),
        standard_component_id=standard_component_id,
        batch_size=[],
    )


def _copy_atom_fields(
    target: Structure,
    target_atom: int,
    source: Structure,
    source_atom: int,
) -> None:
    target.coordinates[target_atom] = source.coordinates[source_atom]
    target.coordinate_mask[target_atom] = source.coordinate_mask[source_atom]
    target.occupancy[target_atom] = source.occupancy[source_atom]
    target.occupancy_mask[target_atom] = source.occupancy_mask[source_atom]
    target.b_factor[target_atom] = source.b_factor[source_atom]
    target.b_factor_mask[target_atom] = source.b_factor_mask[source_atom]


def _remap_structure(
    source: Structure,
    system: MolecularSystem,
    old_to_new: torch.Tensor,
) -> Structure:
    output = _empty_structure(system, source)
    valid = old_to_new >= 0
    for source_atom, target_atom in zip(
        valid.nonzero(as_tuple=False).flatten().tolist(),
        old_to_new[valid].detach().cpu().tolist(),
        strict=True,
    ):
        _copy_atom_fields(output, target_atom, source, source_atom)
    return output
