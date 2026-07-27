r"""
Decoded ODesign protein, nucleic-acid, and ligand design with their released inverse-folding families.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

from ...data import ATOMIC_NUMBER_BY_ELEMENT
from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    HYDROGEN_ELEMENTS,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
    is_water_component,
)
from ...data.structure import Structure
from ...io.msa import read_msa
from ...io.structure import read_structure
from ...pipelines import CompositePair, CompositePipeline, resolve_generator
from ...results import DesignCandidate, DesignResult, Score, best
from ..grnade import GrnadeDesignResult, GrnadePipeline
from ..ligandmpnn import LigandMpnnPipeline
from ..oinvfold import (
    OInvFoldLigandPipeline,
    OInvFoldLigandResult,
    OInvFoldLigandSlots,
    OInvFoldPolymerPipeline,
    OInvFoldPolymerResult,
)
from ..proteinmpnn import ProteinMpnnPipeline
from .model import (
    ODesignGeneratedAtoms,
    ODesignInput,
    ODesignLigandBackbone,
    ODesignLigandInput,
    ODesignModel,
    ODesignPlacements,
    ODesignProvisionalAtomSlot,
    ODesignReferenceCoordinates,
    realize_nucleic_acid_identity,
)
from .processor import ODesignLigandProcessor, ODesignProcessor


@dataclass(eq=False)
class ODesignPolymerChainPrediction:
    r"""
    One OInvFold chain prediction retained inside an ODesign candidate.

    Args:
        modality: Protein, DNA, or RNA chemistry of the predicted chain.
        instance_id: Canonical molecular-instance identifier.
        component_references: Full OInvFold row axis.
        design_mask: Rows whose identities were designed.
        sequence: Full decoded chain sequence.
        logits: Raw OInvFold logits aligned with ``component_references``.
        log_probabilities: Raw log probabilities aligned with the same axis.
        probabilities: Temperature-view probabilities aligned with the same axis.
        total_log_probability: Sum of raw log probabilities over the full row axis.
        mean_log_probability: Mean raw log probability over the full row axis.
    """

    modality: Literal["protein", "dna", "rna"]
    instance_id: str
    component_references: tuple[ComponentReference, ...]
    design_mask: np.ndarray
    sequence: str
    logits: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    total_log_probability: float
    mean_log_probability: float


@dataclass(eq=False)
class ODesignMpnnChainPrediction:
    r"""
    One ProteinMPNN or LigandMPNN chain prediction retained inside an ODesign candidate.

    Args:
        sequence_designer: Exact released MPNN family selected from assembly context.
        instance_id: Canonical protein-instance identifier.
        component_references: Full protein-chain residue axis.
        design_mask: Rows whose identities were designed.
        score_mask: Rows summarized by the released ODesign wrapper score.
        sequence: Full decoded chain sequence.
        total_log_probability: Released ODesign wrapper score over selected rows.
        mean_log_probability: Mean selected-row log probability.
        scores: MPNN-family scientific scores for this chain candidate.
    """

    sequence_designer: Literal["proteinmpnn", "ligandmpnn"]
    instance_id: str
    component_references: tuple[ComponentReference, ...]
    design_mask: np.ndarray
    score_mask: np.ndarray
    sequence: str
    total_log_probability: float
    mean_log_probability: float
    scores: dict[str, Score] = field(default_factory=dict)


@dataclass(eq=False)
class ODesignGrnadeChainPrediction:
    r"""
    One gRNAde chain prediction retained inside an ODesign candidate.

    Args:
        instance_id: Canonical RNA-instance identifier.
        component_references: Full RNA residue axis.
        design_mask: Rows whose identities were designed.
        sequence: Full decoded RNA sequence.
        sequence_tokens: Sampled gRNAde token IDs.
        logits: Realized autoregressive logits for this candidate.
        log_probabilities: Log-softmax values aligned with ``logits``.
        total_log_probability: Sum over designed rows.
        mean_log_probability: Mean over designed rows.
    """

    instance_id: str
    component_references: tuple[ComponentReference, ...]
    design_mask: np.ndarray
    sequence: str
    sequence_tokens: np.ndarray
    logits: np.ndarray
    log_probabilities: np.ndarray
    total_log_probability: float
    mean_log_probability: float


@dataclass(eq=False)
class ODesignPolymerDesignCandidate:
    r"""
    One polymer sequence assigned to one ODesign backbone.

    Args:
        structure: Canonical designed structure.
        chain_predictions: Per-chain inverse-folding predictions.
        scores: Assembly-level scientific scores when child predictions share an
            aggregatable score contract.
        seed_index: Zero-based ODesign seed index.
        backbone_index: Zero-based backbone index within the seed.
        sequence_index: Zero-based sequence index within the backbone.
    """

    structure: Structure
    chain_predictions: tuple[
        ODesignPolymerChainPrediction | ODesignMpnnChainPrediction | ODesignGrnadeChainPrediction,
        ...,
    ]
    scores: dict[str, Score] = field(default_factory=dict)
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignPolymerDesignResult:
    r"""
    Seed-by-backbone-by-sequence ODesign polymer candidates.
    """

    candidates: tuple[ODesignPolymerDesignCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignPolymerDesignCandidate:
        r"""
        Return the earliest best candidate under one named score.
        """

        return best(self.candidates, score)


@dataclass(eq=False)
class ODesignLigandDesignCandidate:
    r"""
    One provisional ligand atom assignment without fabricated connectivity.

    Args:
        context: Canonical context with its ODesign-predicted coordinates.
        context_token_sites: Context groups on the leading bond-logit axes.
        slots: Caller-owned provisional atom identities.
        coordinates: Coordinates aligned with ``slots``.
        elements: OInvFold element labels aligned with ``slots``.
        per_atom_probabilities: Selected temperature-view probabilities.
        logits: Raw OInvFold logits aligned with ``slots``.
        log_probabilities: Raw OInvFold log probabilities.
        probabilities: Full temperature-view probabilities.
        token_bond_type_logits: Full ODesign context-and-slot bond logits.
        scores: Total and mean raw element log probabilities.
        bonds: Explicitly unresolved generated chemical connectivity.
        seed_index: Zero-based ODesign seed index.
        backbone_index: Zero-based backbone index within the seed.
        sequence_index: Zero-based element-assignment index within the backbone.
    """

    context: Structure | None
    context_token_sites: tuple[tuple[AtomSite, ...], ...]
    slots: tuple[ODesignProvisionalAtomSlot, ...]
    coordinates: np.ndarray
    elements: tuple[str, ...]
    per_atom_probabilities: np.ndarray
    logits: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    token_bond_type_logits: np.ndarray
    scores: dict[str, Score] = field(default_factory=dict)
    bonds: None = None
    seed_index: int = 0
    backbone_index: int = 0
    sequence_index: int = 0


@dataclass(eq=False)
class ODesignLigandDesignResult:
    r"""
    Seed-by-backbone-by-assignment ODesign ligand candidates.
    """

    candidates: tuple[ODesignLigandDesignCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best(self, score: str) -> ODesignLigandDesignCandidate:
        r"""
        Return the earliest best candidate under one named score.
        """

        return best(self.candidates, score)


class ODesignProteinPipeline(CompositePipeline):
    r"""
    Official ODesign protein-backbone generation followed by the released MPNN family.

    ProteinMPNN is selected for a protein-only realized assembly. LigandMPNN is
    selected when that realized assembly contains observed nonprotein, nonwater
    heavy-atom context.

    Args:
        model: ODesign backbone model.
        processor: Protein-artifact ODesign Processor.
        protein_designer: Released ProteinMPNN Pipeline.
        ligand_designer: Released LigandMPNN Pipeline.
    """

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "proteinmpnn",
            "proteinmpnn",
            "protein_designer",
            None,
            wrap=ProteinMpnnPipeline,
        ),
        CompositePair(
            "ligandmpnn",
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
        if self.processor.config.generated_modality != "protein":
            raise ValueError(
                "ODesignProteinPipeline requires a protein ODesign Processor; "
                f"got {self.processor.config.generated_modality!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerDesignResult:
        r"""
        Generate protein backbones and sample sequences with the context-selected MPNN.
        """

        seed_list = _resolve_seeds(
            num_seeds,
            num_backbones,
            num_sequences,
            None,
            seeds,
            seed,
            generator,
        )
        model_input = _build_polymer_input(
            self.processor,
            input,
            placement,
            msa,
            conditions,
        )

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=model_input.reference_positions.device,
            )
            realized_input = self.processor.realize_ranges(
                model_input,
                generator=stream,
            )
            design_chains = tuple(chain for chain in _generated_polymer_chains(realized_input) if chain[1] == "protein")
            if not design_chains:
                raise ValueError("ODesign protein design requires at least one generated protein position")
            backbones = self.model.design(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )

            for backbone_index, backbone in enumerate(backbones):
                target = _protein_backbone_for_inverse_folding(
                    backbone,
                    tuple(reference for _, _, positions in design_chains for reference in positions),
                )
                designer_name: Literal["proteinmpnn", "ligandmpnn"] = (
                    "ligandmpnn"
                    if _has_ligand_atom_context(
                        target,
                        filter_by_occupancy=filter_by_occupancy,
                    )
                    else "proteinmpnn"
                )
                chain_results: list[
                    tuple[
                        tuple[ComponentReference, ...],
                        tuple[Structure, ...],
                        tuple[ODesignMpnnChainPrediction, ...],
                    ]
                ] = []
                for instance_id, _, positions in design_chains:
                    component_references = _instance_component_references(
                        target.system,
                        instance_id,
                    )
                    selected = set(positions)
                    chain_design: DesignResult
                    if designer_name == "ligandmpnn":
                        chain_design = self.ligand_designer.design(
                            target,
                            design_positions=positions,
                            num_candidates=num_sequences,
                            temperature=temperature,
                            filter_by_occupancy=filter_by_occupancy,
                            generator=stream,
                        )
                    else:
                        chain_design = self.protein_designer.design(
                            target,
                            design_positions=positions,
                            num_candidates=num_sequences,
                            temperature=temperature,
                            generator=stream,
                        )
                    ranked_candidates: list[
                        tuple[
                            DesignCandidate,
                            float,
                            float,
                            frozenset[ComponentReference],
                        ]
                    ] = []
                    for candidate in chain_design.candidates:
                        total, mean, score_references = _mpnn_log_probabilities(
                            candidate,
                            designer_name,
                            design_references=frozenset(selected),
                        )
                        ranked_candidates.append(
                            (
                                candidate,
                                total,
                                mean,
                                score_references,
                            )
                        )
                    ranked_candidates.sort(key=lambda value: value[1], reverse=True)
                    chain_results.append(
                        (
                            positions,
                            tuple(
                                _structure_with_system(target, candidate.system)
                                for candidate, _, _, _ in ranked_candidates
                            ),
                            tuple(
                                ODesignMpnnChainPrediction(
                                    sequence_designer=designer_name,
                                    instance_id=instance_id,
                                    component_references=component_references,
                                    design_mask=np.asarray(
                                        [reference in selected for reference in component_references],
                                        dtype=np.bool_,
                                    ),
                                    score_mask=np.asarray(
                                        [reference in score_references for reference in component_references],
                                        dtype=np.bool_,
                                    ),
                                    sequence=_protein_sequence(
                                        candidate.system,
                                        component_references,
                                    ),
                                    scores=candidate.scores,
                                    total_log_probability=total_log_probability,
                                    mean_log_probability=mean_log_probability,
                                )
                                for (
                                    candidate,
                                    total_log_probability,
                                    mean_log_probability,
                                    score_references,
                                ) in ranked_candidates
                            ),
                        )
                    )

                for sequence_index in range(num_sequences):
                    predictions = tuple(chain_predictions[sequence_index] for _, _, chain_predictions in chain_results)
                    candidates.append(
                        ODesignPolymerDesignCandidate(
                            structure=_combine_polymer_structures(
                                backbone,
                                tuple(
                                    (positions, structures[sequence_index])
                                    for positions, structures, _ in chain_results
                                ),
                            ),
                            chain_predictions=predictions,
                            scores=_polymer_scores(predictions),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )

        return ODesignPolymerDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignOInvFoldProteinPipeline(CompositePipeline):
    r"""
    Explicit OInvFold alternative for ODesign protein backbones.

    Args:
        model: ODesign backbone model.
        processor: Protein-artifact ODesign Processor.
        sequence_designer: Protein OInvFold Pipeline.
    """

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "oinvfold",
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
        if self.processor.config.generated_modality != "protein":
            raise ValueError(
                "ODesignOInvFoldProteinPipeline requires a protein ODesign Processor; "
                f"got {self.processor.config.generated_modality!r}"
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
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = False,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerDesignResult:
        r"""
        Generate protein backbones and assign their generated residues with OInvFold.

        Greedy OInvFold decoding has one candidate. Set ``use_beam=True`` when
        ``num_sequences`` is greater than one.
        """

        seed_list = _resolve_seeds(
            num_seeds,
            num_backbones,
            num_sequences,
            use_beam,
            seeds,
            seed,
            generator,
        )
        model_input = _build_polymer_input(
            self.processor,
            input,
            placement,
            msa,
            conditions,
        )

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=model_input.reference_positions.device,
            )
            realized_input = self.processor.realize_ranges(
                model_input,
                generator=stream,
            )
            design_chains = tuple(chain for chain in _generated_polymer_chains(realized_input) if chain[1] == "protein")
            if not design_chains:
                raise ValueError("ODesign protein design requires at least one generated protein position")
            backbones = self.model.design(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )

            for backbone_index, backbone in enumerate(backbones):
                chain_results: list[
                    tuple[
                        str,
                        Literal["protein", "dna", "rna"],
                        tuple[ComponentReference, ...],
                        OInvFoldPolymerResult,
                    ]
                ] = []
                for instance_id, modality, positions in design_chains:
                    target = _protein_backbone_for_inverse_folding(backbone, positions)
                    result = self.sequence_designer.design(
                        target,
                        instance_id=instance_id,
                        design_positions=positions,
                        topk=num_sequences,
                        use_beam=use_beam,
                        temperature=temperature,
                        filter_by_occupancy=filter_by_occupancy,
                    )
                    chain_results.append((instance_id, modality, positions, result))

                for sequence_index in range(num_sequences):
                    structures = tuple(
                        (
                            positions,
                            result.candidates[sequence_index].structure,
                        )
                        for _, _, positions, result in chain_results
                    )
                    predictions = tuple(
                        _polymer_chain_prediction(
                            instance_id,
                            modality,
                            result,
                            sequence_index,
                        )
                        for instance_id, modality, _, result in chain_results
                    )
                    candidates.append(
                        ODesignPolymerDesignCandidate(
                            structure=_combine_polymer_structures(
                                backbone,
                                structures,
                            ),
                            chain_predictions=predictions,
                            scores=_polymer_scores(predictions),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )

        return ODesignPolymerDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignNucleicAcidPipeline(CompositePipeline):
    r"""
    Official ODesign nucleic-acid generation followed by DNA OInvFold or RNA gRNAde.

    Args:
        model: ODesign backbone model.
        processor: Nucleic-acid-artifact ODesign Processor.
        dna_designer: DNA OInvFold Pipeline.
        rna_designer: Released gRNAde Pipeline.
    """

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "oinvfold_dna",
            "oinvfold",
            "dna_designer",
            None,
            wrap=OInvFoldPolymerPipeline,
        ),
        CompositePair(
            "grnade",
            "grnade",
            "rna_designer",
            None,
            wrap=GrnadePipeline,
        ),
    )

    def __init__(
        self,
        *,
        model: ODesignModel,
        processor: ODesignProcessor,
        dna_designer: OInvFoldPolymerPipeline,
        rna_designer: GrnadePipeline,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dna_designer = dna_designer
        self.rna_designer = rna_designer
        self._validate_components()

    def _validate_components(self) -> None:
        super()._validate_components()
        if self.processor.config.generated_modality != "nucleic_acid":
            raise ValueError(
                "ODesignNucleicAcidPipeline requires a nucleic-acid ODesign Processor; "
                f"got {self.processor.config.generated_modality!r}"
            )
        if self.dna_designer.processor.config.modality != "dna":
            raise ValueError(
                "ODesignNucleicAcidPipeline requires a DNA OInvFold child; "
                f"got {self.dna_designer.processor.config.modality!r}"
            )

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None,
        *,
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = False,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerDesignResult:
        r"""
        Generate DNA/RNA backbones and realize canonical nucleotide identities.

        ODesign ``DN``/``N`` backbone placeholders are adapted to temporary
        adenine components only at the inverse-folding boundary. DNA uses the
        released OInvFold checkpoint; RNA uses gRNAde stochastic sampling.
        ``use_beam`` applies only when a generated DNA chain is present.
        """

        seed_list = _resolve_seeds(
            num_seeds,
            num_backbones,
            num_sequences,
            None,
            seeds,
            seed,
            generator,
        )
        model_input = _build_polymer_input(
            self.processor,
            input,
            placement,
            msa,
            conditions,
        )

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=model_input.reference_positions.device,
            )
            realized_input = self.processor.realize_ranges(
                model_input,
                generator=stream,
            )
            design_chains = tuple(
                chain for chain in _generated_polymer_chains(realized_input) if chain[1] in ("dna", "rna")
            )
            if not design_chains:
                raise ValueError("ODesign nucleic-acid design requires at least one generated DNA or RNA position")
            if any(modality == "dna" for _, modality, _ in design_chains) and not use_beam and num_sequences != 1:
                raise ValueError(
                    "greedy DNA OInvFold decoding returns one sequence; "
                    "set use_beam=True for num_sequences greater than one"
                )
            backbones = self.model.design(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )

            for backbone_index, backbone in enumerate(backbones):
                chain_results: list[
                    tuple[
                        tuple[ComponentReference, ...],
                        tuple[Structure, ...],
                        tuple[ODesignPolymerChainPrediction | ODesignGrnadeChainPrediction, ...],
                    ]
                ] = []
                for instance_id, modality, positions in design_chains:
                    nucleic_modality = cast(
                        Literal["dna", "rna"],
                        modality,
                    )
                    target = _nucleic_acid_backbone_for_inverse_folding(
                        backbone,
                        positions,
                        nucleic_modality,
                    )
                    if nucleic_modality == "dna":
                        dna_result = self.dna_designer.design(
                            target,
                            instance_id=instance_id,
                            design_positions=positions,
                            topk=num_sequences,
                            use_beam=use_beam,
                            temperature=temperature,
                            filter_by_occupancy=filter_by_occupancy,
                        )
                        chain_results.append(
                            (
                                positions,
                                tuple(candidate.structure for candidate in dna_result.candidates),
                                tuple(
                                    _polymer_chain_prediction(
                                        instance_id,
                                        modality,
                                        dna_result,
                                        sequence_index,
                                    )
                                    for sequence_index in range(num_sequences)
                                ),
                            )
                        )
                    else:
                        rna_result = self.rna_designer.design(
                            target,
                            instance_id=instance_id,
                            design_positions=positions,
                            num_candidates=num_sequences,
                            temperature=temperature,
                            filter_by_occupancy=filter_by_occupancy,
                            generator=stream,
                        )
                        chain_results.append(
                            (
                                positions,
                                tuple(
                                    _rna_structure_with_system(
                                        target,
                                        candidate.system,
                                        positions,
                                    )
                                    for candidate in rna_result.candidates
                                ),
                                tuple(
                                    _grnade_chain_prediction(
                                        instance_id,
                                        rna_result,
                                        sequence_index,
                                    )
                                    for sequence_index in range(num_sequences)
                                ),
                            )
                        )

                for sequence_index in range(num_sequences):
                    structures = tuple(
                        (
                            positions,
                            decoded_structures[sequence_index],
                        )
                        for positions, decoded_structures, _ in chain_results
                    )
                    predictions = tuple(chain_predictions[sequence_index] for _, _, chain_predictions in chain_results)
                    candidates.append(
                        ODesignPolymerDesignCandidate(
                            structure=_combine_polymer_structures(
                                backbone,
                                structures,
                            ),
                            chain_predictions=predictions,
                            scores=_polymer_scores(predictions),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )

        return ODesignPolymerDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignOInvFoldRnaPipeline(CompositePipeline):
    r"""
    Explicit OInvFold alternative for ODesign RNA backbones.

    Args:
        model: ODesign backbone model.
        processor: Nucleic-acid-artifact ODesign Processor.
        sequence_designer: RNA OInvFold Pipeline.
    """

    pair_layout = (
        CompositePair("", "odesign", "model", "processor"),
        CompositePair(
            "oinvfold",
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
        if self.processor.config.generated_modality != "nucleic_acid":
            raise ValueError(
                "ODesignOInvFoldRnaPipeline requires a nucleic-acid ODesign Processor; "
                f"got {self.processor.config.generated_modality!r}"
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
        placement: ODesignPlacements | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = False,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignPolymerDesignResult:
        r"""
        Generate RNA backbones and decode them with the optional OInvFold route.
        """

        seed_list = _resolve_seeds(
            num_seeds,
            num_backbones,
            num_sequences,
            use_beam,
            seeds,
            seed,
            generator,
        )
        model_input = _build_polymer_input(
            self.processor,
            input,
            placement,
            msa,
            conditions,
        )

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=model_input.reference_positions.device,
            )
            realized_input = self.processor.realize_ranges(
                model_input,
                generator=stream,
            )
            generated_chains = _generated_polymer_chains(realized_input)
            design_chains = tuple(chain for chain in generated_chains if chain[1] == "rna")
            if not design_chains:
                raise ValueError("ODesign OInvFold RNA design requires at least one generated RNA position")
            if len(design_chains) != len(generated_chains):
                raise ValueError("ODesignOInvFoldRnaPipeline accepts only generated RNA chains")
            backbones = self.model.design(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )

            for backbone_index, backbone in enumerate(backbones):
                chain_results: list[
                    tuple[
                        str,
                        Literal["protein", "dna", "rna"],
                        tuple[ComponentReference, ...],
                        OInvFoldPolymerResult,
                    ]
                ] = []
                for instance_id, modality, positions in design_chains:
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
                    chain_results.append((instance_id, modality, positions, result))

                for sequence_index in range(num_sequences):
                    predictions = tuple(
                        _polymer_chain_prediction(
                            instance_id,
                            modality,
                            result,
                            sequence_index,
                        )
                        for instance_id, modality, _, result in chain_results
                    )
                    candidates.append(
                        ODesignPolymerDesignCandidate(
                            structure=_combine_polymer_structures(
                                backbone,
                                tuple(
                                    (
                                        positions,
                                        result.candidates[sequence_index].structure,
                                    )
                                    for _, _, positions, result in chain_results
                                ),
                            ),
                            chain_predictions=predictions,
                            scores=_polymer_scores(predictions),
                            seed_index=seed_index,
                            backbone_index=backbone_index,
                            sequence_index=sequence_index,
                        )
                    )

        return ODesignPolymerDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


class ODesignLigandPipeline(CompositePipeline):
    r"""
    ODesign provisional ligand generation followed by ligand OInvFold.

    Args:
        model: ODesign backbone model.
        processor: Ligand-artifact ODesign Processor.
        element_designer: Ligand OInvFold Pipeline.
    """

    pair_layout = (
        CompositePair("", "odesign_ligand", "model", "processor"),
        CompositePair(
            "oinvfold",
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
        generated_atoms: ODesignGeneratedAtoms,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        temperature: float = 1.0,
        use_beam: bool = False,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        conditions: Sequence[Hotspot | ODesignReferenceCoordinates] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> ODesignLigandDesignResult:
        r"""
        Generate provisional ligand slots and assign their element categories.

        The result retains ODesign's full bond logits, but generated connectivity
        remains unresolved.
        """

        seed_list = _resolve_seeds(
            num_seeds,
            num_backbones,
            num_sequences,
            use_beam,
            seeds,
            seed,
            generator,
        )
        model_input = _build_ligand_input(
            self.processor,
            input,
            generated_atoms,
            msa,
            conditions,
        )

        candidates: list[ODesignLigandDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=model_input.reference_positions.device,
            )
            realized_input = self.processor.realize_atoms(
                model_input,
                generator=stream,
            )
            backbones = self.model.design_ligand(
                realized_input,
                num_candidates=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                partial_diffusion_snr=partial_diffusion_snr,
            )

            for backbone_index, backbone in enumerate(backbones):
                slot_ids = tuple(str(index) for index in range(len(backbone.slots)))
                slot_by_id = dict(zip(slot_ids, backbone.slots, strict=True))
                result = self.element_designer.design(
                    OInvFoldLigandSlots(
                        slot_ids=slot_ids,
                        coordinates=backbone.coordinates,
                        context=backbone.context,
                    ),
                    topk=num_sequences,
                    use_beam=use_beam,
                    temperature=temperature,
                )
                candidates.extend(
                    _ligand_candidate(
                        backbone,
                        result,
                        slot_by_id,
                        sequence_index,
                        seed_index,
                        backbone_index,
                    )
                    for sequence_index in range(num_sequences)
                )

        return ODesignLigandDesignResult(
            candidates=tuple(candidates),
            num_seeds=len(seed_list),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
        )


def _resolve_seeds(
    num_seeds: int,
    num_backbones: int,
    num_sequences: int,
    use_beam: bool | None,
    seeds: Sequence[int] | None,
    seed: int | None,
    generator: torch.Generator | None,
) -> tuple[int, ...]:
    if num_seeds <= 0 or num_backbones <= 0 or num_sequences <= 0:
        raise ValueError("num_seeds, num_backbones, and num_sequences must be positive")
    if use_beam is False and num_sequences != 1:
        raise ValueError(
            "greedy OInvFold decoding returns one sequence; " "set use_beam=True for num_sequences greater than one"
        )
    if seeds is not None and (seed is not None or generator is not None):
        raise ValueError("seeds cannot be combined with seed or generator")
    if seeds is not None:
        values = tuple(int(value) for value in seeds)
        if not values:
            raise ValueError("seeds must not be empty")
        return values

    generator = resolve_generator(seed, generator)
    device = None if generator is None else generator.device
    drawn = torch.randint(
        1,
        torch.iinfo(torch.int32).max,
        (num_seeds,),
        generator=generator,
        device=device,
    )
    return tuple(int(value) for value in drawn.tolist())


def _build_polymer_input(
    processor: ODesignProcessor,
    input: MolecularSystem | Structure | str | PathLike[str] | None,
    placement: ODesignPlacements | None,
    msa: MSA | MSACollection | str | PathLike[str] | None,
    conditions: Sequence[Hotspot | SymbolicBond | ODesignReferenceCoordinates],
) -> ODesignInput:
    canonical_input = _resolve_structure(input)
    plan = processor._plan(canonical_input, placement, conditions)
    return processor._lower(plan, _resolve_msa(msa, plan.system))


def _build_ligand_input(
    processor: ODesignLigandProcessor,
    input: MolecularSystem | Structure | str | PathLike[str] | None,
    generated_atoms: ODesignGeneratedAtoms,
    msa: MSA | MSACollection | str | PathLike[str] | None,
    conditions: Sequence[Hotspot | ODesignReferenceCoordinates],
) -> ODesignLigandInput:
    canonical_input = _resolve_structure(input)
    plan = processor._plan(canonical_input, generated_atoms, conditions)
    return processor._lower(
        plan,
        _resolve_msa(msa, plan.context_system),
    )


def _resolve_structure(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    if isinstance(input, (str, PathLike)):
        return read_structure(Path(input).expanduser())
    return input


def _resolve_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem | None,
) -> MSA | MSACollection | None:
    if not isinstance(msa, (str, PathLike)):
        return msa
    if system is None:
        raise ValueError("an MSA path requires canonical polymer context")
    polymer_entity_ids = tuple(
        entity_id for entity_id, entity in system.entities.items() if isinstance(entity, PolymerSequence)
    )
    if len(polymer_entity_ids) != 1:
        raise ValueError("an MSA path is unambiguous only for a single-polymer system; " "pass MSACollection")
    return read_msa(
        Path(msa).expanduser(),
        entity_id=polymer_entity_ids[0],
    )


def _generated_polymer_chains(
    input: ODesignInput,
) -> tuple[
    tuple[
        str,
        Literal["protein", "dna", "rna"],
        tuple[ComponentReference, ...],
    ],
    ...,
]:
    generated = {
        reference
        for placements in input.layout.placement_map.values()
        for occurrences in placements.values()
        for reference in occurrences
    }
    has_placement = input.layout.placement is not None
    chains = []
    for instance_id, entity_id in input.layout.system.instances.items():
        entity = input.layout.system.entities[entity_id]
        if not isinstance(entity, PolymerSequence):
            continue
        if entity.polymer_type not in ("protein", "dna", "rna"):
            continue
        positions = tuple(
            (instance_id, component_index)
            for component_index in range(len(entity))
            if not has_placement or (instance_id, component_index) in generated
        )
        if positions:
            chains.append(
                (
                    instance_id,
                    cast(
                        Literal["protein", "dna", "rna"],
                        entity.polymer_type,
                    ),
                    positions,
                )
            )
    return tuple(chains)


def _protein_backbone_for_inverse_folding(
    structure: Structure,
    positions: tuple[ComponentReference, ...],
) -> Structure:
    replacements = {reference: get_component("GLY") for reference in positions}
    system = structure.system.replace_components(replacements)
    rows = [index for index, site in enumerate(structure.sites) if _contains_site(system, site)]
    indices = np.asarray(rows, dtype=np.int64)
    return Structure(
        system=system,
        sites=tuple(structure.sites[index] for index in rows),
        coordinates=structure.coordinates[indices],
        coordinate_mask=structure.coordinate_mask[indices],
        occupancy=(None if structure.occupancy is None else structure.occupancy[indices]),
        b_factor=(None if structure.b_factor is None else structure.b_factor[indices]),
        residue_numbers=structure.residue_numbers,
    )


def _nucleic_acid_backbone_for_inverse_folding(
    structure: Structure,
    positions: tuple[ComponentReference, ...],
    modality: Literal["dna", "rna"],
) -> Structure:
    identity = realize_nucleic_acid_identity(modality, "A")
    replacements = {reference: get_component(identity.component_id) for reference in positions}
    system = structure.system.replace_components(replacements)
    selected = set(positions)
    sites = tuple(
        (
            AtomSite(
                site.instance_id,
                site.component_index,
                identity.atom_id(site.atom_id),
            )
            if (site.instance_id, site.component_index) in selected
            else site
        )
        for site in structure.sites
    )
    rows = [index for index, site in enumerate(sites) if _contains_site(system, site)]
    indices = np.asarray(rows, dtype=np.int64)
    return Structure(
        system=system,
        sites=tuple(sites[index] for index in rows),
        coordinates=structure.coordinates[indices],
        coordinate_mask=structure.coordinate_mask[indices],
        occupancy=(None if structure.occupancy is None else structure.occupancy[indices]),
        b_factor=(None if structure.b_factor is None else structure.b_factor[indices]),
        residue_numbers=structure.residue_numbers,
    )


def _contains_site(system: MolecularSystem, site: AtomSite) -> bool:
    try:
        system.atom(site)
    except KeyError:
        return False
    return True


def _has_ligand_atom_context(
    structure: Structure,
    *,
    filter_by_occupancy: bool,
) -> bool:
    observed = structure.coordinate_mask & np.isfinite(structure.coordinates).all(axis=-1)
    if filter_by_occupancy and structure.occupancy is not None:
        observed = observed & (np.isnan(structure.occupancy) | (structure.occupancy > 0))
    for index, site in enumerate(structure.sites):
        if not bool(observed[index]):
            continue
        entity = structure.system.entities[structure.system.instances[site.instance_id]]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            continue
        component = entity.components[site.component_index]
        if is_water_component(component):
            continue
        element = structure.system.atom(site).element
        if element not in HYDROGEN_ELEMENTS and element in ATOMIC_NUMBER_BY_ELEMENT:
            return True
    return False


def _structure_with_system(
    structure: Structure,
    system: MolecularSystem,
) -> Structure:
    rows = [index for index, site in enumerate(structure.sites) if _contains_site(system, site)]
    indices = np.asarray(rows, dtype=np.int64)
    return Structure(
        system=system,
        sites=tuple(structure.sites[index] for index in rows),
        coordinates=structure.coordinates[indices],
        coordinate_mask=structure.coordinate_mask[indices],
        occupancy=None if structure.occupancy is None else structure.occupancy[indices],
        b_factor=None if structure.b_factor is None else structure.b_factor[indices],
        residue_numbers=structure.residue_numbers,
    )


def _protein_sequence(
    system: MolecularSystem,
    positions: tuple[ComponentReference, ...],
) -> str:
    return "".join(
        AMINO_ACID_THREE_TO_ONE.get(system.component(reference).component_id, "X") for reference in positions
    )


def _instance_component_references(
    system: MolecularSystem,
    instance_id: str,
) -> tuple[ComponentReference, ...]:
    entity = system.entities[system.instances[instance_id]]
    return tuple((instance_id, component_index) for component_index in range(len(entity.components)))


def _rna_structure_with_system(
    structure: Structure,
    system: MolecularSystem,
    positions: tuple[ComponentReference, ...],
) -> Structure:
    selected = set(positions)
    sites = []
    for site in structure.sites:
        reference = (site.instance_id, site.component_index)
        if reference not in selected:
            sites.append(site)
            continue
        symbol = system.component(reference).component_id
        identity = realize_nucleic_acid_identity("rna", symbol)
        atom_id = identity.base_anchor_atom_id if site.atom_id in ("N", "N1", "N9") else site.atom_id
        sites.append(AtomSite(site.instance_id, site.component_index, atom_id))
    rows = [index for index, site in enumerate(sites) if _contains_site(system, site)]
    indices = np.asarray(rows, dtype=np.int64)
    return Structure(
        system=system,
        sites=tuple(sites[index] for index in rows),
        coordinates=structure.coordinates[indices],
        coordinate_mask=structure.coordinate_mask[indices],
        occupancy=None if structure.occupancy is None else structure.occupancy[indices],
        b_factor=None if structure.b_factor is None else structure.b_factor[indices],
        residue_numbers=structure.residue_numbers,
    )


def _polymer_chain_prediction(
    instance_id: str,
    modality: Literal["protein", "dna", "rna"],
    result: OInvFoldPolymerResult,
    sequence_index: int,
) -> ODesignPolymerChainPrediction:
    candidate = result.candidates[sequence_index]
    return ODesignPolymerChainPrediction(
        modality=modality,
        instance_id=instance_id,
        component_references=result.component_references,
        design_mask=result.design_mask,
        sequence=candidate.sequence,
        logits=result.logits,
        log_probabilities=result.log_probabilities,
        probabilities=result.probabilities,
        total_log_probability=candidate.total_log_probability,
        mean_log_probability=candidate.mean_log_probability,
    )


def _mpnn_log_probabilities(
    candidate: DesignCandidate,
    sequence_designer: Literal["proteinmpnn", "ligandmpnn"],
    *,
    design_references: frozenset[ComponentReference],
) -> tuple[float, float, frozenset[ComponentReference]]:
    if sequence_designer == "proteinmpnn":
        mean_log_probability = -candidate.scores["designed_sequence_nll"].value
        return (
            mean_log_probability * len(design_references),
            mean_log_probability,
            design_references,
        )

    score = candidate.scores["ligand_interface_confidence"]
    if not score.scope:
        score = candidate.scores["designed_sequence_confidence"]
    score_references = design_references if score.scope is None else cast(frozenset[ComponentReference], score.scope)
    mean_log_probability = float(np.log(score.value))
    return (
        mean_log_probability * len(score_references),
        mean_log_probability,
        score_references,
    )


def _grnade_chain_prediction(
    instance_id: str,
    result: GrnadeDesignResult,
    sequence_index: int,
) -> ODesignGrnadeChainPrediction:
    candidate = result.candidates[sequence_index]
    return ODesignGrnadeChainPrediction(
        instance_id=instance_id,
        component_references=result.component_references,
        design_mask=result.design_mask,
        sequence="".join(
            candidate.system.component(reference).component_id for reference in result.component_references
        ),
        sequence_tokens=result.sequence_tokens[sequence_index],
        logits=result.logits[sequence_index],
        log_probabilities=result.log_probabilities[sequence_index],
        total_log_probability=candidate.scores["total_log_probability"].value,
        mean_log_probability=candidate.scores["mean_log_probability"].value,
    )


def _polymer_scores(
    predictions: tuple[
        ODesignPolymerChainPrediction | ODesignMpnnChainPrediction | ODesignGrnadeChainPrediction,
        ...,
    ],
) -> dict[str, Score]:
    total = sum(prediction.total_log_probability for prediction in predictions)
    masks: list[np.ndarray] = []
    for prediction in predictions:
        if isinstance(prediction, ODesignPolymerChainPrediction):
            masks.append(np.ones_like(prediction.design_mask))
        elif isinstance(prediction, ODesignMpnnChainPrediction):
            masks.append(prediction.score_mask)
        else:
            masks.append(prediction.design_mask)
    count = sum(int(np.count_nonzero(mask)) for mask in masks)
    scope = frozenset(
        reference
        for prediction, mask in zip(predictions, masks, strict=True)
        for reference, selected in zip(
            prediction.component_references,
            mask,
            strict=True,
        )
        if selected
    )
    return {
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
    }


def _combine_polymer_structures(
    backbone: Structure,
    selections: tuple[
        tuple[tuple[ComponentReference, ...], Structure],
        ...,
    ],
) -> Structure:
    selected = {reference for positions, _ in selections for reference in positions}
    replacements = {
        reference: candidate.system.component(reference)
        for positions, candidate in selections
        for reference in positions
    }
    system = backbone.system.replace_components(replacements)

    sites: list[AtomSite] = []
    coordinates: list[np.ndarray] = []
    coordinate_mask: list[np.bool_] = []
    occupancy: list[np.generic] | None = None if backbone.occupancy is None else []
    b_factor: list[np.generic] | None = None if backbone.b_factor is None else []

    for index, site in enumerate(backbone.sites):
        if (site.instance_id, site.component_index) in selected:
            continue
        sites.append(site)
        coordinates.append(backbone.coordinates[index])
        coordinate_mask.append(backbone.coordinate_mask[index])
        if occupancy is not None and backbone.occupancy is not None:
            occupancy.append(backbone.occupancy[index])
        if b_factor is not None and backbone.b_factor is not None:
            b_factor.append(backbone.b_factor[index])

    for positions, candidate in selections:
        selected_positions = set(positions)
        for index, site in enumerate(candidate.sites):
            if (site.instance_id, site.component_index) not in selected_positions:
                continue
            sites.append(site)
            coordinates.append(candidate.coordinates[index])
            coordinate_mask.append(candidate.coordinate_mask[index])
            if occupancy is not None:
                occupancy.append(np.nan if candidate.occupancy is None else candidate.occupancy[index])
            if b_factor is not None:
                b_factor.append(np.nan if candidate.b_factor is None else candidate.b_factor[index])

    return Structure(
        system=system,
        sites=tuple(sites),
        coordinates=np.asarray(
            coordinates,
            dtype=backbone.coordinates.dtype,
        ),
        coordinate_mask=np.asarray(coordinate_mask, dtype=np.bool_),
        occupancy=(
            None
            if occupancy is None
            else np.asarray(
                occupancy,
                dtype=cast(np.ndarray, backbone.occupancy).dtype,
            )
        ),
        b_factor=(
            None
            if b_factor is None
            else np.asarray(
                b_factor,
                dtype=cast(np.ndarray, backbone.b_factor).dtype,
            )
        ),
        residue_numbers=backbone.residue_numbers,
    )


def _ligand_candidate(
    backbone: ODesignLigandBackbone,
    result: OInvFoldLigandResult,
    slot_by_id: dict[str, ODesignProvisionalAtomSlot],
    sequence_index: int,
    seed_index: int,
    backbone_index: int,
) -> ODesignLigandDesignCandidate:
    candidate = result.candidates[sequence_index]
    return ODesignLigandDesignCandidate(
        context=candidate.context,
        context_token_sites=backbone.context_token_sites,
        slots=tuple(slot_by_id[slot_id] for slot_id in candidate.slot_ids),
        coordinates=candidate.coordinates,
        elements=candidate.elements,
        per_atom_probabilities=candidate.per_atom_probabilities,
        logits=result.logits,
        log_probabilities=result.log_probabilities,
        probabilities=result.probabilities,
        token_bond_type_logits=backbone.token_bond_type_logits,
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
        seed_index=seed_index,
        backbone_index=backbone_index,
        sequence_index=sequence_index,
    )
