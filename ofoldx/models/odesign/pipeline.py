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

from ...data.bonds import SymbolicBond
from ...data.components import get_component
from ...data.conditions import Hotspot
from ...data.evidence import MSA, MSACollection
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_DNA,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    AtomSite,
    ComponentReference,
    MolecularSystem,
    copy_components,
    replace_components,
    water_component_mask,
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
    ODesignLigandInput,
    ODesignModel,
    ODesignPlacements,
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
class ODesignLigandBackbone:
    r"""
    Decoded ODesign ligand backbone before identities and bonds are resolved.

    Args:
        context: Predicted coordinates over the canonical context system.
        coordinates: Ordered provisional coordinates with shape ``(L, 3)``.
        token_bond_type_logits: Generated-slot pair logits with shape
            ``(L, L, D_b)``.
        bonds: Unresolved generated chemical connectivity.
    """

    context: Structure | None
    coordinates: np.ndarray
    token_bond_type_logits: np.ndarray
    bonds: None = None


@dataclass(eq=False)
class ODesignLigandDesignCandidate:
    r"""
    One provisional ligand atom assignment without fabricated connectivity.

    Args:
        context: Canonical context with its ODesign-predicted coordinates.
        coordinates: Ordered provisional coordinates.
        elements: OInvFold element labels aligned with the same rows.
        per_atom_probabilities: Selected temperature-view probabilities.
        logits: Raw OInvFold logits aligned with ``coordinates``.
        log_probabilities: Raw OInvFold log probabilities.
        probabilities: Full temperature-view probabilities.
        token_bond_type_logits: Generated-slot pair bond logits.
        scores: Total and mean raw element log probabilities.
        bonds: Explicitly unresolved generated chemical connectivity.
        seed_index: Zero-based ODesign seed index.
        backbone_index: Zero-based backbone index within the seed.
        sequence_index: Zero-based element-assignment index within the backbone.
    """

    context: Structure | None
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
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
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
        canonical_input = _resolve_structure(input)
        msa_value = _resolve_msa(msa, _input_system(canonical_input))

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=_model_device(self.model),
            )
            realized_input = self.processor(
                canonical_input,
                placement=placement,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                generator=stream,
            )
            design_chains = tuple(chain for chain in _generated_polymer_chains(realized_input) if chain[1] == "protein")
            if not design_chains:
                raise ValueError("ODesign protein design requires at least one generated protein position")
            backbones = _sample_polymer_backbones(
                self.model,
                realized_input,
                num_backbones=num_backbones,
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
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
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
        canonical_input = _resolve_structure(input)
        msa_value = _resolve_msa(msa, _input_system(canonical_input))

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=_model_device(self.model),
            )
            realized_input = self.processor(
                canonical_input,
                placement=placement,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                generator=stream,
            )
            design_chains = tuple(chain for chain in _generated_polymer_chains(realized_input) if chain[1] == "protein")
            if not design_chains:
                raise ValueError("ODesign protein design requires at least one generated protein position")
            backbones = _sample_polymer_backbones(
                self.model,
                realized_input,
                num_backbones=num_backbones,
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
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
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
        canonical_input = _resolve_structure(input)
        msa_value = _resolve_msa(msa, _input_system(canonical_input))

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=_model_device(self.model),
            )
            realized_input = self.processor(
                canonical_input,
                placement=placement,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
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
            backbones = _sample_polymer_backbones(
                self.model,
                realized_input,
                num_backbones=num_backbones,
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
        hotspots: Sequence[Hotspot] = (),
        cyclic_bonds: Sequence[SymbolicBond] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
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
        canonical_input = _resolve_structure(input)
        msa_value = _resolve_msa(msa, _input_system(canonical_input))

        candidates: list[ODesignPolymerDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=_model_device(self.model),
            )
            realized_input = self.processor(
                canonical_input,
                placement=placement,
                msa=msa_value,
                hotspots=hotspots,
                cyclic_bonds=cyclic_bonds,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                generator=stream,
            )
            generated_chains = _generated_polymer_chains(realized_input)
            design_chains = tuple(chain for chain in generated_chains if chain[1] == "rna")
            if not design_chains:
                raise ValueError("ODesign OInvFold RNA design requires at least one generated RNA position")
            if len(design_chains) != len(generated_chains):
                raise ValueError("ODesignOInvFoldRnaPipeline accepts only generated RNA chains")
            backbones = _sample_polymer_backbones(
                self.model,
                realized_input,
                num_backbones=num_backbones,
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
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        recycling_steps: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> ODesignLigandDesignResult:
        r"""
        Generate provisional ligand slots and assign their element categories.

        The result retains generated-slot pair bond logits, but generated
        connectivity remains unresolved.
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
        canonical_input = _resolve_structure(input)
        msa_value = _resolve_msa(msa, _input_system(canonical_input))

        candidates: list[ODesignLigandDesignCandidate] = []
        for seed_index, backbone_seed in enumerate(seed_list):
            stream = resolve_generator(
                backbone_seed,
                None,
                device=_model_device(self.model),
            )
            realized_input = self.processor(
                canonical_input,
                generated_atoms=generated_atoms,
                msa=msa_value,
                hotspots=hotspots,
                fixed_atoms=fixed_atoms,
                constraint_atoms=constraint_atoms,
                generator=stream,
            )
            output = self.model.sample(
                realized_input,
                num_samples=num_backbones,
                generator=stream,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                num_model_seeds=1,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            backbones = tuple(
                _ligand_backbone_from_coordinates(
                    output.coordinates[0, 0, index],
                    output.token_bond_type_logits[0, 0],
                    realized_input,
                )
                for index in range(num_backbones)
            )

            for backbone_index, backbone in enumerate(backbones):
                slot_ids = tuple(str(index) for index in range(backbone.coordinates.shape[0]))
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


def _resolve_structure(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    if isinstance(input, (str, PathLike)):
        return read_structure(Path(input).expanduser())
    return input


def _input_system(
    input: MolecularSystem | Structure | None,
) -> MolecularSystem | None:
    return input.system if isinstance(input, Structure) else input


def _model_device(model: ODesignModel) -> torch.device:
    return next(model.parameters()).device


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
        _structure_from_atom_coordinates(
            output.coordinates[0, 0, index],
            input,
        )
        for index in range(num_backbones)
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
    complete = torch.zeros(atom_count, 3, dtype=torch.float32)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    complete[canonical_atom_index] = coordinates[valid.to(coordinates.device)].to(device="cpu", dtype=torch.float32)
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
        residue_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
        ),
        insertion_code=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
        ),
        batch_size=[],
    )


def _ligand_backbone_from_coordinates(
    coordinates: torch.Tensor,
    token_bond_type_logits: torch.Tensor,
    input: ODesignLigandInput,
) -> ODesignLigandBackbone:
    context = _structure_from_ligand_context(coordinates, input)
    slot_mask = input.atom_mask[0].bool() & (input.atom_index[0] < 0)
    generated_token_mask = input.token_mask[0].bool() & (input.token_component_index[0] < 0)
    slot_coordinates = coordinates[slot_mask].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    bond_logits = (
        token_bond_type_logits[generated_token_mask][:, generated_token_mask]
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    return ODesignLigandBackbone(
        context=context,
        coordinates=slot_coordinates,
        token_bond_type_logits=bond_logits,
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
    complete = torch.zeros(atom_count, 3, dtype=torch.float32)
    coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
    complete[canonical_atom_index] = coordinates[valid.to(coordinates.device)].to(device="cpu", dtype=torch.float32)
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
        residue_id=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
        ),
        insertion_code=torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
        ),
        batch_size=[],
    )


def _resolve_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem | None,
) -> MSA | MSACollection | None:
    if not isinstance(msa, (str, PathLike)):
        return msa
    if system is None:
        raise ValueError("an MSA path requires canonical polymer context")
    polymer_entity_ids = tuple(
        system.entity_id(entity)
        for entity in range(system.entity_symbol.numel())
        if int(system.entity_type[entity]) != ENTITY_MOLECULE
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
    chains = []
    system = input.system
    modality_by_type = {
        ENTITY_PROTEIN: "protein",
        ENTITY_DNA: "dna",
        ENTITY_RNA: "rna",
    }
    for instance in range(system.instance_symbol.numel()):
        entity = int(system.instance_entity[instance])
        modality = modality_by_type.get(int(system.entity_type[entity]))
        if modality is None:
            continue
        start = int(system.instance_component_ptr[instance])
        stop = int(system.instance_component_ptr[instance + 1])
        positions = tuple(
            (
                system.instance_id(instance),
                component - start,
            )
            for component in range(start, stop)
            if bool(input.design_component_mask[component])
        )
        if positions:
            chains.append(
                (
                    system.instance_id(instance),
                    cast(
                        Literal["protein", "dna", "rna"],
                        modality,
                    ),
                    positions,
                )
            )
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
    ("dna", "C"): ("DC", "N1"),
    ("dna", "G"): ("DG", "N9"),
    ("dna", "T"): ("DT", "N1"),
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
    component_id, base_anchor_atom_id = _NUCLEIC_ACID_IDENTITIES[modality, "A"]
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
        target_atom = system.atom_index(
            AtomSite(
                reference[0],
                reference[1],
                base_anchor_atom_id,
            )
        )
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


def _protein_sequence(
    system: MolecularSystem,
    positions: tuple[ComponentReference, ...],
) -> str:
    return "".join(
        AMINO_ACID_THREE_TO_ONE.get(
            system.symbols[int(system.component_symbol[system.component_index(reference)])],
            "X",
        )
        for reference in positions
    )


def _instance_component_references(
    system: MolecularSystem,
    instance_id: str,
) -> tuple[ComponentReference, ...]:
    instance = system.instance_index(instance_id)
    count = int(system.instance_component_ptr[instance + 1] - system.instance_component_ptr[instance])
    return tuple((instance_id, component) for component in range(count))


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
            target_atom = system.atom_index(
                AtomSite(
                    site.instance_id,
                    site.component_index,
                    atom_id,
                )
            )
        except KeyError:
            continue
        _copy_atom_fields(output, target_atom, structure, source_atom)
    return output


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
            candidate.system.symbols[
                int(candidate.system.component_symbol[candidate.system.component_index(reference)])
            ]
            for reference in result.component_references
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
        system, old_to_new, source_to_new = copy_components(
            output.system,
            target_component,
            candidate.system,
            source_component,
        )
        combined = _remap_structure(output, system, old_to_new)
        valid = source_to_new >= 0
        for source_atom, target_atom in zip(
            valid.nonzero(as_tuple=False).flatten().tolist(),
            source_to_new[valid].detach().cpu().tolist(),
            strict=True,
        ):
            _copy_atom_fields(
                combined,
                target_atom,
                candidate,
                source_atom,
            )
        output = combined
    return output


def _ligand_candidate(
    backbone: ODesignLigandBackbone,
    result: OInvFoldLigandResult,
    sequence_index: int,
    seed_index: int,
    backbone_index: int,
) -> ODesignLigandDesignCandidate:
    candidate = result.candidates[sequence_index]
    return ODesignLigandDesignCandidate(
        context=candidate.context,
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


def _empty_structure(
    system: MolecularSystem,
    source: Structure,
) -> Structure:
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
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
        _copy_atom_fields(
            output,
            target_atom,
            source,
            source_atom,
        )
    return output
