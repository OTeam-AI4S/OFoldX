r"""
Structure and sequence design operations for BoltzGen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch

from ...data.components import get_component
from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io.msa import read_msa
from ...io.structure import read_structure
from ...pipelines import CompositePair, CompositePipeline, resolve_generator
from ...results import Metric, Score
from ...results.values import best
from ..boltz2.model import Boltz2AffinityModel, Boltz2Model
from ..boltz2.processor import Boltz2AffinityProcessor, Boltz2Processor
from .model import (
    BoltzGenInput,
    BoltzGenInverseFoldingDesignOutput,
    BoltzGenInverseFoldingInput,
    BoltzGenInverseFoldingModel,
    BoltzGenLayout,
    BoltzGenModel,
    BoltzGenPlacements,
    BoltzGenResidueConstraint,
    BoltzGenSchema,
)
from .processor import BoltzGenInverseFoldingProcessor, BoltzGenProcessor


class BoltzGenPipeline(CompositePipeline):
    r"""
    Structure and sequence design pipeline for BoltzGen.

    Args:
        design_model: Structure design model.
        design_processor: Processor for structure design.
        inverse_folding_model: Sequence design model.
        inverse_folding_processor: Processor for sequence design.
        fold: Optional Boltz-2 structure prediction model.
        fold_processor: Boltz-2 Processor paired with ``fold``.
        affinity: Optional Boltz-2 affinity model.
        affinity_processor: Boltz-2 affinity Processor paired with ``affinity``.
    """

    pair_layout = (
        CompositePair("", "design_model", "design_processor"),
        CompositePair("ifold", "inverse_folding_model", "inverse_folding_processor"),
        CompositePair("fold", "fold", "fold_processor", required=False),
        CompositePair("affinity", "affinity", "affinity_processor", required=False),
    )

    def __init__(
        self,
        *,
        design_model: BoltzGenModel,
        design_processor: BoltzGenProcessor,
        inverse_folding_model: BoltzGenInverseFoldingModel,
        inverse_folding_processor: BoltzGenInverseFoldingProcessor,
        fold: Boltz2Model | None = None,
        fold_processor: Boltz2Processor | None = None,
        affinity: Boltz2AffinityModel | None = None,
        affinity_processor: Boltz2AffinityProcessor | None = None,
    ) -> None:
        if (fold is None) != (fold_processor is None):
            raise ValueError("fold and fold_processor must either both be present or both be absent")
        if (affinity is None) != (affinity_processor is None):
            raise ValueError("affinity and affinity_processor must either both be present or both be absent")
        self.design_model = design_model
        self.design_processor = design_processor
        self.inverse_folding_model = inverse_folding_model
        self.inverse_folding_processor = inverse_folding_processor
        self.fold = fold
        self.fold_processor = fold_processor
        self.affinity = affinity
        self.affinity_processor = affinity_processor

    def design(
        self,
        input: MolecularSystem | Structure | str | PathLike[str] | None = None,
        *,
        placement: BoltzGenPlacements | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_candidates: int = 1,
        num_sampling_steps: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Generate structures, assign sequences, and evaluate optional stages.

        Args:
            input: Optional target or context system, structure, or local structure path.
                ``None`` requests a fully generated assembly.
            placement: Optional topology of existing and generated molecular segments.
            total_len: Total token budget used to realize generated segments.
            conditions: BoltzGen conditioning records.
            msa: Optional MSA evidence or alignment path.
            templates: Optional structural templates.
            num_candidates: Number of structure and sequence candidates.
            num_sampling_steps: Number of coordinate-diffusion steps. ``None`` uses
                the model configuration.
            temperature: Inverse-folding sampling temperature. ``None`` selects the
                highest-logit residue at each decoded position.
            seed: Integer seed, mutually exclusive with ``generator``.
            generator: PyTorch random generator, mutually exclusive with ``seed``.

        Returns:
            A [`BoltzGenDesignResult`][] containing candidates in generation order.
            Each candidate contains the sequence-assigned structure and any scores or
            metrics produced by configured fold and affinity stages.
        """

        generator = resolve_generator(seed, generator, device=self.design_processor.device)
        canonical_input = _resolve_structure_reference(input)
        plan = self.design_processor._plan(
            canonical_input,
            placement=placement,
            total_len=total_len,
            conditions=tuple(conditions),
        )
        canonical_msa = _resolve_msa_reference(msa, plan.system)
        design_batch = self.design_processor._lower(
            plan,
            canonical_msa,
            templates=templates,
            generator=generator,
        )
        design_output = self.design_model.design(
            design_batch,
            num_samples=num_candidates,
            num_sampling_steps=num_sampling_steps,
            generator=generator,
        )

        # The generated positions marked by structure design become the inverse-fold design mask
        # and drive sequence remapping during analysis.
        design_references = _design_references(design_batch)

        candidates: list[BoltzGenDesignCandidate] = []
        for sample_index in range(num_candidates):
            backbone = _decode_structure(design_batch, design_output.coordinates[0, sample_index])
            inverse_folding_input = self.inverse_folding_processor(backbone, design=design_references)
            inverse_folding_output = self.inverse_folding_model.design(
                inverse_folding_input, temperature=temperature, generator=generator
            )
            candidate = self._analyze(
                backbone=backbone,
                inverse_folding_input=inverse_folding_input,
                inverse_folding_output=inverse_folding_output,
                generator=generator,
            )
            candidates.append(candidate)
        return BoltzGenDesignResult(candidates=tuple(candidates))

    def inverse_fold(
        self,
        input: Structure | str | PathLike[str],
        *,
        design: Sequence[ComponentReference] = (),
        aa_constraints: Sequence[BoltzGenResidueConstraint] = (),
        symmetry: Sequence[Sequence[ComponentReference]] = (),
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        temperature: float | None = None,
        num_sequences: int = 1,
        generator: torch.Generator | None = None,
    ) -> BoltzGenDesignResult:
        r"""
        Design sequences for a fixed structure.

        Args:
            input: Fixed structure or local structure path.
            design: Residue occurrences whose identities may change.
            aa_constraints: Per-residue amino-acid constraints.
            symmetry: Residue groups sharing one sequence identity.
            msa: Optional MSA evidence or alignment path.
            temperature: Categorical sampling temperature. ``None`` selects the
                highest-logit residue at each decoded position.
            num_sequences: Number of sequences to sample.
            generator: Random generator used for decode order and residue sampling.

        Returns:
            A [`BoltzGenDesignResult`][] containing sequence-assigned structures in
            sampling order.
        """

        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        backbone = cast(Structure, _resolve_structure_reference(input))
        canonical_msa = _resolve_msa_reference(msa, backbone.system)
        inverse_folding_input = self.inverse_folding_processor(
            backbone,
            design=design,
            aa_constraints=aa_constraints,
            symmetry=symmetry,
            msa=canonical_msa,
        )
        candidates: list[BoltzGenDesignCandidate] = []
        for _ in range(num_sequences):
            inverse_folding_output = self.inverse_folding_model.design(
                inverse_folding_input, temperature=temperature, generator=generator
            )
            candidates.append(
                self._analyze(
                    backbone=backbone,
                    inverse_folding_input=inverse_folding_input,
                    inverse_folding_output=inverse_folding_output,
                    generator=generator,
                )
            )
        return BoltzGenDesignResult(candidates=tuple(candidates))

    def _analyze(
        self,
        *,
        backbone: Structure,
        inverse_folding_input: BoltzGenInverseFoldingInput,
        inverse_folding_output: BoltzGenInverseFoldingDesignOutput,
        generator: torch.Generator | None,
    ) -> BoltzGenDesignCandidate:
        r"""
        Remap the designed sequence and attach per-residue NLL and optional stage scores.
        """

        layout = inverse_folding_input.layout
        assert layout is not None
        scores: dict[str, Score] = {}
        metrics: dict[str, Metric] = {}

        logits = inverse_folding_output.logits
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        tokens = inverse_folding_output.tokens
        axis = tuple(layout.output_residues)
        per_residue_nll = -log_probs.gather(-1, tokens[0].long().unsqueeze(-1)).squeeze(-1)
        metrics["per_residue_nll"] = Metric(
            values=per_residue_nll.detach().cpu().numpy().astype(np.float32),
            axis=axis,
            unit="nats",
        )

        # Write inverse-fold identities into the candidate before optional refolding.
        structure = _remap_designed_sequence(backbone, inverse_folding_input, inverse_folding_output)
        # Composition IS the switch: refold runs iff a fold Model+Processor is composed in, affinity
        # runs iff a score Model+Processor is composed in. No mode flags.
        if self.fold is not None and self.fold_processor is not None:
            fold_batch = self.fold_processor(structure, generator=generator)
            fold_result = self.fold.fold(fold_batch, num_samples=1, generator=generator)
            prediction = fold_result.predictions[0]
            structure = prediction.structure
            for name in ("plddt", "ptm", "iptm"):
                if name in prediction.scores:
                    scores[name] = prediction.scores[name]

        if self.affinity is not None and self.affinity_processor is not None:
            affinity_batch = self.affinity_processor(structure, generator=generator)
            affinity_result = self.affinity.score(affinity_batch, generator=generator)
            if "binding_affinity" in affinity_result.scores:
                scores["binding_affinity"] = affinity_result.scores["binding_affinity"]
            if "binding_probability" in affinity_result.scores:
                scores["binding_probability"] = affinity_result.scores["binding_probability"]

        return BoltzGenDesignCandidate(structure=structure, scores=scores, metrics=metrics)


@dataclass(eq=False)
class BoltzGenDesignCandidate:
    r"""
    One designed assembly and its aligned scientific values.

    Args:
        structure: Designed structure after inverse folding and optional refolding.
        scores: Scalar scores keyed by stable scientific names.
        metrics: One-axis metrics keyed by stable scientific names.
    """

    structure: Structure
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


@dataclass(eq=False)
class BoltzGenDesignResult:
    r"""
    A generation-ordered collection of BoltzGen designs.

    Args:
        candidates: Designed assemblies in generation order.
    """

    candidates: tuple[BoltzGenDesignCandidate, ...]

    def best(self, score: str) -> BoltzGenDesignCandidate:
        r"""
        Return the earliest best candidate under a named score.

        Args:
            score: Comparable score name present on every candidate.

        Returns:
            The best candidate, with generation order breaking ties.
        """

        return best(self.candidates, score)


def _decode_structure(input: BoltzGenInput, coords: torch.Tensor) -> Structure:
    r"""
    Map one sample's model atom coords back onto canonical sites → ``Structure``.

    Model atom order is placed onto the system's complete heavy-atom sites while preserving
    residue identity.
    """

    layout = input.layout
    if layout is None or layout.system is None:
        raise ValueError("BoltzGenPipeline requires a Processor-populated layout with a system to decode")
    system = layout.system
    complete_sites = [site for site, atom in system.iter_atoms() if atom.element != "H"]
    complete_index = {site: index for index, site in enumerate(complete_sites)}
    coordinates = np.zeros((len(complete_sites), 3), dtype=np.float32)
    coordinate_mask = np.zeros(len(complete_sites), dtype=np.bool_)
    model_coords = coords.float().cpu().numpy()
    if len(model_coords) != len(layout.atom_sites):
        raise ValueError(
            f"BoltzGen returned {len(model_coords)} atom coordinates for a {len(layout.atom_sites)}-atom layout"
        )
    for site, values in zip(layout.atom_sites, model_coords, strict=True):
        index = complete_index[site]
        coordinates[index] = values
        coordinate_mask[index] = True
    return Structure(
        system=system,
        sites=tuple(complete_sites),
        coordinates=coordinates,
        coordinate_mask=coordinate_mask,
    )


def _resolve_structure_reference(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    r"""
    Read one structure reference or preserve an already canonical input.
    """

    if isinstance(input, str | PathLike):
        return read_structure(Path(input).expanduser())
    return input


def _resolve_msa_reference(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Bind an alignment path to the unique protein entity of a planned system.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    proteins = [
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA is unambiguous only for one protein entity; pass MSACollection")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])


# Model token id -> canonical component id in the BoltzGen residue vocabulary
# (0 ``<pad>``, 1 ``-``, 2..21 ALA..VAL, 22 UNK, 23..27 A/G/C/U/N, 28..32 DA/DG/DC/DT/DN) and the
# ``BoltzGenProcessor`` residue-type vocabulary.
_DESIGN_TOKEN_TO_COMPONENT: dict[int, str] = dict(
    enumerate(
        (
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "UNK", "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN",
        ),
        start=2,
    )
)  # fmt: skip


def _design_references(design_batch: BoltzGenInput) -> tuple[ComponentReference, ...]:
    r"""
    The concrete design (generated) residue references the design step marked.

    Read off ``design_batch.design_mask`` over ``layout.output_residues``; only concrete
    [`ComponentReference`][ofoldx.data.molecular.ComponentReference] rows are returned (a symbolic generated row
    from a not-yet-materialized assembly is skipped — the decode path always materializes first).
    """

    layout = cast(BoltzGenLayout, design_batch.layout)
    mask = cast(torch.Tensor, design_batch.design_mask)
    flags = mask[0].bool().tolist()
    return tuple(
        reference for reference, flag in zip(layout.output_residues, flags) if flag and isinstance(reference, tuple)
    )


def _remap_designed_sequence(
    backbone: Structure,
    inverse_folding_input: BoltzGenInverseFoldingInput,
    inverse_folding_output: BoltzGenInverseFoldingDesignOutput,
) -> Structure:
    r"""
    Write the inverse-fold argmax residue identities into the design positions of ``backbone``.

    At every design token, the argmax token id is mapped to its canonical component and written
    into the output system. Backbone ``{N, CA, C, O}`` coordinates are retained; remaining heavy
    atoms of designed residues are represented as present zero-coordinate sites. Fixed residues
    are carried unchanged.
    """

    layout = cast(BoltzGenLayout, inverse_folding_input.layout)
    tokens = inverse_folding_output.tokens
    design_mask = cast(torch.Tensor, inverse_folding_input.design_mask)

    token_ids = tokens[0].long().tolist()
    flags = design_mask[0].bool().tolist()
    designed: dict[ComponentReference, str] = {}
    for token_index, reference in enumerate(layout.output_residues):
        if not flags[token_index]:
            continue
        designed[cast(ComponentReference, reference)] = _DESIGN_TOKEN_TO_COMPONENT[token_ids[token_index]]
    if not designed:
        return backbone
    return _write_designed_structure(backbone, designed)


def _write_designed_structure(backbone: Structure, designed: dict[ComponentReference, str]) -> Structure:
    r"""
    Rebuild ``backbone`` with the designed component identities and dummy designed side chains.
    """

    system = backbone.system
    design_positions = set(designed)

    # Rebuild entities per instance so each designed residue's identity is independent (the shared
    # entity a de-novo chain rides on is minted per chain, so this only ever splits genuinely shared
    # entities the redesign touches).
    new_entities: dict[str, PolymerSequence | object] = {}
    new_instances: dict[str, str] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence):
            components = list(entity.components)
            changed = False
            for index in range(len(components)):
                name = designed.get((instance_id, index))
                if name is not None:
                    components[index] = get_component(name)
                    changed = True
            if changed:
                new_entity_id = f"{entity_id}__designed_{instance_id}"
                new_entities[new_entity_id] = PolymerSequence(entity.polymer_type, tuple(components))
                new_instances[instance_id] = new_entity_id
                continue
        new_entities.setdefault(entity_id, entity)
        new_instances[instance_id] = entity_id

    new_system = MolecularSystem(cast(dict, new_entities), new_instances, system.bonds)

    old_coordinates = {
        site: (backbone.coordinates[index], bool(backbone.coordinate_mask[index]))
        for index, site in enumerate(backbone.sites)
    }
    backbone_atoms = {"N", "CA", "C", "O"}
    sites: list[AtomSite] = []
    coordinates: list[np.ndarray] = []
    coordinate_mask: list[bool] = []
    for site, atom in new_system.iter_atoms():
        if atom.element == "H":
            continue
        reference = (site.instance_id, site.component_index)
        if reference in design_positions and site.atom_id not in backbone_atoms:
            # Designed side-chain atoms are represented at the origin and marked present; the
            # sequence is realized even though side-chain coordinates are not predicted here.
            coordinate, present = np.zeros(3, dtype=np.float32), True
        else:
            coordinate, present = old_coordinates.get(site, (np.zeros(3, dtype=np.float32), False))
        sites.append(site)
        coordinates.append(np.asarray(coordinate, dtype=np.float32))
        coordinate_mask.append(present)

    return Structure(
        system=new_system,
        sites=tuple(sites),
        coordinates=np.asarray(coordinates, dtype=np.float32).reshape(len(sites), 3),
        coordinate_mask=np.asarray(coordinate_mask, dtype=np.bool_),
    )
