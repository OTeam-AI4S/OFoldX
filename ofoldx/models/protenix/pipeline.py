r"""
Protenix structure-prediction pipelines.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import (
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...io import pair_msa_rows_by_key, read_fasta_records, read_msa, read_msa_with_headers, read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldResult
from .model import (
    ProtenixAtomContact,
    ProtenixConstraintModel,
    ProtenixModel,
    ProtenixPocket,
    ProtenixStructureConditionModel,
    ProtenixTemplateModel,
    ProtenixTfgGuidance,
    ProtenixTokenContact,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
    molecular_system_from_input,
    molecular_system_from_named_sequences,
)

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".fna", ".seq"))
_UNIPROT_ENTRY_NAME = re.compile(r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})")
_UNIREF100_ENTRY_NAME = re.compile(r"^UniRef100_[^_]+_([^_/]+)")
_MAX_PAIRED_ROWS = 16384 // 2
_MAX_ROWS_PER_KEY = 600


def pair_msas(
    unpaired_msa_paths: Mapping[str, str | PathLike[str]],
    paired_msa_paths: Mapping[str, str | PathLike[str]],
    *,
    remove_paired_from_unpaired: bool = True,
) -> MSACollection:
    r"""
    Pair independent protein alignments by Protenix's released header keys.

    The first mapping supplies the complete unpaired pools and fixes column
    order. The second supplies header-bearing paired candidates for the same
    entities.
    """

    entity_ids = tuple(unpaired_msa_paths)
    if paired_msa_paths.keys() != unpaired_msa_paths.keys():
        raise ValueError("unpaired and paired MSA mappings must have the same entities")

    msas: list[MSA] = []
    keys_by_msa: list[tuple[str, ...]] = []
    for entity_id in entity_ids:
        unpaired = read_msa(unpaired_msa_paths[entity_id], entity_id=entity_id)
        paired, headers = read_msa_with_headers(paired_msa_paths[entity_id], entity_id=entity_id)
        if unpaired.unpaired_sequences[0] != paired.unpaired_sequences[0]:
            raise ValueError(f"unpaired and paired MSA queries differ for entity {entity_id!r}")
        msas.append(
            MSA(
                entity_id=entity_id,
                unpaired_sequences=unpaired.unpaired_sequences,
                unpaired_deletion_counts=unpaired.unpaired_deletion_counts,
                paired_sequences=paired.unpaired_sequences,
                paired_deletion_counts=paired.unpaired_deletion_counts,
            )
        )
        keys: list[str] = []
        for header in headers:
            match = _UNIPROT_ENTRY_NAME.match(header.strip()) or _UNIREF100_ENTRY_NAME.match(header.strip())
            if match is None:
                keys.append("")
            else:
                keys.append(match.group("SpeciesId") if "SpeciesId" in match.groupdict() else match.group(1))
        keys_by_msa.append(tuple(keys))

    paired_blocks = [np.zeros((1, len(msas)), dtype=np.int64)]
    paired_count = 1
    if len({msa.unpaired_sequences[0] for msa in msas}) > 1:
        for rows in pair_msa_rows_by_key(keys_by_msa, max_rows_per_key=_MAX_ROWS_PER_KEY):
            rank = np.sum(np.log(np.abs(rows.astype(np.float32)) + 1e-10), axis=1)
            paired_blocks.append(rows[np.argsort(rank)])
            paired_count += len(rows)
            if paired_count >= _MAX_PAIRED_ROWS:
                break
    paired_plan = np.concatenate(paired_blocks, axis=0)[:_MAX_PAIRED_ROWS]

    unpaired_sources: list[list[int]] = []
    for column, msa in enumerate(msas):
        if remove_paired_from_unpaired:
            paired_sequences = {msa.paired_sequences[int(row)] for row in paired_plan[:, column] if row != -1}
            unpaired_sources.append(
                [row for row, sequence in enumerate(msa.unpaired_sequences) if sequence not in paired_sequences]
            )
        else:
            unpaired_sources.append(list(range(len(msa.unpaired_sequences))))

    unpaired_plan = np.full(
        (max((len(rows) for rows in unpaired_sources), default=0), len(msas)),
        -1,
        dtype=np.int64,
    )
    for column, rows in enumerate(unpaired_sources):
        unpaired_plan[: len(rows), column] = rows

    row_pairing = np.concatenate((paired_plan, unpaired_plan), axis=0)
    row_is_paired = np.concatenate(
        (paired_plan != -1, np.zeros_like(unpaired_plan, dtype=np.bool_)),
        axis=0,
    )
    return MSACollection(tuple(msas), row_pairing, row_is_paired)


class ProtenixPipeline(PairedPipeline[ProtenixModel, ProtenixProcessor]):
    r"""
    Structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with Protenix.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, PDE, and gPDE
            values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=_canonical_msa(msa, system),
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


class ProtenixTemplatePipeline(PairedPipeline[ProtenixTemplateModel, ProtenixTemplateProcessor]):
    r"""
    Template- and RNA-MSA-capable structure prediction pipeline for Protenix v1 and v2.
    """

    model_architecture = "protenix_template"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures with the released Protenix v1/v2 template and RNA-MSA path.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional protein/RNA MSA value, collection, or alignment path.
            templates: Protein templates aligned to target chains.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and diffusion.
                Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations.
            num_sampling_steps: Number of diffusion steps.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=_canonical_template_msa(msa, system),
            templates=templates,
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


class ProtenixStructureConditionPipeline(
    PairedPipeline[ProtenixStructureConditionModel, ProtenixStructureConditionProcessor]
):
    r"""
    Structure-conditioned prediction pipeline for Protenix mini template.
    """

    model_architecture = "protenix_structure_condition"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        structure_condition: Structure | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict structures using an explicit in-memory coordinate condition.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence
                collection, molecular system, or structure topology.
            structure_condition: Optional in-memory Structure aligned by exact
                atom occurrence to the input assembly.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by preprocessing and folding.
                Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations.
            num_sampling_steps: Number of diffusion transitions.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Optional training-free chemical guidance.

        Returns:
            Generation-ordered structure predictions.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            structure_condition=structure_condition,
            msa=_canonical_msa(msa, system),
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


class ProtenixConstraintPipeline(PairedPipeline[ProtenixConstraintModel, ProtenixConstraintProcessor]):
    r"""
    Constrained structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix_constraint"

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
    ) -> FoldResult:
        r"""
        Predict constrained structures for one molecular assembly with Protenix.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            atom_contacts: Upper-bound constraints between exact atom pairs.
            token_contacts: Upper-bound constraints between token-level endpoints.
            pocket: Optional directed binder-to-residue pocket constraint.
            msa: Optional MSA value, collection, or alignment path.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preprocessing, MSA subsampling, and
                coordinate diffusion. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by constraint preprocessing and
                folding. Mutually exclusive with ``seed``.
            recycling_steps: Number of representation recycling iterations. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it;
                ``ProtenixTfgGuidance()`` uses the released enabled defaults.

        Returns:
            Generation-ordered structure predictions. Structure, PAE, PDE, and gPDE
            values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            msa=_canonical_msa(msa, system),
            tfg_guidance=tfg_guidance,
            generator=generator,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=tfg_guidance,
        )


def _canonical_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""
    Resolve Protenix's inline-sequence, FASTA, and structure conveniences.
    """

    if not isinstance(input, str | PathLike):
        if isinstance(input, MolecularSystem | Structure):
            return input
        return molecular_system_from_input(input)
    classified = classify_reference(input)
    if not isinstance(classified, Path):
        sequence = PolymerSequence.from_one_letter(classified.strip().upper())
        return molecular_system_from_named_sequences((("protein", sequence),))
    if any(suffix.lower() in _FASTA_SUFFIXES for suffix in classified.suffixes):
        sequences = tuple(
            (
                record.id or f"chain_{index + 1}",
                PolymerSequence.from_one_letter(record.sequence.strip().upper()),
            )
            for index, record in enumerate(read_fasta_records(classified))
        )
        return molecular_system_from_named_sequences(sequences)
    return read_structure(classified)


def _canonical_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read a path MSA against the only unambiguous protein entity.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    proteins = [
        system.entity_id(entity_index)
        for entity_index, entity_type in enumerate(system.entity_type.detach().cpu().tolist())
        if entity_type == ENTITY_PROTEIN
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])


def _canonical_template_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
) -> MSA | MSACollection | None:
    r"""
    Read a path MSA against the only unambiguous protein or RNA entity.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    candidates = [
        system.entity_id(entity_index)
        for entity_index, entity_type in enumerate(system.entity_type.detach().cpu().tolist())
        if entity_type in {ENTITY_PROTEIN, ENTITY_RNA}
    ]
    if len(candidates) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein or RNA entity")
    return read_msa(Path(msa).expanduser(), entity_id=candidates[0])
