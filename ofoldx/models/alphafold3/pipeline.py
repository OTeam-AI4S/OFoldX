r"""
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch

from ...data.evidence import MSA, MSACollection, Template
from ...data.molecular import MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io import pair_msa_rows_by_key, read_fasta_records, read_msa, read_msa_with_headers, read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...references import classify_reference
from ...results import FoldResult
from .model import AlphaFold3Model
from .processor import (
    AlphaFold3Processor,
    molecular_system_from_input,
    molecular_system_from_named_sequences,
)

_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".fna", ".seq"))
_UNIPROT_ENTRY_NAME = re.compile(
    r"(?:tr|sp)\|(?:[A-Z0-9]{6,10})(?:_\d+)?\|(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_MAX_PAIRED_ROWS = 16384 // 2
_MAX_ROWS_PER_KEY = 600


def pair_msas(
    unpaired_msa_paths: Mapping[str, str | PathLike[str]],
    paired_msa_paths: Mapping[str, str | PathLike[str]],
    *,
    remove_paired_from_unpaired: bool = True,
) -> MSACollection:
    r"""
    Pair independent protein alignments by AlphaFold 3's UniProt entry key.

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
        keys_by_msa.append(
            tuple(
                match.group("SpeciesId") if (match := _UNIPROT_ENTRY_NAME.match(header.strip())) else ""
                for header in headers
            )
        )

    paired_blocks = [np.zeros((1, len(msas)), dtype=np.int64)]
    paired_count = 1
    if len({msa.unpaired_sequences[0] for msa in msas}) > 1:
        for rows in pair_msa_rows_by_key(keys_by_msa, max_rows_per_key=_MAX_ROWS_PER_KEY):
            rank = np.abs(np.prod(rows.astype(np.float32), axis=1))
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


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 3.
    """

    model_architecture = "alphafold3"

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
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with AlphaFold 3.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            templates: Template structures aligned to target polymers.
            num_samples: Number of diffusion samples. ``None`` uses five.
            seed: Optional diffusion seed, mutually exclusive with ``generator``.
            generator: Optional generator for per-recycle MSA permutations, initial
                coordinates, churn, and rigid augmentations. Mutually exclusive with ``seed``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions. Coordinates, PAE, and PDE values are
            in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        canonical_input = _canonical_input(input)
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        model_input = self.processor(
            canonical_input,
            msa=_canonical_msa(msa, system),
            templates=templates,
        )
        return self.model.fold(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )


def _canonical_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure:
    r"""
    Resolve AlphaFold 3's inline-sequence, FASTA, and structure conveniences.
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
        entity_id
        for entity_id, entity in system.entities.items()
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein"
    ]
    if len(proteins) != 1:
        raise ValueError("an aligned FASTA MSA is unambiguous only for one protein entity")
    return read_msa(Path(msa).expanduser(), entity_id=proteins[0])
