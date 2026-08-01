r"""Canonical path and sequence conveniences for task Pipelines."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from os import PathLike
from pathlib import Path

from ..data.evidence import (
    MSA,
    MSA_ARTIFACT_SUFFIX,
    TEMPLATE_ARTIFACT_SUFFIX,
    MsaCollection,
    Template,
    load_msa_collection,
    load_templates,
)
from ..data.molecular import MolecularSystem, PolymerSequence
from ..data.structure import STRUCTURE_ARTIFACT_SUFFIX, Structure, load_structure
from ..io import read_fasta_records, read_msa, read_structure
from ..references import classify_reference

_PROTEIN_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".seq"))


def resolve_polymer_input(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
) -> MolecularSystem | Structure | tuple[tuple[str, PolymerSequence], ...]:
    r"""Classify materialized, inline-protein, protein-FASTA, and structure inputs.

    Anonymous sequences receive representation-independent instance names. A typed
    ``PolymerSequence`` is required for nucleic acids because one-letter text alone
    cannot distinguish DNA from RNA without guessing.
    """

    if isinstance(input, MolecularSystem | Structure):
        return input
    if isinstance(input, PolymerSequence):
        return (("chain_1", input),)
    if not isinstance(input, str | PathLike):
        sequences = tuple(input)
        if not sequences:
            raise ValueError("sequence collections must not be empty")
        return tuple((f"chain_{index + 1}", sequence) for index, sequence in enumerate(sequences))

    classified = classify_reference(input)
    if not isinstance(classified, Path):
        return (("chain_1", PolymerSequence.from_one_letter(classified.strip().upper())),)
    suffixes = {suffix.lower() for suffix in classified.suffixes}
    if ".fna" in suffixes:
        raise ValueError("FNA does not distinguish DNA from RNA; pass a typed PolymerSequence or MolecularSystem")
    if suffixes & _PROTEIN_FASTA_SUFFIXES:
        return tuple(
            (
                record.id or f"chain_{index + 1}",
                PolymerSequence.from_one_letter(record.sequence.strip().upper()),
            )
            for index, record in enumerate(read_fasta_records(classified))
        )
    return (
        load_structure(classified)
        if classified.name.endswith(STRUCTURE_ARTIFACT_SUFFIX)
        else read_structure(classified)
    )


def resolve_structure_input(
    input: MolecularSystem | Structure | str | PathLike[str] | None,
) -> MolecularSystem | Structure | None:
    r"""Read a structure path and preserve an already materialized value."""

    if isinstance(input, str | PathLike):
        path = Path(input).expanduser()
        return load_structure(path) if path.name.endswith(STRUCTURE_ARTIFACT_SUFFIX) else read_structure(path)
    return input


def resolve_msa_input(
    msa: MSA | MsaCollection | str | PathLike[str] | None,
    system: MolecularSystem | None,
    *,
    entity_types: Collection[int],
    max_rows: int | None = None,
) -> MSA | MsaCollection | None:
    r"""
    Bind a text alignment path to the only accepted polymer entity.

    A native ``MsaCollection`` artifact is an exact prepared value and is loaded
    without row truncation. ``max_rows`` applies only while parsing text alignments.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    path = Path(msa).expanduser()
    if path.name.endswith(MSA_ARTIFACT_SUFFIX):
        return load_msa_collection(path)
    if system is None:
        raise ValueError("an MSA path requires canonical polymer context")
    candidates = tuple(
        system.entity_id(entity)
        for entity, entity_type in enumerate(system.entity_type.detach().cpu().tolist())
        if entity_type in entity_types
    )
    if len(candidates) != 1:
        raise ValueError(
            "an MSA path is unambiguous only for one matching polymer entity; "
            "read it with an explicit entity_id or pass MsaCollection"
        )
    return read_msa(path, entity_id=candidates[0], max_rows=max_rows)


def resolve_template_input(
    templates: Sequence[Template] | str | PathLike[str],
) -> tuple[Template, ...]:
    r"""Load a native template artifact or materialize an in-memory sequence."""

    if not isinstance(templates, str | PathLike):
        return tuple(templates)
    path = Path(templates).expanduser()
    if not path.name.endswith(TEMPLATE_ARTIFACT_SUFFIX):
        raise ValueError(f"template artifact paths must end with {TEMPLATE_ARTIFACT_SUFFIX!r}")
    return load_templates(path)
