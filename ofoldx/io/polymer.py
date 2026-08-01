"""Resolve declared polymer positions from source evidence, without inferring chemistry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.molecular import PolymerType

POLYMER_TYPES: dict[str, PolymerType] = {
    "polypeptide(l)": "protein",
    "polypeptide(d)": "protein",
    "cyclic-pseudo-peptide": "protein",
    "polyribonucleotide": "rna",
    "polydeoxyribonucleotide": "dna",
    "polydeoxyribonucleotide/polyribonucleotide hybrid": "dna_rna_hybrid",
    "peptide nucleic acid": "pna",
}


def resolve_monomers(
    entity_id: str,
    declared: tuple[tuple[int, tuple[str, ...]], ...],
    observed: Mapping[int, str],
) -> tuple[str, ...]:
    """Select a deposited monomer; an unobserved ambiguous position cannot be guessed.

    observed uses zero-based sequence positions. declared retains CIF's one-based
    numbering, which must be contiguous before positions can index a sequence.
    """
    if tuple(number for number, _ in declared) != tuple(range(1, len(declared) + 1)):
        raise ValueError("mmCIF polymer sequence numbering must be contiguous and start at 1")
    if any(index < 0 or index >= len(declared) for index in observed):
        raise ValueError("observed mmCIF label_seq_id is outside the declared polymer sequence")
    identifiers = []
    for index, (_, candidates) in enumerate(declared):
        identifier = observed.get(index)
        if identifier is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"unmodeled mmCIF polymer position {index + 1} of entity {entity_id!r} "
                    f"has ambiguous monomers {candidates!r}"
                )
            identifier = candidates[0]
        if identifier not in candidates:
            raise ValueError(
                f"observed mmCIF component {identifier!r} is not declared at "
                f"entity {entity_id!r} position {index + 1}"
            )
        identifiers.append(identifier)
    return tuple(identifiers)
