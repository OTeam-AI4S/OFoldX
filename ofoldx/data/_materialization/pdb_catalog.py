"""PDB-only declarations; no mmCIF placeholders or tensor graph state."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from ..._arrays import immutable_array


@dataclass(frozen=True, kw_only=True)
class PdbDeclarations:
    lines: tuple[str, ...]
    sequences: Mapping[str, tuple[str, ...]]
    missing_residues: tuple[tuple[str, str, int, str], ...]
    entity_by_chain: Mapping[str, str]
    modified_residues: Mapping[tuple[str, int, str, str], str]
    segment_by_serial: Mapping[str, int]
    assembly_placements: Mapping[str, tuple[np.ndarray, frozenset[str]]] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))
        object.__setattr__(self, "missing_residues", tuple(self.missing_residues))
        for name in ("sequences", "entity_by_chain", "modified_residues", "segment_by_serial"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.assembly_placements is not None:
            placements = {
                key: (immutable_array(transform, np.float64), frozenset(chains))
                for key, (transform, chains) in self.assembly_placements.items()
            }
            object.__setattr__(self, "assembly_placements", MappingProxyType(placements))

    def select_chains(self, chain_ids: Collection[str]) -> PdbDeclarations:
        """Select interpreted declarations without rewriting the original records."""
        selected = frozenset(chain_ids)
        return replace(
            self,
            sequences={key: value for key, value in self.sequences.items() if key in selected},
            missing_residues=tuple(value for value in self.missing_residues if value[0] in selected),
            entity_by_chain={key: value for key, value in self.entity_by_chain.items() if key in selected},
            modified_residues={key: value for key, value in self.modified_residues.items() if key[0] in selected},
            assembly_placements=(
                None
                if self.assembly_placements is None
                else {
                    key: (matrix, chains & selected)
                    for key, (matrix, chains) in self.assembly_placements.items()
                    if chains & selected
                }
            ),
        )

    @classmethod
    def read(
        cls,
        lines: Sequence[str],
        missing_residues: Sequence[tuple[str, str, int, str]],
        assembly_placements: Mapping[str, tuple[np.ndarray, Collection[str]]] | None,
    ) -> PdbDeclarations:
        from ...io.pdb import pdb_entity_by_chain, pdb_modified_residues, pdb_segment_by_serial, pdb_sequences

        lines = tuple(lines)
        return cls(
            lines=tuple(lines),
            sequences=pdb_sequences(lines),
            missing_residues=tuple(missing_residues),
            entity_by_chain=pdb_entity_by_chain(lines),
            modified_residues=pdb_modified_residues(lines),
            segment_by_serial=pdb_segment_by_serial(lines),
            assembly_placements=(
                None
                if assembly_placements is None
                else {key: (matrix, frozenset(chains)) for key, (matrix, chains) in assembly_placements.items()}
            ),
        )
