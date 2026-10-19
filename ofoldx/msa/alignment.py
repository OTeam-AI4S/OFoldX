# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
MSA values: alignment pools for one entity, row plans over several pools, row identity, and projection of a plan
onto the tokens of a molecular system.

Headers, rows and deletion counts are facts; plans and token rows are derived, each by the one function that owns it.
An instance without an MSA has no pool (an absent key), never a fabricated query-only one: ``msa_rows`` gives every
token its own class in row 0, so such an instance has its query row and gaps below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import MolecularSystem, MonomerType
from ..data.tokenization import (
    CLASS_MONOMER,
    TOKEN_INDEX,
    TokenLayout,
    class_sequence,
    encode_sequence,
    token_classes,
)
from ..errors import RequestRef, StructureRef, fail

#: Per monomer type (row) and ASCII code (column), the class of that letter; ``NONE`` reads every letter ``<unk>``.
_DECODE = np.stack(
    [np.full(128, TOKEN_INDEX["<unk>"], dtype=np.uint8)]
    + [
        encode_sequence("".join(map(chr, range(128))), m)
        for m in (MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA)
    ]
)
_GAP = TOKEN_INDEX["-"]


@dataclass(frozen=True, eq=False)
class MSA:
    r"""
    The paired and unpaired pools of one entity.

    Rows are uppercase one-letter symbols with ``-`` for a gap, all of the query's width; row 0 of both pools is the
    same ungapped query. Deletion counts have shape ``(rows, width)``. ``paired_headers`` are the source headers of the
    paired rows (``None``: unknown, so no paired row has a species key).
    """

    entity_id: str
    unpaired_sequences: tuple[str, ...]
    unpaired_deletion_counts: NDArray[np.uint32]
    paired_sequences: tuple[str, ...]
    paired_deletion_counts: NDArray[np.uint32]
    paired_headers: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        ref = RequestRef("msa", self.entity_id)
        if not self.unpaired_sequences or not self.paired_sequences:
            fail("value_invalid", ref, invariant="query_row")
        query = self.unpaired_sequences[0]
        if self.paired_sequences[0] != query or "-" in query:
            fail("value_invalid", ref, invariant="query_row")
        for rows, counts in (
            (self.unpaired_sequences, self.unpaired_deletion_counts),
            (self.paired_sequences, self.paired_deletion_counts),
        ):
            if any(len(row) != len(query) for row in rows) or counts.shape != (len(rows), len(query)):
                fail("value_invalid", ref, invariant="width")
        if self.paired_headers is not None and len(self.paired_headers) != len(self.paired_sequences):
            fail("value_invalid", ref, invariant="paired_headers")


@dataclass(frozen=True, eq=False)
class MsaCollection:
    r"""
    A row plan over one or more pools.

    Column ``c`` is ``msas[c]``; ``row_pairing[r, c]`` selects a row of that pool (``-1`` a gap) and
    ``row_is_paired[r, c]`` the paired or the unpaired pool. Row 0 selects the paired query of every column.
    ``target_instance_ids`` binds each column to one instance; ``None`` binds each column to every instance of its
    entity (the entities are then distinct).
    """

    msas: tuple[MSA, ...]
    row_pairing: NDArray[np.int64]
    row_is_paired: NDArray[np.bool_]
    target_instance_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        pairing, paired = self.row_pairing, self.row_is_paired
        gap = pairing == -1

        def check(bad: bool, invariant: str) -> None:
            if bad:
                fail("value_invalid", RequestRef("msa_collection", str(pairing.shape)), invariant=invariant)

        check(not self.msas or pairing.ndim != 2 or pairing.shape != (pairing.shape[0], len(self.msas)), "shape")
        check(pairing.shape[0] == 0 or paired.shape != pairing.shape, "shape")
        check(not ((pairing[0] == 0).all() and paired[0].all()), "query_row")
        check(bool((gap & paired).any()) or bool(gap.all(axis=1).any()), "gap")
        for column, msa in enumerate(self.msas):
            rows, flags = pairing[:, column], paired[:, column]
            for pool, flag in ((msa.paired_sequences, True), (msa.unpaired_sequences, False)):
                cells = rows[~gap[:, column] & (flags == flag)]
                check(bool(((cells < 0) | (cells >= len(pool))).any()), "row_range")
        targets = self.target_instance_ids
        if targets is None:
            entities = [msa.entity_id for msa in self.msas]
            check(len(set(entities)) != len(entities), "broadcast_entities")
        else:
            check(len(targets) != len(self.msas) or len(set(targets)) != len(targets), "target_instance_ids")


def unique_rows(rows: NDArray, *, exclude: NDArray | None = None) -> NDArray[np.int64]:
    r"""
    The one row identity: first-occurrence indices, in source order, of distinct rows, minus rows equal to a row of
    ``exclude``.

    ``rows`` holds one projected row per element (a fixed-width array, e.g. ``np.array(sequences, dtype="S")``): a row
    is its uppercase residues and gaps, deletions are not part of it.

    Examples:
        >>> unique_rows(np.array(["AG", "A-", "AG", "CG"], dtype="S"), exclude=np.array(["CG"], dtype="S")).tolist()
        [0, 1]
    """

    rows = np.asarray(rows)
    combined = rows if exclude is None else np.concatenate((rows, np.asarray(exclude, dtype=rows.dtype)))
    _, first, inverse = np.unique(combined, return_index=True, return_inverse=True)
    banned = np.zeros(len(first), dtype=bool)
    banned[inverse.reshape(-1)[len(rows) :]] = True
    return np.sort(first[(first < len(rows)) & ~banned]).astype(np.int64)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def instance_monomer(system: MolecularSystem) -> NDArray[np.uint8]:
    r"""Per instance, the one monomer type of its components that have one (``NONE`` when none, or mixed)."""
    pointer = system.instance_component_ptr
    kinds = np.zeros((len(pointer) - 1, 4), dtype=bool)
    kinds[_owner(pointer), CLASS_MONOMER[token_classes(system).token]] = True
    single = kinds[:, 1:].sum(axis=1) == 1
    return np.where(single, kinds[:, 1:].argmax(axis=1) + 1, 0).astype(np.uint8)


@dataclass(frozen=True)
class MsaRows:
    r"""
    A plan spread onto a token axis: per row and token, its class (row 0 is ``layout.token_class``; a token no column
    covers is a gap below row 0), its deletion count and whether it comes from a paired pool (row 0 is paired).
    ``column`` is the binding itself: per token, the plan column covering it, or ``-1``.
    """

    token: NDArray[np.uint8]
    deletions: NDArray[np.uint32]
    paired: NDArray[np.bool_]
    column: NDArray[np.int64]


def _ascii(rows: Sequence[str], width: int) -> NDArray[np.uint8]:
    return np.frombuffer("".join(rows).encode("ascii", "replace"), dtype=np.uint8).reshape(len(rows), width)


def msa_rows(msa: MsaCollection, system: MolecularSystem, layout: TokenLayout) -> MsaRows:
    r"""
    The one binding and token spreading of a plan.

    Each column binds to its target instance (or every instance of its entity); a column naming no instance of its
    entity is ``InvalidRequestError(value_invalid)``. A pool whose width is not the instance's length is
    ``ContradictionError(msa_length_mismatch)``; a query that is not the instance's ``class_sequence`` is
    ``ContradictionError(msa_query_mismatch)``. Each letter is read in the monomer type of its token's class (hybrid
    instances per component), and a residue's column is spread to each of its tokens. The plan is used as given.
    """

    s = system.arrays()
    pointer = s["instance_component_ptr"]
    classes = token_classes(system).token
    entity_text = np.asarray(system.symbols()["entity"], dtype=object)[s["entity_symbol"]][s["instance_entity"]]
    column_of = np.full(len(entity_text), -1, dtype=np.int64)
    for column, pool in enumerate(msa.msas):
        ref = RequestRef("msa", pool.entity_id)
        if msa.target_instance_ids is None:
            instances = np.flatnonzero(entity_text == pool.entity_id)
        else:
            instances = system.instance_index([msa.target_instance_ids[column]])
            if instances[0] < 0 or entity_text[instances[0]] != pool.entity_id:
                instances = instances[:0]
        if not len(instances):
            fail("value_invalid", ref, invariant="msa_target")
        query = pool.unpaired_sequences[0]
        for instance in instances.tolist():
            start, stop = int(pointer[instance]), int(pointer[instance + 1])
            if len(query) != stop - start:
                fail("msa_length_mismatch", ref, columns=len(query), components=stop - start)
            expected = class_sequence(classes[start:stop], StructureRef(instance=instance))
            if query != expected:
                position = next(k for k, (a, b) in enumerate(zip(query, expected)) if a != b)
                fail("msa_query_mismatch", ref, position=position, query=query[position], target=expected[position])
        column_of[instances] = column

    token_instance = _owner(pointer)[layout.token_component]
    token_column = column_of[token_instance]
    position = layout.token_component - pointer[token_instance]
    monomer = CLASS_MONOMER[layout.token_class]
    num_rows, num_tokens = msa.row_pairing.shape[0], len(layout.token_component)
    token = np.full((num_rows, num_tokens), _GAP, dtype=np.uint8)
    token[0] = layout.token_class
    deletions = np.zeros((num_rows, num_tokens), dtype=np.uint32)
    paired = np.zeros((num_rows, num_tokens), dtype=bool)
    paired[0] = True
    for column, pool in enumerate(msa.msas):
        tokens = np.flatnonzero(token_column == column)
        if not len(tokens):
            continue
        rows, flags = msa.row_pairing[:, column], msa.row_is_paired[:, column]
        width = len(pool.unpaired_sequences[0])
        letters = np.full((num_rows, width), ord("-"), dtype=np.uint8)
        counts = np.zeros((num_rows, width), dtype=np.uint32)
        for flag, sequences, pool_counts in (
            (True, pool.paired_sequences, pool.paired_deletion_counts),
            (False, pool.unpaired_sequences, pool.unpaired_deletion_counts),
        ):
            chosen = (rows >= 0) & (flags == flag)
            letters[chosen] = _ascii(sequences, width)[rows[chosen]]
            counts[chosen] = pool_counts[rows[chosen]]
        at = position[tokens]
        token[1:, tokens] = _DECODE[monomer[tokens], letters[1:, at]]
        deletions[:, tokens] = counts[:, at]
        paired[1:, tokens] = flags[1:, None]
    return MsaRows(token, deletions, paired, token_column)
