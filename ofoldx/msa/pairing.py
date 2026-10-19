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
MSA pairing: species keys from paired-pool headers and a depth-limited row plan over the pools of participating
polymer instances.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from ..data.molecular import EntityType, MolecularSystem, MonomerType
from ..data.tokenization import TokenLayout
from ..provenance import known
from .alignment import MSA, MsaCollection, instance_monomer, unique_rows

#: The NCBI TaxID form of UniRef100 headers (ColabFold-style searches; TaxID 0, no taxon, gives no key) and the
#: UniProtKB mnemonic species code (1–5 uppercase alphanumerics with at least one letter).
_TAXID = re.compile(r"UniRef100_[A-Z0-9]+(?:-\d+)?_(?P<taxid>[1-9][0-9]*)")
_MNEMONIC = re.compile(
    r"(?:sp|tr)\|[A-Z0-9]{6,10}(?:[-_]\d+)?\|[A-Z0-9]{1,10}_(?P<code>(?=[A-Z0-9]*[A-Z])[A-Z0-9]{1,5})"
)


def _species(header: str) -> tuple[str, str] | None:
    r"""Return a paired-pool header's species key, or None for an unrecognized organism identifier."""
    token = header.split(maxsplit=1)[0].partition("/")[0] if header.strip() else ""
    if match := _TAXID.fullmatch(token):
        return ("taxid", match["taxid"])
    if match := _MNEMONIC.fullmatch(token):
        return ("mnemonic", match["code"])
    return None


def pair_msas(
    pools: Mapping[str, MSA], system: MolecularSystem, layout: TokenLayout, *, max_rows: int | None = None
) -> MsaCollection | None:
    r"""Pair alignment pools for polymer instances represented by a token layout.

    Args:
        pools: Alignment pools keyed by instance ID. Only participating pools are read.
        system: Molecular system defining the instance and entity identities.
        layout: Token layout, usually after cropping, that identifies participants.
        max_rows: Optional limit on the complete row plan.

    Returns:
        A row plan retaining the original pools, or None when no participating instance
        has a pool.

    Notes:
        Participants follow system order and must be polymer instances with tokens in
        the layout and an available pool. Protein pairing runs when these participants
        include at least two distinct query sequences. Other monomer types contribute
        their query and unpaired rows.

        Species keys come from UniRef100 TaxIDs or UniProt mnemonic codes; the two
        key types remain distinct. For each participant and key, the first paired row
        is used. A species contributes a paired row when it spans two distinct queries;
        missing participants receive gaps. The query row comes first, followed by
        species ordered by descending carrier count, source-row sum, then species key.

        With a row limit, the paired block keeps at most `max(1, max_rows // 2)` rows.
        Remaining unpaired sequences are deduplicated in pool order, exclude sequences
        consumed by that block, and are stacked by depth. The complete plan is capped
        at `max_rows`. Chain length does not affect this ordering or allocation, and
        the returned collection retains the pools unchanged.
    """

    s = system.arrays()
    entity = s["instance_entity"]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    cropped = np.zeros(len(entity), dtype=bool)
    pointer = s["instance_component_ptr"]
    cropped[np.repeat(np.arange(len(entity)), np.diff(pointer))[layout.token_component]] = True
    names = system.symbols()["instance"]
    candidates = [(int(i), names[int(s["instance_symbol"][i])]) for i in np.flatnonzero(polymer & cropped)]
    participants = [(i, name) for i, name in candidates if name in pools]
    if not participants:
        return None
    msas = [pools[name] for _, name in participants]
    protein = instance_monomer(system)[[i for i, _ in participants]] == MonomerType.PROTEIN
    queries = [msa.unpaired_sequences[0] for msa in msas]
    count = len(msas)

    block = [np.zeros(count, dtype=np.int64)]
    if len({query for query, is_protein in zip(queries, protein) if is_protein}) >= 2:
        carriers: dict[tuple[str, str], dict[int, int]] = {}
        for column, msa in enumerate(msas):
            if protein[column] and msa.paired_headers is not None:
                for row, header in enumerate(msa.paired_headers[1:], start=1):
                    key = _species(header)
                    if key is not None:
                        carriers.setdefault(key, {}).setdefault(column, row)
        keyed = []
        for key, rows in carriers.items():
            if len({queries[column] for column in rows}) >= 2:
                cells = np.full(count, -1, dtype=np.int64)
                cells[list(rows)] = list(rows.values())
                keyed.append(((-len(rows), sum(rows.values()), key), cells))
        block += [cells for _, cells in sorted(keyed, key=lambda item: item[0])]
    kept = np.stack(block)[: None if max_rows is None else max(1, max_rows // 2)]

    stacks = []
    for column, msa in enumerate(msas):
        consumed = [msa.paired_sequences[row] for row in kept[:, column] if row >= 0]
        stacks.append(unique_rows(np.array(msa.unpaired_sequences, dtype="S"), exclude=np.array(consumed, dtype="S")))
    stack = np.full((max(len(rows) for rows in stacks), count), -1, dtype=np.int64)
    for column, rows in enumerate(stacks):
        stack[: len(rows), column] = rows
    return MsaCollection(
        tuple(msas),
        np.concatenate((kept, stack))[:max_rows],
        np.concatenate((kept >= 0, np.zeros(stack.shape, dtype=bool)))[:max_rows],
        tuple(name for _, name in participants),
    )
