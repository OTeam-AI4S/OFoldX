r"""
MSA parsing and featurization.

Ports the Boltz precomputed-MSA data path (``boltz.data.parse.a3m`` / ``parse.csv`` and the MSA half of
``boltz.data.feature.featurizerv2.process_msa_features``) into OFoldX's residue-token contract. Parsers
turn a3m/CSV alignment files into per-chain token rows plus run-length deletions; [`featurize_msa`][]
turns those, aligned to the processed tokens, into the ``msa`` / ``has_deletion`` / ``deletion_value`` /
``msa_paired`` / ``msa_mask`` / ``msa_profile`` / ``deletion_mean`` tensors the Boltz-1/Boltz-2 trunks consume.

MSA *generation* (querying the ColabFold MMseqs2 server for a query sequence) is intentionally not ported:
that is a remote-service dependency. This module handles the common inference path of featurizing
precomputed alignment files.

The residue-token vocabulary here is byte-identical to Boltz ``const.tokens`` (``<pad>``, ``-``, the 20 amino
acids, ``UNK``, the RNA/DNA tokens), so token ids match the released checkpoints. Alignments are
protein-only, mirroring upstream (both parsers use the protein letter map).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor

from .residue_token import _NUM_RESIDUE_TOKENS, _RESIDUE_TOKEN_IDS

_ModelInput = TypeVar("_ModelInput")

_GAP_TOKEN_ID = _RESIDUE_TOKEN_IDS["-"]
_UNK_TOKEN_ID = _RESIDUE_TOKEN_IDS["UNK"]

# One-letter -> residue token for MSA columns. Boltz ``const.prot_letter_to_token``: the 20 canonical amino
# acids, gap, and everything else (including ``X``, ``B``, ``Z``, ``J``, ``O``, ``U``) mapped to UNK.
_PROTEIN_LETTER_TO_TOKEN_ID: dict[str, int] = {
    "A": _RESIDUE_TOKEN_IDS["ALA"],
    "R": _RESIDUE_TOKEN_IDS["ARG"],
    "N": _RESIDUE_TOKEN_IDS["ASN"],
    "D": _RESIDUE_TOKEN_IDS["ASP"],
    "C": _RESIDUE_TOKEN_IDS["CYS"],
    "Q": _RESIDUE_TOKEN_IDS["GLN"],
    "E": _RESIDUE_TOKEN_IDS["GLU"],
    "G": _RESIDUE_TOKEN_IDS["GLY"],
    "H": _RESIDUE_TOKEN_IDS["HIS"],
    "I": _RESIDUE_TOKEN_IDS["ILE"],
    "L": _RESIDUE_TOKEN_IDS["LEU"],
    "K": _RESIDUE_TOKEN_IDS["LYS"],
    "M": _RESIDUE_TOKEN_IDS["MET"],
    "F": _RESIDUE_TOKEN_IDS["PHE"],
    "P": _RESIDUE_TOKEN_IDS["PRO"],
    "S": _RESIDUE_TOKEN_IDS["SER"],
    "T": _RESIDUE_TOKEN_IDS["THR"],
    "W": _RESIDUE_TOKEN_IDS["TRP"],
    "Y": _RESIDUE_TOKEN_IDS["TYR"],
    "V": _RESIDUE_TOKEN_IDS["VAL"],
    "-": _GAP_TOKEN_ID,
}


@dataclass(frozen=True, slots=True)
class ParsedMsa:
    r"""
    A parsed multiple-sequence alignment for one query chain.

    Every sequence is aligned to the query and stored column-aligned: ``token_ids[row]`` has one entry per
    query residue (gaps included), and ``deletions[row]`` counts inserted (lowercase) residues immediately
    before each query column. Duplicate ungapped sequences are dropped, matching upstream.

    Attributes:
        token_ids: Residue token id per (row, query column); shape ``(num_sequences, query_length)``.
        deletions: Insertion count per (row, query column); same shape.

    Examples:
        >>> msa = ParsedMsa(token_ids=((2, 6, 5), (2, 1, 5)), deletions=((0, 0, 0), (0, 0, 0)))
        >>> msa.num_sequences, msa.query_length
        (2, 3)
    """

    token_ids: tuple[tuple[int, ...], ...]
    deletions: tuple[tuple[int, ...], ...]

    @property
    def num_sequences(self) -> int:
        r"""
        Number of aligned rows (the query row plus every retained homolog).

        Returns:
            Count of rows in the alignment.

        Examples:
            >>> ParsedMsa(token_ids=((2, 6), (2, 1)), deletions=((0, 0), (0, 0))).num_sequences
            2
        """

        return len(self.token_ids)

    @property
    def query_length(self) -> int:
        r"""
        Number of query columns (residues) each row is aligned to; ``0`` for an empty alignment.

        Returns:
            Column count, or ``0`` when there are no rows.

        Examples:
            >>> ParsedMsa(token_ids=((2, 6, 5),), deletions=((0, 0, 0),)).query_length
            3
            >>> ParsedMsa(token_ids=(), deletions=()).query_length
            0
        """

        return len(self.token_ids[0]) if self.token_ids else 0


def parse_a3m(path: str | PathLike[str], *, max_sequences: int | None = None) -> ParsedMsa:
    r"""
    Parse an a3m alignment file (optionally gzipped).

    Args:
        path: Path to an ``.a3m`` or ``.a3m.gz`` file.
        max_sequences: Optional cap on retained sequences (including the query).

    Returns:
        The parsed alignment.

    Raises:
        ValueError: If the file has no sequences.

    Examples:
        >>> parse_a3m("query.a3m")  # doctest: +SKIP
        ParsedMsa(token_ids=..., deletions=...)
    """

    path = Path(path)
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as file:
            lines = [line for line in (raw.strip() for raw in file) if line and not line.startswith(("#", ">"))]
    else:
        with path.open() as file:
            lines = [line for line in (raw.strip() for raw in file) if line and not line.startswith(("#", ">"))]
    return _rows_to_msa(lines, max_sequences=max_sequences)


def parse_csv(path: str | PathLike[str], *, max_sequences: int | None = None) -> ParsedMsa:
    r"""
    Parse a CSV alignment file with columns ``key`` and ``sequence``.

    Args:
        path: Path to a ``.csv`` file with ``key`` and ``sequence`` columns.
        max_sequences: Optional cap on retained sequences (including the query).

    Returns:
        The parsed alignment.

    Raises:
        ValueError: If the columns are wrong or the file has no sequences.

    Examples:
        >>> parse_csv("query.csv")  # doctest: +SKIP
        ParsedMsa(token_ids=..., deletions=...)
    """

    import csv

    with Path(path).open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(sorted(reader.fieldnames or ()))
        if fieldnames != ("key", "sequence"):
            raise ValueError("invalid CSV format, expected columns ['key', 'sequence']")
        rows = [row["sequence"].strip() for row in reader if row.get("sequence", "").strip()]
    return _rows_to_msa(rows, max_sequences=max_sequences)


def featurize_msa(
    query_token_ids: Tensor,
    chain_index: Tensor,
    chain_msas: dict[int, ParsedMsa],
    *,
    max_sequences: int = 16384,
) -> dict[str, Tensor]:
    r"""
    Build MSA feature tensors aligned to processed tokens.

    Chains without an alignment contribute only their query row (gaps in every other row), so the returned
    tensors always cover every token. Non-protein tokens carry no MSA and stay gap-filled.

    Args:
        query_token_ids: Residue token id per token, shape ``(num_tokens,)`` (from the residue-type one-hot).
        chain_index: Chain index per token, shape ``(num_tokens,)``.
        chain_msas: Parsed alignment per chain index. Each MSA's ``query_length`` must equal that chain's
            token count.
        max_sequences: Cap on retained alignment rows (the query row is always kept).

    Returns:
        Batched tensors ``msa`` (long ``(1, S, N)`` token ids), ``has_deletion``/``deletion_value``/
        ``msa_paired``/``msa_mask`` (``(1, S, N)``), ``msa_profile`` (``(1, N, vocab)``), and
        ``deletion_mean`` (``(1, N)``). Padding rows are gaps with zero mask.

    Raises:
        ValueError: If a chain MSA length does not match its token count.

    Examples:
        >>> import torch
        >>> query = torch.tensor([2, 6, 5])
        >>> chain = torch.tensor([0, 0, 0])
        >>> msas = {0: ParsedMsa(token_ids=((2, 6, 5), (2, 1, 5)), deletions=((0, 0, 0), (1, 0, 0)))}
        >>> features = featurize_msa(query, chain, msas)
        >>> features["msa"].shape
        torch.Size([1, 2, 3])
        >>> features["msa"][0].tolist()
        [[2, 6, 5], [2, 1, 5]]
        >>> features["msa_profile"].shape
        torch.Size([1, 3, 33])
    """

    query_token_ids = query_token_ids.long()
    chain_index = chain_index.long()
    num_tokens = int(query_token_ids.shape[0])

    chain_rows = _chain_row_slices(chain_index)
    num_rows = max((chain_msas[chain].num_sequences for chain in chain_msas), default=1)
    num_rows = min(max(num_rows, 1), max_sequences)

    # Every non-query row defaults to a gap with mask 0; the query row (row 0) is the processed sequence.
    msa = torch.full((num_rows, num_tokens), _GAP_TOKEN_ID, dtype=torch.long)
    deletions = torch.zeros((num_rows, num_tokens), dtype=torch.float32)
    row_mask = torch.zeros((num_rows, num_tokens), dtype=torch.float32)
    msa[0] = query_token_ids
    row_mask[0] = 1.0

    for chain, columns in chain_rows.items():
        parsed = chain_msas.get(chain)
        if parsed is None:
            continue
        if parsed.query_length != len(columns):
            raise ValueError(f"chain {chain} MSA has {parsed.query_length} columns but {len(columns)} tokens")
        column_index = torch.tensor(columns, dtype=torch.long)
        rows = min(parsed.num_sequences, num_rows)
        for row in range(rows):
            msa[row, column_index] = torch.tensor(parsed.token_ids[row], dtype=torch.long)
            deletions[row, column_index] = torch.tensor(parsed.deletions[row], dtype=torch.float32)
            row_mask[row, column_index] = 1.0

    profile = torch.nn.functional.one_hot(msa, num_classes=_NUM_RESIDUE_TOKENS).float().mean(dim=0)
    has_deletion = deletions > 0
    deletion_value = (math.pi / 2) * torch.atan(deletions / 3)
    deletion_mean = deletion_value.mean(dim=0)

    return {
        "msa": msa.unsqueeze(0),
        "has_deletion": has_deletion.float().unsqueeze(0),
        "deletion_value": deletion_value.unsqueeze(0),
        "msa_paired": torch.zeros_like(deletion_value).unsqueeze(0),
        "msa_mask": row_mask.unsqueeze(0),
        "msa_profile": profile.unsqueeze(0),
        "deletion_mean": deletion_mean.unsqueeze(0),
    }


def parse_msa(path: str | PathLike[str], *, max_sequences: int | None = None) -> ParsedMsa:
    r"""
    Parse an MSA file, dispatching on the suffix (``.a3m``/``.a3m.gz`` or ``.csv``).

    Args:
        path: Path to an ``.a3m``, ``.a3m.gz``, or ``.csv`` alignment file.
        max_sequences: Optional cap on retained sequences.

    Returns:
        The parsed alignment.

    Raises:
        ValueError: If the suffix is not a supported MSA format.

    Examples:
        >>> parse_msa("query.a3m")  # doctest: +SKIP
        ParsedMsa(token_ids=..., deletions=...)
    """

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".a3m") or suffixes.endswith(".a3m.gz"):
        return parse_a3m(path, max_sequences=max_sequences)
    if suffixes.endswith(".csv"):
        return parse_csv(path, max_sequences=max_sequences)
    raise ValueError(f"unsupported MSA format: {path.name!r}")


def attach_msa(
    model_input: _ModelInput,
    chain_msas: dict[int, ParsedMsa],
    *,
    max_sequences: int = 16384,
) -> _ModelInput:
    r"""
    Return a copy of a Boltz model input with MSA feature tensors populated.

    Reads the query sequence and chain assignment from the input's own ``residue_type`` (one-hot) and
    ``chain_index`` fields, so the MSA aligns to exactly the tokens the model will see, and overwrites the
    single-sequence closed forms (``msa_profile``, ``deletion_mean``) with the real MSA statistics.

    Args:
        model_input: A processed [`Boltz1Input`][ofoldx.models.boltz1.Boltz1Input] or
            [`Boltz2Input`][ofoldx.models.boltz2.Boltz2Input] (batch size 1).
        chain_msas: Parsed alignment per chain index, in the input's first-seen chain order.
        max_sequences: Cap on retained alignment rows.

    Returns:
        A new input with ``msa``, ``has_deletion``, ``deletion_value``, ``msa_paired``, ``msa_mask``,
        ``msa_profile``, and ``deletion_mean`` set.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> model_input = AutoProcessor.from_pretrained("oteam/boltz1")("complex.cif")  # doctest: +SKIP
        >>> chain_msas = {0: parse_msa("chain_a.a3m")}  # doctest: +SKIP
        >>> model_input = attach_msa(model_input, chain_msas)  # doctest: +SKIP
    """

    query_token_ids = model_input.residue_type[0].argmax(dim=-1)  # type: ignore[attr-defined]
    chain_index = model_input.chain_index[0]  # type: ignore[attr-defined]
    features = featurize_msa(query_token_ids, chain_index, chain_msas, max_sequences=max_sequences)
    return replace(model_input, **features)  # type: ignore[type-var]


def _rows_to_msa(rows: list[str], *, max_sequences: int | None) -> ParsedMsa:
    r"""
    Convert aligned sequence strings into column-aligned token ids and per-column deletion counts.

    Lowercase columns are insertions relative to the query: they advance a deletion counter that is
    attached to the next uppercase/gap column. Duplicate ungapped sequences are skipped (upstream dedup).

    Args:
        rows: Aligned sequence strings, the query first, one uppercase/gap character per query column.
        max_sequences: Optional cap on retained rows (including the query).

    Returns:
        The column-aligned alignment.

    Raises:
        ValueError: If ``rows`` is empty or the rows have inconsistent aligned lengths.

    Examples:
        >>> msa = _rows_to_msa(["ACDE", "A-DE"], max_sequences=None)
        >>> msa.num_sequences, msa.query_length
        (2, 4)
    """

    token_rows: list[tuple[int, ...]] = []
    deletion_rows: list[tuple[int, ...]] = []
    seen: set[str] = set()
    for line in rows:
        ungapped = line.replace("-", "").upper()
        if ungapped in seen:
            continue
        seen.add(ungapped)
        tokens: list[int] = []
        row_deletions: list[int] = []
        insertion_count = 0
        for character in line:
            if character != "-" and character.islower():
                insertion_count += 1
                continue
            tokens.append(_PROTEIN_LETTER_TO_TOKEN_ID.get(character.upper(), _UNK_TOKEN_ID))
            row_deletions.append(insertion_count)
            insertion_count = 0
        token_rows.append(tuple(tokens))
        deletion_rows.append(tuple(row_deletions))
        if max_sequences is not None and len(token_rows) >= max_sequences:
            break
    if not token_rows:
        raise ValueError("MSA file contains no sequences")
    query_length = len(token_rows[0])
    if any(len(row) != query_length for row in token_rows):
        raise ValueError("MSA rows have inconsistent alignment lengths")
    return ParsedMsa(token_ids=tuple(token_rows), deletions=tuple(deletion_rows))


def _chain_row_slices(chain_index: Tensor) -> dict[int, list[int]]:
    r"""
    Return the token positions belonging to each chain, in token order.

    Args:
        chain_index: Chain index per token, shape ``(num_tokens,)``.

    Returns:
        Token positions per chain index, keyed by chain, each list in ascending token order.

    Examples:
        >>> import torch
        >>> _chain_row_slices(torch.tensor([0, 0, 1, 1, 0]))
        {0: [0, 1, 4], 1: [2, 3]}
    """

    columns: dict[int, list[int]] = {}
    for position, chain in enumerate(chain_index.tolist()):
        columns.setdefault(int(chain), []).append(position)
    return columns
