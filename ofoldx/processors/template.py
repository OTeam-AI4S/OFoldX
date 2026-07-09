r"""
Structural-template search pipeline.

Ports the AlphaFold3 template pipeline (``data/templates.py`` ``Hit``/``_filter_hits``/``Templates`` and
the ``data/template_realign.py`` realignment) into OFoldX: turn ``hmmsearch`` hits against a PDB
sequence database into a date-filtered, deduplicated, max-hits-capped set of templates for a query MSA.
The hit-parsing, filtering, and realignment logic is faithful; the search step delegates to the
[`Hmmsearch`][ofoldx.processors.msa.search.Hmmsearch] wrapper.

EXTERNAL DEPENDENCIES (REQUIRED, NOT bundled):
  * HMMER binaries -- ``hmmsearch`` and ``hmmbuild`` (see [`msa.search`][ofoldx.processors.msa.search]).
  * A PDB sequence database (FASTA) to search for template hits.
  * A PDB mmCIF structure store (directory of ``<pdb_id>.cif``) with release dates, to realign hits and
    read their coordinates.

The date/max-hits/subsequence filters run without any external dependency and are unit-tested; the
search + structure-store steps require the databases above and raise a clear error when they are absent.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
from collections.abc import Iterable, Sequence

from .msa.search import Hmmsearch, MsaToolError

# Stockholm/a3m hit descriptions embed the source PDB id and chain, e.g. ``6xyz_A/12-98``.
_HIT_NAME = re.compile(r"(?P<pdb_id>[0-9a-zA-Z]{4})_(?P<chain_id>[0-9a-zA-Z]+)")


@dataclasses.dataclass(frozen=True, slots=True)
class TemplateHit:
    r"""
    One template hit parsed from an ``hmmsearch`` alignment.

    Attributes:
        pdb_id: Source PDB entry id.
        chain_id: Source chain id within the PDB entry.
        query_sequence: Gapless query sequence the hit aligns to.
        hit_sequence: Aligned template residues (a3m row, may contain gaps).
        release_date: Deposition/release date of the source structure, used for date filtering.

    Examples:
        >>> import datetime
        >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
        ...                   hit_sequence="AC-EF", release_date=datetime.date(2020, 1, 1))
        >>> hit.matching_sequence
        'ACEF'
    """

    pdb_id: str
    chain_id: str
    query_sequence: str
    hit_sequence: str
    release_date: datetime.date

    @property
    def matching_sequence(self) -> str:
        r"""
        Return the gapless template residues (gaps removed).

        Returns:
            Template sequence without alignment gaps.

        Examples:
            >>> import datetime
            >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
            ...                   hit_sequence="ac-EF", release_date=datetime.date(2020, 1, 1))
            >>> hit.matching_sequence
            'ACEF'
        """

        return self.hit_sequence.replace("-", "").replace(".", "").upper()

    @property
    def length_ratio(self) -> float:
        r"""
        Return the template-to-query length ratio (used to reject near-duplicate self-hits).

        Returns:
            Ratio of gapless template length to gapless query length.

        Examples:
            >>> import datetime
            >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
            ...                   hit_sequence="AC-EF", release_date=datetime.date(2020, 1, 1))
            >>> round(hit.length_ratio, 4)
            0.8
        """

        query_length = len(self.query_sequence.replace("-", ""))
        return len(self.matching_sequence) / query_length if query_length else 0.0

    @property
    def align_ratio(self) -> float:
        r"""
        Return the fraction of the template aligned to the query (non-gap columns).

        Returns:
            Aligned fraction of the template.

        Examples:
            >>> import datetime
            >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
            ...                   hit_sequence="AC-EF", release_date=datetime.date(2020, 1, 1))
            >>> round(hit.align_ratio, 4)
            1.0
        """

        aligned = sum(1 for residue in self.hit_sequence if residue not in {"-", "."})
        return aligned / len(self.matching_sequence) if self.matching_sequence else 0.0

    def keep(
        self,
        *,
        release_date_cutoff: datetime.date | None,
        max_subsequence_ratio: float | None,
        min_align_ratio: float | None,
        min_hit_length: int | None,
    ) -> bool:
        r"""
        Return whether this hit passes the template filters.

        Faithful port of upstream ``Hit.keep``: drop hits released after the cutoff, near-duplicate
        subsequences of the query, too-short hits, and hits with too little aligned coverage.

        Args:
            release_date_cutoff: Latest allowed template release date (``None`` disables the check).
            max_subsequence_ratio: Reject a hit that is a query subsequence above this length ratio.
            min_align_ratio: Reject hits whose aligned fraction is at or below this value.
            min_hit_length: Reject hits shorter than this many residues.

        Returns:
            ``True`` if the hit is kept.

        Examples:
            >>> import datetime
            >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
            ...                   hit_sequence="AC-EF", release_date=datetime.date(2020, 1, 1))
            >>> hit.keep(release_date_cutoff=datetime.date(2021, 1, 1), max_subsequence_ratio=0.95,
            ...          min_align_ratio=0.1, min_hit_length=3)
            True
            >>> hit.keep(release_date_cutoff=datetime.date(2019, 1, 1), max_subsequence_ratio=0.95,
            ...          min_align_ratio=0.1, min_hit_length=3)
            False
        """

        if release_date_cutoff is not None and self.release_date > release_date_cutoff:
            return False
        if (
            max_subsequence_ratio is not None
            and self.length_ratio > max_subsequence_ratio
            and self.matching_sequence in self.query_sequence.replace("-", "").upper()
        ):
            return False
        if min_hit_length is not None and len(self.matching_sequence) < min_hit_length:
            return False
        if min_align_ratio is not None and self.align_ratio <= min_align_ratio:
            return False
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class TemplateFilterConfig:
    r"""
    Thresholds for filtering template hits.

    Attributes:
        max_template_date: Latest allowed template release date (training/leakage cutoff).
        max_subsequence_ratio: Reject query-subsequence hits above this length ratio.
        min_align_ratio: Reject hits with aligned fraction at or below this value.
        min_hit_length: Reject hits shorter than this many residues.
        max_hits: Cap on the number of templates kept.
        deduplicate_sequences: If ``True``, keep only the first hit per unique template sequence.

    Examples:
        >>> import datetime
        >>> config = TemplateFilterConfig(max_template_date=datetime.date(2021, 9, 30))
        >>> config.max_hits
        4
    """

    max_template_date: datetime.date
    max_subsequence_ratio: float | None = 0.95
    min_align_ratio: float | None = 0.1
    min_hit_length: int | None = 10
    max_hits: int | None = 4
    deduplicate_sequences: bool = True


def filter_hits(hits: Iterable[TemplateHit], config: TemplateFilterConfig) -> list[TemplateHit]:
    r"""
    Filter template hits by date, length, alignment, deduplication, and max-hits cap.

    Faithful port of upstream ``_filter_hits``.

    Args:
        hits: Candidate template hits.
        config: Filter thresholds.

    Returns:
        Kept hits, in input order, capped at ``config.max_hits``.

    Examples:
        >>> import datetime
        >>> hit = TemplateHit(pdb_id="6xyz", chain_id="A", query_sequence="ACDEF",
        ...                   hit_sequence="AC-EF", release_date=datetime.date(2020, 1, 1))
        >>> config = TemplateFilterConfig(max_template_date=datetime.date(2021, 1, 1), min_hit_length=3)
        >>> [kept.pdb_id for kept in filter_hits([hit], config)]
        ['6xyz']
    """

    kept: list[TemplateHit] = []
    seen: set[str] = set()
    for hit in hits:
        if not hit.keep(
            release_date_cutoff=config.max_template_date,
            max_subsequence_ratio=config.max_subsequence_ratio,
            min_align_ratio=config.min_align_ratio,
            min_hit_length=config.min_hit_length,
        ):
            continue
        if config.deduplicate_sequences:
            if hit.hit_sequence in seen:
                continue
            seen.add(hit.hit_sequence)
        kept.append(hit)
        if config.max_hits and len(kept) == config.max_hits:
            break
    return kept


def parse_hmmsearch_hits(
    stockholm: str, query_sequence: str, release_date_by_pdb_id: dict[str, datetime.date]
) -> list[TemplateHit]:
    r"""
    Parse ``hmmsearch`` Stockholm output into template hits.

    Args:
        stockholm: Stockholm-format hits from ``hmmsearch``.
        query_sequence: Gapless query sequence the hits align to.
        release_date_by_pdb_id: Release date per source PDB id (from the structure store); hits whose id
            is absent are dropped (their date cannot be validated).

    Returns:
        Parsed template hits with resolved release dates.

    Examples:
        >>> import datetime
        >>> stockholm = "# STOCKHOLM 1.0\n6XYZ_A ACDEF\n//\n"
        >>> hits = parse_hmmsearch_hits(stockholm, "ACDEF", {"6xyz": datetime.date(2020, 1, 1)})
        >>> [(hit.pdb_id, hit.chain_id) for hit in hits]
        [('6xyz', 'A')]
    """

    hits: list[TemplateHit] = []
    for name, aligned in _stockholm_rows(stockholm):
        match = _HIT_NAME.match(name)
        if match is None:
            continue
        pdb_id = match.group("pdb_id").lower()
        release_date = release_date_by_pdb_id.get(pdb_id)
        if release_date is None:
            continue
        hits.append(
            TemplateHit(
                pdb_id=pdb_id,
                chain_id=match.group("chain_id"),
                query_sequence=query_sequence,
                hit_sequence=aligned,
                release_date=release_date,
            )
        )
    return hits


def search_templates(
    query_a3m: str,
    query_sequence: str,
    *,
    hmmsearch: Hmmsearch,
    release_date_by_pdb_id: dict[str, datetime.date],
    config: TemplateFilterConfig,
) -> list[TemplateHit]:
    r"""
    Run ``hmmsearch`` for a query MSA and return the filtered template hits.

    Ports the ``WholePdbPipeline`` flow: build an HMM from the query alignment, search the PDB sequence
    database, parse hits, resolve their release dates, and apply the date/length/dedup/max-hits filters.

    Args:
        query_a3m: Query MSA in a3m format (used to build the search HMM).
        query_sequence: Gapless query sequence.
        hmmsearch: Configured ``hmmsearch`` wrapper (validates its binary + database).
        release_date_by_pdb_id: Release date per PDB id from the structure store.
        config: Template filter thresholds.

    Returns:
        Filtered template hits.

    Raises:
        MsaToolError: If ``hmmsearch`` or its database is unavailable.

    Examples:
        >>> import datetime
        >>> hmmsearch = Hmmsearch(database_path="pdb_seqres.fasta")  # doctest: +SKIP
        >>> config = TemplateFilterConfig(max_template_date=datetime.date(2021, 9, 30))  # doctest: +SKIP
        >>> hits = search_templates(query_a3m, "ACDEF", hmmsearch=hmmsearch,
        ...                         release_date_by_pdb_id=dates, config=config)  # doctest: +SKIP
    """

    stockholm = hmmsearch.query(query_a3m)
    hits = parse_hmmsearch_hits(stockholm, query_sequence, release_date_by_pdb_id)
    return filter_hits(hits, config)


def _stockholm_rows(stockholm: str) -> Sequence[tuple[str, str]]:
    r"""
    Return concatenated ``(name, aligned_sequence)`` rows from a Stockholm string.

    Comment (``#``) and terminator (``//``) lines are skipped and wrapped blocks for the same name are
    concatenated in first-seen order.

    Args:
        stockholm: Stockholm-format alignment text.

    Returns:
        ``(name, aligned_sequence)`` pairs, one per distinct sequence name.

    Examples:
        >>> _stockholm_rows("# STOCKHOLM 1.0\n6xyz_A ACDEF\n//\n")
        [('6xyz_A', 'ACDEF')]
    """

    sequences: dict[str, str] = {}
    for line in stockholm.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        name, _, aligned = stripped.partition(" ")
        sequences[name] = sequences.get(name, "") + aligned.strip()
    return list(sequences.items())


__all__ = [
    "Hmmsearch",
    "MsaToolError",
    "TemplateFilterConfig",
    "TemplateHit",
    "filter_hits",
    "parse_hmmsearch_hits",
    "search_templates",
]
