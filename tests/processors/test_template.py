from __future__ import annotations

import datetime

from ofoldx.processors.template import (
    TemplateFilterConfig,
    TemplateHit,
    filter_hits,
    parse_hmmsearch_hits,
)

_CUTOFF = datetime.date(2020, 1, 1)


def _hit(pdb_id: str, hit_sequence: str, release: datetime.date, query: str = "ACDEFGHIKLMNPQRSTVWY") -> TemplateHit:
    return TemplateHit(
        pdb_id=pdb_id, chain_id="A", query_sequence=query, hit_sequence=hit_sequence, release_date=release
    )


def _config(**overrides: object) -> TemplateFilterConfig:
    base = {
        "max_template_date": _CUTOFF,
        "max_subsequence_ratio": 0.95,
        "min_align_ratio": 0.0,
        "min_hit_length": 1,
        "max_hits": 10,
        "deduplicate_sequences": True,
    }
    base.update(overrides)
    return TemplateFilterConfig(**base)  # type: ignore[arg-type]


def test_hit_after_cutoff_is_dropped() -> None:
    kept = filter_hits([_hit("1abc", "GHIKLM", datetime.date(2021, 6, 1))], _config())
    assert kept == []


def test_hit_on_or_before_cutoff_is_kept() -> None:
    kept = filter_hits([_hit("1abc", "GHIKLM", datetime.date(2019, 6, 1))], _config())
    assert len(kept) == 1


def test_short_hit_is_dropped() -> None:
    kept = filter_hits([_hit("1abc", "GH", datetime.date(2019, 1, 1))], _config(min_hit_length=5))
    assert kept == []


def test_deduplication_keeps_first() -> None:
    hits = [
        _hit("1abc", "GHIKLM", datetime.date(2019, 1, 1)),
        _hit("2def", "GHIKLM", datetime.date(2018, 1, 1)),  # duplicate sequence
        _hit("3ghi", "NPQRST", datetime.date(2018, 1, 1)),
    ]
    kept = filter_hits(hits, _config())
    assert [hit.pdb_id for hit in kept] == ["1abc", "3ghi"]


def test_max_hits_caps_output() -> None:
    hits = [_hit(f"{i}abc", f"SEQ{i}ABCDE", datetime.date(2019, 1, 1)) for i in range(5)]
    kept = filter_hits(hits, _config(max_hits=2))
    assert len(kept) == 2


def test_subsequence_of_query_is_dropped() -> None:
    query = "ACDEFGHIKLMNPQRSTVWY"
    # A near-full copy of the query is a self-hit and must be excluded.
    kept = filter_hits(
        [_hit("1abc", query, datetime.date(2019, 1, 1), query=query)], _config(max_subsequence_ratio=0.5)
    )
    assert kept == []


def test_parse_hmmsearch_hits_resolves_release_dates() -> None:
    stockholm = "\n".join(
        [
            "# STOCKHOLM 1.0",
            "1abc_A GHIKLMNPQR",
            "2def_B STVWYACDEF",
            "9zzz_C UNKNOWNDATE",  # no release date -> dropped
            "//",
        ]
    )
    dates = {"1abc": datetime.date(2019, 1, 1), "2def": datetime.date(2018, 1, 1)}
    hits = parse_hmmsearch_hits(stockholm, "GHIKLMNPQR", dates)
    assert {hit.pdb_id for hit in hits} == {"1abc", "2def"}
    assert hits[0].chain_id == "A"
