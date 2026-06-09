from __future__ import annotations

import pytest

from ofoldx.io import FastaRecord, parse_fasta, read_fasta


def test_parse_fasta_records() -> None:
    records = parse_fasta(">query desc\nAC\nDE\n>hit\nM-K\n")

    assert records == (
        FastaRecord(name="query", description="query desc", sequence="ACDE"),
        FastaRecord(name="hit", description="hit", sequence="M-K"),
    )


def test_read_fasta_path(tmp_path) -> None:
    path = tmp_path / "query.fasta"
    path.write_text(">query\nACD\n")

    assert read_fasta(path)[0].sequence == "ACD"


def test_parse_fasta_rejects_sequence_before_header() -> None:
    with pytest.raises(ValueError, match="before the first header"):
        parse_fasta("ACD\n")
