from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ofoldx.io.msa import read_msa, read_msa_with_headers

# §3.6 (frozen): read_msa(path, *, entity_id) picks the grammar from the filename suffix
# (.a3m -> A3M, MultiMolecule aligned-FASTA suffixes -> aligned FASTA, anything else
# raises), projects every row to the query width by removing insertion columns and
# counting each removed residue against the next retained column, and produces an
# unpaired pool plus a query-only paired pool. It records entity_id but does not verify
# width or chemistry (no MolecularSystem here); that is the Processor's job.


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_a3m_counts_insertions_and_yields_query_only_paired_pool(tmp_path: Path) -> None:
    # Row 0 is the ungapped query. "ins" has a lowercase insertion between G and V that
    # is removed and counted on the next emitted column; "lead" has a leading lowercase
    # run that attaches to column zero.
    path = _write(tmp_path, "aln.a3m", ">query\nAGV\n>ins\nAGkV\n>lead\nkAGV\n")

    msa = read_msa(path, entity_id="A")

    assert msa.entity_id == "A"
    assert msa.unpaired_sequences == ("AGV", "AGV", "AGV")
    assert msa.unpaired_deletion_counts.dtype == np.uint32
    assert msa.unpaired_deletion_counts.shape == (3, 3)
    assert np.array_equal(
        msa.unpaired_deletion_counts,
        np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint32),
    )
    # The paired pool is query-only and matches unpaired row zero.
    assert msa.paired_sequences == ("AGV",)
    assert msa.paired_deletion_counts.shape == (1, 3)
    assert np.array_equal(msa.paired_deletion_counts, np.array([[0, 0, 0]], dtype=np.uint32))


def test_read_aligned_fasta_removes_query_gap_columns(tmp_path: Path) -> None:
    # The query's "." marks an insertion column (query gap) removed from every row; a
    # non-gap residue there is counted against the next retained column, while a gap is
    # not. Letters are uppercased and "." is normalized to "-" before projection.
    path = _write(tmp_path, "aln.fasta", ">query\nAG.V\n>sub\nAGKV\n>gap\nA--V\n")

    msa = read_msa(path, entity_id="B")

    assert msa.entity_id == "B"
    assert msa.unpaired_sequences == ("AGV", "AGV", "A-V")
    assert np.array_equal(
        msa.unpaired_deletion_counts,
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint32),
    )
    assert msa.paired_sequences == ("AGV",)


def test_unsupported_suffix_raises_before_reading(tmp_path: Path) -> None:
    path = _write(tmp_path, "aln.txt", ">query\nAGV\n")
    with pytest.raises(ValueError, match="unsupported MSA filename suffix"):
        read_msa(path, entity_id="A")


def test_row_that_does_not_project_to_query_width_raises(tmp_path: Path) -> None:
    # The query establishes width 3; a row emitting only two aligned cells cannot project
    # to it, and the reader raises rather than padding or truncating.
    path = _write(tmp_path, "aln.a3m", ">query\nAGV\n>short\nAG\n")
    with pytest.raises(ValueError, match="project to the query width"):
        read_msa(path, entity_id="A")


def test_read_msa_with_headers_keeps_headers_aligned_to_biological_rows(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "aln.a3m",
        ">ss_pred\nHHH\n>query target sequence\nAGV\n>sp|P12345|FOO_HUMAN source note\nAGkV\n",
    )

    msa, headers = read_msa_with_headers(path, entity_id="A")

    assert msa.unpaired_sequences == ("AGV", "AGV")
    assert headers == ("query target sequence", "sp|P12345|FOO_HUMAN source note")


def test_alphafold3_pairing_handles_missing_chains_and_unsupported_headers(tmp_path: Path) -> None:
    from ofoldx.models.alphafold3.pipeline import pair_msas

    unpaired = {
        "A": _write(tmp_path, "a_unpaired.a3m", ">query\nAAA\n>human\nA-A\n>other\nAA-\n"),
        "B": _write(tmp_path, "b_unpaired.a3m", ">query\nCCC\n>human\nC-C\n>other\nCC-\n"),
        "C": _write(tmp_path, "c_unpaired.a3m", ">query\nDDD\n>other\nD-D\n"),
    }
    paired = {
        "C": _write(tmp_path, "c_paired.a3m", ">query\nDDD\n>unsupported\nD-D\n"),
        "B": _write(
            tmp_path,
            "b_paired.a3m",
            ">query\nCCC\n>sp|Q12345|BAR_HUMAN\nC-C\n>OX=9606\nCC-\n",
        ),
        "A": _write(
            tmp_path,
            "a_paired.a3m",
            ">query\nAAA\n>sp|P12345|FOO_HUMAN\nA-A\n>organism prose HUMAN\nAA-\n",
        ),
    }

    collection = pair_msas(unpaired, paired)

    assert tuple(msa.unpaired_sequences for msa in collection.msas) == (
        ("AAA", "A-A", "AA-"),
        ("CCC", "C-C", "CC-"),
        ("DDD", "D-D"),
    )
    assert np.array_equal(
        collection.row_pairing,
        np.array([[0, 0, 0], [1, 1, -1], [2, 2, 1]], dtype=np.int64),
    )
    assert np.array_equal(
        collection.row_is_paired,
        np.array([[True, True, True], [True, True, False], [False, False, False]], dtype=np.bool_),
    )
    assert collection.row_pairing.dtype == np.int64
    assert collection.row_is_paired.dtype == np.bool_


def test_pairing_keeps_protenix_uniref_support_family_owned_and_skips_homomers(tmp_path: Path) -> None:
    from ofoldx.models.alphafold3.pipeline import pair_msas as pair_alphafold3_msas
    from ofoldx.models.protenix.pipeline import pair_msas as pair_protenix_msas

    unpaired = {
        "A": _write(tmp_path, "a_unpaired.a3m", ">query\nAAA\n"),
        "B": _write(tmp_path, "b_unpaired.a3m", ">query\nCCC\n"),
    }
    paired = {
        "A": _write(tmp_path, "a_paired.a3m", ">query\nAAA\n>UniRef100_A_HUMAN\nA-A\n"),
        "B": _write(tmp_path, "b_paired.a3m", ">query\nCCC\n>UniRef100_B_HUMAN\nC-C\n"),
    }

    alphafold3 = pair_alphafold3_msas(unpaired, paired)
    protenix = pair_protenix_msas(unpaired, paired)

    assert alphafold3.row_pairing.shape == (1, 2)
    assert np.array_equal(protenix.row_pairing, np.array([[0, 0], [1, 1]], dtype=np.int64))
    assert np.array_equal(protenix.row_is_paired, np.ones((2, 2), dtype=np.bool_))

    homomer_unpaired = {
        "A": _write(tmp_path, "ha_unpaired.a3m", ">query\nAAA\n"),
        "B": _write(tmp_path, "hb_unpaired.a3m", ">query\nAAA\n"),
    }
    homomer_paired = {
        "A": _write(tmp_path, "ha_paired.a3m", ">query\nAAA\n>sp|P12345|FOO_HUMAN\nA-A\n"),
        "B": _write(tmp_path, "hb_paired.a3m", ">query\nAAA\n>sp|Q12345|BAR_HUMAN\nA-A\n"),
    }

    homomer = pair_alphafold3_msas(homomer_unpaired, homomer_paired)

    assert np.array_equal(homomer.row_pairing, np.zeros((1, 2), dtype=np.int64))
    assert np.array_equal(homomer.row_is_paired, np.ones((1, 2), dtype=np.bool_))
