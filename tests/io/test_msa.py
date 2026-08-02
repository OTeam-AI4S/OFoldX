from __future__ import annotations

from pathlib import Path
from typing import get_args, get_type_hints

import numpy as np
import pytest
import torch

from ofoldx.data import MSA, MSACollection, MolecularSystem, PolymerSequence, Structure
from ofoldx.io.msa import _protein_component_one_letter, read_msa, read_msa_with_headers

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


def _structure(
    entities: dict[str, PolymerSequence],
    instances: dict[str, str],
) -> Structure:
    system = MolecularSystem.from_entities(entities, instances)
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros((atom_count, 3)),
        coordinate_mask=torch.zeros(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _write_indexed_msas(
    root: Path,
    index: str,
    *,
    sequence: str,
    paired_rows: str = "",
    unpaired_rows: str = "",
) -> None:
    directory = root / index
    directory.mkdir(parents=True)
    (directory / "uniref100_hits.a3m").write_text(f">query\n{sequence}\n{paired_rows}")
    (directory / "mmseqs_other_hits.a3m").write_text(f">query\n{sequence}\n{unpaired_rows}")


def test_read_msa_runtime_type_hints_resolve_structure_overload_types() -> None:
    hints = get_type_hints(read_msa)

    assert Structure in get_args(hints["path"])
    assert set(get_args(hints["return"])) == {MSA, MSACollection}


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


def test_read_structure_msa_returns_single_chain_collection(tmp_path: Path) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AGV")}, {"A": "protein"})
    root = tmp_path / "msa"
    _write_indexed_msas(
        root,
        "7",
        sequence="AGV",
        paired_rows=">unsupported\nA-V\n",
        unpaired_rows=">duplicate\nA-V\n>hit\nAG-\n",
    )

    collection = read_msa(structure, {"AGV": 7}, root)

    assert isinstance(collection, MSACollection)
    assert tuple(msa.entity_id for msa in collection.msas) == ("protein",)
    assert collection.target_instance_ids is None
    # UniRef rows remain available as unpaired depth if species pairing does not
    # select them; the duplicate MMseqs row is discarded in favor of its first copy.
    assert collection.msas[0].unpaired_sequences == ("AGV", "A-V", "AG-")
    assert np.array_equal(collection.row_pairing, np.array([[0], [1], [2]], dtype=np.int64))


def test_read_structure_msa_pairs_heteromer_species_from_uniref(tmp_path: Path) -> None:
    structure = _structure(
        {
            "first": PolymerSequence.from_one_letter("AAA"),
            "second": PolymerSequence.from_one_letter("CCC"),
        },
        {"A": "first", "B": "second"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(
        root,
        "1",
        sequence="AAA",
        paired_rows=">UniRef100_A_HUMAN\nA-A\n",
        unpaired_rows=">pool\nAA-\n",
    )
    _write_indexed_msas(
        root,
        "2",
        sequence="CCC",
        paired_rows=">UniRef100_B_HUMAN\nC-C\n",
        unpaired_rows=">pool\nCC-\n",
    )

    collection = read_msa(structure, {"AAA": "1", "CCC": "2"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("first", "second")
    assert np.array_equal(collection.row_pairing[:2], np.array([[0, 0], [1, 1]], dtype=np.int64))
    assert np.array_equal(collection.row_is_paired[:2], np.ones((2, 2), dtype=np.bool_))
    assert tuple(msa.unpaired_sequences for msa in collection.msas) == (
        ("AAA", "A-A", "AA-"),
        ("CCC", "C-C", "CC-"),
    )


def test_read_structure_msa_broadcasts_one_column_to_repeated_entity(tmp_path: Path) -> None:
    structure = _structure(
        {"homomer": PolymerSequence.from_one_letter("AAA")},
        {"A": "homomer", "B": "homomer"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(root, "1", sequence="AAA")

    collection = read_msa(structure, {"AAA": "1"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("homomer",)
    assert collection.target_instance_ids is None


def test_read_structure_msa_keeps_distinct_equal_sequence_entities(tmp_path: Path) -> None:
    structure = _structure(
        {
            "first": PolymerSequence.from_one_letter("AAA"),
            "second": PolymerSequence.from_one_letter("AAA"),
        },
        {"A": "first", "B": "second"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(root, "1", sequence="AAA")

    collection = read_msa(structure, {"AAA": "1"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("first", "second")
    assert collection.row_pairing.shape == (1, 2)


@pytest.mark.parametrize("missing_kind", ["mapping", "directory", "empty_directory"])
def test_read_structure_msa_falls_back_to_query_when_indexed_evidence_is_absent(
    tmp_path: Path,
    missing_kind: str,
) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AAA")}, {"A": "protein"})
    root = tmp_path / "msa"
    root.mkdir()
    mapping: dict[str, object] = {}
    if missing_kind == "directory":
        mapping["AAA"] = 157703
    elif missing_kind != "mapping":
        mapping["AAA"] = "1"
    if missing_kind == "empty_directory":
        (root / "1").mkdir()

    collection = read_msa(structure, mapping, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("protein",)
    msa = collection.msas[0]
    assert msa.unpaired_sequences == ("AAA",)
    assert msa.paired_sequences == ("AAA",)
    assert msa.unpaired_deletion_counts.dtype == np.uint32
    assert msa.paired_deletion_counts.dtype == np.uint32
    assert np.array_equal(msa.unpaired_deletion_counts, np.zeros((1, 3), dtype=np.uint32))
    assert np.array_equal(msa.paired_deletion_counts, np.zeros((1, 3), dtype=np.uint32))
    assert np.array_equal(collection.row_pairing, np.zeros((1, 1), dtype=np.int64))
    assert np.array_equal(collection.row_is_paired, np.ones((1, 1), dtype=np.bool_))


def test_read_structure_msa_all_missing_heteromer_has_one_query_row(tmp_path: Path) -> None:
    structure = _structure(
        {
            "first": PolymerSequence.from_one_letter("AAA"),
            "second": PolymerSequence.from_one_letter("CCC"),
        },
        {"A": "first", "B": "second"},
    )
    root = tmp_path / "msa"
    root.mkdir()

    collection = read_msa(structure, {}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("first", "second")
    assert tuple(msa.unpaired_sequences for msa in collection.msas) == (("AAA",), ("CCC",))
    assert np.array_equal(collection.row_pairing, np.zeros((1, 2), dtype=np.int64))
    assert np.array_equal(collection.row_is_paired, np.ones((1, 2), dtype=np.bool_))


def test_read_structure_msa_mixed_found_and_missing_preserves_available_depth(tmp_path: Path) -> None:
    structure = _structure(
        {
            "found": PolymerSequence.from_one_letter("AAA"),
            "missing": PolymerSequence.from_one_letter("CCC"),
        },
        {"A": "found", "B": "missing"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(
        root,
        "1",
        sequence="AAA",
        paired_rows=">unsupported\nA-A\n",
        unpaired_rows=">pool\nAA-\n",
    )

    collection = read_msa(structure, {"AAA": "1"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("found", "missing")
    assert collection.msas[0].unpaired_sequences == ("AAA", "A-A", "AA-")
    assert collection.msas[1].unpaired_sequences == ("CCC",)
    assert np.array_equal(
        collection.row_pairing,
        np.array([[0, 0], [1, -1], [2, -1]], dtype=np.int64),
    )
    assert np.array_equal(
        collection.row_is_paired,
        np.array([[True, True], [False, False], [False, False]], dtype=np.bool_),
    )


def test_read_structure_msa_mixed_missing_then_found_preserves_entity_order(tmp_path: Path) -> None:
    structure = _structure(
        {
            "missing": PolymerSequence.from_one_letter("CCC"),
            "found": PolymerSequence.from_one_letter("AAA"),
        },
        {"A": "missing", "B": "found"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(
        root,
        "1",
        sequence="AAA",
        paired_rows=">unsupported\nA-A\n",
        unpaired_rows=">pool\nAA-\n",
    )

    collection = read_msa(structure, {"AAA": "1"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("missing", "found")
    assert np.array_equal(
        collection.row_pairing,
        np.array([[0, 0], [-1, 1], [-1, 2]], dtype=np.int64),
    )
    assert np.array_equal(
        collection.row_is_paired,
        np.array([[True, True], [False, False], [False, False]], dtype=np.bool_),
    )


def test_read_structure_msa_species_pairing_leaves_missing_entity_gapped(tmp_path: Path) -> None:
    structure = _structure(
        {
            "first": PolymerSequence.from_one_letter("AAA"),
            "second": PolymerSequence.from_one_letter("CCC"),
            "missing": PolymerSequence.from_one_letter("DDD"),
        },
        {"A": "first", "B": "second", "C": "missing"},
    )
    root = tmp_path / "msa"
    _write_indexed_msas(root, "1", sequence="AAA", paired_rows=">UniRef100_A_HUMAN\nA-A\n")
    _write_indexed_msas(root, "2", sequence="CCC", paired_rows=">UniRef100_B_HUMAN\nC-C\n")

    collection = read_msa(structure, {"AAA": "1", "CCC": "2"}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("first", "second", "missing")
    assert np.array_equal(
        collection.row_pairing,
        np.array([[0, 0, 0], [1, 1, -1]], dtype=np.int64),
    )
    assert np.array_equal(
        collection.row_is_paired,
        np.array([[True, True, True], [True, True, False]], dtype=np.bool_),
    )


def test_read_structure_msa_broadcasts_query_fallback_to_repeated_entity(tmp_path: Path) -> None:
    structure = _structure(
        {"homomer": PolymerSequence.from_one_letter("AAA")},
        {"A": "homomer", "B": "homomer"},
    )
    root = tmp_path / "msa"
    root.mkdir()

    collection = read_msa(structure, {}, root)

    assert tuple(msa.entity_id for msa in collection.msas) == ("homomer",)
    assert collection.target_instance_ids is None
    assert np.array_equal(collection.row_pairing, np.zeros((1, 1), dtype=np.int64))


def test_read_structure_msa_rejects_nonprotein_only_structure(tmp_path: Path) -> None:
    structure = _structure(
        {"rna": PolymerSequence.from_one_letter("AC", polymer_type="rna")},
        {"R": "rna"},
    )
    root = tmp_path / "msa"
    root.mkdir()

    with pytest.raises(ValueError, match="no protein chains"):
        read_msa(structure, {}, root)


def test_protein_component_sequence_uses_single_letter_ccd_values() -> None:
    assert _protein_component_one_letter("MSE") == "M"
    assert _protein_component_one_letter("SEP") == "S"
    assert _protein_component_one_letter("not-a-ccd-code") == "X"


def test_read_structure_msa_reports_root_and_query_errors(tmp_path: Path) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AAA")}, {"A": "protein"})
    root = tmp_path / "msa"

    with pytest.raises(FileNotFoundError, match="MSA root directory"):
        read_msa(structure, {"AAA": "1"}, root)

    root.mkdir()

    _write_indexed_msas(root, "2", sequence="CCC")
    with pytest.raises(ValueError, match="structure chain sequence"):
        read_msa(structure, {"AAA": "2"}, root)

    path = _write(tmp_path, "legacy.a3m", ">query\nAAA\n")
    with pytest.raises(TypeError, match="only valid when reading MSAs for a structure"):
        read_msa(path, {}, root, entity_id="protein")
    with pytest.raises(TypeError, match="only valid when reading a single MSA file"):
        read_msa(structure, {"AAA": "2"}, root, entity_id="protein")


@pytest.mark.parametrize(
    ("present_name", "missing_name"),
    [
        ("uniref100_hits.a3m", "mmseqs_other_hits.a3m"),
        ("mmseqs_other_hits.a3m", "uniref100_hits.a3m"),
    ],
)
def test_read_structure_msa_rejects_partial_artifact(
    tmp_path: Path,
    present_name: str,
    missing_name: str,
) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AAA")}, {"A": "protein"})
    root = tmp_path / "msa"
    directory = root / "1"
    directory.mkdir(parents=True)
    (directory / present_name).write_text(">query\nAAA\n")

    with pytest.raises(FileNotFoundError, match=rf"incomplete MSA artifact.*{missing_name}"):
        read_msa(structure, {"AAA": "1"}, root)


@pytest.mark.parametrize("corrupt_name", ["uniref100_hits.a3m", "mmseqs_other_hits.a3m"])
def test_read_structure_msa_does_not_hide_empty_available_artifact(
    tmp_path: Path,
    corrupt_name: str,
) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AAA")}, {"A": "protein"})
    root = tmp_path / "msa"
    _write_indexed_msas(root, "1", sequence="AAA")
    (root / "1" / corrupt_name).write_text("")

    with pytest.raises(ValueError):
        read_msa(structure, {"AAA": "1"}, root)


def test_read_structure_msa_rejects_uniref_mmseqs_query_disagreement(tmp_path: Path) -> None:
    structure = _structure({"protein": PolymerSequence.from_one_letter("AAA")}, {"A": "protein"})
    root = tmp_path / "msa"
    _write_indexed_msas(root, "1", sequence="AAA")
    (root / "1" / "mmseqs_other_hits.a3m").write_text(">query\nCCC\n")

    with pytest.raises(ValueError, match="uniref100 and mmseqs_other MSA queries differ"):
        read_msa(structure, {"AAA": "1"}, root)
