from __future__ import annotations

import gzip
import json
import lzma
from pathlib import Path

import numpy as np
import pytest
import torch
import zstandard

from ofoldx.data import MSA, MolecularSystem, MsaCollection, PolymerSequence, Structure
from ofoldx.io import (
    IndexedMsaLoader,
    decode_msa_collection,
    encode_msa_collection,
    read_msa,
    read_msa_with_headers,
)

# read_msa(path, *, entity_id) picks the grammar from the filename suffix
# (.a3m -> A3M, MultiMolecule aligned-FASTA suffixes -> aligned FASTA, anything else
# raises), projects every row to the query width by removing insertion columns and
# counting each removed residue against the next retained column, and produces an
# unpaired pool plus a query-only paired pool. It records entity_id but does not verify
# width or chemistry (no MolecularSystem here); that is the Processor's job.


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


class TestReadMsa:
    def test_a3m(self, tmp_path: Path) -> None:
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
        assert not np.shares_memory(msa.paired_deletion_counts, msa.unpaired_deletion_counts)

    @pytest.mark.parametrize(
        ("suffix", "opener"),
        (("gz", gzip.open), ("xz", lzma.open), ("zst", zstandard.open)),
    )
    def test_compressed_a3m(self, tmp_path: Path, suffix: str, opener: object) -> None:
        path = tmp_path / f"aln.a3m.{suffix}"
        assert callable(opener)
        with opener(path, "wt") as file:
            file.write(">query\nAGV\n>hit\nAGkV\n")

        msa = read_msa(path, entity_id="A")

        assert msa.unpaired_sequences == ("AGV", "AGV")
        assert np.array_equal(msa.unpaired_deletion_counts[1], np.array([0, 0, 1], dtype=np.uint32))

    def test_aligned_fasta(self, tmp_path: Path) -> None:
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

    def test_suffix(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "aln.txt", ">query\nAGV\n")
        with pytest.raises(ValueError, match="unsupported MSA filename suffix"):
            read_msa(path, entity_id="A")

    def test_width(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "aln.a3m", ">query\nAGV\n>short\nAG\n")
        with pytest.raises(ValueError, match="project to the query width"):
            read_msa(path, entity_id="A")

    def test_headers(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "aln.a3m",
            ">ss_pred\nHHH\n>query target sequence\nAGV\n>sp|P12345|FOO_HUMAN source note\nAGkV\n",
        )

        msa, headers = read_msa_with_headers(path, entity_id="A")

        assert msa.unpaired_sequences == ("AGV", "AGV")
        assert headers == ("query target sequence", "sp|P12345|FOO_HUMAN source note")


def test_msa_collection_codec() -> None:
    msa = MSA(
        "target",
        ("A", "G"),
        np.zeros((2, 1), dtype=np.uint32),
        ("A",),
        np.zeros((1, 1), dtype=np.uint32),
    )
    collection = MsaCollection(
        (msa,),
        np.array([[0], [1]], dtype=np.int64),
        np.array([[True], [False]], dtype=np.bool_),
        target_instance_ids=("A",),
    )
    restored = decode_msa_collection(json.loads(json.dumps(encode_msa_collection(collection))))

    assert restored.target_instance_ids == ("A",)
    assert restored.msas[0].unpaired_sequences == ("A", "G")
    np.testing.assert_array_equal(
        restored.msas[0].unpaired_deletion_counts,
        msa.unpaired_deletion_counts,
    )
    np.testing.assert_array_equal(restored.row_pairing, collection.row_pairing)
    np.testing.assert_array_equal(restored.row_is_paired, collection.row_is_paired)


class TestIndexedMsaLoader:
    def _case(self, tmp_path: Path) -> tuple[Path, Structure]:
        root = tmp_path / "msa"
        for directory_id, query, paired, unpaired in (
            ("1", "AGV", "CGV", "TGV"),
            ("2", "GAK", "WAK", "AAK"),
        ):
            directory = root / directory_id
            directory.mkdir(parents=True)
            _ = (directory / "paired.a3m").write_text(f">query\n{query}\n>UniRef100_hit_MOUSE\n{paired}\n")
            _ = (directory / "unpaired.a3m").write_text(f">query\n{query}\n>unpaired\n{unpaired}\n")

        system = MolecularSystem.from_entities(
            {
                "left": PolymerSequence.from_one_letter("AGV"),
                "right": PolymerSequence.from_one_letter("GAK"),
            },
            {"A": "left", "B": "right"},
        )
        atom_count = system.atom_symbol.numel()
        component_count = system.component_symbol.numel()
        structure = Structure(  # type: ignore[call-arg]
            system=system,
            coordinates=torch.zeros(atom_count, 3),
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
        return root, structure

    def test_pairing(self, tmp_path: Path) -> None:
        root, structure = self._case(tmp_path)
        loader = IndexedMsaLoader(
            {"AGV": "1", "GAK": "2"},
            root,
            paired_filename="paired.a3m",
            unpaired_filename="unpaired.a3m",
        )

        collection = loader(structure)

        assert collection is not None
        assert tuple(msa.entity_id for msa in collection.msas) == ("left", "right")
        np.testing.assert_array_equal(
            collection.row_pairing,
            np.array([[0, 0], [1, 1], [2, -1], [-1, 2]], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            collection.row_is_paired,
            np.array(
                [[True, True], [True, True], [False, False], [False, False]],
                dtype=np.bool_,
            ),
        )

    def test_fallback(self, tmp_path: Path) -> None:
        root, structure = self._case(tmp_path)

        fallback = IndexedMsaLoader({}, root)(structure)

        assert fallback is not None
        assert tuple(msa.unpaired_sequences for msa in fallback.msas) == (("AGV",), ("GAK",))
