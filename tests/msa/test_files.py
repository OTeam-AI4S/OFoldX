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

"""Behaviour of ofoldx.msa.files: A3M and aligned-FASTA reading, the one indexed layout and the native artifact."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from ofoldx.data.structure import Structure
from ofoldx.errors import ContradictionError, InsufficientInformationError
from ofoldx.msa import MSA, IndexedMsaLoader, MsaCollection, load_msa_collection, read_msa, save_msa_collection

A3M = ">query\nMKV\n>hit desc\nMaaK-\n>ss_pred\nCCC\n"


def rewritten(edit):
    r"""An edit of a saved artifact: ``edit`` changes its header entries and tensors, which are then saved again."""

    def apply(path) -> None:
        with safe_open(str(path), framework="np") as artifact:
            metadata, arrays = artifact.metadata(), {key: artifact.get_tensor(key) for key in artifact.keys()}
        edit(metadata, arrays)
        save_file(arrays, path, metadata=metadata)

    return apply


NOT_ASCII = np.frombuffer(b"MKM\xff", np.uint8).reshape(2, 2)  # the paired rows, the second ending in byte 0xFF
ARTIFACT_EDITS = {
    "truncated": (lambda path: path.write_bytes(path.read_bytes()[:-1]), "header"),
    "entry-missing": (rewritten(lambda metadata, _: metadata.pop("paired_headers")), "paired_headers"),
    "not-json": (rewritten(lambda metadata, _: metadata.update(entity_ids="[")), "entity_ids"),
    "wrong-type": (rewritten(lambda metadata, _: metadata.update(target_instance_ids='"A"')), "target_instance_ids"),
    "tensor-missing": (rewritten(lambda _, arrays: arrays.pop("row_pairing")), "row_pairing"),
    "not-ascii": (
        rewritten(lambda _, arrays: arrays.update({"msa.0.paired_sequences": NOT_ASCII})),
        "msa.0.paired_sequences",
    ),
}


class TestReadMsa:
    @pytest.mark.parametrize("name", ["x.a3m", "x.a3m.gz"])
    def test_a3m_rows(self, tmp_path, name):
        path = tmp_path / name
        if name.endswith(".gz"):
            path.write_bytes(gzip.compress(A3M.encode()))
        else:
            path.write_text(A3M)
        msa = read_msa(path, entity_id="1")
        assert msa.unpaired_sequences == ("MKV", "MK-")
        assert msa.unpaired_deletion_counts.tolist() == [[0, 0, 0], [0, 2, 0]]
        assert msa.paired_sequences == ("MKV",) and msa.paired_headers == ("query",)

    def test_aligned_fasta(self, tmp_path):
        path = tmp_path / "x.fasta"
        path.write_text(">q\nM-KV\n>h\nmaK-\n")
        msa = read_msa(path, entity_id="1")
        assert msa.unpaired_sequences == ("MKV", "MK-")
        assert msa.unpaired_deletion_counts.tolist() == [[0, 0, 0], [0, 1, 0]]

    def test_max_rows(self, tmp_path):
        path = tmp_path / "x.a3m"
        path.write_text(A3M)
        assert read_msa(path, entity_id="1", max_rows=1).unpaired_sequences == ("MKV",)

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (">q\nM-K\n", "msa_syntax"),
            (">q\nmK\n", "msa_syntax"),
            (">q\nMK\n>h\nM*\n", "msa_syntax"),
            ("", "msa_syntax"),
            (">q\nMK\n>h\nMKV\n", "msa_length_mismatch"),
        ],
        ids=["gapped-query", "lowercase-query", "symbol", "empty", "width"],
    )
    def test_grammar(self, tmp_path, text, code):
        path = tmp_path / "x.a3m"
        path.write_text(text)
        with pytest.raises(ContradictionError) as caught:
            read_msa(path, entity_id="1")
        assert caught.value.code == code


@pytest.fixture
def root(tmp_path):
    """Protein directories p1 (query MK) and p2 (query MQ), RNA directory r1 (query AC), and an index file."""
    files = {
        "p1/pairing.a3m": ">query\nMK\n>UniRef100_X_9606/\nMR\n>UniRef100_Y_9606/\nMR\n",
        "p1/non_pairing.a3m": ">query\nMK\n>u1\nMR\n>u2\nMA\n",
        "p2/pairing.a3m": ">query\nMQ\n",
        "p2/non_pairing.a3m": ">query\nMQ\n",
        "r1/r1_all.a3m": ">query\nAC\n>r\nAG\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text(text)
    (tmp_path / "index.json").write_text(json.dumps({"MK": "p1"}))
    return tmp_path


@pytest.fixture
def structure(polymers):
    """Proteins A and B (entity 1, MK), protein C (G), RNA R (AC) and a ligand."""
    system = polymers(
        ("A", "1", ("MET", "LYS")),
        ("B", "1", ("MET", "LYS")),
        ("C", "2", ("GLY",)),
        ("R", "3", ("A", "C")),
        ("L", "4", ("HEM",)),
    )
    return Structure.unobserved(system)


class TestIndexedMsaLoader:
    def test_layout(self, root, structure):
        pools = IndexedMsaLoader(root / "index.json", root, rna_index={"AC": ["r1"]})(structure)
        assert sorted(pools) == ["A", "B", "R"]
        protein = pools["A"]
        assert protein.unpaired_sequences == ("MK", "MR", "MA")
        assert protein.paired_sequences == ("MK", "MR", "MR")
        assert protein.paired_headers[1] == "UniRef100_X_9606/"
        assert pools["B"] is protein
        assert pools["R"].unpaired_sequences == ("AC", "AG") and pools["R"].paired_sequences == ("AC",)

    def test_reads_on_access(self, root, structure):
        pools = IndexedMsaLoader({"MK": "missing"}, root)(structure)
        assert "A" in pools
        with pytest.raises(OSError):
            pools["A"]

    def test_ambiguous_index(self, root, structure):
        with pytest.raises(InsufficientInformationError) as caught:
            IndexedMsaLoader({"MK": ["p1", "p2"]}, root)(structure)
        assert caught.value.code == "msa_index_ambiguous"

    def test_query_mismatch(self, root, structure):
        pools = IndexedMsaLoader({"MK": "p2"}, root)(structure)
        with pytest.raises(ContradictionError) as caught:
            pools["A"]
        assert caught.value.code == "msa_query_mismatch"


class TestMsaCollectionArtifact:
    @pytest.mark.parametrize("headers", [None, ("query", "tr|A0A000|X_HUMAN")])
    def test_round_trip(self, tmp_path, headers):
        counts = np.array([[0, 0], [0, 3]], dtype=np.uint32)
        msa = MSA("1", ("MK", "M-"), counts, ("MK", "MR"), counts, headers)
        plan = MsaCollection((msa,), np.array([[0], [1], [1]]), np.array([[True], [True], [False]]), ("A",))
        save_msa_collection(plan, tmp_path / "x.msa.safetensors")
        loaded = load_msa_collection(tmp_path / "x.msa.safetensors")
        (pool,) = loaded.msas
        assert (pool.entity_id, pool.unpaired_sequences, pool.paired_sequences) == ("1", ("MK", "M-"), ("MK", "MR"))
        assert pool.paired_headers == headers and pool.paired_deletion_counts.tolist() == counts.tolist()
        assert loaded.row_pairing.tolist() == [[0], [1], [1]] and loaded.target_instance_ids == ("A",)

    @pytest.mark.parametrize(("edit", "item"), ARTIFACT_EDITS.values(), ids=ARTIFACT_EDITS)
    def test_corrupt_refused(self, tmp_path, edit, item):
        counts = np.zeros((2, 2), dtype=np.uint32)
        msa = MSA("1", ("MK", "M-"), counts, ("MK", "MR"), counts, ("query", "hit"))
        path = tmp_path / "x.msa.safetensors"
        save_msa_collection(MsaCollection((msa,), np.array([[0], [1]]), np.array([[True], [False]]), ("A",)), path)
        edit(path)
        with pytest.raises(ContradictionError) as caught:
            load_msa_collection(path)
        assert (caught.value.code, caught.value.refs[0].item) == ("native_corrupt", item)
