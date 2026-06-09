from __future__ import annotations

import math

import pytest
import torch

from ofoldx.processors.msa import (
    ParsedMsa,
    attach_msa,
    featurize_msa,
    parse_a3m,
    parse_csv,
    parse_msa,
)
from ofoldx.processors.residue_token import _RESIDUE_TOKEN_IDS

_MET, _LYS, _VAL, _GAP, _UNK = (
    _RESIDUE_TOKEN_IDS["MET"],
    _RESIDUE_TOKEN_IDS["LYS"],
    _RESIDUE_TOKEN_IDS["VAL"],
    _RESIDUE_TOKEN_IDS["-"],
    _RESIDUE_TOKEN_IDS["UNK"],
)


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parse_a3m_insertions_and_dedup(tmp_path) -> None:
    # h1 has a lowercase insertion 'k' before column 1; h2 has a gap; h3 duplicates the query (ungapped) -> dropped.
    path = _write(tmp_path, "aln.a3m", ">query\nMKV\n>h1\nMkKV\n>h2\nM-V\n>h3\nMKV\n")
    msa = parse_a3m(path)

    assert msa.num_sequences == 3  # query + h1 + h2; h3 deduped
    assert msa.query_length == 3
    assert msa.token_ids[0] == (_MET, _LYS, _VAL)
    assert msa.token_ids[1] == (_MET, _LYS, _VAL) and msa.deletions[1] == (0, 1, 0)
    assert msa.token_ids[2] == (_MET, _GAP, _VAL) and msa.deletions[2] == (0, 0, 0)


def test_parse_a3m_unknown_letter_maps_to_unk(tmp_path) -> None:
    path = _write(tmp_path, "aln.a3m", ">q\nMXV\n")
    assert parse_a3m(path).token_ids[0] == (_MET, _UNK, _VAL)


def test_parse_csv(tmp_path) -> None:
    path = _write(tmp_path, "aln.csv", "key,sequence\n100,MKV\n101,MKV\n102,M-V\n")
    msa = parse_csv(path)
    assert msa.num_sequences == 2  # second row deduped
    assert msa.token_ids[0] == (_MET, _LYS, _VAL)


def test_parse_csv_bad_columns(tmp_path) -> None:
    path = _write(tmp_path, "bad.csv", "id,seq\n1,MKV\n")
    with pytest.raises(ValueError, match="CSV"):
        parse_csv(path)


def test_parse_msa_dispatch_and_unsupported(tmp_path) -> None:
    a3m = _write(tmp_path, "a.a3m", ">q\nMK\n")
    csv = _write(tmp_path, "a.csv", "key,sequence\n1,MK\n")
    assert parse_msa(a3m).query_length == 2
    assert parse_msa(csv).query_length == 2
    with pytest.raises(ValueError, match="unsupported"):
        parse_msa(_write(tmp_path, "a.txt", "MK"))


def test_parse_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="no sequences"):
        parse_a3m(_write(tmp_path, "empty.a3m", "# comment only\n"))


def test_featurize_math() -> None:
    r"""profile, has_deletion, deletion_value (pi/2 * atan(d/3)) and deletion_mean match the reference math."""

    parsed = ParsedMsa(
        token_ids=((_MET, _LYS, _VAL), (_MET, _LYS, _VAL)),
        deletions=((0, 0, 0), (0, 2, 0)),
    )
    query = torch.tensor([_MET, _LYS, _VAL])
    feats = featurize_msa(query, torch.tensor([0, 0, 0]), {0: parsed})

    assert feats["msa"].shape == (1, 2, 3) and feats["msa"].dtype == torch.long
    assert feats["msa_profile"].shape == (1, 3, 33)
    assert torch.equal(feats["msa"][0, 0], query)
    # every row has MET at column 0 -> profile mass 1 on MET, 0 elsewhere
    assert feats["msa_profile"][0, 0, _MET].item() == pytest.approx(1.0)
    expected_deletion = (math.pi / 2) * math.atan(2 / 3)
    assert feats["deletion_value"][0, 1, 1].item() == pytest.approx(expected_deletion)
    assert feats["has_deletion"][0, 1, 1].item() == 1.0
    assert feats["deletion_mean"][0, 1].item() == pytest.approx(expected_deletion / 2)


def test_featurize_multichain_and_missing() -> None:
    r"""Chain without an MSA stays gap-filled beyond its query row; provided chain fills its own columns."""

    chain_a = ParsedMsa(token_ids=((_MET, _LYS), (_VAL, _LYS)), deletions=((0, 0), (0, 0)))
    query = torch.tensor([_MET, _LYS, _VAL])  # tokens 0,1 = chain 0; token 2 = chain 1 (no MSA)
    feats = featurize_msa(query, torch.tensor([0, 0, 1]), {0: chain_a})

    assert feats["msa"].shape == (1, 2, 3)
    assert feats["msa"][0, 1].tolist() == [_VAL, _LYS, _GAP]  # chain-1 column gap-filled in row 1
    assert feats["msa_mask"][0, 1].tolist() == [1.0, 1.0, 0.0]  # unmasked only where chain 0 supplied a row


def test_featurize_length_mismatch_raises() -> None:
    parsed = ParsedMsa(token_ids=((_MET, _LYS),), deletions=((0, 0),))
    with pytest.raises(ValueError, match="columns"):
        featurize_msa(torch.tensor([_MET, _LYS, _VAL]), torch.tensor([0, 0, 0]), {0: parsed})


def test_attach_msa_uses_input_query(tmp_path) -> None:
    from ofoldx.data import ResidueKey, Structure
    from ofoldx.models.boltz2.processor import Boltz2Processor
    from tests.processors.conftest import _component, _FakeCcd, _residue

    met, lys = ResidueKey("A", 1), ResidueKey("A", 2)
    structure = Structure(
        (
            *_residue("MET", met, {"N": "N", "CA": "C", "C": "C", "O": "O"}),
            *_residue("LYS", lys, {"N": "N", "CA": "C", "C": "C", "O": "O"}),
        )
    )
    ccd = _FakeCcd(
        {"MET": _component({"N": 0, "CA": 1, "C": 2, "O": 3}), "LYS": _component({"N": 0, "CA": 1, "C": 2, "O": 3})}
    )
    processed = Boltz2Processor(ccd=ccd)(structure)
    assert processed.msa is None

    parsed = parse_a3m(_write(tmp_path, "aln.a3m", ">q\nMK\n>h1\nMkK\n"))
    attached = attach_msa(processed, {0: parsed})

    assert attached.msa is not None and attached.msa[0, 0].tolist() == [_MET, _LYS]
    assert attached.msa_profile.shape == (1, 2, 33)
    assert torch.equal(attached.residue_type, processed.residue_type)  # untouched fields preserved
