from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.data import AtomRecord, ResidueKey, Structure
from ofoldx.models.alphafold2 import AlphaFold2Input
from ofoldx.models.alphafold2.processor import AlphaFold2Processor
from ofoldx.processors import AutoProcessor


def _protein_structure() -> Structure:
    residue = ResidueKey("A", 1)
    return Structure(
        (
            AtomRecord("N", "N", "ALA", residue, -1.2, 0.0, 0.0, mol_type="protein"),
            AtomRecord("CA", "C", "ALA", residue, 0.0, 0.0, 0.0, mol_type="protein"),
            AtomRecord("C", "C", "ALA", residue, 1.2, 0.0, 0.0, mol_type="protein"),
        )
    )


def test_alphafold2_processor_returns_typed_input_for_protein_structure() -> None:
    batch = AlphaFold2Processor()(_protein_structure())

    assert isinstance(batch, AlphaFold2Input)
    assert batch.token_mask.shape == (1, 1)
    assert batch.atom_mask.shape == (1, 3)
    assert batch.token_features.shape[-1] == 449
    assert batch.msa_feat is not None
    assert batch.msa_feat.shape == (1, 1, 1, 49)
    assert batch.extra_msa is not None
    assert batch.extra_msa.shape == (1, 1, 1)
    assert batch.extra_msa_mask is not None
    assert batch.extra_msa_mask.sum().item() == 0.0
    assert batch.template_aatype is not None
    assert batch.template_aatype.shape == (1, 1, 1)
    assert batch.template_mask is not None
    assert batch.template_mask.sum().item() == 0.0


def test_alphafold2_processor_returns_typed_input_for_sequence() -> None:
    batch = AlphaFold2Processor()("ACD")

    assert isinstance(batch, AlphaFold2Input)
    assert batch.token_mask.shape == (1, 3)
    assert batch.atom_mask.shape == (1, 111)
    assert batch.msa_feat is not None
    assert batch.msa_feat.shape == (1, 1, 3, 49)
    assert batch.target_feat is not None
    assert batch.target_feat.shape == (1, 3, 22)
    assert batch.target_feat[0, :, 0].tolist() == [0.0, 0.0, 0.0]
    assert batch.target_feat[0, 0, 1].item() == 1.0
    assert batch.extra_msa is not None
    assert batch.extra_msa.shape == (1, 1, 3)
    assert batch.extra_msa_mask is not None
    assert batch.extra_msa_mask.sum().item() == 0.0
    assert batch.template_aatype is not None
    assert batch.template_aatype.shape == (1, 1, 3)
    assert batch.template_mask is not None
    assert batch.template_mask.sum().item() == 0.0


def test_alphafold2_processor_reads_fasta_sequence(tmp_path: Path) -> None:
    fasta = tmp_path / "query.fasta"
    fasta.write_text(">query\nAG\n")

    batch = AlphaFold2Processor()(fasta)

    assert batch.token_mask.shape == (1, 2)
    assert batch.aatype is not None
    assert batch.aatype.tolist() == [[0, 7]]


def test_alphafold2_processor_reads_a3m_msa(tmp_path: Path) -> None:
    a3m = tmp_path / "query.a3m"
    a3m.write_text(">query\nACD\n>hit\nACcD\n")

    batch = AlphaFold2Processor()(a3m)

    assert batch.token_mask.shape == (1, 3)
    assert batch.msa_feat is not None
    assert batch.msa_feat.shape == (1, 2, 3, 49)
    assert batch.msa_feat[0, 1, 2, 23].item() == 1
    assert batch.extra_msa is not None
    assert batch.extra_msa.shape == (1, 1, 3)


def test_alphafold2_processor_rejects_non_protein_by_default() -> None:
    residue = ResidueKey("L", 1)
    ligand = Structure((AtomRecord("C1", "C", "LIG", residue, 0.0, 0.0, 0.0, hetero=True),))

    with pytest.raises(ValueError, match="protein-only by default"):
        AlphaFold2Processor()(ligand)


def test_alphafold2_processor_can_opt_into_shared_non_protein_scaffold() -> None:
    residue = ResidueKey("L", 1)
    ligand = Structure((AtomRecord("C1", "C", "LIG", residue, 0.0, 0.0, 0.0, hetero=True),))

    batch = AlphaFold2Processor(protein_only=False)(ligand)

    assert isinstance(batch, AlphaFold2Input)
    assert batch.atom_mask.shape == (1, 1)


def test_alphafold2_processor_save_load_preserves_protein_only(tmp_path: Path) -> None:
    AlphaFold2Processor(protein_only=False, single_size=4, token_feature_size=8, atom_feature_size=5).save_pretrained(
        tmp_path
    )

    loaded = AutoProcessor.from_pretrained(tmp_path)

    assert isinstance(loaded, AlphaFold2Processor)
    assert loaded.protein_only is False
    assert loaded.single_size == 4
