from __future__ import annotations

from pathlib import Path

import torch
from chanfig import Config

from ofoldx.data import AtomRecord, ResidueKey, Structure
from ofoldx.models import AutoModel
from ofoldx.models.ligandmpnn import LigandMPNNPacker, LigandMPNNPackerConfig
from ofoldx.pipelines import Pipeline, SideChainPackingPipeline
from ofoldx.pipelines.auto import PIPELINE_MANIFEST_NAME


def protein_backbone_structure() -> Structure:
    atoms: list[AtomRecord] = []
    for residue_number, residue_name in enumerate(("ALA", "GLY", "SER"), start=1):
        residue = ResidueKey("A", residue_number)
        x = float(residue_number) * 3.8
        atoms.extend(
            (
                AtomRecord("N", "N", residue_name, residue, x - 1.2, 0.0, 0.0),
                AtomRecord("CA", "C", residue_name, residue, x, 0.0, 0.0),
                AtomRecord("C", "C", residue_name, residue, x + 1.2, 0.0, 0.0),
                AtomRecord("O", "O", residue_name, residue, x + 1.8, 0.6, 0.0),
            )
        )
    return Structure(tuple(atoms))


def tiny_packer() -> LigandMPNNPacker:
    return LigandMPNNPacker(
        LigandMPNNPackerConfig(
            {
                "hidden_size": 16,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_rbf": 4,
                "k_neighbors": 2,
                "atom_context_num": 2,
                "num_positional_embeddings": 4,
            }
        )
    )


def test_side_chain_packing_pipeline_runs_structure_input() -> None:
    pipeline = SideChainPackingPipeline(tiny_packer())

    output = pipeline(protein_backbone_structure(), num_denoising_steps=1, num_samples=1, num_context_atoms=2)

    assert output["X"].shape == (1, 3, 14, 3)
    assert output["X_m"].shape == (1, 3, 14)
    assert output["mean"].shape[:3] == (1, 3, 4)
    assert output["mix_logits"].shape[-1] == 3


def test_side_chain_packing_pipeline_round_trips_package(tmp_path: Path) -> None:
    pipeline = SideChainPackingPipeline(tiny_packer())

    pipeline.save_pretrained(tmp_path)
    manifest = Config.from_json(tmp_path / PIPELINE_MANIFEST_NAME)
    loaded = Pipeline.from_pretrained(tmp_path)
    loaded_model = AutoModel.from_pretrained(tmp_path)
    output = loaded(protein_backbone_structure(), num_denoising_steps=1, num_samples=1, num_context_atoms=2)

    assert manifest["kind"] == "pipeline"
    assert manifest["task"] == "side_chain_packing"
    assert manifest["entrypoint"] == "ofoldx.pipelines.side_chain_packing.SideChainPackingPipeline"
    assert manifest["components"]["packer"]["path"] == "packer"
    assert isinstance(loaded, SideChainPackingPipeline)
    assert isinstance(loaded.model, LigandMPNNPacker)
    assert isinstance(loaded_model, LigandMPNNPacker)
    assert output["X"].shape[-2:] == (14, 3)


def test_side_chain_packing_pipeline_loads_bare_packer_model(tmp_path: Path) -> None:
    tiny_packer().save_pretrained(tmp_path)

    loaded = Pipeline.from_pretrained(tmp_path)
    output = loaded(protein_backbone_structure(), num_denoising_steps=1, num_samples=1, num_context_atoms=2)

    assert isinstance(loaded, SideChainPackingPipeline)
    assert torch.isfinite(output["X"]).all()
