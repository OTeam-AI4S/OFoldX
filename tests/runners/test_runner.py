from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import Tensor

from ofoldx.data import (
    ChemicalAtom,
    ChemicalComponent,
    Cropper,
    MolecularSystem,
    Molecule,
    Structure,
    TokenLayout,
    save_structure,
)
from ofoldx.runners.runner import Runner

STRUCTURE = Path(__file__).resolve().parents[2] / "examples" / "data" / "tripeptide.pdb"

_ODESIGN_MODEL = {
    "architecture": "odesign",
    "single_size": 16,
    "pair_size": 8,
    "constraint_embeddings": {
        "hidden_size": 8,
        "num_blocks": 1,
        "num_distogram_bins": 4,
        "dropout": 0.0,
    },
    "trunk": {
        "num_msa_stages": 1,
        "num_pairformer_blocks": 1,
        "pairformer_dropout": 0.0,
        "msa_dropout": 0.0,
        "pair_dropout": 0.0,
    },
    "denoiser": {
        "num_token_encoder_blocks": 1,
        "num_token_encoder_attention_heads": 1,
        "atom_hidden_size": 8,
        "atom_pair_hidden_size": 4,
        "token_hidden_size": 16,
        "num_atom_encoder_blocks": 1,
        "num_atom_encoder_attention_heads": 1,
        "num_atom_decoder_blocks": 1,
        "num_atom_decoder_attention_heads": 1,
        "num_query_atoms": 4,
        "num_key_atoms": 4,
    },
    "pairwise": {
        "num_distogram_bins": 4,
        "bond_hidden_size": 8,
    },
}


def _runner(tmp_path: Path, **values: object) -> Runner:
    config = {
        "stack": "torch",
        "model": {
            "architecture": "proteinmpnn",
            "hidden_size": 8,
            "num_encoder_blocks": 1,
            "num_decoder_blocks": 1,
            "num_neighbors": 2,
            "dropout": 0.0,
        },
        "data": {},
        "steps": 1,
        "optim": {"type": "adamw"},
        "logging": {"enabled": False},
        "tensorboard": {"enabled": False},
        "ckpt": {"enabled": False},
        "workspace": {"root": tmp_path},
        **values,
    }
    return Runner(config)


def _odesign_runner(
    tmp_path: Path,
    *,
    cropper: Cropper | None = None,
    **values: object,
) -> Runner:
    return Runner(
        {
            "stack": "torch",
            "model": _ODESIGN_MODEL,
            "data": {},
            "steps": 1,
            "optim": {"type": "adamw"},
            "logging": {"enabled": False},
            "tensorboard": {"enabled": False},
            "ckpt": {"enabled": False},
            "workspace": {"root": tmp_path},
            **values,
        },
        cropper=cropper,
    )


def _first_token(
    structure: Structure,
    layout: TokenLayout,
    *,
    required_token_mask: Tensor | None = None,
    anchor_token_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    del structure
    del required_token_mask, anchor_token_mask, generator
    return layout.token_component_index.new_tensor([0])


class TestRunner:
    def test_ligand(self, tmp_path: Path) -> None:
        model = {**_ODESIGN_MODEL, "architecture": "odesign_ligand"}
        context = Structure.from_sequence("A")
        context.coordinates.zero_()
        context.coordinate_mask.fill_(True)
        context_target = Structure.from_sequence("A", instance_id="A")
        component = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset())
        ligand_system = MolecularSystem.from_entities({"ligand": Molecule((component,))}, {"L": "ligand"})
        ligand_target = Structure(  # type: ignore[call-arg]
            system=ligand_system,
            coordinates=torch.zeros(1, 3),
            coordinate_mask=torch.ones(1, dtype=torch.bool),
            occupancy=torch.zeros(1),
            occupancy_mask=torch.zeros(1, dtype=torch.bool),
            b_factor=torch.zeros(1),
            b_factor_mask=torch.zeros(1, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((1,), -1),
            insertion_code=torch.full((1,), -1),
            batch_size=[],
        )
        target = Structure.batch((context_target, ligand_target))
        target.coordinates.zero_()
        target.coordinate_mask.fill_(True)
        context_path = tmp_path / "context.structure.safetensors"
        target_path = tmp_path / "target.structure.safetensors"
        save_structure(context, context_path)
        save_structure(target, target_path)
        generated_rows = range(int(ligand_target.system.atom_symbol.numel()))
        manifest = tmp_path / "ligand.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "structure": context_path.name,
                    "target": {"path": target_path.name, "instance_ids": ["L"]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        runner = _odesign_runner(tmp_path, model=model, data={"train": str(manifest)})
        batch = runner.collate_fn([runner.datasets["train"][0]])

        input = batch["input"]["input"]
        target = batch["input"]["target"]
        scored = target.coordinate_mask & input.atom_mask & ~input.fixed_coordinate_mask
        assert int(scored.sum()) == len(generated_rows)
        assert batch["loss_normalizer"] == 1

    def test_gradient_checkpointing(self, tmp_path: Path) -> None:
        runner = _odesign_runner(
            tmp_path,
            stack="ddp",
            gradient_checkpointing=True,
            compile={
                "enabled": True,
                "scope": "regions",
                "backend": "eager",
                "fullgraph": True,
                "dynamic": True,
            },
            data={"train": str(STRUCTURE)},
            forward={"num_samples": 1, "sample_chunk_size": 1, "recycling_steps": 1},
        )

        assert runner.model.is_gradient_checkpointing
        batch = runner.to_device(runner.collate_fn([runner.datasets["train"][0]]))
        with runner.train_context():
            prediction = runner.model(**batch["input"])
            assert runner.criterion is not None
            loss = runner.criterion(prediction, batch["target"])
        loss.backward()

        gradients = [parameter.grad for parameter in runner.model.parameters() if parameter.grad is not None]
        assert torch.isfinite(loss)
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    def test_forward(self, tmp_path: Path) -> None:
        runner = _odesign_runner(
            tmp_path,
            data={"train": str(STRUCTURE)},
            forward={"num_samples": 2, "sample_chunk_size": 1, "recycling_steps": 1},
        )
        batch = runner.collate_fn([runner.datasets["train"][0]])

        assert batch["input"]["num_samples"] == 2
        assert batch["input"]["sample_chunk_size"] == 1
        assert batch["input"]["recycling_steps"] == 1

    def test_collate_context(self, tmp_path: Path) -> None:
        runner = _runner(
            tmp_path,
            data={"train": {"source": str(STRUCTURE), "processing": {"positions": []}}},
        )

        with pytest.raises(ValueError, match="no designable residue") as error:
            runner.collate_fn([runner.datasets["train"][0]])

        assert error.value.__notes__ == [f"dataset samples: index=0, structure={STRUCTURE}"]

    def test_training_cropper(self, tmp_path: Path) -> None:
        runner = _odesign_runner(
            tmp_path,
            cropper=_first_token,
            data={"train": str(STRUCTURE), "evaluate": str(STRUCTURE)},
        )

        train_input, _ = runner.datasets["train"][0][0]
        evaluate_input, _ = runner.datasets["evaluate"][0][0]

        assert train_input.token_mask.shape[-1] == 1
        assert evaluate_input.token_mask.shape[-1] == 3

    def test_sampling_conflict(self, tmp_path: Path) -> None:
        structure = tmp_path / "structure.cif"
        structure.touch()
        index = tmp_path / "sampling.jsonl"
        index.write_text(
            json.dumps({"structure": structure.name, "sample_type": "structure", "weight": 1.0}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="cannot be combined"):
            _runner(
                tmp_path,
                data={"train": {"source": str(structure), "sampling": {"index": str(index)}}},
                dataloader={"sampler": "random"},
            )
