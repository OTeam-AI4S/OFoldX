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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ofoldx import pipeline, save_model
from ofoldx.data.molecular import MolecularSystem, MonomerType
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKENS, TokenLayout
from ofoldx.errors import UnsupportedError
from ofoldx.io import save_structure
from ofoldx.models.proteinmpnn import ProteinMpnnScorePipeline
from ofoldx.models.targets import SequenceTarget
from ofoldx.provenance import Basis, Resolution, state
from ofoldx.runners.runner import Runner


@pytest.fixture
def artifact(tmp_path, peptide) -> Path:
    path = tmp_path / "tripeptide.safetensors"
    save_structure(peptide("ALA", "GLY", "LYS"), path)
    return path


def _runner(tmp_path: Path, *, cropper=None, **values: object) -> Runner:
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
        "workspace": {"root": str(tmp_path)},
        **values,
    }
    return Runner(config, cropper=cropper)


def _first_token(structure, layout: TokenLayout, *, required, generator) -> np.ndarray:
    keep = np.zeros(len(layout.token_component), dtype=bool)
    keep[0] = True
    return keep


def _odesign_model(architecture: str) -> dict:
    return {
        "architecture": architecture,
        "single_size": 16,
        "pair_size": 8,
        "constraint_embeddings": {"hidden_size": 8, "num_blocks": 1, "num_distogram_bins": 4},
        "trunk": {"num_msa_stages": 1, "num_pairformer_blocks": 1},
        "denoiser": {
            "atom_hidden_size": 8,
            "atom_pair_hidden_size": 4,
            "token_hidden_size": 16,
            "num_token_encoder_blocks": 1,
            "num_token_encoder_attention_heads": 1,
            "num_atom_encoder_blocks": 1,
            "num_atom_encoder_attention_heads": 1,
            "num_atom_decoder_blocks": 1,
            "num_atom_decoder_attention_heads": 1,
            "num_query_atoms": 4,
            "num_key_atoms": 4,
        },
        "pairwise": {"num_distogram_bins": 4, "bond_hidden_size": 8},
    }


def _required_tokens(structure, layout: TokenLayout, *, required, generator) -> np.ndarray:
    return required.copy()


class TestRunner:
    def test_exports_native_repository(self, tmp_path: Path, artifact: Path, peptide) -> None:
        runner = _runner(tmp_path / "training", data={"train": str(artifact)})
        try:
            before = {
                name: value.detach().to("cpu", copy=True)
                for name, value in runner.unwrap(runner.model).state_dict().items()
            }
            runner.train()
            trained_model = runner.unwrap(runner.model).to("cpu")
            runner.processor.to("cpu")
            trained = trained_model.state_dict()
            assert any(not torch.equal(value, before[name]) for name, value in trained.items())
            repository = tmp_path / "native"
            save_model(repository, model=trained_model, processor=runner.processor)
            loaded = pipeline("score", model=repository, device="cpu")
            assert isinstance(loaded, ProteinMpnnScorePipeline)
            for name, expected in trained.items():
                torch.testing.assert_close(loaded.model.state_dict()[name], expected)

            structure = peptide("ALA", "GLY", "LYS")
            reference = ProteinMpnnScorePipeline(model=trained_model.eval(), processor=runner.processor)
            expected = reference(structure, seed=17)
            actual = loaded(structure, seed=17)
            for name, score in expected.scores.items():
                assert actual.scores[name].value == pytest.approx(score.value)
        finally:
            runner.close()

    def test_forward(self, tmp_path: Path, artifact: Path) -> None:
        runner = _runner(tmp_path, data={"train": str(artifact)}, forward={"coordinate_noise": 0.1})

        batch = runner.collate_fn([runner.datasets["train"][0]])

        assert batch["input"]["coordinate_noise"] == 0.1

    def test_ligand_training(self, tmp_path: Path, peptide) -> None:
        context = tmp_path / "context.safetensors"
        truth = tmp_path / "truth.safetensors"
        save_structure(peptide("ALA", "GLY"), context)
        save_structure(peptide("ALA", "GLY", ligands=("ACT",)), truth)
        manifest = tmp_path / "records.jsonl"
        manifest.write_text(
            json.dumps({"structure": context.name, "target": {"path": truth.name, "instance_ids": ["B"]}}) + "\n",
            encoding="utf-8",
        )
        runner = _runner(
            tmp_path / "training",
            model=_odesign_model("odesign_ligand"),
            data={"train": str(manifest)},
            forward={"num_recycles": 1, "coordinate_augmentation": False},
            dataloader={"batch_size": 1, "num_workers": 0},
        )
        try:
            (input, target), _, _ = runner.datasets["train"][0]
            generated = input.generated_atom_mask[0]
            expected = runner.processor.prepare_input(peptide("ALA", "GLY"), generated_atoms=int(generated.sum()))
            torch.testing.assert_close(input.to_dict(), expected.features.to_dict())
            assert target.coordinate_mask[0, generated].all()
            assert generated[target.covalent_bond_index[0]].all(dim=-1).any()
            model = runner.unwrap(runner.model)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}

            runner.train()

            assert all(torch.isfinite(value).all() for value in model.parameters())
            assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters())
        finally:
            runner.close()

    def test_independent_sequence_truth(self, tmp_path: Path, peptide) -> None:
        source = peptide("ALA", "GLY", "LYS")
        truth = peptide("GLY", "LYS", "ALA")
        source_path, truth_path = tmp_path / "source.safetensors", tmp_path / "truth.safetensors"
        save_structure(source, source_path)
        save_structure(truth, truth_path)
        manifest = tmp_path / "examples.jsonl"
        manifest.write_text(
            json.dumps({"structure": source_path.name, "target": {"path": truth_path.name}}) + "\n",
            encoding="utf-8",
        )
        runner = _runner(tmp_path / "training", data={"train": str(manifest), "validation": str(manifest)}, seed=7)
        captured = []
        model = runner.unwrap(runner.model)

        def record_input(module, args, kwargs):
            actual = kwargs["input"] if "input" in kwargs else args[0]
            captured.append(
                (module.training, actual.sequence_tokens.detach().clone(), actual.backbone_coordinates.detach().clone())
            )

        hook = model.register_forward_pre_hook(record_input, with_kwargs=True)
        try:
            (features, target), _, _ = runner.datasets["train"][0]
            baseline = runner.processor.prepare_input(source)
            assert isinstance(target, SequenceTarget)
            torch.testing.assert_close(features.to_dict(), baseline.features.to_dict())
            assert not torch.equal(target.tokens, features.sequence_tokens)
            original_tokens = features.sequence_tokens.clone()
            runner.train()
            runner.evaluate()
            assert any(training for training, _tokens, _backbone in captured)
            assert any(not training for training, _tokens, _backbone in captured)
            for _training, tokens, backbone in captured:
                torch.testing.assert_close(tokens.cpu(), target.tokens.cpu())
                torch.testing.assert_close(backbone.cpu(), features.backbone_coordinates.cpu())
            torch.testing.assert_close(features.sequence_tokens, original_tokens)
        finally:
            hook.remove()
            runner.close()

    def test_known_unpredicted_sequence_truth_is_rejected(self, tmp_path: Path, peptide, polymers) -> None:
        source = peptide("ALA", "GLY", "LYS")
        arrays = polymers(("A", "P", ("ALA", "SEC", "PYL"))).build()
        arrays.columns["component_monomer_type"][:] = MonomerType.PROTEIN
        arrays.columns["component_monomer_type_state"][:] = state(Resolution.KNOWN, Basis.DEFINED)
        truth = Structure.unobserved(MolecularSystem.from_arrays(arrays))
        source_path, truth_path = tmp_path / "source.safetensors", tmp_path / "truth.safetensors"
        save_structure(source, source_path)
        save_structure(truth, truth_path)
        manifest = tmp_path / "examples.jsonl"
        manifest.write_text(
            json.dumps({"structure": source_path.name, "target": {"path": truth_path.name}}) + "\n",
            encoding="utf-8",
        )
        runner = _runner(tmp_path / "training", data={"train": str(manifest)})
        try:
            sample = runner.datasets["train"][0]
            features, target = sample[0]
            assert target.mask.all()
            assert target.tokens[0, 1:].tolist() == [TOKENS.index("U"), TOKENS.index("O")]
            batch = runner.collate_fn([sample])
            assert batch["loss_normalizer"] == 3
            with pytest.raises(UnsupportedError, match="model_capacity"):
                runner.train_step(batch)
            torch.testing.assert_close(
                features.sequence_tokens, runner.processor.prepare_input(source).features.sequence_tokens
            )
        finally:
            runner.close()

    def test_missing_generated_history_is_not_filled_from_source(self, tmp_path: Path, peptide) -> None:
        runner = _runner(tmp_path)
        try:
            prepared = runner.processor.prepare_input(peptide("ALA", "GLY", "LYS"))
            features = prepared.features
            target = SequenceTarget(features.sequence_tokens.clone(), torch.tensor([[True, False, True]]))
            batch = runner.collate_fn([((features, target), 0, "partial-sequence")])
            before = features.sequence_tokens.clone()
            with pytest.raises(ValueError, match="known truth throughout"):
                runner.train_step(batch)
            torch.testing.assert_close(features.sequence_tokens, before)
        finally:
            runner.close()

    def test_supervision_and_train_only_crop(self, tmp_path: Path, peptide) -> None:
        source = peptide("ALA", "GLY", ligands=("ACT",))
        arrays = source.build()
        arrays.columns["coordinates"] += np.array([11.0, 17.0, 23.0])
        truth = Structure.from_arrays(source.system, arrays)
        source_path, truth_path = tmp_path / "source.safetensors", tmp_path / "truth.safetensors"
        save_structure(source, source_path)
        save_structure(truth, truth_path)
        manifest = tmp_path / "examples.jsonl"
        manifest.write_text(
            json.dumps({"structure": source_path.name, "target": {"path": truth_path.name}}) + "\n",
            encoding="utf-8",
        )
        runner = _runner(
            tmp_path / "training",
            model=_odesign_model("odesign"),
            processor={"artifact": "protein_rigid"},
            seed=17,
            cropper=_required_tokens,
            data={"train": str(manifest), "validation": str(manifest)},
            dataloader={"batch_size": 1, "num_workers": 0},
        )
        try:
            inputs = {}
            for split in ("train", "validation"):
                sample = runner.datasets[split][0]
                (features, target), _, _ = sample
                baseline = runner.processor.prepare_input(
                    source,
                    cropper=_required_tokens if split == "train" else None,
                    generator=np.random.default_rng(17),
                )
                torch.testing.assert_close(features.to_dict(), baseline.features.to_dict())
                fixed = features.fixed_coordinate_mask[0]
                rows = baseline.context.slots.atom[~fixed]
                torch.testing.assert_close(
                    target.coordinates[0, ~fixed],
                    torch.as_tensor(truth.coordinates[rows], dtype=target.coordinates.dtype),
                )
                torch.testing.assert_close(
                    target.coordinates[0, fixed], features.observed_positions[0, fixed].to(target.coordinates.dtype)
                )
                collated = runner.collate_fn([sample])
                torch.testing.assert_close(collated["input"]["input"].to_dict(), baseline.features.to_dict())
                torch.testing.assert_close(collated["target"]["target"].coordinates, target.coordinates)
                inputs[split] = features
            assert inputs["train"].token_mask.shape[-1] == 2
            assert inputs["validation"].token_mask.shape[-1] > 2
        finally:
            runner.close()

    def test_collate_context(self, tmp_path: Path, artifact: Path) -> None:
        runner = _runner(tmp_path, data={"train": str(artifact)})

        with pytest.raises(TypeError) as error:
            runner.collate_fn([(object(), 0, "sample")])

        assert error.value.__notes__ == ["dataset samples: index=0, structure=sample"]

    def test_training_cropper(self, tmp_path: Path, artifact: Path) -> None:
        runner = Runner(
            {
                "stack": "torch",
                "seed": 1,
                "model": {"architecture": "proteinmpnn", "hidden_size": 8, "num_neighbors": 2},
                "data": {"train": str(artifact), "evaluate": str(artifact)},
                "logging": {"enabled": False},
                "tensorboard": {"enabled": False},
                "ckpt": {"enabled": False},
                "workspace": {"root": tmp_path},
            },
            cropper=_first_token,
        )

        train_input, _ = runner.datasets["train"][0][0]
        evaluate_input, _ = runner.datasets["evaluate"][0][0]

        assert train_input.sequence_tokens.shape[-1] == 1
        assert evaluate_input.sequence_tokens.shape[-1] == 3

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
