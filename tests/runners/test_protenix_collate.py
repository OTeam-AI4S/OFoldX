from __future__ import annotations

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard

from ofoldx.data import MolecularSystem, PolymerSequence, Structure
from ofoldx.models.protenix import (
    ProtenixConfig,
    ProtenixInput,
    ProtenixModel,
    ProtenixProcessor,
    ProtenixStructureTarget,
)
from ofoldx.modules.diffusion.loss import edm_denoising_loss
from ofoldx.runners.collate import collate_protenix_denoising, collate_protenix_inputs


def _training_pair(length: int) -> tuple[ProtenixInput, ProtenixStructureTarget]:
    system = MolecularSystem.from_entities(
        {"protein": PolymerSequence.from_one_letter("A" * length)},
        {"A": "protein"},
    )
    atoms_per_token = system.component_atom_ptr[1:] - system.component_atom_ptr[:-1]
    atom_to_token = torch.repeat_interleave(torch.arange(length), atoms_per_token)
    atoms = atom_to_token.numel()
    token_index = torch.arange(length).unsqueeze(0)
    atom_index = torch.arange(atoms).unsqueeze(0)
    zeros_token = torch.zeros(1, length, dtype=torch.long)
    zeros_atom = torch.zeros(1, atoms, dtype=torch.long)
    input = ProtenixInput(  # type: ignore[call-arg]
        token_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=token_index,
        chain_index=zeros_token,
        entity_index=zeros_token,
        symmetry_index=zeros_token,
        token_index=token_index,
        atom_features=torch.zeros(1, atoms, 1),
        atom_to_token=atom_to_token.unsqueeze(0),
        atom_mask=torch.ones(1, atoms, dtype=torch.bool),
        reference_positions=system.reference_coordinates.unsqueeze(0),
        reference_space_id=zeros_atom,
        atom_token_index=zeros_atom,
        token_to_rep_atom=system.component_atom_ptr[:-1].unsqueeze(0),
        token_features=torch.zeros(1, length, 1),
        reference_mask=system.reference_coordinate_mask.unsqueeze(0),
        reference_charges=torch.zeros(1, atoms, 1),
        token_bonds=torch.zeros(1, length, length, dtype=torch.bool),
        msa_tokens=torch.zeros(1, 1, length, dtype=torch.long),
        msa_deletion_counts=torch.zeros(1, 1, length),
        esm_tokens=zeros_token.unsqueeze(1),
        esm_mask=torch.ones(1, 1, length, dtype=torch.bool),
        token_to_esm_residue=token_index,
        token_has_esm=torch.ones(1, length, dtype=torch.bool),
        pocket=torch.zeros(1, length, length, 1),
        token_contact=torch.zeros(1, length, length, 2),
        atom_contact=torch.zeros(1, length, length, 2),
        substructure=torch.zeros(1, length, length, 4),
        structure_rep_positions=torch.zeros(1, length, 3),
        structure_rep_mask=torch.zeros(1, length, dtype=torch.bool),
        template_features=None,
        tfg_features=None,
        system=system,
        atom_index=atom_index,
        token_component_index=token_index,
        token_atom_index=torch.full((1, length), -1),
        batch_size=[],
    )
    target = ProtenixStructureTarget(  # type: ignore[call-arg]
        coordinates=system.reference_coordinates.unsqueeze(0),
        coordinate_mask=system.reference_coordinate_mask.unsqueeze(0),
        resolution=torch.tensor([2.0]),
        resolution_mask=torch.ones(1, dtype=torch.bool),
        batch_size=[],
    )
    return input, target


def _strict_nested_execution():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def _resolved_structure(length: int) -> Structure:
    structure = Structure.from_sequence("A" * length)
    structure.coordinates = structure.system.reference_coordinates.clone()
    structure.coordinate_mask = structure.system.reference_coordinate_mask.clone()
    return structure


def _model() -> ProtenixModel:
    return ProtenixModel(
        ProtenixConfig(
            {
                "single_size": 8,
                "pair_size": 8,
                "trunk": {
                    "msa_size": 8,
                    "num_msa_stages": 1,
                    "msa_outer_product_hidden_size": 4,
                    "num_msa_attention_heads": 1,
                    "msa_triangle_multiplication_hidden_size": 4,
                    "msa_triangle_attention_hidden_size": 4,
                    "num_msa_triangle_attention_heads": 1,
                    "msa_expansion": 2,
                    "num_pairformer_blocks": 1,
                    "pairformer_triangle_multiplication_hidden_size": 4,
                    "pairformer_triangle_attention_hidden_size": 4,
                    "num_pairformer_triangle_attention_heads": 1,
                    "num_pair_attention_heads": 1,
                    "pairformer_expansion": 2,
                    "pairformer_dropout": 0.0,
                    "msa_dropout": 0.0,
                    "pair_dropout": 0.0,
                },
                "recycling_steps": 1,
                "denoiser": {
                    "num_token_encoder_blocks": 1,
                    "num_token_encoder_attention_heads": 1,
                    "atom_hidden_size": 8,
                    "atom_pair_hidden_size": 4,
                    "token_hidden_size": 8,
                    "num_atom_encoder_blocks": 1,
                    "num_atom_encoder_attention_heads": 1,
                    "num_atom_decoder_blocks": 1,
                    "num_atom_decoder_attention_heads": 1,
                    "num_query_atoms": 4,
                    "num_key_atoms": 4,
                },
                "distogram": {"num_bins": 4},
                "confidence": {
                    "num_plddt_bins": 4,
                    "num_pae_bins": 4,
                    "num_pde_bins": 4,
                    "max_atoms_per_token": 8,
                    "num_blocks": 1,
                    "num_pair_attention_heads": 1,
                    "triangle_multiplication_hidden_size": 4,
                    "triangle_attention_hidden_size": 4,
                    "num_triangle_attention_heads": 1,
                },
            }
        )
    )


def test_collate_protenix_denoising_ragged_axes() -> None:
    batch = collate_protenix_denoising([_training_pair(2), _training_pair(3)])
    input = batch["input"]["input"]
    target = batch["input"]["target"]

    assert isinstance(input.token_mask, NestedTensor)
    assert input.token_mask.element_sizes().tolist() == [[2], [3]]
    assert input.token_bonds.ragged_dims == (0, 1)
    assert input.msa_tokens.ragged_dims == (0, 1)
    assert input.atom_index[1, 0] == 0
    assert target.coordinates.ragged_dims == (0,)
    assert batch["loss_normalizer"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps atom kernels require CUDA")
def test_protenix_ragged_denoise() -> None:
    processor = ProtenixProcessor()
    inputs = [
        processor(PolymerSequence.from_one_letter("A" * length)).to("cuda")  # type: ignore[attr-defined]
        for length in (2, 3)
    ]
    input = collate_protenix_inputs(inputs)
    model = _model().to("cuda")
    model.set_backend("oops")
    noisy_coordinates = NestedTensor(
        [torch.zeros(1, value.atom_features.shape[1], 3, device="cuda") for value in inputs],
        ragged_dims=(1,),
    )

    output = model.denoise(
        input,
        noisy_coordinates=noisy_coordinates,
        noise_level=torch.ones(2, 1, device="cuda"),
        recycling_steps=1,
    )

    assert isinstance(output.coordinates, NestedTensor)
    assert output.coordinates.element_sizes()[:, 0].tolist() == [1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps atom kernels require CUDA")
def test_protenix_ragged_training() -> None:
    processor = ProtenixProcessor()
    batch = collate_protenix_denoising([processor.process_training(_resolved_structure(length)) for length in (2, 3)])
    nested_input = {key: value.to("cuda") for key, value in batch["input"].items()}
    resolved = batch["target"].to("cuda")
    model = _model().to("cuda")
    model.set_backend("oops")
    model.gradient_checkpointing_enable()

    with _strict_nested_execution():
        output = model(
            **nested_input,
            num_samples=2,
            sample_chunk_size=1,
            compute_auxiliary_heads=False,
            coordinate_augmentation=False,
        )
        loss = edm_denoising_loss(output, resolved, sigma_data=float(model.config.denoiser.sigma_data))
        loss.backward()

    assert isinstance(output.coordinates, NestedTensor)
    assert output.coordinates.element_sizes()[:, 0].tolist() == [2, 2]
    assert torch.isfinite(loss)
    assert model.coordinate_denoiser.atom_encoder.noisy_position_projection.weight.grad is not None
