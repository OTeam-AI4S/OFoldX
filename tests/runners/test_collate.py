from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from danling.tensors import NestedTensor, nested_execution_guard

from ofoldx.data import (
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    PolymerType,
    Structure,
)
from ofoldx.models.ligandmpnn import (
    LigandMpnnConfig,
    LigandMpnnModel,
    LigandMpnnProcessor,
    LigandMpnnProcessorConfig,
)
from ofoldx.models.ligandmpnn.objectives import criterion as ligandmpnn_criterion
from ofoldx.models.nampnn import NaMpnnConfig, NaMpnnModel, NaMpnnProcessor
from ofoldx.models.nampnn.objectives import criterion as nampnn_criterion
from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnInput, ProteinMpnnModel
from ofoldx.models.proteinmpnn.objectives import criterion as proteinmpnn_criterion
from ofoldx.runners.collate import (
    collate_boltzgen_denoising,
    collate_boltzgen_ifold,
    collate_odesign,
    collate_sequence,
)


def _strict_nested_execution():
    return nested_execution_guard(
        forbid_iteration=True,
        forbid_storage_map=True,
        forbid_eager_fallback=True,
        forbid_padded_materialization=True,
        forbid_dense_repack=True,
    )


def _assert_causal_unit_ranks(ranks: NestedTensor, group_ids: NestedTensor) -> None:
    for item_ranks, item_groups in zip(ranks, group_ids, strict=True):
        for group in item_groups.unique():
            assert item_ranks[item_groups == group].unique().numel() == 1
        assert item_ranks.unique().numel() == item_groups.unique().numel()


def _protein_input(
    length: int,
    *,
    ca_only: bool = False,
    tied_group_ids: tuple[int, ...] | None = None,
    design_positions: tuple[bool, ...] | None = None,
) -> ProteinMpnnInput:
    if tied_group_ids is None:
        tied_group_ids = tuple(range(length))
    if design_positions is None:
        design_positions = (True,) * length
    return ProteinMpnnInput(  # type: ignore[call-arg]
        backbone_coordinates=torch.randn(1, length, 3) if ca_only else torch.randn(1, length, 4, 3),
        sequence_tokens=torch.arange(length).remainder(20).unsqueeze(0),
        residue_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=torch.arange(length).unsqueeze(0),
        chain_index=torch.zeros(1, length, dtype=torch.int64),
        design_positions=torch.tensor(design_positions, dtype=torch.bool).unsqueeze(0),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).unsqueeze(0),
        tied_group_ids=torch.tensor(tied_group_ids).unsqueeze(0),
        batch_size=[],
    )


def _placed_structure(length: int, polymer_type: PolymerType = "protein") -> Structure:
    structure = Structure.from_sequence("A" * length, polymer_type=polymer_type)
    system = structure.system
    atom_component = torch.repeat_interleave(
        torch.arange(length),
        system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
    )
    coordinates = system.reference_coordinates.clone()
    coordinates[:, 0] += atom_component.to(coordinates.dtype) * 4
    structure.coordinates = coordinates
    structure.coordinate_mask = system.reference_coordinate_mask.clone()
    return structure


def _protein_ligand_structure(length: int) -> Structure:
    protein = PolymerSequence.from_one_letter("A" * length)
    ligand = Molecule((ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset()),))
    system = MolecularSystem.from_entities(
        {"protein": protein, "ligand": ligand},
        {"A": "protein", "L": "ligand"},
    )
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.zeros(atom_count, 3),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1),
        insertion_code=torch.full((component_count,), -1),
        batch_size=[],
    )


def test_collate_sequence_ragged() -> None:
    batch = collate_sequence([_protein_input(2), _protein_input(3)])

    input = batch["input"]
    assert all(
        isinstance(value, NestedTensor) for name, value in input.items() if name != "structure" and value is not None
    )
    assert input.sequence_tokens[0].shape == (2,)
    assert input.sequence_tokens[1].shape == (3,)
    assert torch.equal(input.sequence_tokens.element_sizes(), torch.tensor([[2], [3]]))
    assert torch.equal(input.backbone_coordinates.element_sizes(), torch.tensor([[2, 4, 3], [3, 4, 3]]))
    assert input.sequence_tokens.ragged_dims == (0,)
    assert input.backbone_coordinates.ragged_dims == (0,)
    assert input.component_index[1].tolist() == [2, 3, 4]
    assert batch["target"].sequence_tokens.tolist() == input.sequence_tokens.tolist()
    assert batch["loss_normalizer"] == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
@pytest.mark.parametrize("ca_only", [False, True])
def test_protein_mpnn_ragged_training_without_materialization(ca_only: bool) -> None:
    inputs = (
        _protein_input(
            3,
            ca_only=ca_only,
            tied_group_ids=(0, 1, 0),
            design_positions=(True, True, False),
        ).to("cuda"),
        _protein_input(
            5,
            ca_only=ca_only,
            tied_group_ids=(0, 1, 2, 1, 3),
            design_positions=(True, True, True, False, True),
        ).to("cuda"),
    )
    model = ProteinMpnnModel(
        ProteinMpnnConfig(
            {
                "hidden_size": 8,
                "num_encoder_blocks": 1,
                "num_decoder_blocks": 1,
                "num_neighbors": 4,
                "dropout": 0.0,
                "ca_only": ca_only,
            }
        )
    ).to("cuda")
    dense_outputs = tuple(model(input, decoding_ranks=input.tied_group_ids) for input in inputs)
    dense_loss = sum(
        proteinmpnn_criterion(output, input) * (input.residue_mask & input.design_positions).sum()
        for output, input in zip(dense_outputs, inputs, strict=True)
    ) / sum((input.residue_mask & input.design_positions).sum() for input in inputs)

    nested_input = collate_sequence(list(inputs))["input"]
    with _strict_nested_execution():
        nested_output = model(nested_input, decoding_ranks=nested_input.tied_group_ids)
        nested_loss = proteinmpnn_criterion(nested_output, nested_input)
        sampled_order_output = model(
            nested_input,
            coordinate_noise=0.05,
            generator=torch.Generator(device="cuda").manual_seed(20260830),
        )
        sampled_order_loss = proteinmpnn_criterion(sampled_order_output, nested_input)
        sampled_order_loss.backward()

    assert isinstance(nested_output.logits, NestedTensor)
    assert isinstance(nested_output.decoding_ranks, NestedTensor)
    assert torch.equal(nested_output.logits.element_sizes(), torch.tensor([[3, 21], [5, 21]]))
    for index, dense_output in enumerate(dense_outputs):
        torch.testing.assert_close(nested_output.logits[index], dense_output.logits[0])
        assert torch.equal(nested_output.decoding_ranks[index], dense_output.decoding_ranks[0])
    torch.testing.assert_close(nested_loss, dense_loss)
    assert isinstance(sampled_order_output.decoding_ranks, NestedTensor)
    _assert_causal_unit_ranks(sampled_order_output.decoding_ranks, nested_input.tied_group_ids)
    assert torch.isfinite(sampled_order_loss)
    assert model.feature_encoder.edge_projection.weight.grad is not None
    assert torch.isfinite(model.feature_encoder.edge_projection.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
def test_ligand_mpnn_ragged_training_without_materialization() -> None:
    processor = LigandMpnnProcessor(LigandMpnnProcessorConfig({"num_context_atoms": 1}))
    inputs = tuple(processor(_protein_ligand_structure(length)).to("cuda") for length in (3, 5))
    inputs[0].tied_group_ids.copy_(torch.tensor([[0, 1, 0]], device="cuda"))
    inputs[1].tied_group_ids.copy_(torch.tensor([[0, 1, 2, 1, 3]], device="cuda"))
    model = LigandMpnnModel(
        LigandMpnnConfig(
            {
                "hidden_size": 8,
                "num_encoder_blocks": 1,
                "num_decoder_blocks": 1,
                "num_neighbors": 4,
                "dropout": 0.0,
            }
        )
    ).to("cuda")
    dense_outputs = tuple(model(input, decoding_ranks=input.tied_group_ids) for input in inputs)
    dense_loss = sum(
        ligandmpnn_criterion(output, input) * (input.residue_mask & input.design_positions).sum()
        for output, input in zip(dense_outputs, inputs, strict=True)
    ) / sum((input.residue_mask & input.design_positions).sum() for input in inputs)

    nested_input = collate_sequence(list(inputs))["input"]
    with _strict_nested_execution():
        nested_output = model(nested_input, decoding_ranks=nested_input.tied_group_ids)
        nested_loss = ligandmpnn_criterion(nested_output, nested_input)
        sampled_order_output = model(
            nested_input,
            coordinate_noise=0.05,
            generator=torch.Generator(device="cuda").manual_seed(20260830),
        )
        sampled_order_loss = ligandmpnn_criterion(sampled_order_output, nested_input)
        sampled_order_loss.backward()

    assert isinstance(nested_output.logits, NestedTensor)
    for index, dense_output in enumerate(dense_outputs):
        torch.testing.assert_close(nested_output.logits[index], dense_output.logits[0])
    torch.testing.assert_close(nested_loss, dense_loss)
    assert isinstance(sampled_order_output.decoding_ranks, NestedTensor)
    _assert_causal_unit_ranks(sampled_order_output.decoding_ranks, nested_input.tied_group_ids)
    assert torch.isfinite(sampled_order_loss)
    assert model.edge_embedding.weight.grad is not None
    assert torch.isfinite(model.edge_embedding.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OOps masked_knn requires CUDA")
def test_na_mpnn_ragged_training_without_materialization() -> None:
    processor = NaMpnnProcessor()
    inputs = tuple(processor(_placed_structure(length, "rna")).to("cuda") for length in (3, 5))
    inputs[0].token_group_ids.copy_(torch.tensor([[0, 1, 0]], device="cuda"))
    inputs[1].token_group_ids.copy_(torch.tensor([[0, 1, 2, 1, 3]], device="cuda"))
    model = NaMpnnModel(
        NaMpnnConfig(
            {
                "hidden_size": 8,
                "feature_size": 8,
                "num_encoder_blocks": 1,
                "num_decoder_blocks": 1,
                "num_neighbors": 4,
                "dropout": 0.0,
                "num_positional_embeddings": 4,
                "num_rbf": 2,
                "include_predicted_na_n": False,
            }
        )
    ).to("cuda")
    dense_outputs = tuple(model(input, decoding_ranks=input.token_group_ids) for input in inputs)
    dense_loss = sum(
        nampnn_criterion(output, input) * (input.residue_mask & input.design_positions).sum()
        for output, input in zip(dense_outputs, inputs, strict=True)
    ) / sum((input.residue_mask & input.design_positions).sum() for input in inputs)

    nested_input = collate_sequence(list(inputs))["input"]
    with _strict_nested_execution():
        nested_output = model(nested_input, decoding_ranks=nested_input.token_group_ids)
        nested_loss = nampnn_criterion(nested_output, nested_input)
        sampled_order_output = model(
            nested_input,
            protein_coordinate_noise=0.01,
            dna_coordinate_noise=0.02,
            rna_coordinate_noise=0.05,
            generator=torch.Generator(device="cuda").manual_seed(20260830),
        )
        sampled_order_loss = nampnn_criterion(sampled_order_output, nested_input)
        sampled_order_loss.backward()

    assert isinstance(nested_output.logits, NestedTensor)
    for index, dense_output in enumerate(dense_outputs):
        torch.testing.assert_close(nested_output.logits[index], dense_output.logits[0])
    torch.testing.assert_close(nested_loss, dense_loss)
    assert isinstance(sampled_order_output.decoding_ranks, NestedTensor)
    _assert_causal_unit_ranks(sampled_order_output.decoding_ranks, nested_input.token_group_ids)
    assert torch.isfinite(sampled_order_loss)
    assert model.edge_embedding.weight.grad is not None
    assert torch.isfinite(model.edge_embedding.weight.grad).all()


def test_collate_sequence_preserves_single_input() -> None:
    input = _protein_input(2)

    batch = collate_sequence([input])

    assert batch["input"] is input


def test_collate_boltzgen_denoising_mask() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True, True, True]]),
        atom_index=torch.full((1, 4), -1),
    )
    target = SimpleNamespace(
        coordinate_mask=torch.tensor([[True, False, True, True]]),
        coordinate_atom_index=torch.tensor([[2, 0, -1, 1]]),
    )

    batch = collate_boltzgen_denoising([(input, target)])

    assert torch.equal(batch["target"], torch.tensor([[True, True, False, False]]))
    assert batch["loss_normalizer"] == 1


def test_collate_boltzgen_ifold_mask() -> None:
    input = SimpleNamespace(
        token_mask=torch.tensor([[True, True, True, False]]),
        token_resolved_mask=torch.tensor([[True, False, True, True]]),
        design_mask=torch.tensor([[True, True, False, True]]),
    )
    residue_type = torch.zeros(1, 4, 33)

    batch = collate_boltzgen_ifold([(input, residue_type)])

    assert torch.equal(batch["target"]["design_mask"], torch.tensor([[True, False, False, False]]))
    assert batch["loss_normalizer"] == 1


def test_collate_odesign_mask() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True, True, False]]),
        fixed_coordinate_mask=torch.tensor([[True, False, False, False]]),
    )
    target = SimpleNamespace(coordinate_mask=torch.tensor([[True, True, False, True]]))

    batch = collate_odesign([(input, target)])

    assert batch["loss_normalizer"] == 1


def test_collate_odesign_empty() -> None:
    input = SimpleNamespace(
        atom_mask=torch.tensor([[True, True]]),
        fixed_coordinate_mask=torch.tensor([[True, False]]),
    )
    target = SimpleNamespace(coordinate_mask=torch.tensor([[True, False]]))

    with pytest.raises(ValueError, match="no resolved generated atom"):
        collate_odesign([(input, target)])
