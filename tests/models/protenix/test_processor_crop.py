from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch

from ofoldx.data import AtomSite, MolecularSystem, PolymerSequence, Structure
from ofoldx.models.protenix import processor as processor_module
from ofoldx.models.protenix.config import ProtenixProcessorConfig
from ofoldx.models.protenix.processor import (
    ProtenixProcessor,
    _continuous_crop_indices,
    _spatial_crop_indices,
    _tokenize,
)
from ofoldx.processor import UnresolvedTrainingCropError
from ofoldx.runners.dataset import StructureDataset


@dataclass(frozen=True)
class _Index:
    type: str
    reference_instance_ids: tuple[str, ...]


def _structure() -> Structure:
    entities = {
        "1": PolymerSequence.from_one_letter("AAAAAA"),
        "2": PolymerSequence.from_one_letter("GGGGGG"),
    }
    system = MolecularSystem.from_entities(entities, {"A": "1", "B": "2"})
    coordinates = torch.zeros((system.atom_symbol.numel(), 3), dtype=torch.float32)
    for instance_index in range(system.instance_symbol.numel()):
        component_start = int(system.instance_component_ptr[instance_index])
        component_stop = int(system.instance_component_ptr[instance_index + 1])
        for residue_index, component_index in enumerate(range(component_start, component_stop)):
            atom_start = int(system.component_atom_ptr[component_index])
            atom_stop = int(system.component_atom_ptr[component_index + 1])
            coordinates[atom_start:atom_stop, 0] = residue_index * 2.0
            coordinates[atom_start:atom_stop, 1] = instance_index * 4.0
    component_count = system.component_symbol.numel()
    return Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=torch.ones(system.atom_symbol.numel(), dtype=torch.bool),
        occupancy=torch.ones(system.atom_symbol.numel()),
        occupancy_mask=torch.ones(system.atom_symbol.numel(), dtype=torch.bool),
        b_factor=torch.zeros(system.atom_symbol.numel()),
        b_factor_mask=torch.ones(system.atom_symbol.numel(), dtype=torch.bool),
        source_symbols=tuple(str(index) for index in range(component_count)),
        residue_id=torch.arange(component_count),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


def _processor() -> ProtenixProcessor:
    return ProtenixProcessor(ProtenixProcessorConfig(augment_reference_positions=False))


def test_training_crop_noop_does_not_consume_generator() -> None:
    structure = _structure()
    generator = torch.Generator().manual_seed(7)
    expected_state = generator.get_state().clone()
    model_input, target = _processor().process_training(
        structure,
        max_tokens=12,
        generator=generator,
    )
    assert torch.equal(generator.get_state(), expected_state)
    assert model_input.token_mask.shape[-1] == 12
    assert target.coordinates.shape[-2] == structure.coordinates.shape[0]


def test_unresolved_training_crop_error_is_a_value_error() -> None:
    assert issubclass(UnresolvedTrainingCropError, ValueError)


def test_continuous_training_crop_requires_one_resolved_selected_atom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = _structure()
    structure.coordinate_mask.fill_(False)
    tokens = _tokenize(structure.system, frozenset())
    selected = list(range(6))
    monkeypatch.setattr(
        processor_module,
        "_continuous_crop_indices",
        lambda system, crop_tokens, max_tokens, generator: selected,
    )

    with pytest.raises(UnresolvedTrainingCropError) as exc_info:
        _processor().process_training(
            structure,
            max_tokens=6,
            crop_method_weights=(1.0, 0.0, 0.0),
        )
    assert type(exc_info.value) is UnresolvedTrainingCropError

    resolved_atom = cast(tuple[int, ...], tokens[selected[0]]["atoms"])[0]
    structure.coordinate_mask[resolved_atom] = True
    model_input, _ = _processor().process_training(
        structure,
        max_tokens=6,
        crop_method_weights=(1.0, 0.0, 0.0),
    )
    assert model_input.token_mask.shape[-1] == 6


def test_uncropped_all_unresolved_structure_keeps_existing_behavior() -> None:
    structure = _structure()
    structure.coordinate_mask.fill_(False)

    model_input, target = _processor().process_training(
        structure,
        max_tokens=12,
    )

    assert model_input.token_mask.shape[-1] == 12
    assert target.coordinates.shape[-2] == structure.coordinates.shape[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"max_tokens": 0}, "positive integer"),
        ({"crop_method_weights": (1.0, 0.0)}, "exactly three"),
        ({"crop_method_weights": (1.0, float("nan"), 0.0)}, "finite"),
        ({"crop_method_weights": (1.0, -1.0, 1.0)}, "non-negative"),
        ({"crop_method_weights": (0.0, 0.0, 0.0)}, "positive sum"),
    ),
)
def test_training_crop_rejects_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _processor().process_training(_structure(), **kwargs)


@pytest.mark.parametrize(
    ("weights", "indices"),
    (
        ((1.0, 0.0, 0.0), None),
        ((0.0, 1.0, 0.0), _Index("chain", ("A",))),
        ((0.0, 0.0, 1.0), _Index("interface", ("A", "B"))),
    ),
)
def test_one_hot_crop_modes_are_capped_reproducible_and_keep_full_target(
    weights: tuple[float, float, float],
    indices: _Index | None,
) -> None:
    structure = _structure()
    first, first_target = _processor().process_training(
        structure,
        max_tokens=6,
        crop_method_weights=weights,
        indices=indices,
        generator=torch.Generator().manual_seed(19),
    )
    second, _ = _processor().process_training(
        structure,
        max_tokens=6,
        crop_method_weights=weights,
        indices=indices,
        generator=torch.Generator().manual_seed(19),
    )
    assert 0 < first.token_mask.shape[-1] <= 6
    assert torch.equal(first.token_component_index, second.token_component_index)
    assert first.token_index.tolist() == [list(range(first.token_mask.shape[-1]))]
    assert first_target.coordinates.shape[-2] == structure.coordinates.shape[0]
    atom_indices = first.atom_index[0]
    assert torch.equal(first_target.coordinates[0, atom_indices], structure.coordinates[atom_indices])


def test_spatial_probability_requires_indices_before_mode_sampling() -> None:
    with pytest.raises(ValueError, match="indices are required"):
        _processor().process_training(
            _structure(),
            max_tokens=6,
            crop_method_weights=(0.999, 0.001, 0.0),
            generator=torch.Generator().manual_seed(1),
        )


def test_spatial_crop_rejects_invalid_index_type() -> None:
    with pytest.raises(ValueError, match="indices.type"):
        _processor().process_training(
            _structure(),
            max_tokens=6,
            crop_method_weights=(0.0, 1.0, 0.0),
            indices=_Index("assembly", ("A",)),
        )


@pytest.mark.parametrize(
    ("weights", "indices"),
    (
        ((0.0, 1.0, 0.0), _Index("chain", ("A",))),
        ((0.0, 0.0, 1.0), _Index("interface", ("A", "B"))),
    ),
)
def test_spatial_crop_without_resolved_sampled_centres_raises_unresolved_error(
    weights: tuple[float, float, float],
    indices: _Index,
) -> None:
    structure = _structure()
    structure.coordinate_mask.fill_(False)
    if indices.type == "chain":
        other_chain_center = structure.system.atom_index(AtomSite("B", 0, "CA"))
        structure.coordinate_mask[other_chain_center] = True

    with pytest.raises(UnresolvedTrainingCropError) as exc_info:
        _processor().process_training(
            structure,
            max_tokens=6,
            crop_method_weights=weights,
            indices=indices,
        )
    assert type(exc_info.value) is UnresolvedTrainingCropError


def test_spatial_crop_reference_instances_without_model_tokens_remains_value_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        _processor().process_training(
            _structure(),
            max_tokens=6,
            crop_method_weights=(0.0, 1.0, 0.0),
            indices=_Index("chain", ("not-a-model-instance",)),
        )
    assert type(exc_info.value) is ValueError


def test_interface_crop_without_resolved_cross_chain_contact_remains_value_error() -> None:
    structure = _structure()
    structure.coordinate_mask.fill_(False)
    a_center = structure.system.atom_index(AtomSite("A", 0, "CA"))
    b_center = structure.system.atom_index(AtomSite("B", 0, "CA"))
    structure.coordinate_mask[[a_center, b_center]] = True
    structure.coordinates[b_center, 0] = structure.coordinates[a_center, 0] + 100.0

    with pytest.raises(ValueError, match="no resolved <15A cross-chain") as exc_info:
        _processor().process_training(
            structure,
            max_tokens=6,
            crop_method_weights=(0.0, 0.0, 1.0),
            indices=_Index("interface", ("A", "B")),
        )
    assert type(exc_info.value) is ValueError


def test_structure_dataset_retries_actual_unresolved_spatial_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved = _structure()
    unresolved.coordinate_mask.fill_(False)
    other_chain_center = unresolved.system.atom_index(AtomSite("B", 0, "CA"))
    unresolved.coordinate_mask[other_chain_center] = True
    resolved = _structure()
    structures = {"unresolved.cif": unresolved, "resolved.cif": resolved}
    reads: list[str] = []

    def read_structure(path: str, **kwargs: object) -> Structure:
        reads.append(path)
        return structures[path]

    monkeypatch.setattr("ofoldx.io.read_structure", read_structure)
    dataset = StructureDataset(
        None,
        _processor().process_training,
        processor_kwargs={
            "max_tokens": 6,
            "crop_method_weights": (0.0, 1.0, 0.0),
            "indices": _Index("chain", ("A",)),
        },
    )
    dataset.records = ["unresolved.cif", "resolved.cif"]

    model_input, _ = dataset[0]

    assert model_input.token_mask.shape[-1] == 6
    assert reads == ["unresolved.cif", "resolved.cif"]


def test_spatial_crop_centre_is_restricted_to_exact_sampled_instances() -> None:
    model_input, _ = _processor().process_training(
        _structure(),
        max_tokens=1,
        crop_method_weights=(0.0, 1.0, 0.0),
        indices=_Index("chain", ("A",)),
        generator=torch.Generator().manual_seed(23),
    )
    assert model_input.entity_index.tolist() == [[0]]


def test_continuous_crop_completes_multitoken_boundaries_and_excludes_single_atom_chain() -> None:
    structure = _structure()
    # Chain 0 mixes one-, two-, and three-token reference spaces (the same
    # boundary behavior used by ligands and atom-tokenized non-standard
    # residues). Chain 2 is a classified singleton one-atom chain.
    reference_spaces = (0, 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8)
    chain_indices = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2)
    tokens = [
        {
            "atoms": (index,),
            "chain_index": chain_index,
            "reference_space": reference_space,
        }
        for index, (chain_index, reference_space) in enumerate(zip(chain_indices, reference_spaces))
    ]
    by_space = {
        reference_space: {index for index, value in enumerate(reference_spaces) if value == reference_space}
        for reference_space in set(reference_spaces)
    }
    observed_split_boundary = False
    for seed in range(40):
        selected = _continuous_crop_indices(
            structure.system,
            tokens,
            6,
            torch.Generator().manual_seed(seed),
        )
        selected_set = set(selected)
        assert len(selected) <= 6
        assert 11 not in selected_set
        assert all(not (selected_set & group) or group <= selected_set for group in by_space.values())
        observed_split_boundary |= len(selected) < 6
    assert observed_split_boundary


def test_continuous_boundary_completion_recomputes_later_chain_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = _structure()
    tokens = [
        {
            "atoms": (index,),
            "chain_index": 0 if index < 4 else 1,
            "reference_space": (0, 0, 1, 1, 2, 3, 4, 5)[index],
        }
        for index in range(8)
    ]
    # Chain 0 initially samples only token 1. Completing its leading
    # two-token reference space expands that crop to [0, 1], so chain 1's
    # guaranteed minimum must be recomputed from N_added=2 (not the naive
    # pre-boundary size 1).
    random_values = iter((1, 1, 0, 2, 0))
    monkeypatch.setattr(
        processor_module,
        "_random_permutation",
        lambda length, generator: [0, 1],
    )
    monkeypatch.setattr(
        processor_module,
        "_random_integer",
        lambda high, generator, *, low=0: next(random_values),
    )
    selected = _continuous_crop_indices(
        structure.system,
        tokens,
        4,
        torch.Generator().manual_seed(0),
    )
    assert selected == [0, 1, 4, 5]


def test_spatial_crop_drops_partially_selected_reference_space() -> None:
    structure = _structure()
    tokens = _tokenize(structure.system, frozenset())
    for token in tokens[:3]:
        token["reference_space"] = 100
    selected = _spatial_crop_indices(
        structure,
        tokens,
        2,
        _Index("chain", ("A",)),
        torch.Generator().manual_seed(2),
        interface=False,
    )
    assert not set(selected) & {0, 1, 2}


def test_interface_crop_uses_only_resolved_sampled_contact_centres() -> None:
    structure = _structure()
    structure.coordinate_mask.fill_(False)
    a_center = structure.system.atom_index(AtomSite("A", 5, "CA"))
    b_center = structure.system.atom_index(AtomSite("B", 5, "CA"))
    structure.coordinate_mask[[a_center, b_center]] = True
    model_input, _ = _processor().process_training(
        structure,
        max_tokens=1,
        crop_method_weights=(0.0, 0.0, 1.0),
        indices=_Index("chain", ("A",)),
        generator=torch.Generator().manual_seed(3),
    )
    assert model_input.token_component_index.tolist() == [[5]]
