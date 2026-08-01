from __future__ import annotations

from functools import partial

import pytest
import torch

from ofoldx.data import (
    ChemicalAtom,
    ChemicalComponent,
    Cropper,
    MolecularSystem,
    Molecule,
    Structure,
    TokenLayout,
    continuous_crop,
    crop_token_layout,
    instance_anchored_crop,
    interface_anchored_crop,
    interface_crop,
    resolved_token_mask,
    spatial_crop,
    tokenize_molecular_system,
)


def _structure(
    coordinates: torch.Tensor,
    *,
    instances: int = 1,
    coordinate_mask: torch.Tensor | None = None,
    component_sizes: tuple[int, ...] | None = None,
) -> tuple[Structure, TokenLayout]:
    atoms_per_instance = coordinates.shape[0] // instances
    component_sizes = (1,) * atoms_per_instance if component_sizes is None else component_sizes
    components = tuple(
        ChemicalComponent(
            f"L{component_index}",
            tuple(ChemicalAtom(f"C{atom_index}", "C") for atom_index in range(size)),
            frozenset(),
        )
        for component_index, size in enumerate(component_sizes)
    )
    system = MolecularSystem.from_entities(
        {"ligand": Molecule(components)},
        {chr(ord("A") + index): "ligand" for index in range(instances)},
    )
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    structure = Structure(  # type: ignore[call-arg]
        system=system,
        coordinates=coordinates,
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool) if coordinate_mask is None else coordinate_mask,
        occupancy=torch.ones(atom_count),
        occupancy_mask=torch.ones(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.ones(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )
    return structure, tokenize_molecular_system(system)


def test_custom_cropper() -> None:
    structure, layout = _structure(torch.zeros(4, 3))

    def reverse_end(
        structure: Structure,
        layout: TokenLayout,
        *,
        required_token_mask: torch.Tensor | None = None,
        anchor_token_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del structure, required_token_mask, anchor_token_mask, generator
        return torch.arange(layout.token_component_index.numel() - 1, 1, -1).sort().values

    cropped = crop_token_layout(structure, layout, reverse_end)

    assert cropped.token_component_index.tolist() == [2, 3]
    assert cropped.atom_index.tolist() == [2, 3]


def test_instance_anchored_crop() -> None:
    structure, layout = _structure(torch.zeros(4, 3), instances=2)
    processor_anchor = torch.tensor([False, True, True, False])

    def cropper(
        structure: Structure,
        layout: TokenLayout,
        *,
        required_token_mask: torch.Tensor | None = None,
        anchor_token_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del structure, layout, required_token_mask, generator
        assert anchor_token_mask is not None
        return torch.nonzero(anchor_token_mask, as_tuple=False).flatten()

    selection = instance_anchored_crop(
        structure,
        layout,
        cropper=cropper,
        instance_ids=("B",),
        anchor_token_mask=processor_anchor,
    )

    assert selection.tolist() == [2]


def test_interface_anchored_crop() -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
        ]
    )
    structure, layout = _structure(coordinates, instances=3)
    selection = interface_anchored_crop(
        structure,
        layout,
        cropper=partial(continuous_crop, max_tokens=2),
        instance_ids=("A", "B"),
        interface_cutoff=2.0,
    )

    assert set(layout.token_instance_index[selection].tolist()) == {0, 1}
    with pytest.raises(ValueError, match="has no resolved contact"):
        interface_anchored_crop(
            structure,
            layout,
            cropper=partial(continuous_crop, max_tokens=2),
            instance_ids=("A", "C"),
            interface_cutoff=2.0,
        )


@pytest.mark.parametrize(
    "cropper",
    [
        pytest.param(partial(continuous_crop, max_tokens=3), id="continuous"),
        pytest.param(partial(spatial_crop, max_tokens=3), id="spatial"),
        pytest.param(partial(interface_crop, max_tokens=4, interface_cutoff=2.0), id="interface"),
    ],
)
def test_cropper_component_integrity(cropper: Cropper) -> None:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.5, 0.0],
            [200.0, 0.0, 0.0],
            [200.0, 0.5, 0.0],
        ]
    )
    structure, layout = _structure(coordinates, instances=2, component_sizes=(2, 2))

    selection = cropper(structure, layout, generator=torch.Generator().manual_seed(0))

    selected_components = layout.token_component_index[selection]
    for component in selected_components.unique():
        assert int((selected_components == component).sum()) == int((layout.token_component_index == component).sum())


class TestContinuousCrop:
    def test_skips_oversized_group(self) -> None:
        coordinates = torch.arange(4, dtype=torch.float32).unsqueeze(-1).expand(-1, 3)
        structure, layout = _structure(coordinates, component_sizes=(1, 2, 1))
        required = torch.tensor([True, False, False, False])

        selection = continuous_crop(
            structure,
            layout,
            max_tokens=2,
            required_token_mask=required,
            generator=torch.Generator().manual_seed(0),
        )

        assert selection.tolist() == [0, 3]

    @pytest.mark.parametrize(
        ("max_tokens", "required", "anchor", "message"),
        [
            (1, [True, False, False, False, False], None, "required component groups do not fit"),
            (
                3,
                [True, False, False, False, False],
                [False, False, True, False, False],
                "no complete anchor component group fits",
            ),
        ],
    )
    def test_group_capacity(
        self,
        max_tokens: int,
        required: list[bool],
        anchor: list[bool] | None,
        message: str,
    ) -> None:
        structure, layout = _structure(torch.zeros(5, 3), component_sizes=(2, 2, 1))

        with pytest.raises(ValueError, match=message):
            continuous_crop(
                structure,
                layout,
                max_tokens=max_tokens,
                required_token_mask=torch.tensor(required),
                anchor_token_mask=None if anchor is None else torch.tensor(anchor),
            )

    def test_required_and_resolved(self) -> None:
        coordinate_mask = torch.zeros(7, dtype=torch.bool)
        coordinate_mask[6] = True
        structure, layout = _structure(
            torch.arange(7, dtype=torch.float32).unsqueeze(-1).expand(-1, 3),
            coordinate_mask=coordinate_mask,
        )
        required = torch.tensor([True, False, False, False, False, False, False])

        selection = continuous_crop(
            structure,
            layout,
            max_tokens=4,
            required_token_mask=required,
            anchor_token_mask=resolved_token_mask(structure, layout),
            generator=torch.Generator().manual_seed(0),
        )

        assert selection.tolist() == [0, 1, 5, 6]
        assert bool(required[selection].any())
        assert bool(structure.coordinate_mask[layout.token_center_atom_index[selection]].any())

    def test_without_coordinates(self) -> None:
        structure, layout = _structure(
            torch.zeros(4, 3),
            coordinate_mask=torch.zeros(4, dtype=torch.bool),
        )

        selection = continuous_crop(
            structure,
            layout,
            max_tokens=2,
            generator=torch.Generator().manual_seed(0),
        )

        assert selection.shape == (2,)


class TestSpatialCrop:
    def test_reproducible(self) -> None:
        coordinates = torch.arange(7, dtype=torch.float32).unsqueeze(-1) * torch.tensor([1.0, 0.0, 0.0])
        structure, layout = _structure(coordinates)
        required = torch.tensor([False, False, False, True, False, False, False])
        cropper: Cropper = partial(spatial_crop, max_tokens=3)

        first = cropper(
            structure,
            layout,
            required_token_mask=required,
            generator=torch.Generator().manual_seed(7),
        )
        second = cropper(
            structure,
            layout,
            required_token_mask=required,
            generator=torch.Generator().manual_seed(7),
        )

        assert first.tolist() == [2, 3, 4]
        assert torch.equal(first, second)
        assert first.dtype == torch.int64

    @pytest.mark.parametrize("seed", range(8))
    def test_resolved_base(self, seed: int) -> None:
        structure, layout = _structure(
            torch.zeros(2, 3),
            coordinate_mask=torch.tensor([False, True]),
        )

        selection = spatial_crop(
            structure,
            layout,
            max_tokens=1,
            generator=torch.Generator().manual_seed(seed),
        )
        assert selection.tolist() == [1]


class TestInterfaceCrop:
    def test_contact_sides(self) -> None:
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [110.0, 0.0, 0.0],
                [20.0, 1.0, 0.0],
            ]
        )
        structure, layout = _structure(coordinates, instances=2)
        required = torch.tensor([True, False, False, False, False, False])
        anchor = torch.tensor([False, True, False, False, False, False])

        selection = interface_crop(
            structure,
            layout,
            max_tokens=4,
            interface_cutoff=2.0,
            required_token_mask=required,
            anchor_token_mask=anchor,
            generator=torch.Generator().manual_seed(0),
        )

        assert selection.tolist() == sorted(set(selection.tolist()))
        assert {2, 5}.issubset(selection.tolist())
        assert bool(required[selection].any())
        assert bool(anchor[selection].any())

    def test_contact_capacity(self) -> None:
        coordinates = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [110.0, 0.0, 0.0],
                [20.0, 1.0, 0.0],
            ]
        )
        structure, layout = _structure(coordinates, instances=2)

        selection = interface_crop(
            structure,
            layout,
            max_tokens=2,
            interface_cutoff=2.0,
            anchor_token_mask=resolved_token_mask(structure, layout),
            generator=torch.Generator().manual_seed(1),
        )

        assert selection.tolist() == [2, 5]
