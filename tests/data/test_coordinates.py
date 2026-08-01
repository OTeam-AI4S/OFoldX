from __future__ import annotations

import copy
import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from ofoldx.data import ChainCoordinates, PolymerCoordinates, Structure


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gather_owns_values(dtype):
    xyz = np.array([[1.123456789, 2, 3], [np.nan, 9, 9]], dtype=dtype)
    expected = xyz[0].copy()
    mask = np.array([True, False])
    positions = np.array([0, 1])
    parents = [[], ["A", "G"]]
    chain = ChainCoordinates(
        polymer_type="rna",
        component_ids=("A", "XNA"),
        atom_names=("P", "P"),
        component_index=positions,
        coordinates=xyz,
        coordinate_mask=mask,
        parent_component_ids=parents,
    )
    chains = {"R": chain}
    evidence = PolymerCoordinates(chains)
    xyz[:] = 0
    mask[:] = True
    positions[:] = 1
    parents[1].clear()
    chains.clear()

    projected, present = evidence.chains["R"].gather([1, 0, 0], ["P", "C4'", None])
    assert projected.dtype == dtype
    assert present.tolist() == [[False, False, False], [True, False, False], [True, False, False]]
    np.testing.assert_array_equal(projected[1, 0], expected)
    np.testing.assert_array_equal(projected[2, 0], expected)
    assert not projected[~present].any()
    assert chain.parent_component_ids == ((), ("A", "G"))
    assert np.isnan(chain.coordinates[1, 0])
    projected[:] = 0
    present[:] = False
    np.testing.assert_array_equal(chain.gather([0], ["P"])[0][0, 0], expected)
    for array in (chain.coordinates, chain.coordinate_mask, chain.component_index):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        evidence.chains["other"] = chain
    with pytest.raises(FrozenInstanceError):
        chain.component_ids = ("G", "G")


def test_empty_gather():
    chain = ChainCoordinates(
        polymer_type="dna",
        component_ids=("DA",),
        atom_names=(),
        component_index=np.empty(0, dtype=np.int64),
        coordinates=np.empty((0, 3), dtype=np.float64),
        coordinate_mask=np.empty(0, dtype=np.bool_),
    )
    xyz, mask = chain.gather([0], ["P"])
    assert xyz.shape == (1, 1, 3) and not xyz.any() and not mask.any()
    assert chain.gather([], ["P"])[0].shape == (0, 1, 3)
    assert chain.gather([0], [])[0].shape == (1, 0, 3)
    with pytest.raises(IndexError):
        chain.gather([1], ["P"])
    with pytest.raises(TypeError):
        chain.gather([0.5], ["P"])


@pytest.mark.parametrize("warm", [False, True])
def test_pickle_owns_arrays(warm):
    positions = np.array([0])
    coordinates = np.array([[1.123456789, 2, 3]])
    observed = np.array([True])
    chain = ChainCoordinates(
        polymer_type="protein",
        component_ids=("MSE",),
        atom_names=("CA",),
        component_index=positions,
        coordinates=coordinates,
        coordinate_mask=observed,
        parent_component_ids=(("MET", "ALA"),),
    )
    evidence = PolymerCoordinates({"A": chain})
    if warm:
        chain.gather([0], ["CA"])
    snapshots = (pickle.loads(pickle.dumps(evidence)), copy.deepcopy(evidence))
    positions[:] = 7
    coordinates[:] = 9
    observed[:] = False
    np.testing.assert_array_equal(chain.coordinates, [[1.123456789, 2, 3]])
    assert chain.component_index.tolist() == [0]
    assert chain.coordinate_mask.tolist() == [True]
    for restored in snapshots:
        actual = restored.chains["A"]
        assert actual.parent_component_ids == (("MET", "ALA"),)
        for field in ("coordinates", "coordinate_mask", "component_index"):
            array, original = getattr(actual, field), getattr(chain, field)
            np.testing.assert_array_equal(array, original)
            assert array.dtype == original.dtype
            with pytest.raises(ValueError):
                array[...] = 0
            with pytest.raises(ValueError):
                array.setflags(write=True)
        xyz, mask = actual.gather([0], ["CA"])
        assert mask.tolist() == [[True]]
        xyz[:] = 0
        mask[:] = False
        np.testing.assert_array_equal(actual.coordinates, chain.coordinates)
        assert actual.coordinate_mask.tolist() == [True]
        with pytest.raises(TypeError):
            restored.chains["B"] = actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_structure_snapshot(dtype):
    source = Structure.from_sequence("AG").to(dtype=dtype)
    source.coordinates[:] = 1.125
    source.coordinate_mask[:] = True
    source.source_symbols = ("MET", "ALA", "GLY")
    source.component_parent_ptr = torch.tensor([0, 2, 3])
    source.parent_component_id = torch.tensor([0, 1, 2])
    evidence = PolymerCoordinates.from_structure(source)
    source.coordinates[:] = 9
    source.coordinate_mask[:] = False
    source.parent_component_id[:] = 2
    source.system.component_symbol[:] = int(source.system.component_symbol[1])

    chain = evidence.chains["A"]
    assert chain.coordinates.dtype == (np.float64 if dtype == torch.float64 else np.float32)
    assert chain.component_ids == ("ALA", "GLY")
    assert chain.parent_component_ids == (("MET", "ALA"), ("GLY",))
    xyz, mask = chain.gather([0, 1], ["CA"])
    assert (xyz == 1.125).all() and mask.all()


def test_rejects_batched_frames():
    source = Structure.from_sequence("A")
    with pytest.raises(ValueError, match="one molecular system"):
        PolymerCoordinates.from_structure(Structure.batch((source, source)))
