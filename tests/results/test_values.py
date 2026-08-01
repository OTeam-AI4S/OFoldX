from __future__ import annotations

import pytest
import torch

from ofoldx.data import ChemicalAtom, ChemicalComponent, MolecularSystem, Molecule, PolymerSequence, Structure
from ofoldx.results.values import (
    DesignAndPackResult,
    DesignCandidate,
    DesignResult,
    FoldPrediction,
    FoldResult,
    PackResult,
    Score,
)


def _system() -> MolecularSystem:
    residue = ChemicalComponent("ALA", (ChemicalAtom("CA", "C"),), None)
    ligand = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), None)
    return MolecularSystem.from_entities(
        {
            "protein": PolymerSequence("protein", (residue,)),
            "ligand": Molecule((ligand,)),
        },
        {"chain_a": "protein", "chain_b": "protein", "ligand_1": "ligand"},
    )


def _structure() -> Structure:
    system = _system()
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.arange(atom_count * 3, dtype=torch.float32).reshape(atom_count, 3),
        coordinate_mask=torch.ones(atom_count, dtype=torch.bool),
        occupancy=torch.zeros(atom_count),
        occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
        b_factor=torch.zeros(atom_count),
        b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
        source_symbols=(),
        residue_id=torch.full((component_count,), -1, dtype=torch.int64),
        insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
        batch_size=[],
    )


class TestFoldResult:
    def test_best_by(self) -> None:
        structure = _structure()
        first = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        second = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})
        tied = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="higher")})

        assert FoldResult((first, second, tied)).best_by("quality") is second

    def test_direction(self) -> None:
        structure = _structure()
        higher = FoldPrediction(structure, scores={"quality": Score(1.0, preferred="higher")})
        lower = FoldPrediction(structure, scores={"quality": Score(2.0, preferred="lower")})

        with pytest.raises(ValueError):
            FoldResult((higher, lower)).best_by("quality")


class TestDesignResult:
    def test_best_by(self) -> None:
        system = _system()
        first = DesignCandidate(system, scores={"quality": Score(2.0, preferred="lower")})
        second = DesignCandidate(system, scores={"quality": Score(1.0, preferred="lower")})

        assert DesignResult((first, second)).best_by("quality") is second

    def test_candidates(self) -> None:
        candidate = DesignCandidate(_system())
        result = DesignAndPackResult(DesignResult((candidate,)), (PackResult(()),))

        assert result.candidates == (candidate,)
