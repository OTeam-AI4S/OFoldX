from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ofoldx.data import (
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    Structure,
)
from ofoldx.results.values import (
    DesignCandidate,
    DesignResult,
    FoldPrediction,
    FoldResult,
    Metric,
    PackPrediction,
    PackResult,
    PairMetric,
    Score,
    ScoreResult,
)

RESIDUE_A = ("chain_a", 0)
RESIDUE_B = ("chain_b", 0)


def _system() -> MolecularSystem:
    residue = ChemicalComponent(
        "ALA",
        (ChemicalAtom("CA", "C"),),
        None,
    )
    ligand = ChemicalComponent(
        "LIG",
        (ChemicalAtom("C1", "C"),),
        None,
    )
    return MolecularSystem.from_entities(
        {
            "protein": PolymerSequence("protein", (residue,)),
            "ligand": Molecule((ligand,)),
        },
        {
            "chain_a": "protein",
            "chain_b": "protein",
            "ligand_1": "ligand",
        },
    )


def _structure() -> Structure:
    system = _system()
    atom_count = system.atom_symbol.numel()
    component_count = system.component_symbol.numel()
    return Structure(
        system=system,
        coordinates=torch.arange(
            atom_count * 3,
            dtype=torch.float32,
        ).reshape(atom_count, 3),
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


class TestScore:
    def test_carries_scalar_scope_and_raw_evaluations(self) -> None:
        evaluations = np.array([1.0, 2.0], dtype=np.float32)
        score = Score(
            1.5,
            unit="kcal/mol",
            preferred="lower",
            scope=frozenset({RESIDUE_A}),
            evaluations=evaluations,
        )
        assert score.value == 1.5
        assert score.unit == "kcal/mol"
        assert score.scope == frozenset({RESIDUE_A})
        assert score.evaluations is evaluations
        assert math.isnan(Score(float("nan")).value)


class TestMetrics:
    def test_axes_align_one_and_two_dimensional_values(self) -> None:
        axis = (RESIDUE_A, RESIDUE_B)
        metric_values = np.array([0.1, 0.2], dtype=np.float32)
        pair_values = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        metric = Metric(metric_values, axis, "probability")
        pair = PairMetric(pair_values, axis, axis, "Å")

        assert metric.values is metric_values
        assert metric.axis == axis
        assert pair.values is pair_values
        assert pair.row_axis == pair.column_axis == axis


class TestFoldResult:
    def test_best_uses_score_direction_and_stable_ties(self) -> None:
        structure = _structure()
        first = FoldPrediction(
            structure,
            scores={"quality": Score(1.0, preferred="higher")},
        )
        second = FoldPrediction(
            structure,
            scores={"quality": Score(2.0, preferred="higher")},
        )
        tied = FoldPrediction(
            structure,
            scores={"quality": Score(2.0, preferred="higher")},
        )

        result = FoldResult((first, second, tied))
        assert result.best("quality") is second
        assert result.predictions[0].structure is structure

        incompatible = FoldPrediction(
            structure,
            scores={"quality": Score(3.0, preferred="lower")},
        )
        with pytest.raises(ValueError):
            FoldResult((first, incompatible)).best("quality")


class TestDesignResult:
    def test_best_returns_the_original_candidate(self) -> None:
        system = _system()
        low = DesignCandidate(
            system,
            scores={"quality": Score(2.0, preferred="lower")},
        )
        high = DesignCandidate(
            system,
            scores={"quality": Score(1.0, preferred="lower")},
        )
        result = DesignResult((low, high))

        assert result.best("quality") is high
        assert result.candidates[0].system is system


class TestScoreResult:
    def test_binds_values_to_the_scored_structure(self) -> None:
        structure = _structure()
        metric = Metric(
            np.array([0.5], dtype=np.float32),
            (RESIDUE_A,),
        )
        result = ScoreResult(
            structure,
            scores={"global": Score(1.0)},
            metrics={"local": metric},
        )

        assert result.target is structure
        assert result.scores["global"].value == 1.0
        assert result.metrics["local"] is metric


class TestPackResult:
    def test_groups_generation_ordered_predictions(self) -> None:
        structure = _structure()
        metric = Metric(
            np.array([0.3], dtype=np.float32),
            (RESIDUE_A,),
            "nat/radian",
        )
        prediction = PackPrediction(structure, metrics={"chi1": metric})
        result = PackResult((prediction,))

        assert result.predictions == (prediction,)
        assert result.predictions[0].metrics["chi1"] is metric
