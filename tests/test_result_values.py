r"""Tests for the frozen Section 5 scientific result carriers in values.py."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

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
    r"""A minimal two-chain, one-ligand coordinate-free system."""

    alanine = ChemicalComponent("ALA", atoms=(ChemicalAtom("CA", "C"),), bonds=None)
    protein = PolymerSequence("protein", (alanine,))
    ligand = Molecule((ChemicalComponent("LIG", atoms=(ChemicalAtom("C1", "C"),), bonds=None),))
    return MolecularSystem(
        entities={"protein": protein, "ligand": ligand},
        instances={"chain_a": "protein", "chain_b": "protein", "ligand_1": "ligand"},
    )


def _structure(system: MolecularSystem | None = None) -> Structure:
    r"""A coordinate realization of ``system`` over its realized atoms."""

    system = _system() if system is None else system
    sites = tuple(site for site, _ in system.iter_atoms())
    count = len(sites)
    return Structure(
        system=system,
        sites=sites,
        coordinates=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        coordinate_mask=np.ones(count, dtype=np.bool_),
    )


def _fold_prediction(
    value: float,
    *,
    preferred: str | None = "higher",
    unit: str | None = None,
    scope: frozenset | None = None,
) -> FoldPrediction:
    return FoldPrediction(
        _structure(),
        scores={"quality": Score(value, unit=unit, preferred=preferred, scope=scope)},
    )


def _design_candidate(
    value: float,
    *,
    preferred: str | None = "higher",
    unit: str | None = None,
    scope: frozenset | None = None,
) -> DesignCandidate:
    return DesignCandidate(
        _system(),
        scores={"quality": Score(value, unit=unit, preferred=preferred, scope=scope)},
    )


# --------------------------------------------------------------------------- #
# Section 5.1 shared scientific values
# --------------------------------------------------------------------------- #


def test_score_is_a_plain_carrier_with_identity_equality() -> None:
    evaluations = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    score = Score(
        2.0,
        unit="kcal/mol",
        preferred="lower",
        scope=frozenset({RESIDUE_A}),
        evaluations=evaluations,
    )
    assert score.value == 2.0
    assert score.unit == "kcal/mol"
    assert score.preferred == "lower"
    assert score.scope == frozenset({RESIDUE_A})
    # The released aggregate is not recomputed and the array is not copied.
    assert score.evaluations is evaluations

    # eq=False falls back to object identity; no NumPy ambiguous elementwise ==.
    assert Score(1.0) != Score(1.0)
    same = Score(1.0)
    assert same == same

    bare = Score(1.0)
    assert bare.unit is None
    assert bare.preferred is None
    assert bare.scope is None
    assert bare.evaluations is None


def test_score_keeps_nonfinite_values_and_empty_scope_visible() -> None:
    # NaN and infinity are model observations; the constructor never scans or
    # discards them (no __post_init__ finiteness check).
    assert math.isnan(Score(float("nan")).value)
    assert Score(float("inf")).value == float("inf")
    assert Score(float("-inf")).value == float("-inf")
    # An empty frozenset is an explicitly empty scope, distinct from None.
    assert Score(1.0, scope=frozenset()).scope == frozenset()
    assert Score(1.0, scope=None).scope is None


def test_metric_and_pair_metric_carry_arrays_without_scanning() -> None:
    values = np.array([0.1, 0.2], dtype=np.float32)
    axis = (RESIDUE_A, RESIDUE_B)
    metric = Metric(values, axis, "residue")
    # Plain carrier: supplied array is stored as-is, not copied or made read-only.
    assert metric.values is values
    assert metric.axis == axis
    assert metric.unit == "residue"

    pair_values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pair = PairMetric(pair_values, axis, axis, "Å")
    assert pair.values is pair_values
    assert pair.row_axis == axis
    assert pair.column_axis == axis
    assert pair.unit == "Å"

    # NaN and infinity remain visible in array values; unit defaults to None.
    noisy = Metric(np.array([np.nan, np.inf], dtype=np.float32), axis)
    assert np.isnan(noisy.values[0])
    assert np.isinf(noisy.values[1])
    assert noisy.unit is None

    # eq=False: distinct carriers are never equal, avoiding array-broadcast ==.
    assert metric != Metric(values, axis, "residue")
    assert pair != PairMetric(pair_values, axis, axis, "Å")


# --------------------------------------------------------------------------- #
# Section 5.2 FoldResult
# --------------------------------------------------------------------------- #


def test_fold_prediction_shares_supplied_mappings_and_defaults_independently() -> None:
    structure = _structure()
    scores = {"ptm": Score(0.9, preferred="higher")}
    metrics = {"plddt": Metric(np.array([90.0], dtype=np.float32), (RESIDUE_A,))}
    pair_metrics = {
        "pae": PairMetric(np.array([[1.0]], dtype=np.float32), (RESIDUE_A,), (RESIDUE_A,), "Å"),
    }
    prediction = FoldPrediction(structure, scores=scores, metrics=metrics, pair_metrics=pair_metrics)
    assert prediction.structure is structure
    # Supplied mappings are stored as-is, not copied or wrapped read-only.
    assert prediction.scores is scores
    assert prediction.metrics is metrics
    assert prediction.pair_metrics is pair_metrics

    first = FoldPrediction(structure)
    second = FoldPrediction(structure)
    assert first.scores == {}
    assert first.metrics == {}
    assert first.pair_metrics == {}
    # default_factory gives each instance its own empty mapping, never a shared one.
    assert first.scores is not second.scores


def test_fold_best_ranks_by_direction_and_keeps_earliest_tie() -> None:
    low = _fold_prediction(1.0)
    high = _fold_prediction(2.0)
    assert FoldResult((low, high)).best("quality") is high

    worse = _fold_prediction(2.0, preferred="lower")
    better = _fold_prediction(1.0, preferred="lower")
    assert FoldResult((worse, better)).best("quality") is better

    first = _fold_prediction(1.0)
    tied = _fold_prediction(1.0)
    assert FoldResult((first, tied)).best("quality") is first

    # Infinity stays rankable rather than failing.
    finite = _fold_prediction(1.0)
    infinite = _fold_prediction(float("inf"))
    assert FoldResult((finite, infinite)).best("quality") is infinite


def test_fold_best_fails_only_when_ranking_is_impossible() -> None:
    # Missing named score fails through ordinary lookup, not at construction.
    with pytest.raises(KeyError, match="quality"):
        FoldResult((_fold_prediction(1.0), FoldPrediction(_structure()))).best("quality")

    # Undefined direction.
    with pytest.raises(ValueError):
        FoldResult((_fold_prediction(1.0, preferred=None),)).best("quality")

    # Conflicting directions.
    with pytest.raises(ValueError):
        FoldResult((_fold_prediction(1.0, preferred="higher"), _fold_prediction(2.0, preferred="lower"))).best(
            "quality"
        )

    # Units are compared exactly and never normalized or converted.
    with pytest.raises(ValueError):
        FoldResult((_fold_prediction(1.0, unit="kcal/mol"), _fold_prediction(2.0, unit="kJ/mol"))).best("quality")

    # Scopes must match exactly.
    with pytest.raises(ValueError):
        FoldResult(
            (
                _fold_prediction(1.0, scope=frozenset({RESIDUE_A})),
                _fold_prediction(2.0, scope=frozenset({RESIDUE_B})),
            )
        ).best("quality")

    # NaN fails at ranking time, not when the Result is constructed.
    with pytest.raises(ValueError):
        FoldResult((_fold_prediction(float("nan")), _fold_prediction(1.0))).best("quality")


def test_fold_best_matches_scopes_as_unordered_frozensets() -> None:
    scope_ab = frozenset({RESIDUE_A, RESIDUE_B})
    scope_ba = frozenset({RESIDUE_B, RESIDUE_A})
    result = FoldResult((_fold_prediction(1.0, scope=scope_ab), _fold_prediction(2.0, scope=scope_ba)))
    assert result.best("quality").scores["quality"].value == 2.0


# --------------------------------------------------------------------------- #
# Section 5.3.1 DesignResult
# --------------------------------------------------------------------------- #


def test_design_candidate_owns_its_coordinate_free_system() -> None:
    system = _system()
    scores = {"quality": Score(0.8, preferred="higher")}
    candidate = DesignCandidate(system, scores=scores)
    assert candidate.system is system
    # The shared carrier has no input, input-score, provenance, or generated map.
    assert candidate.scores is scores
    assert DesignCandidate(system).scores == {}


def test_design_best_shares_fold_best_ranking_behavior() -> None:
    low = _design_candidate(1.0)
    high = _design_candidate(2.0)
    assert DesignResult((low, high)).best("quality") is high

    worse = _design_candidate(2.0, preferred="lower")
    better = _design_candidate(1.0, preferred="lower")
    assert DesignResult((worse, better)).best("quality") is better

    first = _design_candidate(1.0)
    tied = _design_candidate(1.0)
    assert DesignResult((first, tied)).best("quality") is first

    with pytest.raises(KeyError, match="quality"):
        DesignResult((_design_candidate(1.0), DesignCandidate(_system()))).best("quality")

    # Comparing incommensurate residue subsets fails at ranking time.
    with pytest.raises(ValueError):
        DesignResult(
            (
                _design_candidate(1.0, scope=frozenset({RESIDUE_A})),
                _design_candidate(2.0, scope=frozenset({RESIDUE_B})),
            )
        ).best("quality")

    with pytest.raises(ValueError):
        DesignResult((_design_candidate(float("nan")), _design_candidate(1.0))).best("quality")


# --------------------------------------------------------------------------- #
# Section 5.4.1 ScoreResult
# --------------------------------------------------------------------------- #


def test_score_result_binds_scores_and_metrics_to_its_target() -> None:
    structure = _structure()
    metric = Metric(np.array([0.5], dtype=np.float32), (RESIDUE_A,), "residue")
    result = ScoreResult(structure, scores={"global": Score(1.0)}, metrics={"per_residue": metric})
    # Target is the exact Structure on which the scores were evaluated.
    assert result.target is structure
    assert result.scores["global"].value == 1.0
    assert result.metrics["per_residue"] is metric

    empty = ScoreResult(structure)
    assert empty.scores == {}
    assert empty.metrics == {}


# --------------------------------------------------------------------------- #
# Section 5.5 PackResult
# --------------------------------------------------------------------------- #


def test_pack_carriers_group_metrics_without_scores_or_ranking() -> None:
    structure = _structure()
    metrics = {
        "chi1_log_density": Metric(np.array([0.3], dtype=np.float32), (RESIDUE_A,), "nat/radian"),
    }
    prediction = PackPrediction(structure, metrics=metrics)
    assert prediction.structure is structure
    assert prediction.metrics is metrics

    result = PackResult((prediction,))
    assert result.predictions == (prediction,)

    # Section 5.5: PackPrediction has no scores and PackResult defines no best().
    assert {field.name for field in fields(PackPrediction)} == {"structure", "metrics"}
    assert {field.name for field in fields(PackResult)} == {"predictions"}
    assert not hasattr(PackResult, "best")
