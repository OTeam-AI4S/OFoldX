r"""
Confidence-summary JSON output.

Serializes the pTM / ipTM / ranking-score confidence summary to the JSON layout produced by the
upstream AlphaFold3 post-processing step (``model/confidence_types.py`` ``StructureConfidenceSummary``).
``NaN`` scores (e.g. single-chain ipTM) serialize as JSON ``null`` to match upstream. Structure
coordinates and per-atom pLDDT are written separately by
[`write_structure`][ofoldx.models.pxdesign.io.write_structure] (the B-factor column); this module only
emits the scalar and per-chain confidence summary. Upstream
ModelCIF provenance metadata (citation, licence, ``_ma_qa_metric``) is intentionally omitted as
release-workflow boilerplate rather than core pipeline output.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np


def confidence_summary_json(
    *,
    ptm: float,
    iptm: float,
    ranking_score: float,
    chain_pair_iptm: np.ndarray | None = None,
    chain_ptm: np.ndarray | None = None,
) -> str:
    r"""
    Serialize the scalar confidence summary to the upstream JSON layout.

    Args:
        ptm: Predicted TM-score.
        iptm: Interface predicted TM-score (``NaN`` for single-chain systems).
        ranking_score: Combined ranking score.
        chain_pair_iptm: Optional ``[num_chains, num_chains]`` pairwise ipTM matrix.
        chain_ptm: Optional ``[num_chains]`` per-chain pTM vector.

    Returns:
        JSON string with the confidence summary; ``NaN`` scores become ``null``.

    Examples:
        >>> print(confidence_summary_json(ptm=0.9, iptm=float("nan"), ranking_score=0.8))
        {
          "ptm": 0.9,
          "iptm": null,
          "ranking_score": 0.8
        }
    """

    summary: dict[str, object] = {
        "ptm": _finite_or_none(ptm),
        "iptm": _finite_or_none(iptm),
        "ranking_score": _finite_or_none(ranking_score),
    }
    if chain_pair_iptm is not None:
        summary["chain_pair_iptm"] = [
            [_finite_or_none(value) for value in row] for row in np.asarray(chain_pair_iptm).tolist()
        ]
    if chain_ptm is not None:
        summary["chain_ptm"] = [_finite_or_none(value) for value in np.asarray(chain_ptm).tolist()]
    return json.dumps(summary, indent=2)


def write_confidence_json(
    path: str | PathLike[str],
    *,
    ptm: float,
    iptm: float,
    ranking_score: float,
    chain_pair_iptm: np.ndarray | None = None,
    chain_ptm: np.ndarray | None = None,
) -> None:
    r"""
    Write the confidence summary to a JSON file.

    Args:
        path: Output JSON path.
        ptm: Predicted TM-score.
        iptm: Interface predicted TM-score.
        ranking_score: Combined ranking score.
        chain_pair_iptm: Optional pairwise ipTM matrix.
        chain_ptm: Optional per-chain pTM vector.

    Examples:
        >>> import tempfile, os
        >>> path = os.path.join(tempfile.mkdtemp(), "confidence.json")
        >>> write_confidence_json(path, ptm=0.9, iptm=0.5, ranking_score=0.8)
        >>> print(open(path).read())
        {
          "ptm": 0.9,
          "iptm": 0.5,
          "ranking_score": 0.8
        }
    """

    Path(path).write_text(
        confidence_summary_json(
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking_score,
            chain_pair_iptm=chain_pair_iptm,
            chain_ptm=chain_ptm,
        )
    )


def _finite_or_none(value: float) -> float | None:
    r"""
    Return the float value, or ``None`` for ``NaN`` so it serializes to JSON ``null`` (upstream layout).

    Args:
        value: Scalar confidence score.

    Returns:
        ``float(value)``, or ``None`` when it is ``NaN``.
    """

    value = float(value)
    return None if np.isnan(value) else value
