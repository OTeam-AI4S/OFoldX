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

r"""
Sequence-liability annotations and reports.

This module contains sequence-only operations shared by real model families. It
does not own structural metrics, candidate filtering, ranking, diversity
reduction, or protocol selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias

from .ranking import AnalysisValue

LiabilityModality: TypeAlias = Literal["antibody", "peptide"]
PeptideType: TypeAlias = Literal["linear", "cyclic"]

_LIABILITY_SEVERITY = {
    "OddCysteineCount": 10,
    "DeAmdH": 10,
    "FragH": 10,
    "Isom": 10,
    "TrpOx": 10,
    "IntBind": 10,
    "DeAmdM": 5,
    "FragM": 5,
    "DeAmdL": 1,
    "AspBridge": 10,
    "AspCleave": 10,
    "ProtTryp": 10,
    "DPP4": 5,
    "LowHydrophilic": 7,
    "ConsecIdentical": 7,
    "LongHydrophobic": 7,
}
_DEFAULT_LIABILITY_SEVERITY = 5


@dataclass(frozen=True)
class Liability:
    r"""
    One shared sequence-liability occurrence.

    Attributes:
        motif: Liability name.
        position: One-based sequence position, or ``None`` for a global property.
        num_residues: Number of residues in the matched motif.
        severity: Uncalibrated integer screening weight, not a measured risk.
        details: Optional human-readable value for a global property.
    """

    motif: str
    position: int | None
    num_residues: int
    severity: int
    details: str = ""


def sequence_liabilities(
    sequence: str,
    *,
    modality: LiabilityModality = "antibody",
    peptide_type: PeptideType = "linear",
) -> tuple[Liability, ...]:
    r"""
    Return motif and composition screening annotations.

    These are sequence heuristics with uncalibrated screening weights. Cysteine
    count does not determine disulfide pairing. ``BasicResidueExcess`` reports
    K+R-D-E composition, not a physical net charge at a specified pH.
    """

    if not sequence:
        raise ValueError("sequence is empty")
    if modality not in {"antibody", "peptide"}:
        raise ValueError(f"unsupported liability modality {modality!r}")
    if peptide_type not in {"linear", "cyclic"}:
        raise ValueError(f"unsupported peptide type {peptide_type!r}")
    violations: list[Liability] = []
    for name, pattern in _liability_motifs(modality, peptide_type).items():
        for match in pattern.finditer(sequence):
            violations.append(
                Liability(
                    name,
                    match.start() + 1,
                    len(match.group()),
                    _LIABILITY_SEVERITY.get(name, _DEFAULT_LIABILITY_SEVERITY),
                )
            )

    cysteine_count = sequence.count("C")
    if cysteine_count % 2:
        violations.append(
            Liability(
                "OddCysteineCount",
                None,
                0,
                _LIABILITY_SEVERITY["OddCysteineCount"],
                f"{cysteine_count} cysteines",
            )
        )
    if modality == "antibody":
        basic_excess = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
        if basic_excess > 1:
            violations.append(
                Liability(
                    "BasicResidueExcess",
                    None,
                    0,
                    _DEFAULT_LIABILITY_SEVERITY,
                    f"K+R-D-E={basic_excess}",
                )
            )
    elif peptide_type == "cyclic":
        hydrophilic = sum(sequence.count(residue) for residue in "DEKRHNQST")
        hydrophilic_percentage = hydrophilic / len(sequence) * 100
        if hydrophilic_percentage < 40:
            violations.append(
                Liability(
                    "LowHydrophilic",
                    None,
                    0,
                    _LIABILITY_SEVERITY["LowHydrophilic"],
                    f"{hydrophilic_percentage:.1f}% hydrophilic",
                )
            )

        longest_identical = 1
        current_identical = 1
        for index in range(1, len(sequence)):
            if sequence[index] == sequence[index - 1]:
                current_identical += 1
                longest_identical = max(longest_identical, current_identical)
            else:
                current_identical = 1
        if longest_identical > 1:
            violations.append(
                Liability(
                    "ConsecIdentical",
                    None,
                    0,
                    _LIABILITY_SEVERITY["ConsecIdentical"],
                    f"{longest_identical} consecutive identical",
                )
            )

        longest_hydrophobic = 0
        current_hydrophobic = 0
        for residue in sequence:
            current_hydrophobic = current_hydrophobic + 1 if residue in "FILVWY" else 0
            longest_hydrophobic = max(longest_hydrophobic, current_hydrophobic)
        if longest_hydrophobic > 4:
            violations.append(
                Liability(
                    "LongHydrophobic",
                    None,
                    0,
                    _LIABILITY_SEVERITY["LongHydrophobic"],
                    f"{longest_hydrophobic} consecutive hydrophobic",
                )
            )
    return tuple(violations)


def liability_metrics(
    sequence: str,
    *,
    modality: LiabilityModality = "antibody",
    peptide_type: PeptideType = "linear",
) -> dict[str, AnalysisValue]:
    r"""
    Return shared aggregate and per-motif liability values.
    """

    violations = sequence_liabilities(sequence, modality=modality, peptide_type=peptide_type)
    metrics: dict[str, AnalysisValue] = {
        "liability_score": sum(violation.severity for violation in violations),
        "liability_num_violations": len(violations),
        "liability_high_severity_violations": sum(violation.severity >= 10 for violation in violations),
        "liability_medium_severity_violations": sum(5 <= violation.severity < 10 for violation in violations),
        "liability_low_severity_violations": sum(violation.severity < 5 for violation in violations),
    }
    grouped: dict[str, list[Liability]] = {}
    for violation in violations:
        grouped.setdefault(violation.motif, []).append(violation)
    motif_names = set(_liability_motifs(modality, peptide_type))
    motif_names.update(grouped)
    for motif in sorted(motif_names):
        matches = grouped.get(motif, [])
        positions = [violation.position for violation in matches if violation.position is not None]
        details = [violation.details for violation in matches if violation.details]
        metrics[f"liability_{motif}_count"] = len(matches)
        metrics[f"liability_{motif}_position"] = -1
        metrics[f"liability_{motif}_num_residues"] = 0
        metrics[f"liability_{motif}_severity"] = 0
        metrics[f"liability_{motif}_details"] = ""
        metrics[f"liability_{motif}_positions"] = ""
        metrics[f"liability_{motif}_num_positions"] = 0
        metrics[f"liability_{motif}_global_details"] = ""
        metrics[f"liability_{motif}_avg_severity"] = 0.0
        if len(matches) == 1:
            violation = matches[0]
            metrics[f"liability_{motif}_position"] = -1 if violation.position is None else violation.position
            metrics[f"liability_{motif}_num_residues"] = violation.num_residues
            metrics[f"liability_{motif}_severity"] = violation.severity
            metrics[f"liability_{motif}_details"] = violation.details
        elif matches:
            if positions:
                metrics[f"liability_{motif}_positions"] = f"{min(positions)}-{max(positions)}"
                metrics[f"liability_{motif}_num_positions"] = len(positions)
            metrics[f"liability_{motif}_global_details"] = "; ".join(details)
            metrics[f"liability_{motif}_avg_severity"] = round(
                sum(violation.severity for violation in matches) / len(matches),
                1,
            )

    summary: list[str] = []
    for motif, matches in grouped.items():
        positions = [violation.position for violation in matches if violation.position is not None]
        if len(matches) == 1:
            violation = matches[0]
            if violation.position is not None:
                summary.append(f"{motif}(pos{violation.position},sev{violation.severity})")
            elif violation.details:
                summary.append(f"{motif}({violation.details},sev{violation.severity})")
            else:
                summary.append(f"{motif}(sev{violation.severity})")
        elif positions:
            summary.append(f"{motif}x{len(matches)}(pos{min(positions)}-{max(positions)},sev{matches[0].severity})")
        else:
            summary.append(f"{motif}x{len(matches)}(sev{matches[0].severity})")
    metrics["liability_violations_summary"] = "; ".join(summary)
    return metrics


def _liability_motifs(
    modality: LiabilityModality,
    peptide_type: PeptideType,
) -> dict[str, re.Pattern[str]]:
    if modality == "antibody":
        return {
            "DeAmdH": re.compile(r"N[GS]"),
            "DeAmdM": re.compile(r"N[AHNT]"),
            "DeAmdL": re.compile(r"[STK]N"),
            "Ngly": re.compile(r"N[^P][ST]"),
            "Isom": re.compile(r"D[DGHST]"),
            "Isomer": re.compile(r"DG|DS|DD"),
            "FragH": re.compile(r"DP"),
            "FragM": re.compile(r"TS"),
            "TrpOx": re.compile(r"W"),
            "MetOx": re.compile(r"M"),
            "Hydro": re.compile(r"NP"),
            "IntBind": re.compile(r"GPR|RGD|RYD|LDV|DGE|KGD|NGR"),
            "Polyreactive": re.compile(r"GGG|GG|RR|VG|VVV|WWW|YY|WxW"),
            "AggPatch": re.compile(r"FHW"),
            "ViscPatch": re.compile(r"HYF|HWH"),
        }
    motifs = {
        "AspBridge": re.compile(r"N[GSQA]"),
        "AspCleave": re.compile(r"D[PGS]"),
        "NTCycl": re.compile(r"^[QN]"),
        "ProtTryp": re.compile(r"[KR](?=.)"),
        "DPP4": re.compile(r"^[PX]?[AP]"),
        "MetOx": re.compile(r"M"),
        "TrpOx": re.compile(r"W"),
        "HydroPatch": re.compile(r"[FILVWY]{3,}"),
    }
    if peptide_type == "cyclic":
        motifs.pop("NTCycl")
        motifs.pop("DPP4")
    return motifs
