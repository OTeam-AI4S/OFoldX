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

r"""Score a protein backbone with ProteinMPNN."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ofoldx import pipeline
from ofoldx.io import read_structure
from ofoldx.results import Score

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "tripeptide.pdb"


def score_record(score: Score) -> dict[str, object]:
    r"""Represent one scalar score with its unit, ranking direction and molecular scope.

    Args:
        score: A scalar score returned by the Pipeline.

    Returns:
        A JSON-compatible record. Component and atom references retain their named fields.
    """
    scope = None
    if score.scope is not None:
        scope = [reference if isinstance(reference, str) else asdict(reference) for reference in score.scope]
        scope.sort(key=lambda reference: json.dumps(reference, sort_keys=True))
    return {"value": score.value, "unit": score.unit, "preferred": score.preferred, "scope": scope}


def main(argv: list[str] | None = None) -> None:
    r"""Load a ProteinMPNN scoring Pipeline and write its named scores as JSON.

    Args:
        argv: Command-line arguments; parsed from the process when omitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="oteam/proteinmpnn",
        help="ProteinMPNN Hub repository or native local directory (default: oteam/proteinmpnn)",
    )
    parser.add_argument("--structure", type=Path, default=STRUCTURE, help="Input PDB or mmCIF file")
    parser.add_argument("--output", type=Path, default=Path("scores.json"), help="Destination JSON file")
    parser.add_argument("--device", default="cpu", help="Execution device, for example cpu or cuda")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for decoding-order sampling")
    args = parser.parse_args(argv)

    structure, _ = read_structure(args.structure)
    scorer = pipeline("score", model=args.model, device=args.device)
    result = scorer(structure, seed=args.seed)
    scores = {name: score_record(score) for name, score in result.scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
