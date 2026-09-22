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

r"""Build ``space_group_operators.json``: general-position operators in International Tables order.

The input ``symop.lib`` supplies operators for Hermann-Mauguin settings. ``CORRECTIONS`` retains the explicit
permutations used by the existing table. Supplied named-operator evidence is compared with the resulting order;
the input hashes, observation count, applied corrections and validation outcome are recorded as provenance.

Usage::

    python -m ofoldx.reference.space_groups.build --symop symop.lib --evidence operator_evidence.json \
        --output space_group_operators.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import gemmi

#: 1-based ``symop.lib`` positions listed in ITA order, for settings where ``symop.lib`` departs from ITA.
CORRECTIONS = {"I 2 2 2": [1, 2, 4, 3, 5, 6, 8, 7], "P 1 21/c 1": [1, 3, 2, 4]}

_HEADER = re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+((?:'[^']*'\s*)+)")


def canonical(triplet: str) -> str:
    return gemmi.Op(triplet.lower()).triplet()


def ita_names(names: list[str]) -> list[str]:
    """The ITA names of a rhombohedral group's setting: ``symop.lib``, like wwPDB ``_symmetry`` and ``CRYST1``, writes
    the hexagonal-axes setting ``H …`` and the rhombohedral-axes setting ``R …``; ITA and ``_space_group.name_H-M_alt``
    write them ``R … :H`` and ``R … :R``. Only short symbols take the suffix: ``R -3 2/m :H`` is no ITA name."""
    short = [name for name in names if name[0] in "HR" and "/" not in name]
    return [f"R{name[1:]} :H" if name[0] == "H" else f"{name} :R" for name in short]


def read_symop(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    settings = []
    index = 0
    while index < len(lines):
        match = _HEADER.match(lines[index])
        if not match:
            index += 1
            continue
        number, total, _primitive, short, point_group, system, quoted = match.groups()
        count = int(total)
        operators = [canonical(lines[index + 1 + offset].strip()) for offset in range(count)]
        names = [re.sub(r"\s+", " ", name.strip()) for name in re.findall(r"'([^']*)'", quoted)]
        names += ita_names(names)
        name = names[0]
        provenance = "supplied operator order"
        if name in CORRECTIONS:
            order = CORRECTIONS[name]
            if sorted(order) != list(range(1, count + 1)):
                raise ValueError(f"correction for {name!r} is not a permutation of 1..{count}")
            operators = [operators[position - 1] for position in order]
            provenance = "supplied operator order with explicit correction permutation applied"
        settings.append(
            {
                "number": int(number),
                "name": name,
                "aliases": names[1:],
                "short": short,
                "point_group": point_group,
                "system": system,
                "operators": operators,
                "provenance": provenance,
            }
        )
        index += 1 + count
    return settings


def validate(settings: list[dict], evidence: dict) -> tuple[list[dict], list[str]]:
    """Compare supplied named-operator evidence with the table.

    For each in-range identifier, the most frequent supplied operation is compared. Other observations are not
    independently checked for disagreement.

    Returns:
        Disagreements and the settings the evidence names but ``symop.lib`` does not list. The reader reports the
        latter as unresolved references when an operator other than 1 is needed.
    """
    by_name: dict[str, dict] = {}
    for setting in settings:
        for name in (setting["name"], *setting["aliases"]):
            if name in by_name:
                raise ValueError(f"duplicate setting name {name!r}")
            by_name[name] = setting
    problems = []
    missing: list[str] = []
    for name, entry in evidence.items():
        candidate = by_name.get(name)
        if candidate is None:
            missing.append(name)
            continue
        setting = candidate
        consistent = True
        confirmed = []
        for text, operators in entry["named"].items():
            n = int(text)
            if n > len(setting["operators"]):
                # A non-crystallographic operator carrying an n_klm-looking name; the reader gates on
                # _pdbx_struct_oper_list.type and never reads these as operator numbers.
                continue
            top = max(operators, key=operators.get)
            if gemmi.Op(setting["operators"][n - 1]).wrap() != gemmi.Op(top).wrap():
                consistent = False
                problems.append({"name": name, "n": n, "table": setting["operators"][n - 1], "evidence": top})
                continue
            confirmed.append(n)
        if consistent:
            setting["validation"] = {
                "named_operator_observations": sum(sum(counts.values()) for counts in entry["named"].values()),
                "operators_confirmed": sorted(confirmed),
            }
    return problems, sorted(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symop", type=Path, required=True, help="CCP4 symop.lib")
    parser.add_argument("--evidence", type=Path, required=True, help="JSON containing named-operator evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = read_symop(args.symop)
    evidence_bytes = args.evidence.read_bytes()
    evidence = json.loads(evidence_bytes)["table"]
    observations = sum(sum(counts.values()) for entry in evidence.values() for counts in entry["named"].values())
    problems, missing = validate(settings, evidence)
    if problems:
        for problem in problems:
            print(problem)
        raise SystemExit("operator table disagrees with supplied evidence")
    confirmed_settings = sum(bool(setting.get("validation", {}).get("operators_confirmed")) for setting in settings)
    table = {
        "description": (
            "Space-group general-position operators in International Tables order, "
            "as numbered by wwPDB n_klm symmetry codes."
        ),
        "provenance": {
            "generator": "ofoldx/reference/space_groups/build.py",
            "symop.lib": {
                "sha256": hashlib.sha256(args.symop.read_bytes()).hexdigest(),
            },
            "operator_evidence": {
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                "named_operator_observations": observations,
            },
            "corrections": CORRECTIONS,
            "validation": (
                "supplied operator evidence; highest-frequency operation per selected identifier compared; "
                f"{confirmed_settings} settings confirmed"
            ),
            "settings_declared_by_evidence_but_absent": missing,
        },
        "settings": settings,
    }
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print("settings", len(settings), "validated", confirmed_settings)


if __name__ == "__main__":
    main()
