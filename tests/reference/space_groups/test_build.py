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

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import resources

from ofoldx.reference.space_groups import load_table

TABLE_SHA256 = "dab90cc1c201e07ed6218f5a90500ceb81a1b49a0f823eaa6483ebbe90ebbe06"


def _build(tmp_path, operator: str, identifier: str = "1"):
    source = tmp_path / "symop.lib"
    evidence = tmp_path / "evidence.json"
    output = tmp_path / "table.json"
    source.write_text("1 1 1 P1 1 triclinic 'P 1'\nx,y,z\n", encoding="utf-8")
    evidence.write_text(json.dumps({"table": {"P 1": {"named": {identifier: {operator: 4}}}}}), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ofoldx.reference.space_groups.build",
            "--symop",
            str(source),
            "--evidence",
            str(evidence),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return source, output, result


def test_builds_table_from_matching_evidence(tmp_path):
    source, output, result = _build(tmp_path, "x,y,z")
    assert result.returncode == 0, result.stderr
    table = json.loads(output.read_text(encoding="utf-8"))
    assert table["settings"][0]["name"] == "P 1"
    assert table["settings"][0]["operators"] == ["x,y,z"]
    assert table["settings"][0]["validation"]["named_operator_observations"] == 4
    assert table["provenance"]["symop.lib"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert table["provenance"]["generator"] == "ofoldx/reference/space_groups/build.py"
    assert table["provenance"]["operator_evidence"] == {
        "sha256": hashlib.sha256((tmp_path / "evidence.json").read_bytes()).hexdigest(),
        "named_operator_observations": 4,
    }
    assert table["provenance"]["settings_declared_by_evidence_but_absent"] == []
    assert table["provenance"]["validation"] == (
        "supplied operator evidence; highest-frequency operation per selected identifier compared; "
        "1 settings confirmed"
    )


def test_rejects_disagreeing_evidence_without_writing(tmp_path):
    _, output, result = _build(tmp_path, "x+1/2,y,z")
    assert result.returncode != 0
    assert "operator table disagrees with supplied evidence" in result.stderr
    assert not output.exists()


def test_unmatched_identifier_does_not_count_as_confirmed(tmp_path):
    _, output, result = _build(tmp_path, "x,y,z", identifier="2")
    assert result.returncode == 0, result.stderr
    table = json.loads(output.read_text(encoding="utf-8"))
    assert table["settings"][0]["validation"]["operators_confirmed"] == []
    assert table["provenance"]["validation"].endswith("0 settings confirmed")


def test_bundled_table_retains_original_data_and_provenance():
    data = resources.files("ofoldx.reference.space_groups").joinpath("space_group_operators.json").read_bytes()
    assert hashlib.sha256(data).hexdigest() == TABLE_SHA256
    table = load_table()
    assert table == json.loads(data)
    assert table["provenance"]["generator"] == "tools/build_space_group_operators.py"
    assert len(table["settings"]) == 268
