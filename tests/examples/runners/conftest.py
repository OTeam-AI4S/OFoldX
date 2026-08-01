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

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
# each script in turn as __main__; a failure is written to <name>.error and the next one still runs
RUN = """
import pathlib, runpy, sys, traceback
scripts = list(map(pathlib.Path, sys.argv[1:]))
for script in scripts[:-1]:
    print(f"Running example: {script}", flush=True)
    try:
        runpy.run_path(str(script), run_name="__main__")
    except BaseException:
        pathlib.Path(script.stem + ".error").write_text(traceback.format_exc())
scoring = scripts[-1]
print(f"Running scoring example: {scoring}", flush=True)
sys.argv = [str(scoring), "--model", "runs/proteinmpnn/native", "--output", "proteinmpnn-scores.json"]
try:
    runpy.run_path(str(scoring), run_name="__main__")
except BaseException:
    pathlib.Path("proteinmpnn_score.error").write_text(traceback.format_exc())
"""


@pytest.fixture(scope="session")
def examples_directory(tmp_path_factory) -> Path:
    r"""Workspace containing the examples' exported repositories."""
    return tmp_path_factory.mktemp("examples")


@pytest.fixture(scope="session")
def examples(examples_directory, ccd_artifact) -> dict[str, str]:
    r"""Run the training examples and standalone scorer in one fresh process with the CCD subset.

    Return each runner's error by script name, and the scoring CLI's error under ``proteinmpnn_score``;
    an empty string means the example ran.
    """

    cwd = examples_directory
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if pythonpath is None else os.pathsep.join((str(ROOT), pythonpath))
    env["OFOLDX_CCD_ARTIFACT"] = str(ccd_artifact)
    scripts = sorted((ROOT / "examples" / "runners").glob("*.py"))
    scoring = ROOT / "examples" / "pipelines" / "proteinmpnn.py"
    subprocess.run([sys.executable, "-c", RUN, *map(str, scripts), str(scoring)], cwd=cwd, env=env, check=True)
    errors = {script.stem: cwd / f"{script.stem}.error" for script in scripts}
    errors["proteinmpnn_score"] = cwd / "proteinmpnn_score.error"
    return {name: error.read_text() if error.exists() else "" for name, error in errors.items()}
