from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ("proteinmpnn", "alphafold3", "odesign"))
def test_runner_example(name: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if pythonpath is None else os.pathsep.join((str(ROOT), pythonpath))
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / "runners" / f"{name}.py")],
        cwd=tmp_path,
        env=env,
        check=True,
    )
