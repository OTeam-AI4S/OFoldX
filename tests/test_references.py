from __future__ import annotations

from pathlib import Path

import pytest

from ofoldx.references import classify_reference


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("./missing", Path("./missing")),
        ("~/missing", Path("~/missing").expanduser()),
        (r"C:\missing", Path(r"C:\missing")),
        ("owner/repo", "owner/repo"),
        ("", ""),
    ],
)
def test_classification(reference: str, expected: Path | str) -> None:
    assert classify_reference(reference) == expected


def test_existing_path(tmp_path: Path) -> None:
    assert classify_reference(str(tmp_path)) == tmp_path
