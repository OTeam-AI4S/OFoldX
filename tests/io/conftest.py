from pathlib import Path

import pytest


@pytest.fixture
def cif_text() -> str:
    return (Path(__file__).parent / "fixtures" / "template.cif").read_text(encoding="utf-8")


@pytest.fixture
def cif_path(tmp_path: Path, cif_text: str) -> Path:
    path = tmp_path / "template.cif"
    path.write_text(cif_text, encoding="utf-8")
    return path


@pytest.fixture(
    params=[
        ("struct_asym", "loop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\nB 2\n"),
        ("entity_poly", "loop_\n_entity_poly.entity_id\n_entity_poly.type\n1 'polypeptide(L)'\n"),
        (
            "entity_poly_seq",
            "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n"
            "_entity_poly_seq.mon_id\n_entity_poly_seq.hetero\n1 1 ALA n\n1 2 GLY n\n1 3 MSE n\n",
        ),
    ],
    ids=["chains", "polymer-type", "sequence"],
)
def incomplete_cif(request, cif_text):
    category, declaration = request.param
    return category, cif_text.replace(declaration, "")
