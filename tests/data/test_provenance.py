from __future__ import annotations

import pytest

# The §3.3 value-shape freeze and the §3.5.1 Structure-carrier freeze together removed
# the entire chemical-provenance and private source-identity subsystem this module used
# to cover: ``ChemicalProvenance``/``ChemicalTransform``, per-component/molecule/bond
# provenance, ``Molecule`` SMILES-origin provenance, and the private
# ``_SourceAtomIdentity``/``_StructureSource`` mapping. None of those have a frozen
# replacement — each is explicitly reopened (§3.3 Molecule provenance; §3.4 SystemBond
# origin-domain reconciliation; §3.5 full source-identity provenance). Coverage is kept
# as xfail markers so the record is preserved and any accidental reintroduction surfaces.


@pytest.mark.xfail(
    reason="§3.3 reopened: Molecule/component provenance (incl. exact SMILES parser "
    "provenance) was removed; Molecule deliberately has no provenance field",
    strict=True,
    raises=ImportError,
)
def test_smiles_import_records_exact_parser_provenance() -> None:
    # TODO(§3.3 reopened): ChemicalProvenance and component.provenance were removed.
    from ofoldx.data.molecular import ChemicalProvenance  # noqa: F401


@pytest.mark.xfail(
    reason="§3.3 reopened: component/molecule provenance identity was removed; value "
    "identity now rests on the plain chemical-graph fields alone",
    strict=True,
    raises=ImportError,
)
def test_reference_and_provenance_are_ordinary_component_value_identity() -> None:
    # TODO(§3.3 reopened): ChemicalProvenance and component.provenance were removed.
    from ofoldx.data.molecular import ChemicalProvenance  # noqa: F401


@pytest.mark.xfail(
    reason="§3.3 reopened: ChemicalProvenance/ChemicalTransform and their validation were "
    "removed; the provenance/conditions facet stays reopened",
    strict=True,
    raises=ImportError,
)
def test_provenance_rejects_unearned_bags_and_invalid_mappings() -> None:
    # TODO(§3.3 reopened): ChemicalProvenance/ChemicalTransform were removed.
    from ofoldx.data import ChemicalProvenance, ChemicalTransform  # noqa: F401


@pytest.mark.xfail(
    reason="§3.4 reopened: SystemBond.origin is now a plain field with no provenance and no "
    "non-user-origin provenance requirement; the BondOrigin/provenance domain is reopened",
    strict=True,
    raises=AttributeError,
)
def test_bond_origin_is_semantic_and_non_user_origin_requires_provenance() -> None:
    from ofoldx.data.molecular import AtomSite, SystemBond

    bond = SystemBond(AtomSite("a", 0, "C"), AtomSite("b", 0, "N"), origin="geometry_inferred")
    # TODO(§3.4 reopened): SystemBond.provenance was removed; origin is a plain field.
    _ = bond.provenance  # type: ignore[attr-defined]


@pytest.mark.xfail(
    reason="§3.5.1 reopened: the private _StructureSource/_SourceAtomIdentity mapping was "
    "removed; residue_numbers carries the one retained source fact and the broad source "
    "identity bag stays reopened in §3.5",
    strict=True,
    raises=ImportError,
)
def test_private_structure_source_mapping_is_validated_and_not_semantic_identity() -> None:
    # TODO(§3.5 reopened): _SourceAtomIdentity/_StructureSource were removed by §3.5.1.
    from ofoldx.data.structure import _SourceAtomIdentity, _StructureSource  # noqa: F401
