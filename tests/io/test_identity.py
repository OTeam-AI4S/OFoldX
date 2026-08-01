import pytest

from ofoldx.data.references import AtomSite
from ofoldx.io import CifSource, parse_cif
from ofoldx.io.identity import AtomIndex, Endpoint, IdentityConflictError, InsufficientIdentityEvidenceError, SiteIndex


class TestIdentity:
    def test_namespace_conflict(self, cif_text):
        rows = CifSource(parse_cif(cif_text)).atoms.by_index(1)
        index = AtomIndex(rows)
        correct = Endpoint(("A", 1, None, "ALA", "CA"), ("Q", "7", None, None, None))
        assert index.resolve(correct) == (1,)
        wrong_author = correct._replace(auth=("Q", "9", None, None, None))
        with pytest.raises(IdentityConflictError):
            index.resolve(wrong_author)
        # Projection cannot hide a contradictory source declaration.
        with pytest.raises(IdentityConflictError):
            index.resolve(wrong_author, included_rows=set())
        assert index.resolve(correct, included_rows=set()) == ()

    def test_unobserved_author_name(self):
        from ofoldx.data import AtomSite
        from ofoldx.io.identity import SiteIndex

        first = AtomSite("A", 0, "CA")
        second = AtomSite("A", 1, "CA")
        index = SiteIndex()
        index.add("label", (None, "A", 1, None, "ALA", "CA"), first)
        index.add("label", (None, "A", 2, None, "ALA", "CA"), second)
        index.add("auth", (None, "Q", "7", None, "ALA", None), first)
        index.add("auth", (None, "Q", "8", None, "ALA", None), second)
        assert index.unique(Endpoint(("A", 1, None, "ALA", "CA"), ("Q", "7", None, "ALA", "CX"))) == first
        with pytest.raises(IdentityConflictError):
            index.resolve(Endpoint(("A", 1, None, "ALA", "CA"), ("Q", "8", None, "ALA", "CX")))

        known = SiteIndex()
        known.add("label", (None, "A", 1, None, "ALA", "CA"), first)
        known.add("auth", (None, "Q", "7", None, "ALA", None), first)
        known.add("auth", (None, "Q", "7", None, "ALA", "CA"), first)
        with pytest.raises(IdentityConflictError):
            known.resolve(Endpoint(("A", 1, None, "ALA", "CA"), ("Q", "7", None, "ALA", "CX")))

    def test_unknown_author(self):
        index = SiteIndex()
        site = AtomSite("A", 0, "CA")
        index.add("label", (None, "A", 1, None, "ALA", "CA"), site)
        index.add("auth", (None, None, None, None, None, None), site)
        endpoint = Endpoint((None,) * 5, ("invented", "999", None, None, None))
        with pytest.raises(InsufficientIdentityEvidenceError):
            index.unique(endpoint)

    def test_unknown_alternative(self):
        index = SiteIndex()
        for i, author in enumerate(("Q", None)):
            site = AtomSite("A", i, "CA")
            index.add("label", (None, "A", i + 1, None, "ALA", "CA"), site)
            index.add("auth", (None, author, "1", None, "ALA", "CA"), site)
        with pytest.raises(InsufficientIdentityEvidenceError):
            index.unique(Endpoint((None,) * 5, ("Q", "1", None, "ALA", "CA")))
