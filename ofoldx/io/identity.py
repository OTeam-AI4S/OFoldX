r"""One identity resolver for deposited atoms, selected views and connections."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, TypeVar

import numpy as np

from ..data.references import AtomSite
from .cif import CifCategory, column_integers

if TYPE_CHECKING:
    from .source import CifSource

_IdentityT = TypeVar("_IdentityT", int, AtomSite)
_IdentityKey: TypeAlias = tuple[object, ...]
_SitePatterns: TypeAlias = dict[tuple[int, ...], dict[_IdentityKey, set[AtomSite]]]


class IdentityConflictError(ValueError):
    """Label and author constraints cannot refer to the same source atom."""


class InsufficientIdentityEvidenceError(ValueError):
    """An endpoint is compatible with unknown fields but has no identifying evidence."""


def read_cif_chain_authors(source: CifSource, model_index: int = 1) -> Mapping[str, str]:
    r"""Combine known CIF chain names without hiding contradictory declarations.

    Atom-site names refer to the selected deposited model. Sequence-scheme names
    can also identify wholly unobserved chains. Unknown names supply no constraint.
    Coordinate and graph consumers use the same checks before chain selection.

    Args:
        source: Native CIF source retaining atom-site and sequence-scheme evidence.
        model_index: One-based position of the deposited model to inspect.

    Returns:
        The source's cached, read-only label-to-author mapping for this model.
            Chains without author evidence cannot establish or exclude a match.

    Raises:
        IdentityConflictError: If a label chain has conflicting known author names.
        ValueError: If the model position or decoded atom data is invalid.

    Examples:
        >>> from .source import CifSource, parse_cif
        >>> source = CifSource(parse_cif("data_x\n_pdbx_poly_seq_scheme.asym_id A\n"
        ...                             "_pdbx_poly_seq_scheme.pdb_strand_id Q\n"))
        >>> dict(read_cif_chain_authors(source))
        {'A': 'Q'}
    """
    return source.author_chains(model_index)


def _read_cif_chain_authors(source: CifSource, model_index: int) -> dict[str, str]:
    """Decode one model's chain names for the owning source cache."""
    authors: dict[str, str] = {}

    def add(asym: str | None, author: str | None) -> None:
        if asym is None or author is None:
            return
        previous = authors.setdefault(asym, author)
        if previous != author:
            raise IdentityConflictError(f"asym {asym!r} has conflicting author chain identifiers")

    rows = source.atoms.by_index(model_index)
    for asym, author in zip(rows.chain_id, rows.auth_chain_id, strict=True):
        add(asym, author)
    scheme = source.block.get("pdbx_poly_seq_scheme")
    if scheme is not None:
        for asym, author in zip(scheme.strings("asym_id"), scheme.strings("pdb_strand_id"), strict=True):
            add(asym, author)
    return authors


class Endpoint(NamedTuple):
    label: tuple[object | None, ...]
    auth: tuple[object | None, ...]
    altloc: str | None = None
    symmetry: str | None = None


@dataclass(frozen=True)
class Connection:
    row: int
    identifier: str | None
    kind: str | None
    first: Endpoint
    second: Endpoint


def read_connections(category: CifCategory | None) -> tuple[Connection, ...]:
    """Decode both endpoint namespaces once, without fallback or graph inference."""
    if category is None:
        return ()
    endpoints = []
    for partner in (1, 2):
        label: list[tuple[object | None, ...]] = [
            category.strings(f"ptnr{partner}_label_{field}") for field in ("asym_id", "seq_id", "comp_id", "atom_id")
        ]
        label[1] = column_integers(category, f"ptnr{partner}_label_seq_id")
        author = [
            category.strings(f"ptnr{partner}_auth_{field}") for field in ("asym_id", "seq_id", "comp_id", "atom_id")
        ]
        insertion = category.strings(f"pdbx_ptnr{partner}_PDB_ins_code")
        symmetries = category.strings(f"ptnr{partner}_symmetry")
        alternatives = [
            category.strings(name)
            for name in (
                f"pdbx_ptnr{partner}_label_alt_id",
                f"ptnr{partner}_label_alt_id",
                f"pdbx_ptnr{partner}_auth_alt_id",
            )
            if name in category
        ]
        rows = []
        for i in range(category.row_count):
            alts = {column[i] for column in alternatives if column[i] is not None}
            if len(alts) > 1:
                raise ValueError(f"mmCIF struct_conn row {i} has conflicting alternate-location identities")
            rows.append(
                Endpoint(
                    (
                        label[0][i],
                        label[1][i],
                        insertion[i],
                        label[2][i],
                        label[3][i],
                    ),
                    (author[0][i], author[1][i], insertion[i], author[2][i], author[3][i]),
                    next(iter(alts), None),
                    symmetries[i],
                )
            )
        endpoints.append(rows)
    return tuple(
        Connection(i, identifier, kind, first, second)
        for i, (identifier, kind, first, second) in enumerate(
            zip(
                category.strings("id"),
                category.strings("conn_type_id"),
                endpoints[0],
                endpoints[1],
                strict=True,
            )
        )
    )


def _intersect_namespaces(
    label: frozenset[_IdentityT] | None,
    author: frozenset[_IdentityT] | None,
    endpoint: Endpoint,
) -> frozenset[_IdentityT]:
    """Combine independent namespace evidence; no ranking or cross-namespace union."""
    if label is not None and author is not None:
        matches = label & author
        if not matches and (label or author):
            raise IdentityConflictError(
                f"conflicting label/auth atom identities: {endpoint.label!r}, {endpoint.auth!r}"
            )
        return matches
    if label is not None:
        return label
    if author is not None:
        return author
    raise ValueError("connection endpoint contains no atom identity")


class AtomIndex:
    """Index CIF identities by the fields actually stated by a reference.

    A missing field is an absent constraint, not a manufactured label/auth
    alias. Each namespace/query shape is indexed once, then references are hash
    lookups. Label and author candidate sets are intersected, never ranked by
    specificity. Projection masks are applied only after source consistency.
    """

    def __init__(self, rows: np.recarray) -> None:
        self._columns = {
            "label": tuple(
                rows[name].tolist()
                for name in ("chain_id", "label_seq_id", "insertion_code", "component_id", "atom_name")
            ),
            "auth": tuple(
                rows[name].tolist()
                for name in (
                    "auth_chain_id",
                    "auth_residue_id",
                    "insertion_code",
                    "auth_component_id",
                    "auth_atom_name",
                )
            ),
        }
        self.altlocs = rows.altloc_id.tolist()
        self._indices: dict[tuple[str, tuple[int, ...]], dict[tuple, tuple[int, ...]]] = {}
        self.size = len(rows)

    def _lookup(self, namespace: str, identity: tuple) -> frozenset[int] | None:
        fields = tuple(i for i, value in enumerate(identity) if value is not None)
        # Insertion code by itself is not an author namespace declaration.
        if not fields or fields == (2,):
            return None
        shape = namespace, fields
        if shape not in self._indices:
            grouped: dict[tuple, list[int]] = {}
            for row, key in enumerate(zip(*(self._columns[namespace][i] for i in fields), strict=True)):
                grouped.setdefault(key, []).append(row)
            self._indices[shape] = {key: tuple(rows) for key, rows in grouped.items()}
        return frozenset(self._indices[shape].get(tuple(identity[i] for i in fields), ()))

    def resolve(
        self,
        endpoint: Endpoint,
        *,
        match_altloc: bool = True,
        included_rows: Collection[int] | None = None,
    ) -> tuple[int, ...]:
        label = self._lookup("label", endpoint.label)
        author = self._lookup("auth", endpoint.auth)
        matches = _intersect_namespaces(label, author, endpoint)
        return tuple(
            sorted(
                row
                for row in matches
                if (included_rows is None or row in included_rows)
                and (not match_altloc or endpoint.altloc is None or self.altlocs[row] == endpoint.altloc)
            )
        )

    def unique(self, endpoint: Endpoint) -> int | None:
        candidates = self.resolve(endpoint)
        if len(candidates) > 1:
            raise ValueError("mmCIF struct_conn endpoint is ambiguous in the selected atom rows")
        return candidates[0] if candidates else None


@dataclass(frozen=True)
class NamespaceMatches:
    """Candidates consistent with constraints, and those positively identified."""

    possible: frozenset[AtomSite]
    supported: frozenset[AtomSite]


class SiteIndex:
    """Namespace evidence for canonical atoms, including unobserved atoms.

    A connection can supply an author atom name for an unobserved site uniquely
    identified in label space. That is source evidence, not an inferred alias.
    Known author evidence must still agree; an unknown value is never allowed
    to override an observed one. Index shapes have at most five identity fields.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[tuple[object, AtomSite], _IdentityKey]] = {"label": {}, "auth": {}}
        self._indices: dict[tuple[str, tuple[int, ...], bool], _SitePatterns] = {}
        self._sealed = False
        self._asym_by_site: dict[AtomSite, object] = {}

    def seal(self) -> None:
        """Close evidence collection before publishing a molecular plan."""
        self._sealed = True

    def add(self, namespace: str, key: _IdentityKey, site: AtomSite) -> None:
        if self._sealed:
            raise RuntimeError("cannot mutate a sealed site index")
        entries = self._entries[namespace]
        scope = key[0], site
        identity = key[1:]
        previous = entries.get(scope)
        if previous is not None:
            if any(a is not None and b is not None and a != b for a, b in zip(previous, identity, strict=True)):
                raise IdentityConflictError(f"conflicting {namespace} declarations for canonical site {site!r}")
            identity = tuple(a if a is not None else b for a, b in zip(previous, identity, strict=True))
        entries[scope] = identity
        if namespace == "label":
            self._asym_by_site[site] = identity[0]

    def asym_ids(self, sites: Collection[AtomSite]) -> set[object]:
        return {self._asym_by_site[site] for site in sites}

    def entries(self, namespace: str) -> Iterator[tuple[_IdentityKey, AtomSite]]:
        return (((symmetry, *identity), site) for (symmetry, site), identity in self._entries[namespace].items())

    def _lookup(
        self, namespace: str, identity: _IdentityKey, symmetry: str | None, scoped: bool
    ) -> NamespaceMatches | None:
        fields = tuple(i for i, value in enumerate(identity) if value is not None)
        if not fields or fields == (2,):
            return None
        shape = namespace, fields, scoped
        self._sealed = True
        if shape not in self._indices:
            patterns: _SitePatterns = {}
            for key, site in self.entries(namespace):
                # Only author names may be newly declared by a connection.
                # Label sequence None on a non-polymer is inapplicable.
                known = tuple(i for i in fields if namespace != "auth" or key[i + 1] is not None)
                grouped = patterns.setdefault(known, {})
                identity_key = tuple(key[i + 1] for i in known)
                query = (key[0], *identity_key) if scoped else identity_key
                grouped.setdefault(query, set()).add(site)
            self._indices[shape] = patterns
        candidates: set[AtomSite] = set()
        exact: set[AtomSite] = set()
        for known, grouped in self._indices[shape].items():
            query = tuple(identity[i] for i in known)
            matches = grouped.get((symmetry, *query) if scoped else query, ())
            candidates.update(matches)
            if known == fields:
                exact.update(matches)
        return NamespaceMatches(frozenset(candidates), frozenset(exact))

    def resolve(self, endpoint: Endpoint, *, symmetry: str | None = None, scoped: bool = True) -> frozenset[AtomSite]:
        label = self._lookup("label", endpoint.label, symmetry, scoped)
        author = self._lookup("auth", endpoint.auth, symmetry, scoped)
        candidates = _intersect_namespaces(
            None if label is None else label.possible, None if author is None else author.possible, endpoint
        )
        supported = (frozenset() if label is None else label.supported) | (
            frozenset() if author is None else author.supported
        )
        # Unknown fields retain possible alternatives; they neither prove a
        # match nor disappear when a better documented candidate also exists.
        if candidates and not candidates.issubset(supported):
            raise InsufficientIdentityEvidenceError(
                f"insufficient identity evidence for endpoint {endpoint!r}; unknown candidate fields remain"
            )
        return frozenset(candidates)

    def unique(self, endpoint: Endpoint, *, symmetry: str | None = None) -> AtomSite | None:
        candidates = self.resolve(endpoint, symmetry=symmetry)
        if len(candidates) > 1:
            raise ValueError("mmCIF struct_conn endpoint is ambiguous in the selected molecular declaration")
        return next(iter(candidates), None)
