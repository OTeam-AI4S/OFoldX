r"""Template coordinate projection from the same source used by structure consumers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from ..data.coordinates import PolymerCoordinates
from .coordinates import read_polymer_coordinates
from .identity import InsufficientIdentityEvidenceError
from .polymer import POLYMER_TYPES
from .selection import _ALTLOC_POLICIES, _polymer_declarations
from .source import StructureSource, StructureView, _require_cif_source, read_source

if TYPE_CHECKING:
    from ..data.structure import Structure


def read_template_structure(
    path: str | os.PathLike[str] | StructureSource,
    chain_id: str,
    *,
    chain_id_kind: Literal["label", "auth"] = "label",
    model_index: int | None = None,
    model_id: int | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
) -> Structure:
    r"""Read one declared CIF protein chain without chemistry completion.

    ``chain_id_kind`` selects label or author naming explicitly; the resulting
    structure uses the source label chain identity. PDB is unsupported for this
    sequence projection; [`read_structure`][ofoldx.io.read_structure] accepts PDB.

    Args:
        path: CIF path, optionally gzip-compressed, or an existing source snapshot.
        chain_id: Protein chain to resolve in the requested namespace.
        chain_id_kind: ``"label"`` for CIF label naming or ``"auth"`` for author naming.
        model_index: One-based source position; defaults to the first stored model.
        model_id: Exact stored model ID, mutually exclusive with ``model_index``.
        altloc: Residue-level alternate-location selection policy.

    Returns:
        A Structure with declared sequence positions and observed heavy atoms.
            Chemical completion is not performed, and parent relationships remain complete.

    Raises:
        ofoldx.io.UnsupportedFormatError: If the source is not CIF.
        KeyError: If no protein chain matches in the requested namespace.
        ValueError: If selection is ambiguous, options or source evidence are invalid,
            or the chain has no observed heavy-atom coordinates.

    Examples:
        >>> from .source import parse_source
        >>> read_template_structure(parse_source("END\n", format="pdb"), "A")
        Traceback (most recent call last):
            ...
        ofoldx.io.source.UnsupportedFormatError: this operation supports CIF sources only
    """
    return read_template_structures(
        path, (chain_id,), chain_id_kind=chain_id_kind, model_index=model_index, model_id=model_id, altloc=altloc
    )[chain_id]


def read_template_structures(
    path: str | os.PathLike[str] | StructureSource,
    chain_ids: Sequence[str],
    *,
    chain_id_kind: Literal["label", "auth"] = "label",
    model_index: int | None = None,
    model_id: int | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    on_error: Callable[[str, KeyError | ValueError], None] | None = None,
) -> dict[str, Structure]:
    """Project CIF protein chains from one source and one selected model.

    Source parsing, model decoding, declaration and chain-naming conflicts always
    raise before ``on_error`` is called. The callback can handle individual chain
    lookup or projection failures. Returned keys use the requested namespace;
    contained structures use label identities. The source itself is never filtered.

    Args:
        path: CIF path, optionally gzip-compressed, or an existing source snapshot.
        chain_ids: Protein chain names; duplicates are resolved once in request order.
        chain_id_kind: Namespace of the requested names, ``"label"`` or ``"auth"``.
        model_index: One-based source position; defaults to the first stored model.
        model_id: Exact stored model ID, mutually exclusive with ``model_index``.
        altloc: Residue-level alternate-location selection policy.
        on_error: Optional callback receiving a requested chain name and its lookup
            or projection error. A handled chain is omitted from the result.

    Returns:
        Structures keyed by requested names, each using label chain identities.
            An empty request returns an empty mapping without reading a file.

    Raises:
        ofoldx.io.UnsupportedFormatError: If the source is not CIF.
        TypeError: If ``chain_ids`` is a single string rather than a sequence of names.
        KeyError: If a chain is absent and no callback handles the error.
        ValueError: If options or source evidence are invalid, or a chain failure
            is not handled by ``on_error``. Source-wide failures always propagate.

    Examples:
        >>> read_template_structures("unused.cif", [])
        {}
    """
    from ..data.materialize import materialize_template

    views = _template_views(
        path,
        chain_ids,
        chain_id_kind=chain_id_kind,
        model_index=model_index,
        model_id=model_id,
        altloc=altloc,
        on_error=on_error,
    )
    result = {}
    for chain, view in views.items():
        try:
            result[chain] = materialize_template(view, altloc=altloc)
        except (KeyError, ValueError) as error:
            if on_error is None:
                raise
            on_error(chain, error)
    return result


def read_template_coordinates(
    path: str | os.PathLike[str] | StructureSource,
    chain_ids: Sequence[str],
    *,
    chain_id_kind: Literal["label", "auth"] = "label",
    model_index: int | None = None,
    model_id: int | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    on_error: Callable[[str, KeyError | ValueError], None] | None = None,
) -> dict[str, PolymerCoordinates]:
    r"""Read CIF protein observations and sequence without constructing a graph.

    Returned dictionary keys retain the requested label/auth names; each
    [`PolymerCoordinates`][ofoldx.data.coordinates.PolymerCoordinates] uses label
    chain keys. Raw model decoding and global identity checks occur before any
    ``on_error`` callback. Only individual chain lookup/projection failures can
    be handled by that callback.

    Args:
        path: CIF path, optionally gzip-compressed, or an existing source snapshot.
        chain_ids: Protein chain names; duplicates are resolved once in request order.
        chain_id_kind: Namespace of the requested names, ``"label"`` or ``"auth"``.
        model_index: One-based source position; defaults to the first stored model.
        model_id: Exact stored model ID, mutually exclusive with ``model_index``.
        altloc: Residue-level alternate-location selection policy.
        on_error: Optional callback for individual chain lookup/projection errors;
            handled chains are omitted. Source-wide failures bypass the callback.

    Returns:
        Coordinate sets keyed by requested names. Each contains one label chain's
            declared sequence, heavy-atom observations, validity masks and parent lists.
            An empty request returns an empty mapping without reading a file.

    Raises:
        ofoldx.io.UnsupportedFormatError: If the source is not CIF.
        TypeError: If ``chain_ids`` is a single string rather than a sequence of names.
        KeyError: If a chain is absent and no callback handles the error.
        ValueError: If options or source evidence are invalid, or a chain is ambiguous
            or lacks observations and no callback handles that failure.

    Examples:
        >>> from .source import parse_source
        >>> read_template_coordinates(parse_source("END\n", format="pdb"), ["A"])
        Traceback (most recent call last):
            ...
        ofoldx.io.source.UnsupportedFormatError: this operation supports CIF sources only
    """
    views = _template_views(
        path,
        chain_ids,
        chain_id_kind=chain_id_kind,
        model_index=model_index,
        model_id=model_id,
        altloc=altloc,
        on_error=on_error,
    )
    result = {}
    for chain, view in views.items():
        try:
            coordinates = read_polymer_coordinates(view, altloc=altloc, include_hydrogens=False)
            if not any(value.coordinate_mask.any() for value in coordinates.chains.values()):
                raise ValueError("template chain has no observed heavy-atom coordinates")
            result[chain] = coordinates
        except (KeyError, ValueError) as error:
            if on_error is None:
                raise
            on_error(chain, error)
    return result


def _template_views(
    path: str | os.PathLike[str] | StructureSource,
    chain_ids: Sequence[str],
    *,
    chain_id_kind: Literal["label", "auth"],
    model_index: int | None,
    model_id: int | None,
    altloc: Literal["highest_occupancy", "first"],
    on_error: Callable[[str, KeyError | ValueError], None] | None,
) -> dict[str, StructureView]:
    if isinstance(chain_ids, str):
        raise TypeError("chain_ids must be a sequence of chain identifiers")
    if any(not isinstance(chain, str) or not chain for chain in chain_ids):
        raise ValueError("chain_id must be a non-empty string")
    if chain_id_kind not in {"label", "auth"}:
        raise ValueError(f"unsupported chain_id_kind: {chain_id_kind!r}")
    if altloc not in _ALTLOC_POLICIES:
        raise ValueError(f"unsupported altloc policy: {altloc!r}")
    if not chain_ids:
        return {}
    source = _require_cif_source(path if isinstance(path, StructureSource) else read_source(path))
    model_view = source.view(model_index=model_index, model_id=model_id)
    # Decode the entire selected model before per-chain callbacks. A malformed
    # source must not become an apparently successful empty template collection.
    source.atoms.by_index(model_view.model_index)
    declarations = _polymer_declarations(model_view)
    author_chains = source.author_chains(model_view.model_index)
    # Validate parent declarations before callbacks; both projections reuse this value.
    _ = source.component_parents
    protein_entities = {
        entity for entity, kind in declarations.polymer_types.items() if POLYMER_TYPES.get(kind.casefold()) == "protein"
    }
    protein_chains = {chain for chain, entity in declarations.asym_entities.items() if entity in protein_entities}
    result = {}
    for chain in dict.fromkeys(chain_ids):
        try:
            if chain_id_kind == "label":
                matches = {chain} & protein_chains
            else:
                unknown = protein_chains - author_chains.keys()
                if unknown:
                    raise InsufficientIdentityEvidenceError(
                        f"author template chain {chain!r} cannot be resolved: "
                        f"unknown author names for {sorted(unknown)!r}"
                    )
                matches = {label for label in protein_chains if author_chains[label] == chain}
            if not matches:
                raise KeyError(f"no protein template chain {chain!r} in {chain_id_kind} namespace")
            if len(matches) != 1:
                raise ValueError(f"ambiguous author template chain {chain!r}: {sorted(matches)!r}")
            label = next(iter(matches))
            view = StructureView(source, model_index=model_view.model_index, asym_ids=frozenset({label}))
            if not len(view.rows):
                raise ValueError(f"template chain {label!r} has no atoms in requested model {model_view.model_index}")
            result[chain] = view
        except (KeyError, ValueError) as error:
            if on_error is None:
                raise
            on_error(chain, error)
    return result
