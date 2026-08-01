r"""Source structure declarations, independent of chemical graph construction."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from .cif import CifBlock, CifCategory, CifDocument, parse_cif, read_cif
from .records import _ATOM_ROW_DTYPE, reject_duplicate_atom_rows

if TYPE_CHECKING:
    from .declarations import EntityDeclarations
    from .identity import AtomIndex, Connection


class UnsupportedFormatError(NotImplementedError):
    """The requested operation is not supported for this source format.

    Missing or contradictory evidence in a supported format raises a data error,
    not this exception. Raw source inspection remains available in either case.
    """


class ModelTable(ABC):
    """Decoded observations indexed in source order, without choosing a realization.

    Coordinates use float64 storage. Occupancy and B-factor columns hold Python
    floats or None. Model consumers choose their computation dtype explicitly.
    """

    @property
    @abstractmethod
    def model_ids(self) -> tuple[int | None, ...]:
        """Source identifiers; None denotes an implicit PDB model without MODEL."""

    def index_of(self, model_id: int) -> int:
        """Locate a stored model identifier in source order.

        Args:
            model_id: Exact integer identifier recorded by the source.

        Returns:
            The positive, one-based model position.

        Raises:
            TypeError: If the identifier is not an integer, or is a boolean.
            ValueError: If no model has that identifier.
        """
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            raise TypeError("model_id must be an integer source identifier")
        try:
            return self.model_ids.index(model_id) + 1
        except ValueError:
            raise ValueError(f"source does not contain model_id {model_id!r}") from None

    def by_id(self, model_id: int) -> np.recarray:
        """Read deposited observations for an exact source model identifier.

        Args:
            model_id: Stored identifier, independent of its position in the file.

        Returns:
            Read-only atom rows retaining all alternate locations in source order.

        Raises:
            TypeError: If the identifier is not an integer, or is a boolean.
            ValueError: If the identifier is absent or the model data is invalid.
        """
        return self.by_index(self.index_of(model_id))

    def by_index(self, model_index: int = 1) -> np.recarray:
        """Read deposited observations by position in the source.

        Args:
            model_index: Positive, one-based position; defaults to the first model.

        Returns:
            Read-only atom rows, decoded once and reused for subsequent reads.
                Chain selection and assembly expansion have not been applied.

        Raises:
            ValueError: If the position is invalid or the model data cannot be decoded.
        """
        if (
            isinstance(model_index, bool)
            or not isinstance(model_index, int)
            or not 1 <= model_index <= len(self.model_ids)
        ):
            raise ValueError(f"source does not contain requested model_index {model_index!r}")
        if model_index not in self._models:
            rows = self._read_model(self.model_ids[model_index - 1])
            rows.setflags(write=False)
            self._models[model_index] = rows
        return self._models[model_index]

    def __init__(self) -> None:
        self._models: dict[int, np.recarray] = {}

    @abstractmethod
    def _read_model(self, model_id: int | None) -> np.recarray:
        """Decode one stored model; no assembly expansion or chemical completion."""


class CifAtomTable(ModelTable):
    """Original CIF atom-site columns with stable source keys.

    Args:
        category: The source ``atom_site`` category, or ``None`` if absent.
    """

    def __init__(self, category: CifCategory | None) -> None:
        super().__init__()
        self.category = category
        self._indices: dict[int, AtomIndex] = {}

    def index(self, model_index: int = 1) -> AtomIndex:
        r"""Index CIF label/auth identities in the entire deposited model.

        Chain selection must follow identity resolution. PDB model tables do not
        expose this CIF namespace operation.

        Args:
            model_index: One-based position in source order, independent of model ID.

        Returns:
            The model's reusable [`AtomIndex`][ofoldx.io.identity.AtomIndex],
                with separate label/auth namespaces and positions into deposited rows.

        Raises:
            ValueError: If the model position is invalid or its atom data is invalid.

        Examples:
            >>> source = CifSource(parse_cif("data_x\n_entry.id x\n"))
            >>> source.atoms.index().size
            0
        """
        from .identity import AtomIndex

        rows = self.by_index(model_index)
        if model_index not in self._indices:
            self._indices[model_index] = AtomIndex(rows)
        return self._indices[model_index]

    @cached_property
    def model_ids(self) -> tuple[int | None, ...]:
        if self.category is None or not self.category.row_count:
            return (None,)
        column = self.category.get("pdbx_PDB_model_num")
        if column is None or not column.present.all():
            raise ValueError("mmCIF _atom_site.pdbx_PDB_model_num is required for every atom")
        return tuple(dict.fromkeys(column.as_array(np.int64).tolist()))

    def _read_model(self, model_id: int | None) -> np.recarray:
        category = self.category
        if category is None or not category.row_count:
            return np.empty(0, dtype=_ATOM_ROW_DTYPE).view(np.recarray)
        source_ids = category.strings("id")
        if any(value is None or not value for value in source_ids):
            raise ValueError("mmCIF _atom_site.id is required")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("mmCIF _atom_site.id values must be unique")
        selected = np.flatnonzero(category["pdbx_PDB_model_num"].as_array(np.int64) == model_id)
        rows = np.empty(len(selected), dtype=_ATOM_ROW_DTYPE).view(np.recarray)

        def column(name: str) -> np.ndarray:
            return np.asarray(category.strings(name), dtype=object)[selected]

        def integers(name: str) -> np.ndarray:
            source = category.get(name)
            if source is None:
                return np.full(len(selected), None, dtype=object)
            values = source.as_array(np.int64, 0)[selected].astype(object)
            values[~source.present[selected]] = None
            return values

        def required(name: str) -> np.ndarray:
            values = column(name)
            if np.any(values == None) or np.any(values == ""):  # noqa: E711
                raise ValueError(f"mmCIF _atom_site.{name} is required for every selected atom")
            return values

        rows.row_index = selected
        rows.serial = np.asarray(source_ids, dtype=object)[selected]
        # Label and author namespaces are never spliced together. CIF schema
        # identity is label-based; optional author naming is retained separately.
        rows.chain_id = required("label_asym_id")
        rows.component_id = required("label_comp_id")
        rows.atom_name = required("label_atom_id")
        groups = required("group_PDB")
        if any(group not in {"ATOM", "HETATM"} for group in groups):
            raise ValueError("mmCIF _atom_site.group_PDB must be ATOM or HETATM")
        rows.group = groups
        rows.label_seq_id = integers("label_seq_id")
        for target, source in (
            ("entity_id", "label_entity_id"),
            ("auth_chain_id", "auth_asym_id"),
            ("auth_residue_id", "auth_seq_id"),
            ("auth_component_id", "auth_comp_id"),
            ("auth_atom_name", "auth_atom_id"),
            ("insertion_code", "pdbx_PDB_ins_code"),
            ("altloc_id", "label_alt_id"),
        ):
            rows[target] = column(source)
        # Internal occurrence keys are factorized identities, not inferred
        # sequence numbers or boundaries in the physical ordering of atom rows.
        occurrences: dict[str, dict[tuple, int]] = {}
        residue_ids = []
        for chain, label, author, insertion in zip(
            rows.chain_id,
            rows.label_seq_id,
            rows.auth_residue_id,
            rows.insertion_code,
            strict=True,
        ):
            chain_occurrences = occurrences.setdefault(chain, {})
            occurrence = chain_occurrences.setdefault((label, author, insertion), len(chain_occurrences) + 1)
            residue_ids.append(label if label is not None else occurrence)
        rows.residue_id = residue_ids
        symbols, inverse = np.unique(required("type_symbol"), return_inverse=True)
        elements = [value.capitalize() for value in symbols]
        rows.isotope = np.asarray([{"D": 2, "T": 3}.get(value) for value in elements], dtype=object)[inverse]
        rows.element = np.asarray(["H" if value in {"D", "T"} else value for value in elements], dtype=object)[inverse]
        rows.charge = integers("pdbx_formal_charge")
        for target, source in (("occupancy", "occupancy"), ("b_factor", "B_iso_or_equiv")):
            if source not in category:
                rows[target] = None
            else:
                numbers = category[source].as_array(np.float64, np.nan)[selected]
                values = numbers.astype(object)
                values[~category[source].present[selected]] = None
                rows[target] = values
        if "occupancy" not in category:
            # PDBx _atom_site.occupancy defines 1.0 as its absent-item default.
            # Explicit unknown/inapplicable cells above remain missing.
            rows.occupancy = 1.0
        for axis, name in enumerate(("Cartn_x", "Cartn_y", "Cartn_z")):
            rows.coordinates[:, axis] = category[name].as_array(np.float64, np.nan)[selected]
        coordinate_present = np.isfinite(rows.coordinates).all(axis=1)
        rows.coordinate_present = coordinate_present
        rows.coordinates[~coordinate_present] = 0.0
        rows["symmetry_id"] = None
        reject_duplicate_atom_rows(rows)
        return rows


class StructureSource(ABC):
    """A native source snapshot, reusable by independent coordinate and graph consumers."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def atoms(self) -> ModelTable: ...

    def view(
        self,
        *,
        model_index: int | None = None,
        model_id: int | None = None,
        assembly_id: str | None = None,
        asym_ids: frozenset[str] | None = None,
    ) -> StructureView:
        r"""Select once. Default is the first stored model; ID and index are exclusive.

        asym_ids refers to CIF label_asym_id or the native PDB chain column.
        rows always exposes deposited observations; assembly_id is a request to
        materializers and never changes the source snapshot.

        Args:
            model_index: One-based source position. Omit to use the first model
                or select by ``model_id``.
            model_id: Exact stored identifier; mutually exclusive with ``model_index``.
            assembly_id: Assembly request for materialization; deposited rows stay unexpanded.
            asym_ids: CIF label chain IDs or PDB native chain IDs to retain.
                ``None`` selects all chains; an empty set selects no atoms.

        Returns:
            A [`StructureView`][ofoldx.io.source.StructureView] retaining this source
                and the resolved model position. Atom decoding remains lazy.

        Raises:
            TypeError: If ``model_id`` is not an integer, or is a boolean.
            ValueError: If model options conflict, the model is absent, or source
                model identifiers are invalid.

        Examples:
            >>> source = parse_source("END\n", format="pdb")
            >>> source.view(asym_ids={"A"}).asym_ids == frozenset({"A"})
            True
            >>> source.view(model_index=1, model_id=7)
            Traceback (most recent call last):
                ...
            ValueError: model_index and model_id are mutually exclusive
        """
        if model_index is not None and model_id is not None:
            raise ValueError("model_index and model_id are mutually exclusive")
        index = self.atoms.index_of(model_id) if model_id is not None else (1 if model_index is None else model_index)
        return StructureView(self, model_index=index, assembly_id=assembly_id, asym_ids=asym_ids)


@dataclass(frozen=True)
class CifSource(StructureSource):
    """Original CIF document and its format-specific declarations."""

    document: CifDocument
    _author_chains: dict[int, Mapping[str, str]] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.document.name

    @property
    def block(self) -> CifBlock:
        return self.document.block

    @cached_property
    def atoms(self) -> CifAtomTable:
        """CIF atom observations with model selection and label/auth identity indices."""
        return CifAtomTable(self.block.get("atom_site"))

    @cached_property
    def declarations(self) -> EntityDeclarations:
        from .declarations import read_entity_declarations

        return read_entity_declarations(self.block)

    @cached_property
    def connections(self) -> tuple[Connection, ...]:
        from .identity import read_connections

        return read_connections(self.block.get("struct_conn"))

    def author_chains(self, model_index: int = 1) -> Mapping[str, str]:
        r"""Read known label-to-author chain names once per deposited model.

        Args:
            model_index: One-based model position in source order.

        Returns:
            A read-only mapping shared by this source's coordinate and graph
                consumers. Chains without author evidence are absent from the mapping.

        Raises:
            ofoldx.io.identity.IdentityConflictError: If known author names conflict.
            ValueError: If the model position or decoded observations are invalid.

        Examples:
            >>> source = CifSource(parse_cif("data_x\n_pdbx_poly_seq_scheme.asym_id A\n"
            ...                             "_pdbx_poly_seq_scheme.pdb_strand_id Q\n"))
            >>> source.author_chains()["A"]
            'Q'
        """
        from .identity import _read_cif_chain_authors

        self.atoms.by_index(model_index)
        if model_index not in self._author_chains:
            self._author_chains[model_index] = MappingProxyType(_read_cif_chain_authors(self, model_index))
        return self._author_chains[model_index]

    @cached_property
    def component_parents(self) -> Mapping[str, tuple[str, ...]]:
        r"""Read complete component-parent relationships once for this source.

        Returns:
            A read-only mapping whose parent tuples retain source order. Missing
                declarations create no relationship; no single parent is chosen.

        Raises:
            ValueError: If a parent relationship is malformed or contradictory.

        Examples:
            >>> source = CifSource(parse_cif("data_x\n_chem_comp.id CRO\n"
            ...                             "_chem_comp.mon_nstd_parent_comp_id 'SER,TYR,GLY'\n"))
            >>> source.component_parents["CRO"]
            ('SER', 'TYR', 'GLY')
        """
        from .declarations import read_component_parents

        return MappingProxyType(read_component_parents(self.block.get("chem_comp")))


@dataclass(frozen=True)
class PdbSource(StructureSource):
    """Original PDB records, without inventing CIF categories or chemical topology."""

    text: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.text.isascii():
            raise ValueError("PDB input must be ASCII text")

    @cached_property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.text.splitlines())

    @cached_property
    def atoms(self) -> ModelTable:
        from .pdb_source import PdbAtomTable

        return PdbAtomTable(self.lines)


def read_source(path: str | os.PathLike[str]) -> StructureSource:
    """Read CIF/PDB, optionally gzip-compressed, without selecting or preprocessing.

    An unsupported suffix raises [`UnsupportedFormatError`][ofoldx.io.UnsupportedFormatError].
    Successful source reading does not establish sufficient coordinate or graph
    evidence; those requirements belong to the corresponding consumers.
    CIF parser errors propagate from [`read_cif`][ofoldx.io.cif.read_cif].

    Args:
        path: A ``.cif``, ``.mmcif`` or ``.pdb`` file, optionally ending in ``.gz``.

    Returns:
        A native [`CifSource`][ofoldx.io.source.CifSource] or
            [`PdbSource`][ofoldx.io.source.PdbSource] retaining the file contents.

    Raises:
        UnsupportedFormatError: If the suffix does not identify a supported format.
        OSError: If the file cannot be read.
        ValueError: If PDB text is not ASCII.

    Examples:
        >>> read_source("unavailable.xyz")
        Traceback (most recent call last):
            ...
        ofoldx.io.source.UnsupportedFormatError: structure path suffix must be .pdb, .cif or .mmcif, optionally .gz
    """
    import gzip

    path = Path(os.path.expanduser(os.fspath(path)))
    suffix = path.with_suffix("").suffix.lower() if path.suffix.lower() == ".gz" else path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return CifSource(read_cif(path))
    if suffix != ".pdb":
        raise UnsupportedFormatError("structure path suffix must be .pdb, .cif or .mmcif, optionally .gz")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, mode="rt", encoding="ascii") as handle:
            return PdbSource(handle.read(), path.stem)
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error


def parse_source(text: str, *, format: Literal["pdb", "mmcif"]) -> StructureSource:
    r"""Read native text without selecting coordinates or requiring chemistry.

    CIF parser errors propagate from [`parse_cif`][ofoldx.io.cif.parse_cif].

    Args:
        text: Complete CIF or ASCII PDB document text.
        format: Explicit native format, ``"mmcif"`` or ``"pdb"``.

    Returns:
        A native source snapshot. Atom decoding and evidence validation are deferred
            until a consumer requests those values.

    Raises:
        UnsupportedFormatError: If ``format`` is unsupported.
        ValueError: If PDB text is not ASCII.

    Examples:
        >>> source = parse_source("data_demo\n_entry.id demo\n", format="mmcif")
        >>> source.block["entry"].strings("id")
        ('demo',)
        >>> isinstance(parse_source("END\n", format="pdb"), PdbSource)
        True
    """
    if format == "mmcif":
        return CifSource(parse_cif(text))
    if format == "pdb":
        return PdbSource(text)
    raise UnsupportedFormatError(f"unsupported structure format: {format!r}")


def _require_cif_source(source: StructureSource) -> CifSource:
    """Narrow a source at a CIF consumer boundary, before reading declarations."""
    if not isinstance(source, CifSource):
        raise UnsupportedFormatError("this operation supports CIF sources only")
    return source


@dataclass(frozen=True)
class StructureView:
    r"""A model/chain selection shared by CIF and PDB consumers.

    ``asym_ids`` selects CIF label identifiers or native PDB chain identifiers.
    ``rows`` always contains deposited observations, even with an assembly request.
    CIF declarations and namespace indices belong to [`CifSource`][ofoldx.io.CifSource].

    Args:
        source: Native source snapshot shared by independent consumers.
        model_index: One-based source position. To select by stored ID, use ``source.view``.
        assembly_id: Optional assembly request passed to a materializer.
        asym_ids: Source chain IDs to retain; ``None`` retains all chains.

    Attributes:
        model_id: Stored identifier for the selected position, or ``None`` for an
            implicit PDB model.
        rows: Read-only deposited atoms after model and chain selection, retaining
            alternate locations and coordinate validity masks.

    Raises:
        ValueError: If the model position or stored model identifiers are invalid.

    Examples:
        >>> atoms = (
        ...     "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C  \n"
        ...     "ATOM      2  CA  GLY B   1       4.000   5.000   6.000  1.00 10.00           C  \n"
        ... )
        >>> source = parse_source(f"MODEL        7\n{atoms}ENDMDL\n", format="pdb")
        >>> view = source.view(model_id=7, asym_ids={"B"})
        >>> view.model_index, view.model_id
        (1, 7)
        >>> view.rows.chain_id.tolist()
        ['B']
        >>> view.rows.coordinates.tolist()
        [[4.0, 5.0, 6.0]]
    """

    source: StructureSource
    model_index: int = 1
    assembly_id: str | None = None
    asym_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_index, bool)
            or not isinstance(self.model_index, int)
            or not 1 <= self.model_index <= len(self.source.atoms.model_ids)
        ):
            raise ValueError(f"source does not contain requested model_index {self.model_index!r}")
        if self.asym_ids is not None:
            object.__setattr__(self, "asym_ids", frozenset(self.asym_ids))

    @property
    def model_id(self) -> int | None:
        return self.source.atoms.model_ids[self.model_index - 1]

    @cached_property
    def rows(self) -> np.recarray:
        rows = self.source.atoms.by_index(self.model_index)
        if self.asym_ids is not None:
            rows = rows[np.isin(rows.chain_id, tuple(self.asym_ids))].view(np.recarray)
            rows.setflags(write=False)
        return rows
