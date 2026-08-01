r"""File conveniences for reading sources and writing molecular structures."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from ..data.molecular import ChemicalComponent
from ..data.structure import Structure
from .source import StructureSource, StructureView, read_source

_FORMATS: dict[str, Literal["pdb", "mmcif"]] = {".pdb": "pdb", ".cif": "mmcif", ".mmcif": "mmcif"}


def read_structure(
    path: str | os.PathLike[str] | StructureSource | StructureView,
    *,
    model_index: int | None = None,
    model_id: int | None = None,
    assembly_id: str | None = None,
    altloc: Literal["highest_occupancy", "first"] = "highest_occupancy",
    include_water: bool = True,
    include_hydrogens: bool = True,
    components: Mapping[str, ChemicalComponent] | None = None,
    component_resolver: Callable[[str], ChemicalComponent] | None = None,
) -> Structure:
    r"""
    Read one model and alternate-location realization from PDB or mmCIF.

    Args:
        path: PDB or mmCIF path (optionally gzip-compressed), a native source,
            or an existing view. Model and assembly selection belong to a supplied view.
        model_index: One-based position in source order; default is the first model.
        model_id: Exact stored model identifier; mutually exclusive with model_index.
        assembly_id: Biological assembly identifier, or `None` for the stored
            asymmetric unit.
        altloc: Residue-level alternate-location choice.
        include_water: Whether to retain water entities.
        include_hydrogens: Whether to retain hydrogen atoms in the molecular graph.
        components: Optional component definitions overriding built-in chemistry.
        component_resolver: Optional lazy component lookup used before the built-in
            component library. A missing component must raise ``KeyError``.

    Returns:
        A CPU tensor Structure whose atom axis is the complete retained system graph.

    Raises:
        ValueError: A supplied view is combined with model or assembly options,
            or the selected source has contradictory molecular evidence.

    Examples:
        >>> from ofoldx.data import ChemicalAtom, ChemicalComponent
        >>> from ofoldx.io import parse_source
        >>> source = parse_source(
        ...     "HETATM    1  C1  LIG A   1       1.000   2.000   3.000  1.00 10.00           C  \n",
        ...     format="pdb",
        ... )
        >>> ligand = ChemicalComponent("LIG", (ChemicalAtom("C1", "C"),), frozenset())
        >>> structure = read_structure(source.view(), components={"LIG": ligand})
        >>> structure.coordinates.tolist()
        [[1.0, 2.0, 3.0]]
    """

    from ..data.materialize import materialize_structure

    if isinstance(path, StructureView):
        if model_index is not None or model_id is not None or assembly_id is not None:
            raise ValueError("model/assembly selection belongs to the supplied StructureView")
        view = path
    else:
        source = path if isinstance(path, StructureSource) else read_source(path)
        view = source.view(model_index=model_index, model_id=model_id, assembly_id=assembly_id)
    return materialize_structure(
        view,
        altloc=altloc,
        include_water=include_water,
        include_hydrogens=include_hydrogens,
        components=components,
        component_resolver=component_resolver,
    )


def write_structure(structure: Structure, path: str | os.PathLike[str]) -> None:
    r"""
    Write a Structure as a standard mmCIF or PDB source format projection.

    These interchange formats express molecular and coordinate evidence through their
    own schemas. For complete tensor and graph storage, use
    [`save_structure`][ofoldx.data.structure.save_structure] with a
    ``.structure.safetensors`` artifact.

    mmCIF omits stored values at unobserved coordinate positions. The writer rejects
    chemistry that conflicts with the authoritative definition of its component ID.

    PDB preserves blank occupancy and B-factor fields. Its fixed-width numeric fields
    must preserve values at the input tensor dtype's precision. It likewise refuses a
    structure without author residue identifiers because manufacturing identifiers would
    change the public value; write mmCIF when those source identities are absent.

    PDB ``SEQRES`` records each polymer sequence, ``REMARK 465`` marks its unresolved
    positions, and observed non-polymers use ``HETATM`` plus ``CONECT``. A topology-only
    [`Structure`][ofoldx.data.structure.Structure] with no observed atoms has no PDB
    projection. PDB identifies a residue by chain, number, and insertion code alone,
    so two components in one instance cannot share one of those keys.

    Args:
        structure: Realization to write.
        path: Destination; the suffix selects mmCIF or PDB.

    Raises:
        ValueError: If the structure holds more than one molecular system, or carries
            occupancy or B-factor evidence for atoms without coordinates; or, writing PDB,
            if the structure observes no atom or two components in one instance share one
            author residue key.
    """

    if structure.system.system_entity_ptr.numel() != 2 or structure.system.system_instance_ptr.numel() != 2:
        raise ValueError("one structure file cannot represent multiple molecular systems")
    omitted = ~structure.coordinate_mask
    if bool((omitted & (structure.occupancy_mask | structure.b_factor_mask)).any()):
        raise ValueError(
            "a structure file cannot preserve occupancy or B-factor evidence for atoms without coordinates"
        )
    target, format_name = _structure_path(path)
    if format_name == "mmcif":
        from .mmcif import write_mmcif

        write_mmcif(structure, target)
    else:
        from .pdb import write_pdb

        write_pdb(structure, target)


def _structure_path(
    path: str | os.PathLike[str],
    *,
    allow_compressed: bool = False,
) -> tuple[Path, Literal["pdb", "mmcif"]]:
    resolved = Path(os.path.expanduser(os.fspath(path)))
    suffix = resolved.suffix.lower()
    if allow_compressed and suffix == ".gz":
        suffix = resolved.with_suffix("").suffix.lower()
    try:
        return resolved, _FORMATS[suffix]
    except KeyError:
        expected = ".pdb, .pdb.gz, .cif, .cif.gz, .mmcif, or .mmcif.gz" if allow_compressed else ".pdb, .cif, or .mmcif"
        raise ValueError(f"structure path suffix must be {expected}") from None
