"""Validated mmCIF entity, branch and author-address evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from ...io.declarations import EntityDeclarations
from ...io.records import AtomColumns, BranchResidueKey
from ...io.selection import _select_cif_declarations
from ...io.source import StructureView, _require_cif_source
from ..molecular import Molecule, MoleculeBond
from .plan import AtomRows

BranchBondEvidence: TypeAlias = tuple[tuple[MoleculeBond, bool, bool], ...]


@dataclass(frozen=True, kw_only=True)
class CifCatalog:
    view: StructureView
    declarations: EntityDeclarations
    entity_types: Mapping[str, str]
    branch_entities: Mapping[str, Mapping[str | None, Molecule]]
    branch_component_indices: Mapping[BranchResidueKey, int]
    branch_source_bonds: Mapping[tuple[str, str | None], BranchBondEvidence]

    @property
    def author_chains(self) -> Mapping[str, str]:
        """Borrow this source's read-only author names for the selected model."""
        return _require_cif_source(self.view.source).author_chains(self.view.model_index)

    def __post_init__(self) -> None:
        for name in ("entity_types", "branch_component_indices", "branch_source_bonds"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "branch_entities",
            MappingProxyType({key: MappingProxyType(dict(value)) for key, value in self.branch_entities.items()}),
        )

    @classmethod
    def for_template(cls, view: StructureView, rows: AtomRows) -> CifCatalog:
        return cls._validated(view, rows, {}, {}, {})

    @classmethod
    def _validated(
        cls,
        cif_file: StructureView,
        rows: AtomColumns,
        branch_entities: Mapping[str, Mapping[str | None, Molecule]],
        branch_component_indices: Mapping[BranchResidueKey, int],
        branch_source_bonds: Mapping[tuple[str, str | None], BranchBondEvidence],
    ) -> CifCatalog:
        declarations = _select_cif_declarations(cif_file)
        entity_types = dict(declarations.entity_types)
        asym_entities = declarations.asym_entities
        nonpoly_components = declarations.nonpolymer_components
        for entity_id in branch_entities:
            entity_type = entity_types.get(entity_id)
            if entity_type not in {None, "branched"}:
                raise ValueError(
                    f"mmCIF entity {entity_id!r} is declared {entity_type!r} but also has a branched-entity definition"
                )
            entity_types[entity_id] = "branched"
        missing_branch_definition = {
            entity_id
            for entity_id, entity_type in entity_types.items()
            if entity_type == "branched" and entity_id not in branch_entities
        }
        if missing_branch_definition:
            raise ValueError(
                f"mmCIF branched entities require branch declarations: {sorted(missing_branch_definition)!r}"
            )
        for raw_asym_id, raw_entity_id, raw_component_id in zip(
            rows.chain_id, rows.entity_id, rows.component_id, strict=True
        ):
            asym_id = str(raw_asym_id)
            source_entity_id = None if raw_entity_id is None else str(raw_entity_id)
            declared_entity_id = asym_entities.get(asym_id)
            if declared_entity_id is not None and declared_entity_id != source_entity_id:
                raise ValueError(f"mmCIF atom_site for asym {asym_id!r} disagrees with _struct_asym")
            declared_component_id = nonpoly_components.get(source_entity_id or "")
            if declared_component_id is not None and declared_component_id != str(raw_component_id):
                raise ValueError(
                    f"mmCIF atom_site for non-polymer entity {source_entity_id!r} disagrees with its component"
                )
        _require_cif_source(cif_file.source).author_chains(cif_file.model_index)
        return cls(
            view=cif_file,
            declarations=declarations,
            entity_types=entity_types,
            branch_entities=branch_entities,
            branch_component_indices=branch_component_indices,
            branch_source_bonds=branch_source_bonds,
        )

    def branch_entity(self, entity_id: str, asym_id: str) -> Molecule:
        variants = self.branch_entities[entity_id]
        entity = variants.get(asym_id)
        if entity is None:
            entity = variants.get(None)
        if entity is None:
            raise ValueError(f"cannot select a branched-entity realization for entity {entity_id!r}, asym {asym_id!r}")
        return entity

    def branch_bonds(self, entity_id: str, asym_id: str) -> BranchBondEvidence:
        key = (entity_id, asym_id) if asym_id in self.branch_entities[entity_id] else (entity_id, None)
        return self.branch_source_bonds[key]
