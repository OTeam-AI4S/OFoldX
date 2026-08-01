# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Sequence, MSA and template conveniences for task Pipelines.

Family pipelines take a ``Structure`` (models never import ``ofoldx.io``); structure files are read above them
(``ofoldx.api``, ``ofoldx.runners``). A ``.safetensors`` path is a native OFoldX artifact.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
from multimolecule.io import read_fasta_records

from ..chemistry.construction import PolymerSequence, Recipe, build_system, polymer_sequence
from ..chemistry.definitions import ComponentLibrary
from ..data.molecular import MolecularSystem, MonomerType
from ..data.structure import Structure
from ..errors import RequestRef, SourceRef, StructureRef, fail
from ..msa import MSA, MsaCollection, load_msa_collection, read_msa
from ..msa.alignment import instance_monomer
from ..references import classify_reference
from ..templates import Template, load_templates

_PROTEIN_FASTA_SUFFIXES = frozenset((".fa", ".faa", ".fasta", ".seq"))


def resolve_polymer_input(
    input: Structure | PolymerSequence | Sequence[PolymerSequence] | str | PathLike[str],
    *,
    library: ComponentLibrary | None = None,
) -> Structure:
    r"""
    A structure as given, or one built from polymer sequences with nothing observed.

    Inline text and protein FASTA files (``.fa``, ``.faa``, ``.fasta``, ``.seq``) are protein one-letter sequences
    (``polymer_sequence``); nucleic acids need a typed ``PolymerSequence``, because one-letter text alone does not
    distinguish DNA from RNA. Instances are named by the FASTA record ids, else ``chain_1``, ``chain_2``, …;
    identical sequences are one entity.

    Raises:
        InvalidRequestError: ``format_unknown`` for a path that is not a protein FASTA file.
        ContradictionError: ``sequence_name_duplicate`` when two FASTA records name one instance.
    """

    if isinstance(input, Structure):
        return input
    if isinstance(input, PolymerSequence):
        named = [("chain_1", input)]
    elif not isinstance(input, str | PathLike):
        named = [(f"chain_{index + 1}", sequence) for index, sequence in enumerate(input)]
    else:
        classified = classify_reference(input)
        if not isinstance(classified, Path):
            named = [("chain_1", polymer_sequence(classified.strip().upper(), MonomerType.PROTEIN))]
        elif not {suffix.lower() for suffix in classified.suffixes} & _PROTEIN_FASTA_SUFFIXES:
            fail("format_unknown", RequestRef("input", str(classified)))
        else:
            named = [
                (
                    record.id or f"chain_{index + 1}",
                    polymer_sequence(record.sequence.strip().upper(), MonomerType.PROTEIN),
                )
                for index, record in enumerate(read_fasta_records(classified))
            ]
            names = [name for name, _ in named]
            for index, name in enumerate(names):
                if name in names[:index]:
                    fail("sequence_name_duplicate", SourceRef(str(classified), index), instance=name)
    entities = {sequence: str(index + 1) for index, sequence in enumerate(dict.fromkeys(s for _, s in named))}
    recipe = Recipe({entity: sequence for sequence, entity in entities.items()}, {n: entities[s] for n, s in named})
    system, _, _ = build_system(recipe, library)
    return Structure.unobserved(system)


def entity_pools(msa: MSA, system: MolecularSystem) -> dict[str, MSA]:
    r"""
    ``msa`` bound to every instance of its entity (``msa.entity_id``).

    Raises:
        InvalidRequestError: ``msa_entity_unknown`` when no instance belongs to that entity.
    """

    entity = system.text("entity", system.entity_symbol) == msa.entity_id
    instances = np.flatnonzero(entity[system.instance_entity])
    if not len(instances):
        fail("msa_entity_unknown", RequestRef("entity_id", msa.entity_id))
    return dict.fromkeys(system.text("instance", system.instance_symbol[instances]).tolist(), msa)


def resolve_msa_input(
    msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None,
    system: MolecularSystem,
    *,
    monomer_types: Collection[MonomerType],
) -> Mapping[str, MSA] | MsaCollection | None:
    r"""
    Pools by instance or an exact plan; a path is read.

    A ``.safetensors`` path is a native ``MsaCollection``, loaded exactly. Any other path is a text alignment (A3M or
    aligned FASTA, every row as written) of the one entity whose instances are of ``monomer_types``
    (``instance_monomer``), bound to every instance of that entity.

    Raises:
        InsufficientInformationError: ``msa_entity_unresolved`` when no entity or several entities match.
    """

    if not isinstance(msa, str | PathLike):
        return msa
    path = Path(msa).expanduser()
    if path.suffix == ".safetensors":
        return load_msa_collection(path)
    matching = np.isin(instance_monomer(system), [int(monomer) for monomer in monomer_types])
    entities, first = np.unique(system.instance_entity[matching], return_index=True)
    if len(entities) != 1:
        instances = np.flatnonzero(matching)[first]
        fail(
            "msa_entity_unresolved",
            RequestRef("msa", str(path)),
            candidates=tuple(StructureRef(instance=int(instance)) for instance in instances),
        )
    entity_id = str(system.text("entity", system.entity_symbol[entities[:1]])[0])
    return entity_pools(read_msa(path, entity_id=entity_id), system)


def resolve_template_input(templates: Sequence[Template] | str | PathLike[str]) -> tuple[Template, ...]:
    r"""Templates as given, or the native template artifact at a path."""

    if isinstance(templates, str | PathLike):
        return load_templates(Path(templates).expanduser())
    return tuple(templates)
