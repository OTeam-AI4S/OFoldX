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

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import chanfig
import numpy as np
import pytest
import torch
from torch import nn

from ofoldx.chemistry.construction import Molecule, PolymerSequence, Recipe, build_system
from ofoldx.chemistry.definitions import (
    ChemicalComponent,
    ComponentItems,
    ComponentLibrary,
    define_component,
    load_library,
    write_artifact,
)
from ofoldx.data import Structure
from ofoldx.data.molecular import DefinitionScope
from ofoldx.data.structure import CoordinateEvidence, replace_coordinates
from ofoldx.errors import InvalidRequestError
from ofoldx.io.cif import read_cif
from ofoldx.io.mmcif.decode import decode_components
from ofoldx.model import Model
from ofoldx.models.registry import MODELS
from ofoldx.processor import Processor

from .chemistry.builders import SUBSET, provenance


class RepositoryConfig(chanfig.Config):
    width: int = 3


class RepositoryProcessorConfig(chanfig.Config):
    scale: int = 1


class RepositoryModel(Model):
    config_class = RepositoryConfig

    def __init__(self, config: RepositoryConfig | None = None) -> None:
        config = RepositoryConfig() if config is None else config
        super().__init__(config)
        self.weight = nn.Parameter(torch.arange(config.width, dtype=torch.float32))
        self.bias = nn.Parameter(torch.arange(config.width, dtype=torch.float32) + config.width)


class RepositoryProcessor(Processor):
    config_class = RepositoryProcessorConfig
    config: RepositoryProcessorConfig

    def __init__(
        self,
        config: RepositoryProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = RepositoryProcessorConfig() if config is None else config


MODELS.register("test_repository", model=RepositoryModel, processor=RepositoryProcessor)


@pytest.fixture
def repository_model() -> RepositoryModel:
    return RepositoryModel()


@pytest.fixture
def repository_processor() -> RepositoryProcessor:
    return RepositoryProcessor(RepositoryProcessorConfig({"scale": 2}), dtype=torch.float16)


def _beta_backbone() -> np.ndarray:
    # Residues 1 and 3 have reciprocal C=O...H-N interactions, each across chains.
    # Their O--N, C--H, O--H and C--N distances are 2.8, 2.99, 1.79 and 4 A.
    return np.array(
        [
            [[100, 2, 0], [101, 2, 0], [1.3, 0, 0], [2.5, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [-2.8, 0, 0], [-4, 0, 0]],
            [[-100, 2, 0], [-101, 2, 0], [-8.1, 0, 0], [-9.3, 0, 0]],
            [[-6.8, 0, 0], [-7.8, 0, 0], [-4, 0, 0], [-2.8, 0, 0]],
        ],
        dtype=np.float64,
    )


def _helix_backbone() -> np.ndarray:
    nitrogen = np.stack((np.arange(7) * 10.0, np.zeros(7), np.zeros(7)), axis=-1)
    backbone = np.stack([nitrogen + [offset, 0, 0] for offset in (0.0, 1.0, 11.3, 12.5)], axis=1)
    # Two consecutive i->i+4 bonds define an alpha helix at positions 1..4.
    backbone[0, 2:, 0] = [36.0, 37.2]
    backbone[1, 2:, 0] = [46.0, 47.2]
    return backbone


@pytest.fixture
def secondary_backbones():
    return {"beta": _beta_backbone(), "helix": _helix_backbone()}


_AMINO_ACIDS = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
)


@pytest.fixture
def invalid() -> Callable[..., None]:
    """Assert that a call raises ``InvalidRequestError`` with the given code."""

    def check(call, *args, code: str = "value_invalid", **kwargs) -> None:
        with pytest.raises(InvalidRequestError) as caught:
            call(*args, **kwargs)
        assert caught.value.code == code

    return check


@pytest.fixture
def polymers():
    """
    One canonical system from ``(instance, entity, component ids)`` chains; every component has two atoms (``CA``,
    ``CB``) and no bond. A component's monomer type comes from its id: an amino-acid id is protein, ``A C G U N`` RNA,
    ``DA DC DG DT DN`` DNA, any other id none. An entity with a typed component is a polymer, else a non-polymer.
    """

    from ofoldx.data.molecular import MolecularSystem, SystemArrays
    from ofoldx.provenance import Basis, Resolution, state

    unknown, declared, defined = (
        state(Resolution.UNKNOWN, Basis.NONE),
        state(Resolution.KNOWN, Basis.DECLARED),
        state(Resolution.KNOWN, Basis.DEFINED),
    )

    def monomer(name: str) -> int:
        if name in _AMINO_ACIDS:
            return 1
        if name in ("A", "C", "G", "U", "N"):
            return 2
        return 3 if name in ("DA", "DC", "DG", "DT", "DN") else 0

    def build(*chains: tuple[str, str, tuple[str, ...]]) -> MolecularSystem:
        entities = list(dict.fromkeys(entity for _, entity, _ in chains))
        names = [name for _, _, components in chains for name in components]
        family = tuple(dict.fromkeys(names))
        types = np.array([monomer(name) for name in names], dtype=np.uint8)
        sizes = {"system": 1, "entity": len(entities), "instance": len(chains), "component": len(names)}
        sizes |= {"atom": 2 * len(names), "transform": 1}
        columns = {}
        for field in MolecularSystem.SCHEMA.fields:
            rows = sizes.get(field.axis.removesuffix("+1"), 0) + field.axis.endswith("+1")
            fill = unknown if field.state else -1 if field.symbols or field.name == "component_port" else 0
            columns[field.name] = np.full((*field.leading, rows, *field.trailing), fill, dtype=field.dtype)
        typed = dict.fromkeys(entities, False)
        for _, entity, components in chains:
            typed[entity] |= any(monomer(name) for name in components)
        lengths = [len(components) for _, _, components in chains]
        columns |= {
            "system_entity_ptr": np.array([0, len(entities)]),
            "system_instance_ptr": np.array([0, len(chains)]),
            "entity_symbol": np.arange(len(entities)),
            "entity_type": np.array([1 if typed[entity] else 2 for entity in entities]),
            "entity_type_state": np.full(len(entities), declared),
            "instance_symbol": np.arange(len(chains)),
            "instance_entity": np.array([entities.index(entity) for _, entity, _ in chains]),
            "instance_component_ptr": np.concatenate(([0], np.cumsum(lengths))),
            "instance_connections_complete": np.ones(len(chains), dtype=bool),
            "component_symbol": np.array([family.index(name) for name in names]),
            "component_monomer_type": types,
            "component_monomer_type_state": np.where(types > 0, defined, unknown),
            "component_atom_ptr": 2 * np.arange(len(names) + 1),
            "atom_symbol": np.tile([0, 1], len(names)),
            "atom_reference_row": np.full(2 * len(names), -1),
            "transform_rotation": np.eye(3)[None],
            "transform_code": np.array([-1]),
            "transform_state": np.array([defined]),
        }
        symbols = {name: () for name in ("connection_type", "parent", "polymer_type")}
        symbols |= {"entity": tuple(entities), "instance": tuple(name for name, _, _ in chains), "component": family}
        return MolecularSystem.from_arrays(SystemArrays(symbols | {"atom": ("CA", "CB")}, columns))

    return build


@pytest.fixture(scope="session")
def ccd() -> Mapping[str, ComponentItems]:
    r"""The CCD subset decoded through io, by component id."""
    cif = read_cif(SUBSET)
    return {
        block.name: decode_components(block.categories, block=block.name).components[block.name] for block in cif.blocks
    }


@pytest.fixture(scope="session")
def archive(ccd) -> Callable[[str], ChemicalComponent]:
    r"""The ARCHIVE definition of a subset id (defined once per id)."""
    cache: dict[str, ChemicalComponent] = {}

    def get(component_id: str) -> ChemicalComponent:
        if component_id not in cache:
            cache[component_id] = define_component(ccd[component_id], scope=DefinitionScope.ARCHIVE)
        return cache[component_id]

    return get


@pytest.fixture(scope="session")
def ccd_artifact(ccd, archive, tmp_path_factory) -> Path:
    """The CCD subset of ``tests/fixtures/ccd`` written as an artifact (ALA, GLY, SER, LYS, MSE, ACT, ZN, HOH, …)."""
    path = tmp_path_factory.mktemp("ccd") / "subset.safetensors"
    write_artifact([archive(component_id) for component_id in ccd], path, provenance("subset"))
    return path


@pytest.fixture(scope="session")
def library(ccd_artifact) -> ComponentLibrary:
    """The CCD subset artifact, loaded."""
    return load_library(ccd_artifact)


@pytest.fixture
def peptide(library) -> Callable[..., Structure]:
    """``peptide("ALA", "MSE", ..., copies=1, ligands=("ACT",))``: chains ``A``, ``B``, … of those residues (``copies``
    instances of entity ``P``) and one instance per ligand after them, every atom placed at its reference coordinates
    shifted 4 Å per component, evidence OBSERVED."""

    def build(*components: str, copies: int = 1, ligands: tuple[str, ...] = ()) -> Structure:
        entities: dict = {"P": PolymerSequence("polypeptide(L)", components)}
        instances = {chr(ord("A") + k): "P" for k in range(copies)}
        for k, ligand in enumerate(ligands):
            entities[f"L{k}"] = Molecule((ligand,))
            instances[chr(ord("A") + copies + k)] = f"L{k}"
        system, _, _ = build_system(Recipe(entities, instances), library)
        reference, _ = system.reference_coordinates()
        owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
        coordinates = reference + 4.0 * owner[:, None] * np.array([1.0, 0.0, 0.0])
        every = np.ones(len(owner), dtype=bool)
        structure, _ = replace_coordinates(
            Structure.unobserved(system),
            np.arange(len(owner)),
            coordinates,
            scope=every,
            evidence=CoordinateEvidence.OBSERVED,
        )
        return structure

    return build
