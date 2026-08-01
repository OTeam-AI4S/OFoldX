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
ODesign generation Pipelines: one backbone or ligand-coordinate generation, then the released inverse-folding child.

Each designed chain's identities come from its child family, which builds that chain's final target with the one
design producer; a candidate is the children applied in chain order, so no structure is merged here. The
provisional ligand route keeps ordered coordinates and element probabilities without inventing chemistry.

Complete repositories keep the backbone or ligand-coordinate generator under
``backbone/``. Protein routes also own ``protein_designer/`` and
``ligand_designer/``; DNA, RNA and alternative OInvFold polymer routes own
``sequence_designer/``; the provisional ligand route owns ``element_designer/``.
Each component remains independently loadable through ``load_component``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import ClassVar, Literal, NamedTuple, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ...chemistry.construction import PolymerSequence
from ...chemistry.definitions import ComponentLibrary
from ...data.molecular import (
    AtomFate,
    AtomSite,
    BondKind,
    ComponentRef,
    EntityType,
    MolecularSystem,
    MonomerType,
    endpoint_resolution,
)
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import CLASS_MONOMER, token_classes
from ...errors import RequestRef, StructureRef, fail
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedComponent, resolve_generator
from ...pipelines.base import task_call
from ...pipelines.input import resolve_msa_input, resolve_polymer_input
from ...provenance import Resolution, known
from ...results import DesignCandidate, Score, best, floating_numpy
from ..conditions import Hotspot
from ..design import DesignRecord, GuideAtoms, PositionSelection, resolve_positions
from ..grnade import GrnadePipeline
from ..grnade.model import GRNADE_CLASSES
from ..ligandmpnn import LigandMpnnPipeline
from ..oinvfold import OInvFoldLigandPipeline, OInvFoldPolymerPipeline
from ..preparation import PreparedInput, decode_coordinates
from ..proteinmpnn import ProteinMpnnPipeline
from .model import ODesignInput, ODesignLigandInput, ODesignModel, ODesignSamplingOutput
from .processor import ODesignLigandProcessor, ODesignProcessor, designable_components

_MONOMER_NAME = {MonomerType.PROTEIN: "protein", MonomerType.RNA: "rna", MonomerType.DNA: "dna"}
#: A text MSA names the one polymer entity of the conditioning complex, whatever its type.
_POLYMER_MONOMERS = (MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA)
#: What a generation pipeline conditions on: a canonical structure, or polymer sequences nothing is observed of.
PolymerInput = Structure | PolymerSequence | Sequence[PolymerSequence] | str | PathLike[str]


class _ChainDesign(Protocol):
    r"""What a child's design result gives this composition: ranked candidates and their design records."""

    @property
    def candidates(self) -> tuple[DesignCandidate, ...]: ...

    @property
    def records(self) -> tuple[DesignRecord, ...]: ...


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _samples(output: ODesignSamplingOutput) -> NDArray[np.float64]:
    r"""The coordinate samples of one system and one trunk evaluation, ``[samples, atoms, 3]``."""
    if output.coordinates.shape[:2] != (1, 1):
        fail("value_invalid", RequestRef("coordinates", str(tuple(output.coordinates.shape))))
    return output.coordinates[0, 0].detach().to(dtype=torch.float64, device="cpu").numpy()


_LINKER_NAMES = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "DA": "N9", "DG": "N9", "DC": "N1", "DT": "N1"}


def _linker_atom(structure: Structure, component: int) -> int:
    r"""The verified canonical N attached to this standard nucleotide's C1', or refuse an invalid mapping."""
    system = structure.system
    identity = str(system.text("component", system.component_symbol[component : component + 1])[0])
    name = _LINKER_NAMES.get(identity)
    if name is None:
        fail("value_invalid", StructureRef(component=component), reason="unmappable_linker_identity")
    ref = system.component_refs(np.array([component], dtype=np.int64))[0]
    atom, anchor = system.atom_index(
        (AtomSite(ref.instance, ref.component, name), AtomSite(ref.instance, ref.component, "C1'"))
    )
    edges = system.edge_index
    linked = ((edges[0] == atom) & (edges[1] == anchor)) | ((edges[1] == atom) & (edges[0] == anchor))
    covalent = known(system.bond_kind_state) & (system.bond_kind == BondKind.COVALENT)
    if (
        atom < 0
        or anchor < 0
        or system.atomic_number[atom] != 7
        or system.atomic_number[anchor] != 6
        or not np.any(linked & covalent)
    ):
        fail("value_invalid", StructureRef(component=component), reason="invalid_linker_topology")
    return int(atom)


def _generated_linkers(prepared: PreparedInput[ODesignInput], sample: Structure) -> dict[int, NDArray[np.float64]]:
    r"""Collect only designed nucleic linkers regenerated by this backbone sample."""
    slots = prepared.context.slots
    structure = prepared.context.structure
    system = structure.system
    scope = prepared.context.scope.atoms
    ids = system.text("component", system.component_symbol)
    design = prepared.features.design_features[0, :, 1:3].ne(0).any(dim=-1).detach().cpu().numpy()
    result = {}
    for token, component in enumerate(slots.layout.token_component.tolist()):
        if ids[component] not in _LINKER_NAMES or not design[token]:
            continue
        atom = _linker_atom(structure, component)
        if not scope[atom]:
            continue
        rows = np.flatnonzero((slots.token == token) & (slots.atom == atom))
        if len(rows) != 1 or sample.coordinate_evidence[atom] != CoordinateEvidence.GENERATED:
            fail("value_invalid", StructureRef(atom=atom), reason="invalid_generated_linker")
        if not np.isfinite(sample.coordinates[atom]).all():
            fail("value_invalid", StructureRef(atom=atom), reason="invalid_generated_linker")
        if component in result:
            fail("value_invalid", StructureRef(component=component), reason="duplicate_generated_linker")
        result[component] = sample.coordinates[atom].copy()
    return result


def _materialize_linkers(
    target: Structure, linkers: Mapping[int, NDArray[np.float64]], positions: NDArray[np.int64]
) -> Structure:
    r"""Place generated linker coordinates on the child's verified canonical replacement, never on a fixed atom."""
    atoms, coordinates = [], []
    for component in positions.tolist():
        if component not in linkers:
            continue
        atom = _linker_atom(target, component)
        evidence = target.coordinate_evidence[atom]
        if evidence == CoordinateEvidence.GENERATED:
            if not np.array_equal(target.coordinates[atom], linkers[component]):
                fail("value_invalid", StructureRef(atom=atom), reason="generated_linker_disagrees")
            continue
        if evidence != CoordinateEvidence.NONE:
            fail("value_invalid", StructureRef(atom=atom), reason="linker_is_observed")
        atoms.append(atom)
        coordinates.append(linkers[component])
    if not atoms:
        return target
    scope = np.zeros(len(target.system.atom_symbol), dtype=bool)
    scope[atoms] = True
    result, _ = replace_coordinates(
        target,
        np.asarray(atoms, dtype=np.int64),
        np.asarray(coordinates),
        scope=scope,
        evidence=CoordinateEvidence.GENERATED,
    )
    return result


def _fixed_connections(
    system: MolecularSystem, atom: int, component: int, backbone: frozenset[str]
) -> tuple[tuple[object, ...], ...]:
    r"""Known backbone and external bonds of one fixed atom, with component-stable endpoints."""
    s = system.arrays()
    first, second = s["edge_index"]
    edges = np.flatnonzero((first == atom) | (second == atom))
    owner = _owner(s["component_atom_ptr"])
    names = system.text("atom", s["atom_symbol"])
    signatures = []
    for edge in edges.tolist():
        neighbor = int(second[edge] if first[edge] == atom else first[edge])
        neighbor_component = int(owner[neighbor])
        name = str(names[neighbor])
        if neighbor_component == component:
            if name in backbone:
                endpoint: object = name
            elif names[atom] == "C1'" and name in ("N1", "N9"):
                endpoint = "glycosidic nitrogen"
            else:
                continue
        else:
            endpoint = (system.component_refs(np.array([neighbor_component], dtype=np.int64))[0], name)
        if not known(s["bond_kind_state"][edge]) or not known(s["bond_order_state"][edge]):
            fail("model_capacity", StructureRef(atom=atom), reason="fixed_bond_unresolved")
        if not known(s["atomic_number_state"][neighbor]):
            fail("model_capacity", StructureRef(atom=atom), reason="fixed_neighbor_unresolved")
        signatures.append(
            (endpoint, int(s["atomic_number"][neighbor]), int(s["bond_kind"][edge]), int(s["bond_order"][edge]))
        )
    return tuple(sorted(signatures, key=repr))


def _require_fixed_atoms(source: Structure, target: Structure, sites: Sequence[AtomSite]) -> None:
    r"""Refuse a designed candidate unless every requested fixed atom and its evidence survive safely."""
    if not sites:
        return
    source_system, target_system = source.system, target.system
    source_atoms, target_atoms = source_system.atom_index(sites), target_system.atom_index(sites)
    source_owner = _owner(source_system.component_atom_ptr)
    source_ids = source_system.text("component", source_system.component_symbol)
    target_ids = target_system.text("component", target_system.component_symbol)
    monomer = CLASS_MONOMER[token_classes(source_system).token]
    source_values, target_values = source.arrays(), target.arrays()
    for site, old, new in zip(sites, source_atoms.tolist(), target_atoms.tolist(), strict=True):
        if old < 0 or new < 0:
            fail("model_capacity", RequestRef("fixed_atoms", str(site)), reason="fixed_atom_missing")
        component = int(source_owner[old])
        ref = source_system.component_refs(np.array([component], dtype=np.int64))[0]
        if int(target_system.component_index([ref])[0]) != component:
            fail("model_capacity", StructureRef(atom=old), reason="fixed_component_changed")
        backbone = GuideAtoms.BACKBONE.names.get(MonomerType(int(monomer[component])), frozenset())
        if source_ids[component] != target_ids[component] and site.atom not in backbone:
            fail("model_capacity", StructureRef(atom=old), reason="fixed_atom_identity_change")
        for system, atom in ((source_system, old), (target_system, new)):
            endpoint = endpoint_resolution(system.atom_endpoint_state[np.array([atom])])[0]
            if (
                system.atom_fate[atom] != AtomFate.RETAINED
                or endpoint in (Resolution.UNKNOWN, Resolution.CONFLICT)
                or not known(system.atomic_number_state[atom])
                or not known(system.formal_charge_state[atom])
            ):
                fail("model_capacity", StructureRef(atom=old), reason="fixed_atom_unresolved")
        if (
            source_system.atomic_number[old] != target_system.atomic_number[new]
            or source_system.isotope[old] != target_system.isotope[new]
            or source_system.formal_charge[old] != target_system.formal_charge[new]
        ):
            fail("model_capacity", StructureRef(atom=old), reason="fixed_atom_chemistry_changed")
        if _fixed_connections(source_system, old, component, backbone) != _fixed_connections(
            target_system, new, component, backbone
        ):
            fail("model_capacity", StructureRef(atom=old), reason="fixed_atom_bonds_changed")
        for field in Structure.SCHEMA.fields:
            if field.axis == "atom" and not np.array_equal(
                source_values[field.name][old], target_values[field.name][new]
            ):
                fail("model_capacity", StructureRef(atom=old), reason="fixed_atom_evidence_changed", field=field.name)


class ODesignUnboundLinker(NamedTuple):
    r"""A generated model-only glycosidic linker, not a canonical chemical atom."""

    coordinates: NDArray[np.float64]
    evidence: CoordinateEvidence


class ODesignBackboneSample(NamedTuple):
    r"""A generated structure and model-only linkers keyed by their component."""

    structure: Structure
    unbound_linkers: Mapping[ComponentRef, ODesignUnboundLinker]


class ODesignBackboneComponent(PairedComponent[ODesignModel, ODesignProcessor]):
    r"""Loadable ODesign polymer-backbone component.

    ``prepare`` and ``postprocess`` surround the model's ``sample`` operation.
    ``num_trunk_evaluations`` counts independent trunk evaluations and
    ``num_samples`` counts coordinate samples within each evaluation.
    """

    model_architecture = "odesign"

    def prepare(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[ODesignInput]:
        r"""
        Prepare polymer-backbone generation without executing the Model.

        Notes:
            Generated positions retain their canonical target chemistry but expose the monomer's unknown class, no
            formal
            charge and the design-state feature to the model. Generated protein positions are represented by N/CA/C/O.
            Child
            inverse-folding results record whether source identities were retained, retained under their encoding or
            replaced.

            Row validity, reference availability, source-coordinate availability and fixed-coordinate selection use
            separate
            masks. Available numerical input must be finite and representable in the model dtype; unavailable payload is
            masked before geometry and dtype conversion. Regeneration scope is fixed during preparation.
        """
        return self.processor.prepare_input(
            structure,
            design=design,
            msa=msa,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
        )

    def postprocess(
        self, prepared: PreparedInput[ODesignInput], output: ODesignSamplingOutput
    ) -> tuple[ODesignBackboneSample, ...]:
        r"""
        Decode each sample onto the prepared target and retain its model-only linker coordinates.

        Bound atoms receive GENERATED coordinates in the Structure; unbound linkers are keyed by component and carry
        GENERATED evidence without creating a canonical chemical atom.

        A fixed slot (and, for a rigid-receptor artifact, every context slot) is outside the scope: the model states
        its coordinates as its input states them, so the decode keeps the target's own.

        Notes:
            Regeneration covers every atom of a generated position and every moved context component in a
            flexible-receptor
            artifact. Excluded atoms, including hydrogens, lose old coordinates with their moved component. Fixed atoms
            and
            all rigid-receptor context remain outside that scope and retain their coordinates and observations.
            Generated
            protein positions receive backbone coordinates; other coordinates are unavailable.
        """

        slots = prepared.context.slots
        atoms = slots.atom
        valid = (atoms >= 0) & prepared.context.scope.atoms[np.maximum(atoms, 0)]
        rows = np.flatnonzero((atoms < 0) & (slots.token < len(slots.layout.token_component)))
        components = slots.layout.token_component[slots.token[rows]]
        refs = prepared.context.structure.system.component_refs(components)
        if len(set(refs)) != len(refs):
            fail("value_invalid", RequestRef("backbone", "duplicate unbound linkers"))
        results = []
        for sample in _samples(output):
            if not np.isfinite(sample[rows]).all():
                fail("value_invalid", RequestRef("backbone", "nonfinite unbound linker"))
            structure = decode_coordinates(prepared, sample, valid=valid, evidence=CoordinateEvidence.GENERATED)[0]
            linkers = {
                ref: ODesignUnboundLinker(sample[row].copy(), CoordinateEvidence.GENERATED)
                for ref, row in zip(refs, rows.tolist(), strict=True)
            }
            results.append(ODesignBackboneSample(structure, linkers))
        return tuple(results)


class ODesignLigandCoordinateSample(NamedTuple):
    r"""One decoded provisional-ligand sample: its ordered slot coordinates and the conditioning context."""

    coordinates: NDArray[np.float64]
    slot_ids: tuple[str, ...]
    context: Structure


class ODesignLigandCoordinateComponent(PairedComponent[ODesignModel, ODesignLigandProcessor]):
    r"""Loadable ODesign provisional-ligand component.

    An integer ``generated_atoms`` fixes the provisional atom count. A ``range``
    samples that count using the preparation generator. Context is supplied as
    a ``Structure`` and conditioning uses its observed coordinates.
    """

    model_architecture = "odesign_ligand"

    def prepare(
        self,
        structure: Structure,
        *,
        generated_atoms: int | range,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ODesignLigandInput]:
        r"""
        Prepare provisional-ligand generation without executing the Model.

        Notes:
            Provisional ligand rows have no canonical atom identity and create no canonical chemical vertices. Model row
            validity, reference availability, source observations and fixed coordinates retain separate masks.

            Element category 128 represents unavailable information for both canonically unknown and deliberately hidden
            elements. It does not denote hydrogen or create an element identity. Canonical Z=0 remains unknown; the
            separate
            hydrogen descriptor uses category 0.
        """
        return self.processor.prepare_input(
            structure,
            generated_atoms=generated_atoms,
            msa=msa,
            hotspots=hotspots,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            generator=generator,
        )

    def postprocess(
        self, prepared: PreparedInput[ODesignLigandInput], output: ODesignSamplingOutput
    ) -> tuple[tuple[ODesignLigandCoordinateSample, ...], NDArray]:
        r"""The generated slots of each sample and the pair bond logits of their tokens; the context is unchanged."""
        rows = np.flatnonzero(prepared.context.slots.atom < 0)
        tokens = torch.from_numpy(prepared.context.slots.token[rows]).to(output.token_bond_logits.device)
        logits = output.token_bond_logits[0, 0].index_select(0, tokens).index_select(1, tokens)
        slot_ids = tuple(str(index) for index in range(len(rows)))
        decoded = tuple(
            ODesignLigandCoordinateSample(sample[rows], slot_ids, prepared.context.structure)
            for sample in _samples(output)
        )
        return decoded, floating_numpy(logits)


@dataclass(eq=False)
class ODesignPolymerCandidate:
    r"""
    One assigned ODesign polymer design.

    ``scores`` keys are the child score names of each designed chain, qualified by its instance symbol
    (``"A:designed_sequence_nll"``); ``records`` are the children's design records, in chain order.
    """

    structure: Structure
    scores: dict[str, Score]
    records: tuple[DesignRecord, ...]


@dataclass(eq=False)
class ODesignPolymerResult:
    r"""Polymer candidates in seed-major, backbone-major, sequence-rank order."""

    candidates: tuple[ODesignPolymerCandidate, ...]
    num_seeds: int
    num_backbones: int
    num_sequences: int

    def best_by(self, score: str) -> ODesignPolymerCandidate:
        r"""The earliest best candidate under one explicit score."""
        return best(self.candidates, score)


@dataclass(eq=False)
class ODesignLigandCandidate:
    r"""
    One provisional ligand assignment with its conditioning context.

    The slots have no canonical atom identity and create no chemical vertex: the candidate states their coordinates,
    the predicted elements with their per-atom probabilities, and the pair bond logits of their tokens.
    """

    context: Structure
    slot_ids: tuple[str, ...]
    coordinates: NDArray
    elements: tuple[str, ...]
    per_atom_probabilities: NDArray
    token_bond_logits: NDArray
    scores: dict[str, Score]
    bond_state: Literal["unresolved"] = "unresolved"


@dataclass(eq=False)
class ODesignLigandResult:
    r"""Ligand candidates in seed-major, coordinate-sample-major, assignment-rank order."""

    candidates: tuple[ODesignLigandCandidate, ...]
    num_seeds: int
    num_coordinate_samples: int
    num_assignments: int

    def best_by(self, score: str) -> ODesignLigandCandidate:
        r"""The earliest best candidate under one explicit score."""
        return best(self.candidates, score)


class ODesignPolymerPipelineInput(NamedTuple):
    r"""The conditioning structure, the generated positions and the resolved sample axes of one polymer design."""

    structure: Structure
    design: PositionSelection
    msa: Mapping[str, MSA] | MsaCollection | None
    seeds: tuple[int, ...]
    num_backbones: int
    num_sequences: int
    hotspots: Sequence[Hotspot]
    fixed_atoms: Sequence[AtomSite]
    constraint_atoms: Sequence[AtomSite]
    filter_by_occupancy: bool
    use_beam: bool


class ODesignLigandPipelineInput(NamedTuple):
    r"""The conditioning context and the resolved sample axes of one provisional ligand design."""

    structure: Structure
    generated_atoms: int | range
    msa: Mapping[str, MSA] | MsaCollection | None
    seeds: tuple[int, ...]
    num_coordinate_samples: int
    num_assignments: int
    use_beam: bool
    hotspots: Sequence[Hotspot]
    constraint_atoms: Sequence[AtomSite]
    filter_by_occupancy: bool


def designed_chains(
    structure: Structure, design: PositionSelection, monomer: MonomerType
) -> tuple[tuple[str, NDArray[np.int64]], ...]:
    r"""
    The generated positions grouped by instance, in system order, with their instance symbol.

    Raises:
        UnsupportedError: ``model_capacity`` for a generated position of another monomer type than the route's.
    """

    system = structure.system
    positions = resolve_positions(structure, design, allowed=designable_components(system))
    kinds = CLASS_MONOMER[token_classes(system).token][positions]
    outside = positions[kinds != monomer]
    if len(outside):
        fail("model_capacity", StructureRef(component=int(outside[0])), route=_MONOMER_NAME[monomer])
    instance = _owner(system.instance_component_ptr)[positions]
    symbols = system.text("instance", system.instance_symbol)
    return tuple((str(symbols[value]), positions[instance == value]) for value in dict.fromkeys(instance.tolist()))


def has_ligand_atom_context(structure: Structure) -> bool:
    r"""Whether the structure holds an observed heavy atom of a non-protein, non-water component (LigandMPNN's
    input)."""
    system = structure.system
    s = system.arrays()
    component = _owner(s["component_atom_ptr"])
    entity = s["instance_entity"][_owner(s["instance_component_ptr"])]
    water = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.WATER)
    monomer = CLASS_MONOMER[token_classes(system).token]
    ligand = (monomer != MonomerType.PROTEIN) & ~water
    return bool((structure.coordinate_mask & (s["atomic_number"] > 1) & ligand[component]).any())


class _ODesignPolymerPipeline(CompositePipeline):
    r"""ODesign backbone generation and the released inverse folding of each generated chain."""

    task = "design"

    monomer: ClassVar[MonomerType]
    artifacts: ClassVar[tuple[str, ...]]

    @property
    def backbone(self) -> ODesignBackboneComponent:
        return cast(ODesignBackboneComponent, self.components["backbone"])

    def prepare(
        self,
        input: PolymerInput,
        *,
        design: PositionSelection = PositionSelection(),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        library: ComponentLibrary | None = None,
    ) -> ODesignPolymerPipelineInput:
        r"""
        Resolve the conditioning structure and MSA, the sample axes and the seeds before execution.
        """
        structure = resolve_polymer_input(input, library=library)
        return ODesignPolymerPipelineInput(
            structure=structure,
            design=design,
            msa=resolve_msa_input(msa, structure.system, monomer_types=_POLYMER_MONOMERS),
            seeds=_resolve_seeds(num_seeds, seeds, seed, generator),
            num_backbones=num_backbones,
            num_sequences=num_sequences,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            use_beam=use_beam,
        )

    def forward(
        self,
        input: ODesignPolymerPipelineInput,
        *,
        temperature: float = 1.0,
        num_recycles: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        library: ComponentLibrary | None = None,
    ) -> tuple[ODesignPolymerCandidate, ...]:
        r"""Generate each seeded backbone and assign the identities of its generated chains.

        Results remain in seed-major, backbone-major, sequence-rank order.
        OInvFold children interpret ``num_sequences`` as a per-backbone beam
        budget; greedy decoding returns one assignment and has no sampling
        temperature. ProteinMPNN, LigandMPNN and gRNAde use their own sampling
        temperatures. Every candidate retains the children's final structure
        and design records in chain order.
        """
        chains = designed_chains(input.structure, input.design, self.monomer)
        candidates: list[ODesignPolymerCandidate] = []
        for seed in input.seeds:
            stream = resolve_generator(seed, None, device=self.backbone.processor.device)
            prepared = self.backbone.prepare(
                input.structure,
                design=input.design,
                msa=input.msa,
                hotspots=input.hotspots,
                fixed_atoms=input.fixed_atoms,
                constraint_atoms=input.constraint_atoms,
                filter_by_occupancy=input.filter_by_occupancy,
            )
            output = self.backbone.model.sample(
                prepared.features,
                num_samples=input.num_backbones,
                generator=stream,
                num_recycles=num_recycles,
                num_sampling_steps=num_sampling_steps,
                num_trunk_evaluations=1,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            for sample in self.backbone.postprocess(prepared, output):
                linkers = _generated_linkers(prepared, sample.structure)
                refs = tuple(sample.unbound_linkers)
                components = input.structure.system.component_index(refs)
                for component, ref in zip(components.tolist(), refs, strict=True):
                    if component < 0 or component in linkers:
                        fail("value_invalid", RequestRef("backbone", "unbound linker component"))
                    linkers[component] = sample.unbound_linkers[ref].coordinates
                candidates.extend(
                    self._assign(
                        sample.structure,
                        chains,
                        fixed_source=input.structure,
                        fixed_atoms=input.fixed_atoms,
                        linker_coordinates=linkers,
                        num_candidates=input.num_sequences,
                        temperature=temperature,
                        use_beam=input.use_beam,
                        generator=stream,
                        library=library,
                    )
                )
        return tuple(candidates)

    def postprocess(
        self, input: ODesignPolymerPipelineInput, output: tuple[ODesignPolymerCandidate, ...]
    ) -> ODesignPolymerResult:
        r"""
        Assemble the ordered candidates with their resolved sample counts.
        """
        return ODesignPolymerResult(
            candidates=output,
            num_seeds=len(input.seeds),
            num_backbones=input.num_backbones,
            num_sequences=input.num_sequences,
        )

    def design(
        self,
        input: PolymerInput,
        *,
        design: PositionSelection = PositionSelection(),
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_backbones: int = 1,
        num_sequences: int = 1,
        use_beam: bool = True,
        temperature: float = 1.0,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        num_recycles: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
        library: ComponentLibrary | None = None,
    ) -> ODesignPolymerResult:
        r"""
        Generate backbones for the requested positions and assign their identities.

        Args:
            input: The conditioning complex, including the positions to generate, or the polymer sequences it is
                built from (a de-novo chain is one of unknown residues).
            design: The positions ODesign generates (default every position of the route's monomer type).
            msa: MSA pools by instance, an exact plan, an alignment path, or ``None``.
            num_seeds: Number of independent generation seeds.
            num_backbones: Number of backbone samples per seed.
            num_sequences: Number of identity assignments per backbone.
            use_beam: Whether a beam-decoding child returns ranked assignments.
            temperature: Sampling temperature of a sampling child.
            seeds: Explicit seeds; mutually exclusive with ``seed`` and ``generator``.
            seed: Optional integer seed of the seed draw.
            generator: Optional PyTorch generator of the seed draw.
            hotspots: Residue and atom positions labelled as interaction hotspots.
            fixed_atoms: Atoms whose observed coordinates are kept.
            constraint_atoms: Atoms whose observed pair distances steer the generation.
            num_recycles: Number of trunk recycling passes.
            num_sampling_steps: Number of coordinate sampling steps.
            partial_diffusion_snr: Optional partial-diffusion signal-to-noise ratio.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            library: The definitions a sequence input and the children's replacements are built from.
        """

        prepared = self.prepare(
            input,
            design=design,
            msa=msa,
            num_seeds=num_seeds,
            num_backbones=num_backbones,
            num_sequences=num_sequences,
            use_beam=use_beam,
            seeds=seeds,
            seed=seed,
            generator=generator,
            hotspots=hotspots,
            fixed_atoms=fixed_atoms,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
            library=library,
        )
        output = self.forward(
            prepared,
            temperature=temperature,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            partial_diffusion_snr=partial_diffusion_snr,
            library=library,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)

    def _assign(
        self,
        backbone: Structure,
        chains: tuple[tuple[str, NDArray[np.int64]], ...],
        *,
        fixed_source: Structure,
        fixed_atoms: Sequence[AtomSite],
        linker_coordinates: Mapping[int, NDArray[np.float64]],
        num_candidates: int,
        temperature: float,
        use_beam: bool,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> tuple[ODesignPolymerCandidate, ...]:
        r"""
        One backbone's candidates: each chain's child designs it on the previous chain's target and the candidate of
        the rank being assembled is taken (component indices are stable under a replacement, so one rank addresses
        every intermediate target). The first chain conditions on the backbone alone, so it is designed once and its
        ranked candidates are the stems the later chains continue.
        """

        def design(target: Structure, positions: NDArray[np.int64]) -> _ChainDesign:
            return self._design_chain(
                target,
                PositionSelection(indices=positions),
                linker_coordinates=linker_coordinates,
                num_candidates=num_candidates,
                temperature=temperature,
                use_beam=use_beam,
                generator=generator,
                library=library,
            )

        _require_fixed_atoms(fixed_source, backbone, fixed_atoms)
        if not chains:
            return tuple(ODesignPolymerCandidate(backbone, {}, ()) for _ in range(num_candidates))
        (head, head_positions), rest = chains[0], chains[1:]
        stems = design(backbone, head_positions)
        candidates = []
        for rank in range(num_candidates):
            stem = stems.candidates[rank]
            _require_fixed_atoms(fixed_source, stem.target, fixed_atoms)
            target = _materialize_linkers(stem.target, linker_coordinates, head_positions)
            _require_fixed_atoms(fixed_source, target, fixed_atoms)
            records = [stems.records[rank]]
            scores = {f"{head}:{name}": score for name, score in stem.scores.items()}
            for symbol, positions in rest:
                result = design(target, positions)
                candidate = result.candidates[rank]
                _require_fixed_atoms(fixed_source, candidate.target, fixed_atoms)
                target = _materialize_linkers(candidate.target, linker_coordinates, positions)
                _require_fixed_atoms(fixed_source, target, fixed_atoms)
                scores.update({f"{symbol}:{name}": score for name, score in candidate.scores.items()})
                records.append(result.records[rank])
            candidates.append(ODesignPolymerCandidate(target, scores, tuple(records)))
        return tuple(candidates)

    def _design_chain(
        self,
        target: Structure,
        design: PositionSelection,
        *,
        linker_coordinates: Mapping[int, NDArray[np.float64]],
        num_candidates: int,
        temperature: float,
        use_beam: bool,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> _ChainDesign:
        r"""One chain's identities from this route's child, as its ranked candidates with their design records."""
        raise NotImplementedError

    def _require_artifact(self) -> None:
        artifact = self.backbone.processor.config.artifact
        if artifact not in self.artifacts:
            fail("value_invalid", RequestRef("artifact", artifact), pipeline=type(self).__name__)


class ODesignProteinPipeline(_ODesignPolymerPipeline):
    r"""Official ODesign protein generation with context-selected MPNN inverse folding."""

    monomer = MonomerType.PROTEIN
    artifacts = ("protein_flex", "protein_rigid")

    component_layout = (
        ComponentSpec("backbone", "backbone", ODesignBackboneComponent),
        ComponentSpec("protein_inverse_folder", "protein_designer", ProteinMpnnPipeline),
        ComponentSpec("ligand_inverse_folder", "ligand_designer", LigandMpnnPipeline),
    )

    def __init__(
        self,
        *,
        backbone: ODesignBackboneComponent,
        protein_inverse_folder: ProteinMpnnPipeline,
        ligand_inverse_folder: LigandMpnnPipeline,
    ) -> None:
        super().__init__(
            {
                "backbone": backbone,
                "protein_inverse_folder": protein_inverse_folder,
                "ligand_inverse_folder": ligand_inverse_folder,
            }
        )
        self._require_artifact()

    @property
    def protein_inverse_folder(self) -> ProteinMpnnPipeline:
        return cast(ProteinMpnnPipeline, self.components["protein_inverse_folder"])

    @property
    def ligand_inverse_folder(self) -> LigandMpnnPipeline:
        return cast(LigandMpnnPipeline, self.components["ligand_inverse_folder"])

    def _design_chain(
        self,
        target: Structure,
        design: PositionSelection,
        *,
        linker_coordinates: Mapping[int, NDArray[np.float64]],
        num_candidates: int,
        temperature: float,
        use_beam: bool,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> _ChainDesign:
        r"""LigandMPNN when the complex holds ligand atom context, ProteinMPNN otherwise (the released choice)."""
        child = self.ligand_inverse_folder if has_ligand_atom_context(target) else self.protein_inverse_folder
        return child.design(
            target,
            design=design,
            num_candidates=num_candidates,
            temperature=temperature,
            generator=generator,
            library=library,
        )


class _ODesignOInvFoldPipeline(_ODesignPolymerPipeline):
    r"""ODesign generation with an OInvFold inverse-folding child of the route's modality."""

    component_layout = (
        ComponentSpec("backbone", "backbone", ODesignBackboneComponent),
        ComponentSpec("inverse_folder", "sequence_designer", OInvFoldPolymerPipeline),
    )

    def __init__(self, *, backbone: ODesignBackboneComponent, inverse_folder: OInvFoldPolymerPipeline) -> None:
        super().__init__({"backbone": backbone, "inverse_folder": inverse_folder})
        self._require_artifact()
        modality = inverse_folder.processor.config.modality
        if modality != _MONOMER_NAME[self.monomer]:
            fail("value_invalid", RequestRef("modality", str(modality)), pipeline=type(self).__name__)

    @property
    def inverse_folder(self) -> OInvFoldPolymerPipeline:
        return cast(OInvFoldPolymerPipeline, self.components["inverse_folder"])

    def _design_chain(
        self,
        target: Structure,
        design: PositionSelection,
        *,
        linker_coordinates: Mapping[int, NDArray[np.float64]],
        num_candidates: int,
        temperature: float,
        use_beam: bool,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> _ChainDesign:
        r"""
        The beam candidates of one chain (greedy decoding returns exactly one, whatever the width).

        Raises:
            InvalidRequestError: ``value_invalid`` for several sequences without beam decoding.
        """

        if not use_beam and num_candidates != 1:
            fail("value_invalid", RequestRef("num_sequences", str(num_candidates)), use_beam=False)
        prepared = self.inverse_folder.processor.prepare_input(
            target, design=design, _linker_coordinates=linker_coordinates
        )
        return self.inverse_folder.postprocess(
            prepared,
            self.inverse_folder.forward(prepared.features),
            num_beams=num_candidates,
            use_beam=use_beam,
            library=library,
        )


class ODesignDnaPipeline(_ODesignOInvFoldPipeline):
    r"""Official ODesign DNA generation with the released DNA OInvFold child."""

    monomer = MonomerType.DNA
    artifacts = ("nucleic_acid_rigid",)


class ODesignOInvFoldProteinPipeline(_ODesignOInvFoldPipeline):
    r"""ODesign protein generation with the independently runnable OInvFold protein child."""

    monomer = MonomerType.PROTEIN
    artifacts = ("protein_flex", "protein_rigid")


class ODesignOInvFoldRnaPipeline(_ODesignOInvFoldPipeline):
    r"""ODesign RNA generation with the independently runnable OInvFold RNA child."""

    monomer = MonomerType.RNA
    artifacts = ("nucleic_acid_rigid",)


class ODesignRnaPipeline(_ODesignPolymerPipeline):
    r"""Official ODesign RNA generation with the released gRNAde v1 child."""

    monomer = MonomerType.RNA
    artifacts = ("nucleic_acid_rigid",)

    component_layout = (
        ComponentSpec("backbone", "backbone", ODesignBackboneComponent),
        ComponentSpec("inverse_folder", "sequence_designer", GrnadePipeline),
    )

    def __init__(self, *, backbone: ODesignBackboneComponent, inverse_folder: GrnadePipeline) -> None:
        super().__init__({"backbone": backbone, "inverse_folder": inverse_folder})
        self._require_artifact()

    @property
    def inverse_folder(self) -> GrnadePipeline:
        return cast(GrnadePipeline, self.components["inverse_folder"])

    def _design_chain(
        self,
        target: Structure,
        design: PositionSelection,
        *,
        linker_coordinates: Mapping[int, NDArray[np.float64]],
        num_candidates: int,
        temperature: float,
        use_beam: bool,
        generator: torch.Generator | None,
        library: ComponentLibrary | None,
    ) -> _ChainDesign:
        positions = resolve_positions(target, design, allowed=designable_components(target.system))
        instances = {ref.instance for ref in target.system.component_refs(positions)}
        if len(instances) != 1:
            fail(
                "request_invalid_positions", RequestRef("design", "one RNA instance required"), instances=len(instances)
            )
        instance_id = next(iter(instances))
        system = target.system
        instance = int(system.instance_index([instance_id])[0])
        start, stop = system.instance_component_ptr[instance : instance + 2]
        components = np.arange(start, stop, dtype=np.int64)
        classes = token_classes(system).token
        rna = CLASS_MONOMER[classes[components]] == MonomerType.RNA
        context = ~np.isin(components, positions)
        unsupported = components[rna & context & ~np.isin(classes[components], GRNADE_CLASSES)]
        if len(unsupported):
            fail("model_capacity", StructureRef(component=int(unsupported[0])), reason="unsupported_rna_context")
        prepared = self.inverse_folder.processor.prepare_input(
            target, instance_id=instance_id, design=design, _linker_coordinates=linker_coordinates
        )
        output = self.inverse_folder.forward(
            prepared.features, num_candidates=num_candidates, temperature=temperature, generator=generator
        )
        return self.inverse_folder.postprocess(prepared, output, library=library)


class ODesignLigandPipeline(CompositePipeline):
    r"""Official ODesign provisional ligand generation with the ligand OInvFold element designer."""

    task = "design"

    component_layout = (
        ComponentSpec("coordinate_generator", "backbone", ODesignLigandCoordinateComponent),
        ComponentSpec("inverse_folder", "element_designer", OInvFoldLigandPipeline),
    )

    def __init__(
        self,
        *,
        coordinate_generator: ODesignLigandCoordinateComponent,
        inverse_folder: OInvFoldLigandPipeline,
    ) -> None:
        super().__init__({"coordinate_generator": coordinate_generator, "inverse_folder": inverse_folder})

    @property
    def coordinate_generator(self) -> ODesignLigandCoordinateComponent:
        return cast(ODesignLigandCoordinateComponent, self.components["coordinate_generator"])

    @property
    def inverse_folder(self) -> OInvFoldLigandPipeline:
        return cast(OInvFoldLigandPipeline, self.components["inverse_folder"])

    def prepare(
        self,
        structure: Structure,
        *,
        generated_atoms: int | range,
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_coordinate_samples: int = 1,
        num_assignments: int = 1,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
    ) -> ODesignLigandPipelineInput:
        r"""
        Resolve the conditioning MSA, the sample axes and the seeds before execution.
        """
        if not use_beam and num_assignments != 1:
            fail("value_invalid", RequestRef("num_assignments", str(num_assignments)), use_beam=False)
        return ODesignLigandPipelineInput(
            structure=structure,
            generated_atoms=generated_atoms,
            msa=resolve_msa_input(msa, structure.system, monomer_types=_POLYMER_MONOMERS),
            seeds=_resolve_seeds(num_seeds, seeds, seed, generator),
            num_coordinate_samples=num_coordinate_samples,
            num_assignments=num_assignments,
            use_beam=use_beam,
            hotspots=hotspots,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
        )

    def forward(
        self,
        input: ODesignLigandPipelineInput,
        *,
        probability_temperature: float = 1.0,
        num_recycles: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
    ) -> tuple[ODesignLigandCandidate, ...]:
        r"""Generate each seeded coordinate sample and assign the elements of its slots.

        Preparation fixes ``num_coordinate_samples``, ``num_assignments`` and
        ``use_beam``. Results remain in seed-major, coordinate-sample-major,
        assignment-rank order. ``probability_temperature`` controls the returned
        element-probability view; diffusion-step controls are execution options.
        """
        candidates: list[ODesignLigandCandidate] = []
        for seed in input.seeds:
            stream = resolve_generator(seed, None, device=self.coordinate_generator.processor.device)
            prepared = self.coordinate_generator.prepare(
                input.structure,
                generated_atoms=input.generated_atoms,
                msa=input.msa,
                hotspots=input.hotspots,
                constraint_atoms=input.constraint_atoms,
                filter_by_occupancy=input.filter_by_occupancy,
                generator=np.random.default_rng(seed),
            )
            output = self.coordinate_generator.model.sample(
                prepared.features,
                num_samples=input.num_coordinate_samples,
                generator=stream,
                num_recycles=num_recycles,
                num_sampling_steps=num_sampling_steps,
                num_trunk_evaluations=1,
                partial_diffusion_snr=partial_diffusion_snr,
            )
            decoded, token_bond_logits = self.coordinate_generator.postprocess(prepared, output)
            for sample in decoded:
                assigned = self.inverse_folder.design(
                    sample.coordinates,
                    slot_ids=sample.slot_ids,
                    context=sample.context,
                    num_beams=input.num_assignments,
                    use_beam=input.use_beam,
                    probability_temperature=probability_temperature,
                )
                candidates.extend(
                    ODesignLigandCandidate(
                        context=sample.context,
                        slot_ids=candidate.slot_ids,
                        coordinates=candidate.coordinates,
                        elements=candidate.elements,
                        per_atom_probabilities=candidate.per_atom_probabilities,
                        token_bond_logits=token_bond_logits,
                        scores=dict(candidate.scores),
                    )
                    for candidate in assigned.candidates
                )
        return tuple(candidates)

    def postprocess(
        self, input: ODesignLigandPipelineInput, output: tuple[ODesignLigandCandidate, ...]
    ) -> ODesignLigandResult:
        r"""Assemble the ordered candidates with their resolved sample counts."""
        return ODesignLigandResult(
            candidates=output,
            num_seeds=len(input.seeds),
            num_coordinate_samples=input.num_coordinate_samples,
            num_assignments=input.num_assignments,
        )

    def design(
        self,
        structure: Structure,
        *,
        generated_atoms: int | range,
        msa: Mapping[str, MSA] | MsaCollection | str | PathLike[str] | None = None,
        num_seeds: int = 1,
        num_coordinate_samples: int = 1,
        num_assignments: int = 1,
        probability_temperature: float = 1.0,
        use_beam: bool = True,
        seeds: Sequence[int] | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        hotspots: Sequence[Hotspot] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        num_recycles: int | None = None,
        num_sampling_steps: int = 200,
        partial_diffusion_snr: float | None = None,
        filter_by_occupancy: bool = True,
    ) -> ODesignLigandResult:
        r"""Generate provisional ligand coordinates in the context and assign their element identities."""
        prepared = self.prepare(
            structure,
            generated_atoms=generated_atoms,
            msa=msa,
            num_seeds=num_seeds,
            num_coordinate_samples=num_coordinate_samples,
            num_assignments=num_assignments,
            use_beam=use_beam,
            seeds=seeds,
            seed=seed,
            generator=generator,
            hotspots=hotspots,
            constraint_atoms=constraint_atoms,
            filter_by_occupancy=filter_by_occupancy,
        )
        output = self.forward(
            prepared,
            probability_temperature=probability_temperature,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            partial_diffusion_snr=partial_diffusion_snr,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(design)


def _resolve_seeds(
    num_seeds: int, seeds: Sequence[int] | None, seed: int | None, generator: torch.Generator | None
) -> tuple[int, ...]:
    r"""The explicit seeds, or ``num_seeds`` drawn from the resolved generator."""
    if seeds is not None:
        if seed is not None or generator is not None:
            fail("value_invalid", RequestRef("seeds", "with seed or generator"))
        return tuple(int(value) for value in seeds)
    generator = resolve_generator(seed, generator)
    device = None if generator is None else generator.device
    drawn = torch.randint(1, torch.iinfo(torch.int32).max, (num_seeds,), generator=generator, device=device)
    return tuple(int(value) for value in drawn.cpu().tolist())
