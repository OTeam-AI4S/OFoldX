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
ODesign input processing, written against the shared preparation contract (generation shares the source frame).

Everything that produces the model's inputs is OFoldX's own (layout, gating, reference conformers, crop, MSA
pairing); this module keeps only the ODesign featurisation of those inputs, which matches upstream for identical
inputs. The generated positions are named by a ``PositionSelection``: their chemistry is hidden from the
features (the unknown class of their monomer type, no charge, the design-state plane), the designed protein backbone
is addressed as N, CA, C and O only, and their coordinates are the task's regeneration scope. The provisional ligand
preparation adds ordered slots bound to no target atom, so nothing of them is decoded onto the target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors import SafetensorError
from safetensors.numpy import load as load_safetensors
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import (
    AtomFate,
    AtomSite,
    BondKind,
    ComponentRef,
    EntityType,
    MolecularSystem,
    MonomerType,
    ReferenceSource,
    endpoint_resolution,
)
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, Encoding, TokenLayout, token_classes
from ...errors import RequestRef, SourceRef, StructureRef, fail
from ...msa import MSA, MsaCollection, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Records, Resolution, known
from ..conditions import Hotspot
from ..design import PositionSelection, resolve_positions
from ..layout import LayoutTable, model_tokens, slot_map, token_layout, token_representatives
from ..preparation import (
    ConformerSource,
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
    missing_reference,
    reference_conformers,
)
from ..processing import project_model_coordinates, project_model_features
from ..targets import StructureTarget, SymmetryPolicy, TruthState, match_atoms, prepare_target
from .config import ODesignLigandProcessorConfig, ODesignProcessorConfig
from .model import ODesignInput, ODesignLigandInput, ODesignStructureTarget

_PROTEIN, _RNA, _DNA, _NONE = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA, MonomerType.NONE
_NUCLEIC_CENTRE = "C1'"
#: The hidden class of a generated position, by monomer type.
_UNKNOWN_CLASS = {_PROTEIN: "X", _RNA: "RNA_X", _DNA: "DNA_X", _NONE: "OTHER"}
#: The design-state plane a generated position sets (protein, RNA, DNA, ligand).
_DESIGN_STATE = {_PROTEIN: 0, _RNA: 1, _DNA: 2, _NONE: 3}
#: The atoms of a generated protein position the model places.
_DESIGNED_BACKBONE = ("N", "CA", "C", "O")
# The released nucleic-acid mask retains these backbone atoms and the base's glycosidic nitrogen.
_DESIGNED_NUCLEIC_BACKBONE = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
_NUCLEIC_GLYCOSIDIC_N = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "DA": "N9", "DG": "N9", "DC": "N1", "DT": "N1"}
# Model-only representative conformers from the released ODesign GLY/C/DC reference archive.
_REFERENCE_NAMES = {
    _PROTEIN: _DESIGNED_BACKBONE,
    _RNA: (*_DESIGNED_NUCLEIC_BACKBONE, "N1"),
    _DNA: (*(name for name in _DESIGNED_NUCLEIC_BACKBONE if name != "O2'"), "N1"),
}
#: Element category 128 of the atom feature means an element this feature does not state; it asserts no element.
_UNAVAILABLE_ELEMENT = 128
_NUM_ELEMENTS = 129
_NUM_NAME_CHARACTERS = 64
_GAP = TOKEN_INDEX["-"]

ODESIGN_LAYOUT = LayoutTable(
    family="odesign",
    monomer_types={_PROTEIN: 0, _RNA: 1, _DNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="nonstandard",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA", _RNA: _NUCLEIC_CENTRE, _DNA: _NUCLEIC_CENTRE},
    frame_atoms={},
    representative={
        TOKEN_INDEX[token]: name
        for token, name in (("G", "CA"), ("RNA_A", "C4"), ("RNA_G", "C4"), ("DNA_A", "C4"), ("DNA_G", "C4"))
    },
    representative_default={_PROTEIN: "CB", _RNA: "C2", _DNA: "C2"},
)
#: ODesign feeds the element and the formal charge of every conditioned atom it consumes.
ODESIGN_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
ODESIGN_SYMMETRY = SymmetryPolicy(components="all", resonance=True, pin_unobserved=False)
#: Per artifact, the monomer types its generated positions may have.
ODESIGN_ARTIFACT_MONOMERS: Mapping[str, frozenset[MonomerType]] = {
    "protein_flex": frozenset({_PROTEIN}),
    "protein_rigid": frozenset({_PROTEIN}),
    "nucleic_acid_rigid": frozenset({_RNA, _DNA}),
}


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def designable_components(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""Per component, whether ODesign can generate it: its canonical class is a protein, RNA or DNA monomer."""
    return CLASS_MONOMER[token_classes(system).token] != _NONE


@dataclass(frozen=True)
class _ODesignPreparationContext(PreparationContext):
    filter_by_occupancy: bool = True


class ODesignProcessor(Processor):
    r"""
    Prepare one polymer-generation problem for ODesign.

    Args:
        config: The released artifact this processor prepares for.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ODesignProcessorConfig
    config: ODesignProcessorConfig

    def __init__(
        self,
        config: ODesignProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = self.config_class() if config is None else self.config_class(config)
        if seed is not None:
            self.config.seed = seed
        super().__init__(device=device, dtype=dtype)

    @property
    def seed(self) -> int:
        r"""Seed used to embed missing reference conformers."""
        return self.config.seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self.config.seed = seed

    def prepare_input(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        hotspots: Sequence[Hotspot] = (),
        fixed_atoms: Sequence[AtomSite] = (),
        constraint_atoms: Sequence[AtomSite] = (),
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ODesignInput]:
        r"""
        Prepare one structure: layout, optional training crop, slots, gating, reference conformers, MSA rows and the
        ODesign tensors.

        Args:
            structure: The conditioning complex, including the positions to generate.
            design: The positions ODesign generates (default every protein, RNA and DNA position).
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            hotspots: Residue and atom positions labelled as interaction hotspots.
            fixed_atoms: Atoms whose observed coordinates are kept and fed as context.
            constraint_atoms: Atoms whose observed pair distances steer the generation.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            cropper: Optional training crop over the tokens; needs ``generator``.
            generator: The numpy generator of the crop.

        Returns:
            The prepared task. Its regeneration scope is every generated coordinate: the atoms of the designed
            positions and, for a flexible-receptor artifact, the atoms of the context components it moves.

        Raises:
            UnsupportedError: ``model_capacity`` for a designed position whose monomer type the artifact cannot
                generate.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        designed = np.zeros(len(system.component_symbol), dtype=bool)
        designed[resolve_positions(structure, design, allowed=designable_components(system))] = True
        self._require_artifact_monomers(system, designed)
        fixed_mask = _atom_mask(structure, fixed_atoms)
        constraint = _atom_mask(structure, constraint_atoms)
        conditioned = fixed_mask | constraint
        hotspot_atoms = _hotspot_atoms(structure, hotspots)
        explicit_hotspots = _atom_mask(
            structure, tuple(site for hotspot in hotspots for site in hotspot.residues if isinstance(site, AtomSite))
        )
        layout = token_layout(
            system,
            ODESIGN_LAYOUT,
            atom_mask=_addressed_atoms(system, designed, preserve=conditioned | explicit_hotspots),
        )
        monomer = CLASS_MONOMER[token_classes(system).token]
        atom_component = _owner(system.component_atom_ptr)
        unsupported_fixed = np.flatnonzero(designed[atom_component] & fixed_mask & (layout.atom_token < 0))
        if len(unsupported_fixed):
            fail("model_capacity", StructureRef(atom=int(unsupported_fixed[0])), reason="fixed_atom_not_addressed")
        designed_nucleic = (designed & np.isin(monomer, (_RNA, _DNA)))[atom_component]
        unsupported = np.flatnonzero(designed_nucleic & (conditioned | explicit_hotspots) & (layout.atom_token < 0))
        if len(unsupported):
            fail("model_capacity", StructureRef(atom=int(unsupported[0])), reason="conditioned_atom_not_addressed")
        if cropper is not None:
            required = designed[layout.token_component] | _tokens_of(layout, conditioned | hotspot_atoms)
            layout = crop_token_layout(structure, layout, cropper, required=required, generator=generator)
        slots = _nucleic_model_slots(system, slot_map(system, layout, ODESIGN_LAYOUT), designed)
        accepted = gate(structure, slots, ODESIGN_GATING)
        template_rows, template_coordinates = _representative_reference(system, slots, designed)
        reference_slots = SlotMap(np.where(template_rows, -1, slots.atom), slots.token, slots.layout, slots.unaddressed)
        references, sources, embedded = reference_conformers(
            structure, reference_slots, embed=missing_reference(system), seed=self.seed
        )
        references[template_rows] = template_coordinates[template_rows]
        _, source_pool = system.reference_coordinates()
        template_atoms = slots.atom[template_rows]
        usable = (template_atoms < 0) | np.isin(
            source_pool[np.maximum(template_atoms, 0)], (ReferenceSource.IDEAL, ReferenceSource.MODEL)
        )
        sources[template_rows] = np.where(usable, ConformerSource.REFERENCE, ConformerSource.NONE)
        tokens, losses = model_tokens(system, layout)
        generated = designed[layout.token_component]
        available = _available(structure, filter_by_occupancy)
        _require_observed(conditioned, available)
        _require_constraint_centres(layout, constraint)
        context = (slots.atom >= 0) & ~generated[slots.token]
        fixed = _slot_mask(slots, fixed_mask)
        if self.config.artifact in ("protein_rigid", "nucleic_acid_rigid"):
            fixed |= context
        features = _features(
            structure,
            slots,
            tokens,
            generated,
            references,
            sources,
            _msa_plan(msa, system, layout),
            hotspot=_tokens_of(layout, hotspot_atoms),
            fixed=fixed,
            constrained=fixed | _slot_mask(slots, conditioned),
            available=available,
            device=self.device,
            dtype=self.dtype,
        )
        return PreparedInput(
            ODesignInput(**features, batch_size=[]),  # type: ignore[call-arg]
            _ODesignPreparationContext(
                structure,
                slots,
                RegenerationScope.atoms_of(system, _regenerated_atoms(system, slots, designed, fixed)),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    _conformer_sources(system, slots, sources),
                ),
                spec=self.preparation_spec,
                filter_by_occupancy=filter_by_occupancy,
            ),
        )

    def prepare(
        self,
        structure: Structure,
        *,
        target: Structure | None = None,
        filter_by_occupancy: bool = True,
        **options,
    ) -> tuple[PreparedInput[ODesignInput], ODesignStructureTarget | None]:
        r"""Prepare input and optional truth under the same explicit observation policy."""
        prepared = self.prepare_input(structure, filter_by_occupancy=filter_by_occupancy, **options)
        truth = None if target is None else self.prepare_target(target, prepared=prepared)
        return prepared, truth

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[ODesignInput],
    ) -> ODesignStructureTarget:
        r"""Align coordinate truth to an existing prepared input.

        Args:
            target: Independent truth in the task's Cartesian frame.
            prepared: Input features and correspondence returned by `prepare_input`.

        Returns:
            The family's coordinate target. Measured coordinates retain Float64
            precision; input preparation, crop, and reference sampling are not repeated.

        Notes:
            Truth is matched by instance, component ordinal and atom name. A generated nucleic linker instead
            aligns to the truth's N1/N9 only when its chemical identity and C1' bond are known.
            Conditioning and truth share the caller's Cartesian frame. Explicitly
            zero-occupancy coordinates are unresolved by default; the same
            occupancy policy selected during input preparation also applies to the target.
        """
        self._validate_prepared(prepared)
        truth = prepare_target(
            prepared,
            target,
            symmetry=ODESIGN_SYMMETRY,
            exclude_unoccupied=cast(_ODesignPreparationContext, prepared.context).filter_by_occupancy,
        )
        return _polymer_target_tensors(prepared, truth, target_structure=target, device=self.device)

    def _require_artifact_monomers(self, system: MolecularSystem, designed: NDArray[np.bool_]) -> None:
        allowed = ODESIGN_ARTIFACT_MONOMERS[self.config.artifact]
        monomer = CLASS_MONOMER[token_classes(system).token]
        outside = np.flatnonzero(designed & ~np.isin(monomer, sorted(allowed)))
        if len(outside):
            fail(
                "model_capacity",
                StructureRef(component=int(outside[0])),
                artifact=self.config.artifact,
                monomer=MonomerType(int(monomer[outside[0]])).name,
            )
        symbols = system.text("component", system.component_symbol)
        supported = ((monomer == _RNA) & np.isin(symbols, ("A", "G", "C", "U", "N"))) | (
            (monomer == _DNA) & np.isin(symbols, ("DA", "DG", "DC", "DT", "DN"))
        )
        unsupported = np.flatnonzero(designed & np.isin(monomer, (_RNA, _DNA)) & ~supported)
        if len(unsupported):
            fail("model_capacity", StructureRef(component=int(unsupported[0])), reason="unsupported_nucleic_design")


@dataclass(frozen=True)
class ODesignLigandTarget:
    r"""Ligand-training truth on the provisional slot axis, without assigning those slots chemical identities."""

    coordinates: NDArray[np.float64]
    coordinate_mask: NDArray[np.bool_]
    covalent_bond_index: NDArray[np.int64]
    covalent_pair_mask: NDArray[np.bool_]


class ODesignLigandProcessor(Processor):
    r"""
    Prepare provisional ligand generation for ODesign: the canonical context and the ordered generated slots.

    A generated slot is bound to no target atom: ODesign states its coordinates and its element probabilities, never
    a chemical vertex, so the preparation regenerates no canonical coordinate.

    Args:
        config: Input-processing configuration. Defaults to ``ODesignLigandProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ODesignLigandProcessorConfig
    config: ODesignLigandProcessorConfig

    def __init__(
        self,
        config: ODesignLigandProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = self.config_class() if config is None else self.config_class(config)
        if seed is not None:
            self.config.seed = seed
        super().__init__(device=device, dtype=dtype)

    @property
    def seed(self) -> int:
        r"""Seed used to embed missing reference conformers."""
        return self.config.seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self.config.seed = seed

    def prepare_input(
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
        Prepare one ligand-generation problem: the context as read plus ``generated_atoms`` provisional slots.

        Args:
            structure: The canonical context; its observed coordinates are conditioned and fixed.
            generated_atoms: The number of provisional atoms, or the range this preparation samples it from.
            msa: MSA pools by instance, an exact plan, or ``None``.
            hotspots: Residue and atom positions labelled as interaction hotspots.
            constraint_atoms: Atoms whose observed pair distances steer the generation.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            generator: The numpy generator the atom count is sampled with; a range needs one.
        """

        if isinstance(generated_atoms, range) and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        count = (
            int(generator.choice(np.asarray(generated_atoms)))  # type: ignore[union-attr]
            if isinstance(generated_atoms, range)
            else int(generated_atoms)
        )
        system = structure.system
        no_design = np.zeros(len(system.component_symbol), dtype=bool)
        layout = token_layout(system, ODESIGN_LAYOUT, atom_mask=_addressed_atoms(system, no_design))
        context = slot_map(system, layout, ODESIGN_LAYOUT)
        accepted = gate(structure, context, ODESIGN_GATING)
        references, sources, embedded = reference_conformers(
            structure, context, embed=missing_reference(system), seed=self.seed
        )
        tokens, losses = model_tokens(system, layout)
        available = _available(structure, filter_by_occupancy)
        constraint = _atom_mask(structure, constraint_atoms)
        _require_observed(constraint, available)
        _require_constraint_centres(layout, constraint)
        bound = context.atom >= 0
        slots = SlotMap(
            atom=np.concatenate((context.atom, np.full(count, -1, dtype=np.int64))),
            token=np.concatenate((context.token, len(layout.token_component) + np.arange(count))),
            layout=layout,
            unaddressed=context.unaddressed,
        )
        features = _features(
            structure,
            slots,
            np.concatenate((tokens, np.full(count, TOKEN_INDEX["OTHER"], dtype=np.int64))),
            np.concatenate((np.zeros(len(layout.token_component), dtype=bool), np.ones(count, dtype=bool))),
            np.concatenate((references, np.zeros((count, 3)))),
            np.concatenate((sources, np.full(count, ConformerSource.REFERENCE, dtype=np.uint8))),
            _msa_plan(msa, system, layout),
            hotspot=np.concatenate(
                (_tokens_of(layout, _hotspot_atoms(structure, hotspots)), np.zeros(count, dtype=bool))
            ),
            fixed=np.concatenate((bound, np.zeros(count, dtype=bool))),
            constrained=np.concatenate((bound | _slot_mask(context, constraint), np.zeros(count, dtype=bool))),
            available=available,
            device=self.device,
            dtype=self.dtype,
        )
        return PreparedInput(
            ODesignLigandInput(**features, batch_size=[]),  # type: ignore[call-arg]
            _ODesignPreparationContext(
                structure,
                slots,
                RegenerationScope.none(system),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    _conformer_sources(system, context, sources),
                ),
                spec=self.preparation_spec,
                filter_by_occupancy=filter_by_occupancy,
            ),
        )

    def prepare(
        self,
        structure: Structure,
        *,
        target: Structure | None = None,
        generated_target_instance_ids: Sequence[str] = (),
        generated_atoms: int | range | None = None,
        filter_by_occupancy: bool = True,
        **options,
    ) -> tuple[PreparedInput[ODesignLigandInput], ODesignStructureTarget | None]:
        r"""Prepare a ligand-generation sample, resolving a missing size from explicit truth instances.

        Only the generated atom count can enter input preparation from truth.
        Coordinates and covalent facts are aligned separately by `prepare_target`.

        For separate calls, supply that count to ``prepare_input`` and the same
        generated-instance order to ``prepare_target``. Both structures share
        the caller's Cartesian frame; fixed context coordinates come from input
        conditioning. The selected occupancy policy applies to both operations.
        """
        if target is None:
            if generated_atoms is None or generated_target_instance_ids:
                fail("value_invalid", RequestRef("generated_atoms", "explicit size required without truth"))
            return (
                self.prepare_input(
                    structure, generated_atoms=generated_atoms, filter_by_occupancy=filter_by_occupancy, **options
                ),
                None,
            )
        generated = _generated_target_rows(target, generated_target_instance_ids)
        count = len(generated)
        if generated_atoms is not None and (
            count not in generated_atoms if isinstance(generated_atoms, range) else count != generated_atoms
        ):
            fail("value_invalid", RequestRef("generated_atoms", str(generated_atoms)), target_count=count)
        prepared = self.prepare_input(
            structure, generated_atoms=count, filter_by_occupancy=filter_by_occupancy, **options
        )
        return prepared, self.prepare_target(
            target,
            prepared=prepared,
            generated_target_instance_ids=generated_target_instance_ids,
        )

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[ODesignLigandInput],
        generated_target_instance_ids: Sequence[str],
    ) -> ODesignStructureTarget:
        r"""Align independent coordinate and covalent truth to existing provisional slots.

        Args:
            target: Truth containing context and generated ligand instances in a common Cartesian frame.
            prepared: Existing input features and correspondence from `prepare_input`.
            generated_target_instance_ids: Generated instances, in supervision order.

        Returns:
            Coordinate and covalent supervision in the prepared slot order. The
            source conditioning and task dimensions remain unchanged.

        Raises:
            InvalidRequestError: Generated identities are empty, repeated, absent,
                overlap context, or do not match the prepared generated atom count.
        """
        self._validate_prepared(prepared)
        generated = _generated_target_rows(target, generated_target_instance_ids)
        slots = prepared.context.slots
        count = int(np.count_nonzero(slots.atom < 0))
        if len(generated) != count:
            fail("value_invalid", RequestRef("generated_atoms", str(count)), target_count=len(generated))
        context = slots.atom[:-count]
        context_rows = match_atoms(prepared.context.structure.system, context, target)
        if np.isin(context_rows, generated).any():
            fail(
                "value_invalid",
                RequestRef("generated_target_instance_ids", repr(tuple(generated_target_instance_ids))),
                invariant="outside_context",
            )
        rows = np.concatenate((context_rows, generated))
        available = _available(target, cast(_ODesignPreparationContext, prepared.context).filter_by_occupancy)
        observed = np.append(available, False)[rows]
        coordinates = np.concatenate((target.coordinates, np.zeros((1, 3))))[rows]
        coordinates = np.where(observed[:, None], coordinates, 0.0)
        edges, pair = _target_topology(target.system, rows)
        return _target_tensors(prepared, coordinates, observed, edges, pair, device=self.device)


def _generated_target_rows(target: Structure, instance_ids: Sequence[str]) -> NDArray[np.int64]:
    r"""Select the explicitly ordered ligand truth atoms used to define provisional slots."""
    names = tuple(instance_ids)
    if not names or len(set(names)) != len(names):
        fail("value_invalid", RequestRef("generated_target_instance_ids", repr(names)), invariant="nonempty_unique")
    system = target.system
    instances = system.instance_index(names)
    if (instances < 0).any():
        fail("value_invalid", RequestRef("generated_target_instance_ids", names[int(np.flatnonzero(instances < 0)[0])]))
    component, atom = system.instance_component_ptr, system.component_atom_ptr
    generated = np.concatenate(
        [np.arange(atom[component[index]], atom[component[index + 1]], dtype=np.int64) for index in instances]
    )
    if not len(generated):
        fail("value_invalid", RequestRef("generated_target_instance_ids", repr(names)), target_count=0)
    return generated


def _conformer_sources(system: MolecularSystem, slots: SlotMap, sources: NDArray[np.uint8]) -> NDArray[np.uint8]:
    r"""Per component, where its slot atoms' conformer came from."""
    conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
    bound = slots.atom >= 0
    conformers[_owner(system.component_atom_ptr)[slots.atom[bound]]] = sources[bound]
    return conformers


def _slot_mask(slots: SlotMap, atoms: NDArray[np.bool_]) -> NDArray[np.bool_]:
    r"""The slots whose target atom is in ``atoms``."""
    return (slots.atom >= 0) & atoms[np.maximum(slots.atom, 0)]


def _available(structure: Structure, filter_by_occupancy: bool) -> NDArray[np.bool_]:
    r"""The atoms whose coordinates this preparation reads as observed."""
    available = structure.coordinate_mask.copy()
    if filter_by_occupancy:
        available &= ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
    return available


def _require_observed(named: NDArray[np.bool_], available: NDArray[np.bool_]) -> None:
    r"""An explicitly conditioned atom without observed coordinates is ``request_invalid_positions``."""
    missing = np.flatnonzero(named & ~available)
    if len(missing):
        fail("request_invalid_positions", StructureRef(atom=int(missing[0])), reason="unobserved")


def _require_constraint_centres(layout: TokenLayout, requested: NDArray[np.bool_]) -> None:
    r"""The token distogram can express only an atom token or a residue token's centre atom."""
    atoms = np.flatnonzero(requested)
    tokens = layout.atom_token[atoms]
    valid = tokens >= 0
    valid[valid] &= layout.token_centre_atom[tokens[valid]] == atoms[valid]
    if not valid.all():
        fail(
            "model_capacity",
            StructureRef(atom=int(atoms[np.flatnonzero(~valid)[0]])),
            reason="constraint_atom_not_token_centre",
        )


def _tokens_of(layout: TokenLayout, atoms: NDArray[np.bool_]) -> NDArray[np.bool_]:
    r"""The tokens holding one of ``atoms``."""
    tokens = np.zeros(len(layout.token_component), dtype=bool)
    named = layout.atom_token[atoms]
    tokens[named[named >= 0]] = True
    return tokens


def _atom_mask(structure: Structure, sites: Sequence[AtomSite]) -> NDArray[np.bool_]:
    r"""The atoms named by ``sites``; a site the structure does not hold is ``request_invalid_positions``."""
    mask = np.zeros(len(structure.system.atom_symbol), dtype=bool)
    if not sites:
        return mask
    found = structure.system.atom_index(list(sites))
    if (found < 0).any():
        fail("request_invalid_positions", RequestRef("atoms", str(sites[int(np.flatnonzero(found < 0)[0])])))
    mask[found] = True
    return mask


def _hotspot_atoms(structure: Structure, hotspots: Sequence[Hotspot]) -> NDArray[np.bool_]:
    r"""
    The atoms the hotspot conditions name (a residue names all of its atoms).

    Raises:
        InvalidRequestError: ``request_invalid_positions`` for an endpoint the structure does not hold, or for a
            symbolic endpoint (ODesign prepares a read structure, never a placement).
    """

    system = structure.system
    mask = np.zeros(len(system.atom_symbol), dtype=bool)
    pointer = system.component_atom_ptr
    for condition in hotspots:
        for endpoint in condition.residues:
            if isinstance(endpoint, AtomSite):
                mask |= _atom_mask(structure, (endpoint,))
                continue
            if not isinstance(endpoint, ComponentRef):
                fail("request_invalid_positions", RequestRef("hotspots", str(endpoint)), symbolic=True)
            component = int(system.component_index([endpoint])[0])
            if component < 0:
                fail("request_invalid_positions", RequestRef("hotspots", str(endpoint)))
            mask[pointer[component] : pointer[component + 1]] = True
    return mask


def _addressed_atoms(
    system: MolecularSystem, designed: NDArray[np.bool_], *, preserve: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    r"""Retain designed polymer backbone atoms, plus any explicitly conditioned atoms."""
    names = system.text("atom", system.atom_symbol)
    component = _owner(system.component_atom_ptr)
    monomer = CLASS_MONOMER[token_classes(system).token]
    protein = designed & (monomer == _PROTEIN)
    nucleic = designed & np.isin(monomer, (_RNA, _DNA))
    symbols = system.text("component", system.component_symbol)
    glycosidic = np.array([_NUCLEIC_GLYCOSIDIC_N.get(str(symbol), "") for symbol in symbols])
    keep = (~protein[component] | np.isin(names, _DESIGNED_BACKBONE)) & (
        ~nucleic[component] | np.isin(names, _DESIGNED_NUCLEIC_BACKBONE) | (names == glycosidic[component])
    )
    return keep if preserve is None else keep | (preserve & nucleic[component])


@lru_cache(maxsize=1)
def _representative_templates() -> dict[MonomerType, dict[str, NDArray[np.float64]]]:
    r"""The released model's representative GLY/C/DC conformers, never canonical target chemistry."""
    resource = "models/odesign/reference.safetensors"
    try:
        data = load_safetensors(files("ofoldx").joinpath(resource).read_bytes())
    except (OSError, SafetensorError) as error:
        fail("native_corrupt", SourceRef(resource, 0, item="payload"), reason=str(error))
    result = {}
    for kind, key in ((_PROTEIN, "GLY"), (_RNA, "C"), (_DNA, "DC")):
        coordinates = data.get(key)
        names = _REFERENCE_NAMES[kind]
        if (
            coordinates is None
            or coordinates.shape != (len(names), 3)
            or not np.issubdtype(coordinates.dtype, np.floating)
            or not np.isfinite(coordinates).all()
        ):
            fail("native_corrupt", SourceRef(resource, 0, item=key))
        result[kind] = dict(zip(names, coordinates, strict=True))
    return result


def _nucleic_model_slots(system: MolecularSystem, slots: SlotMap, designed: NDArray[np.bool_]) -> SlotMap:
    r"""Add model-only base-linker slots for designed unknown RNA/DNA, without creating chemical atoms."""
    components = slots.layout.token_component
    symbols = system.text("component", system.component_symbol)
    monomer = CLASS_MONOMER[token_classes(system).token]
    unknown = designed[components] & (
        ((monomer[components] == _RNA) & (symbols[components] == "N"))
        | ((monomer[components] == _DNA) & (symbols[components] == "DN"))
    )
    virtual_tokens = np.flatnonzero(unknown)
    if not len(virtual_tokens):
        return slots
    names = system.text("atom", system.atom_symbol)
    owners = _owner(system.component_atom_ptr)
    existing = np.isin(owners, components[virtual_tokens]) & np.isin(names, ("N1", "N9"))
    if existing.any():
        fail("model_capacity", StructureRef(atom=int(np.flatnonzero(existing)[0])), reason="unknown_linker_present")
    atoms = np.concatenate((slots.atom, np.full(len(virtual_tokens), -1, dtype=np.int64)))
    tokens = np.concatenate((slots.token, virtual_tokens))
    order = np.argsort(tokens, kind="stable")
    return SlotMap(atoms[order], tokens[order], slots.layout, slots.unaddressed)


def _representative_reference(
    system: MolecularSystem, slots: SlotMap, designed: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    r"""Replace only generated polymer-backbone reference features with the upstream model templates."""
    names = _atom_names(system, slots)
    component_ids = system.text("component", system.component_symbol)
    monomer = CLASS_MONOMER[token_classes(system).token]
    selected = np.zeros(len(slots.atom), dtype=bool)
    coordinates = np.zeros((len(slots.atom), 3), dtype=np.float64)
    templates = _representative_templates()
    for row, (atom, token) in enumerate(zip(slots.atom.tolist(), slots.token.tolist())):
        component = int(slots.layout.token_component[token])
        kind = MonomerType(int(monomer[component]))
        if not designed[component] or kind not in templates:
            continue
        if atom < 0:
            if kind in (_RNA, _DNA) and component_ids[component] in ("N", "DN"):
                selected[row] = True
                coordinates[row] = templates[kind]["N1"]
            continue
        name = names[row]
        if kind in (_RNA, _DNA) and name in ("N1", "N9"):
            if name != _NUCLEIC_GLYCOSIDIC_N.get(str(component_ids[component])):
                continue
            name = "N1"
        if name in templates[kind]:
            selected[row] = True
            coordinates[row] = templates[kind][name]
    return selected, coordinates


def _regenerated_atoms(
    system: MolecularSystem, slots: SlotMap, designed: NDArray[np.bool_], fixed: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    r"""
    The task's regeneration scope: every atom of a generated position and of a context component the model moves (a
    flexible receptor), their unaddressed atoms included, so those are invalidated and counted — a hydrogen the
    layout excludes does not keep an observation of a frame that moved — minus the fixed atoms.
    """

    owner = _owner(system.component_atom_ptr)
    bound = slots.atom >= 0
    moved = designed.copy()
    moved[owner[slots.atom[bound & ~fixed]]] = True
    atoms = moved[owner]
    atoms[slots.atom[bound & fixed]] = False
    return atoms


def _msa_plan(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout
) -> MsaCollection | None:
    r"""Pools are paired by the one OFoldX rule; an exact plan is used as given."""
    if msa is None:
        return None
    return msa if isinstance(msa, MsaCollection) else pair_msas(msa, system, layout)


def _unpaired_plan(plan: MsaCollection) -> MsaCollection:
    r"""
    The same pools with one row per unpaired row of each pool: the rows ODesign's profile is taken over.

    The plan is as deep as the deepest pool, so a shallower pool's columns are gaps below its own rows; the statistic
    reads each column over that pool's own depth.
    """
    depth = max(len(pool.unpaired_sequences) for pool in plan.msas)
    pairing = np.full((depth, len(plan.msas)), -1, dtype=np.int64)
    for column, pool in enumerate(plan.msas):
        pairing[: len(pool.unpaired_sequences), column] = np.arange(len(pool.unpaired_sequences))
    paired = np.zeros(pairing.shape, dtype=bool)
    pairing[0] = 0
    paired[0] = True
    return MsaCollection(plan.msas, pairing, paired, plan.target_instance_ids)


def _msa_features(
    plan: MsaCollection | None,
    system: MolecularSystem,
    layout: TokenLayout,
    tokens: NDArray[np.int64],
    generated: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    The MSA rows, deletion counts, per-position profile and mean deletion count over the prepared tokens.

    The profile and the mean are ODesign's own statistic over each pool's unpaired rows (its query included), never
    over the row plan: each position is normalised by the depth of the one pool covering it, and a position no pool
    covers states its own class and no deletion. A generated position states no residue: its row and its profile are
    the gap class.
    """

    count, width = len(tokens), len(layout.token_component)
    identity = np.where(generated, _GAP, tokens)
    if plan is None:
        return (
            identity[None].astype(np.int64),
            np.zeros((1, count), dtype=np.int64),
            np.eye(len(TOKENS))[identity],
            np.zeros(count, dtype=np.float64),
        )
    spread = msa_rows(plan, system, layout)
    rows = np.full((spread.token.shape[0], count), _GAP, dtype=np.int64)
    rows[:, :width] = spread.token
    deletions = np.zeros((spread.token.shape[0], count), dtype=np.int64)
    deletions[:, :width] = spread.deletions
    unpaired = msa_rows(_unpaired_plan(plan), system, layout)
    pool_depth = np.array([len(pool.unpaired_sequences) for pool in plan.msas])
    depth = np.where(unpaired.column < 0, 1, pool_depth[unpaired.column])
    read = np.arange(len(unpaired.token))[:, None] < depth[None, :]
    profile = np.eye(len(TOKENS))[identity]
    profile[:width] = (np.eye(len(TOKENS))[unpaired.token] * read[..., None]).sum(axis=0) / depth[:, None]
    mean = np.zeros(count, dtype=np.float64)
    mean[:width] = np.where(read, unpaired.deletions, 0).sum(axis=0) / depth
    profile[generated] = np.eye(len(TOKENS))[_GAP]
    return (
        np.where(generated, _GAP, rows),
        np.where(generated, 0, deletions),
        profile,
        np.where(generated, 0.0, mean),
    )


def _atom_names(system: MolecularSystem, slots: SlotMap) -> list[str]:
    r"""Per slot, its atom name in four characters; a slot bound to no atom is named ``-``.

    Raises:
        InvalidRequestError: ``value_invalid`` for a name this feature cannot encode (over four characters, or a
            character outside its 64 printable ones).
    """

    bound = slots.atom >= 0
    names = np.full(len(slots.atom), "-", dtype=object)
    names[bound] = system.text("atom", system.atom_symbol[slots.atom[bound]])
    unencodable = [
        name for name in names.tolist() if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name)
    ]
    if unencodable:
        fail("value_invalid", RequestRef("atom_name", unencodable[0]))
    return names.tolist()


def _name_characters(names: Sequence[str]) -> NDArray[np.int64]:
    r"""Per slot, the printable code of each of its four name characters."""
    codes = [[max(ord(character) - 32, 0) for character in name.ljust(4)] for name in names]  # a pad is a space
    return np.array(codes, dtype=np.int64).reshape(len(names), 4)


def _elements(system: MolecularSystem, slots: SlotMap, hidden: NDArray[np.bool_]) -> NDArray[np.int64]:
    r"""Per slot, its element category: ``atomic_number - 1``, 128 where this feature states no element."""
    number = np.append(system.atomic_number, 0)[slots.atom].astype(np.int64)  # no atom (-1): the appended 0
    return np.where((number == 0) | hidden, _UNAVAILABLE_ELEMENT, number - 1)


def _centred_references(
    references: NDArray[np.float64], sources: NDArray[np.uint8], space: NDArray[np.int64], spaces: int
) -> NDArray[np.float64]:
    r"""The reference conformers with each space's available atoms centred on their own centroid (upstream)."""
    available = sources != ConformerSource.NONE
    coordinates = np.where(available[:, None], references, 0.0)
    sums = np.zeros((spaces, 3), dtype=np.float64)
    np.add.at(sums, space, coordinates)
    counts = np.bincount(space, weights=available, minlength=spaces)
    centres = sums / np.maximum(counts, 1)[:, None]
    return np.where(available[:, None], coordinates - centres[space], 0.0)


def _cyclic_periods(system: MolecularSystem) -> NDArray[np.int64]:
    r"""Per instance, its component count when a KNOWN covalent edge joins its first component's previous port to its
    last component's next port (a head-to-tail cyclic polymer), else 0."""
    s = system.arrays()
    pointer = s["instance_component_ptr"]
    periods = np.zeros(len(pointer) - 1, dtype=np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    pairs = {tuple(edge) for edge in s["edge_index"][:, covalent].T.tolist()}
    for instance, (start, stop) in enumerate(zip(pointer[:-1].tolist(), pointer[1:].tolist())):
        if stop - start < 2:
            continue
        head, tail = int(s["component_port"][start, 0]), int(s["component_port"][stop - 1, 1])
        if head >= 0 and tail >= 0 and (min(head, tail), max(head, tail)) in pairs:
            periods[instance] = stop - start
    return periods


def _token_bonds(system: MolecularSystem, layout: TokenLayout, count: int) -> NDArray[np.float64]:
    r"""
    ``[N, N]``: 1 between two tokens a KNOWN covalent edge joins, except polymer junctions — an edge from one
    component's next port to another's previous port in the same instance — which upstream never lists.
    """

    s = system.arrays()
    first, second = s["edge_index"].astype(np.int64)
    component = _owner(s["component_atom_ptr"])
    left, right = component[first], component[second]
    port = s["component_port"].astype(np.int64)
    junction = ((port[left, 1] == first) & (port[right, 0] == second)) | (
        (port[left, 0] == first) & (port[right, 1] == second)
    )
    instance = _owner(s["instance_component_ptr"])
    junction &= (left != right) & (instance[left] == instance[right])
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = covalent & ~junction & (a >= 0) & (b >= 0) & (a != b)
    bonds = np.zeros((count, count), dtype=np.float64)
    bonds[a[linked], b[linked]] = 1
    bonds[b[linked], a[linked]] = 1
    return bonds


def _constraint_distogram(
    structure: Structure,
    slots: SlotMap,
    atom_token: NDArray[np.bool_],
    constrained: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    count: int,
) -> NDArray[np.float64]:
    r"""
    ``[N, N]`` observed pair distances between the constrained tokens: an atom token takes its own atom, a residue
    token its centre atom (one steering distance per token).
    """

    bound = slots.atom >= 0
    centre = np.zeros(len(slots.atom), dtype=bool)
    centre[bound] = np.isin(slots.atom[bound], slots.layout.token_centre_atom)
    rows = np.flatnonzero(constrained & observed & bound & (atom_token[slots.token] | centre))
    values = np.zeros((count, count), dtype=np.float64)
    if not len(rows):
        return values
    tokens = slots.token[rows]
    points = structure.coordinates[slots.atom[rows]]
    values[tokens[:, None], tokens[None]] = np.linalg.norm(points[:, None] - points[None], axis=-1)
    return values


def _copy_index(entity: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Per instance, its ordinal among the instances of its entity."""
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    return copy


def _token_axis(
    system: MolecularSystem, layout: TokenLayout, count: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    r"""Per token (provisional tokens last), its residue index, chain, entity, copy and cyclic period."""
    s = system.arrays()
    instances, entities = len(s["instance_symbol"]), len(s["entity_symbol"])
    component = np.concatenate((layout.token_component, np.full(count - len(layout.token_component), -1)))
    instance = np.append(_owner(s["instance_component_ptr"]), instances)[component]  # provisional (-1): its own
    entity = np.concatenate((s["instance_entity"].astype(np.int64), [entities]))
    polymer = known(s["entity_type_state"][s["instance_entity"]]) & (
        s["entity_type"][s["instance_entity"]] == EntityType.POLYMER
    )
    starts = np.concatenate((s["instance_component_ptr"][:-1], [0]))
    residue = np.where(np.concatenate((polymer, [False]))[instance] & (component >= 0), component - starts[instance], 0)
    copy = np.concatenate((_copy_index(s["instance_entity"].astype(np.int64)), [0]))
    periods = np.concatenate((_cyclic_periods(system), [0]))
    return residue, instance, entity[instance], copy[instance], periods[instance]


def _features(
    structure: Structure,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    generated: NDArray[np.bool_],
    references: NDArray[np.float64],
    sources: NDArray[np.uint8],
    plan: MsaCollection | None,
    *,
    hotspot: NDArray[np.bool_],
    fixed: NDArray[np.bool_],
    constrained: NDArray[np.bool_],
    available: NDArray[np.bool_],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    r"""The ODesign tensors of one prepared task (the upstream featurisation over its slot and token axes)."""
    system = structure.system
    s = system.arrays()
    layout = slots.layout
    num_slots, num_tokens, spaces = len(slots.atom), len(tokens), len(s["component_symbol"]) + 1
    bound = slots.atom >= 0
    classes = token_classes(system)
    monomer = CLASS_MONOMER[classes.token]
    layout_tokens = len(layout.token_component)
    token_monomer = np.full(num_tokens, _NONE, dtype=monomer.dtype)
    token_monomer[:layout_tokens] = monomer[layout.token_component]
    atom_token = np.ones(num_tokens, dtype=bool)
    atom_token[:layout_tokens] = (classes.encoding != Encoding.EXACT)[layout.token_component] | (
        monomer[layout.token_component] == _NONE
    )

    hidden = np.array([TOKEN_INDEX[_UNKNOWN_CLASS[MonomerType(int(m))]] for m in token_monomer.tolist()])
    identity = np.where(generated, hidden, tokens)
    design_state = np.array([_DESIGN_STATE[MonomerType(int(m))] for m in token_monomer.tolist()])
    design_features = np.zeros((num_tokens, len(_DESIGN_STATE)), dtype=np.float64)
    design_features[np.flatnonzero(generated), design_state[generated]] = 1
    rows, deletions, profile, deletion_mean = _msa_features(plan, system, layout, tokens, generated)
    token_features = np.concatenate(
        (np.eye(len(TOKENS))[identity], profile, deletion_mean[:, None], hotspot[:, None].astype(np.float64)), axis=1
    )

    space = np.append(_owner(s["component_atom_ptr"]), spaces - 1)[slots.atom]  # no atom (-1): the last space
    virtual_polymer = ~bound & (slots.token < layout_tokens)
    space[virtual_polymer] = layout.token_component[slots.token[virtual_polymer]]
    reference_mask = sources != ConformerSource.NONE
    generated_atom = generated[slots.token]
    hidden_element = generated_atom & (token_monomer[slots.token] == _NONE)
    charges = np.where(generated_atom, 0, np.append(s["formal_charge"], 0)[slots.atom]).astype(np.float64)
    model_names = _atom_names(system, slots)
    for row in np.flatnonzero(virtual_polymer):
        model_names[row] = "N"
    for row, token in enumerate(slots.token.tolist()):
        if generated[token] and token_monomer[token] in (_RNA, _DNA) and model_names[row] in ("N1", "N9"):
            model_names[row] = "N"
    elements = _elements(system, slots, hidden_element)
    elements[virtual_polymer] = 6  # atomic number 7, generic model-only glycosidic nitrogen
    atom_features = np.concatenate(
        (
            reference_mask[:, None].astype(np.float64),
            np.eye(_NUM_ELEMENTS)[elements],
            np.eye(_NUM_NAME_CHARACTERS)[_name_characters(model_names)].reshape(num_slots, 4 * _NUM_NAME_CHARACTERS),
        ),
        axis=1,
    )

    observed = np.append(available, False)[slots.atom]
    positions = np.concatenate((structure.coordinates, np.zeros((1, 3))))[slots.atom]  # no atom (-1): the appended 0
    observed_positions = np.where(observed[:, None], positions, 0.0)
    residue_index, chain_index, entity_index, symmetry_index, cyclic_period = _token_axis(system, layout, num_tokens)
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[slots.atom[bound]] = np.flatnonzero(bound)
    representative = np.concatenate(
        (slot_of[token_representatives(system, layout, ODESIGN_LAYOUT)], np.flatnonzero(~bound & ~virtual_polymer))
    )

    tensors: dict[str, Tensor] = {
        "token_mask": torch.ones(num_tokens, dtype=torch.bool),
        "residue_index": torch.from_numpy(residue_index),
        "chain_index": torch.from_numpy(chain_index),
        "entity_index": torch.from_numpy(entity_index),
        "symmetry_index": torch.from_numpy(symmetry_index),
        "token_index": torch.arange(num_tokens),
        "cyclic_period": torch.from_numpy(cyclic_period),
        "token_features": torch.from_numpy(token_features),
        "design_features": torch.from_numpy(design_features),
        "token_bonds": torch.from_numpy(
            np.pad(
                _token_bonds(system, layout, layout_tokens),
                ((0, num_tokens - layout_tokens), (0, num_tokens - layout_tokens)),
            )
        ),
        "atom_features": torch.from_numpy(atom_features),
        "atom_to_token": torch.from_numpy(slots.token.copy()),
        "atom_mask": torch.ones(num_slots, dtype=torch.bool),
        "atom_monomer": torch.from_numpy(token_monomer[slots.token].astype(np.uint8)),
        "atom_token_mask": torch.from_numpy(atom_token),
        "generated_atom_mask": torch.from_numpy(~bound),
        "reference_positions": torch.from_numpy(_centred_references(references, sources, space, spaces)),
        "reference_space_id": torch.from_numpy(space),
        "reference_mask": torch.from_numpy(reference_mask),
        "reference_charges": torch.from_numpy(charges).unsqueeze(-1),
        "observed_positions": torch.from_numpy(observed_positions),
        "observed_mask": torch.from_numpy(observed),
        "fixed_coordinate_mask": torch.from_numpy(fixed & observed),
        "constraint_distogram": torch.from_numpy(
            _constraint_distogram(structure, slots, atom_token, constrained, observed, num_tokens)
        ).unsqueeze(-1),
        "msa_tokens": torch.from_numpy(rows),
        "msa_deletion_counts": torch.from_numpy(deletions),
        "msa_mask": torch.ones(rows.shape, dtype=torch.bool),
        "token_representative_atom_index": torch.from_numpy(representative),
    }
    placed = {
        name: (
            project_model_features(
                value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"ODesign prepared {name}"
            ).to(device=device)
            if value.is_floating_point()
            else value.to(device=device)
        )
        for name, value in tensors.items()
    }
    return {name: value.unsqueeze(0) for name, value in placed.items()}


def _polymer_target_tensors(
    prepared: PreparedInput[ODesignInput],
    target: StructureTarget,
    *,
    target_structure: Structure,
    device: torch.device,
) -> ODesignStructureTarget:
    r"""
    Truth on the slot axis: the OBSERVED coordinates, the covalent edges between slot atoms, and the pairs whose
    covalent status is known — a stated edge, or two atoms whose components and instances state a complete topology.
    """

    rows = prepared.context.slots.atom
    observed = np.append(target.truth == TruthState.OBSERVED, False)[rows]
    coordinates = np.concatenate((target.coordinates, np.zeros((1, 3))))[rows]
    coordinates = np.where(observed[:, None], coordinates, 0.0)
    edges, pair = _target_topology(prepared.context.structure.system, rows)
    edges, pair = _align_nucleic_linker_truth(prepared, target_structure, coordinates, observed, edges, pair)
    return _target_tensors(prepared, coordinates, observed, edges, pair, device=device)


def _align_nucleic_linker_truth(
    prepared: PreparedInput[ODesignInput],
    target: Structure,
    coordinates: NDArray[np.float64],
    observed: NDArray[np.bool_],
    edges: NDArray[np.int64],
    pair: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    r"""Align generated linker slots to the truth's verified N1/N9, including unbound input slots."""
    source = prepared.context.structure.system
    slots = prepared.context.slots
    rows = slots.atom
    bound = rows >= 0
    target_rows = np.full(len(rows), -1, dtype=np.int64)
    target_rows[bound] = match_atoms(source, rows[bound], target)
    source_ids = source.text("component", source.component_symbol)
    source_monomer = CLASS_MONOMER[token_classes(source).token]
    target_ids = target.system.text("component", target.system.component_symbol)
    target_monomer = CLASS_MONOMER[token_classes(target.system).token]
    generated = prepared.features.design_features[0, :, 1:3].ne(0).any(dim=-1).detach().cpu().numpy()
    names = _atom_names(source, slots)
    linker_rows = []
    for token, component in enumerate(slots.layout.token_component.tolist()):
        kind = MonomerType(int(source_monomer[component]))
        if not generated[token] or kind not in (_RNA, _DNA):
            continue
        source_id = str(source_ids[component])
        candidates = np.flatnonzero(slots.token == token)
        if source_id in ("N", "DN"):
            candidates = candidates[rows[candidates] < 0]
        else:
            source_name = _NUCLEIC_GLYCOSIDIC_N.get(source_id)
            candidates = np.array([row for row in candidates if names[row] == source_name], dtype=np.int64)
        if len(candidates) != 1:
            continue
        row = int(candidates[0])
        if rows[row] >= 0 and not prepared.context.scope.atoms[rows[row]]:
            continue
        linker_rows.append(row)
        target_rows[row] = -1
        observed[row] = False
        coordinates[row] = 0
        ref = source.component_refs(np.array([component], dtype=np.int64))[0]
        component_index = int(target.system.component_index([ref])[0])
        if component_index < 0 or MonomerType(int(target_monomer[component_index])) != kind:
            continue
        target_name = _NUCLEIC_GLYCOSIDIC_N.get(str(target_ids[component_index]))
        if target_name is None:
            continue
        atom, anchor = target.system.atom_index(
            (AtomSite(ref.instance, ref.component, target_name), AtomSite(ref.instance, ref.component, "C1'"))
        )
        if atom < 0 or anchor < 0:
            continue
        target_edges = target.system.edge_index
        linked = ((target_edges[0] == atom) & (target_edges[1] == anchor)) | (
            (target_edges[1] == atom) & (target_edges[0] == anchor)
        )
        known_bond = known(target.system.bond_kind_state) & (target.system.bond_kind == BondKind.COVALENT)
        endpoints = np.array([atom, anchor], dtype=np.int64)
        known_elements = known(target.system.atomic_number_state[endpoints]).all()
        fate_known = (target.system.atom_fate[endpoints] == AtomFate.RETAINED).all()
        endpoint_state = endpoint_resolution(target.system.atom_endpoint_state[endpoints])
        endpoints_known = not np.isin(endpoint_state, (Resolution.UNKNOWN, Resolution.CONFLICT)).any()
        if (
            not known_elements
            or not fate_known
            or not endpoints_known
            or target.system.atomic_number[atom] != 7
            or target.system.atomic_number[anchor] != 6
            or not np.any(linked & known_bond)
            or np.count_nonzero(target_rows == atom) != 0
        ):
            continue
        target_rows[row] = atom
        available = bool(target.coordinate_mask[atom])
        exclude = cast(_ODesignPreparationContext, prepared.context).filter_by_occupancy
        if exclude and known(target.occupancy_state[atom]) and target.occupancy[atom] <= 0:
            available = False
        observed[row] = available
        if available:
            coordinates[row] = target.coordinates[atom]
    if not linker_rows:
        return edges, pair
    selected = np.asarray(linker_rows, dtype=np.int64)
    edges = edges[~np.isin(edges, selected).any(axis=1)]
    pair[selected, :] = False
    pair[:, selected] = False
    target_edges, target_pair = _target_topology(target.system, target_rows)
    edges = np.concatenate((edges, target_edges[np.isin(target_edges, selected).any(axis=1)]))
    pair[selected, :] = target_pair[selected, :]
    pair[:, selected] = target_pair[:, selected]
    return edges, pair


def _target_topology(system: MolecularSystem, rows: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    r"""Covalent edges and known pair status on a slot-to-canonical-atom map (``-1`` has no truth atom)."""
    s = system.arrays()
    bound = rows >= 0
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[rows[bound]] = np.flatnonzero(bound)

    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    left, right = slot_of[first], slot_of[second]
    edges = covalent & (left >= 0) & (right >= 0) & (left != right)
    closed = s["component_topology_complete"] & s["instance_connections_complete"][_owner(s["instance_component_ptr"])]
    complete = closed[_owner(s["component_atom_ptr"])]
    determined = np.append(complete, False)[rows]
    pair = determined[:, None] & determined[None]
    pair[left[edges], right[edges]] = True
    pair[right[edges], left[edges]] = True
    return np.stack((left[edges], right[edges]), axis=1), pair


def _target_tensors(
    prepared: PreparedInput[ODesignInput] | PreparedInput[ODesignLigandInput],
    coordinates: NDArray[np.float64],
    observed: NDArray[np.bool_],
    edges: NDArray[np.int64],
    pair: NDArray[np.bool_],
    *,
    device: torch.device,
) -> ODesignStructureTarget:
    r"""The shared finite coordinate projection and placement of ODesign's scientific training tensors."""
    rows = prepared.context.slots.atom
    condition = np.concatenate((prepared.context.structure.coordinates, np.zeros((1, 3))))[rows]
    fixed = prepared.features.fixed_coordinate_mask[0].to(device)
    mask = torch.from_numpy(observed).to(device) | fixed
    points = torch.where(
        fixed[:, None], torch.from_numpy(condition).to(device), torch.from_numpy(coordinates).to(device)
    )
    projected = project_model_coordinates(points, mask, feature="ODesign truth")
    return ODesignStructureTarget(  # type: ignore[call-arg]
        coordinates=projected.unsqueeze(0),
        coordinate_mask=mask.unsqueeze(0),
        covalent_bond_index=torch.from_numpy(edges).to(device).unsqueeze(0),
        covalent_pair_mask=torch.from_numpy(pair).to(device).unsqueeze(0),
        batch_size=[1],
    )
