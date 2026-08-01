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
Protenix input processing, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own (layout, gating, crop, MSA pairing, template target
resolution, reference conformers, constraint features); this module keeps only the Protenix featurisation of those
inputs, which matches upstream for identical inputs: the MSA letter classes (``B``, ``Z``, ``U`` as ``D``,
``E``, ``C``; ``J``, ``O`` as ``X``), the MSA profile over the pool's unique unpaired rows, the query-only MSA and empty
template slots of chains of at most four residues, four template slots per chain, the template frame with its ``1e-6``
terms, and the query fill of a non-protein chain's all-gap MSA rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.rdkit import ConstraintFeatures, constraint_features
from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import AtomSite, BondKind, ComponentRef, EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKENS, Encoding, TokenLayout, encode_sequence, token_classes, tokenize
from ...errors import RequestRef, StructureRef, fail
from ...modules.functional import coordinate_distogram, pairwise_distance
from ...modules.geometry import uniform_random_rotation
from ...msa import MSA, MsaCollection, msa_rows, pair_msas, unique_rows
from ...msa.alignment import instance_monomer
from ...processor import Processor
from ...provenance import Records, known
from ...templates import Template, resolve_template, template_alignment_frames, template_atoms
from ..layout import LayoutTable, model_tokens, slot_map, token_frames, token_layout, token_representatives
from ..preparation import (
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
from .config import ProtenixProcessorConfig
from .model import (
    ProtenixAtomContact,
    ProtenixInput,
    ProtenixPocket,
    ProtenixStructureTarget,
    ProtenixTemplateFeatures,
    ProtenixTfgFeatures,
    ProtenixTokenContact,
)

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA
_NUCLEIC_FRAME = ("C1'", "C3'", "C4'")

PROTENIX_LAYOUT = LayoutTable(
    family="protenix",
    monomer_types={_PROTEIN: 0, _RNA: 1, _DNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="nonstandard",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={_PROTEIN: ("N", "CA", "C"), _DNA: _NUCLEIC_FRAME, _RNA: _NUCLEIC_FRAME},
    representative={
        TOKEN_INDEX[token]: name
        for token, name in (
            ("G", "CA"),
            *((f"{kind}_{base}", "C4") for kind in ("RNA", "DNA") for base in "AG"),
            *((f"RNA_{base}", "C2") for base in "CU"),
            *((f"DNA_{base}", "C2") for base in "CT"),
        )
    },
    representative_default={_PROTEIN: "CB", _DNA: "C1'", _RNA: "C1'"},
    exclude_water=True,
)
PROTENIX_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
PROTENIX_SYMMETRY = SymmetryPolicy(components="ligands", resonance=False, pin_unobserved=True)
#: The family's MSA depth constant (upstream ``max_msa_size``): the resource cap passed to ``pair_msas``.
PROTENIX_MSA_DEPTH = 16384

#: Upstream featurisation: a chain of at most this many residues gets a query-only MSA and no template.
_SHORT_CHAIN = 4
_TEMPLATE_SLOTS = 4
_EMBEDDING_SEED = 42
_GAP = TOKEN_INDEX["-"]
_OTHER = TOKEN_INDEX["OTHER"]
_LETTERS = "".join(map(chr, range(128)))
#: Upstream's MSA reading of a class: ``B``, ``Z`` and ``U`` as ``D``, ``E`` and ``C``; ``J`` and ``O`` as ``X``.
_MSA_CLASS = np.arange(len(TOKENS), dtype=np.int64)
for _letter, _reading in zip("BZUJO", "DECXX"):
    _MSA_CLASS[TOKEN_INDEX[_letter]] = TOKEN_INDEX[_reading]
#: Per monomer type, the MSA class of each ASCII letter (the decoding of ``ofoldx.msa`` read as upstream).
_DECODE = {monomer: _MSA_CLASS[encode_sequence(_LETTERS, monomer)] for monomer in (_PROTEIN, _RNA)}
_METALS = frozenset({3, 4, *range(11, 14), *range(19, 32), *range(37, 52), *range(55, 85), *range(87, 119)})
# ESM-2 alphabet: <cls> 0, <pad> 1, <eos> 2, residues from 4; every other protein class reads its X (24).
_ESM_CLS, _ESM_PAD, _ESM_EOS = 0, 1, 2
_ESM_TOKEN = np.full(len(TOKENS), 24, dtype=np.int64)
for _index, _letter in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO", start=4):
    _ESM_TOKEN[TOKEN_INDEX[_letter]] = _index
# Van der Waals radii by atomic number (RDKit's periodic table, as upstream).
_VDW_RADII = np.array(
    (0.0, 1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4)
    + (2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3)
    + (2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47)
    + (2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05)
    + (2.1, 2.05, 2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3)
    + (2.0,) * 26
)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class ProtenixProcessor(Processor):
    r"""
    Prepare one structure for Protenix.

    Args:
        config: Reference-position options. Defaults to ``ProtenixProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ProtenixProcessorConfig
    config: ProtenixProcessorConfig

    def __init__(
        self,
        config: ProtenixProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = ProtenixProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def prepare_input(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg: bool = False,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProtenixInput]:
        r"""
        Prepare one structure: layout, optional training crop, slots, gating, reference conformers, MSA rows and the
        Protenix tensors. The scope is every atom in a new frame.

        Args:
            structure: The target; its chemistry is consumed as read (no library).
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            tfg: Whether to prepare the static geometry training-free guidance needs.
            cropper: Optional training crop policy; needs ``generator``.
            generator: The numpy generator of the crop, the reference-position augmentation and the constraint draws.
        """

        return self._prepare(structure, msa=msa, tfg=tfg, cropper=cropper, generator=generator)

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[ProtenixInput],
    ) -> ProtenixStructureTarget:
        r"""Align coordinate truth to an existing prepared input.

        Args:
            target: Independent truth in the task's Cartesian frame.
            prepared: Input features and correspondence returned by `prepare_input`.

        Returns:
            The family's coordinate target. Measured coordinates retain Float64
            precision; input preparation, crop, and reference sampling are not repeated.
        """
        self._validate_prepared(prepared)
        truth = prepare_target(
            prepared,
            target,
            symmetry=PROTENIX_SYMMETRY,
        )
        return self._target_tensors(prepared, truth)

    def _target_tensors(
        self, prepared: PreparedInput[ProtenixInput], target: StructureTarget
    ) -> ProtenixStructureTarget:
        r"""The model's tensor target: canonical truth on the full atom axis and the stated resolution."""
        observed = torch.from_numpy(target.truth == TruthState.OBSERVED)
        stated = target.resolution is not None
        return ProtenixStructureTarget(  # type: ignore[call-arg]
            coordinates=project_model_coordinates(
                torch.from_numpy(target.coordinates.copy()), observed, feature="Protenix targets"
            )
            .to(self.device)
            .unsqueeze(0),
            coordinate_mask=observed.to(self.device).unsqueeze(0),
            resolution=torch.tensor([target.resolution if stated else -1.0], dtype=torch.float64, device=self.device),
            resolution_mask=torch.tensor([stated], device=self.device),
            batch_size=[1],
        )

    def _prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None,
        tfg: bool,
        cropper: Cropper | None,
        generator: np.random.Generator | None,
        templates: Sequence[Template] | None = None,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        noise: float = 0.0,
        structure_condition: Structure | None = None,
    ) -> PreparedInput[ProtenixInput]:
        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        constraints = _Constraints.resolve(system, atom_contacts, token_contacts, pocket)
        layout = token_layout(system, PROTENIX_LAYOUT)
        if len(constraints.atoms):
            layout = _per_atom(system, layout, _owner(system.component_atom_ptr)[constraints.atoms.ravel()])
        if cropper is not None:
            layout = crop_token_layout(
                structure, layout, cropper, required=constraints.required(system, layout), generator=generator
            )
        slots = slot_map(system, layout, PROTENIX_LAYOUT)
        accepted = gate(structure, slots, PROTENIX_GATING)
        references, sources, embedded = reference_conformers(
            structure, slots, embed=missing_reference(system), seed=_EMBEDDING_SEED
        )
        tokens, losses = model_tokens(system, layout)
        plan = _msa_plan(msa, system, layout, cropped=cropper is not None)
        torch_generator = None if generator is None else torch.Generator().manual_seed(int(generator.integers(2**62)))
        features = _features(
            system,
            slots,
            tokens,
            references,
            plan,
            augment=self.config.augment_reference_positions,
            generator=torch_generator,
            device=self.device,
            dtype=self.dtype,
        )
        features.update(constraints.channels(system, layout, substructure, noise, torch_generator, self.dtype))
        features["structure_rep_positions"], features["structure_rep_mask"] = _condition(
            system, token_representatives(system, layout, PROTENIX_LAYOUT), structure_condition, self.dtype
        )
        placed = _placed(features, device=self.device, dtype=self.dtype)
        placed["template_features"] = (
            None if templates is None else _template_features(templates, system, layout, self.device, self.dtype)
        )
        placed["tfg_features"] = _tfg_features(system, slots, self.device, self.dtype) if tfg else None
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        conformers[_owner(system.component_atom_ptr)[slots.atom]] = sources
        return PreparedInput(
            ProtenixInput(**placed, batch_size=[]),  # type: ignore[call-arg]
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    conformers,
                ),
                spec=self.preparation_spec,
            ),
        )


class ProtenixStructureConditionProcessor(ProtenixProcessor):
    r"""Prepare one structure with an explicit representative-atom coordinate condition (mini template v0.5)."""

    def prepare_input(
        self,
        structure: Structure,
        *,
        structure_condition: Structure | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg: bool = False,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProtenixInput]:
        r"""
        Prepare one structure as ``ProtenixProcessor.prepare_input``. ``structure_condition`` gives coordinates of the
        tokens' representative atoms (matched by instance, component ordinal and atom name).
        """

        return self._prepare(
            structure, msa=msa, tfg=tfg, cropper=cropper, generator=generator, structure_condition=structure_condition
        )


class ProtenixConstraintProcessor(ProtenixProcessor):
    r"""Prepare one structure with spatial constraints (base constraint v0.5)."""

    def prepare_input(
        self,
        structure: Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg: bool = False,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProtenixInput]:
        r"""
        Prepare one structure as ``ProtenixProcessor.prepare_input`` with the four constraint channels.

        Args:
            atom_contacts: Upper bounds between exact atoms; their residues are tokenized per atom before the crop.
            token_contacts: Upper bounds between token endpoints; a residue endpoint draws one of its tokens.
            pocket: Directed binder-to-residue condition.
            substructure: Coordinates of token centres, digitised into the four-channel distance condition.
            substructure_coordinate_noise_scale: Gaussian noise added to those coordinates, in angstroms.
        """

        return self._prepare(
            structure,
            msa=msa,
            tfg=tfg,
            cropper=cropper,
            generator=generator,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            noise=substructure_coordinate_noise_scale,
        )


class ProtenixTemplateProcessor(ProtenixProcessor):
    r"""Prepare one structure with protein templates (base v1 and v2)."""

    def prepare_input(
        self,
        structure: Structure,
        *,
        templates: Sequence[Template] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg: bool = False,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProtenixInput]:
        r"""
        Prepare one structure as ``ProtenixProcessor.prepare_input`` with four template slots: per protein chain of more
        than four residues, the first four templates aligned to it; targets are resolved by ``resolve_template``.
        """

        return self._prepare(
            structure, msa=msa, tfg=tfg, cropper=cropper, generator=generator, templates=tuple(templates)
        )


def _per_atom(system: MolecularSystem, layout: TokenLayout, components: NDArray[np.int64]) -> TokenLayout:
    r"""The layout with ``components`` tokenized per atom (the residues an atom contact names, as upstream)."""
    count = len(system.component_symbol)
    tokens = np.bincount(layout.token_component, minlength=count)
    residue = np.zeros(count, dtype=bool)
    residue[layout.token_component] = (tokens[layout.token_component] == 1) & (layout.token_class != _OTHER)
    residue[components] = False
    centre = np.full(count, -1, dtype=np.int64)
    centre[layout.token_component] = layout.token_centre_atom
    return tokenize(system, residue_token=residue, centre_atom=centre, atom_mask=layout.atom_token >= 0)


def atom_tokens(layout: TokenLayout) -> NDArray[np.bool_]:
    r"""Per token, whether it is one atom of a per-atom tokenized component."""
    counts = np.bincount(layout.token_component)
    return (counts[layout.token_component] > 1) | (layout.token_class == _OTHER)


def _msa_chains(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""Per instance, whether its MSA is read: protein and RNA chains of more than four residues (upstream)."""
    return np.isin(instance_monomer(system), (_PROTEIN, _RNA)) & (np.diff(system.instance_component_ptr) > _SHORT_CHAIN)


def _msa_plan(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout, *, cropped: bool
) -> MsaCollection | None:
    r"""Pair eligible pools or use an explicit plan unchanged."""
    if msa is None:
        return None
    if not isinstance(msa, MsaCollection):
        names = system.symbols()["instance"]
        instances = np.unique(_owner(system.instance_component_ptr)[layout.token_component])
        read = {names[int(system.instance_symbol[i])] for i in instances[_msa_chains(system)[instances]]}
        pools = {name: msa[name] for name in msa if name in read}
        return pair_msas(pools, system, layout, max_rows=PROTENIX_MSA_DEPTH)
    if cropped:
        fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
    if msa.row_pairing.shape[0] > PROTENIX_MSA_DEPTH:
        fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=PROTENIX_MSA_DEPTH)
    return msa


def _msa_features(
    plan: MsaCollection | None, system: MolecularSystem, layout: TokenLayout
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    MSA rows, deletion counts, profile and deletion mean on the token axis (upstream featurisation of a given plan).

    Only protein and RNA chains of more than four residues take MSA rows; every other chain is its query with gaps
    below. Classes are read as upstream's MSA alphabet (``_MSA_CLASS``). The profile and deletion mean of a chain with
    a pool are over the pool's unique unpaired rows (its query included); every other token's profile is its own
    class. A non-protein chain's rows that are gaps over all its tokens are its query row.
    """

    pointer = system.instance_component_ptr
    token_instance = _owner(pointer)[layout.token_component]
    position = layout.token_component - pointer[token_instance]
    monomer = instance_monomer(system)
    used = _msa_chains(system)
    num_tokens = len(token_instance)
    if plan is None:
        token, deletions = layout.token_class[None].astype(np.int64), np.zeros((1, num_tokens), dtype=np.int64)
    else:
        rows = msa_rows(plan, system, layout)
        token, deletions = rows.token.astype(np.int64), rows.deletions.astype(np.int64)
    token = _MSA_CLASS[token]
    token[1:, ~used[token_instance]] = _GAP
    deletions[:, ~used[token_instance]] = 0
    profile = np.eye(len(TOKENS))[_MSA_CLASS[layout.token_class]]
    deletion_mean = np.zeros(num_tokens)
    entity = system.text("entity", system.entity_symbol)[system.instance_entity]
    target_instance_ids = None if plan is None else plan.target_instance_ids
    for column, pool in enumerate(() if plan is None else plan.msas):
        if target_instance_ids is None:
            instances = np.flatnonzero(entity == pool.entity_id)
        else:
            instances = system.instance_index([target_instance_ids[column]])
        unique = unique_rows(np.array(pool.unpaired_sequences, dtype="S"))
        for instance in instances[used[instances]].tolist():
            letters = np.frombuffer("".join(pool.unpaired_sequences[r] for r in unique).encode(), dtype=np.uint8)
            classes = _DECODE[MonomerType(int(monomer[instance]))][letters.reshape(len(unique), -1)]
            width = classes.shape[1]
            counts = np.bincount((np.arange(width) * len(TOKENS) + classes).ravel(), minlength=width * len(TOKENS))
            covered = np.flatnonzero(token_instance == instance)
            profile[covered] = (counts.reshape(width, len(TOKENS)) / len(unique))[position[covered]]
            deletion_mean[covered] = pool.unpaired_deletion_counts[unique].mean(axis=0)[position[covered]]
    for instance in np.unique(token_instance[monomer[token_instance] != _PROTEIN]).tolist():
        columns = np.flatnonzero(token_instance == instance)
        gapped = (token[:, columns] == _GAP).all(axis=1)
        token[np.ix_(gapped, columns)] = token[0, columns]
    return token, deletions, profile, deletion_mean


def _esm_features(system: MolecularSystem, layout: TokenLayout) -> dict[str, Tensor]:
    r"""ESM-2 rows of the distinct protein chain sequences and each token's residue in them."""
    pointer = system.instance_component_ptr
    classes = token_classes(system).token
    rows: dict[tuple[int, ...], int] = {}
    instance_row = np.full(len(pointer) - 1, -1, dtype=np.int64)
    for instance in np.flatnonzero(instance_monomer(system) == _PROTEIN).tolist():
        sequence = tuple(_ESM_TOKEN[classes[pointer[instance] : pointer[instance + 1]]].tolist())
        instance_row[instance] = rows.setdefault(sequence, len(rows))
    length = max((len(sequence) + 2 for sequence in rows), default=0)
    tokens = np.full((len(rows), length), _ESM_PAD, dtype=np.int64)
    mask = np.zeros((len(rows), length), dtype=bool)
    for sequence, row in rows.items():
        tokens[row, : len(sequence) + 2] = (_ESM_CLS, *sequence, _ESM_EOS)
        mask[row, : len(sequence) + 2] = True
    token_instance = _owner(pointer)[layout.token_component]
    row = instance_row[token_instance]
    residue = row * length + layout.token_component - pointer[token_instance] + 1
    return {
        "esm_tokens": torch.from_numpy(tokens),
        "esm_mask": torch.from_numpy(mask),
        "token_to_esm_residue": torch.from_numpy(np.where(row >= 0, residue, 0)),
        "token_has_esm": torch.from_numpy(row >= 0),
    }


def _token_bonds(system: MolecularSystem, layout: TokenLayout) -> Tensor:
    r"""
    ``[N, N]``: 1 between two tokens a KNOWN covalent edge joins, except an edge between two polymer components that
    are neighbours in one chain or inside one standard polymer residue (upstream inference keeps every other
    polymer–polymer bond, e.g. a disulfide, with polymer–ligand, ligand–ligand and the internal bonds of a non-standard
    residue tokenized per atom).
    """

    s = system.arrays()
    first, second = s["edge_index"].astype(np.int64)
    component = _owner(s["component_atom_ptr"])
    instance = _owner(s["instance_component_ptr"])
    entity = s["instance_entity"][instance][component]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    a_component, b_component = component[first], component[second]
    adjacent = (instance[a_component] == instance[b_component]) & (np.abs(a_component - b_component) == 1)
    standard = token_classes(system).encoding[a_component] == Encoding.EXACT
    internal = (a_component == b_component) & standard
    junction = polymer[first] & polymer[second] & (adjacent | internal)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = covalent & ~junction & (a >= 0) & (b >= 0) & (a != b)
    bonds = torch.zeros(len(layout.token_component), len(layout.token_component))
    bonds[torch.from_numpy(a[linked]), torch.from_numpy(b[linked])] = 1
    bonds[torch.from_numpy(b[linked]), torch.from_numpy(a[linked])] = 1
    return bonds


def _reference_positions(
    references: NDArray[np.float64], space: NDArray[np.int64], *, augment: bool, generator: torch.Generator | None
) -> Tensor:
    r"""Each reference space centred on its atoms, then (with ``augment``) translated by U(-1, 1)³ and rotated by a
    uniform rotation, as upstream's ``random_transform``."""
    spaces, group = np.unique(space, return_inverse=True)
    centres = np.zeros((len(spaces), 3))
    np.add.at(centres, group, references)
    positions = torch.from_numpy(references - (centres / np.bincount(group)[:, None])[group])
    if augment and len(spaces):
        translation = torch.empty(len(spaces), 3, dtype=positions.dtype).uniform_(-1.0, 1.0, generator=generator)
        rotation = uniform_random_rotation(len(spaces), dtype=positions.dtype, generator=generator)
        index = torch.from_numpy(group)
        positions = torch.einsum("aj,akj->ak", positions + translation[index], rotation[index])
    return positions


def _frames(
    system: MolecularSystem, slots: SlotMap, positions: Tensor, space: NDArray[np.int64]
) -> tuple[Tensor, Tensor]:
    r"""
    Per token, its three frame slots and whether the frame is usable. A residue token uses its named frame atoms
    (all present, sine ≥ 1e-8); an atom token its two nearest atoms of the same reference space around itself, as
    upstream (sine of the angle at least that of a 25° deviation).
    """

    layout = slots.layout
    slot_of = np.full(len(system.atom_symbol), -1, dtype=np.int64)
    slot_of[slots.atom] = np.arange(len(slots.atom))
    frames = slot_of[token_frames(system, layout, PROTENIX_LAYOUT)]
    named = (frames[:, 0] != frames[:, 1]) & (frames[:, 1] != frames[:, 2]) & (frames[:, 0] != frames[:, 2])
    atomic = atom_tokens(layout)
    frames[~atomic & ~named] = -1
    coordinates = positions.numpy()
    for token in np.flatnonzero(atomic).tolist():
        centre = slot_of[layout.token_centre_atom[token]]
        neighbours = np.flatnonzero(space == space[centre])
        neighbours = neighbours[neighbours != centre]
        if len(neighbours) < 2:
            frames[token] = -1
            continue
        distance = ((coordinates[neighbours] - coordinates[centre]) ** 2).sum(axis=-1)
        nearest = neighbours[np.argsort(distance, kind="stable")[:2]]
        frames[token] = (nearest[0], centre, nearest[1])
    selected = torch.from_numpy(np.maximum(frames, 0))
    first = positions[selected[:, 0]] - positions[selected[:, 1]]
    second = positions[selected[:, 2]] - positions[selected[:, 1]]
    length = torch.linalg.vector_norm(first, dim=-1) * torch.linalg.vector_norm(second, dim=-1)
    sine = torch.from_numpy(np.where(atomic, (1 - (1 - 0.0937) ** 2) ** 0.5, 1e-8))
    usable = (length > 1e-16) & (torch.linalg.vector_norm(torch.linalg.cross(first, second), dim=-1) >= sine * length)
    return torch.from_numpy(frames), usable & torch.from_numpy((frames >= 0).all(axis=1))


def _features(
    system: MolecularSystem,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    references: NDArray[np.float64],
    plan: MsaCollection | None,
    *,
    augment: bool,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    r"""The Protenix token, atom, MSA and ESM tensors of one prepared task (upstream featurisation on the slot axis)."""
    s = system.arrays()
    layout = slots.layout
    atom = slots.atom
    num_atoms, num_tokens = len(atom), len(layout.token_component)
    atom_component = _owner(s["component_atom_ptr"])
    component_instance = _owner(s["instance_component_ptr"])
    token_instance = component_instance[layout.token_component]
    entity = s["instance_entity"]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[atom] = np.arange(num_atoms)

    space = atom_component[atom]
    positions = _reference_positions(references, space, augment=augment, generator=generator)
    frame_atom_index, frame_mask = _frames(system, slots, positions, space)
    names = system.text("atom", s["atom_symbol"][atom]).tolist()
    characters = torch.tensor(
        [[min(63, max(0, ord(c) - 32)) for c in name.ljust(4)[:4]] for name in names], dtype=torch.long
    ).reshape(num_atoms, 4)
    atomic_number = torch.from_numpy(s["atomic_number"][atom].astype(np.int64))
    reference_mask = torch.ones(num_atoms, dtype=torch.bool)
    # Protenix reference-atom features: mask (1), element (128), name characters (4 x 64).
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            torch.nn.functional.one_hot(atomic_number - 1, 128).float(),
            torch.nn.functional.one_hot(characters, 64).reshape(num_atoms, 256).float(),
        ),
        dim=-1,
    )
    msa_tokens, deletions, profile, deletion_mean = _msa_features(plan, system, layout)
    token_features = torch.cat(
        (
            torch.zeros(num_tokens, 384),
            torch.nn.functional.one_hot(torch.from_numpy(tokens), len(TOKENS)).float(),
            torch.from_numpy(profile).float(),
            torch.from_numpy(deletion_mean).float()[:, None],
        ),
        dim=-1,
    )
    return {
        "token_mask": torch.ones(num_tokens, dtype=torch.bool),
        "residue_index": torch.from_numpy(layout.token_component - s["instance_component_ptr"][token_instance]),
        "chain_index": torch.from_numpy(token_instance),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64)),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "token_index": torch.arange(num_tokens),
        "atom_features": atom_features,
        "atom_to_token": torch.from_numpy(slots.token),
        "atom_mask": torch.ones(num_atoms, dtype=torch.bool),
        "reference_positions": project_model_coordinates(
            positions, reference_mask, dtype=dtype, feature="Protenix prepared references"
        ),
        "reference_space_id": torch.from_numpy(space),
        "atom_token_index": torch.from_numpy(np.arange(num_atoms) - layout.token_atom_ptr[slots.token]),
        "token_representative_atom_index": torch.from_numpy(
            slot_of[token_representatives(system, layout, PROTENIX_LAYOUT)]
        ),
        "frame_atom_index": frame_atom_index,
        "frame_mask": frame_mask,
        "token_features": token_features,
        "reference_mask": reference_mask,
        "reference_charges": torch.from_numpy(s["formal_charge"][atom].astype(np.float64))[:, None],
        "token_bonds": _token_bonds(system, layout),
        "msa_tokens": torch.from_numpy(msa_tokens),
        "msa_deletion_counts": torch.from_numpy(deletions),
        **_esm_features(system, layout),
        "atom_is_polymer": torch.from_numpy(polymer[component_instance[space]]),
        "atom_index": torch.from_numpy(atom.copy()),
    }


def _placed(features: dict[str, Tensor], *, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor | None]:
    r"""Every tensor on ``device`` with a leading batch axis; floating features projected to ``dtype``."""
    placed: dict[str, Tensor | None] = {}
    for name, value in features.items():
        if value.is_floating_point() and name not in ("reference_positions", "structure_rep_positions"):
            value = project_model_features(
                value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"Protenix prepared {name}"
            )
        placed[name] = value.to(device=device).unsqueeze(0)
    return placed


def _matched(
    system: MolecularSystem, atoms: NDArray[np.int64], other: Structure
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    r"""The coordinates of ``atoms`` in ``other``, matched by instance, component ordinal and atom name."""
    found = match_atoms(system, atoms, other)
    present = found >= 0
    present[present] = other.coordinate_mask[found[present]]
    coordinates = np.zeros((len(atoms), 3))
    coordinates[present] = other.coordinates[found[present]]
    return coordinates, present


def _condition(
    system: MolecularSystem, representatives: NDArray[np.int64], condition: Structure | None, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    r"""The representative-atom coordinates of each token in the structure condition (none without one)."""
    if condition is None:
        coordinates, present = np.zeros((len(representatives), 3)), np.zeros(len(representatives), dtype=bool)
    else:
        coordinates, present = _matched(system, representatives, condition)
    positions = project_model_coordinates(
        torch.from_numpy(coordinates), torch.from_numpy(present), dtype=dtype, feature="Protenix structure conditioning"
    )
    return positions, torch.from_numpy(present)


class _Constraints:
    r"""The resolved spatial constraints of one request: atoms, endpoints (atom or component) and the pocket."""

    def __init__(self, atom_contacts, atoms, token_contacts, endpoints, pocket, binder, residues) -> None:
        self.atom_contacts: Sequence[ProtenixAtomContact] = atom_contacts
        self.atoms: NDArray[np.int64] = atoms  # [k, 2]
        self.token_contacts: Sequence[ProtenixTokenContact] = token_contacts
        self.endpoints: NDArray[np.int64] = endpoints  # [k, 2, 2]: (atom, component), one of them -1
        self.pocket: ProtenixPocket | None = pocket
        self.binder: int = binder
        self.residues: NDArray[np.int64] = residues

    @classmethod
    def resolve(
        cls,
        system: MolecularSystem,
        atom_contacts: Sequence[ProtenixAtomContact],
        token_contacts: Sequence[ProtenixTokenContact],
        pocket: ProtenixPocket | None,
    ) -> _Constraints:
        r"""Every named atom, component and instance in ``system``; naming nothing is ``request_invalid_positions``."""

        def atom(site: AtomSite, argument: str) -> int:
            index = int(system.atom_index([site])[0])
            if index < 0:
                fail("request_invalid_positions", RequestRef(argument, str(site)))
            return index

        def component(ref: ComponentRef, argument: str) -> int:
            index = int(system.component_index([ref])[0])
            if index < 0:
                fail("request_invalid_positions", RequestRef(argument, str(ref)))
            return index

        def endpoint(value: AtomSite | ComponentRef) -> tuple[int, int]:
            if isinstance(value, AtomSite):
                return atom(value, "token_contacts"), -1
            return -1, component(value, "token_contacts")

        atoms = [(atom(c.atom_a, "atom_contacts"), atom(c.atom_b, "atom_contacts")) for c in atom_contacts]
        endpoints = [(endpoint(c.endpoint_a), endpoint(c.endpoint_b)) for c in token_contacts]
        binder, residues = -1, []
        if pocket is not None:
            binder = int(system.instance_index([pocket.binder])[0])
            if binder < 0:
                fail("request_invalid_positions", RequestRef("pocket.binder", pocket.binder))
            residues = [component(ref, "pocket.contact_residues") for ref in pocket.contact_residues]
        return cls(
            tuple(atom_contacts),
            np.array(atoms, dtype=np.int64).reshape(-1, 2),
            tuple(token_contacts),
            np.array(endpoints, dtype=np.int64).reshape(-1, 2, 2),
            pocket,
            binder,
            np.array(residues, dtype=np.int64),
        )

    def required(self, system: MolecularSystem, layout: TokenLayout) -> NDArray[np.bool_]:
        r"""The tokens a crop must keep: every token that can realise a constraint."""
        required = np.zeros(len(layout.token_component), dtype=bool)
        atoms = np.concatenate((self.atoms.ravel(), self.endpoints[..., 0].ravel()))
        tokens = layout.atom_token[atoms[atoms >= 0]]
        required[tokens[tokens >= 0]] = True
        components = np.concatenate((self.endpoints[..., 1].ravel(), self.residues))
        required |= np.isin(layout.token_component, components[components >= 0])
        required |= _owner(system.instance_component_ptr)[layout.token_component] == self.binder
        return required

    def channels(
        self,
        system: MolecularSystem,
        layout: TokenLayout,
        substructure: Structure | None,
        noise: float,
        generator: torch.Generator | None,
        dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        r"""
        The four constraint pair channels: an atom contact writes ``[max, 0]`` and a token contact ``[0, max]``
        (symmetric), the pocket a directed binder → residue ``max``, and the substructure the digitised distances of
        the resolved token centres. A residue endpoint with several tokens draws one from ``generator``.
        """

        num_tokens = len(layout.token_component)
        token_instance = _owner(system.instance_component_ptr)[layout.token_component]
        entity = system.instance_entity[token_instance]
        polymer = known(system.entity_type_state[entity]) & (system.entity_type[entity] == EntityType.POLYMER)
        pocket = torch.zeros(num_tokens, num_tokens, 1, dtype=dtype)
        token_contact = torch.zeros(num_tokens, num_tokens, 2, dtype=dtype)
        atom_contact = torch.zeros(num_tokens, num_tokens, 2, dtype=dtype)
        distances = torch.zeros(num_tokens, num_tokens, 4, dtype=dtype)

        def atom_token(atom: int, argument: str) -> int:
            token = int(layout.atom_token[atom])
            if token < 0:
                fail(
                    "request_invalid_positions", StructureRef(atom=atom), argument=argument, reason="atom_not_tokenized"
                )
            return token

        def residue_token(component: int, argument: str) -> int:
            candidates = np.flatnonzero((layout.token_component == component) & polymer)
            if not len(candidates):
                fail("request_invalid_positions", RequestRef(argument, str(system.component_refs([component])[0])))
            return int(candidates[_draw(len(candidates), generator)])

        def endpoint_token(atom: int, component: int) -> int:
            return atom_token(atom, "token_contacts") if atom >= 0 else residue_token(component, "token_contacts")

        for (first, second), atom_constraint in zip(self.atoms.tolist(), self.atom_contacts):
            _write(
                atom_contact,
                atom_token(first, "atom_contacts"),
                atom_token(second, "atom_contacts"),
                atom_constraint,
                0,
            )
        for (first, second), token_constraint in zip(self.endpoints.tolist(), self.token_contacts):
            _write(token_contact, endpoint_token(*first), endpoint_token(*second), token_constraint, 1)
        if self.pocket is not None:
            binder = np.flatnonzero(token_instance == self.binder)
            if not len(binder):
                fail("request_invalid_positions", RequestRef("pocket.binder", self.pocket.binder))
            for component in self.residues.tolist():
                pocket[torch.from_numpy(binder), residue_token(component, "pocket.contact_residues"), 0] = (
                    self.pocket.max_distance
                )
        if substructure is not None:
            coordinates, present = _matched(system, layout.token_centre_atom, substructure)
            chosen = torch.from_numpy(np.flatnonzero(present))
            if len(chosen):
                points = project_model_coordinates(
                    torch.from_numpy(coordinates[present]),
                    torch.ones(len(chosen), dtype=torch.bool),
                    dtype=dtype,
                    feature="Protenix substructure coordinates",
                )
                if noise != 0.0:
                    points = points + noise * torch.randn(points.shape, dtype=points.dtype, generator=generator)
                distance = pairwise_distance(points, points)
                bins = torch.bucketize(distance, distance.new_tensor((0.0, 4.0, 8.0, 16.0, float("inf"))), right=True)
                bins = bins - 1
                row, column = chosen[:, None].expand_as(bins), chosen[None, :].expand_as(bins)
                inside = bins > 0
                distances[row[inside], column[inside], bins[inside]] = 1
        return {
            "pocket": pocket,
            "token_contact": token_contact,
            "atom_contact": atom_contact,
            "substructure": distances,
        }


def _draw(count: int, generator: torch.Generator | None) -> int:
    return 0 if count == 1 else int(torch.randint(0, count, (1,), generator=generator).item())


def _write(feature: Tensor, first: int, second: int, contact: ProtenixAtomContact | ProtenixTokenContact, channel: int):
    values = torch.zeros(2, dtype=feature.dtype)
    values[channel] = contact.max_distance
    previous = feature[first, second]
    if bool((previous != 0).any()) and not torch.equal(previous, values):
        fail("value_invalid", RequestRef("contacts", str(contact)), invariant="one_constraint_per_token_pair")
    feature[first, second] = feature[second, first] = values


def _template_features(
    templates: Sequence[Template],
    system: MolecularSystem,
    layout: TokenLayout,
    device: torch.device,
    dtype: torch.dtype,
) -> ProtenixTemplateFeatures:
    r"""
    The four template slots. Per protein chain of more than four residues, slot ``k`` holds the ``k``-th template
    aligned to it (its unaligned tokens gap); slot 0 of every other chain is gap, the remaining slots ``A`` (upstream's
    zero padding class), as upstream's empty features padded to four. Pair masks keep token pairs placed in one frame
    of one template.
    """

    pointer = system.instance_component_ptr
    token_instance = _owner(pointer)[layout.token_component]
    position = layout.token_component - pointer[token_instance]
    chosen: dict[int, list] = {}
    for index, template in enumerate(templates):
        resolved = resolve_template(template, system, monomer_types=(_PROTEIN,))
        placed: dict[int, list] = {}
        for alignment, frame in zip(resolved, template_alignment_frames(resolved)):
            chain = template.coordinates.chains[alignment.template_instance_id]
            placed.setdefault(alignment.target_instance_index, []).append((chain, alignment.residue_pairs, frame))
        for instance, alignments in placed.items():
            slots = chosen.setdefault(instance, [])
            if pointer[instance + 1] - pointer[instance] > _SHORT_CHAIN and len(slots) < _TEMPLATE_SLOTS:
                slots.append((index, alignments))
    num_tokens = len(token_instance)
    positions = np.zeros((_TEMPLATE_SLOTS, num_tokens, 4, 3))
    present = np.zeros((_TEMPLATE_SLOTS, num_tokens, 4), dtype=bool)
    residue = np.full((_TEMPLATE_SLOTS, num_tokens), TOKEN_INDEX["A"], dtype=np.int64)
    residue[0] = _GAP
    frame_of = np.full((_TEMPLATE_SLOTS, num_tokens), -1, dtype=np.int64)
    packed: dict[tuple[int, int], int] = {}
    for instance, slots in chosen.items():
        tokens = np.flatnonzero(token_instance == instance)
        for slot, (index, alignments) in enumerate(slots):
            residue[slot, tokens] = _GAP
            for chain, pairs, frame in alignments:
                packed_frame = packed.setdefault((index, frame), len(packed))
                for target, source in pairs:
                    hit = tokens[position[tokens] == target]
                    positions[slot, hit], present[slot, hit] = template_atoms(chain, source, ("N", "CA", "C", "CB"))
                    residue[slot, hit] = chain.token_class[source]
                    frame_of[slot, hit] = packed_frame
    pseudo_beta, pseudo_beta_mask, unit_vector, backbone_mask = _template_geometry(
        torch.from_numpy(positions), torch.from_numpy(present).double(), torch.from_numpy(residue)
    )
    frame = torch.from_numpy(frame_of)
    same = ((frame[:, :, None] == frame[:, None, :]) & (frame[:, :, None] >= 0)).double()
    pseudo_beta_2d = pseudo_beta_mask[:, :, None] * pseudo_beta_mask[:, None, :] * same
    backbone_2d = backbone_mask[:, :, None] * backbone_mask[:, None, :] * same
    distogram = coordinate_distogram(pseudo_beta, min_distance=3.25, max_distance=50.75, num_bins=39)
    return ProtenixTemplateFeatures(  # type: ignore[call-arg]
        distogram=(distogram * pseudo_beta_2d[..., None]).to(device=device, dtype=dtype).unsqueeze(0),
        pseudo_beta_mask=pseudo_beta_2d.to(device=device, dtype=dtype).unsqueeze(0),
        residue_type_ids=torch.from_numpy(residue).to(device).unsqueeze(0),
        unit_vector=(unit_vector * backbone_2d[..., None]).to(device=device, dtype=dtype).unsqueeze(0),
        backbone_frame_mask=backbone_2d.to(device=device, dtype=dtype).unsqueeze(0),
        batch_size=[1],
    )


def _template_geometry(positions: Tensor, mask: Tensor, residue: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""
    Upstream's template geometry from N, CA, C, CB: the pseudo-β (CA for glycine) with its mask, the unit vector of
    every CA in each residue's backbone frame (``e1 = (C − CA)/(|C − CA| + ε)``, ``e2`` the normalised N − CA
    component orthogonal to it, ``ε = 1e-6`` also in the vector norm) and the frame mask by atom presence.
    """

    nitrogen, alpha, carbon, beta = positions.unbind(dim=-2)
    nitrogen_mask, alpha_mask, carbon_mask, beta_mask = mask.unbind(dim=-1)
    glycine = residue == TOKEN_INDEX["G"]
    pseudo_beta = torch.where(glycine[..., None], alpha, beta)
    pseudo_beta_mask = torch.where(glycine, alpha_mask, beta_mask)
    first, second = carbon - alpha, nitrogen - alpha
    e1 = first / (torch.linalg.vector_norm(first, dim=-1, keepdim=True) + 1e-6)
    e2 = second - (second * e1).sum(dim=-1, keepdim=True) * e1
    e2 = e2 / (torch.linalg.vector_norm(e2, dim=-1, keepdim=True) + 1e-6)
    rotation = torch.stack((e1, e2, torch.linalg.cross(e1, e2)), dim=-1)
    local = torch.einsum("...ilk,...ijl->...ijk", rotation, alpha[..., None, :, :] - alpha[..., :, None, :])
    unit_vector = local / (torch.linalg.vector_norm(local, dim=-1, keepdim=True) + 1e-6)
    return pseudo_beta, pseudo_beta_mask, unit_vector, nitrogen_mask * alpha_mask * carbon_mask


def _tfg_scopes(system: MolecularSystem, slots: SlotMap) -> list[NDArray[np.int64]]:
    r"""
    The components training-free guidance takes geometry for: each component holding a slot that is not a standard
    polymer residue, holds no metal atom (upstream's two exclusions) and projects to RDKit on its own — every claim
    True, connections complete, no edge or external connection leaving it.
    """

    s = system.arrays()
    count = len(s["component_symbol"])
    atom_component = _owner(s["component_atom_ptr"])
    component_instance = _owner(s["instance_component_ptr"])
    entity = s["instance_entity"][component_instance]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    standard = polymer & (token_classes(system).encoding == Encoding.EXACT)
    metal = np.zeros(count, dtype=bool)
    metal[atom_component[np.isin(s["atomic_number"], list(_METALS))]] = True
    complete = s["component_topology_complete"] & s["component_heavy_atoms_complete"]
    complete &= system.component_hydrogens_complete & s["instance_connections_complete"][component_instance]
    first, second = s["edge_index"].astype(np.int64)
    crossing = atom_component[first] != atom_component[second]
    leaving = np.zeros(count, dtype=bool)
    leaving[atom_component[first[crossing]]] = leaving[atom_component[second[crossing]]] = True
    leaving[atom_component[s["external_atom"].astype(np.int64)]] = True
    used = np.zeros(count, dtype=bool)
    used[atom_component[slots.atom]] = True
    chosen = np.flatnonzero(used & complete & ~standard & ~metal & ~leaving)
    return [np.flatnonzero(atom_component == c) for c in chosen.tolist()]


def _tfg_features(
    system: MolecularSystem, slots: SlotMap, device: torch.device, dtype: torch.dtype
) -> ProtenixTfgFeatures:
    r"""The static guidance geometry on the slot axis: per-component constraint features (rows naming an atom without a
    slot dropped), KNOWN covalent edges between chains, atom chains and van der Waals radii."""
    s = system.arrays()
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[slots.atom] = np.arange(len(slots.atom))
    parts: list[ConstraintFeatures] = [
        constraint_features(system, atoms, default_bounds=True) for atoms in _tfg_scopes(system, slots)
    ]

    def joined(name: str, width: int) -> NDArray:
        values = [getattr(part, name) for part in parts]
        return np.concatenate(values) if values else np.zeros((0, width) if width else 0)

    def rows(index: NDArray, *values: NDArray) -> list[Tensor]:
        mapped = slot_of[index.astype(np.int64)]
        keep = (mapped >= 0).all(axis=1)
        return [torch.from_numpy(mapped[keep].T.copy()).to(device)] + [
            torch.from_numpy(value[keep]).to(device=device, dtype=dtype if value.dtype.kind == "f" else None)
            for value in values
        ]

    pairs, lower, upper, bond, angle = rows(
        joined("pairs", 2), joined("lower", 0), joined("upper", 0), joined("bond_pair", 0), joined("angle_pair", 0)
    )
    torsions, force_constant, sign = rows(
        joined("torsions", 4), joined("torsion_force_constant", 6), joined("torsion_sign", 6).astype(np.float64)
    )
    chiral, chiral_orientation = rows(joined("chiral", 4), np.where(joined("chiral_positive", 0), 1.0, -1.0))
    stereo, stereo_orientation = rows(joined("stereo_bond", 4), joined("stereo_trans", 0).astype(np.float64))
    (linear,) = rows(joined("linear", 3))
    (planar,) = rows(joined("planar_improper", 4))
    component_instance = _owner(s["instance_component_ptr"])
    atom_instance = component_instance[_owner(s["component_atom_ptr"])]
    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    interchain = covalent & (atom_instance[first] != atom_instance[second])
    (interchain_bonds,) = rows(np.stack((first[interchain], second[interchain]), axis=1))
    return ProtenixTfgFeatures(  # type: ignore[call-arg]
        atom_chain=torch.from_numpy(atom_instance[slots.atom]).to(device),
        atom_vdw_radius=torch.from_numpy(_VDW_RADII[s["atomic_number"][slots.atom]]).to(device=device, dtype=dtype),
        interchain_bond_index=interchain_bonds,
        pairwise_distance_index=pairs,
        pairwise_distance_lower_bound=lower,
        pairwise_distance_upper_bound=upper,
        pairwise_distance_is_bond=bond,
        pairwise_distance_is_angle=angle,
        experimental_torsion_index=torsions,
        experimental_torsion_force_constant=force_constant,
        experimental_torsion_sign=sign,
        linear_bond_index=linear,
        chiral_index=chiral,
        chiral_orientation=chiral_orientation,
        stereo_bond_index=stereo,
        stereo_bond_orientation=stereo_orientation,
        planar_improper_index=planar,
        batch_size=[],
    )
