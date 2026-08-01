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
Boltz-2 input processing for folding and affinity scoring, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own (layout, gating, crop, MSA pairing, template resolution,
reference conformers, constraint features); this module keeps only the Boltz-2 featurisation of those inputs, which
matches upstream for identical inputs. The affinity ligand's SMILES standardisation and atom naming are the
family's own rule (upstream ``parse_boltz_schema``), computed through ``chemistry.smiles``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.definitions import ChemicalComponent, stated_component
from ...chemistry.rdkit import ConstraintFeatures, constraint_features, molecular_weight
from ...chemistry.smiles import canonical_atom_ranks, read_smiles, standardize_smiles
from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ELEMENTS, AtomSite, BondKind, ComponentRef, EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import (
    CLASS_MONOMER,
    TOKEN_INDEX,
    TOKENS,
    Encoding,
    TokenLayout,
    select_tokens,
    token_classes,
)
from ...errors import RequestRef, StructureRef, fail
from ...msa import MSA, MsaCollection, MsaRows, msa_rows, pair_msas
from ...processor import PreparationSpec, Processor
from ...provenance import Records, known
from ...templates import ChainCoordinates, Template, resolve_template, template_alignment_frames
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
from ..processing import augment_reference_positions, project_model_coordinates, project_model_features
from ..targets import StructureTarget, SymmetryPolicy, TruthState, prepare_target
from ..tensors import to_tensors
from .config import Boltz2ProcessorConfig
from .model import Boltz2Contact, Boltz2Input, Boltz2Pocket, Boltz2StructureTarget, Boltz2Template

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA
_NUCLEIC_FRAME = ("C1'", "C3'", "C4'")

BOLTZ2_LAYOUT = LayoutTable(
    family="boltz2",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    # Boltz-2 tokenizes a modified polymer residue at residue level; only class-less components are per atom.
    per_atom_components="none",
    exclude_designated_leaving=True,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={_PROTEIN: ("N", "CA", "C"), _DNA: _NUCLEIC_FRAME, _RNA: _NUCLEIC_FRAME},
    representative={
        TOKEN_INDEX["G"]: "CA",
        **{TOKEN_INDEX[f"{p}_{b}"]: "C4" for p in ("RNA", "DNA") for b in "AG"},
        **{TOKEN_INDEX[token]: "C2" for token in ("RNA_C", "RNA_U", "DNA_C", "DNA_T")},
    },
    representative_default={_PROTEIN: "CB", _DNA: "C1'", _RNA: "C1'"},
)
BOLTZ2_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
BOLTZ2_SYMMETRY = SymmetryPolicy(components="all", resonance=True, pin_unobserved=False)
#: The family's MSA depth constant (upstream ``max_msa_seqs``): the resource cap passed to ``pair_msas``.
BOLTZ2_MSA_DEPTH = 8192

_NONPOLYMER = 3
# Contact-conditioning classes: UNSPECIFIED, UNSELECTED, POCKET>BINDER, BINDER>POCKET, CONTACT.
_UNSPECIFIED, _UNSELECTED, _POCKET_TO_BINDER, _BINDER_TO_POCKET, _CONTACT = range(5)
_METHOD_TYPES = {
    "md": 0,
    "x-ray diffraction": 1,
    "electron microscopy": 2,
    "solution nmr": 3,
    "solid-state nmr": 4,
    "neutron diffraction": 4,
    "electron crystallography": 4,
    "fiber diffraction": 4,
    "powder diffraction": 4,
    "infrared spectroscopy": 4,
    "fluorescence transfer": 4,
    "epr": 4,
    "theoretical model": 4,
    "solution scattering": 4,
    "other": 4,
    "afdb": 5,
    "boltz-1": 6,
    "future1": 7,
    "future2": 8,
    "future3": 9,
    "future4": 10,
    "future5": 11,
}
# The upstream affinity crop (AffinityCropper and the affinity token and atom budgets) and ligand size limit.
_AFFINITY_TOKENS, _AFFINITY_POLYMER_TOKENS, _AFFINITY_ATOMS, _NEIGHBOURHOOD = 256, 200, 2048, 10
_AFFINITY_HEAVY_ATOMS = 128
# The fields stated_component derives itself.
_DERIVED_FIELDS = frozenset({"scope", "digest"})


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def standardize_boltz2_affinity_smiles(smiles: str) -> ChemicalComponent:
    r"""
    The Boltz-2 affinity ligand definition of a SMILES string (upstream ``parse_boltz_schema``): the ChEMBL-standardised
    largest fragment, read by ``read_smiles`` as component ``LIG1`` (upstream's name of the first SMILES ligand, absent
    from the CCD: a library entry of the same id would conflict with it), its atoms named by element and RDKit canonical
    rank (``C3``, ``N1``). Pass it to ``build_system(components=...)``.

    Raises:
        UnsupportedError: ``model_capacity`` when a name does not fit Boltz's four characters; the refusals of
            ``standardize_smiles`` and ``read_smiles``.
    """

    text = standardize_smiles(smiles)
    component = read_smiles(text, component_id="LIG1").component
    ranks = canonical_atom_ranks(text)
    assert len(ranks) == len(component.atom_names), "read_smiles and RDKit disagree on the atoms of a SMILES"
    names = tuple(f"{ELEMENTS[e].upper()}{r + 1}" for e, r in zip(component.element.tolist(), ranks.tolist()))
    for name in names:
        if len(name) > 4:
            fail("model_capacity", RequestRef("smiles", smiles), atom_name=name)
    values = {f.name: getattr(component, f.name) for f in fields(component) if f.name not in _DERIVED_FIELDS}
    return stated_component({**values, "atom_names": names}, scope=component.scope)


class Boltz2Processor(Processor):
    r"""
    Prepare one structure for Boltz-2 folding.

    Args:
        config: Input-processing configuration. Defaults to ``Boltz2ProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.

    Notes:
        Atom grouping and representative atoms follow the target's resolved biological classes. Reference conformers
        and experimental observations have independent availability; generated reference geometry remains separate
        from observed coordinates.

        Learned relation categories summarize positive source edges and lose chemical detail. The canonical target
        retains the complete typed and directed relations. The ``modified`` feature marks membership in the model's
        label table; it is not a chemical assessment of a residue modification.
    """

    config_class = Boltz2ProcessorConfig
    config: Boltz2ProcessorConfig

    def __init__(
        self,
        config: Boltz2ProcessorConfig | None = None,
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
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        designed_positions: Sequence[ComponentRef] = (),
        self_template: Sequence[ComponentRef] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[Boltz2Input]:
        r"""
        Prepare one structure: layout, optional training crop, slots, gating, reference conformers, constraint
        features, MSA rows, templates, conditions and the Boltz-2 tensors. The scope is every atom in a new frame.

        Args:
            structure: The target; its chemistry is consumed as read (no library).
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            templates: Protein templates, one model template row each.
            designed_positions: Components designed by the calling protocol (design mask and design scores).
            self_template: Components whose own coordinates form one more template row.
            contacts: Pairwise contact conditions.
            pockets: Binder-to-pocket conditions.
            method: Experimental method of the learned method embedding.
            cropper: Optional training crop policy; needs ``generator``.
            generator: The numpy generator of the crop and of the reference-position augmentation.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        layout = token_layout(system, BOLTZ2_LAYOUT)
        if cropper is not None:
            required = _required_tokens(system, layout, self_template, contacts, pockets)
            layout = crop_token_layout(structure, layout, cropper, required=required, generator=generator)
        return _prepare(
            structure,
            layout,
            seed=self.seed,
            spec=self.preparation_spec,
            device=self.device,
            dtype=self.dtype,
            msa=msa,
            cropped=cropper is not None,
            templates=templates,
            designed_positions=designed_positions,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            ligand=None,
            generator=generator,
        )

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[Boltz2Input],
    ) -> Boltz2StructureTarget:
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
            symmetry=BOLTZ2_SYMMETRY,
        )
        return self._target_tensors(prepared, truth)

    def _target_tensors(self, prepared: PreparedInput[Boltz2Input], target: StructureTarget) -> Boltz2StructureTarget:
        r"""The model's tensor target: canonical truth (one conformer), chain spans, chain equivalence and slot-axis
        symmetry groups."""
        system = prepared.context.structure.system
        atom_ptr = system.component_atom_ptr[system.instance_component_ptr]
        observed = torch.from_numpy(target.truth == TruthState.OBSERVED)
        coordinates = torch.from_numpy(target.coordinates.copy())
        group_atoms, permutations = _symmetry_groups(prepared.context.slots, target)
        return Boltz2StructureTarget(  # type: ignore[call-arg]
            coordinates=project_model_coordinates(coordinates, observed, feature="Boltz2 target coordinates").to(
                self.device
            )[None, None],
            coordinate_mask=observed.to(self.device).unsqueeze(0),
            chain_span=torch.from_numpy(np.stack((atom_ptr[:-1], atom_ptr[1:]), axis=1)).to(self.device).unsqueeze(0),
            chain_equivalence_index=torch.from_numpy(system.instance_entity.astype(np.int64))
            .to(self.device)
            .unsqueeze(0),
            symmetry_atom_index=group_atoms.to(self.device).unsqueeze(0),
            symmetry_permutation=permutations.to(self.device).unsqueeze(0),
            batch_size=[1],
        )


class Boltz2AffinityProcessor(Processor):
    r"""
    Prepare one folded protein-ligand structure for Boltz-2 affinity scoring.

    Args:
        config: Input-processing configuration. Defaults to ``Boltz2ProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = Boltz2ProcessorConfig
    config: Boltz2ProcessorConfig

    def __init__(
        self,
        config: Boltz2ProcessorConfig | None = None,
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
        ligand: str | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[Boltz2Input]:
        r"""
        Prepare the ligand-centred crop of one folded complex (upstream ``AffinityCropper``) for affinity scoring.

        Args:
            structure: The folded complex; the crop is centred on the ligand's observed token centres.
            ligand: The instance scored, of a non-polymer entity; may be omitted when the structure has exactly one.
            msa: MSA pools by instance, an exact plan, or ``None``.
            contacts: Pairwise contact conditions.
            pockets: Binder-to-pocket conditions.
            method: Experimental method of the learned method embedding.
            generator: The numpy generator of the reference-position augmentation.

        Raises:
            InvalidRequestError: ``request_invalid_positions`` when ``ligand`` names no ligand instance (or is omitted
                with several).
            UnsupportedError: ``model_capacity`` for a structure without a protein instance, a ligand entity with
                several instances or a ligand of more than 128 heavy atoms.
            InsufficientInformationError: ``affinity_ligand_unobserved`` when no ligand token centre is observed.
        """

        system = structure.system
        instance = _affinity_ligand(system, ligand)
        layout = token_layout(system, BOLTZ2_LAYOUT)
        layout = select_tokens(layout, _affinity_crop(structure, layout, instance))
        return _prepare(
            structure,
            layout,
            seed=self.seed,
            spec=self.preparation_spec,
            device=self.device,
            dtype=self.dtype,
            msa=msa,
            cropped=False,
            templates=(),
            designed_positions=(),
            self_template=(),
            contacts=contacts,
            pockets=pockets,
            method=method,
            ligand=instance,
            generator=generator,
        )


def _prepare(
    structure: Structure,
    layout: TokenLayout,
    *,
    seed: int,
    spec: PreparationSpec,
    device: torch.device,
    dtype: torch.dtype,
    msa: Mapping[str, MSA] | MsaCollection | None,
    cropped: bool,
    templates: Sequence[Template | Boltz2Template],
    designed_positions: Sequence[ComponentRef],
    self_template: Sequence[ComponentRef],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
    method: str,
    ligand: int | None,
    generator: np.random.Generator | None,
) -> PreparedInput[Boltz2Input]:
    r"""The one preparation of both processors over a settled (cropped) layout."""
    if method.lower() not in _METHOD_TYPES:
        fail("value_invalid", RequestRef("method", method))
    system = structure.system
    slots = slot_map(system, layout, BOLTZ2_LAYOUT)
    accepted = gate(structure, slots, BOLTZ2_GATING)
    references, sources, embedded = reference_conformers(structure, slots, embed=missing_reference(system), seed=seed)
    scopes = _constraint_scopes(system, slots)
    constraints = [constraint_features(system, atoms) for atoms in scopes]
    tokens, losses = model_tokens(system, layout)
    rows = _msa_rows(msa, system, layout, cropped=cropped)
    torch_generator = None if generator is None else torch.Generator().manual_seed(int(generator.integers(2**62)))
    residue = _residue_tokens(layout, len(system.component_symbol))
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    design = np.isin(layout.token_component, _components(system, designed_positions, "designed_positions"))
    msa_features = _msa_features(rows, tokens)
    tensors: dict[str, Tensor] = {
        **_token_features(system, layout, tokens, residue, _METHOD_TYPES[method.lower()]),
        **_atom_features(system, slots, references, torch_generator),
        **_steering(system, slots, constraints),
        **_contact_features(system, slots, residue, contacts, pockets),
        **msa_features,
        **_template_features(structure, slots, residue, templates, self_template),
        "design_mask": torch.from_numpy(design),
        "chain_design_mask": torch.from_numpy(_chain_design_mask(system, token_instance, design)),
    }
    affinity_mw = None
    if ligand is not None:
        # upstream's affinity MSA features: the profile and deletion mean of the first plan row alone (max_seqs=1)
        tensors["affinity_token_mask"] = torch.from_numpy(token_instance == ligand)
        tensors["affinity_msa_profile"] = torch.nn.functional.one_hot(
            msa_features["msa_tokens"][0], len(TOKENS)
        ).float()
        tensors["affinity_deletion_mean"] = msa_features["deletion_value"][0]
        atoms = np.flatnonzero(_owner(system.instance_component_ptr)[_owner(system.component_atom_ptr)] == ligand)
        if any(np.array_equal(scope, atoms) for scope in scopes):  # a determined graph: its weight is a fact
            affinity_mw = torch.tensor([[molecular_weight(system, atoms)]], dtype=dtype, device=device)
    placed = {
        name: (
            (
                value.to(device=device, dtype=dtype)
                if name == "template_force_threshold"  # infinity is the stated "no threshold"
                else project_model_features(
                    value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"Boltz2 prepared {name}"
                ).to(device=device)
            )
            if value.is_floating_point()
            else value.to(device=device)
        )
        for name, value in tensors.items()
    }
    features = Boltz2Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        atom_index=torch.from_numpy(slots.atom.copy()).to(device).unsqueeze(0),
        affinity_mw=affinity_mw,
        batch_size=[],
    )
    conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
    bound = slots.atom >= 0
    conformers[_owner(system.component_atom_ptr)[slots.atom[bound]]] = sources[bound]
    return PreparedInput(
        features,
        PreparationContext(
            structure,
            slots,
            RegenerationScope.all(system),
            PreparationRecord(
                Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)), conformers
            ),
            spec=spec,
        ),
    )


def _residue_tokens(layout: TokenLayout, count: int) -> NDArray[np.int64]:
    r"""Per component, its residue token (its one token, of a polymer class), else ``-1``."""
    tokens = np.bincount(layout.token_component, minlength=count)
    residue = (tokens[layout.token_component] == 1) & (layout.token_class != TOKEN_INDEX["OTHER"])
    found = np.full(count, -1, dtype=np.int64)
    found[layout.token_component[residue]] = np.flatnonzero(residue)
    return found


def _components(system: MolecularSystem, refs: Sequence[ComponentRef | AtomSite], argument: str) -> NDArray[np.int64]:
    r"""The component of each reference (an atom site names its component); an unknown one is a request error."""
    components = system.component_index([ComponentRef(ref.instance, ref.component) for ref in refs])
    if (components < 0).any():
        fail("request_invalid_positions", RequestRef(argument, str(refs[int(np.flatnonzero(components < 0)[0])])))
    return components


def _instance(system: MolecularSystem, symbol: str, argument: str) -> int:
    instance = int(system.instance_index([symbol])[0])
    if instance < 0:
        fail("request_invalid_positions", RequestRef(argument, symbol))
    return instance


def _required_tokens(
    system: MolecularSystem,
    layout: TokenLayout,
    self_template: Sequence[ComponentRef],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
) -> NDArray[np.bool_]:
    r"""The tokens a crop must keep: the components the conditions name and every token of each pocket binder."""
    endpoints = [e for c in contacts for e in (c.endpoint_a, c.endpoint_b)] + [e for p in pockets for e in p.contacts]
    required = np.isin(layout.token_component, _components(system, [*self_template, *endpoints], "conditions"))
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    for pocket in pockets:
        required |= token_instance == _instance(system, pocket.binder, "pockets.binder")
    return required


def _chain_types(system: MolecularSystem) -> NDArray[np.int64]:
    r"""Per instance, its Boltz molecule type: a polymer by the one monomer type of its typed components, everything
    else (and a mixed polymer) non-polymer."""
    s = system.arrays()
    entity = s["instance_entity"]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    pointer = s["instance_component_ptr"]
    kinds = np.zeros((len(pointer) - 1, len(MonomerType)), dtype=bool)
    kinds[_owner(pointer), CLASS_MONOMER[token_classes(system).token]] = True
    single = polymer & (kinds[:, 1:].sum(axis=1) == 1)
    types = np.array([BOLTZ2_LAYOUT.monomer_types[m] for m in (_PROTEIN, _RNA, _DNA)], dtype=np.int64)
    return np.where(single, types[kinds[:, 1:].argmax(axis=1)], _NONPOLYMER)


def _cyclic_periods(system: MolecularSystem) -> NDArray[np.int64]:
    r"""Per instance, its component count when a KNOWN covalent edge joins its first component's previous port to its
    last component's next port (a head-to-tail cyclic polymer), else 0."""
    s = system.arrays()
    pointer = s["instance_component_ptr"]
    periods = np.zeros(len(pointer) - 1, dtype=np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    pairs = {tuple(edge) for edge in np.sort(s["edge_index"][:, covalent], axis=0).T.tolist()}
    for instance, (start, stop) in enumerate(zip(pointer[:-1].tolist(), pointer[1:].tolist())):
        if stop - start < 2:
            continue
        head, tail = int(s["component_port"][start, 0]), int(s["component_port"][stop - 1, 1])
        if head >= 0 and tail >= 0 and (min(head, tail), max(head, tail)) in pairs:
            periods[instance] = stop - start
    return periods


def _token_bonds(system: MolecularSystem, layout: TokenLayout) -> dict[str, Tensor]:
    r"""
    ``token_bonds`` and ``type_bonds`` (``[N, N]``) over the KNOWN covalent edges upstream lists: every edge between two
    tokens except polymer junctions (one component's next port to another's previous port in one instance), and the
    edges inside a modified residue's token (its own diagonal). The type is upstream's bond type + 1: a connection
    between components 6, an aromatic bond 5, a KNOWN order 1/2/3 as 2/3/4, else 1; several edges of different types
    between two tokens give 1.
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
    standard = token_classes(system).encoding == Encoding.EXACT
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = covalent & ~junction & (a >= 0) & (b >= 0) & ~((a == b) & standard[left])
    order = s["bond_order"].astype(np.int64)
    kind = np.where(known(s["bond_order_state"]) & np.isin(order, (1, 2, 3)), order + 1, 1)
    kind[known(s["bond_aromatic_state"]) & s["bond_aromatic"]] = 5
    kind[left != right] = 6
    count = len(layout.token_component)
    low_token, high_token = np.minimum(a, b)[linked], np.maximum(a, b)[linked]
    pairs, inverse = np.unique(low_token * count + high_token, return_inverse=True)
    low, high = np.full(len(pairs), 7, dtype=np.int64), np.zeros(len(pairs), dtype=np.int64)
    np.minimum.at(low, inverse, kind[linked])
    np.maximum.at(high, inverse, kind[linked])
    category = np.where(high == 6, 6, np.where(low == high, low, 1))
    i, j = torch.from_numpy(pairs // count), torch.from_numpy(pairs % count)
    bonds = torch.zeros(count, count)
    types = torch.zeros(count, count, dtype=torch.long)
    bonds[i, j] = bonds[j, i] = 1
    types[i, j] = types[j, i] = torch.from_numpy(category)
    return {"token_bonds": bonds, "type_bonds": types}


def _token_features(
    system: MolecularSystem, layout: TokenLayout, tokens: NDArray[np.int64], residue: NDArray[np.int64], method: int
) -> dict[str, Tensor]:
    r"""The token-axis features: indices, types, the modified flag, method, cyclic periods and token bonds."""
    s = system.arrays()
    pointer = s["instance_component_ptr"]
    count = len(tokens)
    token_instance = _owner(pointer)[layout.token_component]
    entity = s["instance_entity"]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    is_residue = residue[layout.token_component] == np.arange(count)
    modified = is_residue & (token_classes(system).encoding[layout.token_component] != Encoding.EXACT)
    return {
        "token_mask": torch.ones(count, dtype=torch.bool),
        "residue_index": torch.from_numpy(layout.token_component - pointer[token_instance]),
        "chain_index": torch.from_numpy(token_instance),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64)),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "token_index": torch.arange(count),
        "residue_type_features": torch.nn.functional.one_hot(torch.from_numpy(tokens), len(TOKENS)).float(),
        "molecule_type": torch.from_numpy(_chain_types(system)[token_instance]),
        "method_feature": torch.full((count,), method, dtype=torch.long),
        "modified": torch.from_numpy(modified.astype(np.int64)),
        "cyclic_period": torch.from_numpy(_cyclic_periods(system)[token_instance].astype(np.float32)),
        **_token_bonds(system, layout),
    }


def _atom_names(system: MolecularSystem, atoms: NDArray[np.int64]) -> list[str]:
    r"""Atom names in four characters; a longer or non-printable name becomes its element and a per-instance count."""
    names = system.text("atom", system.atom_symbol[atoms]).tolist()
    instance = _owner(system.instance_component_ptr)[_owner(system.component_atom_ptr)[atoms]].tolist()
    elements = system.atomic_number[atoms].tolist()
    counts: dict[tuple[int, str], int] = {}
    encoded = []
    for name, owner, number in zip(names, instance, elements):
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = ELEMENTS[number].upper()
            counts[owner, element] = counts.get((owner, element), 0) + 1
            name = f"{element}{counts[owner, element]}"
        encoded.append(name)
    return encoded


def _atom_features(
    system: MolecularSystem, slots: SlotMap, references: NDArray[np.float64], generator: torch.Generator | None
) -> dict[str, Tensor]:
    r"""The slot-axis features: reference positions (augmented), charge, element, name characters and atom indices."""
    atom, layout = slots.atom, slots.layout
    num_atoms = len(atom)
    slot_of = np.full(len(system.atom_symbol), -1, dtype=np.int64)
    slot_of[atom] = np.arange(num_atoms)
    reference_positions = torch.from_numpy(references.astype(np.float32))
    reference_space_id = torch.from_numpy(_owner(system.component_atom_ptr)[atom])
    augment_reference_positions(
        reference_positions, reference_space_id, torch.ones(num_atoms, dtype=torch.bool), generator=generator
    )
    characters = torch.tensor(
        [[ord(c) - 32 if c != " " else 0 for c in name.ljust(4)] for name in _atom_names(system, atom)],
        dtype=torch.long,
    ).reshape(num_atoms, 4)
    canonical = to_tensors(system, ("atomic_number", "formal_charge"), device=torch.device("cpu"))
    index = torch.from_numpy(atom)
    # Boltz-2 reference-atom features: positions (3), charge (1), element (128), name characters (4x64).
    atom_features = torch.cat(
        (
            reference_positions,
            canonical["formal_charge"][index].float().unsqueeze(-1),
            torch.nn.functional.one_hot(canonical["atomic_number"][index].long(), 128).float(),
            torch.nn.functional.one_hot(characters, 64).reshape(num_atoms, 256).float(),
        ),
        dim=-1,
    )
    return {
        "atom_features": atom_features,
        "atom_to_token": torch.from_numpy(slots.token),
        "atom_mask": torch.ones(num_atoms, dtype=torch.bool),
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "token_representative_atom_index": torch.from_numpy(
            slot_of[token_representatives(system, layout, BOLTZ2_LAYOUT)]
        ),
        "frame_atom_index": torch.from_numpy(slot_of[token_frames(system, layout, BOLTZ2_LAYOUT)]),
    }


def _constraint_scopes(system: MolecularSystem, slots: SlotMap) -> list[NDArray[np.int64]]:
    r"""
    The atom sets Boltz-2 derives steering constraints for: each non-polymer instance, and each non-standard polymer
    component other than MSE (standard residues and MSE are constraint-free upstream), that holds a slot atom and
    projects to RDKit on its own — every claim True, connections complete, no edge or external connection leaving it.
    """

    s = system.arrays()
    count, instances = len(s["component_symbol"]), len(s["instance_symbol"])
    atom_component = _owner(s["component_atom_ptr"])
    component_instance = _owner(s["instance_component_ptr"])
    entity = s["instance_entity"][component_instance]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    standard = token_classes(system).encoding == Encoding.EXACT
    standard |= system.text("component", s["component_symbol"]) == "MSE"
    unit = np.where(polymer, np.arange(count), count + component_instance)
    complete = s["component_topology_complete"] & s["component_heavy_atoms_complete"]
    complete &= system.component_hydrogens_complete & s["instance_connections_complete"][component_instance]
    atom_unit = unit[atom_component]
    first, second = s["edge_index"].astype(np.int64)
    crossing = atom_unit[first] != atom_unit[second]
    leaving = np.zeros(count + instances, dtype=bool)
    leaving[atom_unit[first[crossing]]] = leaving[atom_unit[second[crossing]]] = True
    leaving[atom_unit[s["external_atom"].astype(np.int64)]] = True
    blocked = np.zeros(count + instances, dtype=bool)
    blocked[unit[~complete | (polymer & standard)]] = True
    used = np.zeros(count + instances, dtype=bool)
    used[atom_unit[slots.atom[slots.atom >= 0]]] = True
    scopes = np.flatnonzero(used & ~leaving & ~blocked)
    return [np.flatnonzero(atom_unit == u) for u in scopes.tolist()]


def _steering(system: MolecularSystem, slots: SlotMap, constraints: list[ConstraintFeatures]) -> dict[str, Tensor]:
    r"""Constraint features on the slot axis, covalent connections between instances and same-entity chain pairs."""
    s = system.arrays()
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[slots.atom[slots.atom >= 0]] = np.flatnonzero(slots.atom >= 0)

    def rows(index: NDArray[np.int64], *flags: NDArray) -> tuple[Tensor, ...]:
        mapped = slot_of[index]
        keep = (mapped >= 0).all(axis=1)
        return (torch.from_numpy(mapped[keep].T.copy()), *(torch.from_numpy(flag[keep].copy()) for flag in flags))

    def joined(name: str, width: int) -> NDArray:
        parts = [getattr(c, name) for c in constraints]
        return np.concatenate(parts) if parts else np.zeros((0, width) if width else 0)

    pairs, bond, angle, lower, upper = rows(
        joined("pairs", 2).astype(np.int64),
        joined("bond_pair", 0).astype(bool),
        joined("angle_pair", 0).astype(bool),
        joined("lower", 0).astype(np.float64),
        joined("upper", 0).astype(np.float64),
    )
    chiral, positive = rows(joined("chiral", 4).astype(np.int64), joined("chiral_positive", 0).astype(bool))
    stereo, trans = rows(joined("stereo_bond", 4).astype(np.int64), joined("stereo_trans", 0).astype(bool))
    (planar,) = rows(joined("planar", 6).astype(np.int64))
    atom_instance = _owner(s["instance_component_ptr"])[_owner(s["component_atom_ptr"])]
    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    between = covalent & (atom_instance[first] != atom_instance[second])
    between &= (slot_of[first] >= 0) & (slot_of[second] >= 0)
    active = np.unique(atom_instance[slots.atom[slots.atom >= 0]])
    entity = s["instance_entity"]
    left, right = np.triu_indices(len(active), 1)
    same = entity[active[left]] == entity[active[right]]
    return {
        "geometry_pair_index": pairs,
        "geometry_bond_mask": bond,
        "geometry_angle_mask": angle,
        "geometry_upper_bounds": upper,
        "geometry_lower_bounds": lower,
        "chiral_atom_index": chiral,
        "chiral_atom_positive": positive,
        "stereo_bond_index": stereo,
        "stereo_bond_trans": trans,
        "planar_bond_index": planar,
        "connected_chain_index": torch.from_numpy(
            np.stack((atom_instance[first[between]], atom_instance[second[between]]))
        ),
        "connected_atom_index": torch.from_numpy(np.stack((slot_of[first[between]], slot_of[second[between]]))),
        "symmetric_chain_index": torch.from_numpy(np.stack((active[left][same], active[right][same]))),
    }


def _endpoint_token(
    system: MolecularSystem, layout: TokenLayout, residue: NDArray[np.int64], endpoint: ComponentRef | AtomSite
) -> int:
    r"""The token of a condition endpoint (upstream): a residue names its residue token, an atom its own atom token."""
    if isinstance(endpoint, AtomSite):
        atom = int(system.atom_index([endpoint])[0])
        token = int(layout.atom_token[atom]) if atom >= 0 else -1
        if token >= 0 and residue[layout.token_component[token]] == token:
            token = -1
    else:
        component = int(system.component_index([endpoint])[0])
        token = int(residue[component]) if component >= 0 else -1
    if token < 0:
        fail("request_invalid_positions", RequestRef("conditions", str(endpoint)))
    return token


def _contact_features(
    system: MolecularSystem,
    slots: SlotMap,
    residue: NDArray[np.int64],
    contacts: Sequence[Boltz2Contact],
    pockets: Sequence[Boltz2Pocket],
) -> dict[str, Tensor]:
    r"""
    The learned contact conditioning (``[N, N, 5]`` one-hot and thresholds; every pair UNSPECIFIED without a condition)
    and, for forced conditions, the slot atom pairs of the contact potential with their union group and threshold.
    """

    layout = slots.layout
    count = len(layout.token_component)
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    classes = np.full((count, count), _UNSELECTED, dtype=np.int64)
    threshold = np.zeros((count, count), dtype=np.float32)
    potential: list[tuple[NDArray[np.int64], NDArray[np.int64], float]] = []

    def slots_of(tokens: NDArray[np.int64]) -> NDArray[np.int64]:
        return np.flatnonzero(np.isin(slots.token, tokens))

    for pocket in pockets:
        binder = np.flatnonzero(token_instance == _instance(system, pocket.binder, "pockets.binder"))
        if not len(binder):
            fail("request_invalid_positions", RequestRef("pockets.binder", pocket.binder))
        for endpoint in pocket.contacts:
            token = _endpoint_token(system, layout, residue, endpoint)
            classes[binder, token], classes[token, binder] = _BINDER_TO_POCKET, _POCKET_TO_BINDER
            threshold[binder, token] = threshold[token, binder] = pocket.max_distance
            if pocket.force:
                potential.append((slots_of(binder), slots_of(np.array([token])), pocket.max_distance))
    for contact in contacts:
        a = _endpoint_token(system, layout, residue, contact.endpoint_a)
        b = _endpoint_token(system, layout, residue, contact.endpoint_b)
        classes[a, b] = classes[b, a] = _CONTACT
        threshold[a, b] = threshold[b, a] = contact.max_distance
        if contact.force:
            potential.append((slots_of(np.array([a])), slots_of(np.array([b])), contact.max_distance))
    if (classes == _UNSELECTED).all():
        classes[:] = _UNSPECIFIED
    pairs = [np.stack(np.meshgrid(left, right, indexing="ij")).reshape(2, -1) for left, right, _ in potential]
    union = [np.full(pair.shape[1], k, dtype=np.int64) for k, pair in enumerate(pairs)]
    limits = [np.full(pair.shape[1], limit, dtype=np.float32) for pair, (_, _, limit) in zip(pairs, potential)]
    return {
        "contact_conditioning": torch.nn.functional.one_hot(torch.from_numpy(classes), 5),
        "contact_threshold": torch.from_numpy(threshold),
        "contact_pair_index": torch.from_numpy(np.concatenate([np.zeros((2, 0), dtype=np.int64), *pairs], axis=1)),
        "contact_union_index": torch.from_numpy(np.concatenate([np.zeros(0, dtype=np.int64), *union])),
        "contact_thresholds": torch.from_numpy(np.concatenate([np.zeros(0, dtype=np.float32), *limits])),
    }


def _chain_design_mask(
    system: MolecularSystem, token_instance: NDArray[np.int64], design: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    r"""Per token, whether its instance is joined to a designed instance through KNOWN covalent connections."""
    s = system.arrays()
    atom_instance = _owner(s["instance_component_ptr"])[_owner(s["component_atom_ptr"])]
    first, second = s["edge_index"][:, known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)]
    a, b = atom_instance[first.astype(np.int64)], atom_instance[second.astype(np.int64)]
    selected = np.zeros(len(s["instance_symbol"]), dtype=bool)
    selected[token_instance[design]] = True
    while True:
        grown = selected.copy()
        grown[b[selected[a]]] = grown[a[selected[b]]] = True
        if (grown == selected).all():
            return selected[token_instance]
        selected = grown


def _msa_rows(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout, *, cropped: bool
) -> MsaRows | None:
    r"""Pools are paired by the one OFoldX rule after the crop; an exact plan is used as given."""
    if msa is None:
        return None
    plan: MsaCollection | None
    if isinstance(msa, MsaCollection):
        if cropped:
            fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
        if msa.row_pairing.shape[0] > BOLTZ2_MSA_DEPTH:
            fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=BOLTZ2_MSA_DEPTH)
        plan = msa
    else:
        plan = pair_msas(msa, system, layout, max_rows=BOLTZ2_MSA_DEPTH)
    return None if plan is None else msa_rows(plan, system, layout)


def _msa_features(rows: MsaRows | None, tokens: NDArray[np.int64]) -> dict[str, Tensor]:
    r"""Boltz's MSA features: deletions as ``π/2·atan(d/3)``, the profile and deletion mean over every plan row."""
    if rows is None:
        token = torch.from_numpy(tokens[None].copy())
        deletions = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        token = torch.from_numpy(rows.token.astype(np.int64))
        deletions = torch.from_numpy(rows.deletions.astype(np.float32))
        paired = torch.from_numpy(rows.paired.astype(np.float32))
    deletion_value = (math.pi / 2) * torch.atan(deletions / 3)
    return {
        "msa_tokens": token,
        "has_deletion": deletions > 0,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(token, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(token, len(TOKENS)).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _template_row(
    token_instance: NDArray[np.int64],
    tokens: NDArray[np.int64],
    classes: NDArray[np.int64],
    atoms: NDArray[np.float64],
    present: NDArray[np.bool_],
    framed: NDArray[np.bool_],
    visibility: NDArray[np.int64],
) -> dict[str, NDArray]:
    r"""
    One template row over the tokens of ``token_instance``. ``atoms``/``present`` (``[k, 4, 3]``/``[k, 4]``) are the
    N, CA, C and pseudo-beta observations placed on ``tokens``; ``framed`` marks the positions that carry a backbone
    frame (standard protein residues, upstream), computed from N, CA, C as upstream ``compute_frame`` where all three
    are observed. Visibility is upstream's per chain: every token of an instance holding templated tokens of one frame
    takes that frame, every other instance its own ``-1 - instance``.
    """

    count = len(token_instance)
    n, ca, c, pseudo = (atoms[:, k] for k in range(4))
    frame = framed & present[:, :3].all(axis=1)
    e1 = (c - ca) / (np.linalg.norm(c - ca, axis=-1, keepdims=True) + 1e-10)
    u2 = (n - ca) - e1 * (e1 * (n - ca)).sum(axis=-1, keepdims=True)
    e2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + 1e-10)
    rotation = np.where(frame[:, None, None], np.stack((e1, e2, np.cross(e1, e2)), axis=-1), np.eye(3))
    residue_type = np.full(count, TOKEN_INDEX["<pad>"])  # upstream's untemplated class 0: <pad> on both alphabets
    residue_type[tokens] = classes
    row = {
        "template_residue_type_features": np.eye(len(TOKENS), dtype=np.int64)[residue_type],
        "template_frame_rotations": np.zeros((count, 3, 3)),
        "template_frame_translations": np.zeros((count, 3)),
        "template_frame_mask": np.zeros(count),
        "template_pseudo_beta_positions": np.zeros((count, 3)),
        "template_alpha_carbon_positions": np.zeros((count, 3)),
        "template_pseudo_beta_mask": np.zeros(count),
        "template_mask": np.zeros(count),
        "template_visibility_index": -1 - token_instance,
    }
    row["template_frame_rotations"][tokens] = rotation
    row["template_frame_translations"][tokens[frame]] = ca[frame]
    row["template_frame_mask"][tokens] = frame
    row["template_pseudo_beta_positions"][tokens] = pseudo
    row["template_alpha_carbon_positions"][tokens] = ca
    row["template_pseudo_beta_mask"][tokens] = present[:, 3]
    row["template_mask"][tokens] = 1
    for instance in np.unique(token_instance[tokens]):
        frames = np.unique(visibility[token_instance[tokens] == instance])
        if len(frames) == 1:
            row["template_visibility_index"][token_instance == instance] = frames[0]
    row["template_visibility_index"][tokens] = visibility
    return row


def _chain_row(
    template: Template, system: MolecularSystem, residue: NDArray[np.int64], token_instance: NDArray[np.int64]
) -> dict[str, NDArray]:
    r"""
    One ``Template`` as one row: each alignment resolved once (``resolve_template``, protein targets) and placed on the
    residue tokens of its target positions (positions outside the crop are dropped), in its placement frame.
    """

    resolved = resolve_template(template, system, monomer_types=(_PROTEIN,))
    pointer = system.instance_component_ptr
    parts = []
    for alignment, frame in zip(resolved, template_alignment_frames(resolved), strict=True):
        chain: ChainCoordinates = template.coordinates.chains[alignment.template_instance_id]
        target, source = np.array(alignment.residue_pairs, dtype=np.int64).T
        tokens = residue[pointer[alignment.target_instance_index] + target]
        kept = tokens >= 0
        source = source[kept]
        classes = chain.token_class[source].astype(np.int64)
        atoms, present = chain.gather(source.tolist(), ("N", "CA", "C", "CB", "CA"))
        glycine = classes == TOKEN_INDEX["G"]
        atoms[glycine, 3], present[glycine, 3] = atoms[glycine, 4], present[glycine, 4]
        framed = chain.encoding[source] == Encoding.EXACT
        parts.append((tokens[kept], classes, atoms[:, :4], present[:, :4], framed, np.full(len(source), frame)))
    columns = [np.concatenate(column) for column in zip(*parts)]
    return _template_row(token_instance, *columns)


def _self_row(
    structure: Structure, slots: SlotMap, residue: NDArray[np.int64], refs: Sequence[ComponentRef]
) -> dict[str, NDArray]:
    r"""The named components' own coordinates as one template row (frame 0): each token's centre as CA, its
    representative as pseudo-beta, and the N, CA, C frame of a standard protein residue token."""
    system, layout = structure.system, slots.layout
    tokens = np.flatnonzero(np.isin(layout.token_component, _components(system, refs, "self_template")))
    frames = token_frames(system, layout, BOLTZ2_LAYOUT)[tokens]
    centre = layout.token_centre_atom[tokens]
    representative = token_representatives(system, layout, BOLTZ2_LAYOUT)[tokens]
    atoms = np.column_stack((frames[:, 0], centre, frames[:, 2], representative))
    observed = structure.coordinate_mask[atoms]
    component = layout.token_component[tokens]
    framed = (residue[component] == tokens) & (frames[:, 0] != centre) & (frames[:, 2] != centre)
    framed &= _chain_types(system)[_owner(system.instance_component_ptr)[component]] == 0
    framed &= token_classes(system).encoding[component] == Encoding.EXACT
    coordinates = np.where(observed[..., None], structure.coordinates[atoms], 0.0)
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    return _template_row(
        token_instance,
        tokens,
        layout.token_class[tokens].astype(np.int64),
        coordinates,
        observed,
        framed,
        np.zeros_like(tokens),
    )


def _template_features(
    structure: Structure,
    slots: SlotMap,
    residue: NDArray[np.int64],
    templates: Sequence[Template | Boltz2Template],
    self_template: Sequence[ComponentRef],
) -> dict[str, Tensor]:
    r"""One model template row per ``Template`` (with its force and threshold), then the self-template row."""
    token_instance = _owner(structure.system.instance_component_ptr)[slots.layout.token_component]
    rows, force, thresholds = [], [], []
    for value in templates:
        template, forced, limit = (
            (value.template, value.force, value.threshold)
            if isinstance(value, Boltz2Template)
            else (value, False, None)
        )
        rows.append(_chain_row(template, structure.system, residue, token_instance))
        force.append(forced)
        thresholds.append(math.inf if limit is None else limit)
    if self_template:
        rows.append(_self_row(structure, slots, residue, self_template))
        force.append(False)
        thresholds.append(math.inf)
    if not rows:
        return {}
    return {
        **{name: torch.from_numpy(np.stack([row[name] for row in rows])) for name in rows[0]},
        "template_force": torch.tensor(force, dtype=torch.bool),
        "template_force_threshold": torch.tensor(thresholds, dtype=torch.float32),
    }


def _affinity_ligand(system: MolecularSystem, ligand: str | None) -> int:
    r"""The instance affinity scores: a ligand instance (of no polymer or water entity) of a single-copy entity of at
    most 128 heavy atoms, in a structure holding a protein instance."""
    s = system.arrays()
    entity = s["instance_entity"]
    stated = known(s["entity_type_state"][entity])
    candidates = stated & ~np.isin(s["entity_type"][entity], (EntityType.POLYMER, EntityType.WATER))
    if ligand is None:
        if candidates.sum() != 1:
            fail("request_invalid_positions", RequestRef("ligand", "None"), count=int(candidates.sum()))
        instance = int(np.flatnonzero(candidates)[0])
    else:
        instance = _instance(system, ligand, "ligand")
        if not candidates[instance]:
            fail("request_invalid_positions", RequestRef("ligand", ligand))
    if not (_chain_types(system) == 0).any():
        fail("model_capacity", StructureRef(), item="no protein receptor")
    if (entity == entity[instance]).sum() > 1:
        fail("model_capacity", StructureRef(instance=instance), item="copied ligand entity")
    atom_instance = _owner(s["instance_component_ptr"])[_owner(s["component_atom_ptr"])]
    heavy = int(((atom_instance == instance) & (s["atomic_number"] != 1)).sum())
    if heavy > _AFFINITY_HEAVY_ATOMS:
        fail("model_capacity", StructureRef(instance=instance), item="heavy atoms", count=heavy)
    return instance


def _affinity_crop(structure: Structure, layout: TokenLayout, ligand: int) -> NDArray[np.bool_]:
    r"""
    Upstream ``AffinityCropper``: tokens with an observed centre, nearest the ligand's first, each adding its chain
    when the chain has at most 10 tokens, else the 10-token residue neighbourhood around it, until 256 tokens, 2,048
    atoms or 200 non-ligand tokens would be exceeded.
    """

    system = structure.system
    count = len(layout.token_component)
    pointer = system.instance_component_ptr
    token_instance = _owner(pointer)[layout.token_component]
    position = layout.token_component - pointer[token_instance]
    nonpolymer = _chain_types(system)[token_instance] == _NONPOLYMER
    sizes = np.diff(layout.token_atom_ptr)
    centre = layout.token_centre_atom
    valid = np.flatnonzero(structure.coordinate_mask[centre])
    anchors = valid[token_instance[valid] == ligand]
    if not len(anchors):
        fail("affinity_ligand_unobserved", StructureRef(instance=ligand))
    points = structure.coordinates[centre]
    distance = np.sqrt(((points[valid, None] - points[None, anchors]) ** 2).sum(axis=-1)).min(axis=1)
    cropped = np.zeros(count, dtype=bool)
    for query in valid[np.argsort(distance, kind="stable")].tolist():
        chain = token_instance == token_instance[query]
        if chain.sum() <= _NEIGHBOURHOOD:
            new = chain
        else:
            window = chain & (np.abs(position - position[query]) <= _NEIGHBOURHOOD)
            radius = 0
            new = window & (position == position[query])
            while new.sum() < _NEIGHBOURHOOD and radius < _NEIGHBOURHOOD:
                radius += 1
                new = window & (np.abs(position - position[query]) <= radius)
        new &= ~cropped
        if (
            new.sum() > _AFFINITY_TOKENS - cropped.sum()
            or sizes[cropped | new].sum() > _AFFINITY_ATOMS
            or ((cropped | new) & ~nonpolymer).sum() > _AFFINITY_POLYMER_TOKENS
        ):
            break
        cropped |= new
    return cropped


def _symmetry_groups(slots: SlotMap, target: StructureTarget) -> tuple[Tensor, Tensor]:
    r"""The symmetry candidates on the slot axis: ``[G, W]`` slot indices and ``[G, P, W]`` permutations, identity
    first, ``-1`` padded; a candidate that moves a slot atom onto an atom without a slot is dropped."""
    candidates = target.symmetry
    slot_of = np.full(len(target.truth), -1, dtype=np.int64)
    slot_of[slots.atom[slots.atom >= 0]] = np.flatnonzero(slots.atom >= 0)
    groups: list[tuple[list[int], list[list[int]]]] = []
    for g in range(len(candidates.component)):
        atoms = candidates.atoms[candidates.atoms_ptr[g] : candidates.atoms_ptr[g + 1]]
        rows = candidates.permutation[candidates.permutation_ptr[g] : candidates.permutation_ptr[g + 1]]
        rows = rows.reshape(-1, len(atoms))
        visible = slot_of[atoms] >= 0
        if visible.sum() < 2:
            continue
        identity = slot_of[atoms[visible]].tolist()
        found = [identity]
        for row in rows:
            mapped = row[visible]
            if set(mapped.tolist()) == set(atoms[visible].tolist()):
                candidate = slot_of[mapped].tolist()
                if candidate not in found:
                    found.append(candidate)
        if len(found) > 1:
            groups.append((identity, found))
    width = max((len(identity) for identity, _ in groups), default=0)
    count = max((len(found) for _, found in groups), default=0)
    group_atoms = torch.full((len(groups), width), -1, dtype=torch.long)
    permutations = torch.full((len(groups), count, width), -1, dtype=torch.long)
    for g, (identity, found) in enumerate(groups):
        group_atoms[g, : len(identity)] = torch.tensor(identity)
        permutations[g, : len(found), : len(identity)] = torch.tensor(found)
    return group_atoms, permutations
