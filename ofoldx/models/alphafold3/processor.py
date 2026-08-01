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
AlphaFold 3 input processing, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own (layout, gating, crop, MSA pairing, template resolution,
reference conformers); this module keeps only the AlphaFold 3 featurisation of those inputs, which matches upstream
(``alphafold3/model/features.py``) for identical inputs: the MSA profile over each pool's unpaired rows,
query-only MSA columns for short and non-protein/RNA chains, per-chain template packing into four slots and the
upstream frame rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import BondKind, EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TOKENS, TokenLayout
from ...errors import RequestRef, StructureRef, fail
from ...modules.functional import coordinate_distogram
from ...msa import MSA, MsaCollection, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Records, known
from ...templates import Template, resolve_template, template_alignment_frames
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
from ..processing import project_model_coordinates
from ..targets import StructureTarget, SymmetryPolicy, TruthState, prepare_target
from .config import AlphaFold3ProcessorConfig
from .model import AlphaFold3Input, AlphaFold3StructureTarget, AlphaFold3TemplateFeatures

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA

ALPHAFOLD3_LAYOUT = LayoutTable(
    family="alphafold3",
    monomer_types={},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="nonstandard",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA", _RNA: "C1'", _DNA: "C1'"},
    frame_atoms={},
    representative={
        TOKEN_INDEX[token]: name
        for token, name in (("G", "CA"), *((f"{prefix}{base}", "C4") for prefix in ("RNA_", "DNA_") for base in "AG"))
    },
    representative_default={_PROTEIN: "CB", _RNA: "C2", _DNA: "C2"},
)
ALPHAFOLD3_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
ALPHAFOLD3_SYMMETRY = SymmetryPolicy(components="ligands", resonance=False, pin_unobserved=True)
#: The family's MSA depth constant (upstream ``msa_size``): the resource cap passed to ``pair_msas``.
ALPHAFOLD3_MSA_DEPTH = 16384
#: Upstream ``max_templates``: every chain has this many template slots, empty ones included.
ALPHAFOLD3_TEMPLATE_SLOTS = 4

# Upstream chain types (``constants/mmcif_names.py``).
#: Upstream ``STANDARD_POLYMER_CHAIN_TYPES``: protein, RNA and DNA chains.
ALPHAFOLD3_STANDARD_CHAINS = ("polypeptide(L)", "polyribonucleotide", "polydeoxyribonucleotide")
_MSA_CHAINS = ("polypeptide(L)", "polyribonucleotide")
# ``PEPTIDE_CHAIN_TYPES`` and ``NUCLEIC_ACID_CHAIN_TYPES``.
_FRAMED_CHAINS = (
    "cyclic-pseudo-peptide",
    "polypeptide(D)",
    "polypeptide(L)",
    "peptide nucleic acid",
    "polyribonucleotide",
    "polydeoxyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
)
_POLYMER_CHAINS = (*_FRAMED_CHAINS, "other")
_LIGAND_CHAINS = ("branched", "macrolide", "non-polymer")
_NON_POLYMER_CHAINS = (*_LIGAND_CHAINS, "water")
_ENTITY_CHAINS = {
    EntityType.NON_POLYMER: "non-polymer",
    EntityType.BRANCHED: "branched",
    EntityType.MACROLIDE: "macrolide",
    EntityType.WATER: "water",
}

_GAP = TOKEN_INDEX["-"]
# Template classes upstream reads as unknown (``_STANDARDIZED_AA``); its dense-atom row for UNK points every slot at N.
_TEMPLATE_UNKNOWN = [TOKEN_INDEX[letter] for letter in "XJO"]
# Upstream template distogram: 39 bins with lower edges over [3.25, 50.75] angstrom.
_TEMPLATE_DISTOGRAM = (3.25, 50.75, 39)
# Upstream clips norms at 1e-6 (``jax/geometry/vector.py``).
_NORM_EPSILON = 1e-6


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class AlphaFold3Processor(Processor):
    r"""
    Prepare one structure for AlphaFold 3.

    Args:
        config: Input-processing configuration. Defaults to ``AlphaFold3ProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = AlphaFold3ProcessorConfig
    config: AlphaFold3ProcessorConfig

    def __init__(
        self,
        config: AlphaFold3ProcessorConfig | None = None,
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
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[AlphaFold3Input]:
        r"""
        Prepare one structure: layout, optional training crop, slots, gating, reference conformers, MSA rows, template
        slots and the AlphaFold 3 tensors. The scope is every atom in a new frame.

        Args:
            structure: The target; its chemistry is consumed as read (no library).
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            templates: Template evidence; its targets are resolved against the structure (protein only).
            cropper: Optional training crop policy; needs ``generator``.
            generator: The numpy generator of the crop.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        layout = token_layout(system, ALPHAFOLD3_LAYOUT)
        if cropper is not None:
            layout = crop_token_layout(structure, layout, cropper, generator=generator)
        slots = slot_map(system, layout, ALPHAFOLD3_LAYOUT)
        accepted = gate(structure, slots, ALPHAFOLD3_GATING)
        references, sources, embedded = reference_conformers(
            structure, slots, embed=missing_reference(system), seed=self.seed
        )
        tokens, losses = model_tokens(system, layout)
        chain = alphafold3_chain_types(system)
        fed = _msa_fed(system, layout, chain)
        plan = _msa_plan(msa, system, layout, fed, cropped=cropper is not None)
        features = _features(
            system,
            slots,
            tokens,
            references,
            sources,
            chain,
            fed,
            plan,
            tuple(templates),
            device=self.device,
            dtype=self.dtype,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        conformers[_owner(system.component_atom_ptr)[slots.atom]] = sources
        return PreparedInput(
            features,
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

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[AlphaFold3Input],
    ) -> AlphaFold3StructureTarget:
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
            symmetry=ALPHAFOLD3_SYMMETRY,
        )
        return self._target_tensors(prepared, truth)

    def _target_tensors(
        self, prepared: PreparedInput[AlphaFold3Input], target: StructureTarget
    ) -> AlphaFold3StructureTarget:
        r"""The model's tensor target: canonical truth and its observed mask on the full atom axis."""
        observed = torch.from_numpy(target.truth == TruthState.OBSERVED)
        return AlphaFold3StructureTarget(  # type: ignore[call-arg]
            coordinates=project_model_coordinates(
                torch.from_numpy(target.coordinates.copy()), observed, feature="AlphaFold3 truth"
            )
            .to(self.device)
            .unsqueeze(0),
            coordinate_mask=observed.to(self.device).unsqueeze(0),
            batch_size=[1],
        )


def alphafold3_chain_types(system: MolecularSystem) -> NDArray[np.str_]:
    r"""
    Per instance, its chain type as upstream names it: the stated polymer type of a polymer entity, else the entity
    type (``non-polymer``, ``branched``, ``macrolide``, ``water``).

    An instance whose entity type, or whose polymer entity's polymer type, is not KNOWN has no chain type to featurise
    or score it by: ``InsufficientInformationError(prepare_unknown_feature)`` naming the instance.
    """
    s = system.arrays()
    entity = s["instance_entity"]
    stated = known(s["entity_type_state"])
    polymer = stated & (s["entity_type"] == EntityType.POLYMER)
    for column, missing in (
        ("entity_type", ~stated),
        ("entity_polymer_type", polymer & ~known(s["entity_polymer_type_state"])),
    ):
        if missing[entity].any():
            fail("prepare_unknown_feature", StructureRef(instance=int(np.argmax(missing[entity]))), column=column)
    polymer_type = system.text("polymer_type", np.where(polymer, s["entity_polymer_type"], -1)).tolist()
    kinds = s["entity_type"].tolist()
    names = [polymer_type[e] if polymer[e] else _ENTITY_CHAINS[EntityType(k)] for e, k in enumerate(kinds)]
    return np.array(names, dtype=np.str_)[entity]


def _msa_fed(system: MolecularSystem, layout: TokenLayout, chain: NDArray[np.str_]) -> NDArray[np.bool_]:
    r"""Protein/RNA instances with more than four residues represented in the layout."""
    pointer = system.instance_component_ptr
    residues = np.bincount(_owner(pointer)[np.unique(layout.token_component)], minlength=len(pointer) - 1)
    return np.isin(chain, _MSA_CHAINS) & (residues > 4)


def _msa_plan(
    msa: Mapping[str, MSA] | MsaCollection | None,
    system: MolecularSystem,
    layout: TokenLayout,
    fed: NDArray[np.bool_],
    *,
    cropped: bool,
) -> MsaCollection | None:
    r"""Pair eligible pools or use an explicit plan unchanged."""
    if msa is None:
        return None
    if isinstance(msa, MsaCollection):
        if cropped:
            fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
        if msa.row_pairing.shape[0] > ALPHAFOLD3_MSA_DEPTH:
            fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=ALPHAFOLD3_MSA_DEPTH)
        return msa
    names = set(system.text("instance", system.instance_symbol[fed]).tolist())
    pools = {name: msa[name] for name in msa if name in names}
    return pair_msas(pools, system, layout, max_rows=ALPHAFOLD3_MSA_DEPTH)


def _msa_features(
    plan: MsaCollection | None,
    system: MolecularSystem,
    layout: TokenLayout,
    chain: NDArray[np.str_],
    fed: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    MSA rows, deletion counts, profile and deletion mean as upstream featurises a given MSA
    (``MSA.compute_features``).

    A token of a polymer-(L), RNA or DNA chain reads its row 0 class; any other token (ligands, other polymer types)
    reads a gap in every row. Rows below 0 come from the plan only for fed instances, else they are gaps (the plan
    may cover an instance that is not fed). Deletions are clipped to 127 (upstream's int8 cast). The profile and
    the raw deletion mean of a fed token are over its pool's unpaired rows as given, read through ``msa_rows``; every
    other token's profile is the one-hot of its row 0 and its deletion mean zero.
    """

    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    standard = np.isin(chain, ALPHAFOLD3_STANDARD_CHAINS)[token_instance]
    fed = fed[token_instance]
    if plan is None:
        rows = layout.token_class[None].astype(np.int64)
        deletions = np.zeros(rows.shape, dtype=np.int32)
    else:
        spread = msa_rows(plan, system, layout)
        rows = spread.token.astype(np.int64)
        deletions = np.minimum(spread.deletions, 127).astype(np.int32)
    rows[1:, ~fed] = _GAP
    deletions[:, ~fed] = 0
    rows[:, ~standard] = _GAP
    size = len(TOKENS)
    profile = np.eye(size)[rows[0]]
    deletion_mean = np.zeros(len(token_instance))
    target_instance_ids = None if plan is None else plan.target_instance_ids
    for c, pool in enumerate(plan.msas if plan is not None else ()):
        # The pool's unpaired rows as a one-column plan: row 1 (its query) is never a gap on a token it covers.
        depth = len(pool.unpaired_sequences)
        targets = None if target_instance_ids is None else (target_instance_ids[c],)
        unpaired = MsaCollection(
            (pool,),
            np.arange(-1, depth).clip(0)[:, None],
            (np.arange(depth + 1) == 0)[:, None],
            targets,
        )
        spread = msa_rows(unpaired, system, layout)
        chosen = np.flatnonzero(fed & (spread.token[1] != _GAP))
        if not len(chosen):
            continue
        codes = spread.token[1:, chosen].astype(np.int64) + size * np.arange(len(chosen))
        profile[chosen] = np.bincount(codes.ravel(), minlength=size * len(chosen)).reshape(len(chosen), size) / depth
        deletion_mean[chosen] = spread.deletions[1:, chosen].mean(axis=0)
    return rows, deletions, profile, deletion_mean


def _normalized(vectors: Tensor) -> Tensor:
    r"""Unit vectors with the norm clipped at ``1e-6``, as upstream's ``Vec3Array.normalized``."""
    return vectors / vectors.square().sum(dim=-1, keepdim=True).clamp_min(_NORM_EPSILON**2).sqrt()


def _template_features(
    templates: tuple[Template, ...], system: MolecularSystem, layout: TokenLayout, chain: NDArray[np.str_]
) -> dict[str, Tensor]:
    r"""
    The template slots as upstream packs and featurises them (``Templates.compute_features``,
    ``template_modules.py``).

    Every chain has ``ALPHAFOLD3_TEMPLATE_SLOTS`` slots; slot ``k`` of a polymer-(L) chain of more than four tokens is
    the ``k``-th template (in order) resolving onto it, its aligned positions carrying the template residue class and
    N, CA, C, CB, the others a gap. A residue of class X, J or O (upstream's UNK) reads N in every atom slot, as
    upstream's UNK dense-atom row: its pseudo-beta is N and its frame degenerate (zero unit vectors). Empty slots and
    other chains read class ``A`` with no atom, as upstream's zero padding. Pair features are kept within one
    placement frame; the backbone frame needs N, CA and C present (no degeneracy test) and the unit vectors clip norms
    at 1e-6.
    """

    slots, count = ALPHAFOLD3_TEMPLATE_SLOTS, len(layout.token_component)
    pointer = system.instance_component_ptr
    token_instance = _owner(pointer)[layout.token_component]
    token_position = layout.token_component - pointer[token_instance]
    eligible = (chain == "polypeptide(L)") & (np.bincount(token_instance, minlength=len(pointer) - 1) > 4)
    residue = np.full((slots, count), TOKEN_INDEX["A"], dtype=np.int64)
    positions = np.zeros((slots, count, 4, 3))
    present = np.zeros((slots, count, 4), dtype=bool)
    frame = np.full((slots, count), -1, dtype=np.int64)
    used = np.zeros(len(pointer) - 1, dtype=np.int64)
    offset = 0
    for template in templates:
        resolved = resolve_template(template, system, monomer_types=(_PROTEIN,))
        frames = template_alignment_frames(resolved)
        slot_of: dict[int, int | None] = {}
        for alignment, placement in zip(resolved, frames):
            instance = alignment.target_instance_index
            if instance not in slot_of:
                free = eligible[instance] and used[instance] < slots
                slot_of[instance] = int(used[instance]) if free else None
                if free:
                    residue[used[instance], token_instance == instance] = _GAP
                    used[instance] += 1
            k = slot_of[instance]
            if k is None:
                continue
            source = np.full(pointer[instance + 1] - pointer[instance], -1, dtype=np.int64)
            target, aligned = np.array(alignment.residue_pairs, dtype=np.int64).T
            source[target] = aligned
            tokens = np.flatnonzero(token_instance == instance)
            tokens = tokens[source[token_position[tokens]] >= 0]
            hit = source[token_position[tokens]]
            chain_coordinates = template.coordinates.chains[alignment.template_instance_id]
            positions[k, tokens], present[k, tokens] = chain_coordinates.gather(hit.tolist(), ("N", "CA", "C", "CB"))
            residue[k, tokens] = chain_coordinates.token_class[hit]
            unknown = tokens[np.isin(residue[k, tokens], _TEMPLATE_UNKNOWN)]
            positions[k, unknown], present[k, unknown] = positions[k, unknown, :1], present[k, unknown, :1]
            frame[k, tokens] = offset + placement
        offset += max(frames, default=-1) + 1

    mask = torch.from_numpy(present)
    coordinates = torch.from_numpy(positions) * mask[..., None]
    nitrogen, alpha, carbon, beta = coordinates.unbind(dim=-2)
    glycine = torch.from_numpy(residue == TOKEN_INDEX["G"])
    pseudo_beta = torch.where(glycine[..., None], alpha, beta)
    pseudo_beta_mask = torch.where(glycine, mask[..., 1], mask[..., 3])
    frame_ids = torch.from_numpy(frame)
    same_frame = (frame_ids[..., :, None] == frame_ids[..., None, :]) & (frame_ids[..., :, None] >= 0)
    pseudo_beta_mask_2d = pseudo_beta_mask[..., :, None] & pseudo_beta_mask[..., None, :] & same_frame
    min_distance, max_distance, num_bins = _TEMPLATE_DISTOGRAM
    distogram = (
        coordinate_distogram(pseudo_beta, min_distance=min_distance, max_distance=max_distance, num_bins=num_bins)
        * pseudo_beta_mask_2d[..., None]
    )
    x_axis = _normalized(carbon - alpha)
    xy = nitrogen - alpha
    y_axis = _normalized(xy - (xy * x_axis).sum(dim=-1, keepdim=True) * x_axis)
    rotation = torch.stack((x_axis, y_axis, torch.linalg.cross(x_axis, y_axis)), dim=-1)
    local = torch.einsum("tinm,tijn->tijm", rotation, alpha[..., None, :, :] - alpha[..., :, None, :])
    backbone_mask = mask[..., 0] & mask[..., 1] & mask[..., 2]
    backbone_mask_2d = backbone_mask[..., :, None] & backbone_mask[..., None, :] & same_frame
    return {
        "distogram": distogram,
        "pseudo_beta_mask": pseudo_beta_mask_2d.double(),
        "residue_type_ids": torch.from_numpy(residue),
        "unit_vector": _normalized(local) * backbone_mask_2d[..., None],
        "backbone_frame_mask": backbone_mask_2d.double(),
    }


def _frame_mask(
    system: MolecularSystem, layout: TokenLayout, centres: NDArray[np.float64], chain: NDArray[np.str_]
) -> NDArray[np.bool_]:
    r"""
    Per token, whether it has a confidence frame (upstream ``Frames.compute_features``): every token of a peptide or
    nucleic chain; a ligand token when its component has at least three tokens and its two nearest tokens (by
    reference distance) are not colinear with it (``1 - |cos| >= 0.0937``, identical points colinear); no other
    token.
    """

    token_chain = chain[_owner(system.instance_component_ptr)[layout.token_component]]
    mask = np.isin(token_chain, _FRAMED_CHAINS)
    points = centres.astype(np.float32)
    for component in np.unique(layout.token_component[np.isin(token_chain, _NON_POLYMER_CHAINS)]).tolist():
        members = np.flatnonzero(layout.token_component == component)
        if len(members) < 3:
            continue
        local = points[members]
        cost = np.linalg.norm(local[:, None] - local[None], axis=-1) + 1e8 * np.eye(len(members), dtype=np.float32)
        nearest = np.argsort(cost, axis=1)[:, :2]
        first, second = local[nearest[:, 0]] - local, local[nearest[:, 1]] - local
        first_norm, second_norm = np.linalg.norm(first, axis=-1), np.linalg.norm(second, axis=-1)
        degenerate = np.isclose(first_norm, 0) | np.isclose(second_norm, 0)
        safe_first, safe_second = np.where(degenerate, 1, first_norm), np.where(degenerate, 1, second_norm)
        cosine = ((first / safe_first[:, None]) * (second / safe_second[:, None])).sum(axis=-1)
        mask[members] = ~degenerate & (1 - np.abs(cosine) >= 0.0937)
    return mask


def _token_bonds(system: MolecularSystem, layout: TokenLayout, chain: NDArray[np.str_]) -> Tensor:
    r"""
    ``[N, N]``: upstream's polymer-ligand and ligand-ligand bond contacts (``inter_chain_bonds.get_bond_layout``,
    ``evoformer.py``). A KNOWN covalent edge between two components, one of a ligand chain (branched, macrolide,
    non-polymer) and the other of a ligand or polymer chain, sets 1 at its endpoints' tokens; intra-component,
    polymer-polymer and water edges set nothing. One direction per bond, in edge order (upstream's is its source
    order).
    """
    s = system.arrays()
    first, second = s["edge_index"].astype(np.int64)
    component = _owner(s["component_atom_ptr"])
    left, right = component[first], component[second]
    component_chain = chain[_owner(s["instance_component_ptr"])]
    ligand, polymer = np.isin(component_chain, _LIGAND_CHAINS), np.isin(component_chain, _POLYMER_CHAINS)
    partners = (ligand[left] & (ligand[right] | polymer[right])) | (polymer[left] & ligand[right])
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = covalent & partners & (left != right) & (a >= 0) & (b >= 0)
    bonds = torch.zeros(len(layout.token_component), len(layout.token_component))
    bonds[torch.from_numpy(a[linked]), torch.from_numpy(b[linked])] = 1
    return bonds


def _features(
    system: MolecularSystem,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    references: NDArray[np.float64],
    sources: NDArray[np.uint8],
    chain: NDArray[np.str_],
    fed: NDArray[np.bool_],
    plan: MsaCollection | None,
    templates: tuple[Template, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold3Input:
    r"""The AlphaFold 3 tensors of one prepared task (upstream featurisation over the slot axis)."""
    s = system.arrays()
    layout = slots.layout
    atom = slots.atom
    num_atoms, num_tokens = len(atom), len(layout.token_component)
    atom_component = _owner(s["component_atom_ptr"])
    token_instance = _owner(s["instance_component_ptr"])[layout.token_component]
    entity = s["instance_entity"]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[atom] = np.arange(num_atoms)

    # Reference positions are centred per component (one reference space each).
    space = atom_component[atom]
    centre = np.zeros((len(s["component_symbol"]), 3))
    np.add.at(centre, space, references)
    centre /= np.maximum(np.bincount(space, minlength=len(centre)), 1)[:, None]
    reference_mask = torch.from_numpy(sources != ConformerSource.NONE)
    names = system.text("atom", s["atom_symbol"][atom]).tolist()
    characters = torch.tensor(
        [[min(63, max(0, ord(c) - 32)) for c in name.ljust(4)[:4]] for name in names], dtype=torch.long
    ).reshape(num_atoms, 4)
    atomic_number = torch.from_numpy(s["atomic_number"][atom].astype(np.int64))
    atom_features = torch.cat(
        (
            reference_mask[:, None].float(),
            F.one_hot(atomic_number, 128).float(),
            F.one_hot(characters, 64).reshape(num_atoms, 256).float(),
        ),
        dim=-1,
    )
    rows, deletions, profile, deletion_mean = _msa_features(plan, system, layout, chain, fed)
    token_features = torch.cat(
        (
            torch.zeros(num_tokens, 384),
            F.one_hot(torch.from_numpy(tokens), len(TOKENS)).float(),
            torch.from_numpy(profile).float(),
            torch.from_numpy(deletion_mean).float()[:, None],
        ),
        dim=-1,
    )
    frame_mask = _frame_mask(system, layout, references[slot_of[layout.token_centre_atom]], chain)

    def floating(value: Tensor) -> Tensor:
        return value.to(device=device, dtype=dtype).unsqueeze(0)

    def discrete(value: Tensor | NDArray) -> Tensor:
        value = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        return value.to(device=device).unsqueeze(0)

    template = _template_features(templates, system, layout, chain)
    return AlphaFold3Input(  # type: ignore[call-arg]
        token_mask=discrete(torch.ones(num_tokens, dtype=torch.bool)),
        frame_mask=discrete(frame_mask),
        residue_index=discrete(layout.token_component - s["instance_component_ptr"][token_instance]),
        chain_index=discrete(token_instance),
        entity_index=discrete(entity[token_instance].astype(np.int64)),
        symmetry_index=discrete(copy[token_instance]),
        token_index=discrete(torch.arange(num_tokens)),
        atom_features=floating(atom_features),
        atom_to_token=discrete(slots.token),
        atom_mask=discrete(torch.ones(num_atoms, dtype=torch.bool)),
        reference_positions=floating(torch.from_numpy(references - centre[space])),
        reference_space_id=discrete(space),
        atom_token_index=discrete(np.arange(num_atoms) - layout.token_atom_ptr[slots.token]),
        token_representative_atom_index=discrete(slot_of[token_representatives(system, layout, ALPHAFOLD3_LAYOUT)]),
        token_features=floating(token_features),
        reference_mask=discrete(reference_mask),
        reference_charges=floating(torch.from_numpy(s["formal_charge"][atom].astype(np.float64))[:, None]),
        token_bonds=floating(_token_bonds(system, layout, chain)),
        msa_tokens=discrete(rows),
        msa_deletion_counts=discrete(deletions),
        msa_mask=discrete(torch.ones(rows.shape, dtype=torch.bool)),
        template_features=AlphaFold3TemplateFeatures(
            distogram=floating(template["distogram"]),
            pseudo_beta_mask=floating(template["pseudo_beta_mask"]),
            residue_type_ids=discrete(template["residue_type_ids"]),
            unit_vector=floating(template["unit_vector"]),
            backbone_frame_mask=floating(template["backbone_frame_mask"]),
            batch_size=[1],
        ),
        atom_index=discrete(atom.copy()),
        batch_size=[],
    )
