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
AlphaFold 2 input processing, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own: the residue layout (the atom37 slots of every protein
component holding a CA; a cap such as ACE is no residue), gating, the crop, MSA pairing and binding (``ofoldx.msa``)
and template target resolution (``ofoldx.templates``). This module keeps only the AlphaFold 2 featurisation of those
inputs — target features, the MSA pool and its profile, template atom37 coordinates and torsions — which matches
upstream for identical inputs. Structure prediction regenerates every coordinate in a new frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from ...data import residue_constants
from ...data.crop import Cropper
from ...data.molecular import MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import TokenLayout, select_tokens
from ...errors import RequestRef, StructureRef, fail
from ...modules.geometry import atom37_to_torsion_angles, virtual_beta_carbon
from ...msa import MSA, MsaCollection, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Records
from ...templates import Template, resolve_template
from ..layout import LayoutTable, model_tokens, slot_map, token_layout
from ..preparation import (
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
)
from .config import AlphaFold2MultimerProcessorConfig, AlphaFold2ProcessorConfig
from .model import AlphaFold2InitialGuessInput, AlphaFold2Input, AlphaFold2MultimerInput
from .vocabulary import HHBLITS_CLASS, NUM_CLASSES, RESIDUE_TYPE, target_class

_PROTEIN = MonomerType.PROTEIN
_ATOMS = residue_constants.NUM_ATOM_TYPES
_GAP_TYPE = residue_constants.NUM_AMINO_ACID_TYPES + 1  # the template residue type of an unaligned position
_SD = residue_constants.atom_order["SD"]

ALPHAFOLD2_LAYOUT = LayoutTable(
    family="alphafold2",
    monomer_types={_PROTEIN: 0},
    residue_slots=None,
    monomer_slots={_PROTEIN: tuple(residue_constants.atom_types)},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: AlphaFold 2 reads each residue's class (and, for the initial guess, its placed coordinates); no hydrogen count,
#: bond, charge or edge kind of an atom is its input.
ALPHAFOLD2_GATING = GatingPolicy(
    refuse_undetermined_in_slots=False,
    refuse_unknown_kind_in_slots=False,
    refuse_incomplete_topology_in_slots=False,
    refuse_valence_excess=False,
)
#: AlphaFold-Multimer's MSA depth constant (upstream ``MSA_CROP_SIZE``): the resource cap passed to ``pair_msas``.
ALPHAFOLD2_MULTIMER_MSA_DEPTH = 2048


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _residue_layout(system: MolecularSystem) -> TokenLayout:
    r"""One token per protein component holding a CA, its atom37 atoms addressed (upstream reads residues only)."""
    component = _owner(system.component_atom_ptr)
    holds = np.zeros(len(system.component_symbol), dtype=bool)
    holds[component[system.text("atom", system.atom_symbol) == "CA"]] = True
    return token_layout(system, ALPHAFOLD2_LAYOUT, atom_mask=holds[component])


def _crop(
    structure: Structure,
    layout: TokenLayout,
    cropper: Cropper | None,
    generator: np.random.Generator | None,
    required: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    r"""The tokens a training crop keeps (every token without a crop); a crop needs a numpy generator."""
    if cropper is None:
        return np.ones(len(layout.token_component), dtype=bool)
    if generator is None:
        fail("request_invalid_generator", RequestRef("generator", "None"))
    return cropper(structure, layout, required=required, generator=generator)


def _token_instance(system: MolecularSystem, layout: TokenLayout) -> NDArray[np.int64]:
    return _owner(system.instance_component_ptr)[layout.token_component]


def _residue_index(system: MolecularSystem, layout: TokenLayout) -> NDArray[np.int64]:
    r"""Each token's component position within its instance."""
    return layout.token_component - system.instance_component_ptr[_token_instance(system, layout)]


def _msa_tokens(
    msa: Mapping[str, MSA] | MsaCollection | None,
    system: MolecularSystem,
    layout: TokenLayout,
    *,
    cropped: bool,
    depth: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    r"""
    The MSA pool on the token axis as the upstream featuriser reads it (HHblits letter aliases) and its deletion
    counts. Pools are paired by the one OFoldX rule after the crop; an exact plan is used as given; without
    an MSA the pool is the query row.
    """

    rows = None
    if isinstance(msa, MsaCollection):
        if cropped:
            fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
        if depth is not None and msa.row_pairing.shape[0] > depth:
            fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=depth)
        rows = msa_rows(msa, system, layout)
    elif msa is not None:
        plan = pair_msas(msa, system, layout, max_rows=depth)
        rows = None if plan is None else msa_rows(plan, system, layout)
    if rows is None:
        return HHBLITS_CLASS[layout.token_class[None].astype(np.int64)], np.zeros((1, len(layout.token_class)), "f4")
    return HHBLITS_CLASS[rows.token.astype(np.int64)], rows.deletions.astype(np.float32)


def _template_atoms(
    templates: Sequence[Template], system: MolecularSystem, layout: TokenLayout
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    r"""
    Per template on the token axis: the upstream residue type of the aligned template position (HHblits letter
    aliases; a gap where nothing is aligned), its atom37 coordinates and their presence. Targets are resolved once by
    ``resolve_template``; positions outside the layout (a cap, a cropped residue) are dropped.
    """

    count = len(layout.token_component)
    token_of = np.full(len(system.component_symbol), -1, dtype=np.int64)
    token_of[layout.token_component] = np.arange(count)
    pointer = system.instance_component_ptr
    types = np.full((len(templates), count), _GAP_TYPE, dtype=np.int64)
    positions = np.zeros((len(templates), count, _ATOMS, 3))
    present = np.zeros((len(templates), count, _ATOMS), dtype=bool)
    for t, template in enumerate(templates):
        for alignment in resolve_template(template, system, monomer_types=(_PROTEIN,)):
            chain = template.coordinates.chains[alignment.template_instance_id]
            pairs = np.array(alignment.residue_pairs, dtype=np.int64)
            tokens = token_of[pointer[alignment.target_instance_index] + pairs[:, 0]]
            source, tokens = pairs[tokens >= 0, 1], tokens[tokens >= 0]
            types[t, tokens] = RESIDUE_TYPE[HHBLITS_CLASS[chain.token_class[source].astype(np.int64)]]
            positions[t, tokens], present[t, tokens] = chain.gather(source, residue_constants.atom_types)
            selenium = np.array([chain.component_ids[p] == "MSE" for p in source.tolist()], dtype=bool)
            if selenium.any():  # upstream puts an MSE's SE in the SD column
                se, seen = chain.gather(source[selenium], ("SE",))
                positions[t, tokens[selenium], _SD], present[t, tokens[selenium], _SD] = se[:, 0], seen[:, 0]
    return types, positions, present


def _template_torsions(types: Tensor, positions: Tensor, present: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    r"""Template torsions and their alternatives (upstream ``atom37_to_torsion_angles``), on the axis given."""
    table = types.clamp(max=residue_constants.NUM_AMINO_ACID_TYPES)
    unknown = ((0.0, 0.0, 0.0, 0.0),)
    return atom37_to_torsion_angles(
        positions,
        present,
        chi_atom_indices=torch.as_tensor(residue_constants.AMINO_ACID_CHI_ATOM_INDICES)[table],
        chi_angles_mask=torch.tensor((*residue_constants.chi_angles_mask, *unknown), dtype=torch.bool)[table],
        chi_pi_periodic=torch.tensor((*residue_constants.chi_pi_periodic, *unknown), dtype=torch.bool)[table],
    )


def _record(system: MolecularSystem, layout: TokenLayout, accepted: Records) -> PreparationRecord:
    _, losses = model_tokens(system, layout)
    read_as_x = layout.token_component[target_class(layout.token_class) != layout.token_class]  # e.g. U, O
    lossy = np.union1d(losses.component, read_as_x)
    records = Records.concat((component_records("encoding_loss", lossy), accepted))
    return PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))


def _too_many_templates(templates: Sequence[Template], limit: int) -> None:
    if len(templates) > limit:
        fail("model_capacity", RequestRef("templates", str(len(templates))), limit=limit)


class AlphaFold2Processor(Processor):
    r"""
    Prepare one protein chain for AlphaFold 2.

    Args:
        config: Template limits. Defaults to ``AlphaFold2ProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = AlphaFold2ProcessorConfig
    config: AlphaFold2ProcessorConfig

    def __init__(
        self,
        config: AlphaFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def prepare_input(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[AlphaFold2Input]:
        r"""
        Prepare one structure whose residues form one protein chain: layout, optional training crop, slots (the atom37
        names of each residue), gating, the MSA pool and the template features. The scope is every atom in a new frame.

        Args:
            structure: The target; components without a protein residue token (ligands, water, caps) are not inputs
                and lose their coordinates on decode.
            msa: MSA pools by instance (paired after the crop), an exact plan, or ``None``.
            templates: Templates with explicit target alignments.
            cropper: Optional training crop over the residue tokens; needs ``generator``.
            generator: The numpy generator of the crop.
        """

        _too_many_templates(templates, self.config.max_templates)
        system = structure.system
        full = _residue_layout(system)
        instances = np.unique(_token_instance(system, full))
        if len(instances) != 1:
            fail("model_capacity", *(StructureRef(instance=int(i)) for i in instances), instances=len(instances))
        keep = _crop(structure, full, cropper, generator)
        layout = select_tokens(full, keep)
        slots = slot_map(system, layout, ALPHAFOLD2_LAYOUT)
        accepted = gate(structure, slots, ALPHAFOLD2_GATING)
        msa_tokens, deletions = _msa_tokens(msa, system, layout, cropped=cropper is not None, depth=None)
        templated = _monomer_templates(*_template_atoms(templates, system, full), keep)
        features = _monomer_input(layout, _residue_index(system, layout), msa_tokens, deletions, templated, self)
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                _record(system, layout, accepted),
                spec=self.preparation_spec,
            ),
        )


def _monomer_templates(
    types: NDArray[np.int64], positions: NDArray[np.float64], present: NDArray[np.bool_], keep: NDArray[np.bool_]
) -> dict[str, Tensor]:
    r"""
    The monomer template features of the kept tokens; torsions are computed on the uncropped axis first, since they
    depend on sequence neighbours. Without a template, one masked all-gap template.
    """

    if not len(types):
        count = int(keep.sum())
        return {
            "template_mask": torch.zeros(1),
            "template_residue_type_ids": torch.full((1, count), _GAP_TYPE, dtype=torch.long),
            "template_all_atom_positions": torch.zeros(1, count, _ATOMS, 3),
            "template_all_atom_mask": torch.zeros(1, count, _ATOMS),
            "template_torsion_angles_sin_cos": torch.zeros(1, count, 7, 2),
            "template_alternative_torsion_angles_sin_cos": torch.zeros(1, count, 7, 2),
            "template_torsion_angles_mask": torch.zeros(1, count, 7),
        }
    types_, positions_, present_ = torch.from_numpy(types), torch.from_numpy(positions), torch.from_numpy(present)
    torsion, alternative, torsion_mask = _template_torsions(types_, positions_, present_)
    kept = torch.from_numpy(keep)
    return {
        "template_mask": torch.ones(len(types)),
        "template_residue_type_ids": types_[:, kept],
        "template_all_atom_positions": positions_[:, kept],
        "template_all_atom_mask": present_[:, kept].float(),
        "template_torsion_angles_sin_cos": torsion[:, kept],
        "template_alternative_torsion_angles_sin_cos": alternative[:, kept],
        "template_torsion_angles_mask": torsion_mask[:, kept].float(),
    }


def _monomer_input(
    layout: TokenLayout,
    residue_index: NDArray[np.int64],
    msa_tokens: NDArray[np.int64],
    deletions: NDArray[np.float32],
    templated: Mapping[str, Tensor],
    processor: Processor,
) -> AlphaFold2Input:
    r"""The monomer tensors (upstream featurisation): target features, the MSA pool and its full-pool profile, and
    the template features, placed on the processor's device and dtype."""

    classes = torch.from_numpy(target_class(layout.token_class))
    tokens = torch.from_numpy(msa_tokens)
    count = len(classes)
    floating = {
        "token_features": torch.cat((torch.zeros(count, 1), F.one_hot(classes, NUM_CLASSES).float()), dim=-1),
        "residue_mask": torch.ones(count),
        "msa_deletion_counts": torch.from_numpy(deletions),
        "msa_profile": F.one_hot(tokens, NUM_CLASSES).float().mean(dim=0),
        **{name: value for name, value in templated.items() if value.is_floating_point()},
    }
    integral = {
        "residue_type_ids": torch.from_numpy(RESIDUE_TYPE[classes.numpy()]),
        "residue_index": torch.from_numpy(residue_index),
        "msa_tokens": tokens,
        "template_residue_type_ids": templated["template_residue_type_ids"],
    }
    device, dtype = processor.device, processor.dtype
    return AlphaFold2Input(  # type: ignore[call-arg]
        **{name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()},
        **{name: value.to(device=device).unsqueeze(0) for name, value in integral.items()},
        batch_size=[],
    )


class AlphaFold2InitialGuessProcessor(Processor):
    r"""
    Prepare an in-memory target--binder structure for AF2 initial-guess evaluation.

    The released path uses the pTM monomer graph over every protein chain, query-only MSA features, the input atom37
    coordinates as the first recycled ``prev_pos`` state, and one all-GAP backbone/CB template whose target--binder
    pair features are masked by the Model. Residue indices keep every chain break (target chains in system order,
    then the binder, 50 apart), while the asymmetric-unit index groups all target chains apart from the binder. It does
    not initialise Structure Module frames.

    Args:
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def prepare_input(
        self,
        structure: Structure,
        *,
        binder_instance_id: str,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[AlphaFold2InitialGuessInput]:
        r"""
        Prepare the released target--binder pTM evaluation input.

        Args:
            structure: In-memory structure holding the target chains and the designed binder.
            binder_instance_id: The protein instance treated as the binder.
            cropper: Optional training crop; one target and one binder residue, drawn from ``generator``, are kept.
            generator: The numpy generator of the crop and its role anchors.
        """

        system = structure.system
        full = _residue_layout(system)
        instance = _token_instance(system, full)
        binder = int(system.instance_index([binder_instance_id])[0])
        role = instance == binder
        if not role.any() or role.all():  # a binder residue and a target residue are needed
            fail("request_invalid_positions", RequestRef("binder_instance_id", binder_instance_id))
        required = None
        if cropper is not None and generator is not None:
            required = np.zeros(len(role), dtype=bool)
            for candidates in (np.flatnonzero(~role), np.flatnonzero(role)):
                required[generator.choice(candidates)] = True
        keep = _crop(structure, full, cropper, generator, required)
        layout = select_tokens(full, keep)
        slots = slot_map(system, layout, ALPHAFOLD2_LAYOUT)
        accepted = gate(structure, slots, ALPHAFOLD2_GATING)
        msa_tokens, deletions = _msa_tokens(None, system, layout, cropped=False, depth=None)
        empty = np.zeros((0, len(keep)), dtype=np.int64)
        templated = _monomer_templates(empty, np.zeros((0, len(keep), _ATOMS, 3)), empty.astype(bool), keep)
        residue_index = _chain_break_index(system, full, role)[keep]
        base = _monomer_input(layout, residue_index, msa_tokens, deletions, templated, self)
        positions, present = _initial_positions(structure, slots)
        template_present = present.clone()
        template_present[:, 5:] = False  # the backbone and CB template (N, CA, C, CB, O)
        template_positions = positions * template_present[..., None]
        count = len(layout.token_component)
        values = {field.name: getattr(base, field.name) for field in fields(base)}  # type: ignore[arg-type]
        values.update(
            template_mask=torch.ones(1, 1, dtype=self.dtype, device=self.device),
            template_residue_type_ids=torch.full((1, 1, count), _GAP_TYPE, dtype=torch.long, device=self.device),
            template_all_atom_positions=template_positions.to(self.device, self.dtype)[None, None],
            template_all_atom_mask=template_present.to(self.device, self.dtype)[None, None],
            initial_recycling_positions=positions.to(self.device, self.dtype)[None],
            chain_index=torch.from_numpy(role[keep].astype(np.int64)).to(self.device)[None],
            binder_mask=torch.from_numpy(role[keep]).to(self.device)[None],
        )
        features = AlphaFold2InitialGuessInput(**values, batch_size=[])  # type: ignore[call-arg]
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                _record(system, layout, accepted),
                spec=self.preparation_spec,
            ),
        )


def _chain_break_index(system: MolecularSystem, layout: TokenLayout, binder: NDArray[np.bool_]) -> NDArray[np.int64]:
    r"""Residue indices with chain breaks: the target chains in system order, then the binder, each starting 50 after
    the previous chain's last residue."""
    instance = _token_instance(system, layout)
    position = _residue_index(system, layout)
    chains = list(dict.fromkeys(instance[~binder].tolist())) + list(dict.fromkeys(instance[binder].tolist()))
    start, offset = {}, 0
    for chain in chains:
        start[chain] = offset
        offset += int(position[instance == chain].max()) + 50
    return np.array([start[i] for i in instance.tolist()], dtype=np.int64) + position


def _initial_positions(structure: Structure, slots: SlotMap) -> tuple[Tensor, Tensor]:
    r"""The placed atom37 coordinates of each residue (``[N, 37, 3]``, presence ``[N, 37]``), with a virtual CB where N,
    CA and C are placed and CB is not."""
    atom = slots.atom.reshape(-1, _ATOMS)
    present = torch.from_numpy((atom >= 0) & structure.coordinate_mask[np.maximum(atom, 0)])
    positions = torch.from_numpy(structure.coordinates[np.maximum(atom, 0)]) * present[..., None]
    n, ca, c, cb = (residue_constants.atom_order[name] for name in ("N", "CA", "C", "CB"))
    rebuild = present[:, n] & present[:, ca] & present[:, c] & ~present[:, cb]
    positions[rebuild, cb] = virtual_beta_carbon(positions[rebuild, n], positions[rebuild, ca], positions[rebuild, c])
    present[rebuild, cb] = True
    return positions, present


class AlphaFold2MultimerProcessor(Processor):
    r"""
    Prepare a protein assembly for AlphaFold-Multimer.

    Args:
        config: Template limits. Defaults to ``AlphaFold2MultimerProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = AlphaFold2MultimerProcessorConfig
    config: AlphaFold2MultimerProcessorConfig

    def __init__(
        self,
        config: AlphaFold2MultimerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2MultimerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def prepare_input(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[AlphaFold2MultimerInput]:
        r"""
        Prepare one structure: every protein chain's residues on one axis, optional training crop, slots, gating, the
        paired MSA pool and the template features. The scope is every atom in a new frame.

        Args:
            structure: The target; components without a protein residue token are not inputs.
            msa: MSA pools by instance (paired after the crop, depth ``ALPHAFOLD2_MULTIMER_MSA_DEPTH``), an exact
                plan, or ``None``.
            templates: Templates with explicit target alignments.
            cropper: Optional training crop over the residue tokens; needs ``generator``.
            generator: The numpy generator of the crop.
        """

        _too_many_templates(templates, self.config.max_templates)
        system = structure.system
        full = _residue_layout(system)
        layout = select_tokens(full, _crop(structure, full, cropper, generator))
        slots = slot_map(system, layout, ALPHAFOLD2_LAYOUT)
        accepted = gate(structure, slots, ALPHAFOLD2_GATING)
        msa_tokens, deletions = _msa_tokens(
            msa, system, layout, cropped=cropper is not None, depth=ALPHAFOLD2_MULTIMER_MSA_DEPTH
        )
        types, positions, present = _template_atoms(templates, system, layout)
        if not len(templates):
            count = len(layout.token_component)
            types = np.full((1, count), _GAP_TYPE, dtype=np.int64)
            positions, present = np.zeros((1, count, _ATOMS, 3)), np.zeros((1, count, _ATOMS), dtype=bool)
        instance = _token_instance(system, layout)
        chains = list(dict.fromkeys(instance.tolist()))
        entity = system.instance_entity.astype(np.int64)
        entities = list(dict.fromkeys(entity[chains].tolist()))
        copy = {chain: int((entity[chains[: k + 1]] == entity[chain]).sum()) for k, chain in enumerate(chains)}
        classes = torch.from_numpy(target_class(layout.token_class))
        tokens = torch.from_numpy(msa_tokens)
        floating = {
            "token_features": F.one_hot(classes, NUM_CLASSES).float(),
            "residue_mask": torch.ones(len(classes)),
            "msa_deletion_counts": torch.from_numpy(deletions),
            "msa_profile": F.one_hot(tokens, NUM_CLASSES).float().mean(dim=0),
            "template_mask": torch.full((len(types),), float(len(templates) > 0)),
            "template_all_atom_positions": torch.from_numpy(positions),
            "template_all_atom_mask": torch.from_numpy(present).float(),
        }
        integral = {
            "residue_type_ids": torch.from_numpy(RESIDUE_TYPE[classes.numpy()]),
            "residue_index": torch.from_numpy(_residue_index(system, layout)),
            "chain_index": torch.tensor([chains.index(i) + 1 for i in instance.tolist()], dtype=torch.long),
            "entity_index": torch.tensor([entities.index(entity[i]) + 1 for i in instance.tolist()], dtype=torch.long),
            "symmetry_index": torch.tensor([copy[i] for i in instance.tolist()], dtype=torch.long),
            "msa_tokens": tokens,
            "template_residue_type_ids": torch.from_numpy(types),
        }
        features = AlphaFold2MultimerInput(  # type: ignore[call-arg]
            **{name: value.to(device=self.device, dtype=self.dtype).unsqueeze(0) for name, value in floating.items()},
            **{name: value.to(device=self.device).unsqueeze(0) for name, value in integral.items()},
            batch_size=[],
        )
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                _record(system, layout, accepted),
                spec=self.preparation_spec,
            ),
        )
