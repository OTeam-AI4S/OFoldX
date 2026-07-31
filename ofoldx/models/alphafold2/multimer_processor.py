r"""
Prepare assembly, paired-MSA, and template features for AlphaFold-Multimer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from ...data.evidence import MSA, MSACollection, Template, TemplateAlignment, resolve_msa_bindings
from ...data.molecular import ENTITY_PROTEIN, MolecularSystem, PolymerSequence
from ...modules import residue_constants
from ...processor import Processor
from .config import AlphaFold2MultimerProcessorConfig
from .model import AlphaFold2MultimerInput
from .processor import (
    GAP_RESIDUE_TYPE_INDEX,
    RESIDUE_TYPE_INDEX_BY_COMPONENT,
    RESIDUE_TYPE_INDEX_BY_SYMBOL,
    UNKNOWN_RESIDUE_TYPE_INDEX,
    _atom37_indices,
    _component_ids,
    select_template_alignment,
)


class AlphaFold2MultimerProcessor(Processor):
    r"""
    Convert supported protein assemblies into `AlphaFold2MultimerInput`.

    Args:
        config: MSA and template feature limits. Defaults to
            ``AlphaFold2MultimerProcessorConfig()``.
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

    def __call__(
        self,
        input: MolecularSystem | Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
        *,
        msa: MSACollection | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold2MultimerInput:
        r"""
        Prepare model features for one protein assembly.

        Args:
            input: Molecular system or named/ordered protein sequences forming one
                assembly.
            msa: Optional paired and unpaired MSA evidence.
            templates: Template structures with explicit target alignments.

        Returns:
            Batched target, paired-MSA, and template tensors. Atom coordinates in
            template features are in angstroms. This method performs no random
            sampling.
        """

        system = input if isinstance(input, MolecularSystem) else multimer_system(input)
        chains = multimer_chains(system)
        msa_bindings = () if msa is None else resolve_msa_bindings(msa, system)
        normalized_templates = tuple(templates)
        if len(normalized_templates) > self.config.max_templates:
            raise ValueError(f"AlphaFold-Multimer accepts at most {self.config.max_templates} templates")
        return prepare_multimer_features(
            system,
            chains=chains,
            msa=msa,
            msa_bindings=msa_bindings,
            templates=normalized_templates,
            device=self.device,
            dtype=self.dtype,
        )


class MultimerChain:
    r"""
    One resolved chain: its instance, entity identity indices, and token range.
    """

    __slots__ = (
        "instance_index",
        "system_entity_index",
        "chain_index",
        "model_entity_index",
        "symmetry_index",
        "component_index",
        "start",
        "stop",
    )

    def __init__(
        self,
        *,
        instance_index: int,
        system_entity_index: int,
        chain_index: int,
        model_entity_index: int,
        symmetry_index: int,
        component_index: Tensor,
        start: int,
        stop: int,
    ) -> None:
        self.instance_index = instance_index
        self.system_entity_index = system_entity_index
        self.chain_index = chain_index
        self.model_entity_index = model_entity_index
        self.symmetry_index = symmetry_index
        self.component_index = component_index
        self.start = start
        self.stop = stop


def multimer_system(
    input: Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
) -> MolecularSystem:
    r"""
    Lift a named/ordered ``PolymerSequence`` collection into one MolecularSystem.

    Distinct sequences become distinct entities; repeated identical sequences share
    one entity, so the symmetry-copy structure is preserved.
    """

    if isinstance(input, Mapping):
        named = list(input.items())
    else:
        named = [(f"chain_{index}", sequence) for index, sequence in enumerate(input)]
    if not named:
        raise ValueError("AlphaFold-Multimer requires at least one PolymerSequence")

    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    sequence_to_entity: dict[PolymerSequence, str] = {}
    for instance_id, sequence in named:
        entity_id = sequence_to_entity.get(sequence)
        if entity_id is None:
            entity_id = f"entity_{len(entities)}"
            sequence_to_entity[sequence] = entity_id
            entities[entity_id] = sequence
        instances[instance_id] = entity_id
    return MolecularSystem.from_entities(entities=entities, instances=instances)


def multimer_chains(system: MolecularSystem) -> tuple[MultimerChain, ...]:
    r"""
    Resolve chain identity and reject monomer/unsupported molecular meaning.
    """

    if system.instance_entity.numel() < 2:
        raise ValueError("AlphaFold-Multimer requires at least two chain instances; use alphafold2 for monomers")
    model_entity_index: dict[int, int] = {}
    symmetry_count: dict[int, int] = {}
    chains: list[MultimerChain] = []
    cursor = 0
    for instance_index, system_entity_index in enumerate(system.instance_entity.tolist()):
        if int(system.entity_type[system_entity_index]) != ENTITY_PROTEIN:
            raise ValueError("AlphaFold-Multimer requires protein PolymerSequence chains")
        component_index = torch.arange(
            int(system.instance_component_ptr[instance_index]),
            int(system.instance_component_ptr[instance_index + 1]),
            device=system.instance_component_ptr.device,
        )
        for component_id in _component_ids(system, component_index):
            if component_id not in RESIDUE_TYPE_INDEX_BY_COMPONENT:
                raise ValueError(f"AlphaFold-Multimer does not support protein component {component_id!r}")
        if system_entity_index not in model_entity_index:
            model_entity_index[system_entity_index] = len(model_entity_index) + 1
        symmetry_count[system_entity_index] = symmetry_count.get(system_entity_index, 0) + 1
        length = component_index.numel()
        chains.append(
            MultimerChain(
                instance_index=instance_index,
                system_entity_index=system_entity_index,
                chain_index=len(chains) + 1,
                model_entity_index=model_entity_index[system_entity_index],
                symmetry_index=symmetry_count[system_entity_index],
                component_index=component_index,
                start=cursor,
                stop=cursor + length,
            )
        )
        cursor += length
    return tuple(chains)


def prepare_multimer_features(
    system: MolecularSystem,
    *,
    chains: tuple[MultimerChain, ...],
    msa: MSACollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    templates: tuple[Template, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold2MultimerInput:
    r"""
    Build the concatenated multimer tensor and semantic axes.
    """

    total_length = chains[-1].stop
    component_index = torch.cat([chain.component_index for chain in chains])
    residue_type_ids = torch.tensor(
        [RESIDUE_TYPE_INDEX_BY_COMPONENT[component_id] for component_id in _component_ids(system, component_index)],
        dtype=torch.long,
    )
    token_features = F.one_hot(residue_type_ids, num_classes=residue_constants.NUM_AMINO_ACID_TYPES + 1).to(
        torch.float32
    )

    residue_index = torch.cat([torch.arange(chain.stop - chain.start) for chain in chains]).long()
    chain_index = torch.cat([torch.full((chain.stop - chain.start,), chain.chain_index) for chain in chains]).long()
    entity_index = torch.cat(
        [torch.full((chain.stop - chain.start,), chain.model_entity_index) for chain in chains]
    ).long()
    symmetry_index = torch.cat(
        [torch.full((chain.stop - chain.start,), chain.symmetry_index) for chain in chains]
    ).long()

    query_tokens = residue_type_ids.clone()
    tokens, deletions = materialize_collection(
        msa,
        msa_bindings,
        chains,
        total_length,
        query_tokens,
    )
    # The Processor emits the raw paired alignment pool and its full-pool profile.
    # The Model samples clustered and extra rows once per recycle. ``msa_profile``
    # is the mean of ``one_hot(tokens, 22)`` before sampling or masking.
    msa_profile = F.one_hot(tokens, num_classes=22).to(torch.float32).mean(dim=0)
    template_values = multimer_template_features(
        templates,
        system=system,
        chains=chains,
        total_length=total_length,
    )

    floating = {
        "token_features": token_features,
        "sequence_mask": torch.ones(total_length),
        "msa_deletions": deletions,
        "msa_profile": msa_profile,
        "template_mask": template_values["template_mask"],
        "template_all_atom_positions": template_values["template_all_atom_positions"],
        "template_all_atom_mask": template_values["template_all_atom_mask"],
        "template_pseudo_beta": template_values["template_pseudo_beta"],
        "template_pseudo_beta_mask": template_values["template_pseudo_beta_mask"],
        "template_backbone_frame": template_values["template_backbone_frame"],
        "template_backbone_frame_mask": template_values["template_backbone_frame_mask"],
    }
    placed = {name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()}
    integral = {
        "residue_type_ids": residue_type_ids,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "entity_index": entity_index,
        "symmetry_index": symmetry_index,
        "msa_tokens": tokens,
        "template_residue_type_ids": template_values["template_residue_type_ids"],
    }
    placed_integral = {name: value.to(device=device).unsqueeze(0) for name, value in integral.items()}
    return AlphaFold2MultimerInput(  # type: ignore[call-arg]
        system=system,
        component_index=component_index.to(device=device).unsqueeze(0),
        atom37_index=_atom37_indices(system, component_index).to(device=device).unsqueeze(0),
        **placed,
        **placed_integral,
        batch_size=[1],
    )


def materialize_collection(
    msa: MSACollection | None,
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    chains: tuple[MultimerChain, ...],
    total_length: int,
    query_tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""
    Expand the paired MSACollection row plan onto the concatenated token axis.

    Each plan row is materialized into one full-width token row: every chain's
    token range receives the symbols selected for its entity's column (paired or
    unpaired pool), and gap cells leave the ungapped-elsewhere range as
    ``GAP_RESIDUE_TYPE_INDEX``.
    Row 0 is the concatenated query. Paired and unpaired rows retain the explicit
    order encoded by the collection's row plan.
    """

    if msa is None:
        tokens = query_tokens.unsqueeze(0)
        deletions = torch.zeros_like(tokens, dtype=torch.float32)
        return tokens, deletions

    columns_for_chain = _columns_for_chain(msa_bindings, chains)
    row_count = msa.row_pairing.shape[0]
    token_rows: list[list[int]] = []
    deletion_rows: list[list[float]] = []
    for row in range(row_count):
        token_row = [GAP_RESIDUE_TYPE_INDEX] * total_length
        deletion_row = [0.0] * total_length
        for chain in chains:
            column = columns_for_chain.get(chain.instance_index)
            if column is None:
                continue
            source = int(msa.row_pairing[row, column])
            if source == -1:
                continue
            paired = bool(msa.row_is_paired[row, column])
            entry = msa.msas[column]
            sequence = entry.paired_sequences[source] if paired else entry.unpaired_sequences[source]
            counts = entry.paired_deletion_counts[source] if paired else entry.unpaired_deletion_counts[source]
            for offset, symbol in enumerate(sequence):
                token_row[chain.start + offset] = RESIDUE_TYPE_INDEX_BY_SYMBOL.get(symbol, UNKNOWN_RESIDUE_TYPE_INDEX)
                deletion_row[chain.start + offset] = float(counts[offset])
        token_rows.append(token_row)
        deletion_rows.append(deletion_row)
    tokens = torch.tensor(token_rows, dtype=torch.long)
    deletions = torch.tensor(deletion_rows, dtype=torch.float32)
    tokens[0] = query_tokens
    return tokens, deletions


def _columns_for_chain(
    msa_bindings: tuple[tuple[MSA, int | None, int], ...],
    chains: tuple[MultimerChain, ...],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for column, (_msa, target_instance_index, entity_index) in enumerate(msa_bindings):
        for chain in chains:
            if target_instance_index is None:
                if chain.system_entity_index == entity_index:
                    mapping[chain.instance_index] = column
            elif chain.instance_index == target_instance_index:
                mapping[chain.instance_index] = column
    return mapping


def multimer_template_features(
    templates: tuple[Template, ...],
    *,
    system: MolecularSystem,
    chains: tuple[MultimerChain, ...],
    total_length: int,
) -> dict[str, Tensor]:
    r"""
    Project aligned template coordinates onto the concatenated residue axis.

    Produces atom37 positions/masks plus the multimer pseudo-beta and backbone-frame
    projections. The model recomputes template geometry from atom37; the derived
    pseudo-beta and backbone-frame fields retain the processor-level representation.
    """

    atom_type_num = residue_constants.atom_type_num
    if not templates:
        return {
            "template_residue_type_ids": torch.full((1, total_length), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long),
            "template_mask": torch.zeros(1),
            "template_all_atom_positions": torch.zeros(1, total_length, atom_type_num, 3),
            "template_all_atom_mask": torch.zeros(1, total_length, atom_type_num),
            "template_pseudo_beta": torch.zeros(1, total_length, 3),
            "template_pseudo_beta_mask": torch.zeros(1, total_length),
            "template_backbone_frame": torch.zeros(1, total_length, 4, 4),
            "template_backbone_frame_mask": torch.zeros(1, total_length),
        }

    instance_to_chain = {chain.instance_index: chain for chain in chains}
    entity_to_chains: dict[int, list[MultimerChain]] = {}
    for chain in chains:
        entity_to_chains.setdefault(chain.system_entity_index, []).append(chain)

    template_residue_type_ids: list[Tensor] = []
    positions_list: list[Tensor] = []
    masks_list: list[Tensor] = []
    for template in templates:
        residue_type_ids = torch.full((total_length,), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long)
        positions = torch.zeros(total_length, atom_type_num, 3)
        masks = torch.zeros(total_length, atom_type_num)
        projected_instances: set[int] = set()
        for alignment in template.alignments:
            try:
                target_entity_index = system.entity_index(alignment.target_entity_id)
                target_instance_index = (
                    None
                    if alignment.target_instance_id is None
                    else system.instance_index(alignment.target_instance_id)
                )
            except KeyError:
                continue
            target_chains = (
                entity_to_chains.get(target_entity_index, [])
                if target_instance_index is None
                else [instance_to_chain[target_instance_index]] if target_instance_index in instance_to_chain else []
            )
            for chain in target_chains:
                if chain.instance_index in projected_instances:
                    continue
                selected_alignment = select_template_alignment(
                    template,
                    system=system,
                    target_length=chain.stop - chain.start,
                    target_entity_index=chain.system_entity_index,
                    target_instance_index=chain.instance_index,
                )
                _project_template_chain(
                    template,
                    selected_alignment,
                    chain,
                    residue_type_ids,
                    positions,
                    masks,
                )
                projected_instances.add(chain.instance_index)
        template_residue_type_ids.append(residue_type_ids)
        positions_list.append(positions)
        masks_list.append(masks)

    residue_type_ids = torch.stack(template_residue_type_ids)
    positions = torch.stack(positions_list)
    masks = torch.stack(masks_list)
    pseudo_beta, pseudo_beta_mask = _template_pseudo_beta(residue_type_ids, positions, masks)
    backbone_frame, backbone_frame_mask = _template_backbone_frame(positions, masks)
    return {
        "template_residue_type_ids": residue_type_ids,
        "template_mask": torch.ones(len(templates)),
        "template_all_atom_positions": positions,
        "template_all_atom_mask": masks,
        "template_pseudo_beta": pseudo_beta,
        "template_pseudo_beta_mask": pseudo_beta_mask,
        "template_backbone_frame": backbone_frame,
        "template_backbone_frame_mask": backbone_frame_mask,
    }


def _project_template_chain(
    template: Template,
    alignment: TemplateAlignment,
    chain: MultimerChain,
    residue_type_ids: Tensor,
    positions: Tensor,
    masks: Tensor,
) -> None:
    structure = template.structure
    system = structure.system
    template_instance_index = system.instance_index(alignment.template_instance_id)
    template_entity_index = int(system.instance_entity[template_instance_index])
    if int(system.entity_type[template_entity_index]) != ENTITY_PROTEIN:
        raise ValueError("AlphaFold-Multimer templates must align a protein instance")
    component_start = int(system.instance_component_ptr[template_instance_index])
    for target_index, template_index in alignment.residue_pairs:
        global_index = chain.start + target_index
        component = component_start + template_index
        component_id = system.symbols[int(system.component_symbol[component])]
        residue_type_ids[global_index] = RESIDUE_TYPE_INDEX_BY_COMPONENT.get(component_id, UNKNOWN_RESIDUE_TYPE_INDEX)
        atom_start = int(system.component_atom_ptr[component])
        atom_stop = int(system.component_atom_ptr[component + 1])
        for atom in range(atom_start, atom_stop):
            atom_index = residue_constants.atom_order.get(system.symbols[int(system.atom_symbol[atom])])
            if atom_index is None or not bool(structure.coordinate_mask[atom]):
                continue
            positions[global_index, atom_index] = (
                structure.coordinates[atom]
                .detach()
                .to(
                    device=positions.device,
                    dtype=positions.dtype,
                )
            )
            masks[global_index, atom_index] = 1.0


def _template_pseudo_beta(residue_type_ids: Tensor, positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    ca = residue_constants.atom_order["CA"]
    cb = residue_constants.atom_order["CB"]
    gly = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("G")
    atom_index = torch.where(
        residue_type_ids == gly,
        torch.full_like(residue_type_ids, ca),
        torch.full_like(residue_type_ids, cb),
    )
    gather = atom_index.unsqueeze(-1).unsqueeze(-1).expand(*atom_index.shape, 1, 3)
    pseudo_beta = positions.gather(-2, gather).squeeze(-2)
    mask_gather = atom_index.unsqueeze(-1).expand(*atom_index.shape, 1)
    pseudo_beta_mask = masks.gather(-1, mask_gather).squeeze(-1)
    return pseudo_beta, pseudo_beta_mask


def _template_backbone_frame(positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Construct the N/CA/C backbone rigid with Gram-Schmidt orthogonalization.

    The rotation is formed from ``C - CA`` and ``N - CA`` with translation ``CA``.
    ``from_two_vectors(e0_raw, e1_raw)`` sets the x-axis ``e0 = normalize(C - CA)``, makes
    ``e1 = normalize((N - CA) - (e1_raw·e0) e0)`` orthogonal to it, and ``e2 = e0 × e1``.
    The 3x3 block stores these as columns (``R[:, 0] = e0``), so the consumer's
    ``R_iᵀ (t_j - t_i)`` reproduces ``rigid.inverse().apply_to_point`` exactly.
    """

    n = residue_constants.atom_order["N"]
    ca = residue_constants.atom_order["CA"]
    c = residue_constants.atom_order["C"]
    backbone_mask = masks[..., n] * masks[..., ca] * masks[..., c]

    ca_xyz = positions[..., ca, :]
    e0_raw = positions[..., c, :] - ca_xyz
    e1_raw = positions[..., n, :] - ca_xyz

    e0 = e0_raw / e0_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dot = (e1_raw * e0).sum(dim=-1, keepdim=True)
    e1 = e1_raw - dot * e0
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    e2 = torch.linalg.cross(e0, e1, dim=-1)

    frame = torch.zeros(*positions.shape[:-2], 4, 4)
    frame[..., :3, 0] = e0
    frame[..., :3, 1] = e1
    frame[..., :3, 2] = e2
    frame[..., :3, 3] = ca_xyz
    frame[..., 3, 3] = 1.0
    return frame, backbone_mask
