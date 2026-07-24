r"""
Prepare monomer protein, MSA, and template features for AlphaFold 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import torch
import torch.nn.functional as F
from multimolecule.io import read_fasta_records
from torch import Tensor

from ...data.evidence import MSA, MSACollection, Template, TemplateAlignment
from ...data.molecular import AtomSite, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io.msa import read_msa
from ...modules import residue_constants
from ...processor import Processor
from ...references import classify_reference
from .config import AlphaFold2ProcessorConfig
from .model import AlphaFold2Input, AlphaFold2Layout

RESIDUE_TYPE_INDEX_BY_COMPONENT = {
    residue_constants.AMINO_ACID_ONE_TO_THREE[letter]: index
    for index, letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES)
}
RESIDUE_TYPE_INDEX_BY_COMPONENT["UNK"] = residue_constants.NUM_AMINO_ACID_TYPES
_COMPONENT_BY_SYMBOL = {
    **{
        letter: residue_constants.AMINO_ACID_ONE_TO_THREE[letter]
        for letter in residue_constants.AMINO_ACID_ONE_LETTER_CODES
    },
    "B": "ASP",
    "J": "UNK",
    "O": "UNK",
    "U": "CYS",
    "X": "UNK",
    "Z": "GLU",
}
UNKNOWN_RESIDUE_TYPE_INDEX = residue_constants.NUM_AMINO_ACID_TYPES
GAP_RESIDUE_TYPE_INDEX = residue_constants.NUM_AMINO_ACID_TYPES + 1
# MSA cells use uppercase one-letter symbols or ``-``. Ambiguous and uncommon
# symbols follow the HHblits mapping: B->D, U->C, Z->E, J/O/X->UNK.
RESIDUE_TYPE_INDEX_BY_SYMBOL = {
    symbol: RESIDUE_TYPE_INDEX_BY_COMPONENT[component_id] for symbol, component_id in _COMPONENT_BY_SYMBOL.items()
}
RESIDUE_TYPE_INDEX_BY_SYMBOL["-"] = GAP_RESIDUE_TYPE_INDEX
_CHI_PI_PERIODIC = (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
)


class AlphaFold2Processor(Processor):
    r"""
    Convert supported protein inputs into `AlphaFold2Input`.

    Args:
        config: Feature limits and preprocessing options. Defaults to
            ``AlphaFold2ProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        config: AlphaFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2ProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold2Input:
        r"""
        Prepare model features for one protein monomer.

        Args:
            input: Inline sequence, FASTA path, protein sequence, molecular system, or structure
                topology containing exactly one protein instance.
            msa: Optional monomer MSA value or alignment path.
            templates: Template structures with explicit target alignments.

        Returns:
            Batched target, MSA, and template tensors. Atom coordinates in template
            features are in angstroms. This method performs no random sampling.
        """

        if isinstance(input, str | PathLike):
            classified = classify_reference(input)
            if isinstance(classified, Path) and not {
                ".fa",
                ".faa",
                ".fasta",
                ".seq",
            }.intersection(suffix.lower() for suffix in classified.suffixes):
                raise ValueError(
                    "AlphaFold 2 monomer cannot consume a coordinate-bearing structure path; use a FASTA path, "
                    "input.system, or templates="
                )
        topology = input.system if isinstance(input, Structure) else input
        system = topology if isinstance(topology, MolecularSystem) else _molecular_system(topology)
        target, instance_id = _protein_monomer(system)
        target_entity_id = system.instances[instance_id]
        normalized_msa = _normalize_msa(msa, system, target)
        normalized_templates = _normalize_templates(
            templates,
            target=target,
            target_entity_id=target_entity_id,
            target_instance_id=instance_id,
            max_templates=self.config.max_templates,
        )
        return _prepare_alphafold2_features(
            system,
            target=target,
            instance_id=instance_id,
            target_entity_id=target_entity_id,
            msa=normalized_msa,
            templates=normalized_templates,
            device=self.device,
            dtype=self.dtype,
        )


def _molecular_system(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence],
) -> MolecularSystem:
    r"""
    Construct the one-polymer convenience input accepted by monomer AlphaFold 2.
    """

    if isinstance(input, str | PathLike):
        classified = classify_reference(input)
        if isinstance(classified, Path):
            records = read_fasta_records(classified)
            named = [
                (record.id or "protein", PolymerSequence.from_one_letter(record.sequence.strip().upper()))
                for record in records
            ]
        else:
            named = [("protein", PolymerSequence.from_one_letter(classified.strip().upper()))]
    elif isinstance(input, PolymerSequence):
        named = [("protein", input)]
    else:
        named = [("protein", sequence) for sequence in input]
    if len(named) != 1 or not isinstance(named[0][1], PolymerSequence):
        raise ValueError("AlphaFold 2 monomer requires exactly one PolymerSequence")
    entity_id, sequence = named[0]
    return MolecularSystem(
        entities={entity_id: sequence},
        instances={entity_id: entity_id},
    )


def _protein_monomer(system: MolecularSystem) -> tuple[PolymerSequence, str]:
    r"""
    Return the single protein target and its instance id.
    """

    if len(system.instances) != 1 or len(system.entities) != 1:
        raise ValueError("AlphaFold 2 monomer requires one protein instance")
    instance_id, entity_id = next(iter(system.instances.items()))
    entity = system.entities[entity_id]
    if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
        raise ValueError("AlphaFold 2 monomer requires a protein PolymerSequence")
    for component in entity.components:
        component_id = component.component_id
        if component_id not in RESIDUE_TYPE_INDEX_BY_COMPONENT:
            raise ValueError(f"AlphaFold 2 monomer does not support protein component {component_id!r}")
    return entity, instance_id


def _normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
    target: PolymerSequence,
) -> MSA | None:
    r"""
    Bind one monomer alignment to the exact target sequence.

    An [`MSA`][] names its entity by ``entity_id``; the Processor resolves that
    entity against ``system`` and verifies the query width. Path inputs are
    projected to the entity query axis by [`io.msa.read_msa`][].
    """

    if msa is None:
        return None
    if isinstance(msa, MSACollection):
        raise ValueError("AlphaFold 2 monomer accepts an entity-bound MSA, not a multimer MSACollection")
    if isinstance(msa, MSA):
        entity = system.entities.get(msa.entity_id)
        if not isinstance(entity, PolymerSequence) or entity != target:
            raise ValueError("MSA entity disagrees with the AlphaFold 2 target")
        if len(msa.unpaired_sequences[0]) != len(target):
            raise ValueError("MSA query length disagrees with the AlphaFold 2 target")
        return msa
    entity_id = next(iter(system.entities))
    parsed = read_msa(Path(msa).expanduser(), entity_id=entity_id)
    if len(parsed.unpaired_sequences[0]) != len(target):
        raise ValueError("MSA query length disagrees with the AlphaFold 2 target")
    return parsed


def _normalize_templates(
    templates: Sequence[Template],
    *,
    target: PolymerSequence,
    target_entity_id: str,
    target_instance_id: str,
    max_templates: int,
) -> tuple[Template, ...]:
    r"""
    Bind explicit template alignments to the AlphaFold 2 target.
    """

    if len(templates) > max_templates:
        raise ValueError(f"AlphaFold 2 accepts at most {max_templates} templates")
    normalized: list[Template] = []
    for template in templates:
        if not isinstance(template, Template):
            raise TypeError("AlphaFold 2 templates must be explicit Template values with canonical alignments")
        _select_template_alignment(
            template,
            target_length=len(target),
            target_entity_id=target_entity_id,
            target_instance_id=target_instance_id,
        )
        normalized.append(template)
    return tuple(normalized)


def _select_template_alignment(
    template: Template,
    *,
    target_length: int,
    target_entity_id: str,
    target_instance_id: str,
) -> TemplateAlignment:
    r"""
    Return the one Template alignment that binds to the single AlphaFold 2 target.

    AlphaFold 2 folds one monomer target, so exactly one of a Template's
    [`TemplateAlignment`][] values may name that target entity and, when it
    pins an instance, that exact target instance. The target-side upper bound is
    the Processor's job: [`TemplateAlignment`][] holds no target sequence.
    """

    matches = [
        alignment
        for alignment in template.alignments
        if alignment.target_entity_id == target_entity_id and alignment.target_instance_id in (None, target_instance_id)
    ]
    if not matches:
        raise ValueError("Template has no alignment for the AlphaFold 2 target")
    if len(matches) != 1:
        raise ValueError("Template has multiple alignments for the AlphaFold 2 target")
    alignment = matches[0]
    if alignment.residue_pairs[-1][0] >= target_length:
        raise ValueError("template alignment target index is outside the AlphaFold 2 target")
    return alignment


def _prepare_alphafold2_features(
    system: MolecularSystem,
    *,
    target: PolymerSequence,
    instance_id: str,
    target_entity_id: str,
    msa: MSA | None,
    templates: tuple[Template, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold2Input:
    r"""
    Build the monomer AlphaFold 2 raw MSA pool, profile, and semantic axes.

    The Model samples and masks MSA rows for each prediction. This function
    preserves the full query-projected alignment pool, per-cell deletion counts,
    and full-pool HHblits profile.
    """

    residue_type = torch.tensor(
        [RESIDUE_TYPE_INDEX_BY_COMPONENT[component.component_id] for component in target.components], dtype=torch.long
    )
    token_features = torch.cat(
        [
            torch.zeros(len(target), 1),
            F.one_hot(residue_type, num_classes=residue_constants.NUM_AMINO_ACID_TYPES + 1),
        ],
        dim=-1,
    ).to(torch.float32)
    msa_tokens, msa_deletions, msa_profile = _msa_pool_features(target, msa)
    template_values = _template_features(
        templates,
        target_length=len(target),
        target_entity_id=target_entity_id,
        target_instance_id=instance_id,
    )
    floating = {
        "token_features": token_features,
        "sequence_mask": torch.ones(len(target)),
        "msa_deletions": msa_deletions,
        "msa_profile": msa_profile,
    }
    names = (
        "template_mask",
        "template_all_atom_positions",
        "template_all_atom_mask",
        "template_torsion_angles_sin_cos",
        "template_alternative_torsion_angles_sin_cos",
        "template_torsion_angles_mask",
    )
    for name, value in zip(names, template_values[1:], strict=True):
        floating[name] = value
    placed = {name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()}
    template_residue_type = template_values[0]
    return AlphaFold2Input(
        residue_type=residue_type.to(device=device).unsqueeze(0),
        residue_index=torch.arange(len(target), device=device).unsqueeze(0),
        msa_tokens=msa_tokens.to(device=device).unsqueeze(0),
        layout=AlphaFold2Layout(
            system=system,
            residue_sites=tuple((instance_id, index) for index in range(len(target))),
            msa=msa,
            templates=templates,
        ),
        template_residue_type=template_residue_type.to(device=device).unsqueeze(0),
        **placed,
    )


def _msa_pool_features(
    target: PolymerSequence,
    msa: MSA | None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Build the query-projected alignment pool and its HHblits profile.

    Returns ``msa_tokens [M, N]`` (long residue-type indices, row 0 is the query),
    ``msa_deletions [M, N]`` (float per-cell insertion counts), and
    ``msa_profile [N, 22]`` -- the mean of ``one_hot(tokens, 22)`` over the
    whole pool, computed before sampling or masking.
    """

    query = torch.tensor(
        [RESIDUE_TYPE_INDEX_BY_COMPONENT[component.component_id] for component in target.components], dtype=torch.long
    )
    if msa is None:
        tokens = query.unsqueeze(0)
        deletion_counts = torch.zeros_like(tokens, dtype=torch.float32)
    else:
        # The pool is query-width, row 0 is the ungapped query, and column ``i``
        # addresses ``target.components[i]``. Deletions are per-cell insertion counts.
        tokens = torch.tensor(
            [
                [RESIDUE_TYPE_INDEX_BY_SYMBOL.get(symbol, UNKNOWN_RESIDUE_TYPE_INDEX) for symbol in row]
                for row in msa.unpaired_sequences
            ],
            dtype=torch.long,
        )
        deletion_counts = torch.tensor(msa.unpaired_deletion_counts, dtype=torch.float32)
        tokens[0] = query
    msa_profile = F.one_hot(tokens, num_classes=22).to(torch.float32).mean(dim=0)
    return tokens, deletion_counts, msa_profile


def _template_features(
    templates: tuple[Template, ...],
    *,
    target_length: int,
    target_entity_id: str,
    target_instance_id: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""
    Project aligned template coordinates into atom37 and torsion features.
    """

    if not templates:
        return (
            torch.zeros(1, target_length, dtype=torch.long),
            torch.zeros(1),
            torch.zeros(1, target_length, residue_constants.atom_type_num, 3),
            torch.zeros(1, target_length, residue_constants.atom_type_num),
            torch.zeros(1, target_length, 7, 2),
            torch.zeros(1, target_length, 7, 2),
            torch.zeros(1, target_length, 7),
        )
    template_residue_types: list[Tensor] = []
    template_positions: list[Tensor] = []
    template_masks: list[Tensor] = []
    for template in templates:
        alignment = _select_template_alignment(
            template,
            target_length=target_length,
            target_entity_id=target_entity_id,
            target_instance_id=target_instance_id,
        )
        structure = template.structure
        template_entity_id = structure.system.instances[alignment.template_instance_id]
        template_entity = structure.system.entities[template_entity_id]
        if not isinstance(template_entity, PolymerSequence) or template_entity.polymer_type != "protein":
            raise ValueError("AlphaFold 2 templates must align a protein instance")
        sites = {site: index for index, site in enumerate(structure.sites)}
        residue_type = torch.full((target_length,), GAP_RESIDUE_TYPE_INDEX, dtype=torch.long)
        positions = torch.zeros(target_length, residue_constants.atom_type_num, 3)
        masks = torch.zeros(target_length, residue_constants.atom_type_num)
        for target_index, template_index in alignment.residue_pairs:
            component_id = template_entity.components[template_index].component_id
            residue_type[target_index] = RESIDUE_TYPE_INDEX_BY_COMPONENT.get(component_id, UNKNOWN_RESIDUE_TYPE_INDEX)
            for atom_name, atom_index in residue_constants.atom_order.items():
                site = AtomSite(alignment.template_instance_id, template_index, atom_name)
                coordinate_index = sites.get(site)
                if coordinate_index is None or not structure.coordinate_mask[coordinate_index]:
                    continue
                positions[target_index, atom_index] = torch.from_numpy(structure.coordinates[coordinate_index].copy())
                masks[target_index, atom_index] = 1.0
        template_residue_types.append(residue_type)
        template_positions.append(positions)
        template_masks.append(masks)
    residue_type = torch.stack(template_residue_types)
    positions = torch.stack(template_positions)
    masks = torch.stack(template_masks)
    torsion, alternative_torsion, torsion_mask = _atom37_to_torsion_angles(
        residue_type,
        positions,
        masks,
    )
    return (
        residue_type,
        torch.ones(len(templates)),
        positions,
        masks,
        torsion,
        alternative_torsion,
        torsion_mask,
    )


def _atom37_to_torsion_angles(
    residue_type: Tensor,
    positions: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""
    Compute pre-omega, phi, psi, and chi sin/cos values from atom37 coordinates.
    """

    previous_positions = torch.cat([torch.zeros_like(positions[..., :1, :, :]), positions[..., :-1, :, :]], dim=-3)
    previous_mask = torch.cat([torch.zeros_like(atom_mask[..., :1, :]), atom_mask[..., :-1, :]], dim=-2)
    pre_omega = torch.cat([previous_positions[..., 1:3, :], positions[..., :2, :]], dim=-2)
    phi = torch.cat([previous_positions[..., 2:3, :], positions[..., :3, :]], dim=-2)
    psi = torch.cat([positions[..., :3, :], positions[..., 4:5, :]], dim=-2)
    pre_omega_mask = previous_mask[..., 1:3].prod(dim=-1) * atom_mask[..., :2].prod(dim=-1)
    phi_mask = previous_mask[..., 2] * atom_mask[..., :3].prod(dim=-1)
    psi_mask = atom_mask[..., :3].prod(dim=-1) * atom_mask[..., 4]

    chi_indices = torch.zeros(21, 4, 4, dtype=torch.long, device=residue_type.device)
    for residue_type_index, letter in enumerate(residue_constants.AMINO_ACID_ONE_LETTER_CODES):
        for chi_index, atom_names in enumerate(
            residue_constants.chi_angles_atoms[residue_constants.AMINO_ACID_ONE_TO_THREE[letter]]
        ):
            chi_indices[residue_type_index, chi_index] = torch.tensor(
                [residue_constants.atom_order[name] for name in atom_names],
                device=residue_type.device,
            )
    selected_indices = chi_indices[residue_type.clamp(max=20)]
    expanded = selected_indices.unsqueeze(-1).expand(*selected_indices.shape, 3)
    chi_positions = positions.unsqueeze(-3).expand(*positions.shape[:-2], 4, *positions.shape[-2:]).gather(-2, expanded)
    chi_atom_mask = (
        atom_mask.unsqueeze(-2).expand(*atom_mask.shape[:-1], 4, atom_mask.shape[-1]).gather(-1, selected_indices)
    )
    chi_exists = atom_mask.new_tensor((*residue_constants.chi_angles_mask, (0.0, 0.0, 0.0, 0.0)))[
        residue_type.clamp(max=20)
    ]
    chi_mask = chi_atom_mask.prod(dim=-1) * chi_exists
    torsion_atoms = torch.cat(
        [pre_omega.unsqueeze(-3), phi.unsqueeze(-3), psi.unsqueeze(-3), chi_positions],
        dim=-3,
    )
    torsion_mask = torch.cat(
        [pre_omega_mask.unsqueeze(-1), phi_mask.unsqueeze(-1), psi_mask.unsqueeze(-1), chi_mask],
        dim=-1,
    )
    p0, p1, p2, p3 = torsion_atoms.unbind(dim=-2)
    e0_x, e0_y, e0_z = (p2 - p1).unbind(dim=-1)
    e0_norm = torch.sqrt(e0_x.square() + e0_y.square() + e0_z.square() + 1e-8)
    e0_x, e0_y, e0_z = e0_x / e0_norm, e0_y / e0_norm, e0_z / e0_norm

    e1_x, e1_y, e1_z = (p0 - p2).unbind(dim=-1)
    projection = e1_x * e0_x + e1_y * e0_y + e1_z * e0_z
    e1_x = e1_x - projection * e0_x
    e1_y = e1_y - projection * e0_y
    e1_z = e1_z - projection * e0_z
    e1_norm = torch.sqrt(e1_x.square() + e1_y.square() + e1_z.square() + 1e-8)
    e1_x, e1_y, e1_z = e1_x / e1_norm, e1_y / e1_norm, e1_z / e1_norm

    e2_x = e0_y * e1_z - e0_z * e1_y
    e2_y = e0_z * e1_x - e0_x * e1_z
    e2_z = e0_x * e1_y - e0_y * e1_x
    fourth_x, fourth_y, fourth_z = (p3 - p2).unbind(dim=-1)
    sine = fourth_x * e2_x + fourth_y * e2_y + fourth_z * e2_z
    cosine = fourth_x * e1_x + fourth_y * e1_y + fourth_z * e1_z
    angle_norm = torch.sqrt(sine.square() + cosine.square() + 1e-8)
    angles = torch.stack((sine / angle_norm, cosine / angle_norm), dim=-1)
    angles = angles * angles.new_tensor((1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0))[..., None]
    periodic = angles.new_tensor(_CHI_PI_PERIODIC)[residue_type.clamp(max=20)]
    mirror = torch.cat([torch.ones_like(periodic[..., :3]), 1.0 - 2.0 * periodic], dim=-1)
    return angles, angles * mirror.unsqueeze(-1), torsion_mask
