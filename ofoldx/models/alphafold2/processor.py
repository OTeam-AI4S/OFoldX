r"""Lower monomer protein identity, MSA evidence, and templates to AlphaFold 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import torch
import torch.nn.functional as F
from multimolecule.io import read_fasta_records
from torch import Tensor
from typing_extensions import Self

from ...data import residue_constants
from ...data.components import get_component
from ...data.evidence import MSA, MSACollection, Template, TemplateAlignment
from ...data.molecular import AtomSite, Molecule, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io.msa import read_msa
from ...references import classify_reference
from .config import AlphaFold2MultimerProcessorConfig, AlphaFold2ProcessorConfig
from .model import AlphaFold2Input, AlphaFold2Layout, AlphaFold2MultimerInput, AlphaFold2MultimerLayout

_RESTYPE_BY_COMPONENT = {
    residue_constants.restype_1to3[letter]: index for index, letter in enumerate(residue_constants.restypes)
}
_RESTYPE_BY_COMPONENT["UNK"] = residue_constants.restype_num
_COMPONENT_BY_SYMBOL = {
    **{letter: residue_constants.restype_1to3[letter] for letter in residue_constants.restypes},
    "B": "ASN",
    "J": "UNK",
    "O": "UNK",
    "U": "CYS",
    "X": "UNK",
    "Z": "GLU",
}
_UNKNOWN = residue_constants.restype_num
_GAP = residue_constants.restype_num + 1
# MSA cells are UPPERCASE one-letter symbols (or the ``-`` gap), not ChemicalComponent IDs. This
# composes the same two maps the pre-migration three-letter path used -- one-letter symbol ->
# component id (``_COMPONENT_BY_SYMBOL``) -> AF2 restype index (``_RESTYPE_BY_COMPONENT``) -- so
# every symbol lowers to the exact restype it did before (B->N, U->C, Z->E, J/O/X->UNK, gap->_GAP).
_RESTYPE_BY_SYMBOL = {
    symbol: _RESTYPE_BY_COMPONENT[component_id] for symbol, component_id in _COMPONENT_BY_SYMBOL.items()
}
_RESTYPE_BY_SYMBOL["-"] = _GAP
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


class AlphaFold2Processor:
    r"""Build the released monomer AF2 processed-feature contract."""

    config_class = AlphaFold2ProcessorConfig

    def __init__(
        self,
        config: AlphaFold2ProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2ProcessorConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MSACollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold2Input:
        r"""Lower one protein monomer and its explicit evidence."""

        if isinstance(input, Structure):
            raise ValueError(
                "AlphaFold 2 monomer cannot consume observed coordinates; pass input.system for topology-only "
                "folding or supply aligned coordinates through templates="
            )
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
        system = input if isinstance(input, MolecularSystem) else molecular_system(input)
        target, instance_id = protein_monomer(system)
        target_entity_id = system.instances[instance_id]
        normalized_msa = normalize_msa(msa, system, target)
        normalized_templates = normalize_templates(
            templates,
            target=target,
            target_entity_id=target_entity_id,
            target_instance_id=instance_id,
            max_templates=self.config.max_templates,
        )
        return prepare_alphafold2_features(
            system,
            target=target,
            instance_id=instance_id,
            target_entity_id=target_entity_id,
            msa=normalized_msa,
            templates=normalized_templates,
            device=self.device,
            dtype=self.dtype,
        )


def molecular_system(
    input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence],
) -> MolecularSystem:
    r"""Construct the one-polymer convenience input accepted by monomer AF2."""

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


def protein_monomer(system: MolecularSystem) -> tuple[PolymerSequence, str]:
    r"""Return the exact released target or reject unsupported molecular meaning."""

    if len(system.instances) != 1 or len(system.entities) != 1 or system.bonds:
        raise ValueError("AlphaFold 2 monomer requires one unbound protein instance")
    instance_id, entity_id = next(iter(system.instances.items()))
    entity = system.entities[entity_id]
    if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
        raise ValueError("AlphaFold 2 monomer requires a protein PolymerSequence")
    for component in entity.components:
        component_id = component.component_id
        if component_id not in _RESTYPE_BY_COMPONENT:
            raise ValueError(f"AlphaFold 2 monomer does not support protein component {component_id!r}")
        if component != get_component(component_id):
            raise ValueError(f"AlphaFold 2 requires the package-audited {component_id!r} component graph")
    return entity, instance_id


def normalize_msa(
    msa: MSA | MSACollection | str | PathLike[str] | None,
    system: MolecularSystem,
    target: PolymerSequence,
) -> MSA | None:
    r"""Bind one monomer alignment to the exact target sequence.

    Evidence holds no :class:`PolymerSequence`: an :class:`MSA` names its entity
    by ``entity_id`` and closes over ``system``, so the Processor resolves the
    entity and verifies query width. A path is projected to the entity query axis
    by :func:`io.msa.read_msa`, whose one-letter/deletion projection is the frozen
    equivalent of the old three-letter row reader.
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


def normalize_templates(
    templates: Sequence[Template],
    *,
    target: PolymerSequence,
    target_entity_id: str,
    target_instance_id: str,
    max_templates: int,
) -> tuple[Template, ...]:
    r"""Bind explicit canonical template alignments to the AF2 target."""

    if len(templates) > max_templates:
        raise ValueError(f"AlphaFold 2 accepts at most {max_templates} templates")
    normalized: list[Template] = []
    for template in templates:
        if not isinstance(template, Template):
            raise TypeError("AlphaFold 2 templates must be explicit Template values with canonical alignments")
        select_template_alignment(
            template,
            target_length=len(target),
            target_entity_id=target_entity_id,
            target_instance_id=target_instance_id,
        )
        normalized.append(template)
    return tuple(normalized)


def select_template_alignment(
    template: Template,
    *,
    target_length: int,
    target_entity_id: str,
    target_instance_id: str,
) -> TemplateAlignment:
    r"""Return the one Template alignment that binds to the single AF2 target.

    AlphaFold 2 folds one monomer target, so exactly one of a Template's
    :class:`TemplateAlignment` values may name that target entity and, when it
    pins an instance, that exact target instance. The target-side upper bound is
    the Processor's job: :class:`TemplateAlignment` holds no target sequence.
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


def prepare_alphafold2_features(
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
    r"""Build the released monomer AF2 raw pool, profile, and semantic axes.

    The stochastic ``sample_msa`` / ``make_masked_msa`` / clustering / crop draw is
    the Model's job (once per recycle, off its per-sample generator), so this
    lowers only the deterministic inputs to that draw: the full query-projected
    alignment pool, its per-cell deletion counts, and the full-pool HHblits
    profile.
    """

    aatype = torch.tensor(
        [_RESTYPE_BY_COMPONENT[component.component_id] for component in target.components], dtype=torch.long
    )
    target_feat = torch.cat(
        [
            torch.zeros(len(target), 1),
            F.one_hot(aatype, num_classes=residue_constants.restype_num + 1),
        ],
        dim=-1,
    ).to(torch.float32)
    msa_tokens, msa_deletions, msa_profile = msa_pool_features(target, msa)
    template_values = template_features(
        templates,
        target_length=len(target),
        target_entity_id=target_entity_id,
        target_instance_id=instance_id,
    )
    floating = {
        "target_feat": target_feat,
        "token_mask": torch.ones(len(target)),
        "msa_deletions": msa_deletions,
        "msa_profile": msa_profile,
    }
    names = (
        "template_mask",
        "template_all_atom_positions",
        "template_all_atom_masks",
        "template_torsion_angles_sin_cos",
        "template_alt_torsion_angles_sin_cos",
        "template_torsion_angles_mask",
    )
    for name, value in zip(names, template_values[1:], strict=True):
        floating[name] = value
    placed = {name: value.to(device=device, dtype=dtype).unsqueeze(0) for name, value in floating.items()}
    template_aatype = template_values[0]
    return AlphaFold2Input(
        aatype=aatype.to(device=device).unsqueeze(0),
        residue_index=torch.arange(len(target), device=device).unsqueeze(0),
        msa_tokens=msa_tokens.to(device=device).unsqueeze(0),
        layout=AlphaFold2Layout(
            system=system,
            residue_sites=tuple((instance_id, index) for index in range(len(target))),
            msa=msa,
            templates=templates,
        ),
        template_aatype=template_aatype.to(device=device).unsqueeze(0),
        **placed,
    )


def msa_pool_features(
    target: PolymerSequence,
    msa: MSA | None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Lower the raw query-projected alignment pool and its HHblits profile.

    Returns ``msa_tokens [M, N]`` (long restype ids, row 0 is the query),
    ``msa_deletions [M, N]`` (float per-cell insertion counts), and
    ``msa_profile [N, 22]`` -- the mean of ``one_hot(tokens, 22)`` over the
    whole pool, exactly OpenFold ``make_hhblits_profile`` (computed once, before
    any sampling or masking; upstream AF2 calls this the HHblits profile).
    """

    query = torch.tensor(
        [_RESTYPE_BY_COMPONENT[component.component_id] for component in target.components], dtype=torch.long
    )
    if msa is None:
        tokens = query.unsqueeze(0)
        deletion_counts = torch.zeros_like(tokens, dtype=torch.float32)
    else:
        # The unpaired pool is already projected to the query axis: it is query-width, row 0 is the
        # ungapped query, and column ``i`` binds ``target.components[i]`` (query_to_entity is now the
        # identity map). Each cell is an uppercase one-letter symbol or ``-`` lowered by
        # ``_RESTYPE_BY_SYMBOL``, and the deletion array is the frozen per-cell insertion count.
        tokens = torch.tensor(
            [[_RESTYPE_BY_SYMBOL.get(symbol, _UNKNOWN) for symbol in row] for row in msa.unpaired_sequences],
            dtype=torch.long,
        )
        deletion_counts = torch.tensor(msa.unpaired_deletion_counts, dtype=torch.float32)
        tokens[0] = query
    msa_profile = F.one_hot(tokens, num_classes=22).to(torch.float32).mean(dim=0)
    return tokens, deletion_counts, msa_profile


def template_features(
    templates: tuple[Template, ...],
    *,
    target_length: int,
    target_entity_id: str,
    target_instance_id: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    r"""Project canonical aligned template coordinates into atom37 and torsion features."""

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
    template_types: list[Tensor] = []
    template_positions: list[Tensor] = []
    template_masks: list[Tensor] = []
    template_continuity: list[Tensor] = []
    for template in templates:
        alignment = select_template_alignment(
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
        aatype = torch.full((target_length,), _GAP, dtype=torch.long)
        positions = torch.zeros(target_length, residue_constants.atom_type_num, 3)
        masks = torch.zeros(target_length, residue_constants.atom_type_num)
        continuity = torch.zeros(target_length, dtype=torch.bool)
        previous_target = None
        previous_template = None
        for target_index, template_index in alignment.residue_pairs:
            component_id = template_entity.components[template_index].component_id
            aatype[target_index] = _RESTYPE_BY_COMPONENT.get(component_id, _UNKNOWN)
            for atom_name, atom_index in residue_constants.atom_order.items():
                site = AtomSite(alignment.template_instance_id, template_index, atom_name)
                coordinate_index = sites.get(site)
                if coordinate_index is None or not structure.coordinate_mask[coordinate_index]:
                    continue
                positions[target_index, atom_index] = torch.from_numpy(structure.coordinates[coordinate_index].copy())
                masks[target_index, atom_index] = 1.0
            if previous_target is not None and previous_template is not None:
                continuity[target_index] = (
                    target_index == previous_target + 1 and template_index == previous_template + 1
                )
            previous_target = target_index
            previous_template = template_index
        template_types.append(aatype)
        template_positions.append(positions)
        template_masks.append(masks)
        template_continuity.append(continuity)
    aatype = torch.stack(template_types)
    positions = torch.stack(template_positions)
    masks = torch.stack(template_masks)
    torsion, alternate_torsion, torsion_mask = atom37_to_torsion_angles(
        aatype,
        positions,
        masks,
        torch.stack(template_continuity),
    )
    return (
        aatype,
        torch.ones(len(templates)),
        positions,
        masks,
        torsion,
        alternate_torsion,
        torsion_mask,
    )


def atom37_to_torsion_angles(
    aatype: Tensor,
    positions: Tensor,
    atom_mask: Tensor,
    continuity: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Compute pre-omega, phi, psi, and chi sin/cos values from atom37 coordinates."""

    previous_positions = torch.cat([torch.zeros_like(positions[..., :1, :, :]), positions[..., :-1, :, :]], dim=-3)
    previous_mask = torch.cat([torch.zeros_like(atom_mask[..., :1, :]), atom_mask[..., :-1, :]], dim=-2)
    pre_omega = torch.cat([previous_positions[..., 1:3, :], positions[..., :2, :]], dim=-2)
    phi = torch.cat([previous_positions[..., 2:3, :], positions[..., :3, :]], dim=-2)
    psi = torch.cat([positions[..., :3, :], positions[..., 4:5, :]], dim=-2)
    pre_omega_mask = previous_mask[..., 1:3].prod(dim=-1) * atom_mask[..., :2].prod(dim=-1) * continuity
    phi_mask = previous_mask[..., 2] * atom_mask[..., :3].prod(dim=-1) * continuity
    psi_mask = atom_mask[..., :3].prod(dim=-1) * atom_mask[..., 4]

    chi_indices = torch.zeros(21, 4, 4, dtype=torch.long, device=aatype.device)
    for restype_index, letter in enumerate(residue_constants.restypes):
        for chi_index, atom_names in enumerate(
            residue_constants.chi_angles_atoms[residue_constants.restype_1to3[letter]]
        ):
            chi_indices[restype_index, chi_index] = torch.tensor(
                [residue_constants.atom_order[name] for name in atom_names],
                device=aatype.device,
            )
    selected_indices = chi_indices[aatype.clamp(max=20)]
    expanded = selected_indices.unsqueeze(-1).expand(*selected_indices.shape, 3)
    chi_positions = positions.unsqueeze(-3).expand(*positions.shape[:-2], 4, *positions.shape[-2:]).gather(-2, expanded)
    chi_atom_mask = (
        atom_mask.unsqueeze(-2).expand(*atom_mask.shape[:-1], 4, atom_mask.shape[-1]).gather(-1, selected_indices)
    )
    chi_exists = atom_mask.new_tensor((*residue_constants.chi_angles_mask, (0.0, 0.0, 0.0, 0.0)))[aatype.clamp(max=20)]
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
    e0 = p2 - p1
    e0 = e0 / torch.sqrt(e0.square().sum(dim=-1, keepdim=True) + 1e-8)
    e1 = p0 - p2
    e1 = e1 - e0 * (e0 * e1).sum(dim=-1, keepdim=True)
    e1 = e1 / torch.sqrt(e1.square().sum(dim=-1, keepdim=True) + 1e-8)
    e2 = torch.cross(e0, e1, dim=-1)
    fourth = p3 - p2
    angles = torch.stack([(fourth * e2).sum(dim=-1), (fourth * e1).sum(dim=-1)], dim=-1)
    angles = angles / torch.sqrt(angles.square().sum(dim=-1, keepdim=True) + 1e-8)
    angles = angles * angles.new_tensor((1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0))[..., None]
    periodic = angles.new_tensor(_CHI_PI_PERIODIC)[aatype.clamp(max=20)]
    mirror = torch.cat([torch.ones_like(periodic[..., :3]), 1.0 - 2.0 * periodic], dim=-1)
    return angles, angles * mirror.unsqueeze(-1), torsion_mask


# ===========================================================================
# AlphaFold-Multimer Processor.
#
# Lowers a multi-chain assembly, paired MSA evidence, and templates to the released
# AlphaFold-Multimer processed-feature contract. The multimer Processor has a
# genuinely different required call signature from the monomer Processor above: it
# requires chain identity (a multi-instance ``MolecularSystem`` or named
# ``PolymerSequence`` collection), derives ``chain_index`` / ``entity_index`` /
# ``symmetry_index``, and consumes a paired :class:`MSACollection` rather than a bare
# entity-bound :class:`MSA`. That signature/type difference is why a second
# registered pair exists (Principle 7). It reuses the monomer restype tables,
# ``select_template_alignment``, and the ``_GAP`` / ``_UNKNOWN`` constants above.
#
# VALIDATED (parity vs OpenFold >= 2.0 multimer, model_1_multimer_v3): the released
# multimer MSA channelization here -- the 22-class one-hot / paired token-axis layout,
# cluster-profile construction, and released tensor widths (msa 49, extra 25) -- was
# checked against a reference checkout. Feeding these processor features into both the
# OFoldX AlphaFold2MultimerModel and the OpenFold multimer trunk under the shared official
# input-embedder weights reproduces the single / pair / distogram / masked-MSA outputs to
# fp32 round-off (rel-L2 ~3e-7 at matched recycle count); see
# scripts/parity/alphafold2/multimer_forward.py.
# ===========================================================================


class AlphaFold2MultimerProcessor:
    r"""Build the released AlphaFold-Multimer processed-feature contract."""

    config_class = AlphaFold2MultimerProcessorConfig

    def __init__(
        self,
        config: AlphaFold2MultimerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = AlphaFold2MultimerProcessorConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update output placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: MolecularSystem | Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
        *,
        msa: MSACollection | None = None,
        templates: Sequence[Template] = (),
    ) -> AlphaFold2MultimerInput:
        r"""Lower one multi-chain assembly and its explicit paired evidence."""

        system = input if isinstance(input, MolecularSystem) else multimer_system(input)
        chains = multimer_chains(system)
        normalized_msa = normalize_collection(msa, system, chains)
        normalized_templates = tuple(templates)
        if len(normalized_templates) > self.config.max_templates:
            raise ValueError(f"AlphaFold-Multimer accepts at most {self.config.max_templates} templates")
        return prepare_multimer_features(
            system,
            chains=chains,
            msa=normalized_msa,
            templates=normalized_templates,
            device=self.device,
            dtype=self.dtype,
        )


class MultimerChain:
    r"""One resolved chain: its instance, entity identity indices, and token range."""

    __slots__ = ("instance_id", "entity_id", "entity", "asym_id", "entity_index", "sym_id", "start", "stop")

    def __init__(
        self,
        *,
        instance_id: str,
        entity_id: str,
        entity: PolymerSequence,
        asym_id: int,
        entity_index: int,
        sym_id: int,
        start: int,
        stop: int,
    ) -> None:
        self.instance_id = instance_id
        self.entity_id = entity_id
        self.entity = entity
        self.asym_id = asym_id
        self.entity_index = entity_index
        self.sym_id = sym_id
        self.start = start
        self.stop = stop


def multimer_system(
    input: Sequence[PolymerSequence] | Mapping[str, PolymerSequence],
) -> MolecularSystem:
    r"""Lift a named/ordered ``PolymerSequence`` collection into one MolecularSystem.

    Distinct sequences become distinct entities; repeated identical sequences share
    one entity, so ``sym_id`` copy structure is preserved.
    """

    if isinstance(input, Mapping):
        named = list(input.items())
    else:
        named = [(f"chain_{index}", sequence) for index, sequence in enumerate(input)]
    if not named:
        raise ValueError("AlphaFold-Multimer requires at least one PolymerSequence")
    for _, sequence in named:
        if not isinstance(sequence, PolymerSequence):
            raise TypeError("AlphaFold-Multimer requires PolymerSequence chains")

    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    sequence_to_entity: dict[tuple[object, ...], str] = {}
    for instance_id, sequence in named:
        if instance_id in instances:
            raise ValueError(f"duplicate chain identifier {instance_id!r}")
        key = (sequence.polymer_type, tuple(component.component_id for component in sequence.components))
        entity_id = sequence_to_entity.get(key)
        if entity_id is None:
            entity_id = f"entity_{len(entities)}"
            sequence_to_entity[key] = entity_id
            entities[entity_id] = sequence
        instances[instance_id] = entity_id
    return MolecularSystem(entities=entities, instances=instances)


def multimer_chains(system: MolecularSystem) -> tuple[MultimerChain, ...]:
    r"""Resolve chain identity and reject monomer/unsupported molecular meaning."""

    if len(system.instances) < 2:
        raise ValueError("AlphaFold-Multimer requires at least two chain instances; use ofoldx.alphafold2 for monomers")
    entity_index: dict[str, int] = {}
    sym_counter: dict[str, int] = {}
    chains: list[MultimerChain] = []
    cursor = 0
    for asym_index, (instance_id, entity_id) in enumerate(system.instances.items()):
        entity = system.entities[entity_id]
        if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
            raise ValueError("AlphaFold-Multimer requires protein PolymerSequence chains")
        for component in entity.components:
            component_id = component.component_id
            if component_id not in _RESTYPE_BY_COMPONENT:
                raise ValueError(f"AlphaFold-Multimer does not support protein component {component_id!r}")
            if component != get_component(component_id):
                raise ValueError(f"AlphaFold-Multimer requires the package-audited {component_id!r} component graph")
        if entity_id not in entity_index:
            entity_index[entity_id] = len(entity_index) + 1
        sym_counter[entity_id] = sym_counter.get(entity_id, 0) + 1
        length = len(entity.components)
        chains.append(
            MultimerChain(
                instance_id=instance_id,
                entity_id=entity_id,
                entity=entity,
                asym_id=asym_index + 1,
                entity_index=entity_index[entity_id],
                sym_id=sym_counter[entity_id],
                start=cursor,
                stop=cursor + length,
            )
        )
        cursor += length
    return tuple(chains)


def normalize_collection(
    msa: MSACollection | None,
    system: MolecularSystem,
    chains: tuple[MultimerChain, ...],
) -> MSACollection | None:
    r"""Validate a paired MSACollection against the assembly, or reject a bare MSA."""

    if msa is None:
        return None
    if not isinstance(msa, MSACollection):
        raise ValueError("AlphaFold-Multimer requires a paired MSACollection, not a bare entity-bound MSA")
    entity_length = {chain.entity_id: chain.stop - chain.start for chain in chains}
    for column in msa.msas:
        if column.entity_id not in entity_length and msa.target_instance_ids is None:
            raise ValueError(f"MSACollection column entity {column.entity_id!r} is not an assembly entity")
        if len(column.unpaired_sequences[0]) != entity_length.get(column.entity_id, len(column.unpaired_sequences[0])):
            raise ValueError("MSACollection column query width disagrees with its entity length")
    return msa


def prepare_multimer_features(
    system: MolecularSystem,
    *,
    chains: tuple[MultimerChain, ...],
    msa: MSACollection | None,
    templates: tuple[Template, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaFold2MultimerInput:
    r"""Build the concatenated multimer tensor and semantic axes."""

    total_length = chains[-1].stop
    aatype = torch.tensor(
        [_RESTYPE_BY_COMPONENT[component.component_id] for chain in chains for component in chain.entity.components],
        dtype=torch.long,
    )
    target_feat = F.one_hot(aatype, num_classes=residue_constants.restype_num + 1).to(torch.float32)

    residue_index = torch.cat([torch.arange(chain.stop - chain.start) for chain in chains]).long()
    chain_index = torch.cat([torch.full((chain.stop - chain.start,), chain.asym_id) for chain in chains]).long()
    entity_index = torch.cat([torch.full((chain.stop - chain.start,), chain.entity_index) for chain in chains]).long()
    symmetry_index = torch.cat([torch.full((chain.stop - chain.start,), chain.sym_id) for chain in chains]).long()

    query_tokens = aatype.clone()
    tokens, deletions = materialize_collection(msa, chains, total_length, query_tokens)
    # Under the released stochastic-fold contract the Processor lowers only the raw
    # paired alignment pool and its full-pool profile; the Model draws the
    # clustered/extra realization itself, once per recycle, off its per-sample
    # generator (see ``AlphaFold2MultimerModel.fold``). ``msa_profile`` is the mean of
    # ``one_hot(tokens, 22)`` over the whole pool, mirroring OpenFold multimer
    # ``make_msa_profile`` (computed once, before any sampling or masking).
    msa_profile = F.one_hot(tokens, num_classes=22).to(torch.float32).mean(dim=0)
    template_values = multimer_template_features(templates, chains=chains, total_length=total_length)

    residue_sites = tuple((chain.instance_id, index) for chain in chains for index in range(chain.stop - chain.start))
    chain_boundaries = tuple((chain.instance_id, chain.start, chain.stop) for chain in chains)

    floating = {
        "target_feat": target_feat,
        "token_mask": torch.ones(total_length),
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
        "aatype": aatype,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "entity_index": entity_index,
        "symmetry_index": symmetry_index,
        "msa_tokens": tokens,
        "template_aatype": template_values["template_aatype"],
    }
    placed_integral = {name: value.to(device=device).unsqueeze(0) for name, value in integral.items()}
    return AlphaFold2MultimerInput(
        layout=AlphaFold2MultimerLayout(
            system=system,
            residue_sites=residue_sites,
            chain_boundaries=chain_boundaries,
            msa=msa,
            templates=templates,
        ),
        **placed,
        **placed_integral,
    )


def materialize_collection(
    msa: MSACollection | None,
    chains: tuple[MultimerChain, ...],
    total_length: int,
    query_tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Expand the paired MSACollection row plan onto the concatenated token axis.

    Each plan row is materialized into one full-width token row: every chain's
    token range receives the symbols selected for its entity's column (paired or
    unpaired pool), and gap cells leave the ungapped-elsewhere range as ``_GAP``.
    Row 0 is the concatenated query. FLAGGED: crop ordering and the paired vs
    unpaired block ordering follow the plan row order here, not a verified released
    ordering.
    """

    if msa is None:
        tokens = query_tokens.unsqueeze(0)
        deletions = torch.zeros_like(tokens, dtype=torch.float32)
        return tokens, deletions

    columns_for_chain = _columns_for_chain(msa, chains)
    row_count = msa.row_pairing.shape[0]
    token_rows: list[list[int]] = []
    deletion_rows: list[list[float]] = []
    for row in range(row_count):
        token_row = [_GAP] * total_length
        deletion_row = [0.0] * total_length
        for chain in chains:
            column = columns_for_chain.get(chain.instance_id)
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
                token_row[chain.start + offset] = _RESTYPE_BY_SYMBOL.get(symbol, _UNKNOWN)
                deletion_row[chain.start + offset] = float(counts[offset])
        token_rows.append(token_row)
        deletion_rows.append(deletion_row)
    tokens = torch.tensor(token_rows, dtype=torch.long)
    deletions = torch.tensor(deletion_rows, dtype=torch.float32)
    tokens[0] = query_tokens
    return tokens, deletions


def _columns_for_chain(msa: MSACollection, chains: tuple[MultimerChain, ...]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if msa.target_instance_ids is None:
        entity_to_column = {entry.entity_id: column for column, entry in enumerate(msa.msas)}
        for chain in chains:
            column = entity_to_column.get(chain.entity_id)
            if column is not None:
                mapping[chain.instance_id] = column
    else:
        for column, instance_id in enumerate(msa.target_instance_ids):
            mapping[instance_id] = column
    return mapping


def multimer_template_features(
    templates: tuple[Template, ...],
    *,
    chains: tuple[MultimerChain, ...],
    total_length: int,
) -> dict[str, Tensor]:
    r"""Project canonical aligned template coordinates onto the concatenated axis.

    Produces atom37 positions/masks plus the multimer pseudo-beta and backbone-frame
    projections. FLAGGED: multi-chain template selection/assignment and the exact
    backbone-frame featurization are a best-effort port; the model recomputes
    template geometry from atom37, so these derived fields are advisory.
    """

    atom_type_num = residue_constants.atom_type_num
    if not templates:
        return {
            "template_aatype": torch.full((1, total_length), _GAP, dtype=torch.long),
            "template_mask": torch.zeros(1),
            "template_all_atom_positions": torch.zeros(1, total_length, atom_type_num, 3),
            "template_all_atom_mask": torch.zeros(1, total_length, atom_type_num),
            "template_pseudo_beta": torch.zeros(1, total_length, 3),
            "template_pseudo_beta_mask": torch.zeros(1, total_length),
            "template_backbone_frame": torch.zeros(1, total_length, 4, 4),
            "template_backbone_frame_mask": torch.zeros(1, total_length),
        }

    instance_to_chain = {chain.instance_id: chain for chain in chains}
    entity_to_chains: dict[str, list[MultimerChain]] = {}
    for chain in chains:
        entity_to_chains.setdefault(chain.entity_id, []).append(chain)

    aatypes: list[Tensor] = []
    positions_list: list[Tensor] = []
    masks_list: list[Tensor] = []
    for template in templates:
        aatype = torch.full((total_length,), _GAP, dtype=torch.long)
        positions = torch.zeros(total_length, atom_type_num, 3)
        masks = torch.zeros(total_length, atom_type_num)
        for alignment in template.alignments:
            target_chains = (
                [instance_to_chain[alignment.target_instance_id]]
                if alignment.target_instance_id in instance_to_chain
                else entity_to_chains.get(alignment.target_entity_id, [])
            )
            for chain in target_chains:
                select_template_alignment(
                    template,
                    target_length=chain.stop - chain.start,
                    target_entity_id=chain.entity_id,
                    target_instance_id=chain.instance_id,
                )
                _project_template_chain(template, alignment, chain, aatype, positions, masks)
        aatypes.append(aatype)
        positions_list.append(positions)
        masks_list.append(masks)

    aatype = torch.stack(aatypes)
    positions = torch.stack(positions_list)
    masks = torch.stack(masks_list)
    pseudo_beta, pseudo_beta_mask = _template_pseudo_beta(aatype, positions, masks)
    backbone_frame, backbone_frame_mask = _template_backbone_frame(positions, masks)
    return {
        "template_aatype": aatype,
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
    aatype: Tensor,
    positions: Tensor,
    masks: Tensor,
) -> None:
    structure = template.structure
    template_entity_id = structure.system.instances[alignment.template_instance_id]
    template_entity = structure.system.entities[template_entity_id]
    if not isinstance(template_entity, PolymerSequence) or template_entity.polymer_type != "protein":
        raise ValueError("AlphaFold-Multimer templates must align a protein instance")
    sites = {site: index for index, site in enumerate(structure.sites)}
    for target_index, template_index in alignment.residue_pairs:
        global_index = chain.start + target_index
        component_id = template_entity.components[template_index].component_id
        aatype[global_index] = _RESTYPE_BY_COMPONENT.get(component_id, _UNKNOWN)
        for atom_name, atom_index in residue_constants.atom_order.items():
            site = AtomSite(alignment.template_instance_id, template_index, atom_name)
            coordinate_index = sites.get(site)
            if coordinate_index is None or not structure.coordinate_mask[coordinate_index]:
                continue
            positions[global_index, atom_index] = torch.from_numpy(structure.coordinates[coordinate_index].copy())
            masks[global_index, atom_index] = 1.0


def _template_pseudo_beta(aatype: Tensor, positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    ca = residue_constants.atom_order["CA"]
    cb = residue_constants.atom_order["CB"]
    gly = residue_constants.restypes.index("G")
    atom_index = torch.where(aatype == gly, torch.full_like(aatype, ca), torch.full_like(aatype, cb))
    gather = atom_index.unsqueeze(-1).unsqueeze(-1).expand(*atom_index.shape, 1, 3)
    pseudo_beta = positions.gather(-2, gather).squeeze(-2)
    mask_gather = atom_index.unsqueeze(-1).expand(*atom_index.shape, 1)
    pseudo_beta_mask = masks.gather(-1, mask_gather).squeeze(-1)
    return pseudo_beta, pseudo_beta_mask


def _template_backbone_frame(positions: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
    r"""Gram-Schmidt N/CA/C backbone rigid, matching AF-Multimer ``make_backbone_affine``.

    Reproduces OpenFold ``all_atom_multimer.make_transform_from_reference``:
    ``rotation = Rot3Array.from_two_vectors(C - CA, N - CA)`` with translation ``CA``.
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
