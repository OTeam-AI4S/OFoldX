r"""
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np
import torch

from ...data.evidence import MSA, MsaCollection, Template
from ...data.molecular import (
    AMINO_ACID_THREE_TO_ONE,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_msa_input, resolve_polymer_input, resolve_template_input
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score
from .model import (
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3SamplingOutput,
    alphafold3_pae_bin_centers,
    alphafold3_ranking_score,
    alphafold3_tm_adjusted_pae,
)
from .processor import AlphaFold3Processor, molecular_system_from_named_sequences


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 3.
    """

    model_architecture = "alphafold3"

    def prepare_fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
    ) -> AlphaFold3Input:
        r"""Prepare one molecular assembly and its evidence for AlphaFold 3."""

        resolved_input = resolve_polymer_input(input)
        canonical_input = (
            resolved_input
            if isinstance(resolved_input, MolecularSystem | Structure)
            else molecular_system_from_named_sequences(
                resolved_input,
                revision=self.processor.config.ccd_revision,
            )
        )
        system = canonical_input.system if isinstance(canonical_input, Structure) else canonical_input
        return self.processor(
            canonical_input,
            msa=resolve_msa_input(msa, system, entity_types={ENTITY_PROTEIN, ENTITY_RNA}),
            templates=resolve_template_input(templates),
        )

    def fold(
        self,
        input: str | PathLike[str] | PolymerSequence | Sequence[PolymerSequence] | MolecularSystem | Structure,
        *,
        msa: MSA | MsaCollection | str | PathLike[str] | None = None,
        templates: Sequence[Template] | str | PathLike[str] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        recycling_steps: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one molecular assembly with AlphaFold 3.

        Args:
            input: Inline sequence, FASTA or structure path, polymer sequence collection,
                molecular system, or structure topology.
            msa: Optional MSA value, collection, or alignment path.
            templates: Template structures aligned to target polymers.
            num_samples: Number of diffusion samples. ``None`` uses five.
            seed: Optional diffusion seed, mutually exclusive with ``generator``.
            generator: Optional generator for per-recycle MSA permutations, initial
                coordinates, churn, and rigid augmentations. Mutually exclusive with ``seed``.
            recycling_steps: Number of complete trunk passes. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions. Coordinates, PAE, and PDE values are
                in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        model_input = self.prepare_fold(
            input,
            msa=msa,
            templates=templates,
        )
        output = self.model.sample(
            model_input,
            num_samples=num_samples,
            generator=generator,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
        )
        return self.decode_fold(model_input, output)

    def decode_fold(
        self,
        input: AlphaFold3Input,
        output: AlphaFold3SamplingOutput,
    ) -> FoldResult:
        r"""
        Decode one raw AlphaFold 3 sampling output without executing the model.

        Model atom order is mapped back onto the complete canonical atom axis;
        confidence tensors are converted into public scores and metrics.
        """

        system = input.system.cpu()  # type: ignore[attr-defined]
        atom_count = int(input.atom_mask[0].sum().item())
        token_count = int(input.token_mask[0].sum().item())
        atom_index = input.atom_index[0, :atom_count].detach().cpu().long()
        token_component_index = input.token_component_index[0, :token_count].detach().cpu().long()
        token_atom_index = input.token_atom_index[0, :token_count].detach().cpu().long()
        atom_axis = tuple(system.atom_site(index) for index in atom_index.tolist())
        token_axis = tuple(
            system.atom_site(atom) if atom >= 0 else system.component_reference(component)
            for component, atom in zip(
                token_component_index.tolist(),
                token_atom_index.tolist(),
            )
        )

        coordinates = output.coordinates[0, :, :atom_count].detach().to(device="cpu", dtype=torch.float32)
        atom_plddt = output.plddt[0, :, :atom_count].detach().to(device="cpu", dtype=torch.float32)
        pae_logits = output.pae_logits[0, :, :token_count, :token_count].detach().float().cpu()
        pde_logits = output.pde_logits[0, :, :token_count, :token_count].detach().float().cpu()
        resolved_logits = output.resolved_logits[0, :, :atom_count].detach().float().cpu()
        distogram_logits = output.distogram_logits[0, :token_count, :token_count].detach().float().cpu()
        ptm, iptm, ptm_iptm_average = (
            value[0].detach().cpu() for value in self.model.confidence_summary(output, input)
        )
        num_samples = coordinates.shape[0]

        pae = _expected_bin_value(pae_logits, alphafold3_pae_bin_centers(pae_logits.shape[-1]).cpu())
        pde = _expected_bin_value(pde_logits, _uniform_bin_centers(pde_logits.shape[-1], 0.0, 32.0))
        resolved_probability = resolved_logits.softmax(dim=-1)[..., 1]
        contact_probability = _contact_probability(distogram_logits).numpy()

        # Chain-level confidence summaries operate on decoded per-sample arrays.
        full_pae = pae.numpy()
        full_pde = pde.numpy()
        chain_indices = input.chain_index[0, :token_count].cpu().numpy().astype(np.int64)
        frame_mask = input.token_mask[0, :token_count].to(torch.bool).cpu().numpy()
        unique_chain = np.unique(chain_indices)
        chain_axis = tuple(system.instance_id(int(index)) for index in unique_chain)
        multi_chain = unique_chain.shape[0] > 1

        pae_single_mask = np.tile(frame_mask[:, None], (1, token_count))
        frame_pair_mask = frame_mask[:, None] & frame_mask[None, :]
        tm_interface = alphafold3_tm_adjusted_pae(
            pae_logits,
            pair_mask=torch.from_numpy(frame_pair_mask).unsqueeze(0).expand(pae_logits.shape[0], -1, -1),
            chain_index=torch.from_numpy(chain_indices).unsqueeze(0).expand(pae_logits.shape[0], -1),
            interface=True,
        ).numpy()

        chain_pair_pde_mean, chain_pair_pde_min = _chain_pair_pde(full_pde, chain_indices)
        _, chain_pair_pae_min, _ = _chain_pair_pae(full_pae, chain_indices, mask=pae_single_mask)
        pde_ichain, pde_xchain, pde_full_chain = _pde_single(chain_indices, full_pde, contact_probability)
        chain_metrics = _pae_metrics(chain_indices, full_pae, pae_single_mask, contact_probability, tm_interface)
        ranking_confidence_pae = _rank_metric(
            full_pae,
            contact_probability * frame_mask[:, None].astype(float),
            rank_metric_eps=self.model.config.rank_metric_eps,
        )

        predictions: list[FoldPrediction] = []
        for sample_index in range(num_samples):
            num_atoms = system.atom_symbol.numel()
            complete_coordinates = torch.zeros(num_atoms, 3, dtype=torch.float32)
            coordinate_mask = torch.zeros(num_atoms, dtype=torch.bool)
            complete_coordinates[atom_index] = coordinates[sample_index]
            coordinate_mask[atom_index] = True
            num_components = system.component_symbol.numel()
            structure = Structure(  # type: ignore[call-arg]
                system=system,
                coordinates=complete_coordinates,
                coordinate_mask=coordinate_mask,
                occupancy=torch.zeros(num_atoms, dtype=torch.float32),
                occupancy_mask=torch.zeros(num_atoms, dtype=torch.bool),
                b_factor=torch.zeros(num_atoms, dtype=torch.float32),
                b_factor_mask=torch.zeros(num_atoms, dtype=torch.bool),
                source_symbols=(),
                residue_id=torch.full((num_components,), -1, dtype=torch.long),
                insertion_code=torch.full((num_components,), -1, dtype=torch.long),
                batch_size=[],
            )
            disorder = alphafold3_fraction_disordered(structure)
            clash = alphafold3_has_clash(structure)
            sample_ranking_score = alphafold3_ranking_score(
                float(ptm[sample_index]),
                float(iptm[sample_index]),
                disorder,
                clash,
            )
            ranking_confidence = float(ptm_iptm_average[sample_index]) if multi_chain else float(ptm[sample_index])
            predictions.append(
                FoldPrediction(
                    structure=structure,
                    scores={
                        "mean_plddt": Score(float(atom_plddt[sample_index].mean()), preferred="higher"),
                        "ptm": Score(float(ptm[sample_index]), preferred="higher"),
                        # A single-chain target has no interface, so the interface pTM is
                        # a mean over no chain pair and is simply not reported.
                        **({"iptm": Score(float(iptm[sample_index]), preferred="higher")} if multi_chain else {}),
                        "ptm_iptm_average": Score(float(ptm_iptm_average[sample_index]), preferred="higher"),
                        "ranking_score": Score(sample_ranking_score, preferred="higher"),
                        "ranking_confidence": Score(ranking_confidence, preferred="higher"),
                        "ranking_confidence_pae": Score(
                            float(ranking_confidence_pae[sample_index]),
                            preferred="higher",
                        ),
                        "fraction_disordered": Score(disorder, preferred=None),
                        "has_clash": Score(float(clash), preferred="lower"),
                    },
                    metrics={
                        "plddt": Metric(values=atom_plddt[sample_index].numpy(), axis=atom_axis),
                        "resolved_probability": Metric(
                            values=resolved_probability[sample_index].numpy(),
                            axis=atom_axis,
                        ),
                        "predicted_distance_error": Metric(
                            values=pde_full_chain[sample_index],
                            axis=token_axis,
                            unit="angstrom",
                        ),
                        "chain_ptm": Metric(values=chain_metrics["iptm_ichain"][sample_index], axis=chain_axis),
                        "chain_iptm": Metric(values=chain_metrics["iptm_xchain"][sample_index], axis=chain_axis),
                        "pae_ichain": Metric(
                            values=chain_metrics["pae_ichain"][sample_index],
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                        "pae_xchain": Metric(
                            values=chain_metrics["pae_xchain"][sample_index],
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                        "intra_chain_single_pde": Metric(
                            values=pde_ichain[sample_index],
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                        "cross_chain_single_pde": Metric(
                            values=pde_xchain[sample_index],
                            axis=chain_axis,
                            unit="angstrom",
                        ),
                    },
                    pair_metrics={
                        "pae": PairMetric(
                            values=pae[sample_index].numpy(),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=pde[sample_index].numpy(),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_probability,
                            row_axis=token_axis,
                            column_axis=token_axis,
                        ),
                        "chain_pair_iptm": PairMetric(
                            values=chain_metrics["chain_pair_iptm"][sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                        ),
                        "chain_pair_pae_min": PairMetric(
                            values=chain_pair_pae_min[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_pde_mean": PairMetric(
                            values=chain_pair_pde_mean[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                        "chain_pair_pde_min": PairMetric(
                            values=chain_pair_pde_min[sample_index],
                            row_axis=chain_axis,
                            column_axis=chain_axis,
                            unit="angstrom",
                        ),
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))


# Maximum accessible surface areas from Sander & Rost 1994
# (https://doi.org/10.1002/prot.340200303), used by AlphaFold-RSA.
_MAX_ACCESSIBLE_SURFACE_AREA = {
    "ALA": 106.0,
    "ARG": 248.0,
    "ASN": 157.0,
    "ASP": 163.0,
    "CYS": 135.0,
    "GLN": 198.0,
    "GLU": 194.0,
    "GLY": 84.0,
    "HIS": 184.0,
    "ILE": 169.0,
    "LEU": 164.0,
    "LYS": 205.0,
    "MET": 188.0,
    "PHE": 197.0,
    "PRO": 136.0,
    "SER": 130.0,
    "THR": 142.0,
    "TRP": 227.0,
    "TYR": 222.0,
    "VAL": 142.0,
}

# AlphaFold-RSA rASA disorder threshold.
_RASA_DISORDER_CUTOFF = 0.581
_RASA_WINDOW = 25

_POLYMER_ENTITY_TYPES = frozenset((ENTITY_PROTEIN, ENTITY_RNA, ENTITY_DNA, ENTITY_DNA_RNA_HYBRID))


def alphafold3_fraction_disordered(
    structure: Structure,
    rasa_disorder_cutoff: float = _RASA_DISORDER_CUTOFF,
) -> float:
    r"""
    Fraction of protein residues predicted disordered by windowed AlphaFold-RSA.

    Restricts to protein chains, computes a windowed (window 25) relative solvent
    accessible surface area per residue with DSSP 4.4.7 semantics, and returns the mean
    fraction of residues whose rASA exceeds ``rasa_disorder_cutoff``. Repeated
    chain sequences reuse a cached rASA profile. Calculation errors propagate
    rather than being reinterpreted as an
    absent chain.

    Args:
        structure: The decoded per-sample structure to score.
        rasa_disorder_cutoff: The rASA threshold above which a residue counts as
            disordered.

    Returns:
        The fraction of protein residues predicted disordered, in ``[0, 1]``.
    """

    rasa: list[float] = []
    sequence_rasa: dict[str, np.ndarray] = {}
    for instance_index in _protein_instance_indices(structure):
        chain_sequence = _chain_sequence(structure, instance_index)
        cached = sequence_rasa.get(chain_sequence)
        if cached is not None:
            rasa.extend(cached.tolist())
            continue
        per_residue = _windowed_solvent_accessible_area(structure, instance_index)
        sequence_rasa[chain_sequence] = per_residue
        rasa.extend(per_residue.tolist())

    if not rasa:
        return 0.0
    return float(np.mean(np.asarray(rasa) > rasa_disorder_cutoff))


def alphafold3_has_clash(
    structure: Structure,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
    r"""
    Whether the structure has at least one clashing polymer chain.

    Restricts to protein / RNA / DNA polymer atoms, then flags a chain as
    overlapping when more than ``min_clashes_for_overlap`` of its atoms lie within
    ``cutoff_radius`` Angstrom of a polymer atom in another chain (or a residue more
    than one apart in the same chain), or when more than ``min_fraction_for_overlap``
    of the chain's atoms clash.

    Args:
        structure: The decoded per-sample structure to score.
        cutoff_radius: Atom-pair distances below this (Angstrom) count as a clash.
        min_clashes_for_overlap: Minimum clashing atoms for a chain to overlap.
        min_fraction_for_overlap: Minimum clashing fraction for a chain to overlap.

    Returns:
        ``True`` if at least one polymer chain overlaps another.
    """

    coordinates, residue_indices, instance_indices = _polymer_atoms(structure)
    if coordinates.shape[0] == 0:
        return False

    neighbors = _query_ball(coordinates, cutoff_radius)
    per_atom_has_clash = np.zeros(coordinates.shape[0], dtype=bool)
    for atom_index, clashing in enumerate(neighbors):
        for other in clashing:
            if abs(int(residue_indices[atom_index]) - int(residue_indices[other])) > 1 or (
                instance_indices[atom_index] != instance_indices[other]
            ):
                per_atom_has_clash[atom_index] = True
                break

    for instance_index in np.unique(instance_indices):
        mask = instance_indices == instance_index
        num_atoms = int(np.sum(mask))
        if num_atoms == 0:
            continue
        num_clashes = int(np.sum(per_atom_has_clash & mask))
        fraction_clashes = num_clashes / num_atoms
        if num_clashes > min_clashes_for_overlap or fraction_clashes > min_fraction_for_overlap:
            return True
    return False


def _windowed_solvent_accessible_area(structure: Structure, instance_index: int) -> np.ndarray:
    r"""
    Windowed per-residue rASA for one protein chain (AlphaFold-RSA).

    Runs the DSSP 4.4.7 solvent-accessibility algorithm over the chain, normalizes each residue's
    accessible surface area by [`_MAX_ACCESSIBLE_SURFACE_AREA`][], clips to ``1.0``, and
    averages over a reflect-padded window.
    Raises ``RuntimeError`` when the chain has no complete DSSP residues.
    """

    accessible = _dssp_accessible_area(structure, instance_index)
    if not accessible:
        raise RuntimeError("DSSP produced no accessible-area rows")
    rasa = []
    for residue_name, area in accessible:
        max_area = _MAX_ACCESSIBLE_SURFACE_AREA.get(residue_name, _MAX_ACCESSIBLE_SURFACE_AREA["ALA"])
        normalized = area / max_area
        rasa.append(min(normalized, 1.0))

    half_window = (_RASA_WINDOW - 1) // 2
    padded = np.pad(np.asarray(rasa, dtype=float), (half_window, half_window), "reflect")
    return np.convolve(padded, np.ones(_RASA_WINDOW), "valid") / _RASA_WINDOW


# DSSP 4.4.7 accessibility parameters.
_DSSP_ATOM_RADII = {"N": 1.65, "CA": 1.87, "C": 1.76, "O": 1.4}
_DSSP_SIDE_CHAIN_RADIUS = 1.8
_DSSP_WATER_RADIUS = 1.4
_DSSP_SURFACE_POINT_COUNT = 401


def _dssp_surface_dots() -> np.ndarray:
    r"""
    Return the 401-point Fibonacci sphere used by DSSP 4.4.7.
    """

    n = (_DSSP_SURFACE_POINT_COUNT - 1) // 2
    indices = np.arange(-n, n + 1, dtype=np.int32)
    golden_ratio = np.float32((1.0 + np.sqrt(np.float32(5.0))) / 2.0)
    latitude = np.arcsin(np.float32(2.0) * indices.astype(np.float32) / np.float32(_DSSP_SURFACE_POINT_COUNT))
    longitude = np.asarray(
        np.fmod(indices.astype(np.float64), np.float64(golden_ratio)) * 2.0 * np.pi / np.float64(golden_ratio),
        dtype=np.float32,
    )
    return np.stack(
        (
            np.sin(longitude) * np.cos(latitude),
            np.cos(longitude) * np.cos(latitude),
            np.sin(latitude),
        ),
        axis=-1,
    ).astype(np.float32)


_DSSP_SURFACE_DOTS = _dssp_surface_dots()
_DSSP_SURFACE_POINT_WEIGHT = np.float32(4.0 * np.pi / _DSSP_SURFACE_POINT_COUNT)


def _dssp_accessible_area(structure: Structure, instance_index: int) -> list[tuple[str, float]]:
    r"""
    Return per-residue ``(residue_name, accessible_area)`` for one protein chain.

    This implements the DSSP 4.4.7 accessibility calculation with its atom radii,
    401-point Fibonacci sphere, water radius, overlap test, and integer output
    rounding. DSSP drops residues that
    do not contain all four backbone atoms (N, CA, C, O), then scores each retained
    residue against the complete retained chain in isolation.
    """

    residues: dict[int, tuple[str, dict[str, np.ndarray], list[np.ndarray]]] = {}
    system = structure.system
    component_start = int(system.instance_component_ptr[instance_index])
    component_stop = int(system.instance_component_ptr[instance_index + 1])
    component_atom_ptr = system.component_atom_ptr.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    atom_symbol = system.atom_symbol.detach().cpu().tolist()
    atomic_number = system.atomic_number.detach().cpu()
    structure_coordinates = structure.coordinates.detach().to(device="cpu", dtype=torch.float32)
    coordinate_mask = structure.coordinate_mask.detach().cpu()
    for component_index in range(component_start, component_stop):
        residue_name = system.symbols[int(component_symbol[component_index])]
        backbone: dict[str, np.ndarray] = {}
        side_chain: list[np.ndarray] = []
        for atom_index in range(
            int(component_atom_ptr[component_index]),
            int(component_atom_ptr[component_index + 1]),
        ):
            if not bool(coordinate_mask[atom_index]) or int(atomic_number[atom_index]) == 1:
                continue
            atom_name = system.symbols[int(atom_symbol[atom_index])]
            coordinate = structure_coordinates[atom_index].numpy()
            if atom_name in _DSSP_ATOM_RADII:
                backbone[atom_name] = coordinate
            else:
                side_chain.append(coordinate)
        residues[component_index - component_start] = (
            residue_name,
            backbone,
            side_chain,
        )

    complete_residues = [
        (component_index, residue_name, backbone, side_chain)
        for component_index, (residue_name, backbone, side_chain) in sorted(residues.items())
        if backbone.keys() >= _DSSP_ATOM_RADII.keys()
    ]
    if not complete_residues:
        raise RuntimeError("protein chain has no complete residues")

    atom_coordinates: list[np.ndarray] = []
    atom_radii: list[float] = []
    atom_residue_indices: list[int] = []
    residue_names: list[str] = []
    for residue_index, (_, residue_name, backbone, side_chain) in enumerate(complete_residues):
        residue_names.append(residue_name)
        for atom_name in ("N", "CA", "C", "O"):
            atom_coordinates.append(backbone[atom_name])
            atom_radii.append(_DSSP_ATOM_RADII[atom_name])
            atom_residue_indices.append(residue_index)
        atom_coordinates.extend(side_chain)
        atom_radii.extend([_DSSP_SIDE_CHAIN_RADIUS] * len(side_chain))
        atom_residue_indices.extend([residue_index] * len(side_chain))

    coordinates = np.asarray(atom_coordinates, dtype=np.float32)
    radii = np.asarray(atom_radii, dtype=np.float32)
    residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
    atom_areas = np.empty(coordinates.shape[0], dtype=np.float32)
    surface_radii = radii + np.float32(_DSSP_WATER_RADIUS)
    occlusion_radii = radii.astype(np.float64) + np.float64(np.float32(_DSSP_WATER_RADIUS))

    for atom_index, (coordinate, surface_radius, occlusion_radius) in enumerate(
        zip(coordinates, surface_radii, occlusion_radii)
    ):
        relative = coordinates - coordinate
        center_distances_sq = np.sum(relative * relative, axis=-1, dtype=np.float32)
        candidate_mask = (center_distances_sq.astype(np.float64) > 0.0001) & (
            center_distances_sq.astype(np.float64) < np.square(occlusion_radius + occlusion_radii)
        )
        candidates = relative[candidate_mask]
        candidate_radii_sq = np.square(occlusion_radii[candidate_mask])
        surface_points = _DSSP_SURFACE_DOTS * surface_radius
        if candidates.shape[0] == 0:
            free_count = _DSSP_SURFACE_POINT_COUNT
        else:
            point_distances_sq = np.sum(
                np.square(surface_points[:, None, :] - candidates[None, :, :]),
                axis=-1,
                dtype=np.float32,
            )
            free_count = int(np.sum(np.all(candidate_radii_sq < point_distances_sq.astype(np.float64), axis=-1)))
        surface = np.float32(0.0)
        for _ in range(free_count):
            surface = np.float32(surface + _DSSP_SURFACE_POINT_WEIGHT)
        atom_areas[atom_index] = np.float32(np.float32(surface * surface_radius) * surface_radius)

    rows = []
    for residue_index, residue_name in enumerate(residue_names):
        area = np.float32(0.0)
        for atom_area in atom_areas[residue_indices == residue_index]:
            area = np.float32(area + atom_area)
        rows.append((residue_name, float(np.floor(area + np.float32(0.5)))))
    return rows


def _protein_instance_indices(structure: Structure) -> list[int]:
    r"""
    Canonical instance indices of protein polymer chains, in system order.
    """

    system = structure.system
    instance_entity = system.instance_entity.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    return [
        instance_index
        for instance_index, entity_index in enumerate(instance_entity)
        if int(entity_type[entity_index]) == ENTITY_PROTEIN
    ]


def _chain_sequence(structure: Structure, instance_index: int) -> str:
    r"""
    Return the protein sequence used by the cross-chain rASA cache.
    """

    system = structure.system
    start = int(system.instance_component_ptr[instance_index])
    stop = int(system.instance_component_ptr[instance_index + 1])
    return "".join(
        AMINO_ACID_THREE_TO_ONE.get(
            system.symbols[int(system.component_symbol[component_index])],
            "X",
        )
        for component_index in range(start, stop)
    )


def _polymer_atoms(structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Observed polymer atom coordinates, component ordinals, and instance indices.
    """

    system = structure.system
    instance_component_ptr = system.instance_component_ptr.detach().cpu()
    component_atom_ptr = system.component_atom_ptr.detach().cpu()
    component_counts = instance_component_ptr[1:] - instance_component_ptr[:-1]
    atom_counts = component_atom_ptr[1:] - component_atom_ptr[:-1]
    component_instance = torch.repeat_interleave(
        torch.arange(system.instance_symbol.numel()),
        component_counts,
    )
    component_local = torch.arange(system.component_symbol.numel()) - torch.repeat_interleave(
        instance_component_ptr[:-1],
        component_counts,
    )
    atom_component = torch.repeat_interleave(
        torch.arange(system.component_symbol.numel()),
        atom_counts,
    )
    atom_instance = component_instance[atom_component]
    atom_entity = system.instance_entity.detach().cpu()[atom_instance]
    entity_type = system.entity_type.detach().cpu()
    polymer_entity = torch.tensor(
        [int(current_entity_type) in _POLYMER_ENTITY_TYPES for current_entity_type in entity_type],
        dtype=torch.bool,
    )
    observed = structure.coordinate_mask.detach().cpu().bool() & polymer_entity[atom_entity]
    if not bool(observed.any()):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    return (
        structure.coordinates.detach().to(device="cpu", dtype=torch.float64)[observed].numpy(),
        component_local[atom_component][observed].numpy(),
        atom_instance[observed].numpy(),
    )


def _query_ball(coordinates: np.ndarray, cutoff_radius: float) -> list[list[int]]:
    r"""
    Find atom neighbors within ``cutoff_radius`` using a cKDTree.
    """

    from scipy import spatial

    tree = spatial.cKDTree(coordinates)
    return tree.query_ball_point(coordinates, p=2.0, r=cutoff_radius)


def _uniform_bin_centers(num_bins: int, minimum: float, maximum: float) -> torch.Tensor:
    r"""Return the midpoints of ``num_bins`` equal bins tiling ``[minimum, maximum]``."""

    edges = torch.linspace(minimum, maximum, num_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def _expected_bin_value(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    r"""Take the softmax expectation of ``logits`` over categorical distance/error ``centers``."""

    return logits.softmax(dim=-1) @ centers.to(logits.dtype)


def _contact_probability(distogram_logits: torch.Tensor) -> torch.Tensor:
    r"""Return the probability that each token pair is within 8 angstrom."""

    centers = _uniform_bin_centers(distogram_logits.shape[-1], 2.3125, 21.6875)
    return distogram_logits.softmax(dim=-1)[..., centers < 8.0].sum(dim=-1)


def _weighted_mean(
    mask: np.ndarray,
    value: np.ndarray,
    axis: int | tuple[int, ...],
) -> np.ndarray:
    r"""Compute a contact/frame-weighted mean; empty slices return NaN."""

    # An empty slice reports NaN rather than a fudged zero, so silence the accompanying NumPy warning
    # the same way :func:`_weighted_nanmean` does -- this is called per chain and per chain pair.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.mean(mask * value, axis=axis) / np.mean(mask, axis=axis)


def _weighted_nanmean(value: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    r"""Compute a NaN-aware weighted mean; empty slices return NaN."""

    mask = mask.astype(float)
    nan_idxs = np.where(np.isnan(value))
    mask_with_nan = mask.copy()
    mask_with_nan[nan_idxs] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(value * mask_with_nan, axis=axis) / np.nanmean(mask_with_nan, axis=axis)


def _chain_pair_pde(full_pde: np.ndarray, chain_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute per-chain-pair mean and minimum predicted distance error."""

    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        subset = full_pde[:, chain_indices == chain_first, :]
        for second, chain_second in enumerate(unique):
            block = subset[:, :, chain_indices == chain_second]
            mean[:, first, second] = np.mean(block, axis=(1, 2))
            minimum[:, first, second] = np.min(block, axis=(1, 2))
    return mean, minimum


def _pde_single(
    chain_indices: np.ndarray,
    full_pde: np.ndarray,
    contact_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute contact-weighted intra-chain, cross-chain, and per-token PDE."""

    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pde.shape[0]
    chain_indices_row = chain_indices[None]
    contact = contact_probs[None]
    ichain = np.zeros((num_samples, num_chains))
    xchain = np.zeros((num_samples, num_chains))
    for index, chain_index in enumerate(unique):
        mine = chain_indices_row == chain_index
        intra = mine[:, :, None] * mine[:, None, :] * contact
        cross = mine[:, :, None] * (~mine[:, None, :]) * contact
        ichain[:, index] = _weighted_mean(intra, full_pde, axis=(-2, -1))
        xchain[:, index] = _weighted_mean(cross, full_pde, axis=(-2, -1))
    full_chain = _weighted_mean(contact, full_pde, axis=(-1,))
    return ichain, xchain, full_chain


def _chain_pair_pae(
    full_pae: np.ndarray,
    chain_indices: np.ndarray,
    *,
    mask: np.ndarray,
    contact_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute per-chain-pair mean and minimum predicted aligned error."""

    if contact_probs is None:
        contact_probs = np.ones(full_pae.shape[1:], dtype=float)
    unique = np.unique(chain_indices)
    num_chains = len(unique)
    num_samples = full_pae.shape[0]
    mean = np.zeros((num_samples, num_chains, num_chains))
    minimum = np.zeros((num_samples, num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        subset = full_pae[:, chain_indices == chain_first, :]
        subset_mask = mask[chain_indices == chain_first, :]
        subset_contact = contact_probs[chain_indices == chain_first, :]
        for second, chain_second in enumerate(unique):
            block = subset[:, :, chain_indices == chain_second]
            block_mask = subset_mask[:, chain_indices == chain_second]
            block_contact = subset_contact[:, chain_indices == chain_second]
            (kept,) = np.where(block_mask.flatten() > 0)
            flat_block = block.reshape(num_samples, -1)
            flat_contact = block_contact.flatten()
            if not kept.size:
                mean[:, first, second] = np.nan
                minimum[:, first, second] = np.nan
            else:
                minimum[:, first, second] = np.min(flat_block[:, kept], axis=1)
                mean[:, first, second] = _weighted_mean(flat_contact[kept], flat_block[:, kept], axis=-1)
    return mean, minimum, unique


def _predicted_tm_score_np(
    tm_adjusted_pae: np.ndarray,
    pair_mask: np.ndarray,
    chain_index: np.ndarray,
    interface: bool,
) -> float:
    r"""Compute scalar pTM or ipTM for one sample."""

    if interface:
        pair_mask = pair_mask * (chain_index[:, None] != chain_index[None, :])
    if pair_mask.sum() == 0:
        return float("nan")
    normed = pair_mask / (np.finfo(tm_adjusted_pae.dtype).tiny + np.sum(pair_mask, axis=-1, keepdims=True))
    per_alignment = np.sum(tm_adjusted_pae * normed, axis=-1)
    return float(per_alignment.max())


def _chain_pairwise_iptm(
    tm_adjusted_pae: np.ndarray,
    pair_mask: np.ndarray,
    chain_index: np.ndarray,
) -> np.ndarray:
    r"""Compute the per-chain-pair ipTM matrix for one sample."""

    unique = list(np.unique(chain_index))
    num_chains = len(unique)
    pairwise = np.zeros((num_chains, num_chains))
    for first, chain_first in enumerate(unique):
        first_mask = chain_index == chain_first
        for offset, chain_second in enumerate(unique[first:]):
            second_mask = chain_index == chain_second
            selected = first_mask | second_mask
            (indices,) = np.where(selected)
            is_interface = chain_first != chain_second
            grid = np.ix_(indices, indices)
            value = _predicted_tm_score_np(
                tm_adjusted_pae[grid],
                pair_mask[grid],
                chain_index[selected],
                is_interface,
            )
            pairwise[first, first + offset] = value
            pairwise[first + offset, first] = value
    return pairwise


def _reduce_chain_pair(chain_pair: np.ndarray, *, agg_over_col: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""Reduce chain pairs to intra-chain and cross-chain means."""

    num_samples, num_chains, _ = chain_pair.shape
    ichain = chain_pair.diagonal(axis1=-2, axis2=-1)
    agg_axis = -1 if agg_over_col else -2
    weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
    weight -= np.eye(num_chains, dtype=float)
    xchain = _weighted_nanmean(chain_pair, mask=weight, axis=agg_axis)
    return ichain, xchain


def _pae_metrics(
    chain_indices: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray,
    contact_probs: np.ndarray,
    tm_adjusted_pae_interface: np.ndarray,
) -> dict[str, np.ndarray]:
    r"""Compute per-chain ipTM/PAE aggregates and the chain-pair ipTM matrix."""

    chain_pair_contact_weighted, _, _ = _chain_pair_pae(
        full_pae,
        chain_indices,
        mask=mask,
        contact_probs=contact_probs,
    )
    chain_pair_iptm = np.stack(
        [
            _chain_pairwise_iptm(
                tm_adjusted_pae=sample_tm,
                chain_index=chain_indices,
                pair_mask=mask,
            )
            for sample_tm in tm_adjusted_pae_interface
        ],
        axis=0,
    )

    def reduce(chain_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ichain, xchain_row = _reduce_chain_pair(chain_pair, agg_over_col=False)
        _, xchain_col = _reduce_chain_pair(chain_pair, agg_over_col=True)
        with np.errstate(invalid="ignore"):
            xchain = np.nanmean(np.stack([xchain_row, xchain_col], axis=0), axis=0)
        return ichain, xchain

    pae_ichain, pae_xchain = reduce(chain_pair_contact_weighted)
    iptm_ichain, iptm_xchain = reduce(chain_pair_iptm)
    return {
        "chain_pair_iptm": chain_pair_iptm,
        "iptm_ichain": iptm_ichain,
        "iptm_xchain": iptm_xchain,
        "pae_ichain": pae_ichain,
        "pae_xchain": pae_xchain,
    }


def _rank_metric(
    full_pae: np.ndarray,
    contact_probs: np.ndarray,
    *,
    rank_metric_eps: float,
) -> np.ndarray:
    r"""Compute a contact-weighted PAE ranking metric; higher is better."""

    return -np.sum(full_pae * contact_probs[None, :, :], axis=(-2, -1)) / (np.sum(contact_probs) + rank_metric_eps)
