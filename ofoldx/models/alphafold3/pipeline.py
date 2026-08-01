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
AlphaFold 3 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from multimolecule import graph

from ...data.molecular import AtomSite, ComponentRef
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, token_classes
from ...metrics import (
    categorical_expectation,
    categorical_mean,
    mean_partner_tm_scores,
    predicted_tm_score,
    resolved_probability_from_logits,
)
from ...modules.functional import weighted_mean
from ...msa import MSA, MsaCollection
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, floating_numpy
from ...templates import Template
from ..preparation import PreparedInput, decode_coordinates
from .model import (
    AlphaFold3Input,
    AlphaFold3Model,
    AlphaFold3SamplingOutput,
    alphafold3_pae_bin_centers,
    alphafold3_ranking_score,
    alphafold3_tm_adjusted_pae,
)
from .processor import ALPHAFOLD3_STANDARD_CHAINS, AlphaFold3Processor, alphafold3_chain_types


class AlphaFold3Pipeline(PairedPipeline[AlphaFold3Model, AlphaFold3Processor]):
    r"""
    Structure prediction pipeline for AlphaFold 3.
    """

    task = "fold"

    model_architecture = "alphafold3"

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
    ) -> PreparedInput[AlphaFold3Input]:
        r"""
        Prepare one structure and its evidence for AlphaFold 3 folding.

        Notes:
            Entity and polymer types determine chain types and must be stated. Preparation, cropping, MSA pairing,
            template resolution and reference conformers use the shared OFoldX preparation rules; AlphaFold 3 feature
            construction follows upstream conventions for the supplied inputs.

            Residue classes, MSA rows, profiles and template residue types use the 72 canonical classes. Conversion
            expands
            the 31 checkpoint rows, copying upstream-equivalent rows: B to D, Z to E, U to C, J and O to X, ligand
            tokens to
            UNK, and other nucleic classes to N. Profiles and mean deletion counts use each pool's supplied unpaired
            rows.
            All pools are paired by ``pair_msas``. Protein and RNA chains longer than four residues consume MSA rows;
            other
            polypeptide(L), RNA and DNA chains use their query, and other tokens use gaps.

            Token bonds represent the upstream polymer-ligand and ligand-ligand convention: one direction for covalent
            edges
            between components when one end is branched, macrolide or non-polymer. Each chain has four template
            positions,
            filled in resolution order. Only polypeptide(L) chains longer than four tokens consume templates. Backbone
            frames
            require N, CA and C; normalization clips frame norms at 1e-6.
        """
        return self.processor.prepare_input(structure, msa=msa, templates=templates)

    def fold(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> FoldResult:
        r"""
        Predict structures for one structure with AlphaFold 3.

        Args:
            structure: The target (read or constructed); its chemistry is consumed as is.
            msa: Optional MSA pools by instance, or an exact plan.
            templates: Template evidence aligned to the target's protein chains.
            num_samples: Number of diffusion samples. ``None`` uses five.
            seed: Optional diffusion seed, mutually exclusive with ``generator``.
            generator: Optional generator for per-recycle MSA permutations, initial
                coordinates, churn, and rigid augmentations. Mutually exclusive with ``seed``.
            num_recycles: Number of complete trunk passes. ``None`` uses the
                Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.

        Returns:
            Generation-ordered predictions decoded onto the target (every atom regenerated in a new frame).
            Coordinates, PAE, and PDE values are in angstroms; pLDDT is on ``0--100``.
        """

        generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(structure, msa=msa, templates=templates)
        output = self.forward(
            prepared.features,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: AlphaFold3Input,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
    ) -> AlphaFold3SamplingOutput:
        r"""Compute raw outputs for the prepared input."""

        return self.model.sample(
            input,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
        )

    def postprocess(self, prepared: PreparedInput[AlphaFold3Input], output: AlphaFold3SamplingOutput) -> FoldResult:
        r"""
        Decode one raw output without executing the model: every sample through ``decode_coordinates`` (evidence
        PREDICTED, scope every atom, new frame); confidence values on the prepared slot, token and chain axes.

        Notes:
            Every target atom is regenerated in a new frame; unpredicted atoms, including hydrogens, remain without
            coordinates and are counted. Confidence frames follow upstream: every polymer token, and ligand tokens whose
            two
            nearest reference atoms in the component are noncollinear with the token.

            Clash, disorder and relative accessibility use upstream ranking descriptors: a blockwise dense radius search
            for
            clashes, and DSSP 4.4.7 accessibility with Sander-Rost maxima for disorder. These descriptors do not
            establish
            physical bond exclusion, chemical validity or exact solvent accessibility.
        """

        input = prepared.features
        samples = output.coordinates[0].detach().cpu().double().numpy()
        decoded = [decode_coordinates(prepared, sample)[0] for sample in samples]
        atom_axis, token_axis, chain_axis = _axes(prepared)
        atom_plddt = output.plddt[0].detach().cpu()
        pae_logits = output.pae_logits[0].detach().cpu()
        pde_logits = output.pde_logits[0].detach().cpu()
        resolved_logits = output.resolved_logits[0].detach().cpu()
        distogram_logits = output.distogram_logits[0].detach().cpu()
        ptm, iptm, ptm_iptm_average = (
            value[0].detach().cpu() for value in self.model.confidence_summary(output, input)
        )

        pae = categorical_expectation(
            pae_logits.softmax(dim=-1), alphafold3_pae_bin_centers(pae_logits.shape[-1]).to(pae_logits)
        )
        pde = categorical_mean(pde_logits, end=32.0)
        resolved_probability = resolved_probability_from_logits(resolved_logits)
        contact_probability = floating_numpy(
            _contact_probability(
                distogram_logits,
                first_break=self.model.config.distogram.first_break,
                last_break=self.model.config.distogram.last_break,
            )
        )

        # Chain-level confidence summaries operate on decoded per-sample arrays.
        full_pae = floating_numpy(pae)
        full_pde = floating_numpy(pde)
        chain_indices = input.chain_index[0].detach().cpu().numpy().astype(np.int64)
        frame_mask = input.frame_mask[0].detach().cpu().bool().numpy()
        token_mask = np.ones(len(chain_indices), dtype=bool)
        multi_chain = len(chain_axis) > 1

        pae_single_mask = frame_mask[:, None] & token_mask[None, :]
        scored_pae_logits = torch.where(torch.from_numpy(pae_single_mask)[None, ..., None], pae_logits, 0)
        tm_interface = floating_numpy(
            alphafold3_tm_adjusted_pae(
                scored_pae_logits,
                token_mask=torch.from_numpy(token_mask).unsqueeze(0).expand(pae_logits.shape[0], -1),
                chain_index=torch.from_numpy(chain_indices).unsqueeze(0).expand(pae_logits.shape[0], -1),
                interface=True,
            )
        )

        chain_pair_pde_mean, chain_pair_pde_min = _chain_pair_pde(full_pde, chain_indices)
        _, chain_pair_pae_min, _ = _chain_pair_pae(
            full_pae, chain_indices, mask=pae_single_mask, backend=self.model.backend
        )
        pde_ichain, pde_xchain, pde_full_chain = _pde_single(
            chain_indices, full_pde, contact_probability, backend=self.model.backend
        )
        chain_metrics = _pae_metrics(
            chain_indices, full_pae, pae_single_mask, contact_probability, tm_interface, backend=self.model.backend
        )
        ranking_confidence_pae = _rank_metric(
            full_pae,
            contact_probability * frame_mask[:, None].astype(float),
            rank_metric_eps=self.model.config.rank_metric_eps,
        )

        predictions: list[FoldPrediction] = []
        for sample_index, structure in enumerate(decoded):
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
                        "plddt": Metric(values=floating_numpy(atom_plddt[sample_index]), axis=atom_axis),
                        "resolved_probability": Metric(
                            values=floating_numpy(resolved_probability[sample_index]),
                            axis=atom_axis,
                        ),
                        "predicted_distance_error": Metric(
                            values=pde_full_chain[sample_index],
                            axis=token_axis,
                            unit="angstrom",
                        ),
                        "chain_ptm": Metric(values=chain_metrics["chain_ptm"][sample_index], axis=chain_axis),
                        "chain_mean_partner_iptm": Metric(
                            values=chain_metrics["chain_mean_partner_iptm"][sample_index], axis=chain_axis
                        ),
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
                            values=floating_numpy(pae[sample_index]),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "pde": PairMetric(
                            values=floating_numpy(pde[sample_index]),
                            row_axis=token_axis,
                            column_axis=token_axis,
                            unit="angstrom",
                        ),
                        "contact_probs": PairMetric(
                            values=contact_probability,
                            row_axis=token_axis,
                            column_axis=token_axis,
                        ),
                        "chain_pair_complex_iptm": PairMetric(
                            values=chain_metrics["chain_pair_complex_iptm"][sample_index],
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

    __call__ = task_call(fold)


def _owner(pointer: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _axes(
    prepared: PreparedInput[AlphaFold3Input],
) -> tuple[tuple[AtomSite, ...], tuple[ComponentRef | AtomSite, ...], tuple[str, ...]]:
    r"""The prepared slot axis (one atom each), token axis (a residue token is its component, an atom token its atom)
    and chain axis (the instances with a token, in system order)."""
    system = prepared.context.structure.system
    slots, layout = prepared.context.slots, prepared.context.slots.layout
    atom_component = _owner(system.component_atom_ptr)
    names = system.text("atom", system.atom_symbol)
    atom_axis = tuple(
        AtomSite(ref.instance, ref.component, name)
        for ref, name in zip(system.component_refs(atom_component[slots.atom]), names[slots.atom].tolist())
    )
    counts = np.bincount(layout.token_component, minlength=len(system.component_symbol))
    atomic = (counts[layout.token_component] > 1) | (layout.token_class == TOKEN_INDEX["OTHER"])
    token_axis = tuple(
        AtomSite(ref.instance, ref.component, name) if atom else ref
        for ref, name, atom in zip(
            system.component_refs(layout.token_component),
            names[layout.token_centre_atom].tolist(),
            atomic.tolist(),
        )
    )
    instances = np.unique(_owner(system.instance_component_ptr)[layout.token_component])
    return atom_axis, token_axis, tuple(system.text("instance", system.instance_symbol[instances]).tolist())


# Maximum accessible surface areas from Sander & Rost 1994
# (https://doi.org/10.1002/prot.340200303), used by AlphaFold-RSA; any other class reads ALA, as upstream.
_MAX_ACCESSIBLE_SURFACE_AREA = {
    TOKEN_INDEX[letter]: area
    for letter, area in {
        "A": 106.0,
        "R": 248.0,
        "N": 157.0,
        "D": 163.0,
        "C": 135.0,
        "Q": 198.0,
        "E": 194.0,
        "G": 84.0,
        "H": 184.0,
        "I": 169.0,
        "L": 164.0,
        "K": 205.0,
        "M": 188.0,
        "F": 197.0,
        "P": 136.0,
        "S": 130.0,
        "T": 142.0,
        "W": 227.0,
        "Y": 222.0,
        "V": 142.0,
    }.items()
}

# AlphaFold-RSA rASA disorder threshold.
_RASA_DISORDER_CUTOFF = 0.581
_RASA_WINDOW = 25


def alphafold3_fraction_disordered(
    structure: Structure, *, rasa_disorder_cutoff: float = _RASA_DISORDER_CUTOFF
) -> float:
    r"""
    AlphaFold-RSA disorder (upstream ``confidences.fraction_disordered``): the fraction of protein residues whose
    windowed relative accessible surface area exceeds ``rasa_disorder_cutoff``.

    Each polypeptide(L) chain is scored in isolation with the DSSP 4.4.7 accessibility calculation over its residues
    with placed N, CA, C and O, each area divided by its canonical class's Sander--Rost maximum (ALA for a class
    without one), clipped to one and averaged over a mirrored window of 25 residues; chains with the same class
    sequence share the first one's values, as upstream. The 0.581 operating point was fitted to CAID DisProt; it is a
    ranking descriptor, not a physical determination of disorder.

    Returns:
        The fraction in ``[0, 1]``; zero without a scored residue.
    """

    system = structure.system
    classes = token_classes(system).token
    pointer = system.instance_component_ptr
    rasa: list[float] = []
    seen: dict[tuple[int, ...], list[float]] = {}
    for instance in np.flatnonzero(alphafold3_chain_types(system) == "polypeptide(L)").tolist():
        key = tuple(classes[pointer[instance] : pointer[instance + 1]].tolist())
        if key not in seen:
            seen[key] = _windowed_solvent_accessible_area(structure, instance, classes)
        rasa.extend(seen[key])
    return float(np.mean(np.asarray(rasa) > rasa_disorder_cutoff)) if rasa else 0.0


# Query atoms per dense block of the clash search.
_CLASH_BLOCK = 1024


def alphafold3_has_clash(
    structure: Structure,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
) -> bool:
    r"""
    Whether the structure has at least one clashing polymer chain (upstream ``confidences.has_clash``).

    Over the placed atoms of polypeptide(L), RNA and DNA chains, an atom clashes when it lies within
    ``cutoff_radius`` angstrom of an atom of another chain or of a residue more than one position away; a chain
    overlaps when more than ``min_clashes_for_overlap`` of its atoms, or more than ``min_fraction_for_overlap`` of
    them, clash. Distances are in double precision, as upstream's ``cKDTree``, but the radius search is strict
    (``<``) where ``query_ball_point`` is inclusive. The search is dense, ``_CLASH_BLOCK`` query atoms at a time
    against every placed polymer atom: quadratic time, memory bounded by the block.
    """

    system = structure.system
    atom_component = _owner(system.component_atom_ptr)
    pointer = system.instance_component_ptr
    atom_instance = _owner(pointer)[atom_component]
    polymer = np.isin(alphafold3_chain_types(system), ALPHAFOLD3_STANDARD_CHAINS)[atom_instance]
    polymer &= structure.coordinate_mask
    atoms = np.flatnonzero(polymer)
    if not len(atoms):
        return False
    residue = atom_component[atoms] - pointer[atom_instance[atoms]]
    chain = atom_instance[atoms]
    points = torch.from_numpy(structure.coordinates[atoms])
    per_atom = np.zeros(len(atoms), dtype=bool)
    for start in range(0, len(atoms), _CLASH_BLOCK):
        query, partner = graph.radius(
            points, points[start : start + _CLASH_BLOCK], cutoff_radius, max_num_neighbors=len(atoms)
        )
        query = query.numpy() + start
        partner = partner.numpy()
        clashing = (np.abs(residue[query] - residue[partner]) > 1) | (chain[query] != chain[partner])
        per_atom[query[clashing]] = True
    totals = np.bincount(chain, minlength=len(pointer) - 1)
    clashes = np.bincount(chain[per_atom], minlength=len(pointer) - 1)
    overlap = (clashes > min_clashes_for_overlap) | (clashes > min_fraction_for_overlap * totals)
    return bool(overlap[totals > 0].any())


def _windowed_solvent_accessible_area(structure: Structure, instance: int, classes: np.ndarray) -> list[float]:
    r"""Windowed per-residue rASA for one protein chain (AlphaFold-RSA): areas normalised by class, clipped to one,
    averaged over a reflect-padded window."""

    accessible = _dssp_accessible_area(structure, instance)
    if not accessible:
        return []
    rasa = [min(area / _MAX_ACCESSIBLE_SURFACE_AREA.get(int(classes[c]), 106.0), 1.0) for c, area in accessible]
    half_window = (_RASA_WINDOW - 1) // 2
    padded = np.pad(np.asarray(rasa, dtype=float), (half_window, half_window), "reflect")
    return (np.convolve(padded, np.ones(_RASA_WINDOW), "valid") / _RASA_WINDOW).tolist()


# DSSP 4.4.7 accessibility parameters.
_DSSP_ATOM_RADII = {"N": 1.65, "CA": 1.87, "C": 1.76, "O": 1.4}
_DSSP_SIDE_CHAIN_RADIUS = 1.8
_DSSP_WATER_RADIUS = 1.4
_NUM_DSSP_SURFACE_POINTS = 401


def _dssp_surface_dots() -> np.ndarray:
    r"""
    Return the 401-point Fibonacci sphere used by DSSP 4.4.7.
    """

    n = (_NUM_DSSP_SURFACE_POINTS - 1) // 2
    indices = np.arange(-n, n + 1, dtype=np.int32)
    golden_ratio = np.float32((1.0 + np.sqrt(np.float32(5.0))) / 2.0)
    latitude = np.arcsin(np.float32(2.0) * indices.astype(np.float32) / np.float32(_NUM_DSSP_SURFACE_POINTS))
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
_DSSP_SURFACE_POINT_WEIGHT = np.float32(4.0 * np.pi / _NUM_DSSP_SURFACE_POINTS)


def _dssp_accessible_area(structure: Structure, instance: int) -> list[tuple[int, float]]:
    r"""
    ``(component, area)`` rows of one chain by the DSSP 4.4.7 accessibility calculation (its atom radii, 401-point
    Fibonacci sphere, water radius, overlap test and integer rounding, in single precision as DSSP).

    DSSP keeps the residues with placed N, CA, C and O and scores the chain in isolation over their placed heavy atoms.
    """

    system = structure.system
    names = system.text("atom", system.atom_symbol)
    heavy = system.atomic_number != 1
    placed = structure.coordinate_mask
    pointer = system.component_atom_ptr
    atom_coordinates: list[np.ndarray] = []
    atom_radii: list[float] = []
    atom_residue_indices: list[int] = []
    components: list[int] = []
    for component in range(
        int(system.instance_component_ptr[instance]), int(system.instance_component_ptr[instance + 1])
    ):
        atoms = np.arange(pointer[component], pointer[component + 1])
        by_name = {str(names[a]): a for a in atoms}
        if not all(name in by_name and placed[by_name[name]] for name in _DSSP_ATOM_RADII):
            continue
        side_chain = [a for a in atoms if heavy[a] and placed[a] and str(names[a]) not in _DSSP_ATOM_RADII]
        residue_index = len(components)
        components.append(component)
        for name, radius in _DSSP_ATOM_RADII.items():
            atom_coordinates.append(structure.coordinates[by_name[name]])
            atom_radii.append(radius)
            atom_residue_indices.append(residue_index)
        atom_coordinates.extend(structure.coordinates[side_chain])
        atom_radii.extend([_DSSP_SIDE_CHAIN_RADIUS] * len(side_chain))
        atom_residue_indices.extend([residue_index] * len(side_chain))
    if not components:
        return []

    coordinates = np.asarray(atom_coordinates, dtype=np.float32)
    scalar = coordinates.dtype.type
    radii = np.asarray(atom_radii, dtype=coordinates.dtype)
    residue_indices = np.asarray(atom_residue_indices, dtype=np.int64)
    atom_areas = np.empty(coordinates.shape[0], dtype=coordinates.dtype)
    surface_radii = radii + scalar(_DSSP_WATER_RADIUS)
    occlusion_radii = radii + scalar(_DSSP_WATER_RADIUS)

    for atom_index, (coordinate, surface_radius, occlusion_radius) in enumerate(
        zip(coordinates, surface_radii, occlusion_radii)
    ):
        relative = coordinates - coordinate
        center_distances_sq = np.sum(relative * relative, axis=-1)
        candidate_mask = (center_distances_sq > 0.0001) & (
            center_distances_sq < np.square(occlusion_radius + occlusion_radii)
        )
        candidates = relative[candidate_mask]
        candidate_radii_sq = np.square(occlusion_radii[candidate_mask])
        surface_points = _DSSP_SURFACE_DOTS.astype(coordinates.dtype, copy=False) * surface_radius
        if candidates.shape[0] == 0:
            free_count = _NUM_DSSP_SURFACE_POINTS
        else:
            point_distances_sq = np.sum(
                np.square(surface_points[:, None, :] - candidates[None, :, :]),
                axis=-1,
            )
            free_count = int(np.sum(np.all(candidate_radii_sq < point_distances_sq, axis=-1)))
        surface = scalar(0.0)
        for _ in range(free_count):
            surface = surface + scalar(_DSSP_SURFACE_POINT_WEIGHT)
        atom_areas[atom_index] = surface * surface_radius * surface_radius

    rows = []
    for residue_index, component in enumerate(components):
        area = scalar(0.0)
        for atom_area in atom_areas[residue_indices == residue_index]:
            area = area + atom_area
        rows.append((component, float(np.floor(area + scalar(0.5)))))
    return rows


def _contact_probability(
    distogram_logits: torch.Tensor,
    *,
    first_break: float,
    last_break: float,
) -> torch.Tensor:
    r"""Return the AlphaFold3 mass of distance bins with top at most8.001 angstrom."""

    breaks = torch.linspace(
        first_break,
        last_break,
        distogram_logits.shape[-1] - 1,
        device=distogram_logits.device,
        dtype=distogram_logits.dtype,
    )
    bin_tops = torch.cat((breaks, breaks[-1:] + (breaks[-1] - breaks[-2])))
    return categorical_expectation(distogram_logits.softmax(dim=-1), bin_tops <= 8.001)


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    dim: int | tuple[int, ...],
    *,
    backend: str = "auto",
) -> np.ndarray:
    r"""Compute a contact/frame-weighted mean; empty slices return NaN."""

    values, weights = np.broadcast_arrays(values, weights)
    dims = dim if isinstance(dim, tuple) else (dim,)
    dims = tuple(index % values.ndim for index in dims)
    retained = tuple(index for index in range(values.ndim) if index not in dims)
    order = retained + dims
    shape = tuple(values.shape[index] for index in retained) + (int(np.prod([values.shape[index] for index in dims])),)
    observations = torch.from_numpy(values.transpose(order).reshape(shape))
    coefficients = torch.from_numpy(weights.transpose(order).reshape(shape))
    return weighted_mean(observations, coefficients, backend=backend).numpy()


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray, dim: int, *, backend: str = "auto") -> np.ndarray:
    r"""Compute a NaN-aware weighted mean; empty slices return NaN."""

    return _weighted_mean(values, np.where(np.isnan(values), 0.0, weights), dim=dim, backend=backend)


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
    *,
    backend: str = "auto",
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
        ichain[:, index] = _weighted_mean(full_pde, intra, dim=(-2, -1), backend=backend)
        xchain[:, index] = _weighted_mean(full_pde, cross, dim=(-2, -1), backend=backend)
    full_chain = _weighted_mean(full_pde, contact, dim=(-1,), backend=backend)
    return ichain, xchain, full_chain


def _chain_pair_pae(
    full_pae: np.ndarray,
    chain_indices: np.ndarray,
    *,
    mask: np.ndarray,
    contact_probs: np.ndarray | None = None,
    backend: str = "auto",
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
                mean[:, first, second] = _weighted_mean(
                    flat_block[:, kept], flat_contact[kept], dim=-1, backend=backend
                )
    return mean, minimum, unique


def _chain_pairwise_iptm(
    tm_adjusted_pae: np.ndarray,
    pair_mask: np.ndarray,
    chain_index: np.ndarray,
    *,
    backend: str = "auto",
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
            scope_mask = pair_mask[grid]
            if is_interface:
                scope_chains = chain_index[selected]
                scope_mask = scope_mask * (scope_chains[:, None] != scope_chains[None, :])
            value = (
                float(
                    predicted_tm_score(
                        torch.from_numpy(tm_adjusted_pae[grid]),
                        pair_weights=torch.from_numpy(scope_mask),
                        backend=backend,
                    ).item()
                )
                if scope_mask.any()
                else float("nan")
            )
            pairwise[first, first + offset] = value
            pairwise[first + offset, first] = value
    return pairwise


def _reduce_chain_pair(
    chain_pair: np.ndarray, *, agg_over_col: bool, backend: str = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    r"""Reduce chain pairs to intra-chain and cross-chain means."""

    num_samples, num_chains, _ = chain_pair.shape
    ichain = chain_pair.diagonal(axis1=-2, axis2=-1)
    agg_dim = -1 if agg_over_col else -2
    weight = np.ones((num_samples, num_chains, num_chains), dtype=float)
    weight -= np.eye(num_chains, dtype=float)
    xchain = _weighted_nanmean(chain_pair, weights=weight, dim=agg_dim, backend=backend)
    return ichain, xchain


def _pae_metrics(
    chain_indices: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray,
    contact_probs: np.ndarray,
    tm_adjusted_pae_interface: np.ndarray,
    *,
    backend: str = "auto",
) -> dict[str, np.ndarray]:
    r"""Compute per-chain ipTM/PAE aggregates and the chain-pair ipTM matrix."""

    chain_pair_contact_weighted, _, _ = _chain_pair_pae(
        full_pae, chain_indices, mask=mask, contact_probs=contact_probs, backend=backend
    )
    chain_pair_complex_iptm = np.stack(
        [
            _chain_pairwise_iptm(tm_adjusted_pae=sample_tm, chain_index=chain_indices, pair_mask=mask, backend=backend)
            for sample_tm in tm_adjusted_pae_interface
        ],
        axis=0,
    )

    def reduce(chain_pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ichain, xchain_row = _reduce_chain_pair(chain_pair, agg_over_col=False, backend=backend)
        _, xchain_col = _reduce_chain_pair(chain_pair, agg_over_col=True, backend=backend)
        partner_means = np.stack([xchain_row, xchain_col], axis=0)
        xchain = _weighted_nanmean(partner_means, np.ones_like(partner_means), dim=0, backend=backend)
        return ichain, xchain

    pae_ichain, pae_xchain = reduce(chain_pair_contact_weighted)
    chain_ptm = chain_pair_complex_iptm.diagonal(axis1=-2, axis2=-1).copy()
    chain_mean_partner_iptm = mean_partner_tm_scores(torch.from_numpy(chain_pair_complex_iptm), backend=backend).numpy()
    # The union interface matrix excludes intra-chain pTM, reported separately.
    diagonal = np.arange(chain_pair_complex_iptm.shape[-1])
    chain_pair_complex_iptm[:, diagonal, diagonal] = 0.0
    return {
        "chain_pair_complex_iptm": chain_pair_complex_iptm,
        "chain_ptm": chain_ptm,
        "chain_mean_partner_iptm": chain_mean_partner_iptm,
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

    observed_pae = np.where(contact_probs[None, :, :] != 0, full_pae, 0)
    return -np.sum(observed_pae * contact_probs[None, :, :], axis=(-2, -1)) / (np.sum(contact_probs) + rank_metric_eps)
