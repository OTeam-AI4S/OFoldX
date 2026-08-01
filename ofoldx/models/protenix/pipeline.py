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
Protenix structure-prediction Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from ...data.molecular import AtomSite, ComponentRef
from ...data.structure import Structure
from ...msa import MSA, MsaCollection
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, floating_numpy
from ...templates import Template
from ..preparation import PreparedInput, decode_coordinates
from .model import (
    ProtenixAtomContact,
    ProtenixInput,
    ProtenixModel,
    ProtenixPocket,
    ProtenixSamplingOutput,
    ProtenixTfgGuidance,
    ProtenixTokenContact,
    protenix_confidence_summary,
)
from .processor import (
    ProtenixConstraintProcessor,
    ProtenixProcessor,
    ProtenixStructureConditionProcessor,
    ProtenixTemplateProcessor,
    atom_tokens,
)

_ATOM_METRICS = ("plddt", "resolved_probability")
_CHAIN_METRICS = ("chain_plddt", "chain_ptm", "chain_mean_partner_iptm", "chain_gpde")
_TOKEN_PAIRS = ("pae", "pde")
_CHAIN_PAIRS = ("chain_pair_complex_iptm", "chain_pair_mean_partner_iptm", "chain_pair_gpde", "chain_pair_plddt")
_ANGSTROM = frozenset(("chain_gpde", "pae", "pde", "chain_pair_gpde"))
_PER_SAMPLE = (*_ATOM_METRICS, *_CHAIN_METRICS, *_TOKEN_PAIRS, *_CHAIN_PAIRS)


class ProtenixPipelineInput(NamedTuple):
    r"""One prepared task and the training-free guidance retained for sampling."""

    prepared: PreparedInput[ProtenixInput]
    tfg_guidance: ProtenixTfgGuidance | None


class ProtenixPipeline(PairedPipeline[ProtenixModel, ProtenixProcessor]):
    r"""
    Structure prediction pipeline for Protenix.

    Notes:
        All four registered variants prepare ``ProtenixPipelineInput``, which retains the canonical prepared task and
        its training-free guidance policy. Set ``tfg_guidance`` during preparation; forward execution uses that saved
        policy, and postprocessing decodes the retained task. Preparation uses a NumPy generator for reference-position
        augmentation and constraint draws; sampling uses a PyTorch generator.

        Template variants accept aligned templates and RNA-MSA evidence. Constraint variants accept atom contacts,
        token contacts, pockets, and substructures. Their substructure branch digitizes resolved token-center distances
        into the released four-channel representation. The mini template checkpoint instead aligns representative-atom
        coordinates and applies a noisy-structure pair update on each recycle. Direct callers of
        ``ProtenixNoisyStructureEmbedder`` can enable ``add_noise`` and supply its PyTorch generator.

        Per-call ``num_recycles``, ``num_sampling_steps``, ``churn_rate``, and ``step_scale`` override the saved
        checkpoint configuration; ``None`` preserves it. Ranking is ``0.8 * ipTM + 0.2 * pTM - 100 * has_clash``
        without a disorder estimate.
    """

    task = "fold"

    model_architecture = "protenix"

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: np.random.Generator | None = None,
    ) -> ProtenixPipelineInput:
        r"""
        Prepare one structure for the Protenix sampling graph (``generator`` drives the reference-position
        augmentation and the constraint draws).

        Notes:
            Represented atoms require known elements and formal charges. The shared preparation rules determine
            atom grouping, input checks, reference conformers and constraint features. Residue and profile features use
            72
            canonical classes, with checkpoint axes expanded during conversion.

            MSA pools are bound by instance and paired after cropping; an explicit ``MsaCollection`` plan is used
            unchanged.
            Upstream MSA letter mappings are retained: B, Z and U read as D, E and C; J and O read as X. Profiles and
            deletion
            means use unique unpaired rows. Chains of at most four residues use query-only rows, and a non-protein
            chain's
            otherwise all-gap rows contain its query. Templates resolve through ``resolve_template``; a protein chain
            longer
            than four residues consumes at most four templates.

            The token-bond feature contains covalent token edges except junctions between neighboring chain components.
        """
        prepared = self.processor.prepare_input(structure, msa=msa, tfg=tfg_guidance is not None, generator=generator)
        return ProtenixPipelineInput(prepared, tfg_guidance)

    def fold(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        **conditions,
    ) -> FoldResult:
        r"""
        Predict structures for one structure with Protenix.

        Args:
            structure: The target (read or constructed); its chemistry is consumed as is.
            msa: Optional MSA pools by instance, or an exact plan.
            num_samples: Number of structure predictions. ``None`` uses five.
            seed: Optional seed for preparation and coordinate sampling. Mutually exclusive with ``generator``.
            generator: Optional generator advanced by coordinate sampling. Mutually exclusive with ``seed``.
            num_recycles: Number of representation recycling iterations. ``None`` uses the Model configuration.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            churn_rate: Per-call extra-noise fraction. ``None`` uses the Model configuration.
            step_scale: Per-call solver-step multiplier. ``None`` uses the Model configuration.
            tfg_guidance: Training-free chemical guidance. ``None`` disables it; ``ProtenixTfgGuidance()`` uses the
                released enabled defaults.
            conditions (Any): The variant's further preparation inputs (its ``prepare``).

        Returns:
            Generation-ordered predictions decoded onto the target (every atom regenerated in a new frame).
            Structure, PAE, PDE and gPDE values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        torch_generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(
            structure,
            msa=msa,
            tfg_guidance=tfg_guidance,
            generator=None if seed is None else np.random.default_rng(seed),
            **conditions,
        )
        output = self.forward(
            prepared,
            num_samples=num_samples,
            generator=torch_generator,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: ProtenixPipelineInput,
        *,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        churn_rate: float | None = None,
        step_scale: float | None = None,
    ) -> ProtenixSamplingOutput:
        r"""Compute raw outputs with the guidance retained during preparation."""

        return self.model.sample(
            input.prepared.features,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            churn_rate=churn_rate,
            step_scale=step_scale,
            tfg_guidance=input.tfg_guidance,
        )

    def postprocess(self, input: ProtenixPipelineInput, output: ProtenixSamplingOutput) -> FoldResult:
        r"""
        Decode one raw output without executing the model: every sample through ``decode_coordinates`` (evidence
        PREDICTED, scope every atom, new frame); confidence values on the prepared atom, token and chain axes.

        Notes:
            Prepared atom correspondences place predictions on the target, regenerating every atom in a new frame.
            Ranking is ``0.8 * ipTM + 0.2 * pTM - 100 * has_clash``. No disorder estimator or disorder value is
            included.
        """

        prepared = input.prepared
        model_input = prepared.features
        atom_valid = model_input.atom_mask[0].detach().cpu().numpy().astype(bool)
        token_valid = model_input.token_mask[0].detach().cpu().numpy().astype(bool)
        samples = output.coordinates[0].detach().cpu().double().numpy()
        decoded = [decode_coordinates(prepared, sample, valid=atom_valid)[0] for sample in samples]
        confidence = protenix_confidence_summary(output, model_input, backend=self.model.backend)
        atom_axis, token_axis, chain_axis = _axes(prepared, atom_valid, token_valid)
        contact = floating_numpy(confidence["contact_probability"])
        predictions = []
        for index, structure in enumerate(decoded):
            value = {name: floating_numpy(confidence[name][index]) for name in _PER_SAMPLE}
            scores = {
                "mean_plddt": Score(float(confidence["plddt"][index].mean()), preferred="higher"),
                "gpde": Score(float(confidence["gpde"][index]), unit="angstrom", preferred="lower"),
                "ptm": Score(float(confidence["ptm"][index]), preferred="higher"),
                "iptm": Score(float(confidence["iptm"][index]), preferred="higher"),
                "has_clash": Score(float(confidence["has_clash"][index]), preferred="lower"),
                "ranking_score": Score(float(confidence["ranking_score"][index]), preferred="higher"),
            }
            metrics = {name: Metric(values=value[name], axis=atom_axis) for name in _ATOM_METRICS}
            metrics |= {
                name: Metric(values=value[name], axis=chain_axis, unit="angstrom" if name in _ANGSTROM else None)
                for name in _CHAIN_METRICS
            }
            pair_metrics = {
                name: PairMetric(values=value[name], row_axis=token_axis, column_axis=token_axis, unit="angstrom")
                for name in _TOKEN_PAIRS
            }
            pair_metrics["contact_probs"] = PairMetric(values=contact, row_axis=token_axis, column_axis=token_axis)
            pair_metrics |= {
                name: PairMetric(
                    values=value[name],
                    row_axis=chain_axis,
                    column_axis=chain_axis,
                    unit="angstrom" if name in _ANGSTROM else None,
                )
                for name in _CHAIN_PAIRS
            }
            predictions.append(FoldPrediction(structure, scores, metrics, pair_metrics))
        return FoldResult(predictions=tuple(predictions))

    __call__ = task_call(fold)


class ProtenixStructureConditionPipeline(ProtenixPipeline):
    r"""
    Structure-conditioned prediction pipeline for Protenix mini template.
    """

    model_architecture = "protenix_structure_condition"
    processor: ProtenixStructureConditionProcessor

    def prepare(
        self,
        structure: Structure,
        *,
        structure_condition: Structure | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: np.random.Generator | None = None,
    ) -> ProtenixPipelineInput:
        r"""Prepare one structure with the representative-atom coordinates of ``structure_condition``."""
        prepared = self.processor.prepare_input(
            structure,
            structure_condition=structure_condition,
            msa=msa,
            tfg=tfg_guidance is not None,
            generator=generator,
        )
        return ProtenixPipelineInput(prepared, tfg_guidance)


class ProtenixConstraintPipeline(ProtenixPipeline):
    r"""
    Constrained structure prediction pipeline for Protenix.
    """

    model_architecture = "protenix_constraint"
    processor: ProtenixConstraintProcessor

    def prepare(
        self,
        structure: Structure,
        *,
        atom_contacts: Sequence[ProtenixAtomContact] = (),
        token_contacts: Sequence[ProtenixTokenContact] = (),
        pocket: ProtenixPocket | None = None,
        substructure: Structure | None = None,
        substructure_coordinate_noise_scale: float = 0.0,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: np.random.Generator | None = None,
    ) -> ProtenixPipelineInput:
        r"""Prepare one structure with the four constraint channels (``ProtenixConstraintProcessor.prepare``)."""
        prepared = self.processor.prepare_input(
            structure,
            atom_contacts=atom_contacts,
            token_contacts=token_contacts,
            pocket=pocket,
            substructure=substructure,
            substructure_coordinate_noise_scale=substructure_coordinate_noise_scale,
            msa=msa,
            tfg=tfg_guidance is not None,
            generator=generator,
        )
        return ProtenixPipelineInput(prepared, tfg_guidance)


class ProtenixTemplatePipeline(ProtenixPipeline):
    r"""
    Template- and RNA-MSA-capable structure prediction pipeline for Protenix v1 and v2.
    """

    model_architecture = "protenix_template"
    processor: ProtenixTemplateProcessor

    def prepare(
        self,
        structure: Structure,
        *,
        templates: Sequence[Template] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        tfg_guidance: ProtenixTfgGuidance | None = None,
        generator: np.random.Generator | None = None,
    ) -> ProtenixPipelineInput:
        r"""Prepare one structure with protein templates aligned to its chains."""
        prepared = self.processor.prepare_input(
            structure, templates=templates, msa=msa, tfg=tfg_guidance is not None, generator=generator
        )
        return ProtenixPipelineInput(prepared, tfg_guidance)


def _axes(
    prepared: PreparedInput[ProtenixInput], atom_valid: NDArray[np.bool_], token_valid: NDArray[np.bool_]
) -> tuple[tuple[AtomSite, ...], tuple[ComponentRef | AtomSite, ...], tuple[str, ...]]:
    r"""
    The axes of the confidence values: the valid slot atoms, the valid tokens (a residue token is its component, an
    atom token its atom) and the chains (the instances of the valid tokens, in system order).
    """

    system = prepared.context.structure.system
    layout = prepared.context.slots.layout
    owner = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))

    def sites(atoms: NDArray[np.int64]) -> list[AtomSite]:
        names = system.text("atom", system.atom_symbol[atoms]).tolist()
        return [AtomSite(r.instance, r.component, n) for r, n in zip(system.component_refs(owner[atoms]), names)]

    atomic = atom_tokens(layout)
    centres = sites(layout.token_centre_atom)
    refs = system.component_refs(layout.token_component)
    tokens = [centre if atom else ref for centre, ref, atom in zip(centres, refs, atomic.tolist())]
    instance = np.searchsorted(system.instance_component_ptr, layout.token_component[token_valid], side="right") - 1
    chains = system.text("instance", system.instance_symbol[np.unique(instance)]).tolist()
    return (
        tuple(sites(prepared.context.slots.atom[atom_valid])),
        tuple(t for t, valid in zip(tokens, token_valid.tolist()) if valid),
        tuple(chains),
    )
