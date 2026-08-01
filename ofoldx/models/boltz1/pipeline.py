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
Boltz-1 folding Pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from ...data.molecular import AtomSite, ComponentRef
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX
from ...metrics import resolved_probability_from_logits
from ...msa import MSA, MsaCollection
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, floating_numpy
from ..preparation import PreparedInput, decode_coordinates
from .model import Boltz1Input, Boltz1Model, Boltz1Pocket, Boltz1SamplingWithConfidenceOutput
from .processor import Boltz1Processor

_HIGHER = ("mean_plddt", "complex_iplddt", "ptm", "iptm", "ligand_iptm", "protein_iptm", "confidence_score")
_LOWER = ("complex_pde", "complex_ipde")


class Boltz1Pipeline(PairedPipeline[Boltz1Model, Boltz1Processor]):
    r"""
    Structure prediction pipeline for Boltz 1.
    """

    task = "fold"

    model_architecture = "boltz1"

    def prepare(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        pocket: Boltz1Pocket | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[Boltz1Input]:
        r"""
        Prepare one structure and its evidence for Boltz-1 folding (``generator`` drives the reference-position
        augmentation).

        Notes:
            Biological classes, token grouping and representative atoms are resolved once for the prepared target.
            The token-relation bit is symmetric positive-edge presence and discards kind, role, order and direction. It
            does
            not establish covalence or known absence.

            Complete reference conformers are model features with their own availability mask, independent of source
            observations. Reference augmentation uses that mask. Generated reference coordinates do not create canonical
            observations. A conformer operation unable to represent directed dative geometry raises a capability error.
        """
        return self.processor.prepare_input(structure, msa=msa, pocket=pocket, generator=generator)

    def fold(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        pocket: Boltz1Pocket | None = None,
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> FoldResult:
        r"""
        Predict structures for one structure with Boltz 1.

        Args:
            structure: The target (read or constructed); its chemistry is consumed as is.
            msa: Optional MSA pools by instance, or an exact plan.
            pocket: Optional binder and pocket-residue specification.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-position augmentation and coordinate sampling. Mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by coordinate sampling. Mutually exclusive with ``seed``.
            num_recycles: Number of representation recycling iterations. ``None`` uses the Model configuration.
            msa_max_depth: Optional rows independently sampled for each MSA-module invocation.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            use_potentials: Enable the released Feynman--Kac and physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            num_particles: Number of particles per requested sample under Feynman--Kac steering.
            feynman_kac_lambda: Energy-log-weight scale used for particle resampling.
            feynman_kac_resampling_interval: Reverse-diffusion step interval between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided reverse-diffusion step.

        Returns:
            Generation-ordered predictions decoded onto the target (every atom regenerated in a new frame).
            Structure, PAE and PDE values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        torch_generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(
            structure, msa=msa, pocket=pocket, generator=None if seed is None else np.random.default_rng(seed)
        )
        with torch.inference_mode():
            output = self.forward(
                prepared.features,
                num_samples=num_samples,
                generator=torch_generator,
                num_recycles=num_recycles,
                msa_max_depth=msa_max_depth,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                use_potentials=use_potentials,
                feynman_kac=feynman_kac,
                physical_guidance=physical_guidance,
                num_particles=num_particles,
                feynman_kac_lambda=feynman_kac_lambda,
                feynman_kac_resampling_interval=feynman_kac_resampling_interval,
                num_guidance_steps=num_guidance_steps,
            )
        return self.postprocess(prepared, output)

    @torch.inference_mode()
    def forward(
        self,
        input: Boltz1Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> Boltz1SamplingWithConfidenceOutput:
        r"""Compute raw outputs for the prepared input."""

        return self.model.sample_with_confidence(
            input,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
        )

    def postprocess(
        self, prepared: PreparedInput[Boltz1Input], output: Boltz1SamplingWithConfidenceOutput
    ) -> FoldResult:
        r"""
        Decode one raw output without executing the model: every sample through ``decode_coordinates`` (evidence
        PREDICTED, scope every atom, new frame); confidence values on the prepared token axis.
        """

        samples = output.coordinates[0].detach().cpu().double().numpy()
        decoded = [decode_coordinates(prepared, sample)[0] for sample in samples]
        token_axis, chain_axis = _axes(prepared)
        confidence = output.confidence

        def per_sample(name: str) -> np.ndarray | None:
            value: Tensor | None = None if confidence is None else getattr(confidence, name)
            return None if value is None else floating_numpy(value[0])

        scores = {name: per_sample(name) for name in (*_HIGHER, *_LOWER)}
        plddt, pae, pde, chain_pair_iptm = (per_sample(name) for name in ("plddt", "pae", "pde", "chain_pair_iptm"))
        resolved = (
            None
            if confidence is None
            else floating_numpy(resolved_probability_from_logits(confidence.resolved_logits[0].detach().cpu()))
        )
        predictions = []
        for index, structure in enumerate(decoded):
            sample_scores = {
                name: Score(
                    float(values[index]),
                    unit="angstrom" if name in _LOWER else None,
                    preferred="lower" if name in _LOWER else "higher",
                )
                for name, values in scores.items()
                if values is not None
            }
            metrics = {
                name: Metric(values=values[index], axis=token_axis)
                for name, values in (("plddt", plddt), ("resolved_probability", resolved))
                if values is not None
            }
            pair_metrics = {
                name: PairMetric(values=values[index], row_axis=token_axis, column_axis=token_axis, unit="angstrom")
                for name, values in (("pae", pae), ("pde", pde))
                if values is not None
            }
            if chain_pair_iptm is not None:
                pair_metrics["chain_pair_iptm"] = PairMetric(
                    values=chain_pair_iptm[index], row_axis=chain_axis, column_axis=chain_axis
                )
            predictions.append(FoldPrediction(structure, sample_scores, metrics, pair_metrics))
        return FoldResult(predictions=tuple(predictions))

    __call__ = task_call(fold)


def _axes(prepared: PreparedInput[Boltz1Input]) -> tuple[tuple[ComponentRef | AtomSite, ...], tuple[str, ...]]:
    r"""The prepared token axis (a residue token is its component, an atom token its atom) and the chain axis (the
    instances with a token, in system order)."""
    system = prepared.context.structure.system
    layout = prepared.context.slots.layout
    pointer = system.instance_component_ptr
    counts = np.bincount(layout.token_component, minlength=len(system.component_symbol))
    atomic = (counts[layout.token_component] > 1) | (layout.token_class == TOKEN_INDEX["OTHER"])
    refs = system.component_refs(layout.token_component)
    names = system.text("atom", system.atom_symbol[layout.token_centre_atom]).tolist()
    axis = tuple(
        AtomSite(ref.instance, ref.component, name) if atom else ref
        for ref, name, atom in zip(refs, names, atomic.tolist())
    )
    instances = np.unique(np.searchsorted(pointer, layout.token_component, side="right") - 1)
    return axis, tuple(system.text("instance", system.instance_symbol[instances]).tolist())
