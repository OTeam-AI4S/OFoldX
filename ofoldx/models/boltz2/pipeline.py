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
Boltz-2 folding and two-stage affinity-scoring Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import torch
from torch import Tensor

from ...data.molecular import AtomSite, ComponentRef
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX
from ...msa import MSA, MsaCollection
from ...pipelines import ComponentSpec, CompositePipeline, PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import FoldPrediction, FoldResult, Metric, PairMetric, Score, ScoreResult, floating_numpy
from ...templates import Template
from ..preparation import PreparedInput, decode_coordinates
from .model import (
    Boltz2AffinityModel,
    Boltz2AffinityScoreOutput,
    Boltz2ConfidenceOutput,
    Boltz2Contact,
    Boltz2Input,
    Boltz2Model,
    Boltz2Pocket,
    Boltz2Template,
)
from .processor import Boltz2AffinityProcessor, Boltz2Processor

_HIGHER = ("mean_plddt", "ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_iplddt")
_LOWER = ("complex_pde", "complex_ipde")
_DESIGN_HIGHER = (
    "design_to_target_iptm",
    "design_residue_iptm",
    "design_iptm",
    "design_iiptm",
    "target_ptm",
    "design_ptm",
    "design_ipsae_min",
    "design_to_target_ipsae",
    "target_to_design_ipsae",
)
_DESIGN_LOWER = ("interaction_pae", "min_interaction_pae", "min_design_to_target_pae")
_AFFINITY_VALUES = ("affinity_pred_value", "affinity_pred_value1", "affinity_pred_value2")
_AFFINITY_PROBABILITIES = (
    "affinity_probability_binary",
    "affinity_probability_binary1",
    "affinity_probability_binary2",
)


class Boltz2Pipeline(PairedPipeline[Boltz2Model, Boltz2Processor]):
    r"""
    Structure prediction pipeline for Boltz 2.
    """

    task = "fold"

    model_architecture = "boltz2"

    def prepare(
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
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[Boltz2Input]:
        r"""
        Prepare one structure and its evidence for Boltz-2 folding (``generator`` drives the reference-position
        augmentation).

        Notes:
            Token groups and representative atoms are resolved from biological classes. Complete reference conformers
            are augmented according to their own availability, independent of source observations; generated reference
            geometry
            does not establish a canonical observation.

            Relation category 0 means no represented positive edge; 1 means a mixed or unknown category; 2, 3 and 4 mean
            unanimous known scalar orders 1, 2 and 3; 5 means unanimous known aromaticity; and 6 means any assembly-role
            edge.
            Per-pair aggregation gives 6 precedence, then a unanimous category, then 1. The categories do not establish
            physical covalence, chemical equivalence or known absence. The ``modified`` bit denotes nominal model-label
            table
            membership. Full typed and directed source relations remain unchanged.
        """
        return self.processor.prepare_input(
            structure,
            msa=msa,
            templates=templates,
            designed_positions=designed_positions,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

    def fold(
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
        num_samples: int = 1,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> FoldResult:
        r"""
        Predict structures for one structure with Boltz 2.

        Args:
            structure: The target (read or constructed); its chemistry is consumed as is.
            msa: Optional MSA pools by instance, or an exact plan.
            templates: Protein templates, one model template row each.
            designed_positions: Components designed by the calling protocol (adds the design scores).
            self_template: Components whose own coordinates form one more template row.
            contacts: Pairwise contact conditions.
            pockets: Binder-to-pocket conditions.
            method: Experimental method of the learned method embedding.
            num_samples: Number of structure predictions to generate.
            seed: Optional seed for reference-position augmentation and coordinate sampling. Mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by coordinate sampling. Mutually exclusive with ``seed``.
            num_recycles: Number of representation recycling iterations. ``None`` uses the Model configuration.
            msa_max_depth: Optional rows independently sampled for each MSA-module invocation.
            num_sampling_steps: Number of diffusion steps. ``None`` uses the Model configuration.
            step_scale: Optional scale on each reverse-diffusion Euler update.
            max_parallel_samples: Maximum samples evaluated by one denoiser call.
            use_potentials: Enable the released Feynman--Kac and physical-guidance preset.
            feynman_kac: Whether to use particle resampling.
            physical_guidance: Whether to apply physical-potential gradients.
            contact_guidance: Whether forced contact, pocket and template conditions apply their potential gradients.
            num_particles: Number of particles per requested sample under Feynman--Kac steering.
            feynman_kac_lambda: Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: Reverse-diffusion step interval between particle resampling operations.
            num_guidance_steps: Gradient updates applied at each guided reverse-diffusion step.

        Returns:
            Generation-ordered predictions decoded onto the target (every atom regenerated in a new frame).
            Structure, PAE and PDE values are in angstroms; pLDDT is on the ``0--100`` scale.
        """

        torch_generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(
            structure,
            msa=msa,
            templates=templates,
            designed_positions=designed_positions,
            self_template=self_template,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=None if seed is None else np.random.default_rng(seed),
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
                max_parallel_samples=max_parallel_samples,
                use_potentials=use_potentials,
                feynman_kac=feynman_kac,
                physical_guidance=physical_guidance,
                contact_guidance=contact_guidance,
                num_particles=num_particles,
                feynman_kac_lambda=feynman_kac_lambda,
                feynman_kac_resampling_interval=feynman_kac_resampling_interval,
                num_guidance_steps=num_guidance_steps,
            )
        return self.postprocess(prepared, output)

    @torch.inference_mode()
    def forward(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
    ) -> Boltz2ConfidenceOutput:
        r"""Compute raw outputs for the prepared input."""

        return self.model.sample_with_confidence(
            input,
            num_samples=num_samples,
            generator=generator,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
        )

    def postprocess(self, prepared: PreparedInput[Boltz2Input], output: Boltz2ConfidenceOutput) -> FoldResult:
        r"""
        Decode one raw output without executing the model: every sample through ``decode_coordinates`` (evidence
        PREDICTED, scope every atom, new frame); confidence values on the prepared token axis and the chain axis of
        the instances with a token. Design scores are reported when the input designs a position.
        """

        samples = output.coordinates[0].detach().cpu().double().numpy()
        token_axis, chain_axis, chains = _axes(prepared)

        def values(name: str) -> np.ndarray:
            return floating_numpy(cast(Tensor, getattr(output, name))[0])

        plddt, resolved, pae, pde = (values(name) for name in ("plddt", "resolved_probability", "pae", "pde"))
        chain_pair_iptm = values("chain_pair_iptm")[:, chains][:, :, chains]
        ptm, iptm = values("ptm"), values("iptm")
        confidence_score = (4.0 * values("mean_plddt") / 100.0 + np.where(np.isclose(iptm, 0.0), ptm, iptm)) / 5.0
        higher = (*_HIGHER, *(_DESIGN_HIGHER if bool(prepared.features.design_mask.any()) else ()))
        lower = (*_LOWER, *(_DESIGN_LOWER if bool(prepared.features.design_mask.any()) else ()))
        scores = {name: values(name) for name in (*higher, *lower)}
        predictions = []
        for index, sample in enumerate(samples):
            structure, _ = decode_coordinates(prepared, sample)
            sample_scores = {
                name: Score(
                    float(value[index]),
                    unit="angstrom" if name in lower else None,
                    preferred="lower" if name in lower else "higher",
                )
                for name, value in scores.items()
            }
            sample_scores["confidence_score"] = Score(float(confidence_score[index]), preferred="higher")
            predictions.append(
                FoldPrediction(
                    structure,
                    sample_scores,
                    {
                        "plddt": Metric(values=plddt[index], axis=token_axis),
                        "resolved_probability": Metric(values=resolved[index], axis=token_axis),
                    },
                    {
                        "pae": PairMetric(pae[index], token_axis, token_axis, unit="angstrom"),
                        "pde": PairMetric(pde[index], token_axis, token_axis, unit="angstrom"),
                        "chain_pair_iptm": PairMetric(chain_pair_iptm[index], chain_axis, chain_axis),
                    },
                )
            )
        return FoldResult(predictions=tuple(predictions))

    __call__ = task_call(fold)


class Boltz2AffinityScorePipeline(PairedPipeline[Boltz2AffinityModel, Boltz2AffinityProcessor]):
    r"""
    Binding-affinity scoring pipeline for one folded protein-ligand complex.
    """

    task = "score"

    model_architecture = "boltz2_affinity"

    def prepare(
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
        r"""Prepare the ligand-centred crop of one folded protein-ligand complex for affinity scoring."""
        return self.processor.prepare_input(
            structure,
            ligand=ligand,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )

    def score(
        self,
        structure: Structure,
        *,
        ligand: str | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "other",
        num_samples: int = 5,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ScoreResult:
        r"""
        Score affinity for one already folded protein-ligand complex.

        Args:
            structure: The folded complex.
            ligand: Instance of the ligand to score; may be omitted when the structure holds exactly one ligand.
            msa: Optional MSA pools by instance, or an exact plan.
            contacts: Pairwise contact conditions.
            pockets: Binder-to-pocket conditions.
            method: Experimental method of the learned method embedding.
            num_samples: Number of affinity-pass coordinate samples.
            num_recycles: Number of affinity trunk recycling iterations.
            msa_max_depth: Optional rows independently sampled for every MSA-module invocation.
            num_sampling_steps: Number of affinity-pass diffusion transitions.
            step_scale: Optional reverse-diffusion Euler-step scale.
            molecular_weight_correction: Whether to apply molecular-weight calibration (needs the ligand's weight,
                known only for a determined ligand graph).
            seed: Optional seed for reference-position augmentation and scoring; mutually exclusive with
                ``generator``.
            generator: Optional generator advanced by scoring; mutually exclusive with ``seed``.

        Returns:
            The affinity pass's selected structure, decoded onto the complex, and its affinity scores.
        """

        torch_generator = resolve_generator(seed, generator, device=self.processor.device)
        prepared = self.prepare(
            structure,
            ligand=ligand,
            msa=msa,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=None if seed is None else np.random.default_rng(seed),
        )
        output = self.forward(
            prepared.features,
            num_samples=num_samples,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=torch_generator,
        )
        return self.postprocess(prepared, output)

    def forward(
        self,
        input: Boltz2Input,
        *,
        num_samples: int = 5,
        num_recycles: int | None = None,
        msa_max_depth: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityScoreOutput:
        r"""Compute raw outputs for the prepared input."""

        return self.model.score(
            input,
            num_samples=num_samples,
            num_recycles=num_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )

    def postprocess(self, prepared: PreparedInput[Boltz2Input], output: Boltz2AffinityScoreOutput) -> ScoreResult:
        r"""Decode the selected affinity-pass sample through ``decode_coordinates`` and the affinity scores, scoped to
        the ligand instance."""

        selected = int(output.selected_sample_index[0])
        coordinates = output.structure.coordinates[0, selected].detach().cpu().double().numpy()
        structure, _ = decode_coordinates(prepared, coordinates)
        affinity = output.affinity
        tokens = np.flatnonzero(cast(Tensor, prepared.features.affinity_token_mask)[0].cpu().numpy())
        scope = frozenset(
            {
                prepared.context.structure.system.component_refs(
                    prepared.context.slots.layout.token_component[tokens[:1]]
                )[0].instance
            }
        )

        def value(name: str) -> float:
            return float(getattr(affinity, name)[0, 0, 0])

        scores = {
            **{
                name: Score(value(name), unit="log10(IC50/µM)", preferred="lower", scope=scope)
                for name in _AFFINITY_VALUES
            },
            **{name: Score(value(name), preferred="higher", scope=scope) for name in _AFFINITY_PROBABILITIES},
        }
        return ScoreResult(target=structure, scores=scores)

    __call__ = task_call(score)


@dataclass(eq=False)
class Boltz2AffinityPipelineInput:
    r"""
    The prepared folding pass and the evidence the affinity pass reuses.

    Attributes:
        fold_input: The prepared folding task.
        msa: Alignment evidence reused for the selected structure.
        ligand: Ligand instance scored.
        contacts: Pairwise contact conditions of both passes.
        pockets: Pocket conditions of both passes.
        generator: The numpy generator of both passes' reference-position augmentation.
    """

    fold_input: PreparedInput[Boltz2Input]
    msa: Mapping[str, MSA] | MsaCollection | None
    ligand: str | None
    contacts: Sequence[Boltz2Contact]
    pockets: Sequence[Boltz2Pocket]
    generator: np.random.Generator | None


@dataclass(eq=False)
class Boltz2AffinityPipelineOutput:
    r"""
    Ranked folding predictions and affinity scores for the selected structure.

    Attributes:
        structure_predictions: Predictions in stable descending confidence order.
        affinity: Scores and selected target from the affinity pass.
    """

    structure_predictions: tuple[FoldPrediction, ...]
    affinity: ScoreResult


class Boltz2AffinityPipeline(CompositePipeline):
    r"""
    Two-pass structure prediction and binding-affinity pipeline for Boltz 2.

    Args:
        affinity_scorer: Independently deployable affinity-scoring stage.
        folder: Independently deployable structure-prediction stage.

    Notes:
        Preparation retains the folding features, evidence, and NumPy stream reused to augment reference positions
        for the affinity pass. Execution folds the complex, selects its highest-confidence structure, and scores
        that structure with the affinity child. The sampling generator advances through both passes.

        The folder and affinity scorer are saved in ``structure/`` and ``affinity/``, respectively. Each child retains
        its concrete pipeline identity when the composition is saved and loaded.
    """

    task = "score"

    component_layout = (
        ComponentSpec("affinity_scorer", "affinity", Boltz2AffinityScorePipeline),
        ComponentSpec("folder", "structure", Boltz2Pipeline),
    )

    def __init__(self, *, affinity_scorer: Boltz2AffinityScorePipeline, folder: Boltz2Pipeline) -> None:
        super().__init__({"affinity_scorer": affinity_scorer, "folder": folder})

    @property
    def affinity_scorer(self) -> Boltz2AffinityScorePipeline:
        return cast(Boltz2AffinityScorePipeline, self.components["affinity_scorer"])

    @property
    def folder(self) -> Boltz2Pipeline:
        return cast(Boltz2Pipeline, self.components["folder"])

    def score(
        self,
        structure: Structure,
        *,
        ligand: str | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        num_samples: int = 1,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        num_affinity_samples: int = 5,
        num_affinity_recycles: int | None = None,
        num_affinity_sampling_steps: int | None = None,
        affinity_step_scale: float | None = None,
        msa_max_depth: int | None = None,
        molecular_weight_correction: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityResult:
        r"""
        Predict candidate structures and score protein-ligand affinity on the most confident one.

        Args:
            structure: The complex to fold and score.
            ligand: Instance of the ligand to score; may be omitted when the structure holds exactly one ligand.
            msa: Optional MSA pools by instance, or an exact plan, used by both passes.
            templates: Protein templates of the structure pass.
            contacts: Pairwise contact conditions of both passes.
            pockets: Binder-to-pocket conditions of both passes.
            method: Experimental method of the structure pass.
            num_samples: Number of structure-pass samples.
            num_recycles: Number of structure-pass recycling iterations.
            num_sampling_steps: Number of structure-pass diffusion transitions.
            step_scale: Optional structure-pass Euler-step scale.
            max_parallel_samples: Maximum structure-pass samples evaluated by one denoiser call.
            use_potentials: Enable the released structure-pass Feynman--Kac and physical-guidance preset.
            feynman_kac: Whether the structure pass uses particle resampling.
            physical_guidance: Whether the structure pass applies physical-potential gradients.
            contact_guidance: Whether forced structure-pass conditions apply their potential gradients.
            num_particles: Number of structure-pass particles per requested sample.
            feynman_kac_lambda: Structure-pass Feynman--Kac energy-log-weight scale.
            feynman_kac_resampling_interval: Structure-pass step interval between particle resampling operations.
            num_guidance_steps: Gradient updates at each guided structure-pass step.
            num_affinity_samples: Number of affinity-pass samples.
            num_affinity_recycles: Number of affinity-pass recycling iterations.
            num_affinity_sampling_steps: Number of affinity-pass diffusion transitions.
            affinity_step_scale: Optional affinity-pass Euler-step scale.
            msa_max_depth: Optional rows independently sampled for every MSA-module invocation of both passes.
            molecular_weight_correction: Whether to apply molecular-weight calibration.
            seed: Optional seed shared by preparation and both passes; mutually exclusive with ``generator``.
            generator: Optional generator advanced by both passes; mutually exclusive with ``seed``.

        Returns:
            Structure-pass predictions sorted by descending confidence, the affinity pass's selected structure, and
            affinity scores (``log10(IC50 / micromolar)``; binder probabilities in ``[0, 1]``).
        """

        torch_generator = resolve_generator(seed, generator, device=self.folder.processor.device)
        prepared = self.prepare(
            structure,
            ligand=ligand,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=None if seed is None else np.random.default_rng(seed),
        )
        output = self.forward(
            prepared,
            num_samples=num_samples,
            num_recycles=num_recycles,
            num_sampling_steps=num_sampling_steps,
            step_scale=step_scale,
            max_parallel_samples=max_parallel_samples,
            use_potentials=use_potentials,
            feynman_kac=feynman_kac,
            physical_guidance=physical_guidance,
            contact_guidance=contact_guidance,
            num_particles=num_particles,
            feynman_kac_lambda=feynman_kac_lambda,
            feynman_kac_resampling_interval=feynman_kac_resampling_interval,
            num_guidance_steps=num_guidance_steps,
            num_affinity_samples=num_affinity_samples,
            num_affinity_recycles=num_affinity_recycles,
            num_affinity_sampling_steps=num_affinity_sampling_steps,
            affinity_step_scale=affinity_step_scale,
            msa_max_depth=msa_max_depth,
            molecular_weight_correction=molecular_weight_correction,
            generator=torch_generator,
        )
        return self.postprocess(prepared, output)

    __call__ = task_call(score)

    def prepare(
        self,
        structure: Structure,
        *,
        ligand: str | None = None,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template | Boltz2Template] = (),
        contacts: Sequence[Boltz2Contact] = (),
        pockets: Sequence[Boltz2Pocket] = (),
        method: str = "x-ray diffraction",
        generator: np.random.Generator | None = None,
    ) -> Boltz2AffinityPipelineInput:
        r"""Prepare the folding pass and keep the evidence the affinity pass reuses."""

        fold_input = self.folder.prepare(
            structure,
            msa=msa,
            templates=templates,
            contacts=contacts,
            pockets=pockets,
            method=method,
            generator=generator,
        )
        return Boltz2AffinityPipelineInput(fold_input, msa, ligand, contacts, pockets, generator)

    def forward(
        self,
        input: Boltz2AffinityPipelineInput,
        *,
        num_samples: int = 1,
        num_recycles: int | None = None,
        num_sampling_steps: int | None = None,
        step_scale: float | None = None,
        max_parallel_samples: int | None = None,
        use_potentials: bool = False,
        feynman_kac: bool = False,
        physical_guidance: bool = False,
        contact_guidance: bool = True,
        num_particles: int = 3,
        feynman_kac_lambda: float = 4.0,
        feynman_kac_resampling_interval: int = 3,
        num_guidance_steps: int = 20,
        num_affinity_samples: int = 5,
        num_affinity_recycles: int | None = None,
        num_affinity_sampling_steps: int | None = None,
        affinity_step_scale: float | None = None,
        msa_max_depth: int | None = None,
        molecular_weight_correction: bool = False,
        generator: torch.Generator | None = None,
    ) -> Boltz2AffinityPipelineOutput:
        r"""Predict structures, select the most confident, and score the selected complex."""

        with torch.inference_mode():
            fold_output = self.folder.forward(
                input.fold_input.features,
                num_samples=num_samples,
                num_recycles=num_recycles,
                num_sampling_steps=num_sampling_steps,
                step_scale=step_scale,
                msa_max_depth=msa_max_depth,
                max_parallel_samples=max_parallel_samples,
                use_potentials=use_potentials,
                feynman_kac=feynman_kac,
                physical_guidance=physical_guidance,
                contact_guidance=contact_guidance,
                num_particles=num_particles,
                feynman_kac_lambda=feynman_kac_lambda,
                feynman_kac_resampling_interval=feynman_kac_resampling_interval,
                num_guidance_steps=num_guidance_steps,
                generator=generator,
            )
        fold_result = self.folder.postprocess(input.fold_input, fold_output)
        structure_predictions = tuple(
            sorted(fold_result.predictions, key=lambda p: p.scores["confidence_score"].value, reverse=True)
        )
        prepared = self.affinity_scorer.prepare(
            structure_predictions[0].structure,
            ligand=input.ligand,
            msa=input.msa,
            contacts=input.contacts,
            pockets=input.pockets,
            generator=input.generator,
        )
        affinity_output = self.affinity_scorer.forward(
            prepared.features,
            num_samples=num_affinity_samples,
            num_recycles=num_affinity_recycles,
            msa_max_depth=msa_max_depth,
            num_sampling_steps=num_affinity_sampling_steps,
            step_scale=affinity_step_scale,
            molecular_weight_correction=molecular_weight_correction,
            generator=generator,
        )
        affinity = self.affinity_scorer.postprocess(prepared, affinity_output)
        return Boltz2AffinityPipelineOutput(structure_predictions=structure_predictions, affinity=affinity)

    def postprocess(
        self, input: Boltz2AffinityPipelineInput, output: Boltz2AffinityPipelineOutput
    ) -> Boltz2AffinityResult:
        r"""Assemble the ranked structures and the scoped affinity scores."""

        return Boltz2AffinityResult(
            target=output.affinity.target,
            structure_predictions=output.structure_predictions,
            scores=output.affinity.scores,
            metrics=output.affinity.metrics,
        )


@dataclass(eq=False)
class Boltz2AffinityResult:
    r"""
    One two-stage Boltz-2 affinity result.

    Attributes:
        target: Structure selected for affinity scoring.
        structure_predictions: First-stage predictions in descending confidence order.
        scores: Scalar affinity scores keyed by stable scientific names.
        metrics: Affinity metrics keyed by stable scientific names.
    """

    target: Structure
    structure_predictions: tuple[FoldPrediction, ...]
    scores: dict[str, Score] = field(default_factory=dict)
    metrics: dict[str, Metric] = field(default_factory=dict)


def _axes(
    prepared: PreparedInput[Boltz2Input],
) -> tuple[tuple[ComponentRef | AtomSite, ...], tuple[str, ...], np.ndarray]:
    r"""The prepared token axis (a residue token is its component, an atom token its atom), the chain axis (the
    instances with a token, in system order) and those instances' indices."""
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
    return axis, tuple(system.text("instance", system.instance_symbol[instances]).tolist()), instances
