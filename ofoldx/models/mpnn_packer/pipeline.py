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
MPNN Packer side-chain packing Pipeline.

Each sample is decoded by the one decode (``decode_coordinates``) over the prepared scope — the side chains of the
packed residues — with GENERATED evidence; side-chain atoms the packer has no slot for (hydrogens) become unobserved
and are counted. Chi log densities are metrics of the result, never an observation field.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...data.molecular import AtomSite, ComponentRef, MolecularSystem
from ...data.structure import CoordinateEvidence, Structure
from ...errors import RequestRef, fail
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.base import task_call
from ...results import Metric, PackPrediction, PackResult
from ..design import PositionSelection
from ..preparation import PreparedInput, decode_coordinates
from ..processing import project_model_features
from .model import MpnnPackerInput, MpnnPackerModel, MpnnPackerSamplingOutput
from .processor import MpnnPackerProcessor


class MpnnPackerPipeline(PairedPipeline[MpnnPackerModel, MpnnPackerProcessor]):
    r"""
    Side-chain packing pipeline for MPNN Packer.

    Packing places side chains onto an observed backbone, so it takes a read structure or a design's final target.
    """

    task = "pack"

    model_architecture = "mpnn_packer"

    def prepare(
        self,
        structure: Structure,
        *,
        positions: PositionSelection = PositionSelection(),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[MpnnPackerInput]:
        r"""
        Prepare one structure for the MPNN Packer sampling graph.

        Notes:
            Compatibility with a standard geometry template is checked from the actual heavy-atom sites, elements,
            adjacency, and known boundary/stereochemical facts. Biological labels select model features but do not
            establish chemical compatibility. Unsupported modifications raise an unsupported-capability error;
            missing prerequisites raise insufficient-information errors. Packing preserves the input chemical graph.
        """
        return self.processor.prepare_input(
            structure, positions=positions, use_atom_context=use_atom_context, filter_by_occupancy=filter_by_occupancy
        )

    def pack(
        self,
        structure: Structure,
        *,
        positions: PositionSelection = PositionSelection(),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PackResult:
        r"""
        Pack the side chains of ``positions`` with the iterative sampler.

        Args:
            structure: A read structure or a design's final target.
            positions: Residues to pack (default every protein residue the atom14 tables represent); a residue they
                cannot represent (MSE) refuses with ``model_capacity`` when packed and enters as backbone context
                otherwise.
            use_atom_context: Whether non-protein heavy atoms are visible to the packer.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            num_samples: Number of packed structures to generate.
            num_denoising_steps: Number of iterative side-chain denoising steps.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            A ``PackResult`` whose predictions follow sampling order.
        """

        prepared = self.prepare(
            structure, positions=positions, use_atom_context=use_atom_context, filter_by_occupancy=filter_by_occupancy
        )
        output = self.forward(
            prepared.features,
            num_samples=num_samples,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            seed=seed,
            generator=generator,
        )
        return self.postprocess(prepared, output, num_samples=num_samples)

    __call__ = task_call(pack)

    def forward(
        self,
        input: MpnnPackerInput,
        *,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MpnnPackerSamplingOutput, ...]:
        r"""Return raw side-chain samples in sample order (none when no residue is packed)."""
        generator = resolve_generator(seed, generator, device=input.sequence_tokens.device)
        if num_samples <= 0:
            fail("value_invalid", RequestRef("num_samples", str(num_samples)))
        if not bool(input.pack_mask.any()):
            return ()
        return tuple(
            self.model.sample(
                input,
                num_denoising_steps=num_denoising_steps,
                num_torsion_samples=num_torsion_samples,
                generator=generator,
            )
            for _ in range(num_samples)
        )

    def postprocess(
        self,
        prepared: PreparedInput[MpnnPackerInput],
        output: Sequence[MpnnPackerSamplingOutput],
        *,
        num_samples: int,
    ) -> PackResult:
        r"""
        Decode raw samples without executing the Model; with nothing packed every prediction is the target.

        Notes:
            The prepared atom/torsion correspondence places each raw draw on the source target in sampling order.
            Backbone coordinates retain their source precision. Regenerated side chains lose old coordinate
            observations, and unpredicted atoms such as hydrogens become unavailable within that scope. Chi log
            densities are result metrics with explicit residue/atom references, independently of observation metadata.
        """
        if not bool(prepared.features.pack_mask.any()):
            return PackResult(predictions=tuple(PackPrediction(prepared.context.structure) for _ in range(num_samples)))
        return PackResult(predictions=tuple(_pack_candidate(prepared, sample) for sample in output))


def _pack_candidate(prepared: PreparedInput[MpnnPackerInput], packed: MpnnPackerSamplingOutput) -> PackPrediction:
    r"""One sampled atom14 state decoded onto the target, with its chi log densities as metrics."""
    input, slots = prepared.features, prepared.context.slots
    system = prepared.context.structure.system
    tokens = slots.token[slots.token >= 0]
    starts = np.searchsorted(tokens, np.arange(len(slots.layout.token_component)))
    pack = input.pack_mask[0].cpu().numpy()
    mapping = input.atom14_atom_index[0].cpu().numpy()
    available = packed.atom14_mask[0].cpu().numpy() & pack[:, None] & (mapping >= 0) & (np.arange(14) >= 4)
    rows = np.flatnonzero(pack)
    where = starts[rows][:, None] + np.arange(14)
    coordinates = np.zeros((len(slots.atom), 3), dtype=np.float64)
    valid = np.zeros(len(slots.atom), dtype=bool)
    coordinates[where] = packed.atom14_coordinates[0].detach().cpu().double().numpy()[rows]
    valid[where] = available[rows]
    structure, _ = decode_coordinates(prepared, coordinates, valid=valid, evidence=CoordinateEvidence.GENERATED)

    chi_mask = input.chi_mask[0] & input.pack_mask[0][:, None]
    densities = (
        project_model_features(packed.chi_log_densities[0], chi_mask, feature="packed chi log densities")
        .detach()
        .double()
        .cpu()
        .numpy()
    )
    chi_mask = chi_mask.cpu().numpy()
    references = system.component_refs(input.component_index[0].cpu().numpy())
    metrics: dict[str, Metric] = {}
    for chi in range(4):
        chosen = np.flatnonzero(chi_mask[:, chi])
        if len(chosen):
            metrics[f"chi{chi + 1}_log_density"] = Metric(
                values=densities[chosen, chi], axis=tuple(references[row] for row in chosen), unit="nat/radian"
            )
    scored = available & (input.atom14_chi_index[0].cpu().numpy() >= 0)
    row, slot = np.nonzero(scored)
    if len(row):
        chis = input.atom14_chi_index[0].cpu().numpy()[row, slot]
        metrics["chi_log_density"] = Metric(
            values=densities[row, chis],
            axis=_atom_sites(system, mapping[row, slot], references, row),
            unit="nat/radian",
        )
    return PackPrediction(structure=structure, metrics=metrics)


def _atom_sites(
    system: MolecularSystem, atoms: np.ndarray, references: tuple[ComponentRef, ...], rows: np.ndarray
) -> tuple[AtomSite, ...]:
    r"""The atom site of each scored atom (its row's component and its name)."""
    names = system.text("atom", system.atom_symbol[atoms]).tolist()
    return tuple(AtomSite(references[r].instance, references[r].component, n) for r, n in zip(rows.tolist(), names))
