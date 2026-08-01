r"""
MPNN Packer side-chain packing Pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import cast

import torch

from ...data import residue_constants as rc
from ...data.molecular import ComponentReference
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...pipelines.input import resolve_structure_input
from ...results import Metric, PackPrediction, PackResult
from .model import MpnnPackerInput, MpnnPackerModel, MpnnPackerSamplingOutput
from .processor import MpnnPackerProcessor


class MpnnPackerPipeline(PairedPipeline[MpnnPackerModel, MpnnPackerProcessor]):
    r"""
    Side-chain packing pipeline for MPNN Packer.

    Load a published pair with [`ofoldx.pipeline`][]. Packing places side chains onto an
    observed backbone, so it takes a structure read from a file rather than a topology
    built from a sequence.
    """

    model_architecture = "mpnn_packer"

    def prepare_pack(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        pack_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> MpnnPackerInput:
        r"""Prepare one structure for the MPNN Packer sampling graph."""

        structure = cast(Structure, resolve_structure_input(input))
        return self.processor(
            structure,
            sequence=sequence,
            pack_positions=pack_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )

    def pack(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        pack_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> PackResult:
        r"""
        Pack selected side chains with the iterative sampler.

        Args:
            input: A ``Structure`` or local structure path.
            sequence: Optional replacement protein sequence.
            pack_positions: Residue occurrences to repack. ``None`` selects every eligible
                protein residue.
            use_atom_context: Whether nonprotein heavy atoms are visible to the packer.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_samples: Number of packed structures to generate.
            num_denoising_steps: Number of iterative side-chain denoising steps.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            A ``PackResult`` whose predictions follow sampling order.
        """

        model_input = self.prepare_pack(
            input,
            sequence=sequence,
            pack_positions=pack_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.sequence_tokens.device)
        output = self.sample_pack(
            model_input,
            num_samples=num_samples,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return self.decode_pack(model_input, output, num_samples=num_samples)

    def pack_sequences(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequences: Sequence[str | Mapping[str, str | Sequence[str]]],
        pack_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[PackResult, ...]:
        r"""
        Pack several target sequences against one source structure.

        Args:
            input: A source ``Structure`` or local structure path.
            sequences: Replacement protein sequences in output order.
            pack_positions: Residue occurrences to repack. ``None`` selects every eligible
                protein residue.
            use_atom_context: Whether nonprotein heavy atoms are visible to the packer.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_samples: Number of packed structures generated per sequence.
            num_denoising_steps: Number of iterative side-chain denoising steps.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed; mutually exclusive with ``generator``.
            generator: Optional PyTorch random generator; mutually exclusive with ``seed``.

        Returns:
            One ``PackResult`` per input sequence, preserving sequence order.
        """

        structure = cast(Structure, resolve_structure_input(input))
        inputs = tuple(
            self.processor(
                structure,
                sequence=sequence,
                pack_positions=pack_positions,
                use_atom_context=use_atom_context,
                filter_by_occupancy=filter_by_occupancy,
            )
            for sequence in sequences
        )
        if not inputs:
            return ()
        generator = resolve_generator(seed, generator, device=inputs[0].sequence_tokens.device)
        if len(inputs) == 1:
            output = self.sample_pack(
                inputs[0],
                num_samples=num_samples,
                num_denoising_steps=num_denoising_steps,
                num_torsion_samples=num_torsion_samples,
                generator=generator,
            )
            return (self.decode_pack(inputs[0], output, num_samples=num_samples),)

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        component_offsets = []
        component_offset = 0
        for item in inputs:
            component_offsets.append(component_offset)
            component_offset += item.structure.system.component_symbol.numel()
        batched_structure = Structure.batch([item.structure for item in inputs])
        batched = MpnnPackerInput(  # type: ignore[call-arg]
            backbone_coordinates=torch.cat([item.backbone_coordinates for item in inputs]),
            sequence_tokens=torch.cat([item.sequence_tokens for item in inputs]),
            residue_mask=torch.cat([item.residue_mask for item in inputs]),
            residue_index=torch.cat([item.residue_index for item in inputs]),
            chain_index=torch.cat([item.chain_index for item in inputs]),
            context_coordinates=torch.cat([item.context_coordinates for item in inputs]),
            context_atomic_numbers=torch.cat([item.context_atomic_numbers for item in inputs]),
            context_mask=torch.cat([item.context_mask for item in inputs]),
            atom37_coordinates=torch.cat([item.atom37_coordinates for item in inputs]),
            atom37_mask=torch.cat([item.atom37_mask for item in inputs]),
            pack_positions=torch.cat([item.pack_positions for item in inputs]),
            structure=batched_structure,
            component_index=torch.cat(
                [
                    item.component_index + offset
                    for item, offset in zip(
                        inputs,
                        component_offsets,
                        strict=True,
                    )
                ]
            ),
            batch_size=[],
        )
        realization_inputs = tuple(
            item.replace(structure=target)  # type: ignore[attr-defined]
            for item, target in zip(
                inputs,
                batched_structure.unbatch(),
                strict=True,
            )
        )

        predictions: list[list[PackPrediction]] = [[] for _ in inputs]
        for _ in range(num_samples):
            packed = self.model.sample(
                batched,
                num_denoising_steps=num_denoising_steps,
                num_torsion_samples=num_torsion_samples,
                generator=generator,
            )
            for index, item in enumerate(realization_inputs):
                predictions[index].append(_pack_candidate(item, packed[index : index + 1]))

        return tuple(PackResult(predictions=tuple(items)) for items in predictions)

    def sample_pack(
        self,
        input: MpnnPackerInput,
        *,
        num_samples: int = 1,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        generator: torch.Generator | None = None,
    ) -> tuple[MpnnPackerSamplingOutput, ...]:
        r"""Return raw side-chain samples in generation order."""

        if input.sequence_tokens.shape[0] != 1:
            raise ValueError("pack requires batch size one")
        if input.structure.system.system_instance_ptr.numel() != 2:
            raise ValueError("pack requires one molecular system")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_denoising_steps <= 0:
            raise ValueError("num_denoising_steps must be positive")
        if num_torsion_samples <= 0:
            raise ValueError("num_torsion_samples must be positive")
        if not bool(input.pack_positions.any()):
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

    def decode_pack(
        self,
        input: MpnnPackerInput,
        output: Sequence[MpnnPackerSamplingOutput],
        *,
        num_samples: int,
    ) -> PackResult:
        r"""Decode raw MPNN Packer samples without executing the Model."""

        if not bool(input.pack_positions.any()):
            return PackResult(predictions=tuple(PackPrediction(input.structure) for _ in range(num_samples)))
        if len(output) != num_samples:
            raise ValueError("one raw packing output is required per requested sample")
        return PackResult(predictions=tuple(_pack_candidate(input, sample) for sample in output))


def _pack_candidate(
    input: MpnnPackerInput,
    packed: MpnnPackerSamplingOutput,
) -> PackPrediction:
    r"""Decode one sampled atom14 state onto the canonical structure axis."""

    structure = input.structure
    system = structure.system
    realized = structure.clone()  # type: ignore[attr-defined]
    selected = input.pack_positions[0].detach().cpu().tolist()
    component_indices = input.component_index[0].detach().cpu().tolist()
    component_ids = [system.symbols[int(system.component_symbol[component])] for component in component_indices]
    atom14_coordinates = (
        packed.atom14_coordinates[0]
        .detach()
        .to(
            device=realized.coordinates.device,
            dtype=realized.coordinates.dtype,
        )
    )
    atom14_mask = packed.atom14_mask[0].detach().to(device=realized.coordinate_mask.device)
    for row, (component, component_id, is_selected) in enumerate(
        zip(component_indices, component_ids, selected, strict=True)
    ):
        if not is_selected:
            continue
        start = int(system.component_atom_ptr[component])
        stop = int(system.component_atom_ptr[component + 1])
        atom_by_name = {
            system.symbols[symbol]: atom
            for atom, symbol in enumerate(
                system.atom_symbol[start:stop].detach().cpu().tolist(),
                start=start,
            )
        }
        allowed_atoms = {atom_name for atom_name in rc.AMINO_ACID_ATOM14_NAMES[component_id] if atom_name}
        allowed_atoms.add("OXT")
        for atom_name, atom in atom_by_name.items():
            if atom_name in allowed_atoms:
                continue
            realized.coordinates[atom] = 0
            realized.coordinate_mask[atom] = False
            realized.occupancy[atom] = 0
            realized.occupancy_mask[atom] = False
            realized.b_factor[atom] = 0
            realized.b_factor_mask[atom] = False
        for atom14_index, atom_name in enumerate(rc.AMINO_ACID_ATOM14_NAMES[component_id][4:], start=4):
            if not atom_name:
                continue
            atom = atom_by_name[atom_name]
            realized.coordinates[atom] = atom14_coordinates[row, atom14_index]
            realized.coordinate_mask[atom] = atom14_mask[row, atom14_index]
            realized.occupancy[atom] = 0
            realized.occupancy_mask[atom] = False
            realized.b_factor[atom] = 0
            realized.b_factor_mask[atom] = False

    metrics: dict[str, Metric] = {}
    chi_log_densities = packed.chi_log_densities[0].detach().cpu()
    if chi_log_densities.dtype == torch.bfloat16:
        chi_log_densities = chi_log_densities.float()
    chi_log_density_values = chi_log_densities.numpy()
    references = tuple(system.component_reference(component) for component in component_indices)
    for chi_index in range(4):
        rows = tuple(
            row
            for row, (component_id, is_selected) in enumerate(zip(component_ids, selected, strict=True))
            if is_selected and len(rc.chi_angles_atoms[component_id]) > chi_index
        )
        if rows:
            metrics[f"chi{chi_index + 1}_log_density"] = Metric(
                values=chi_log_density_values[rows, chi_index],
                axis=tuple(references[row] for row in rows),
                unit="nat/radian",
            )
    return PackPrediction(structure=realized, metrics=metrics)
