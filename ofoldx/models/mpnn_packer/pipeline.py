r"""MPNN Packer side-chain packing Pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import torch

from ...data.molecular import ComponentReference
from ...data.structure import Structure
from ...io import read_mpnn_pdb
from ...pipelines import PairedPipeline, resolve_generator
from ...results import PackPrediction, PackResult
from .model import MpnnPackerInput, MpnnPackerModel, MpnnPackerSamplingOutput, pack_candidate
from .processor import MpnnPackerProcessor


class MpnnPackerPipeline(PairedPipeline[MpnnPackerModel, MpnnPackerProcessor]):
    r"""Bind the MPNN Packer Model and Processor."""

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
        r"""Pack selected side chains with the iterative sampler.

        Args:
            input: Canonical structure or local structure path.
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

        model_input = self.processor(
            input,
            sequence=sequence,
            pack_positions=pack_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.sequence_tokens.device)
        return self.model.pack(
            model_input,
            num_samples=num_samples,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )

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
        r"""Pack several target sequences against one source structure.

        Args:
            input: Canonical source structure or local structure path.
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

        structure = read_mpnn_pdb(input) if isinstance(input, (str, PathLike)) else input
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
        generator = resolve_generator(seed, generator, device=inputs[0].sequence_tokens.device)
        if len(inputs) == 1:
            return (
                self.model.pack(
                    inputs[0],
                    num_samples=num_samples,
                    num_denoising_steps=num_denoising_steps,
                    num_torsion_samples=num_torsion_samples,
                    generator=generator,
                ),
            )

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        # Model.sample consumes only the tensor fields. Each singleton input remains
        # beside the tensor batch to retain its own canonical realization mapping.
        batched = MpnnPackerInput(
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
            structure=inputs[0].structure,
            component_references=inputs[0].component_references,
        )

        predictions: list[list[PackPrediction]] = [[] for _ in inputs]
        for _ in range(num_samples):
            packed = self.model.sample(
                batched,
                num_denoising_steps=num_denoising_steps,
                num_torsion_samples=num_torsion_samples,
                generator=generator,
            )
            for index, item in enumerate(inputs):
                candidate = MpnnPackerSamplingOutput(
                    atom14_coordinates=packed.atom14_coordinates[index : index + 1],
                    atom14_mask=packed.atom14_mask[index : index + 1],
                    sampled_chi_angles=packed.sampled_chi_angles[index : index + 1],
                    chi_log_densities=packed.chi_log_densities[index : index + 1],
                    mean=packed.mean[index : index + 1],
                    concentration=packed.concentration[index : index + 1],
                    mix_logits=packed.mix_logits[index : index + 1],
                    input_chi_sin_cos=packed.input_chi_sin_cos[index : index + 1],
                )
                predictions[index].append(pack_candidate(item, candidate))

        return tuple(PackResult(predictions=tuple(items)) for items in predictions)
