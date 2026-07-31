r"""
LigandMPNN sequence design and probability Pipelines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from typing_extensions import Self

from ...data import ENTITY_PROTEIN
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...io import read_structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import DesignResult, MpnnProbabilityResult, PackResult
from .model import LigandMpnnModel
from .processor import LigandMpnnProcessor

if TYPE_CHECKING:
    from ..mpnn_packer import MpnnPackerPipeline


class LigandMpnnPipeline(PairedPipeline[LigandMpnnModel, LigandMpnnProcessor]):
    r"""
    Protein sequence design pipeline for LigandMPNN.
    """

    model_architecture = "ligandmpnn"

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""
        Design LigandMPNN sequences with ligand and residue context.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with complete backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            global_logit_bias: Global bias vector over the 20 generatable identities.
            residue_logit_bias: Per-residue bias vectors over the 20 generatable identities.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            ligand_interface_cutoff: Distance cutoff in angstroms used to mark interface residues.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_sidechain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            A ``DesignResult`` containing the sampled sequence candidates.

        Notes:
            Candidates follow sampling order. ``seed`` and ``generator`` are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            design_positions=design_positions,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute sequence-conditioned LigandMPNN autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_sidechain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with sequence-conditioned logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            design_positions=design_positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute sequence-free LigandMPNN autoregressive probabilities.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences treated as designable. ``None`` selects every
                valid protein occurrence.
            conditions: Tied-identity conditions applied to decoding order.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_sidechain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with sequence-free logits and decoding ranks.

        Notes:
            The leading tensor axis enumerates decoding orders, and the residue axis follows
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            design_positions=design_positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.sequence_free_autoregressive_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute target-last conditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_sidechain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with each target decoded after all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            design_positions=design_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.conditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""
        Compute target-first unconditional probabilities for every residue row.

        Args:
            input: A ``Structure`` or local structure path.
            design_positions: Residue occurrences marked as designable in the returned carrier.
                ``None`` selects every valid protein occurrence.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_decoding_orders: Number of independent base decoding orders to evaluate.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates.
            use_sidechain_context: Whether fixed protein side-chain atoms are used as context.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator.

        Returns:
            An ``MpnnProbabilityResult`` with each target decoded before all other residues.

        Notes:
            ``logits`` use decoding-order, target-residue, and vocabulary axes;
            ``decoding_ranks`` add a final residue-order axis. Residue axes follow
            the processed canonical component indices. ``seed`` and ``generator``
            are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.processor(
            structure,
            design_positions=design_positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=model_input.backbone_coordinates.device)
        return self.model.unconditional_probabilities(
            model_input,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )


class LigandMpnnDesignAndPackPipeline:
    r"""
    Design protein sequences with LigandMPNN and pack their side chains.

    Args:
        designer: LigandMPNN sequence-design pipeline.
        packer: Side-chain packing pipeline.
    """

    def __init__(
        self,
        *,
        designer: LigandMpnnPipeline,
        packer: MpnnPackerPipeline,
    ) -> None:
        self.designer = designer
        self.packer = packer

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""
        Move the designer and packer to a device and dtype.

        Args:
            device: Destination device. ``None`` preserves the current device.
            dtype: Destination floating-point dtype. ``None`` preserves the current dtype.

        Returns:
            This pipeline.
        """

        self.designer.to(device=device, dtype=dtype)
        self.packer.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_sidechain_context: bool = False,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DesignResult, tuple[PackResult, ...]]:
        r"""
        Design LigandMPNN sequences and pack every resulting candidate.

        Args:
            input: A ``Structure`` or local structure path, parsed once for both stages.
            design_positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with complete backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            global_logit_bias: Global bias vector over the 20 generatable identities.
            residue_logit_bias: Per-residue bias vectors over the 20 generatable identities.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible during design.
            ligand_interface_cutoff: Distance cutoff in angstroms used to mark interface residues.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            num_candidates: Number of complete sequence candidates to sample.
            temperature: Positive autoregressive sampling temperature.
            coordinate_noise: Standard deviation of noise applied to backbone and atom-context
                coordinates during design.
            use_sidechain_context: Whether fixed protein side-chain atoms are used during design.
            use_packing_atom_context: Whether the packing stage uses nonprotein atom context.
            repack_everything: Whether to pack every eligible residue instead of only designed
                positions.
            num_packs_per_candidate: Number of packed structures generated per sequence candidate.
            num_denoising_steps: Number of side-chain denoising steps per packed structure.
            num_torsion_samples: Number of torsion proposals evaluated per denoising step.
            seed: Optional integer seed.
            generator: Optional PyTorch random generator advanced through design before packing.

        Returns:
            A pair containing the ``DesignResult`` and one ``PackResult`` per sequence candidate.

        Notes:
            The ``PackResult`` tuple follows candidate order, and each result's predictions follow
            packing-sample order. ``seed`` and ``generator`` are mutually exclusive.
        """

        structure = read_structure(input) if isinstance(input, (str, PathLike)) else input
        model_input = self.designer.processor(
            structure,
            design_positions=design_positions,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        device = model_input.residue_mask.device
        generator = resolve_generator(seed, generator, device=device)
        design_result = self.designer.model.design(
            model_input,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_sidechain_context=use_sidechain_context,
            generator=generator,
        )
        if repack_everything:
            pack_positions: Sequence[ComponentReference] | None = None
        else:
            component_indices = model_input.component_index[0].detach().cpu().tolist()
            references = tuple(
                model_input.structure.system.component_reference(component) for component in component_indices
            )
            selected = (model_input.residue_mask[0] & model_input.design_positions[0]).tolist()
            pack_positions = [reference for reference, keep in zip(references, selected, strict=True) if keep]
        packs = self.packer.pack_sequences(
            structure,
            sequences=[_candidate_protein_sequences(candidate.system) for candidate in design_result.candidates],
            pack_positions=pack_positions,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return design_result, packs


def _candidate_protein_sequences(
    system: MolecularSystem,
) -> dict[str, tuple[str, ...]]:
    r"""
    Map each protein instance to its realized component-ID sequence.

    The composition passes only complete per-instance component-ID sequences to the
    Packer; it never forwards logits, ranks, interface masks, or feature tensors.
    Candidate systems partition changed copies within each original entity exactly as
    the designer realizes them, so reading instances here follows that realization
    without a second split.
    """

    instance_entity = system.instance_entity.detach().cpu().tolist()
    entity_type = system.entity_type.detach().cpu().tolist()
    instance_symbol = system.instance_symbol.detach().cpu().tolist()
    component_ptr = system.instance_component_ptr.detach().cpu().tolist()
    component_symbol = system.component_symbol.detach().cpu().tolist()
    sequences: dict[str, tuple[str, ...]] = {}
    for instance, entity in enumerate(instance_entity):
        if entity_type[entity] != ENTITY_PROTEIN:
            continue
        instance_id = system.symbols[instance_symbol[instance]]
        sequences[instance_id] = tuple(
            system.symbols[symbol] for symbol in component_symbol[component_ptr[instance] : component_ptr[instance + 1]]
        )
    return sequences
