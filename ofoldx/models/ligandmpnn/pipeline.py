r"""OFoldX's biological operation boundary for released LigandMPNN computations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from typing_extensions import Self

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import ComponentReference, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...pipelines import PairedPipeline, resolve_generator
from ...results import DesignResult, MpnnProbabilityResult, PackResult
from ..mpnn_pdb import read_mpnn_pdb
from .model import LigandMpnnModel
from .processor import LigandMpnnProcessor

if TYPE_CHECKING:
    from ..mpnn_packer import MpnnPackerPipeline


def _candidate_protein_sequences(
    system: MolecularSystem,
) -> dict[str, tuple[str, ...]]:
    r"""Map each protein instance to its realized component-ID sequence.

    The composition passes only complete per-instance component-ID sequences to the
    Packer; it never forwards logits, ranks, interface masks, or feature tensors.
    Candidate systems partition changed copies within each original entity exactly as
    the designer realizes them, so reading instances here follows that realization
    without a second split.
    """

    sequences: dict[str, tuple[str, ...]] = {}
    for instance_id, entity_id in system.instances.items():
        entity = system.entities[entity_id]
        if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
            sequences[instance_id] = tuple(component.component_id for component in entity.components)
    return sequences


class LigandMpnnPipeline(PairedPipeline[LigandMpnnModel, LigandMpnnProcessor]):
    r"""Bind one LigandMpnn Model-Processor pair for biological operations."""

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> DesignResult:
        r"""Design sequences over selected positions with ligand and residue context."""

        features = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=features.backbone_coordinates.device)
        return self.model.design(
            features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return sequence-conditioned autoregressive probabilities over orders."""

        features = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=features.backbone_coordinates.device)
        return self.model.autoregressive_probabilities(
            features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def sequence_free_autoregressive_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[TiedResidueIdentities] = (),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return order-dependent sequence-free autoregressive probabilities."""

        features = self.processor(
            input,
            positions=positions,
            conditions=conditions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=features.backbone_coordinates.device)
        return self.model.sequence_free_autoregressive_probabilities(
            features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def conditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-last conditional probabilities for every row."""

        features = self.processor(
            input,
            positions=positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=features.backbone_coordinates.device)
        return self.model.conditional_probabilities(
            features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )

    def unconditional_probabilities(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
        num_decoding_orders: int = 1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MpnnProbabilityResult:
        r"""Return target-first sequence-free probabilities for every row."""

        features = self.processor(
            input,
            positions=positions,
            use_atom_context=use_atom_context,
            filter_by_occupancy=filter_by_occupancy,
        )
        generator = resolve_generator(seed, generator, device=features.backbone_coordinates.device)
        return self.model.unconditional_probabilities(
            features,
            num_decoding_orders=num_decoding_orders,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )


class LigandMpnnDesignAndPackPipeline:
    r"""Compose the LigandMpnn designer with the MpnnPacker over one source.

    The released combined command designs sequences with ligand and residue context
    and then packs each design. Both stages read one canonical ``Structure`` parsed
    exactly once, keep three genuinely distinct context controls, and share a single
    random stream advanced design-first then pack-major across every candidate.
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
        r"""Place both children on one device and dtype in place."""

        self.designer.to(device=device, dtype=dtype)
        self.packer.to(device=device, dtype=dtype)
        return self

    def design(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        num_candidates: int = 1,
        temperature: float = 0.1,
        coordinate_noise: float = 0.0,
        use_side_chain_context: bool = False,
        use_packing_atom_context: bool = True,
        repack_everything: bool = False,
        num_packs_per_candidate: int = 4,
        num_denoising_steps: int = 3,
        num_torsion_samples: int = 16,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[DesignResult, tuple[PackResult, ...]]:
        r"""Design sequences on one canonical Structure and pack each candidate."""

        structure = read_mpnn_pdb(input) if isinstance(input, (str, PathLike)) else input
        features = self.designer.processor.process_structure(
            structure,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )
        device = features.valid_mask.device
        generator = resolve_generator(seed, generator, device=device)
        design_result = self.designer.model.design(
            features,
            num_candidates=num_candidates,
            temperature=temperature,
            coordinate_noise=coordinate_noise,
            use_side_chain_context=use_side_chain_context,
            generator=generator,
        )
        if repack_everything:
            packing_positions: Sequence[ComponentReference] | None = None
        else:
            references = features.component_references
            selected = (features.valid_mask[0] & features.design_mask[0]).tolist()
            packing_positions = [
                reference for reference, keep in zip(references, selected) if keep and reference is not None
            ]
        packs = self.packer.pack_sequences(
            structure,
            sequences=[_candidate_protein_sequences(candidate.system) for candidate in design_result.candidates],
            positions=packing_positions,
            use_atom_context=use_packing_atom_context,
            filter_by_occupancy=filter_by_occupancy,
            num_samples=num_packs_per_candidate,
            num_denoising_steps=num_denoising_steps,
            num_torsion_samples=num_torsion_samples,
            generator=generator,
        )
        return design_result, packs
