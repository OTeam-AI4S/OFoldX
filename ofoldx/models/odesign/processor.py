r"""
ODesign structure processing.

Defines the ODesign structure processor, which turns a parsed structure (or structure file path) into a
batched typed [`ODesignInput`][ofoldx.models.odesign.model.ODesignInput] following the ODesign token
schema (restype + profile + deletion_mean + hotspot), and the helper that supplies ODesign's canonical
restype vocabulary. The processor registers itself under the ODesign architecture keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from ...data.ccd import CcdProvider
from ...data.structure import ResidueKey
from ...processors.registry import PROCESSORS
from ...processors.structure import StructureProcessor
from ...processors.utils import StructureInput
from .model import ODesignInput


@PROCESSORS.register("odesign")
@PROCESSORS.register("odesign_design")
class ODesignProcessor(StructureProcessor):
    r"""
    Process structure inputs with the ODesign token schema.

    Args:
        single_size: Width of the single representation prefix reserved in token features.
        token_feature_size: Total token feature width emitted by the processor.
        atom_feature_size: Total atom feature width emitted by the processor.
        residue_vocabulary: Optional residue vocabulary override.
        element_vocabulary: Optional element vocabulary override.
        element_channels: Number of element one-hot channels.
        atom_name_characters_count: Number of atom-name characters to encode.
        atom_name_character_channels: Number of one-hot channels per atom-name character.
        ccd: Optional CCD provider override. Processors with custom providers cannot be saved.
        reference_conformer_augment: If ``True``, randomly rigid-transform reference conformers.
        augment_seed: Optional seed for reference-conformer augmentation.
        ligand_atom_rename: If ``True``, rename ligand atom-name features to ``ELEM+count``.
        materialize_missing_atoms: If ``True``, synthesize expected-but-unresolved heavy atoms.
        hotspot_value: Constant value written into the ODesign hotspot feature channel.
        constraint_distogram: If ``True``, populate the ``constraint_distogram`` field from resolved
            condition-token distances.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("ofoldx/odesign")  # doctest: +SKIP
        >>> model_input = processor("complex.cif")  # doctest: +SKIP
    """

    input_class = ODesignInput

    def __init__(
        self,
        *,
        single_size: int = 384,
        token_feature_size: int = 453,
        atom_feature_size: int = 386,
        residue_vocabulary: Sequence[str] | None = None,
        element_vocabulary: Sequence[str] | None = None,
        element_channels: int = 129,
        atom_name_characters_count: int = 4,
        atom_name_character_channels: int = 64,
        ccd: CcdProvider | None = None,
        reference_conformer_augment: bool = False,
        augment_seed: int | None = None,
        ligand_atom_rename: bool = False,
        materialize_missing_atoms: bool = False,
        hotspot_value: float = 1.0,
        constraint_distogram: bool = False,
    ) -> None:
        super().__init__(
            single_size=single_size,
            token_feature_size=token_feature_size,
            atom_feature_size=atom_feature_size,
            residue_vocabulary=(
                default_odesign_residue_vocabulary() if residue_vocabulary is None else residue_vocabulary
            ),
            element_vocabulary=element_vocabulary,
            element_channels=element_channels,
            atom_name_characters_count=atom_name_characters_count,
            atom_name_character_channels=atom_name_character_channels,
            ccd=ccd,
            reference_conformer_augment=reference_conformer_augment,
            augment_seed=augment_seed,
            ligand_atom_rename=ligand_atom_rename,
            materialize_missing_atoms=materialize_missing_atoms,
        )
        self.hotspot_value = hotspot_value
        self.constraint_distogram = constraint_distogram

    def __call__(self, structure: StructureInput) -> ODesignInput:  # type: ignore[override]
        r"""
        Return a batched ``ODesignInput`` for one structure input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Batched ODesign model input with leading batch size 1. When ``constraint_distogram`` is enabled
            the ``constraint_distogram`` field is populated from representative-atom distances between
            resolved hotspot (condition) tokens.
        """

        model_input = cast(ODesignInput, super().__call__(structure))
        if self.constraint_distogram:
            model_input.constraint_distogram = self._build_constraint_distogram(model_input)
        return model_input

    def _build_constraint_distogram(self, model_input: ODesignInput) -> Tensor:
        r"""
        Return the ``(B, N_token, N_token, 1)`` constraint-distance distogram.

        Mirrors upstream ``generate_spec_constraint_distogram``: pairwise distances between each token's
        representative-atom coordinate, kept only for resolved condition-token pairs (``0`` elsewhere). The
        condition tokens are the hotspot tokens the ODesign token schema marks in its trailing feature
        channel; representative-atom coordinates come from the experimental ``target_positions``.
        """

        target_positions = model_input.target_positions
        token_to_rep_atom = model_input.token_to_rep_atom
        if target_positions is None or token_to_rep_atom is None:
            num_tokens = model_input.token_mask.shape[-1]
            return model_input.token_mask.new_zeros(1, num_tokens, num_tokens, 1, dtype=torch.float32)

        rep_index = token_to_rep_atom.to(dtype=torch.long)
        rep_coords = torch.gather(
            target_positions, -2, rep_index.unsqueeze(-1).expand(*rep_index.shape, target_positions.shape[-1])
        ).to(torch.float32)
        distances = torch.cdist(rep_coords, rep_coords)

        condition = self._condition_token_mask(model_input)
        condition_pair = condition.unsqueeze(-1) & condition.unsqueeze(-2)
        distances = torch.where(condition_pair, distances, torch.zeros_like(distances))
        return distances.unsqueeze(-1)

    def _condition_token_mask(self, model_input: ODesignInput) -> Tensor:
        r"""
        Return the ``(B, N_token)`` boolean condition mask (resolved hotspot tokens).
        """

        token_features = model_input.token_features
        restype_size = 35
        profile_size = 32
        deletion_mean_size = 1
        hotspot_index = self.single_size + restype_size + profile_size + deletion_mean_size
        if token_features.shape[-1] > hotspot_index:
            hotspot = token_features[..., hotspot_index] != 0
        else:
            hotspot = model_input.token_mask.bool()
        return hotspot & model_input.token_mask.bool()

    def _processor_config(self) -> dict[str, object]:
        r"""
        Extend the base processor config with the ODesign-specific serialization knobs.

        Returns:
            Processor config mapping including ``hotspot_value`` and ``constraint_distogram``.
        """

        config = super()._processor_config()
        config["hotspot_value"] = self.hotspot_value
        config["constraint_distogram"] = self.constraint_distogram
        return config

    def _token_features(self, residues: tuple[ResidueKey, ...], residue_names: dict[ResidueKey, str]) -> Tensor:
        r"""
        Build the ODesign per-token feature matrix (restype + profile + deletion_mean + hotspot).

        Falls back to the base processor's token features when the configured ``token_feature_size`` does not
        match the ODesign schema width. Otherwise writes the one-hot restype, the clamped profile one-hot, and
        the constant hotspot channel into the reserved trailing feature slots.

        Args:
            residues: Ordered residue keys for the tokens to featurize.
            residue_names: Mapping from residue key to residue name.

        Returns:
            Per-token feature matrix `[N_token, token_feature_size]`.
        """

        restype_size = 35
        profile_size = 32
        deletion_mean_size = 1
        hotspot_size = 1
        expected_size = self.single_size + restype_size + profile_size + deletion_mean_size + hotspot_size
        if self.token_feature_size != expected_size:
            return super()._token_features(residues, residue_names)

        features = torch.zeros(len(residues), expected_size)
        if not residues:
            return features

        residue_indices = torch.tensor(
            [
                self._residue_indices.get(residue_names.get(residue, "UNK"), self._residue_unknown_index)
                for residue in residues
            ],
            dtype=torch.long,
        )
        rows = torch.arange(len(residues))
        restype_start = self.single_size
        profile_start = restype_start + restype_size
        deletion_mean_start = profile_start + profile_size
        hotspot_start = deletion_mean_start + deletion_mean_size

        features[rows, restype_start + residue_indices] = 1
        profile_indices = residue_indices.clamp(max=profile_size - 1)
        features[rows, profile_start + profile_indices] = 1
        features[:, hotspot_start] = self.hotspot_value
        return features


def default_odesign_residue_vocabulary() -> tuple[str, ...]:
    r"""
    Return ODesign's 35-class restype vocabulary in reference order.

    Returns:
        Residue vocabulary tuple indexed by emitted ODesign residue-type ids.

    Examples:
        >>> vocabulary = default_odesign_residue_vocabulary()
        >>> len(vocabulary)
        35
        >>> vocabulary[:3]
        ('ALA', 'ARG', 'ASN')
        >>> vocabulary[20]
        'UNK'
    """

    return (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "UNK",
        "A",
        "G",
        "C",
        "U",
        "N",
        "DA",
        "DG",
        "DC",
        "DT",
        "DN",
        "-",
        "-P",
        "-N",
        "-L",
    )
