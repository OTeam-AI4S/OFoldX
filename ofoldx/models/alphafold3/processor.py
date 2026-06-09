r"""
AlphaFold3 structure processing.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from ...data.structure import ResidueKey
from ...processors.registry import PROCESSORS
from ...processors.structure import StructureProcessor
from ...processors.utils import StructureInput
from .model import AlphaFold3Input

_OFFICIAL_AF3_RESIDUE_VOCABULARY = (
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
    "GAP",
    "A",
    "G",
    "C",
    "U",
    "DA",
    "DG",
    "DC",
    "DT",
    "N",
)


@PROCESSORS.register("alphafold3")
class AlphaFold3Processor(StructureProcessor):
    r"""
    Featurize a parsed structure (or a structure file) into the typed tensors
    [`AlphaFold3Model`][ofoldx.models.alphafold3.AlphaFold3Model] consumes.

    Registered under ``"alphafold3"`` so it can be resolved by architecture name. Inherits the shared
    featurization pipeline from [`StructureProcessor`][ofoldx.processors.structure.StructureProcessor]
    and only pins ``input_class`` to [`AlphaFold3Input`][ofoldx.models.alphafold3.AlphaFold3Input] so the
    output is the model's typed input dataclass.

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
        >>> batch = processor("complex.cif")  # doctest: +SKIP
    """

    input_class = AlphaFold3Input

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("token_feature_size", 447)
        kwargs.setdefault("residue_vocabulary", _OFFICIAL_AF3_RESIDUE_VOCABULARY)
        super().__init__(**kwargs)

    def __call__(self, structure: StructureInput) -> AlphaFold3Input:
        r"""
        Return a batched ``AlphaFold3Input`` for one structure input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Batched AlphaFold3 model input with leading batch size 1.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/alphafold3")  # doctest: +SKIP
            >>> batch = processor("complex.cif")  # doctest: +SKIP
            >>> batch.token_mask.shape  # doctest: +SKIP
            torch.Size([1, 37])
        """

        return cast(AlphaFold3Input, super().__call__(structure))

    def _token_features(self, residues: tuple[ResidueKey, ...], residue_names: dict[ResidueKey, str]) -> Tensor:
        features = torch.zeros(len(residues), self.token_feature_size)
        tail_size = max(self.token_feature_size - self.single_size, 0)
        if tail_size != 63:
            return super()._token_features(residues, residue_names)

        residue_indices = torch.tensor(
            [self._residue_index(residue_names[residue]) for residue in residues],
            dtype=torch.long,
        )
        token_indices = torch.arange(len(residues))
        features[token_indices, self.single_size + residue_indices] = 1
        features[token_indices, self.single_size + 31 + residue_indices] = 1
        return features

    def _residue_index(self, residue_name: str) -> int:
        if residue_name == "DN":
            residue_name = "N"
        return super()._residue_index(residue_name)
