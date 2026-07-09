r"""
Friendly Tier-0 result objects.

The task verbs ([`ofoldx.fold`][ofoldx.fold] / [`ofoldx.design`][ofoldx.design]) return these instead of a raw
tensor contract: a biologist gets ``.plddt`` (decoded confidence), ``.coords``, ``.sequence``, and
``.save_pdb(path)`` / ``.save_cif(path)`` directly, while the underlying typed contract stays reachable via
``.raw`` for the builder who wants tensors. ``save_pdb`` reconstructs a [`Structure`][ofoldx.data.structure.Structure]
by putting the predicted coordinates onto the *input* structure's atom records, which works whenever the input
was a structure/path (the common "fold this complex" case). Sequence-input structure export needs a
model-specific atom layout and is left to each family's writer.
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any

from .data import coords_to_structure
from .io import write_structure

if TYPE_CHECKING:
    from torch import Tensor

    from .data.structure import Structure
    from .models.contract import DesignOutput, StructurePredictionOutput
    from .modules.representation import Coords


class StructurePredictionResult:
    r"""
    Friendly wrapper around a structure prediction.

    Args:
        output: The raw [`StructurePredictionOutput`][ofoldx.models.StructurePredictionOutput].
        source: The input structure (when folding a structure/path), used to reconstruct atom identities for
            ``save_pdb``; ``None`` for sequence input.

    Examples:
        >>> result = ofoldx.fold("complex.cif", model="oteam/alphafold3")  # doctest: +SKIP
        >>> result.plddt.mean()  # doctest: +SKIP
        >>> result.save_pdb("out.pdb")  # doctest: +SKIP
    """

    def __init__(self, output: StructurePredictionOutput, *, source: Structure | None = None) -> None:
        self.raw = output
        self._source = source

    @property
    def plddt(self) -> Tensor | None:
        r"""Decoded per-token/atom pLDDT in ``[0, 100]`` (``None`` if the model has no pLDDT head)."""

        return self.raw.plddt

    @property
    def coords(self) -> Coords:
        r"""Predicted coordinates as a typed [`Coords`][ofoldx.modules.representation.Coords] carrier."""

        return self.raw.coords

    def to_structure(self, *, sample: int = 0) -> Structure:
        r"""
        Reconstruct a [`Structure`][ofoldx.data.structure.Structure] for one predicted sample.

        Args:
            sample: Index of the predicted sample to export.

        Returns:
            The input structure with predicted coordinates.

        Raises:
            ValueError: If there is no source structure (sequence input), or the predicted atom count does not
                match the input.
        """

        if self._source is None:
            raise ValueError(
                "save_pdb needs a structure/path input to reconstruct atom identities; sequence-input structure "
                "export is model-specific and not yet wired for this family"
            )
        return coords_to_structure(self._source, self.raw.pred_coords, sample=sample)

    def save_pdb(self, path: str | PathLike[str], *, sample: int = 0) -> None:
        r"""Write the predicted structure to a PDB file."""

        write_structure(self.to_structure(sample=sample), path, format="pdb")

    def save_cif(self, path: str | PathLike[str], *, sample: int = 0) -> None:
        r"""Write the predicted structure to an mmCIF file."""

        write_structure(self.to_structure(sample=sample), path, format="cif")


class DesignResult:
    r"""
    Friendly wrapper around a design output: iterate candidates, each with optional structure export.

    Args:
        output: The raw [`DesignOutput`][ofoldx.models.DesignOutput].
        source: Optional input structure used to reconstruct atom identities for ``save_pdb``.
    """

    def __init__(self, output: DesignOutput, *, source: Structure | None = None) -> None:
        self.raw = output
        self._source = source

    @property
    def candidates(self) -> list[Any]:
        r"""The raw design candidates."""

        return self.raw.candidates

    @property
    def sequences(self) -> list[str]:
        r"""
        Decoded residue sequences, one per candidate that carries tokens and a registered decoder.

        Dispatches through the per-architecture decoder registry (``metadata["architecture"]`` → decoder), so a
        family plugs its decoder in without this module importing the family.
        """

        from .results_registry import _decode

        decoded: list[str] = []
        for candidate in self.raw.candidates:
            sequence_tokens = candidate.sequence_tokens
            if sequence_tokens is None:
                continue
            architecture = (candidate.metadata or {}).get("architecture")
            sequence = _decode(architecture, sequence_tokens)
            if sequence is not None:
                decoded.append(sequence)
        return decoded

    @property
    def sequence(self) -> str | None:
        r"""The first decoded residue sequence, or ``None`` when nothing is decodable."""

        sequences = self.sequences
        return sequences[0] if sequences else None

    def __len__(self) -> int:
        return len(self.raw.candidates)

    def save_pdb(self, path: str | PathLike[str], *, candidate: int = 0, sample: int = 0) -> None:
        r"""Write one candidate's predicted structure to a PDB file (requires a structure/path input)."""

        prediction = self.raw.candidates[candidate].structure_prediction
        if prediction is None:
            raise ValueError("this design candidate carries no predicted structure to write")
        if self._source is None:
            raise ValueError(
                "save_pdb needs a structure/path input to reconstruct atom identities; sequence-input structure "
                "export is model-specific and not yet wired for this family"
            )
        structure = coords_to_structure(self._source, prediction.pred_coords, sample=sample)
        write_structure(structure, path, format="pdb")
