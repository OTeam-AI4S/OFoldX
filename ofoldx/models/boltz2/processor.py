r"""
Boltz2 structure processing.
"""

from __future__ import annotations

from typing import cast

from ...processors.registry import PROCESSORS
from ...processors.residue_token import ResidueTokenStructureProcessor
from ...processors.utils import StructureInput
from .model import Boltz2Input


@PROCESSORS.register("boltz2")
@PROCESSORS.register("boltz2_affinity")
class Boltz2Processor(ResidueTokenStructureProcessor):
    r"""
    Convert structure inputs into the Boltz2Input contract.

    Reuses the shared residue-token structure processor and only pins the model-specific
    ``input_class`` so tokenized features are returned as a
    [`Boltz2Input`][ofoldx.models.boltz2.model.Boltz2Input].

    Examples:
        >>> from ofoldx import AutoProcessor  # doctest: +SKIP
        >>> processor = AutoProcessor.from_pretrained("oteam/boltz2")  # doctest: +SKIP
        >>> batch = processor("complex.cif")  # doctest: +SKIP
    """

    input_class = Boltz2Input

    def __call__(self, structure: StructureInput) -> Boltz2Input:  # type: ignore[override]
        r"""
        Return a batched ``Boltz2Input`` for one structure input.

        Args:
            structure: Parsed structure object or structure file path.

        Returns:
            Batched Boltz2 model input with leading batch size 1.
        """

        return cast(Boltz2Input, super().__call__(structure))
