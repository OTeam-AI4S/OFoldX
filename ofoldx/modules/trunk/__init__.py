r"""
Representation-refinement trunks.

The trunk is the block-stacked core that iteratively refines the single, MSA, and pair
representations before structure prediction. This package groups the representation
trunks used by OFoldX: the AlphaFold 2 Evoformer stack, the Pairformer stack
(single + pair, no MSA), and MSA-conditioned pair-update trunks.

Exports:
    - [`PairformerStack`][ofoldx.modules.trunk.pairformer.PairformerStack] and its
      [`PairformerBlock`][ofoldx.modules.trunk.pairformer.PairformerBlock].
    - [`EvoformerStack`][ofoldx.modules.trunk.evoformer.EvoformerStack].
    - [`MsaTrunk`][ofoldx.modules.trunk.msa.MsaTrunk],
      [`CategoricalMsaTrunk`][ofoldx.modules.trunk.msa.CategoricalMsaTrunk], and the pair-only
      [`FinalMsaPairBlock`][ofoldx.modules.trunk.msa.FinalMsaPairBlock] that caps it.
"""

from __future__ import annotations

from .evoformer import EvoformerStack
from .msa import CategoricalMsaTrunk, FinalMsaPairBlock, MsaTrunk
from .pairformer import PairformerBlock, PairformerStack
