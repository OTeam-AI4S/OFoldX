r"""
Explicit ownership contracts for regional compilation.
"""

from __future__ import annotations

from torch import nn

CompilationRegionGroup = tuple[nn.Module, ...]
CompilationRegionGroups = tuple[CompilationRegionGroup, ...]


class CompilationRegionOwner:
    r"""Mixin for modules that explicitly own local compilation regions."""

    def local_compilation_region_groups(self) -> CompilationRegionGroups:
        r"""Return this owner's local groups of interchangeable regions."""

        raise NotImplementedError


class HomogeneousBlockRegions(CompilationRegionOwner):
    r"""Expose a homogeneous ``blocks`` collection as one compilation group."""

    blocks: nn.ModuleList

    def local_compilation_region_groups(self) -> CompilationRegionGroups:
        blocks = tuple(self.blocks)
        return (blocks,) if blocks else ()
