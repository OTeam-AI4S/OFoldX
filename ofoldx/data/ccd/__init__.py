r"""
wwPDB Chemical Component Dictionary resolution.

Components are read from a revision-pinned Hub dataset or local Parquet snapshot
and lowered to canonical ``ChemicalComponent`` values at an input boundary.
"""

from __future__ import annotations

from .resolver import CCD_DATASET_REPOSITORY, resolve_ccd_component

__all__ = ["CCD_DATASET_REPOSITORY", "resolve_ccd_component"]
