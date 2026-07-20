r"""Shared wwPDB Chemical Component Dictionary (CCD) resolver.

`ofoldx.data.components.get_component` stays audited-only and does zero I/O
(Principle 7): it never guesses a chemical graph and never touches the network.
This package is the *other* half — the import-time resolver a CCD-code-folding
family calls at its importer / Processor boundary to lower a wwPDB CCD component
into a canonical `ChemicalComponent` VALUE. Once lowered, the family embeds those
values; it never performs a live registry lookup at tensor-compute time.

The CCD is distributed as a Hub `dataset` repository (`ofoldx/ccd`, Parquet, one
component per row). The `revision` argument is the exact wwPDB snapshot date, so a
resolved value is fully reproducible and provenance-pinned.
"""

from __future__ import annotations

from .resolver import CCD_DATASET_REPOSITORY, resolve_ccd_component
