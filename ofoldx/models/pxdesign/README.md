# PxDesign

PxDesign is a protein-backbone design generator that samples backbone coordinates
from design conditioning via AlphaFold 3-style EDM diffusion. Beyond backbone
generation it drives the official design pipeline: ProteinMPNN sequence assignment
and structure-predictor filtering.

Distinctive to this family are its condition embedder (fusing an atom-to-token
encoder with restype/pLDDT/hotspot/design token features), the conditional-template
pair embedder, the EDM coordinate sampler with step-scale schedules, and the design
distogram head.

Shared neural network building blocks live in `ofoldx.modules`; this directory
holds the family's config, typed I/O, checkpoint conversion, and model.
