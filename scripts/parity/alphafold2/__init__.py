r"""Deterministic deep-forward golden-dump parity for AlphaFold 2 (monomer + multimer).

Unlike ``scripts/parity/alphafold2.py`` (which runs each framework's own stochastic
feature pipeline and is therefore blocked on the released MSA-sampling contract), the
modules here feed a single set of FIXED processed features into both the reference
(OpenFold / DeepMind) trunk+structure module and the OFoldX trunk+structure module and
compare single / pair / positions / plddt-logits at relative tolerance. No stochastic
MSA sampling, ensembling, or relaxation is exercised.
"""
