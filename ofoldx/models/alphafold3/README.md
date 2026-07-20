---
language:
  - protein
  - rna
  - dna
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
library_name: ofoldx
pipeline_tag: other
---

# AlphaFold 3

> [!IMPORTANT]
All-atom biomolecular structure prediction for proteins, nucleic acids, ligands, ions, and modified residues.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) by Josh Abramson, Jonas Adler, Jack Dunger, et al.

The OFFICIAL repository of AlphaFold 3 is at [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3).

**The team releasing AlphaFold 3 did not write this model card, so this model card has been written by the OFoldX team.**

> [!NOTE]
> Official DeepMind AlphaFold 3 parameters are governed by the DeepMind model-parameter terms and are NOT redistributed by OFoldX. This card describes the AlphaFold 3 architecture entry point in the OFoldX runtime; the model weights must be obtained separately from DeepMind.

> [!NOTE]
> The architecture id `ofoldx.alphafold3` is registered as one pair `(AlphaFold3Model,
> AlphaFold3Processor)`. `fold` is a public family method that runs the released seed-and-sample
> diffusion pipeline and returns an `AlphaFold3FoldResult` preserving the per-seed sample groups
> (it does not flatten into the shared `FoldResult`), so it is called directly rather than through a
> generic standard Pipeline (`standard_tasks=()`). CCD-code ligands, ions, modified residues, and
> glycan monomers are lowered to embedded chemistry via the shared `ofoldx/ccd` resolver at the
> Processor boundary (values in, no live registry lookup at tensor-compute time).

## Model Details

AlphaFold 3 is an all-atom structure-prediction model that jointly predicts the folded coordinates of complexes spanning proteins, nucleic acids, small-molecule ligands, ions, and covalent modifications. The OFoldX implementation follows the released architecture: input features are embedded into single (per-token) and pair representations, a deep Pairformer trunk refines those representations across recycling iterations, and a diffusion coordinate module denoises atom positions with an EDM-style reverse-diffusion sampler. A confidence head reads the trunk and sampled coordinates to produce pLDDT, PAE, PDE, and resolved-atom estimates, and a distogram head predicts binned inter-residue distances. An optional template stack embeds structural templates into the pair path.

### Model Specification

| Trunk Blocks | Single Size | Pair Size | Denoiser Blocks | Recycling Steps | Sampler Steps |
| ------------ | ----------- | --------- | --------------- | --------------- | ------------- |
| 48           | 384         | 128       | 24              | 4               | 200           |

> [!NOTE]
> These dimensions are the OFoldX default `AlphaFold3Config` values for the released architecture. The parameter count is not listed because AlphaFold 3 is a large trunk-plus-diffusion model and the default config is not instantiated on CPU for this card. The template stack is disabled by default (`template.num_blocks = 0`) and is enabled when template features are provided.

### Links

- **Code**: [ofoldx.models.alphafold3](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/alphafold3)
- **Paper**: [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w)
- **Developed by**: Josh Abramson, Jonas Adler, Jack Dunger, et al.
- **Model type**: All-atom biomolecular structure predictor with a Pairformer trunk, diffusion coordinate module, and confidence head
- **Original Repository**: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)

## Training Details

AlphaFold 3 is trained end-to-end to predict all-atom coordinates of biomolecular complexes, combining a diffusion coordinate objective with distogram and confidence losses. OFoldX records AlphaFold 3 as a distinct architecture and does not train these weights.

### Training Data

The upstream AlphaFold 3 publication describes training on biomolecular structure and sequence resources. OFoldX does not redistribute the training corpus.

### Training Procedure

OFoldX can strict-load local official AlphaFold 3 parameter archives into a native model in memory. It does not persist a repository until the family has a complete Processor, and official DeepMind AlphaFold 3 parameters are not uploaded by the converter.

## Citation

```bibtex
@article{abramson2024accurate,
  author = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  title = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  journal = {Nature},
  volume = {630},
  pages = {493--500},
  year = {2024},
  doi = {10.1038/s41586-024-07487-w}
}
```

> [!NOTE]
> The artifacts distributed in this repository are part of the OFoldX project.
> If OFoldX supports your research, please cite the OFoldX project as follows:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use GitHub issues of [OFoldX](https://github.com/OTeam-AI4S/OFoldX/issues) for questions or comments on the model card.

Please contact the authors of the [AlphaFold 3 paper](https://doi.org/10.1038/s41586-024-07487-w) for questions or comments on the paper/model.

## Numerical epsilon note

- **Confidence normalization eps** is per-family and **upstream-faithful — keep, do not unify.** AF3 uses `1e-8` for the pTM per-alignment normalization (`model.py:976,1046,1144`) and `1e-6` for the contact-weighted PAE rank metric (`model.py:1232`), both faithful to DeepMind `confidences.py`. AF3 and Protenix already agree at `1e-8` on the pTM reduction; Boltz's `1e-5` is the outlier. A shared confidence-eps constant would regress whichever family it does not match. See `EPSILON_ANALYSIS.md`.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- The pair representation reaches a relative-L2 of **0.0587** against the reference
  in this session — marginal and above the tight per-atom targets; it is flagged as
  not-yet-tight rather than fully validated.
- The full diffusion sampler end-to-end and the DSSP-derived features are **not
  validated** in this session's audit.
- Confidence heads and global scores are audited; the confidence-epsilon spread is
  documented in the Numerical epsilon note and kept upstream-faithful.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code:** AlphaFold3 source is Apache-2.0
  ([alphafold3 LICENSE](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE)).
- **AlphaFold3 model parameters:** RESTRICTIVE. The
  [Weights Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
  and
  [Prohibited Use Policy](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md)
  permit only certain **non-commercial** uses, forbid use by or on behalf of
  commercial organizations, and do not grant third-party redistribution.
- **Redistribution status: RED.** No public upload of AF3 parameters (or weights
  derived from them) is permissible without an explicit owner/DeepMind decision.
