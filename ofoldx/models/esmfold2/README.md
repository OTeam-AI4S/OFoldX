---
language: protein
tags:
  - Biology
  - Protein
  - Structure Prediction
license: agpl-3.0
base_model: biohub/ESMFold2 + multimolecule/esmc-6b
library_name: ofoldx
pipeline_tag: other
---

# ESMFold2

> [!IMPORTANT]
All-atom biomolecular structure prediction driven by ESMC-6B representations and optional protein MSAs.

> [!CAUTION]
> This family is intentionally absent from the OFoldX registry and no OFoldX ESMFold2 repository is published.
> Registration remains blocked on a published MultiMolecule ESMC asset, caller-owned production assets for maintained
> ESMFold2 and ESMFold2-Fast end-to-end parity, and reference parity for canonical N/DN generic-base atom/full inputs.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf) by Salvatore Candido, Thomas Hayes, Alexander Derry, et al.

The OFFICIAL repository of ESMFold2 is at [Biohub/esm](https://github.com/Biohub/esm).

**The team releasing ESMFold2 did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

ESMFold2 is an all-atom biomolecular structure predictor. Its released input supports standard and modified proteins, DNA, RNA, reference-conformer ligands, covalent assemblies, A3M deletion features, and paired protein MSA evidence. OFoldX implements the released input embedder, MSA encoder, pair-only folding trunk, diffusion structure head, and confidence head. The required ESMC-6B encoder is the ordinary public `model.esmc` child module implemented by MultiMolecule; its weights live in the same OFoldX state dict and repository as the folding head. `model.embeddings(input)` exposes the released token and pair embeddings directly, without running diffusion or wrapping them in another public object.

The upstream `esm` input builder also emits `disto_cond` and `disto_cond_mask`, but the released `ESMFold2Model.forward` accepts them only through `**kwargs` and never reads them; the released checkpoint has no corresponding input parameters. OFoldX therefore rejects conditions instead of presenting this upstream no-op as supported conditioning. Likewise, canonical SMILES chemistry must carry an explicit reference conformer: OFoldX does not silently generate geometry inside a Processor.

ESMFold2 and ESMFold2-Fast are checkpoint variants of this release graph. Fast uses 24 rather than 48 pair-folding blocks, disables the MSA encoder, rejects explicit MSA evidence through its paired Processor, and retains the published `0.1` language-model masking default. The full checkpoint uses the 48-block trunk, MSA encoder, and zero language-model masking default. The older `type="experimental"` checkpoints used by the released binder-inversion notebook have a different recurrent graph and confidence contract; they are not silently loaded as variants of `ESMFold2Model`.

### Binder design and family verb inventory

Beyond the released single-sequence `fold`, ESMFold2 owns a released binder-**design** operation: gradient descent on per-position soft-sequence logits through the experimental recurrent graph, followed by a discrete refold of the optimized sequence. Its output is a folded `target|binder` complex whose binder length varies across candidates, so it is coordinate-producing and variable-length and cannot reuse the shared coordinate-free, fixed-length `DesignResult`. OFoldX therefore defines a distinct, family-owned `ESMFold2DesignResult` (with `ESMFold2DesignCandidate`) in this directory; it is not a subclass of the shared carrier. `design` is not a `standard_task`: it is exposed only through the concrete `ESMFold2Pipeline`, alongside a `fold` convenience wrapper over the shared `FoldResult`.

This wave freezes the `ESMFold2Pipeline.design` signature at contract level only. Its body is blocked on the experimental recurrent-graph checkpoints, which are not local, so it raises `NotImplementedError` and the Pipeline is intentionally **not** registered (no `register_pipeline` / `pipeline.json`). When those checkpoints land, the experimental recurrent graph — which has a different confidence contract than `ESMFold2Model` — becomes a separate registered architecture `ofoldx.esmfold2_experimental`; it is never silently loaded as a variant of the released model.

### Model Specification

| Variant | Trunk Blocks | MSA Encoder | Single Size | Pair Size | Checkpoint Loops | Checkpoint Sampler Steps |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ESMFold2 | 48 | Yes | 384 | 256 | 3 | 14 |
| ESMFold2-Fast | 24 | No | 384 | 256 | 3 | 14 |

> [!NOTE]
> The folding head is driven by ESMC-6B representations (2560 hidden size, 80 language-model layers). A published
> OFoldX ESMFold2 repository is self-contained: `AutoModel(...)` restores both the MultiMolecule ESMC child and the
> folding head, so sequence folding never performs a hidden second Hub resolution or requires `load_esmc()`.

The Biohub checkpoint contains four tensors under three `ConfidenceHead` modules that its released `forward` never
reads. The converter discards exactly those named tensors and rejects every other unexpected key. No production-weight
result is stored in this repository. The maintained `scripts/parity/esmfold2.py` public-output comparison reads the
reference checkout from `~/reference` and caller-owned weights from `~/weights`, leaves generated reports local, and
remains outside frequent CI. Its weight-free semantic mode covers the public molecular, MSA, and Result boundaries;
production runs of both released heads and canonical N/DN generic-base atom/full-input comparison remain required.

The released input builder and current audited confidence implementation both encode non-polymers as molecule type `3`;
OFoldX preserves that identity so ligand interface pLDDT uses the intended ligand weights.

The concrete `fold` method exposes every effective released inference control directly: recurrence loops, diffusion
samples and steps, language-model masking and per-loop dropout, MSA depth/subsampling/column masking, sampler noise and
step scales, maximum inference sigma, and convergence-based denoising exit. The upstream standard-model `early_exit`
boolean is not copied because that argument is swallowed by `**kwargs` and has no effect; OFoldX instead exposes the
actual `denoising_early_exit_rmsd` computation implemented by the released structure head.

`AutoModel("ofoldx/esmfold2")` and `AutoProcessor("ofoldx/esmfold2")` will become valid only after every blocker above is
closed and the architecture is registered.

### Links

- **Code**: [ofoldx.models.esmfold2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/esmfold2)
- **Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf)
- **Developed by**: Salvatore Candido, Thomas Hayes, Alexander Derry, et al.
- **Model type**: All-atom structure predictor with a pair-only folding trunk and a diffusion structure head, conditioned on ESMC-6B language-model representations
- **Original Repository**: [Biohub/esm](https://github.com/Biohub/esm)

## Training

This implementation exposes the differentiable single-step denoising `forward` primitive. The paper's diffusion MSE, smooth-LDDT, polymer–ligand bond,
distogram, and confidence losses are useful provenance; reproducing every unpublished target, mask, mini-rollout, and
reduction is not a registration requirement. OFoldX combines the released Biohub folding head with the paired
MultiMolecule ESMC checkpoint and does not redistribute the upstream training corpora.

## Citation

```bibtex
@misc{candido2026language,
  title = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and others},
  year = {2026},
  url = {https://biohub.ai/papers/esm_protein.pdf},
  note = {Preprint}
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

Please contact the authors of the [ESMFold2 paper](https://biohub.ai/papers/esm_protein.pdf) for questions or comments on the paper/model.

## Numerical epsilon note

- **Confidence eps `1e-6`** (`_CONFIDENCE_EPS` `model.py:3015`, plddt floor `clamp(min=1e-6)` `:3178`) is **upstream-faithful — keep, do not unify.** Part of the inherent per-family confidence-eps spread (esmfold2 1e-6 / af3 1e-8+1e-6 / protenix 1e-8 / boltz 1e-5).
- **OPM denominator** uses a divide-before-projection form with `n_valid.clamp(min=1.0)` and `denom_eps=0.0` (`model.py:2923,2931`) — a structurally different normalization from the `count + 1e-3` divide-after form used by AF2/AF3/Boltz. Not a candidate to fold into the `1e-3` eps.
- **Softmax mask sentinels `-1e5` (`model.py:3005`) / `-1e9` (`model.py:2445`)** are numerically inert: in fp32/bf16 the post-softmax output is bit-identical to `torch.finfo(dtype).min` (measured max-diff 0.00e+00). `finfo.min` is the dtype-safe unified form (`-1e5`/`-1e9` overflow fp16). Unify for hygiene, zero error. See `EPSILON_ANALYSIS.md`.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- Per-atom coordinate parity has a cross-implementation floor of **~0.215 Å**;
  OFoldX's global scores (pLDDT, PAE, PTM) sit **at parity**. Coordinates below that
  floor are a known cross-implementation limit, not a regression.
- Softmax mask sentinels (`-1e5`, `-1e9`) are numerically inert (measured post-softmax
  max-diff `0.0`); `finfo.min` is the dtype-safe unified form. See the Numerical
  epsilon note.
- The confidence epsilon `1e-6` and the OPM divide-before-projection normalization
  are kept upstream-faithful and are not unified with AF2/AF3/Boltz forms.
- A parity run applies only to the checkpoint and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **ESMFold2 / ESMFold2-Fast weights:** released by Chan Zuckerberg Biohub under
  MIT with an additional "other" third-party notice
  ([ESM THIRD_PARTY_NOTICE](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md)).
- **Redistribution status: GREEN** for the ESMFold2 / ESMFold2-Fast folding weights
  (MIT), provided the third-party notice is retained. **Caveat:** the separate
  ESM-C-6B trunk weights are copyleft **AGPL-3.0**; if any variant bundles them,
  redistribution carries AGPL source obligations and must be treated as AMBER.
