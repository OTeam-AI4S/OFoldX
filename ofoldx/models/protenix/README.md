---
language: protein
tags:
  - Biology
  - Protein
  - RNA
  - DNA
  - Structure Prediction
license: agpl-3.0
base_model: bytedance/Protenix
library_name: ofoldx
pipeline_tag: other
---

# Protenix

> [!IMPORTANT]
An AlphaFold 3-shaped all-atom structure predictor for proteins, nucleic acids, and their complexes.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967) by ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al.

The OFFICIAL repository of Protenix is at [bytedance/Protenix](https://github.com/bytedance/Protenix).

**The team releasing Protenix did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

Protenix is an AlphaFold 3-shaped all-atom biomolecular structure predictor. Input tokens and atoms are embedded, refined through a recycled Pairformer trunk that jointly updates the single, pair, and MSA representations, and then handed to an EDM diffusion sampler that denoises atom coordinates in the reverse-diffusion loop. A distogram head predicts binned inter-residue distances, and a confidence head produces per-atom pLDDT together with PAE, PDE, and resolved estimates. The OFoldX implementation maps the Protenix v0.5 checkpoint onto this AF3-style trunk, sampler, and heads.

Distinctive to this family is the Protenix constraint pair-bias path: pocket, token-contact, atom-contact, and substructure constraint features are embedded by four dedicated sub-embedders and added to the initial pair representation before the recycling trunk. The default checkpoint omits the constraint subgraph; the constraint checkpoint always contains all four embedders.

> [!NOTE]
> The two on-disk v0.5 checkpoints are two genuinely different graphs (the constraint checkpoint carries a live four-branch `constraint_embedder`; the default has none) with two different INPUT contracts, so under §3.7.2 they register as two architecture ids: `ofoldx.protenix` (default, artifact repo `ofoldx/protenix`) and `ofoldx.protenix_constraint` (constraint, artifact repo `ofoldx/protenix-constraint`). Each returns `FoldResult`. What separates them is the Processor `__call__` signature: only `ProtenixConstraintProcessor` declares the pocket / token-contact / atom-contact keywords; the default `ProtenixProcessor` signature omits them, so passing any of them raises `TypeError` at the call boundary rather than being gated by a shared runtime flag.

### Model Specification

| Trunk Pairformer Blocks | Single Size | Pair Size | Recycling Steps | Denoiser Token Blocks | Sampler Steps | Confidence Blocks | Distogram Bins |
| ----------------------- | ----------- | --------- | --------------- | --------------------- | ------------- | ----------------- | -------------- |
| 48                      | 384         | 128       | 10              | 24                    | 200           | 4                 | 64             |

> [!NOTE]
> The default configuration disables the template stack and all four constraint embedders, matching
> `protenix_base_default_v0.5.0`. The constraint checkpoint selects the graph containing all four
> embedders. The EDM sampler runs 200 reverse-diffusion steps at inference.

### Variants

| Arch id | Artifact repo | Authoritative code | Checkpoint graph | Constraints |
| --- | --- | --- | --- | --- |
| `ofoldx.protenix` | `ofoldx/protenix` | Protenix v0.5.0 (`26816e9fcb2cd1ff16dc9eaa8feb913f853b9528`) | `protenix_base_default_v0.5.0` | none -- absent from the Processor signature (`TypeError`) |
| `ofoldx.protenix_constraint` | `ofoldx/protenix-constraint` | Protenix v0.5.3 (`a0f95f62430093b46c677f04e3d292ccc5d28d64`) | `protenix_base_constraint_v0.5.0` | pocket, token contact, atom contact (substructure structurally present, zero grid only) |

Both runtimes use ten recycles and 200 reverse-diffusion steps. Direct Model folding retains the released five-sample default and sequential sample chunk of one; the ordinary OFoldX task API can request another positive sample count. Templates are disabled in both authoritative runtimes. Post-v0.5 training-free guidance is not part of either graph.

Seven further v0.5-era model-type entries (`protenix_mini_default_v0.5.0`, `protenix_tiny_default_v0.5.0`, `protenix_mini_esm_v0.5.0`, `protenix_mini_ism_v0.5.0`, `protenix_base_default_v1.0.0`, `protenix_base_20250630_v1.0.0`, `protenix-v2`) are staged and NOT implemented this wave -- their weights are not on disk. Each is a size or internal-graph variant of the single `ofoldx.protenix` architecture (same input contract, same public Model/Processor/Result classes): `mini`/`tiny` shrink the trunk dims, `esm`/`ism` add an internal ESM2-3B feature branch, and `protenix-v2` widens `c_z` to 256 and enables 2-block templates. All are expressible as `ProtenixConfig` field values selected through the checkpoint's `model.json` (§5.4) -- exactly like ProteinMPNN's `ca_only` graph -- and will ship as their own future checkpoint repos (§7.1), NOT as new architecture-id pairs. Only a change to the INPUT contract (as with the constraint checkpoint) earns a second architecture id (§3.7.2).

The constraint runtime maps an atom-contact upper bound to the channel actually trained by the released checkpoint and rejects a nonzero lower bound. This intentionally corrects the documented v0.5.3 training/inference channel swap; it does not add a legacy channel-order mode.

### Links

- **Code**: [ofoldx.models.protenix](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/protenix)
- **Paper**: [Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction](https://doi.org/10.1101/2025.01.08.631967)
- **Developed by**: ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, et al.
- **Model type**: AlphaFold 3-shaped all-atom structure predictor with a recycled Pairformer trunk, EDM diffusion sampler, and distogram / confidence heads
- **Original Repository**: [bytedance/Protenix](https://github.com/bytedance/Protenix)

## Training Details

Protenix is an AlphaFold 3-shaped all-atom structure-prediction model. Training denoises noised ground-truth coordinates through the diffusion objective while the trunk, distogram, and confidence heads are supervised jointly. OFoldX converts the released weights and does not retrain the model.

### Training Data

The upstream Protenix v0.5.0 release reports a 2021-09-30 training-data cutoff. The Protenix training-data pipeline covers released wwPDB/mmCIF bioassemblies, MSAs, CCD cache data, and RCSB sequence clusters. OFoldX does not redistribute the training corpus.

### Training Procedure

The upstream default and constraint artifacts are full checkpoints, including the shared trunk weights. OFoldX constructs the matching ordinary Model graph, translates the released tensor names, and loads it with PyTorch `strict=True`. Conversion returns the loaded Model in memory; it does not retrain the model.

## Citation

```bibtex
@article{protenix2025advancing,
  author = {ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and others},
  title = {Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  year = {2025},
  doi = {10.1101/2025.01.08.631967},
  journal = {bioRxiv}
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

Please contact the authors of the [Protenix paper](https://doi.org/10.1101/2025.01.08.631967) for questions or comments on the paper/model.

## Numerical epsilon note

- **Confidence normalization eps `1e-8`** (`model.py:868` ipTM per-alignment, `:956` contact-weighted PDE) is **upstream-faithful — keep, do not unify.** Faithful to Protenix `sample_confidence`; matches AF3's `1e-8` pTM reduction. The per-family confidence-eps spread (protenix 1e-8 / af3 1e-8+1e-6 / boltz 1e-5 / esmfold2 1e-6) is an inherent cross-model difference, not an OFoldX inconsistency; a shared constant would regress a family. See `EPSILON_ANALYSIS.md`.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- Protenix is an AF3-class folding model shipped with the `default` and `constraint`
  v0.5.0 checkpoints; the diffusion sampler is stochastic and global scores are the
  qualified surface.
- The confidence epsilon `1e-8` is kept upstream-faithful (see the Numerical epsilon
  note) and is not unified with other families.
- A parity run applies only to the checkpoints and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & model parameters:** Apache-2.0, explicitly free for academic
  research and commercial use
  ([Protenix LICENSE](https://github.com/bytedance/Protenix/blob/main/LICENSE)).
- **Redistribution status: GREEN.** The `protenix_base_default_v0.5.0` and
  `protenix_base_constraint_v0.5.0` checkpoints are permissively licensed and freely
  redistributable with the upstream Apache-2.0 notice preserved.
