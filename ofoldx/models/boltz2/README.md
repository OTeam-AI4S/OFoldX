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
  - Binding Affinity
license: agpl-3.0
base_model: boltz-community/boltz-2
library_name: ofoldx
pipeline_tag: other
---

# Boltz-2

> [!IMPORTANT]
Biomolecular structure-prediction and protein-ligand binding-affinity model converted for the OFoldX runtime.

> [!NOTE]
> The family is registered. Two Model/Processor pairs ship over two distinct
> checkpoints: `ofoldx.boltz2` — `Boltz2Model` + `Boltz2Processor`,
> `standard_tasks=("fold",)`, the standalone all-atom folder that resolves through
> the generic standard fold Pipeline and returns a `FoldResult`; and
> `ofoldx.boltz2_affinity` — `Boltz2AffinityModel` + `Boltz2AffinityProcessor`,
> `standard_tasks=()` (its `score` is the concrete two-stage op, not a lossless
> standard `ScorePipeline`). The concrete two-stage `Boltz2AffinityPipeline`
> registers under the same `ofoldx.boltz2_affinity` string in the independent
> pipeline registry.
>
> **Affinity checkpoint provenance.** The affinity weights ship at
> `/data/users/z/weights/boltzgen-1/boltz2_aff.ckpt` — physically co-located with
> BoltzGen-1 weights but scientifically the released Boltz-2 affinity checkpoint
> (`convert_boltz2_affinity_checkpoint`), not a BoltzGen model. The directory
> layout is not a BoltzGen dependency.

## Disclaimer

This is an UNOFFICIAL OFoldX conversion of [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707) by Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al.

The OFFICIAL repository of Boltz-2 is at [jwohlwend/boltz](https://github.com/jwohlwend/boltz).

**The team releasing Boltz-2 did not write this model card, so this model card has been written by the OFoldX team.**

## Model Details

Boltz-2 is an AlphaFold 3-style all-atom model for biomolecular complexes. An atom-conditioned input embedder builds the initial single and pair representations from residue-, MSA-, and atom-level features; a recycled trunk (on-the-fly V2 template embedder, an MSA module, and a deep Pairformer stack) iteratively refines those representations; an EDM diffusion sampler with in-loop weighted rigid alignment draws all-atom coordinates; and a token-level confidence head scores each sample with separate intra- and inter-chain PAE/PDE predictions.

This family ships two architectures. `boltz2` is the full folding model described above. `boltz2_affinity` reuses the Boltz-2 trunk and diffusion module but disables templates and the B-factor head, and adds a pair-only affinity head: a two-module ensemble (8- and 4-block pair-only Pairformer stacks) that scores protein-ligand binding affinity. The released molecular-weight calibration is an optional call-time scoring control, not persistent model configuration.

The affinity path selects a canonical ligand instance explicitly when the
system contains more than one molecule. Raw affinity SMILES must first pass
through `standardize_boltz2_affinity_smiles(...)`. That function applies the
released ChEMBL largest-fragment standardization and returns the canonical
`Molecule` used to construct the system for both model passes. The Processor
derives molecular weight from that same final canonical graph, so one graph
determines token topology, atom identities, and molecular weight. Neither raw
SMILES nor CCD/direct-graph lowering reads chemical provenance.

The audited upstream SMILES parser calls `RemoveHs(..., sanitize=False)` before
`MolWt`, leaving ordinary implicit hydrogens absent from RDKit's property cache;
the CCD path and the paper's stated molecular-weight semantics do not. OFoldX
therefore derives chemically valid molecular weight from the final canonical
graph. The three released correction constants are preserved, but their
holdout-fit artifacts are not public, so the optional corrected value is not
claimed to be bitwise-equivalent to that buggy parser path. The uncorrected
dual-head outputs do not consume molecular weight.

```python
from ofoldx.models.boltz2 import (
    Boltz2AffinityProcessor,
    standardize_boltz2_affinity_smiles,
)

ligand_entity = standardize_boltz2_affinity_smiles(
    "CC(=O)[O-].[Na+]",
    entity_id="ligand",
)
# Build `system`/`structure` with `ligand_entity`; both passes must use that graph.

batch = Boltz2AffinityProcessor(device="cuda")(
    structure,
    ligand="L",  # canonical Molecule instance_id
)
result = affinity_model.score(batch, num_samples=5)
```

The model samples candidate structures, chooses the single greatest-iPTM
sample, and scores only that sample. It reports an IC50-like general
`log10(measurement / µM)` affinity value (lower is stronger) and the mean of
the two heads' binder probabilities (higher is stronger). It also retains both
per-head values and probabilities under the six official Score keys
(`affinity_pred_value`, `affinity_probability_binary`, `affinity_pred_value1`,
`affinity_pred_value2`, `affinity_probability_binary1`,
`affinity_probability_binary2`), each scoped to the selected ligand instance.
The optional molecular-weight correction changes only the ensemble
`affinity_pred_value`. The returned `ScoreResult.target` is the selected
structure actually consumed by the low-level affinity heads.

The concrete `Boltz2AffinityPipeline` runs the full two-stage workflow with two
explicit model/processor components (`structure_model`/`structure_processor` for
the fold+confidence pass, `model`/`processor` for the affinity pass); it is not
hidden inside the Model or the standard `ScorePipeline`. Its `score(...)` folds
the whole system with the confidence checkpoint, ranks the first-stage
predictions by descending whole-call `confidence_score` (interface = ipTM unless
the entire call's ipTM tensor is all-zero, then pTM;
`confidence_score = (4·complex_plddt + interface) / 5` on the 0–1 pLDDT
fraction) into `structure_predictions`, passes the rank-0 canonical `Structure`
through the affinity crop, and runs the affinity checkpoint's independent
second-pass sampling (max-iPTM selection, **no pTM fallback**). The second-pass
coordinates are the returned `target`; first-pass coordinates are never merged
into it.

It returns a flat `Boltz2AffinityResult(target, structure_predictions, scores,
metrics)`. The six official Scores, each scoped to the selected ligand instance,
are: `affinity_pred_value` (two-head ensemble affinity + MW correction when
enabled, `log10(IC50/µM)`, lower is stronger), `affinity_probability_binary`
(mean of the two head binder probabilities, `[0,1]`, higher), the per-head
uncorrected affinities `affinity_pred_value1`/`affinity_pred_value2`
(`log10(IC50/µM)`, lower), and the per-head binder probabilities
`affinity_probability_binary1`/`affinity_probability_binary2` (higher). The
molecular-weight correction applies only to `affinity_pred_value`, never to the
individual heads.

Per DESIGN 6.2 the two passes are two independent upstream `predict_args` dicts,
so `score(...)` splits the sampling counts faithfully: the structure/confidence
pass takes `num_samples`/`num_sampling_steps` (upstream `diffusion_samples`/
`sampling_steps`, defaults 1/200) and the affinity pass takes
`affinity_num_samples`/`affinity_num_sampling_steps` (upstream
`diffusion_samples_affinity`/`sampling_steps_affinity`, defaults 5/200). Per-pass
recycling is faithful without an argument (each Model defaults to its own config:
3 for the structure pass, 5 for the affinity pass). A `seed` seeds one RNG once
and both passes draw from that single continued stream, mirroring upstream's one
`seed_everything(seed)` before the two sequential passes. MSA is routed to both
Processors because upstream reads both passes from the one shared `msa_dir`; the
affinity crop / `method="other"` divergence lives inside
`Boltz2AffinityProcessor`. The Pipeline's `to(...)` placement remains deferred
(DESIGN 6.3).

### Model Specification

| Architecture      | Pairformer Blocks | MSA Blocks | Single Size | Pair Size | Recycling Steps | Diffusion Blocks | Sampler Steps | Confidence Blocks |
| ----------------- | ----------------- | ---------- | ----------- | --------- | --------------- | ---------------- | ------------- | ----------------- |
| `boltz2`          | 64                | 4          | 384         | 128       | 3               | 24               | 200           | 8                 |
| `boltz2_affinity` | 64                | 4          | 384         | 128       | 5               | 24               | 200           | 8                 |

> [!NOTE]
> `boltz2_affinity` (`Boltz2AffinityConfig`) subclasses `Boltz2Config`, turns off templates and B-factor prediction,
> retains the complete folding/confidence path, and adds the checkpoint's fixed pair of affinity heads: an 8-block
> and a 4-block pair-only Pairformer stack. The released trunk is large, so a parameter count is not instantiated here.

### Links

- **Code**: [ofoldx.models.boltz2](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz2)
- **Paper**: [Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
- **Developed by**: Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al.
- **Model type**: All-atom biomolecular structure-prediction trunk with an EDM diffusion sampler, a token-level confidence head, and a pair-only binding-affinity head
- **Original Repository**: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)

## Training Details

Boltz-2 training is staged across structure, confidence, and affinity objectives, learning to reconstruct all-atom biomolecular complexes and to predict protein-ligand binding affinity. This describes the released weights; it is not an OFoldX training contract.

### Training Data

The Boltz-2 report describes structure data from PDB structures before 2023-06-01, experimental NMR collections, MD trajectory datasets, AF2 monomer distillation, Boltz-1 complex distillation, and affinity data from PubChem, ChEMBL, BindingDB, and related filtered assay sources. OFoldX does not redistribute the training set.

The Boltz-2 affinity heads use filtered assay data from sources such as PubChem, ChEMBL, and BindingDB, with decoy generation and structural-confidence filters described by the upstream report. OFoldX does not redistribute the training set.

### Training Procedure

Upstream Boltz-2 training is staged across structure, confidence, and affinity objectives. OFoldX converts a released checkpoint into a strict-loaded native model in memory. Registration still requires the complete fold/confidence boundary and concrete two-pass workflow. Training orchestration is separate from the current Model Hub gate.

The technical report describes detached-trunk absolute/difference Huber losses
and a focal binary objective, but neither the report nor the released source
defines the exact pair construction, censor-mask denominators and reductions,
focal-alpha convention/reduction, or normalization when a batch lacks one
objective. The inspected source contains no affinity labels or affinity loss and
states that updated Boltz-2 training code is forthcoming. Those details are useful
provenance, but reproducing them exactly is not a registration requirement; OFoldX
will define and document its own scientifically useful affinity objective.

The maintained script compares both released affinity heads and their combined
low-level score math with the real checkpoint. Generated reports remain local. Run:

```bash
python -m scripts.parity.boltz2.affinity_parity --device cuda
```

## Citation

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
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

Please contact the authors of the [Boltz-2 paper](https://doi.org/10.1101/2025.06.14.659707) for questions or comments on the paper/model.

## Numerical epsilon note

- **Confidence normalization eps `1e-5`** (`model.py:2803`) is **upstream-faithful — keep, do not unify.** Faithful to Boltz `confidence_utils`, and uses the `+ 1e-5` form consistently (unlike Boltz-1, which mixes in a `clamp_min(1e-5)` at `boltz1/model.py:3760` — audit D5). The per-family confidence-eps spread (boltz 1e-5 / af3 1e-8+1e-6 / protenix 1e-8 / esmfold2 1e-6) is an inherent cross-model difference; a shared constant would regress a family. Measured cost of unifying boltz 1e-5→1e-6 is ~2e-7 on pTM. See `EPSILON_ANALYSIS.md`.

## License

This model implementation is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.html).

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```

## Limitations & intentional upstream divergences

- Boltz2 publishes the structure/confidence pair and the affinity pass as the
  registered `Boltz2AffinityPipeline`, whose root is the affinity pass over a
  `structure/` subpair. Global confidence and affinity scores are the qualified
  surface; the diffusion sampler is stochastic.
- The confidence epsilon is kept upstream-faithful (see the Numerical epsilon note).
- A parity run applies only to the checkpoints and settings supplied to the script.

## Licensing & redistribution

- **OFoldX implementation & docs:** `AGPL-3.0-or-later`.
- **Upstream code & weights:** MIT
  ([boltz LICENSE](https://github.com/jwohlwend/boltz/blob/main/LICENSE)).
- **Redistribution status: GREEN.** The `boltz2_conf` and affinity checkpoints are
  permissively licensed and freely redistributable with the upstream MIT notice
  preserved.
