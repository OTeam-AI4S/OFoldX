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

Boltz-2 is OFoldX's all-atom biomolecular structure-prediction and
protein-ligand binding-affinity family. It provides a folding pair and a
separate two-stage affinity pipeline built from released structure/confidence
and affinity checkpoints.

## Disclaimer

This is an unofficial OFoldX conversion of
[Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction](https://doi.org/10.1101/2025.06.14.659707)
by Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al. The upstream authors did
not write, review, or endorse this conversion or model card.

The official Boltz-2 repository is
[jwohlwend/boltz](https://github.com/jwohlwend/boltz). OFoldX converts released
weights into its native parameter layout; it does not train, relicense, or claim
ownership of the upstream models.

## Model Details

`Boltz2Model` embeds atom-, residue-, MSA-, method-, modification-, cyclic-, and
molecule-level features; applies a recycled v2 template, MSA, and Pairformer
trunk; samples atom coordinates with EDM reverse diffusion and in-loop weighted
rigid alignment; and predicts distogram, B-factor, confidence, and ranking
outputs.

`Boltz2AffinityModel` retains the folding and confidence graph, disables
templates and B-factor prediction, and adds two pair-only affinity heads. Its
low-level `score` operation samples candidate structures, chooses the single
greatest-iPTM sample without a pTM fallback, and returns a `ScoreResult` for that
second-pass target. `Boltz2AffinityPipeline` owns the released two-stage
operation: first fold and rank with the structure/confidence checkpoint, then
crop, resample, and score with the affinity checkpoint. It returns
`Boltz2AffinityResult`, preserving confidence-sorted first-pass predictions and
the selected second-pass target.

### Variants

| Architecture             | Released checkpoint | Model / Processor                                 | Task exposure                      | Concrete Pipeline        |
| ------------------------ | ------------------- | ------------------------------------------------- | ---------------------------------- | ------------------------ |
| `ofoldx.boltz2`          | `boltz2_conf.ckpt`  | `Boltz2Model` / `Boltz2Processor`                 | standard `fold`                    | `Boltz2Pipeline`         |
| `ofoldx.boltz2_affinity` | `boltz2_aff.ckpt`   | `Boltz2AffinityModel` / `Boltz2AffinityProcessor` | concrete `score`; no standard task | `Boltz2AffinityPipeline` |

The affinity repository is composite: the affinity pair owns the root and the
structure/confidence pair lives under `structure/`. The affinity checkpoint is a
Boltz-2 checkpoint even when a source distribution stores it beside BoltzGen
assets; storage location does not change model provenance.

### Model Specification

| Property                          | `ofoldx.boltz2`       | `ofoldx.boltz2_affinity` |
| --------------------------------- | --------------------- | ------------------------ |
| Single / pair size                | 384 / 128             | 384 / 128                |
| Trunk Pairformer blocks           | 64                    | 64                       |
| MSA blocks                        | 4                     | 4                        |
| Template blocks                   | 2                     | 0 (disabled)             |
| Denoiser token Transformer blocks | 24                    | 24                       |
| Denoiser conditioning Transitions | 2                     | 2                        |
| Confidence Pairformer blocks      | 8                     | 8                        |
| Affinity Pairformer blocks        | none                  | 8 and 4                  |
| Total trunk passes                | 4 (`recycling_steps`) | 6 (`recycling_steps`)    |
| Reverse-diffusion steps           | 200                   | 200                      |
| Default samples                   | 1 fold sample         | 5 affinity-pass samples  |
| B-factor head                     | enabled               | disabled                 |

The names above follow the native OFoldX graph: repeated units are Blocks,
stored under `blocks` and counted by fields ending in `num_blocks`; channel MLP
units are Transitions. Source checkpoint spellings are confined to conversion.

Both graphs use `sigma_data=16`, a Karras schedule with `rho=7`, noise levels
from `160` to `1e-4` in `sigma_data` units, `churn_rate=0.8`,
`noise_scale=1.003`, and `step_scale=1.5`. The released inference default is 200
Euler steps.

For folding, each `FoldPrediction` contains a `Structure`, token pLDDT on the
0-100 scale, resolved probability, PAE/PDE in angstroms, and the scores `plddt`,
`ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, and `confidence_score`. First-pass
ranking uses one whole-call fallback decision:

```text
interface = ptm if every sampled iptm is zero else iptm
confidence_score = (4 * complex_plddt + interface) / 5
```

Here `complex_plddt` is the internal 0-1 fraction. The affinity Pipeline sorts
first-pass predictions by this score, passes the rank-0 `Structure` through the
affinity crop, and runs an independent second-pass sample selection by maximum
iPTM.

The two affinity heads publish six ligand-scoped scores:
`affinity_pred_value`, `affinity_probability_binary`,
`affinity_pred_value1`, `affinity_pred_value2`,
`affinity_probability_binary1`, and `affinity_probability_binary2`. Affinity
values use `log10(IC50/µM)` and prefer lower values; binder probabilities are in
`[0, 1]` and prefer higher values. The ensemble values are the means of the two
heads. Optional molecular-weight calibration changes only the ensemble affinity
value `y`:

```text
y_corrected = 1.03525938 * y - 0.59992683 * MW^0.3 + 2.83288489
```

`Boltz2AffinityProcessor` selects one canonical `Molecule` instance and derives
its molecular weight from that same final graph. Raw affinity SMILES should first
pass through `standardize_boltz2_affinity_smiles(smiles)`, which applies the
released ChEMBL standardization and largest-fragment selection. The returned
`Molecule` must be the ligand graph used to build the system consumed by both
passes.

### Links

- [OFoldX implementation](https://github.com/OTeam-AI4S/OFoldX/tree/main/ofoldx/models/boltz2)
- [Official source](https://github.com/jwohlwend/boltz)
- [Boltz-2 paper](https://doi.org/10.1101/2025.06.14.659707)
- [Upstream model repository](https://huggingface.co/boltz-community/boltz-2)
- Developed by Saro Passaro, Gabriele Corso, Jeremy Wohlwend, et al.

## Usage

```python
from ofoldx import AutoModel, AutoProcessor, fold, pipeline
from ofoldx.models.boltz2 import standardize_boltz2_affinity_smiles

fold_repository = "./boltz2"
affinity_repository = "./boltz2-affinity"

fold_result = fold(
    "complex.cif",
    model=fold_repository,
    device="cuda",
    num_samples=5,
    seed=7,
)

ligand_entity = standardize_boltz2_affinity_smiles("CC(=O)[O-].[Na+]")
system = ...  # Build one MolecularSystem that uses ligand_entity as instance "L".

scorer = pipeline("score", model=affinity_repository, device="cuda")
affinity_result = scorer.score(
    system,
    ligand="L",
    num_samples=1,
    affinity_num_samples=5,
    molecular_weight_correction=True,
    seed=7,
)

model = AutoModel(fold_repository, device="cuda")
processor = AutoProcessor(fold_repository, device="cuda")
direct_fold = model.fold(processor("complex.cif"))
```

The two Pipeline passes share one advancing RNG stream and receive the same MSA
evidence. `num_samples` and `num_sampling_steps` control the first pass;
`affinity_num_samples` and `affinity_num_sampling_steps` control the second.
Direct `Boltz2AffinityModel.score` is only the low-level second pass and does not
replace `Boltz2AffinityPipeline`.

## Checkpoints and Conversion

Convert the standalone folding checkpoint:

```bash
python -m ofoldx.models.boltz2.convert fold \
  boltz2_conf.ckpt \
  ./boltz2
```

Convert the affinity and structure checkpoints into one composite repository:

```bash
python -m ofoldx.models.boltz2.convert affinity \
  boltz2_aff.ckpt \
  boltz2_conf.ckpt \
  ./boltz2-affinity
```

Both converters translate released source namespaces to native Block and
Transition names, pack adaptive-normalization projections, and strict-load the
result. The standalone repository contains one Model/Processor pair. The
composite affinity repository contains the affinity pair at its root, the
structure pair under `structure/`, and a pipeline marker selecting
`Boltz2AffinityPipeline`. `--dtype` may cast floating tensors, and `--repo-id`
optionally uploads the converted repository.

## Training Details

### Training Data

The Boltz-2 report describes structure data from PDB entries before 2023-06-01,
experimental NMR collections, molecular-dynamics trajectories, AlphaFold 2
monomer distillation, and Boltz-1 complex distillation. It describes affinity
data from PubChem, ChEMBL, BindingDB, and related filtered assay sources, with
decoy generation and structural-confidence filtering. OFoldX does not
redistribute these datasets.

### Training Procedure

Upstream training is staged across structure, confidence, and affinity
objectives. The report describes detached-trunk absolute and difference Huber
losses for affinity values and a focal binary objective for binder
classification.

The inspected report and released source do not specify enough detail to
reproduce every affinity pair construction, censor-mask denominator and
reduction, focal-alpha convention, or empty-objective normalization. The
inspected source also does not contain the complete affinity label/loss training
path. OFoldX conversion does not retrain the checkpoints or claim a reproduction
of the upstream training loop.

## Limitations

- Structure and affinity sampling are stochastic. The two-stage result depends
  on both first-pass ranking and independent second-pass maximum-iPTM selection.
- Affinity scoring requires a protein receptor and one selected, non-copied
  `Molecule` instance. The released crop supports at most 256 tokens, 200 polymer
  tokens, 2048 atoms, and 128 ligand atoms after hydrogen removal.
- The audited upstream raw-SMILES path removes hydrogens without sanitizing
  before molecular-weight calculation, which can leave ordinary implicit
  hydrogens absent from RDKit's property cache. OFoldX computes molecular weight
  from the chemically valid final canonical graph. The three correction
  constants are preserved, but corrected values are not claimed bitwise equal to
  that parser behavior; uncorrected head outputs do not use molecular weight.
- Native pTM reductions use the dtype-aware `torch.finfo(dtype).tiny` denominator
  for empty masks rather than a fixed cross-family epsilon.
- Recorded released-checkpoint comparisons cover forward, confidence, diffusion,
  sampler, and affinity calculations (5/5), with observed maximum differences
  between 0 and `7.6e-5`; the corrected folding Processor features compared
  exactly. This evidence is scoped to the compared checkpoints and inputs and
  does not establish parity for every crop, ligand chemistry, Pipeline setting,
  or stochastic sample.

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

If OFoldX supports your research, also cite the OFoldX software:

```bibtex
@software{chen2026ofoldx,
  author = {Chen, Zhiyuan},
  title  = {OFoldX},
  year   = {2026},
  url    = {https://github.com/OTeam-AI4S/OFoldX},
}
```

## Contact

Please use [OFoldX GitHub issues](https://github.com/OTeam-AI4S/OFoldX/issues)
for questions about this implementation or model card. Questions about the
upstream model should be directed to the authors of the
[Boltz-2 paper](https://doi.org/10.1101/2025.06.14.659707).

## License

- **OFoldX implementation and documentation:** AGPL-3.0-or-later.
- **Upstream source and checkpoints:** MIT under the
  [Boltz license](https://github.com/jwohlwend/boltz/blob/main/LICENSE).
- `boltz2_conf.ckpt` and `boltz2_aff.ckpt` may be redistributed with the
  upstream MIT notice preserved. Converted weights are not relicensed by OFoldX.

```spdx
SPDX-License-Identifier: AGPL-3.0-or-later
```
