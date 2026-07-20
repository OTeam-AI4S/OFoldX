# Model-output parity

Local parity tooling for comparing OFoldX against upstream/reference implementations.
Real checkpoints and reference checkouts are release evidence, not frequent-CI
fixtures.

## Comparison contract

A direct parity claim has one criterion:

```text
same public input -> same public output
```

A maintained parity claim names all of the evidence needed to reproduce it:

- the upstream code revision observed for that run and the exact checkpoint release and
  file; these are distinct facts, because a later code revision may execute an
  earlier checkpoint without changing that checkpoint's release identity;
- the exact public biological input, public output fields, execution controls, and
  deterministic or stochastic boundary;
- the allowed coordinate alignment and chemically equivalent atom permutations,
  plus the named semantic axes for logits, confidence, affinity, and scores; and
- the numerical tolerances and any scientific acceptance criteria. A quality metric
  can complement an output comparison, but cannot replace it.

Coverage may be split across maintained scripts, but each documented parity claim
must name its exact public input and public output surface. These scripts provide
manual scientific evidence; they are never frequent-CI, registration, or publication
gates.

For stochastic models, an integer seed promises reproducibility within one
implementation; it does not make different random-number algorithms produce the same
stream. Cross-project coordinate equality therefore requires a shared generator
algorithm, injected identical stochastic draws, a captured deterministic boundary,
or stochastic terms disabled. Reproducibility of OFoldX's public
`torch.Generator` path is required separately and is not cross-project sampler
parity.

Strict checkpoint loading, converter key coverage, module, trunk, denoiser, and
sampler-step comparisons, and quality-only evaluations are diagnostics. They help
locate an output mismatch, but they are not parity signoff and must not be reported
as model parity.

Reference checkouts live under `~/reference`. Checkpoints are read from
`OFOLDX_WEIGHTS_ROOT`, which defaults to `~/weights`. Both locations remain
caller-owned; parity scripts never copy checkpoints or generated results into the
repository. Real-weight parity never runs in frequent CI.

When a script accepts `--json` or `--output`, write it under `artifacts/parity/`
(already ignored) or another local scratch directory. Do not add those outputs to
the source tree.

## Protenix evidence boundary

Protenix checkpoint-release identity and inspected parity-code identity are recorded separately:

- `protenix_base_default_v0.5.0.pt` belongs to the v0.5.0 release at
  `26816e9fcb2cd1ff16dc9eaa8feb913f853b9528`;
- `protenix_base_constraint_v0.5.0.pt` belongs to the v0.5.3 constraint release at
  `a0f95f62430093b46c677f04e3d292ccc5d28d64`; and
- `protenix.py` records that v0.5.3 commit as the inspected parity-code revision for both
  graphs. Using the later checkout for the default graph does not relabel the default
  checkpoint as a v0.5.3 artifact.

The current evidence is deliberately separated. `protenix.py forward` compares the
released graph at a caller-supplied noisy-coordinate/noise-level boundary.
`protenix.py released` and `repository` exercise OFoldX-local sampling,
`torch.Generator` reproducibility, `FoldResult`, and public repository paths.
`ref_forward.py` compares these final tensor heads under its declared deterministic
settings:

```text
same synthetic reference features
same checkpoint
same deterministic settings
  -> reference Protenix.forward(...)
  -> OFoldX public model/pipeline API
compare final outputs
```

These are separate claims. None is a cross-project stochastic public-sampler signoff.
That signoff requires the reference and OFoldX public sampling paths to share an
explicit stochastic boundary and to compare their final public outputs. Private
intermediate tensors, hooks, and module-level `f_forward` outputs remain localization
diagnostics.

## Run

```bash
cd /path/to/ofoldx
export OFOLDX_WEIGHTS_ROOT="${OFOLDX_WEIGHTS_ROOT:-$HOME/weights}"
srun -p gpu -N1 -n1 -w node4 --gres=gpu:1 -c8 --mem=64G \
  bash -lc 'source /data/miniforge3/etc/profile.d/conda.sh; conda activate zyc;
            python -m scripts.parity.ref_forward \
              --checkpoint "$OFOLDX_WEIGHTS_ROOT/protenix_base_default_v0.5.0.pt" \
              --device cuda:0 --n-cycle 4 --n-sample 2 --n-step 1'
```

To submit the configured end-to-end public-output jobs through Slurm:

```bash
bash scripts/parity/submit_e2e.sh
```

To submit the complete maintained matrix, including diagnostic module/trunk/localization
jobs:

```bash
bash scripts/parity/submit_all.sh
```

The script prints a local JSON report with one row per comparable final output.
The report is run output, not a repository artifact:

- `coordinate`: Protenix `pred_dict["coordinate"]` vs OFoldX `ModelOutput.pred_coords`
- `plddt`: Protenix `pred_dict["plddt"]` vs OFoldX `ModelOutput.plddt_logits`
- `pae`: Protenix `pred_dict["pae"]` vs OFoldX `ModelOutput.pae_logits`
- `pde`: Protenix `pred_dict["pde"]` vs OFoldX `ModelOutput.pde_logits`
- `resolved`: Protenix `pred_dict["resolved"]` vs OFoldX `ModelOutput.resolved_logits`

## Supporting scripts

- `boltz2/` and `boltzgen/`: family-owned checkpoint and numerical parity
  harnesses using the current public family modules.
- `esmfold2.py --semantic-only` checks mixed polymer/ligand token identity,
  covalent bonds and reference coordinates, canonical and modified nucleotide
  tokenization, paired/unpaired MSA deletion features, Fast MSA rejection, and
  token-granularity Result persistence without executing a model, converter, or
  checkpoint load. The ordinary mode compares production ESMFold2 heads and stays
  outside frequent CI. It requires explicit local `--head`, `--upstream-esmc`,
  `--ofoldx-esmc`, and `--ccd` paths, validates them before importing model stacks,
  and never resolves or downloads a Hub asset. `--semantic-only` is self-contained
  and needs none of those assets.
- `nampnn_released.py` is investigation tooling for both released NA-MPNN
  checkpoints. Its intended matrix includes raw probability paths, sampling,
  specificity, independent-order scoring, and core objectives, but the NaMpnn
  contract is currently reopened and the script is not complete parity evidence
  until it is rewritten against that frozen public API and rerun.
- `proteinmpnn.py` checks conversion and explicit-rank forward computation for the
  twelve vanilla, soluble, and CA ProteinMPNN checkpoints, plus part of the earlier
  full/CA operation surface. It does not yet cover the two membrane checkpoints,
  original 21-class sampling support, the updated four probability operations, or
  the two distinct PDB lowerings. Those omissions are why the expanded family
  contract and parity remain open.
- `mpnn_packer.py` exercises the released Packer checkpoint, Processor, raw Model
  operations, canonical `PackResult`, PDB writer, and local Auto/concrete-Pipeline
  repository round-trip. It is the maintained standalone Packer evidence route;
  generated logs and repositories stay under ignored scratch paths.
  Corrected tied-group semantics in the sequence-design scripts are native contract
  evidence, not claimed as direct parity with defective upstream paths.
  `pxdesign/sampler_parity.py` compares the released PXDesign EDM
  schedule and every sampler step with identical injected stochastic draws while the
  real checkpoint denoiser is checked separately. `odesign/forward_parity.py` keeps
  every released trunk, atom-encoder,
  diffusion-transformer, and atom-decoder block and compares both pairwise heads,
  the raw coordinate update, and the EDM-combined coordinates. `rfdiffusion_design.py`
  replays captured initial coordinates and raw self-conditioning with reverse noise
  disabled across the released checkpoint/mode matrix, and separately checks public
  Processor + `Model.design(generator=...)` reproducibility. Its self-contained
  `--compile-only` mode runs every denoiser output leaf through default Inductor with
  `fullgraph=True`. `rfdiffusion_allatom.py`,
  `rfdiffusion_network.py`, `rfdiffusion_se3.py`, and
  `rfdiffusion_symmetry.py`: maintained
  family/reference diagnostics. The RFdiffusion network script captures all
  released denoiser heads in the official environment, then compares them in the
  OFoldX environment; the atom14, SE(3), and symmetry scripts cover their named
  components. The symmetry diagnostic compares OFoldX's deterministic T/O/I matrix
  order directly with the inspected upstream `sym_rots.npz` asset.
  Those component diagnostics are not public sampler signoff. The design harness
  covers deterministic `pX0`, zero-noise `x_{t-1}`, raw self-conditioning,
  C/D/T/O/I ordering, sequence/index projection, and a reproducible canonical
  `DesignResult`. Both sides run with TF32 disabled so different PyTorch defaults
  do not masquerade as algorithmic differences. Production checkpoints remain
  caller-supplied local files.
- `analyze_trace.py`: streams large profiler traces without loading them fully
  into memory.

External parity checkouts are not adapted through import aliases or runtime
monkey-patching. An external harness must import OFoldX's current public API
itself before it can be added to the maintained Slurm matrix.
