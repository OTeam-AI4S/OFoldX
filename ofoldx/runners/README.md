# Training with `ofoldx.runners.train`

`ofoldx.runners.train` is the command-line entry point for the native OFoldX
Runner. Run commands from the repository root so local paths and the default
`experiments/` workspace are predictable.

## Inputs

The required positional `train_source` may be one PDB/CIF/mmCIF file, a
directory (searched recursively), a glob, or a text/JSONL manifest. A text
manifest contains one structure path per line. Blank lines and `#` comments are
ignored. A JSONL manifest can attach explicit MSA evidence:

```json
{"structure": "structures/1abc.cif", "msa": {"path": "msa/1abc_A.a3m", "entity_id": "A"}}
{"structure": "structures/2xyz.cif", "msa_collection": "msa/2xyz_pairing.json"}
```

Relative paths in a manifest are relative to that manifest. An
`msa_collection` is JSON produced by `ofoldx.data.encode_msa_collection`.
One manifest record cannot contain both `msa` and `msa_collection`.
Alternatively, use `--seq-to-pdb-index INDEX.json --msa-path MSA_ROOT` together;
explicit evidence in a manifest takes precedence. MSA input requires an
MSA-capable processor (including Protenix); LigandMPNN and ProteinMPNN do not
accept it.

## Invocations

Direct, one-GPU training (run this only inside an interactive/batch Slurm
compute-node allocation, never on a management node):

```bash
python -m ofoldx.runners.train data/train \
  --device cuda --precision bf16 --epochs 10
```

The repository Slurm launcher rejects execution outside an allocation. Its
default header requests one A800, and command arguments after the script (and an
optional `--`) are forwarded unchanged:

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --job-name=protenix-train ofoldx/runners/submit_train.slurm -- \
  data/train --device cuda --precision bf16 --epochs 10
```

Single-node, four-GPU training:

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --gres=gpu:a800:4 --cpus-per-task=32 \
  --export=ALL,NPROC_PER_NODE=4 \
  ofoldx/runners/submit_train.slurm -- \
  data/train --device cuda --precision bf16 --epochs 10
```

Two nodes with eight A800 GPUs per node (16 ranks total):

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:a800:8 \
  --cpus-per-task=64 --mem=0 --time=24:00:00 \
  --export=ALL,NPROC_PER_NODE=8 \
  ofoldx/runners/submit_train.slurm -- \
  data/train --device cuda --precision bf16 --epochs 10 \
  --batch-size 1 --accum-steps 4 --workers 4 \
  --workspace /shared/ofoldx/experiments --run-name protenix-2x8
```

The launcher uses direct Python for one node/one process, standalone `torchrun`
for one node/multiple processes, and `srun` to start exactly one c10d `torchrun`
agent per node for multi-node jobs. Its printed, shell-escaped `Command:` line is
the authoritative launch record. Both the native and squashfs launchers default
`NCCL_SOCKET_FAMILY` to `AF_INET`, preventing NCCL from automatically selecting
an unroutable IPv6 link-local address. The squashfs launcher explicitly passes
the value through Pyxis to the container; the native launcher's `srun` inherits
the exported value. Override the default only when the allocated nodes have
routable IPv6 connectivity:

```bash
sbatch --nodes=2 --export=ALL,NPROC_PER_NODE=8,NCCL_SOCKET_FAMILY=AF_INET6 \
  ofoldx/runners/submit_train.slurm -- \
  data/train --device cuda --epochs 10
```

Only `AF_INET` and `AF_INET6` are accepted. `NCCL_SOCKET_FAMILY` selects the IP
address family; it does not select a network interface. `NCCL_SOCKET_IFNAME`
instead filters NCCL's interface selection and should not be set unless the
cluster administrator has identified the correct interface. An IPv6-only
cluster must explicitly use `NCCL_SOCKET_FAMILY=AF_INET6` (and have routable
IPv6 addresses rather than link-local-only addresses).

For multi-node jobs, the `sbatch` options allocate one task slot per node. The
script then explicitly runs `srun --nodes NNODES --ntasks NNODES
--ntasks-per-node=1`, so there is exactly one torchrun agent on each allocated
node; each agent creates `NPROC_PER_NODE` local worker ranks.

## Command-line reference

Numeric arguments described as positive must be greater than zero; non-negative
arguments may be zero. `--epochs` and `--steps` are mutually exclusive.

### Data and evaluation

| Argument | Default and behavior |
| --- | --- |
| `train_source` | Required file, directory, glob, or manifest. |
| `--val-source PATH` | Adds the evaluation split named `val`. |
| `--eval-source NAME=PATH` | Repeatable named evaluation split. Names must match `[A-Za-z][A-Za-z0-9_-]*`, be unique, and cannot be `train`; `val` conflicts with `--val-source`. |
| `--score-split NAME` | Split whose loss selects the best checkpoint. It must name a configured evaluation split. Default: `val` if present, otherwise the first named split. |
| `--seq-to-pdb-index PATH` | Sequence-to-MSA-directory JSON index; requires `--msa-path`. Applied to training and every evaluation split. |
| `--msa-path DIR` | Root of indexed MSA directories; requires `--seq-to-pdb-index`. |

All configured evaluation splits run separately. Scoring uses only
`--score-split`; the others are still evaluated and logged.

### Model and CCD

| Argument | Default and behavior |
| --- | --- |
| `--model ID` | Registered architecture. A fresh run defaults to `protenix`. |
| `--pretrained PATH_OR_REPO` | Load a local or Hugging Face model repository. |
| `--revision REVISION` | Branch, tag, or commit for `--pretrained`; invalid without it. |
| `--model-config JSON_OR_PATH` | Inline JSON object or path to a JSON object; default `{}`. Cannot set reserved `name`, `pretrained`, or `revision`; a nonempty value cannot accompany `--pretrained`. |
| `--processor-config JSON_OR_PATH` | Inline JSON object or path; default `{}`. A nonempty value cannot accompany `--pretrained`. |
| `--ccd-revision REVISION` | Nonempty CCD repository revision for a fresh processor. Mutually exclusive with `--ccd-snapshot-path`, `--pretrained`, and CCD locator keys in processor config. |
| `--ccd-snapshot-path PATH` | Nonempty local CCD snapshot for a fresh processor; the same exclusions apply. |

If `--model` and `--pretrained` are both given, the explicit model ID must match
the architecture recorded by the repository. In `--processor-config`,
`ccd_revision` and `ccd_snapshot_path` must each be either a nonempty string or
`null`, and at most one may be non-null. If either key is present in processor
config, neither explicit CCD locator flag may be used.

For a freshly initialized Protenix model, the native Runner currently optimizes
the built-in EDM coordinate-denoising objective. It does **not** reproduce the
complete upstream Protenix multi-stage total loss.

### Supported training architectures

The registry contains additional architectures for construction, loading, and
inference, but this Runner trains only the exact architecture IDs below. Any
other registered ID fails with “has no built-in Runner objective.”

| Architecture ID(s) | Runner objective | Batching |
| --- | --- | --- |
| `proteinmpnn` | Uniform label-smoothed protein sequence recovery over designable residues. | Ragged multi-structure batches supported. |
| `ligandmpnn` | Protein sequence recovery with ligand context; the same uniform label-smoothed loss as ProteinMPNN. | `--batch-size 1` required. |
| `nampnn` | Polymer-specific label-smoothed protein/DNA/RNA sequence recovery. | Ragged multi-structure batches supported. |
| `alphafold3`, `boltz1`, `boltz2`, `boltzgen`, `protenix`, `protenix_template`, `protenix_structure_condition`, `protenix_constraint` | Shared EDM coordinate-denoising loss over resolved atoms. | `--batch-size 1` required. |
| `alphafold2`, `alphafold2_initial_guess`, `alphafold2_multimer` | Masked-MSA reconstruction cross entropy. | `--batch-size 1` required. |
| `grnade` | Released RNA sequence cross entropy over designable canonical-RNA residues. | `--batch-size 1` required. |
| `oinvfold` | Released polymer negative log likelihood over designable nodes. | `--batch-size 1` required. |
| `odesign` | Molecule-weighted, aligned EDM coordinate denoising; fixed conditioning atoms are not scored. | `--batch-size 1` required. |

The batch-size-one families use the Runner's `collate_single` path, which
rejects larger per-rank batches. Use `--accum-steps` (and multiple DDP ranks) to
increase effective batch size without changing `--batch-size 1`.

### Duration, optimization, and precision

| Argument | Default and behavior |
| --- | --- |
| `--epochs N` | Positive epoch count; default `1` when neither duration flag is given. |
| `--steps N` | Positive total number of optimizer updates. On resume this remains the target total, not an additional number of updates. |
| `--lr FLOAT` | Positive AdamW learning rate; default `1e-4`. |
| `--weight-decay FLOAT` | Non-negative; default `0.01`. |
| `--batch-size N` | Positive samples per rank; default `1`. |
| `--accum-steps N` | Positive gradient accumulation factor; default `1`. |
| `--workers N` | Non-negative data-loader workers per rank; default `0`. |
| `--precision {bf16,fp16}` | Optional reduced precision; unset by default. |
| `--max-grad-norm FLOAT` | Optional positive gradient-clipping norm. |
| `--seed N` | Non-negative random seed; default `1016`. |
| `--deterministic`, `--no-deterministic` | Deterministic mode; default off. |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | Activation recomputation; default off. This is unrelated to saving training-state checkpoints. |

### Protenix training crop

| Argument | Default and behavior |
| --- | --- |
| `--max-tokens N`, `--crop-size N` | Positive maximum training tokens per sample. Only Protenix supports crop options. |
| `--crop-method-weights CONTINUOUS SPATIAL INTERFACE_SPATIAL` | Three non-negative weights, not all zero; requires `--max-tokens`. Default is `1 0 0`, or `0.2 0.4 0.4` with cluster sampling. Positive spatial weights require `--cluster-file`. |

Cropping applies only to training. Evaluation uses full structures and can run
out of memory even when cropped training fits; reduce evaluation inputs or run
evaluation with sufficient memory.

### Learning-rate scheduler

| Argument | Default and behavior |
| --- | --- |
| `--scheduler {none,cosine,linear}` | Default `none`. |
| `--warmup OPTIMIZER_STEPS` | Non-negative linear warmup length; default `0`. |
| `--final-lr-ratio FLOAT` | Non-negative final/initial learning-rate ratio; default `0.0`. |

With `--scheduler none`, the scheduler configuration is omitted entirely;
`--warmup` and `--final-lr-ratio` therefore have no effect unless `cosine` or
`linear` is selected.

### Runtime, workspace, checkpoints, and logs

| Argument | Default and behavior |
| --- | --- |
| `--device {auto,cpu,cuda}` | Default `auto`. `cuda` fails if CUDA is unavailable; `cpu` hides CUDA. |
| `--workspace PATH` | Workspace root; default `experiments`. |
| `--run-name NAME` | Run/experiment name; default model ID, or `pretrained` when appropriate. |
| `--resume` | Resume the latest checkpoint for this run. Mutually exclusive with `--resume-from`; loading remains available with `--no-checkpoint`. |
| `--resume-from CHECKPOINT` | Resume an explicit checkpoint; loading remains available with `--no-checkpoint`. |
| `--checkpoint`, `--no-checkpoint` | Save training-state checkpoints; default on. |
| `--checkpoint-interval N` | Optional positive save interval. Units are optimizer updates in step mode and epochs in epoch mode. Default: `ceil(steps / 20)` (at least 1) in step mode, or 1 epoch in epoch mode. |
| `--keep-latest N` | Number of framework-generated history checkpoints to retain; non-negative, default `0`. Zero disables history cleanup—it does **not** mean retain zero checkpoints. |
| `--logging`, `--no-logging` | Text/file logging; default on. This switch is independent of TensorBoard and W&B. |
| `--log-interval N` | Optional positive shared metric interval. In step mode it counts global optimizer updates and defaults to `ceil(steps / 100)` (at least 1). In epoch mode it counts training steps (loop iterations/batches) and defaults to 1,024. |
| `--tensorboard`, `--no-tensorboard` | TensorBoard metric logging; default off. |
| `--tensorboard-log-dir PATH` | Nonempty event-file directory; requires `--tensorboard`. By default Danling uses `<resolved-workspace>/tensorboard/<timestamp>`. |
| `--tensorboard-flush-secs N` | Optional positive event flush interval in seconds; requires `--tensorboard`. TensorBoard's writer default applies when omitted. |
| `--wandb`, `--no-wandb` | Weights & Biases metric logging; default off. |
| `--wandb-mode {online,offline,disabled}` | W&B operating mode; requires `--wandb`. Omitted means the W&B/environment default. Use `offline` when compute nodes have no network access. |
| `--wandb-project PROJECT` | Nonempty W&B project; requires `--wandb`. Default: workspace lineage `ofoldx`. |
| `--wandb-entity ENTITY` | Optional nonempty W&B entity/team; requires `--wandb`. |
| `--wandb-id ID` | Optional nonempty stable W&B run ID; requires `--wandb`. |
| `--wandb-name NAME` | Optional nonempty W&B display name; requires `--wandb`. Default: Danling runner ID. |
| `--wandb-group GROUP` | Optional nonempty W&B group; requires `--wandb`. Default: the resolved run/experiment name. |
| `--wandb-resume {allow,must,never,auto}` | W&B resume policy; requires `--wandb`. |
| `--wandb-tag TAG` | Repeatable nonempty W&B tag; requires `--wandb`. Tag order is preserved. |
| `--print-config` | Print resolved JSON without constructing a model, Runner, or dataset or initializing TensorBoard/W&B. Optional logger packages are not required. Useful for launcher smokes. |

The three logging backends are freely combinable. For example,
`--no-logging --tensorboard --wandb --wandb-mode offline` disables text/file
logging while retaining both metric backends; `--log-interval` controls their
shared metric cadence. Under DDP, only rank 0 initializes TensorBoard and W&B.

TensorBoard and W&B are optional dependencies and are imported only when their
backend is initialized. Install them in the Python environment visible on each
compute node (for example, `python -m pip install tensorboard wandb`) before a
real enabled run. `--print-config --tensorboard --wandb` needs neither optional
logger package.
For online W&B, provide the secret as `WANDB_API_KEY` in the submitted job's
environment; an API key is deliberately never accepted as a CLI argument.
Offline mode writes locally and can later be synchronized with W&B tooling.

Danling stores run output, logs, W&B's default local directory, and state below
the configured workspace using lineage `ofoldx` and the run name. Before the
first submission, create `/data/users/qinghanw313/AlloDesign/logs`; Slurm opens
its output files before the script body runs. The launcher writes stdout and
stderr to `logs/<job-name>-<jobid>.out` and `.err`: `%x` expands to the Slurm job
name and `%j` to the job ID. Set a descriptive name with `sbatch --job-name=NAME`.
Use a shared workspace for multi-node jobs.

### Cluster-weighted CIF sampling

All flags in this section other than `--cluster-file` require
`--cluster-file`. Cluster sampling is training-only and accepts asymmetric-unit
CIF/mmCIF records (not PDB or assembly-selected manifest records). It eagerly
indexes chain/interface sampling rows, then maps a sampled row back to the full
CIF model input.

| Argument | Default and behavior |
| --- | --- |
| `--cluster-file PATH` | RCSB entity-cluster file; enables cluster sampling. |
| `--cluster-indices-file PATH` | Optional versioned OFoldX JSONL cache of final weighted indices; otherwise the cache is under the workspace. |
| `--cluster-epoch-size N` | Positive **global** samples drawn per epoch. If omitted, defaults to the number of weighted index rows. |
| `--cluster-seed N` | Optional non-negative sampling seed. If omitted, inherits global `--seed`; sampling epoch `E` uses seed `+ E`. |
| `--cluster-replacement`, `--no-cluster-replacement` | Sample with replacement; default on. |
| `--cluster-contact-cutoff ANGSTROMS`, `--cluster-cutoff ANGSTROMS` | Positive contact cutoff; default `5.0`. |
| `--cluster-all-chain-pairs`, `--cluster-all-pairs`, and their `--no-...` forms | Include all chain pairs; default off. |
| `--cluster-exclude-mol-types [TYPE ...]` | Molecular types excluded from sampling rows; default `ions`. Passing the flag with no values gives an empty list. |
| `--cluster-beta-chain FLOAT` | Non-negative chain weight exponent; default `0.5`. |
| `--cluster-beta-interface FLOAT` | Non-negative interface weight exponent; default `1.0`. |
| `--cluster-alpha-prot FLOAT` | Non-negative protein type weight; default `3.0`. |
| `--cluster-alpha-nuc FLOAT` | Non-negative nucleic-acid type weight; default `3.0`. |
| `--cluster-alpha-ligand FLOAT` | Non-negative ligand type weight; default `1.0`. |

Distributed sampling pads the global cluster epoch size so every rank receives
the same number of rows; the processed total can therefore be slightly larger
than the requested value. With `--no-cluster-replacement`, the weighted index
must contain at least that DDP-padded total number of positive-weight rows.

## Launcher environment

| Variable | Default and purpose |
| --- | --- |
| `REPO_ROOT` | `/data/users/qinghanw313/AlloDesign`; working tree added to `PYTHONPATH`. |
| `PYTHON` | `/data/users/qinghanw313/miniforge3/envs/allodesign-cu132/bin/python`; must be executable and visible on every node. |
| `NPROC_PER_NODE` | `1`; positive number of local training ranks/GPUs. |
| `OMP_NUM_THREADS` | If unset, `max(1, SLURM_CPUS_PER_TASK / NPROC_PER_NODE)`; an explicit positive integer is preserved. |
| `MASTER_ADDR` | Multi-node only: first hostname in `SLURM_JOB_NODELIST`; may be overridden. |
| `MASTER_PORT` | Multi-node only: deterministic legal port derived from `SLURM_JOB_ID`; may be overridden with `1`–`65535`. |
| `RDZV_ID` | Multi-node only: `SLURM_JOB_ID`; may be overridden with a nonempty ID. |
| `HF_HUB_OFFLINE` | `1`; set `0` only when compute nodes may contact Hugging Face. |
| `MPLCONFIGDIR` | `${REPO_ROOT}/tmp/mplconfig`; set to another writable shared/local directory if needed. |

With Slurm, `world_size = SLURM_NNODES * NPROC_PER_NODE`. Both `--batch-size`
and `--workers` are per rank, so the effective batch per optimizer update is
`batch_size * accum_steps * world_size`, and total loader workers are normally
`workers * world_size`. Allocate CPUs and memory accordingly. Every node must
see the same repository, Python environment, data, CCD/MSA files, pretrained
model, workspace, and resume checkpoint paths.

## Troubleshooting

- A rendezvous timeout usually means nodes disagree on `MASTER_ADDR`,
  `MASTER_PORT`, `RDZV_ID`, node count, or environment. Start with the printed
  command and the Slurm `.out`/`.err` files; avoid reusing an overridden port for
  concurrent jobs.
- An immediate launcher error usually identifies an invalid/missing environment
  value or executable. Confirm `SLURM_NNODES`, `NPROC_PER_NODE`, and `PYTHON`.
- NCCL connection failures are cluster/network issues. Check that the derived
  master hostname resolves on every node and ask the administrator which network
  interface and NCCL settings are required.
- CUDA OOM during validation can occur because crop and cluster selection affect
  training only; evaluation processes full structures.
- Resume failures commonly come from a changed `--workspace`/`--run-name` or a
  checkpoint path that is not shared across nodes. `--steps` is the final global
  optimizer-update target on resume.
