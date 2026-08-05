# Training with `ofoldx.runners.train`

`ofoldx.runners.train` is the command-line entry point for the native OFoldX
Runner. Run commands from the repository root so local paths and the default
`experiments/` workspace are predictable.

## Recommended YAML configuration

Keep the complete training configuration in one YAML file and pass only that
file to the training entry point:

```bash
python -m ofoldx.runners.train --config configs/train.yaml
python -m ofoldx.runners.train -c configs/train.yaml --print-config
```

[`configs/train.yaml`](../../configs/train.yaml) is the repository AFDB training
example. The YAML document must be one top-level mapping. Its keys are the
canonical snake-case argparse destination names: for example, the CLI flags
`--train-epoch-size`, `--batch-size`, and `--no-checkpoint` correspond to
`train_epoch_size`, `batch_size`, and `checkpoint: false`. Repeatable or
multi-value arguments use YAML sequences, and booleans and numbers use native
YAML values. In particular, `train_datasets`, `model_config`, and
`processor_config` are native nested mappings, not inline JSON or paths to JSON
files:

```yaml
train_datasets:
  afdb_swissprot:
    source: /data/projects/allodesign/datasets/afdb_swissprot/mmcif
    sampling_ratio: 1.0
model: protenix
model_config: {}
processor_config:
  max_msa_rows: 2048
objective_recipe: protenix_baseline_v1
diffusion_batch_size: 4
diffusion_chunk_size: 2
steps: 1
device: cuda
precision: bf16
```

Config mode is intentionally config-only: do not combine `--config` with a
positional training source or any training CLI option. Its non-training
companions are `--print-config` and `-h`/`--help`; help prints the CLI reference
without loading the YAML. `--print-config` checks launch-schema types and
cross-option rules and prints the resolved Runner configuration without
starting training. It does not check that input paths are readable, construct
or validate the selected model or Processor, open a dataset, load a checkpoint,
or perform other runtime validation. Legacy CLI-only invocations remain
supported when `--config` is absent. Do not put launcher controls such as GPU
count, node count, partition, memory, time limit, or Slurm job name in the YAML;
those are scheduler resources supplied to `sbatch`.

YAML is loaded as data only. It does not interpolate environment variables,
follow include directives, or execute custom YAML tags. Top-level launch keys
and the complete `train_datasets` component/cluster schema are closed, so
unknown keys there are rejected. Known `model_config` and `processor_config`
cross-option constraints are checked while parsing, but their family-specific
contents remain open mappings and are validated later when the model and
Processor are constructed. Consequently, `--print-config` is a launch-schema
smoke, not proof that training can start.

Relative paths, including the YAML path and paths stored inside it, resolve
from the training process's current working directory. The repository Slurm
launchers change to `REPO_ROOT` before invoking Python, so their relative paths
resolve from `REPO_ROOT`; direct invocations use the shell's current directory.
For multi-node jobs, the YAML and every referenced data, MSA, CCD, pretrained,
workspace, and checkpoint path must be visible at the same path on every node.
The squashfs launcher additionally requires those paths to be visible through
its container mounts. The AFDB example intentionally retains the source
`/data/projects/allodesign/datasets/afdb_swissprot/mmcif`; actual training
requires that exact path and its associated files to be readable and shared as
described above. `--print-config` does not verify them.

## Inputs

Training input is selected in exactly one of two ways: the legacy positional
`train_source`, or `--train-datasets JSON_OR_PATH` for a mixture of corpora. A
source may be one PDB/CIF/mmCIF file, a directory (searched recursively), a
glob, or a text/JSONL manifest. A text
manifest contains one structure path per line. Blank lines and `#` comments are
ignored. A JSONL manifest can attach explicit MSA evidence:

```json
{"structure": "structures/1abc.cif", "msa": {"path": "msa/1abc_A.a3m", "entity_id": "A"}}
{"structure": "structures/2xyz.cif", "msa_collection": "msa/2xyz_pairing.json"}
```

Relative paths in a manifest are relative to that manifest. An
`msa_collection` is JSON produced by `ofoldx.data.encode_msa_collection`.
One manifest record cannot contain both `msa` and `msa_collection`.
Alternatively, use the nonempty global pair
`--seq-to-pdb-index INDEX.json --msa-path MSA_ROOT`; explicit evidence in a
manifest takes precedence. In legacy mode this pair configures the training
split and every evaluation split. In multidataset mode it is inherited only by
training components that omit both local MSA keys, while every evaluation split
still uses the global pair. MSA input requires an MSA-capable processor
(including Protenix); LigandMPNN and ProteinMPNN do not accept it.

By default, native Runner training datasets apply a 30-second timeout to each
candidate's runtime structure read, MSA read, and processor/crop work. The
`processing_timeout_seconds` key on a `train_datasets` component can override
that training default, while an explicit `null` disables the deadline for that
component. A handled warning is
logged before scanning continues. In cluster sampling, structure/MSA timeouts
quarantine other rows backed by the same CIF for that bounded request, while a
processor/crop or unknown-stage timeout skips only its logical row. In a
multidataset run, exhausting the selected child because of handled timeouts
tries each other child at most once in sorted cyclic order while timeout
exhaustion continues, and the configured sampling weights are unchanged.
Evaluation disables this timeout by
default, although a low-level Runner split configuration may explicitly set
`processing_timeout_seconds` to opt in. Eager cluster-index cache construction
is not timed, and a dataset whose candidates are all unusable still fails
terminally. Timeout warnings report both the configured deadline and actual
elapsed wall time. The deadline uses POSIX real-time signals and is best effort:
native code can defer Python signal delivery.

Training data loading is fault tolerant by default. For each requested sample,
the dataset scans replacement candidates after a recoverable structure-read,
MSA-read, processor/crop, or timeout error. A successful replacement resets the
consecutive-error count. The request terminates on its fifth consecutive
recoverable error, or earlier when its finite candidate set is exhausted; the
last failure is then propagated so DDP does not silently lose a batch. This
limit is local to one dataset request and therefore remains well defined with
multiple DataLoader workers. Evaluation remains strict: it does not substitute
another structure after a data-loading error.

Every skipped training candidate emits an error-provenance warning to Slurm
stderr. When text/file logging is enabled, it also records structured
provenance below the resolved workspace as
`logs/<DanLing run timestamp>.data-errors.rank-*.worker-*.jsonl`. If the low-level
`logging.file` setting explicitly names the main log, that configured path is
authoritative; its directory and stem also determine the data-error sidecars
instead of the default main-log stem. Do not reuse the same explicit
`logging.file` for concurrently running jobs, because equal rank/worker sidecar
stems would collide. Each JSONL record identifies the failed source/candidate,
stage, exception, rank, and worker so a bad sample can be reproduced after training.
Workers deliberately write separate files rather than all appending to the
main training log: concurrent multi-process appends to one shared NFS file can
interleave records, add lock contention, and stall the DataLoader. These JSONL
files are part of the training logs and preserve the full error content
referenced by stderr. This recovery boundary covers dataset item construction
only. CUDA/host OOM, `SIGKILL` or other process failures, collation errors, and
model forward/backward errors are not caught and still terminate training.

### Multiple training datasets

`--train-datasets` accepts either an inline JSON object or the path to a UTF-8
JSON file containing that object; `~` is expanded in the JSON file path.
Top-level keys are unique component names
matching `[A-Za-z][A-Za-z0-9_-]*`. This example shows two named corpora and the
complete per-dataset shape:

```json
{
  "pdb": {
    "source": "data/pdb",
    "sampling_ratio": 3.0,
    "seq_to_pdb_index": "data/pdb-seq-index.json",
    "msa_path": "data/pdb-msa",
    "processing_timeout_seconds": 240,
    "max_consecutive_processing_errors": 5,
    "cluster_sampling": {
      "enabled": false,
      "cluster_file": null,
      "indices_file": null,
      "epoch_size": null,
      "replacement": true,
      "contact_cutoff": 5.0,
      "include_all_chain_pairs": false,
      "exclude_mol_types": ["ions"],
      "seed": null,
      "beta": {"chain": 0.5, "interface": 1.0},
      "alpha": {"prot": 3.0, "nuc": 3.0, "ligand": 1.0}
    }
  },
  "distillation": {
    "source": "data/distillation",
    "sampling_ratio": 1.0,
    "seq_to_pdb_index": "data/distillation-seq-index.json",
    "msa_path": "data/distillation-msa",
    "cluster_sampling": {
      "enabled": true,
      "cluster_file": "data/distillation-clusters.txt",
      "indices_file": "experiments/cache/distillation-indices.jsonl",
      "epoch_size": null,
      "replacement": true,
      "contact_cutoff": 5.0,
      "include_all_chain_pairs": false,
      "exclude_mol_types": ["ions"],
      "seed": null,
      "beta": {"chain": 0.5, "interface": 1.0},
      "alpha": {"prot": 3.0, "nuc": 3.0, "ligand": 1.0}
    }
  }
}
```

`source` is required and nonempty; `sampling_ratio` defaults to `1` and must be
finite and positive. A component's `processing_timeout_seconds` must be a
positive finite number or `null`, and `max_consecutive_processing_errors` must
be a positive integer or `null`; booleans are rejected for both. Omitting these
keys retains the Runner training defaults of 30 seconds and five consecutive
recoverable errors, while explicit `null` disables the corresponding bound for
that component. `cluster_file` is required and nonempty when
`cluster_sampling.enabled` is true; `indices_file` may remain `null` to use the
workspace cache. Explicit component `indices_file` values are expanded and
resolved to absolute paths, and no two components may resolve to the same cache
file. In a mixture, component-local `epoch_size` and `seed` must be
`null` or omitted, and `replacement` must remain `true`; the mixture-level
epoch size and seed control the shared draw. `contact_cutoff` must be positive,
the `beta`/`alpha` values must be non-negative, and unrecognized nested keys are
rejected.

Dataset-local `seq_to_pdb_index` and `msa_path` must be given together as
nonempty strings. When both local keys are absent, a configured global
`--seq-to-pdb-index`/`--msa-path` pair is inherited; if no global pair is
configured, that component has no indexed-MSA fallback. A complete local pair
overrides the global pair. A partial local pair is rejected. Local pairs apply
only to their training component; evaluation splits use the global pair only.

Ratios are relative probabilistic per-sample weights, not epoch quotas. For
dataset `d`, let `r_d` be `sampling_ratio`. Conditional on choosing an ordinary
dataset, its logical candidates are uniform; conditional on choosing a
clustered dataset, its index rows use their cluster weights. Thus

```text
P(dataset=d, candidate=i) = r_d / sum_j(r_j)
                            * (1 / N_d)                  # ordinary
P(dataset=d, candidate=i) = r_d / sum_j(r_j)
                            * (w_di / sum_k(w_dk))       # clustered
```

`--train-epoch-size N` sets the positive global number of samples requested per
epoch. By default it is the sum of logical candidates: ordinary record counts
plus clustered weighted-index row counts. Sampling is with replacement and is
recreated deterministically from `--seed` and the epoch. Under DDP the shared
draw is padded to `ceil(train_epoch_size / world_size) * world_size`; rank `r`
receives the strided subsequence `draw[r::world_size]`, so the processed total
can be slightly larger than the requested epoch size.

The global training crop settings apply to every corpus. Continuous cropping
works for ordinary and clustered corpora. Positive spatial or
interface-spatial crop weight requires cluster sampling for every corpus,
because ordinary candidates do not carry `reference_instance_ids`. Cropping,
mixture sampling, and dataset-local MSA settings are training-only; evaluation
split definitions, scoring, full-structure evaluation, and global indexed-MSA
routing for evaluation are unchanged.

Legacy positional mode remains supported with its existing `--cluster-*`
flags. It cannot be combined with `--train-datasets`. The legacy cluster flags
are rejected in multidataset mode because it would be ambiguous which corpus
they configure; put `enabled`, `cluster_file`, and `indices_file` inside each
dataset object instead.

## Invocations

The recommended Slurm form keeps scheduler resources before the launcher and
forwards only the YAML configuration after `--`:

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch ofoldx/runners/submit_train.slurm -- --config configs/train.yaml
```

Add scheduler overrides before the launcher when needed:

```bash
sbatch [resources] ofoldx/runners/submit_train.slurm -- --config configs/train.yaml
```

Here `[resources]` means `sbatch` options such as `--job-name`, `--nodes`,
`--gres`, `--cpus-per-task`, `--mem`, and `--time`; they are not YAML training
keys. The repository launcher rejects execution outside an allocation. Its
established header requests the `gpu` partition, one node, one task, one A800,
96 CPUs, and `--mem=0` (the node's full memory); it has no explicit time limit,
so the site's default applies. Command arguments after the script (and an
optional `--`) are forwarded unchanged.

Slurm opens the `#SBATCH --output` and `--error` files before the shell script
runs, so the launcher cannot create their parent directory. Create
`/data/users/qinghanw313/AlloDesign/logs` before every first submission from a
clean checkout. For a conservative launch-schema smoke, override the established
CPU, memory, and time defaults explicitly:

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --job-name=afdb-config-smoke \
  --cpus-per-task=4 --mem=96G --time=00:10:00 \
  ofoldx/runners/submit_train.slurm -- \
  --config configs/train.yaml --print-config
```

This smoke validates and prints only the launch configuration; it does not
prove that the AFDB paths or runtime model/data setup are usable.

Direct config-based training is also supported, but run it only inside an
interactive or batch Slurm compute-node allocation, never on a management node:

```bash
python -m ofoldx.runners.train --config configs/train.yaml
```

Single-node, four-GPU training:

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --gres=gpu:a800:4 --cpus-per-task=32 \
  --export=ALL,NPROC_PER_NODE=4 \
  ofoldx/runners/submit_train.slurm -- \
  --config configs/train.yaml
```

Two nodes with eight A800 GPUs per node (16 ranks total):

```bash
mkdir -p /data/users/qinghanw313/AlloDesign/logs
sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:a800:8 \
  --cpus-per-task=64 --mem=0 --time=24:00:00 \
  --export=ALL,NPROC_PER_NODE=8 \
  ofoldx/runners/submit_train.slurm -- \
  --config configs/train.yaml
```

Put per-rank batch size, accumulation, workers, shared workspace, and run name
in the YAML. Keep node/GPU allocation and the Slurm job name in `sbatch`.

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
  --config configs/train.yaml
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

Legacy CLI-only training remains available, including JSON-backed multidataset
arguments using a JSON file that you create:

```bash
python -m ofoldx.runners.train data/train \
  --device cuda --precision bf16 --epochs 10

sbatch --job-name=protenix-mixture ofoldx/runners/submit_train.slurm -- \
  --train-datasets path/to/train-datasets.json \
  --train-epoch-size 100000 \
  --device cuda --precision bf16 --epochs 10
```

## Command-line reference

Numeric arguments described as positive must be greater than zero; non-negative
arguments may be zero. `--epochs` and `--steps` are mutually exclusive.

| Argument | Default and behavior |
| --- | --- |
| `-c YAML_PATH`, `--config YAML_PATH` | Recommended YAML-only interface. Cannot accompany a positional source or training CLI options; only `--print-config` and `-h`/`--help` are accepted as non-training companions. |

The following legacy CLI arguments remain supported without `--config`. Their
canonical destination names, with hyphens replaced by underscores, are the
corresponding YAML top-level keys.

### Data and evaluation

| Argument | Default and behavior |
| --- | --- |
| `train_source` | Legacy file, directory, glob, or manifest. Required unless `--train-datasets` is used; mutually exclusive with it. |
| `--train-datasets JSON_OR_PATH` | Inline JSON object or path to a JSON object of named training dataset components; mutually exclusive with positional `train_source`. |
| `--train-epoch-size N` | Positive global mixture draws per epoch. Only valid with `--train-datasets`; default is the sum of all corpora's logical candidate counts. |
| `--val-source PATH` | Adds the evaluation split named `val`. |
| `--eval-source NAME=PATH` | Repeatable named evaluation split. Names must match `[A-Za-z][A-Za-z0-9_-]*`, be unique, and cannot be `train`; `val` conflicts with `--val-source`. |
| `--score-split NAME` | Split whose loss selects the best checkpoint. It must name a configured evaluation split. Default: `val` if present, otherwise the first named split. |
| `--seq-to-pdb-index PATH` | Nonempty sequence-to-MSA-directory JSON index; requires `--msa-path`. In legacy mode it configures training and every evaluation split. In multidataset mode it is inherited by training components without a local pair and independently configures every evaluation split. |
| `--msa-path DIR` | Nonempty root of indexed MSA directories; requires `--seq-to-pdb-index`. Dataset-local complete pairs override the global pair for that training component only. |

All configured evaluation splits run separately. Scoring uses only
`--score-split`; the others are still evaluated and logged. For
`protenix_baseline_v1`, labeled evaluation runs reverse diffusion and
target-aware prediction-side permutation, preferring EMA weights only when
`ema_decay` is enabled.

### Model and CCD

| Argument | Default and behavior |
| --- | --- |
| `--model ID` | Registered architecture. A fresh run defaults to `protenix`. |
| `--pretrained PATH_OR_REPO` | Load a local or Hugging Face model repository. |
| `--revision REVISION` | Branch, tag, or commit for `--pretrained`; invalid without it. |
| `--objective-recipe {protenix_baseline_v1,ofoldx_edm_v1}` | Versioned objective contract. A fresh Protenix run defaults to the baseline aggregate recipe and may explicitly select legacy EDM compatibility. Non-Protenix architectures reject the Protenix baseline; absent or explicit EDM is persisted without an `objective_recipe` key to preserve historical configuration identity. |
| `--model-config JSON_OR_PATH` | Inline JSON object or path to a JSON object; default `{}`. Cannot set reserved `name`, `pretrained`, or `revision`; a nonempty value cannot accompany `--pretrained`. |
| `--processor-config JSON_OR_PATH` | Inline JSON object or path; default `{}`. A nonempty value cannot accompany `--pretrained`. |
| `--ccd-revision REVISION` | Nonempty CCD repository revision for a fresh processor. Mutually exclusive with `--ccd-snapshot-path`, `--pretrained`, and CCD locator keys in processor config. |
| `--ccd-snapshot-path PATH` | Nonempty local CCD snapshot for a fresh processor; the same exclusions apply. |

If `--model` and `--pretrained` are both given, the explicit model ID must match
the architecture recorded by the repository. In `--processor-config`,
`ccd_revision` and `ccd_snapshot_path` must each be either a nonempty string or
`null`, and at most one may be non-null. If either key is present in processor
config, neither explicit CCD locator flag may be used.

For Protenix, the recipe is resolved before DanLing hashes the Runner
configuration and is stored in full checkpoints. Omitting it on a fresh run
selects `protenix_baseline_v1`: the family-local baseline training forward, the
full configured aggregate criterion (coordinate MSE, smooth-LDDT, bond,
distogram, pLDDT, PDE, resolved, and PAE), the required auxiliary heads, and
structure-level accumulation weighting. Explicitly selecting `ofoldx_edm_v1`
retains the historical Protenix coordinate-denoising objective and keeps its
auxiliary heads disabled.

The Protenix baseline is an initial aggregate recipe, not a reproduction of the
complete upstream four-stage training program. It does not supply the released
stage curriculum and transitions, learning-rate scheduler, or dataset
selection, mixture, and sampling policy; configure those training-application
choices separately.

For the baseline recipe, Runner also expands the complete initial objective
preset into the checkpointed `objective_config` before hashing the run. This
records every loss, chemistry, binning, resolution, epsilon, and zero-weight-term
setting rather than leaving the run identity dependent on implicit code defaults.
The raw stage factors and their expanded effective weights are both available
through that preset. When `train_confidence_only: true`, Runner passes the flag
to the model forward and the objective executes only confidence terms; the raw
stage-factor defaults remain part of the preset and evaluation ignores this
training-only execution switch.

Finite detached raw and weighted objective components are registered with
DanLing's meters under `objective/*`. Runner pre-registers the complete metric
schema on every rank so resolution-gated confidence terms may be absent on one
rank without changing collective order. A term absent from the current
microbatch is omitted from step logging rather than reusing a stale value.

A historical Protenix full checkpoint without `objective_recipe`
unambiguously means `ofoldx_edm_v1`. It may resume under that explicit legacy
recipe, but it is rejected under `protenix_baseline_v1` so an old Protenix run
cannot silently change its objective.

For non-Protenix architectures, Runner canonicalizes both an absent recipe and
an explicit `ofoldx_edm_v1` selection to an absent persisted
`objective_recipe` key, preserving the historical configuration hash and run
identity. At runtime, an internal EDM fallback still dispatches the family's
existing objective; it does not replace that objective with coordinate
denoising. The supported-architecture table below remains authoritative, and
`protenix_baseline_v1` is rejected for these architectures.

### Protenix diffusion sample batching

`diffusion_batch_size` in YAML (or `--diffusion-batch-size S`) draws `S`
independent rigid augmentations, noise levels, and coordinate-noise realizations
for each processed structure. This is the diffusion sample axis in model outputs:
coordinates and augmented targets have shape `(B, S, A, 3)`, and noise levels
have shape `(B, S)`. The objective averages over this sample axis; it does not
multiply the loss scale by `S`. Omission or YAML `null` selects `S=48` for
`protenix_baseline_v1`. The legacy recipe retains the historical effective
`S=1` without adding a diffusion-size key.

`diffusion_chunk_size` (or `--diffusion-chunk-size S_CHUNK`) limits how many of
those samples one Protenix denoiser call evaluates. The trunk is computed once,
and all random corruptions are drawn before chunking, so changing the chunk size
does not redraw training examples. A final remainder chunk is allowed. Chunking
is not a DataLoader batch and is not a strict training microbatch: the chunk
graphs remain connected for one backward pass and their outputs are concatenated.
The baseline recipe defaults to `C=4`; the legacy recipe omits chunking unless
configured. Chunking can lower peak denoiser activation memory, but it does not
guarantee linear memory savings.

Both controls support only `protenix`, `protenix_template`,
`protenix_structure_condition`, and `protenix_constraint`. They apply through
the shared collate function to training; reverse-diffusion inference sampling
is unchanged. Labeled `protenix_baseline_v1` evaluation uses five samples, 200
diffusion transitions, and the model's full recycling count. It disables TFG,
draws the released 0.4 per-structure Monte Carlo-dropout decision, and applies
target-aware prediction-side atom/token permutation to every coordinate sample
and confidence head. It uses EMA shadow weights only when `ema_decay` is
enabled, otherwise the live model. Evaluation requires a full, uncropped atom
axis and rejects incomplete or duplicate `atom_index` mappings. These
architectures still require
DataLoader `batch_size: 1`. With `W` DDP ranks and `K` accumulation steps, one
optimizer update draws `W * K * S` diffusion corruptions (assuming every rank
contributes one structure at each accumulation step).

### Supported training architectures

The registry contains additional architectures for construction, loading, and
inference, but this Runner trains only the exact architecture IDs below. Any
other registered ID fails with “has no built-in Runner objective.”

| Architecture ID(s) | Runner objective | Batching |
| --- | --- | --- |
| `proteinmpnn` | Uniform label-smoothed protein sequence recovery over designable residues. | Ragged multi-structure batches supported. |
| `ligandmpnn` | Protein sequence recovery with ligand context; the same uniform label-smoothed loss as ProteinMPNN. | `--batch-size 1` required. |
| `nampnn` | Polymer-specific label-smoothed protein/DNA/RNA sequence recovery. | Ragged multi-structure batches supported. |
| `alphafold3`, `boltz1`, `boltz2`, `boltzgen` | Shared `ofoldx_edm_v1` coordinate-denoising loss over resolved atoms. | `--batch-size 1` required. |
| `protenix`, `protenix_template`, `protenix_structure_condition`, `protenix_constraint` | Family-local `protenix_baseline_v1` aggregate objective by default; `ofoldx_edm_v1` is the historical opt-in. | `--batch-size 1` required; baseline accumulation is structure-normalized. |
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
| `--diffusion-batch-size S` | Optional positive number of independent Protenix diffusion corruptions per input; omitted means `S=48` for baseline and historical effective `S=1` for EDM. |
| `--diffusion-chunk-size S_CHUNK` | Optional positive maximum Protenix samples per denoiser call; omitted means `C=4` for baseline and no chunking for EDM. |
| `--accum-steps N` | Positive gradient accumulation factor; default `1`. |
| `--workers N` | Non-negative data-loader workers per rank; default `0`. |
| `--precision {bf16,fp16}` | Optional reduced precision; unset by default. |
| `--max-grad-norm FLOAT` | Optional positive gradient-clipping norm. |
| `--ema-decay FLOAT` | Optional model-parameter exponential moving average decay; disabled when omitted. When set, it must satisfy `0 < decay < 1` (for example, `0.999`). |
| `--seed N` | Non-negative random seed; default `1016`. |
| `--deterministic`, `--no-deterministic` | Deterministic mode; default off. |
| `--gradient-checkpointing`, `--no-gradient-checkpointing` | Activation recomputation; default off. This is unrelated to saving training-state checkpoints. |

### Exponential moving average (EMA)

Enable a shadow model from the legacy CLI with `--ema-decay 0.999`, or in YAML
with:

```yaml
ema_decay: 0.999
```

EMA is disabled by default. It is updated after every successful optimizer
update, so with gradient accumulation it advances only at an accumulation
boundary, not after each micro-batch. Evaluation and inference prefer the EMA
shadow when enabled. A DanLing full training-state checkpoint stores both the
live model and EMA state; resuming with EMA enabled from a checkpoint that has
no EMA state fails strictly instead of silently reinitializing it. The final
model-only artifact remains the live model weights, not EMA weights.

Each rank keeps one additional model-parameter copy. The shadow preserves the
live model parameter dtype; with the current bf16/fp16 autocast training path,
the master parameters (and therefore the shadow) are normally FP32, costing
about 4 bytes per parameter. EMA also increases checkpoint size and adds an
`O(P)` parameter update after each optimizer step. EMA and gradient
checkpointing can be enabled together: the former stores parameter history,
while the latter trades activation memory for recomputation. The Slurm launcher
needs no additional option; put `ema_decay` in the YAML passed after `--`, or
pass `--ema-decay` in a legacy CLI-only invocation.

### Protenix training crop

| Argument | Default and behavior |
| --- | --- |
| `--max-tokens N`, `--crop-size N` | Positive maximum training tokens per sample. Only Protenix supports crop options. |
| `--crop-method-weights CONTINUOUS SPATIAL INTERFACE_SPATIAL` | Three non-negative weights, not all zero; requires `--max-tokens`. Default is `1 0 0`; it becomes `0.2 0.4 0.4` when legacy `--cluster-file` is set or every multidataset component enables cluster sampling. Positive spatial weights require legacy `--cluster-file`, or cluster sampling on every multidataset component. |

Cropping applies only to training: legacy mode writes it to the single training
dataset, while multidataset mode copies it to every training component.
Evaluation uses full structures and can run out of memory even when cropped
training fits; reduce evaluation inputs or run evaluation with sufficient
memory.

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
| `--resume` | Resume the latest checkpoint for this Runner ID from its native hash workspace. Mutually exclusive with `--resume-from`; loading remains available with `--no-checkpoint`. |
| `--resume-from CHECKPOINT` | Resume an explicit checkpoint while writing to the native hash workspace; loading remains available with `--no-checkpoint`. |
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
| `--print-config` | Run launch-schema and cross-option validation, then print resolved JSON without filesystem or model/Processor/dataset runtime validation. It does not initialize TensorBoard/W&B, so optional logger packages are not required. |

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

OFoldX uses DanLing's native hash-based workspace layout:

```text
<workspace.root>/<lineage[-code_id]>/<runner.id>/
```

The training entry point sets `lineage: ofoldx`, so a repository checkout with
a code hash normally produces:

```text
<workspace.root>/ofoldx-<code_id>/<code_id>-<config_id>/
├── logs/<runner.timestamp>.log
└── checkpoints/
```

All ranks in a distributed run use this same directory. Identical code and
semantic configuration resolve to the same workspace, and `--resume` uses
DanLing's native latest-checkpoint lookup there. `--resume-from` loads the
specified checkpoint and writes subsequent output to the same native hash
workspace. Use a shared workspace path for multi-node jobs, and do not launch
concurrent jobs with an identical Runner identity because their logs and
checkpoints would share paths. A low-level YAML `workspace.dir` remains an
exact-path override.

Runs created by the former UTC-session layout remain readable, but native
`--resume` does not scan timestamp children below the hash workspace. Resume
one of those runs once with an explicit `--resume-from` path to its checkpoint.

Before the first submission, create
`/data/users/qinghanw313/AlloDesign/logs`; Slurm opens its output files before
the script body runs. The launcher writes stdout and stderr to
`logs/<job-name>-<jobid>.out` and `.err`: `%x` expands to the Slurm job name and
`%j` to the job ID. Set a descriptive name with `sbatch --job-name=NAME`. The
main training log uses DanLing's native run timestamp as its basename, for
example `W1K4M1Xb1.log`. An explicit low-level `logging.file` remains
authoritative and supplies the basename for both the main log and its
data-error sidecars. Per-worker dataset error provenance uses the effective log
basename described under Inputs and is created only while text/file logging is
enabled.

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

DanLing remains the owner of backward, accumulation, AMP/scaler, optimizer,
scheduler, and EMA lifecycle. Its audited Torch runner synchronizes a
non-finite **gradient** skip decision at the optimizer boundary when gradient
skipping or a GradScaler is active. For `protenix_baseline_v1`, OFoldX adds a
pre-backward maximum reduction of the non-finite **loss** decision over the same
DDP replica group. If any rank reports a non-finite loss, every rank performs a
zero graph-connected backward to complete DDP reducer hooks, restores its exact
pre-backward gradient state, and continues the ordinary accumulation/optimizer/
scheduler/EMA lifecycle. At an accumulation boundary, the restored gradients
from earlier finite microbatches are explicitly averaged across replicas before
the ordinary normalization and optimizer step; parameters absent on every rank
remain gradient-free. The invalid microbatch contributes no objective metrics,
and subsequent step logging cannot reuse its previous metric values. The legacy
`ofoldx_edm_v1` path is unchanged.

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
