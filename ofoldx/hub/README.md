# Hub

Use `ofoldx.hub` to download model or dataset snapshots, resolve local repository
paths, and publish repository files on the Hugging Face Hub.

## Usage

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

Resolve the official ProteinMPNN repository, then use its local snapshot to score
the protein backbone in `backbone.cif`:

```python
from ofoldx import pipeline
from ofoldx.hub import resolve_repository
from ofoldx.io import read_structure

repository = resolve_repository("oteam/proteinmpnn")
structure, report = read_structure("backbone.cif")
scorer = pipeline("score", model=repository.path)
result = scorer(structure, seed=7)
print(repository.commit)
print(result.scores["designed_sequence_nll"].value)
```

The returned `ResolvedRepository` contains the local path and the Hub commit.
Keep the commit with your run configuration to identify the snapshot you used.
For a local directory, the result contains that path and `commit=None`.

## Select a Snapshot

`resolve_repository` accepts a Hub repository ID or a local path.
`resolve_hub_repository` downloads a Hub snapshot directly. Both support:

- `revision`: a branch, tag or commit; use a full commit for a fixed snapshot.
- `allow_patterns`: the repository files to download.
- `local_files_only=True`: use files already in the local cache.
- `repo_type`: `"model"` by default, or `"dataset"` for a dataset.

A `ResolvedRepository` also retains its repository type and component
subdirectory. Passing it back to the resolver keeps that snapshot identity
when fetching additional files.

## Load Model Repositories

`pipeline` loads the model, processor and task workflow together. `AutoModel`
and `AutoProcessor` load individual components. The lower-level helpers in
`ofoldx.hub.model` read native repository metadata and restore model weights;
they are also available from `ofoldx.hub`.

## Publish a Repository

After converting or exporting a local model, publish its directory to your Hub
namespace:

```python
from ofoldx.hub import upload_repository

commit = upload_repository("./models/proteinmpnn", "YOUR_NAMESPACE/proteinmpnn")
print(commit.oid)
```

Replace `YOUR_NAMESPACE` with your destination namespace. `upload_repository`
updates the model files while retaining unrelated remote files. A converter
writes the model card, AGPL documents and `conversion.json` before publishing
when supplied with `repo_id`. A pipeline's `save(directory, repo_id=...)` saves
its local repository and uploads that directory.

For other repository files, `upload_folder` accepts `repo_type`, `allow_patterns`
and `delete_patterns` to control the destination and uploaded file set.

## Related Modules

- [Task API](../api/README.md): load and execute a biological operation.
- [Models](../models/README.md): checkpoint conversion and native model repositories.
- [Pipelines](../pipelines/README.md): paired and composite workflows.
