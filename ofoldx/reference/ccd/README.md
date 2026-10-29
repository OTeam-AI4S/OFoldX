# CCD

OFoldX structure readers and chemical preparation use the Chemical Component
Dictionary from the Hugging Face dataset [`oteam/ccd`](https://huggingface.co/datasets/oteam/ccd).
The first use downloads the default `main` snapshot; subsequent calls reuse the
Hugging Face cache. Each snapshot includes the artifact and its provenance;
OFoldX validates the schema and checksums before using it.

## Usage

Download the default dataset in advance:

```bash
python -m ofoldx.reference.ccd download
```

The command prints the cached `components.safetensors` path. To select a fixed
snapshot, pass a dataset tag or full commit and configure the returned path:

```bash
export OFOLDX_CCD_ARTIFACT="$(python -m ofoldx.reference.ccd download \
  --repo-id oteam/ccd \
  --revision YOUR_TAG_OR_FULL_COMMIT)"
```

Replace `YOUR_TAG_OR_FULL_COMMIT` with the revision you selected. Add
`--local-files-only` to use a previously cached snapshot without downloading.
The Python API has the same defaults:

```python
from ofoldx.reference.ccd import download

artifact = download()
```

## Use a Local Artifact

Set a local override before reading structures or preparing designs:

```bash
export OFOLDX_CCD_ARTIFACT="$PWD/ccd/components.safetensors"
```

```python
from ofoldx.io import read_structure
from ofoldx.reference.ccd import load_library

library = load_library("ccd/components.safetensors")
structure, report = read_structure("input.cif", library=library)
```

An explicit local path takes precedence over the default dataset. A missing or
invalid local artifact raises an error. See the [I/O README](../../io/README.md)
for reading options and reports.

## Build an Artifact

The build and publication commands are included in the installed package. Choose
the source CCD release and give its identity as `--content-version`:

```bash
python -m ofoldx.reference.ccd build \
  --content-version YOUR_CCD_RELEASE \
  --output ./ccd
```

The default source is the official wwPDB CCD download. To use a local release or
another source URL, pass `--source`:

```bash
python -m ofoldx.reference.ccd build \
  --source ./components.cif.gz \
  --content-version YOUR_CCD_RELEASE \
  --output ./ccd
```

`--source` accepts `.cif` and `.cif.gz` input. Replace `YOUR_CCD_RELEASE` with the
identity of the source you selected. The new output directory contains:

- `components.safetensors`: the versioned OFoldX CCD artifact.
- `components.safetensors.provenance.json`: the source digest and build provenance.
- `README.md`: a card describing the artifact and how it was built.

The artifact uses the `ofoldx.ccd/7` schema. Build provenance records the source
used to construct it; `--content-version` identifies that source release.

## Upload an Artifact

Upload the complete built directory to the official Hugging Face dataset:

```bash
python -m ofoldx.reference.ccd upload ./ccd --repo-id oteam/ccd
```

Authenticate with Hugging Face before uploading; select another `--repo-id` for
an independent dataset and add `--private` to create a private repository. Keep
the artifact, provenance, and card together so consumers can identify its source
and build.
