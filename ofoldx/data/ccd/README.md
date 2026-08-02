# data/ccd

`ofoldx.data.ccd` is the shared wwPDB Chemical Component Dictionary (CCD)
resolver. It is the boundary-time counterpart to
`ofoldx.data.components.get_component`.

- `get_component` is **audited-only** and does **zero I/O** (Principle 7). It
  returns one of the package-audited standard components (the 31 audited IDs) and
  never guesses a chemical graph from a residue name.
- `resolve_ccd_component` is the tool for the _other_ job: a family that folds
  arbitrary CCD-code ligands calls it **once, at its importer / Processor
  boundary**, to fetch the wwPDB CCD snapshot and lower one component into a
  canonical `ChemicalComponent` VALUE. The family then embeds those values in its
  `MolecularSystem`. There is **no live registry lookup at tensor-compute time**.

## Resolver signature

```python
from ofoldx.data.ccd import resolve_ccd_component

component = resolve_ccd_component("HEM", revision="2026-07-23")
# -> ofoldx.data.molecular.ChemicalComponent

# Or resolve from an explicitly prepared local snapshot with no Hub access.
component = resolve_ccd_component("HEM", snapshot_path="/data/ccd-v20240608")
```

- `component_id: str` — an exact wwPDB CCD code (for example `"HEM"`, `"ATP"`).
- `revision: str | None` — an explicit Hub revision. It pins the remote dataset
  commit so a resolved value is reproducible and provenance-tagged.
- `snapshot_path: str | None` — an existing local directory containing one or
  more resolver-schema `*.parquet` shards. This mode performs no Hub access.
- Exactly one of `revision` and `snapshot_path` is required. There is no implicit
  latest revision and no fallback between remote and local resolution.
- Unknown codes raise `KeyError`; a malformed row raises `ValueError` /
  `TypeError`.

The returned value has the same shape produced by
`ofoldx.data.components._build_protein_component` /
`_build_nucleic_component`: `ChemicalAtom` / `ChemicalBond` /
`PolymerAttachment` values, per-atom `reference_coordinates` aligned to the atom
tuple, and polymer attachments only for polymer components.

## Dataset repository

The CCD is distributed as a Hub **`dataset`** repository, `ofoldx/ccd`, stored as
Parquet with **one component per row**. The resolver downloads it through the same
`huggingface_hub.snapshot_download` mechanism used by
`ofoldx.hub.resolve_repository` for model repositories (here with
`repo_type="dataset"` and `allow_patterns=["*.parquet"]`).

**Revision = wwPDB snapshot date.** Each dataset revision corresponds to one wwPDB
CCD snapshot; passing that date as `revision` fixes exact provenance.

For offline work, build or copy the same Parquet schema into a versioned local
directory and pass that directory explicitly as `snapshot_path`. The directory
name and accompanying metadata should record the source CCD snapshot date; a
local snapshot must never be labeled as a newer remote revision.

### Parquet row schema

Each row is one component. List-valued columns for atoms are all aligned to
`atom_ids`; list-valued bond columns are all aligned to `bond_atom_a`.

| Column                     | Type                                | Meaning                                                                                                     |
| -------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `component_id`             | `str`                               | wwPDB CCD code (row key).                                                                                   |
| `chem_type`                | `str`                               | `protein` / `rna` / `dna` / `dna_rna_hybrid` / `non-polymer`. Determines whether polymer attachments apply. |
| `atom_ids`                 | `list[str]`                         | Component-local atom IDs, in canonical order.                                                               |
| `elements`                 | `list[str]`                         | Element symbol per atom.                                                                                    |
| `formal_charges`           | `list[int \| None]`                 | Formal charge per atom; `None` when the CCD source is unspecified.                                          |
| `aromatic`                 | `list[bool \| None]`                | Aromatic flag per atom; `None` when incident bond aromaticity is unspecified.                               |
| `stereo`                   | `list[str \| None]`                 | Per-atom stereo config (`R`/`S`/…), `None` when unspecified.                                                |
| `bond_atom_a`              | `list[str]`                         | First endpoint atom ID per bond.                                                                            |
| `bond_atom_b`              | `list[str]`                         | Second endpoint atom ID per bond.                                                                           |
| `bond_orders`              | `list[float \| None]`               | Bond order per bond (`1.0`, `1.5` aromatic, `2.0`, `3.0`); `None` when unspecified.                         |
| `bond_aromatic`            | `list[bool \| None]`                | Aromatic flag per bond; `None` when unspecified.                                                            |
| `bond_stereo`              | `list[str \| None]`                 | Per-bond stereo config, `None` when unspecified.                                                            |
| `reference_coordinates`    | `list[[float,float,float] \| None]` | Ideal reference coordinate per atom (aligned to `atom_ids`); `None` where absent.                           |
| `previous_attachment_atom` | `str \| None`                       | Polymer previous-link anchor atom (e.g. protein `N`, nucleic `P`).                                          |
| `previous_leaving_atoms`   | `list[str]`                         | Atoms removed on forming the previous link.                                                                 |
| `next_attachment_atom`     | `str \| None`                       | Polymer next-link anchor atom (e.g. protein `C`, nucleic `O3'`).                                            |
| `next_leaving_atoms`       | `list[str]`                         | Atoms removed on forming the next link.                                                                     |

For `non-polymer` components the attachment columns are `None` / empty and the
resolver omits polymer attachments.

## Building the dataset

`build.py` is the offline converter: wwPDB `components.cif` (optionally
gzipped) → the Parquet rows above. Its parse follows the reference generator
`alphafold3/constants/converters/ccd_pickle_gen.py` (read the multi-data mmCIF,
split into one dict per `data_` block), and the field lowering mirrors
`alphafold3/structure/chemical_components.py` (`parse_atom_data` /
`parse_bond_data`). The mmCIF and Parquet dependencies are imported lazily by the
build entry points.

A complete offline build is:

```python
from ofoldx.data.ccd.build import build_rows, write_parquet

rows = build_rows("/data/components.v20240608.cif")
write_parquet(rows, "/data/ccd-v20240608/components.parquet")
```

The converter materializes the parsed CIF and normalized rows, so a full CCD
build should run as a dedicated data-preparation job with adequate memory. At
runtime, the resolver reads only the resulting Parquet shards. The RDKit pickle
distributed by some model datasets is not resolver input and is not used to
generate reference coordinates.

`build.py` is a build-time tool only — it is never imported during modeling and
never runs during tensor compute.
