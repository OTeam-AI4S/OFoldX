# data/ccd

`ofoldx.data.ccd` is the shared wwPDB Chemical Component Dictionary (CCD)
resolver. It is the boundary-time counterpart to
`ofoldx.data.components.get_component`.

- `get_component` is **audited-only** and does **zero I/O** (Principle 7). It
  returns one of the package-audited standard components (the 34 audited IDs) and
  never guesses a chemical graph from a residue name.
- `resolve_ccd_component` lowers one entry from a pinned wwPDB CCD snapshot into
  a canonical `ChemicalComponent` value.
- `prepare_structure` is the offline boundary that reads a source structure,
  fills chemistry the source does not state, validates the completed graph, and
  writes one exact `.structure.safetensors` artifact. In this offline workflow,
  dataset workers can load the prepared graph without accessing the CCD.

Direct CIF workflows can resolve components during reading. Allodesign's
`CifStructureLoader` passes `resolve_ccd_component` with its local CCD snapshot
to `materialize_structure`. See [sources and projections](../../../docs/io_sources.md)
for the source reading and graph construction boundaries.

Source-declared chemistry wins. CCD values fill missing atoms, bonds,
leaving-atom facts, attachments, and reference coordinates, but never rewrite
an explicit source charge or bond fact.

Completion can expose an occurrence-level reaction that the source bond records
state without naming every consumed template atom. `prepare_structure` removes
such atoms only when they are unobserved and RDKit validation proves one unique
chemical result: an ordinary terminal proton at an atom that forms a new
covalent bond. Heavy-atom leaving groups must already be declared by the source
or CCD and are realized while the structure graph is built. Missing evidence, a
source contradiction, or multiple non-equivalent repairs remains an error; this
is not a general bad-structure correction pass.

An observed hydrogen absent from the generic CCD component is treated as an
occurrence proton only when its coordinates identify one neutral nitrogen, the
source does not assign that nitrogen's charge, the remaining component graph
exactly matches the CCD, and no unobserved CCD hydrogen already occupies that
valence. The parent charge is increased with the added proton; ambiguous
geometry or contradictory source chemistry remains incomplete and therefore
fails.

## Resolver signature

```python
from ofoldx.data.ccd import resolve_ccd_component

component = resolve_ccd_component("HEM", revision="2026-07-23")
# Or resolve from an existing local Parquet snapshot without Hub access.
local_component = resolve_ccd_component("HEM", snapshot_path="/data/ccd")
# -> ofoldx.data.molecular.ChemicalComponent

from ofoldx.data.ccd import prepare_structure

prepare_structure(
    "input.cif",
    "input.structure.safetensors",
    snapshot_path="/data/ccd",
)
```

- `component_id: str` — an exact wwPDB CCD code (for example `"HEM"`, `"ATP"`).
- `revision: str | None` — the wwPDB snapshot date. It pins the Hub dataset
  commit so a resolved value is fully reproducible and provenance-tagged.
- `snapshot_path: path-like | None` — a local Parquet file or directory
  containing the same schema. Exactly one of `revision` and `snapshot_path` is
  required.
- Unknown codes raise `KeyError`; a malformed row raises `ValueError` /
  `TypeError`.

The returned value has the same shape produced by
`ofoldx.data.components._build_protein_component` /
`_build_nucleic_component`: `ChemicalAtom` / `ChemicalBond` /
`PolymerAttachment` values, per-atom `reference_coordinates` aligned to the atom
tuple, and polymer attachments only for polymer components.

## Dataset repository

The CCD is distributed as a Hub **`dataset`** repository, `oteam/ccd`, stored as
Parquet with **one component per row**. The resolver downloads it through the same
`huggingface_hub.snapshot_download` mechanism used by
`ofoldx.hub.resolve_repository` for model repositories (here with
`repo_type="dataset"` and `allow_patterns=["*.parquet"]`).

**Revision = wwPDB snapshot date.** Each dataset revision corresponds to one wwPDB
CCD snapshot; passing that date as `revision` fixes exact provenance.

### Parquet row schema

Each row is one component. List-valued columns for atoms are all aligned to
`atom_ids`; list-valued bond columns are all aligned to `bond_atom_a`.

| Column                     | Type                                | Meaning                                                                                                     |
| -------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `component_id`             | `str`                               | wwPDB CCD code (row key).                                                                                   |
| `chem_type`                | `str`                               | `protein` / `rna` / `dna` / `dna_rna_hybrid` / `non-polymer`. Determines whether polymer attachments apply. |
| `parent_component_ids`     | `list[str]`                         | Standard parent CCD IDs declared for a nonstandard component.                                               |
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
| `leaving_atom_ids`         | `list[str]`                         | CCD atoms explicitly marked as leaving.                                                                     |
| `non_leaving_atom_ids`     | `list[str]`                         | CCD atoms explicitly marked as non-leaving; atoms in neither list have an unknown flag.                     |
| `previous_attachment_atom` | `str \| None`                       | Polymer previous-link anchor atom (e.g. protein `N`, nucleic `P`).                                          |
| `previous_leaving_atoms`   | `list[str]`                         | Atoms removed on forming the previous link.                                                                 |
| `next_attachment_atom`     | `str \| None`                       | Polymer next-link anchor atom (e.g. protein `C`, nucleic `O3'`).                                            |
| `next_leaving_atoms`       | `list[str]`                         | Atoms removed on forming the next link.                                                                     |

For `non-polymer` components the attachment columns are `None` / empty and the
resolver omits polymer attachments. Their generic `leaving_atom_ids` are still
retained for declared covalent reactions.

## Building the dataset

`build.py` is the offline converter: wwPDB `components.cif` (optionally
gzipped) → the Parquet rows above. Its parse follows the reference generator
`alphafold3/constants/converters/ccd_pickle_gen.py` (read the multi-data mmCIF,
split into one dict per `data_` block), and the field lowering mirrors
`alphafold3/structure/chemical_components.py` (`parse_atom_data` /
`parse_bond_data`). The mmCIF and Parquet dependencies are imported lazily by the
build entry points.

`build.py` is a build-time tool only — it is never imported during modeling and
never runs during tensor compute.
