# Multiple Sequence Alignments

`ofoldx.msa` reads existing alignments, retains paired and unpaired sequence pools, and builds row plans for model
features. It also stores those plans as native artifacts. Sequence search and alignment generation run outside this
module; supply the resulting A3M or aligned-FASTA files.

## Usage

Read an A3M file whose first biological record is the ungapped query. `entity_id` identifies the target entity that
the query describes:

```python
from ofoldx.msa import read_msa

msa = read_msa("query.a3m", entity_id="protein", max_rows=256)
print(msa.unpaired_sequences[0])
print(msa.unpaired_deletion_counts.shape)
```

`read_msa` retains the file's rows in the unpaired pool and uses the query as the paired pool. A3M lowercase
insertions are removed from the aligned sequence and retained as deletion counts on the query axis. Aligned FASTA
can contain query-gap columns, which are projected away when building query-width rows. The reader also supports
compressed A3M files; see its API reference for formats and validation.

## Pools, Plans, and Model Inputs

| Value           | Meaning                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------- |
| `MSA`           | One entity's query, paired/unpaired rows, deletion counts, and available paired-row headers |
| `MsaCollection` | A row plan over pools, with optional bindings to target instances                           |
| `MsaRows`       | A plan projected onto a model's token layout                                                |

An entity and an instance are different identifiers: one entity may occur in several physical chain instances.
For pipelines whose concrete signature accepts `msa`, provide the documented alignment path, native collection,
or mapping from instance IDs to `MSA` pools. The query must agree with the target's sequence classes.

`pair_msas` constructs a plan for polymer instances represented in a token layout. It pairs protein rows using
recognized species identifiers in paired-pool headers and retains the original pools. Its `max_rows` limits the
resulting plan; a reader's `max_rows` limits source rows read. `msa_rows` binds a plan to canonical components and
spreads it onto tokens. Family processors own when these operations occur relative to cropping and feature
construction.

Use `save_msa_collection` and `load_msa_collection` to retain an explicit row plan, its pools, deletion counts, and
instance bindings together.

## Indexed Collections

Set `OFOLDX_CCD_ARTIFACT` to an existing CCD artifact before reading a structure.

`IndexedMsaLoader` reads existing protein/RNA alignment directories lazily:

```python
from ofoldx.io import read_structure
from ofoldx.msa import IndexedMsaLoader

structure, report = read_structure("complex.cif")
loader = IndexedMsaLoader(index="sequence_to_directory.json", root="msa")
pools = loader(structure)
print(tuple(pools))
```

The JSON index maps query sequences to directory names under `root`. Protein directories contain `pairing.a3m`
and `non_pairing.a3m`; optional RNA indices use the RNA directory layout documented by the loader. The returned
mapping is keyed by instance ID, and a pool's files are read on access. Instances without an index entry remain
absent from that mapping.

## Related Modules

- [Models](../models/README.md): each family's accepted MSA evidence and sampling controls.
- [Templates](../templates/README.md): aligned structural evidence.
- [Data](../data/README.md): target instances, sequence classes, and token layouts.
