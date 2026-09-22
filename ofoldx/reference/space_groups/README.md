# Space Groups

The bundled space-group table supplies crystallographic operations used when reading structures.

```python
from ofoldx.reference.space_groups import load_table

table = load_table()
```

To build a table, provide a CCP4-format `symop.lib` and named-operator evidence:

```shell
python -m ofoldx.reference.space_groups.build --symop symop.lib --evidence operator_evidence.json --output space_group_operators.json
```

The evidence JSON records observed operations by space-group setting and 1-based operator number. For example,
four observations of the identity operation for `P 1` are represented as:

```json
{
  "table": {
    "P 1": {
      "named": {
        "1": { "x,y,z": 4 }
      }
    }
  }
}
```

The builder compares the most frequently observed operation for each in-range number with the generated table;
it does not check every observation. It records the input digests and validation summary in the result. OFoldX
does not currently provide a script to collect this evidence. The inputs used to create the bundled table are not
included in the repository.
