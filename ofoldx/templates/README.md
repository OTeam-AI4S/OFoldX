# Structural Templates

`ofoldx.templates` represents polymer coordinates aligned to a molecular target. It projects template chains from a
read structure, binds caller-supplied alignments to target instances, and stores the evidence as a native artifact.

## Usage

Read a template structure whose author chain `A` contains at
least two polymer positions. This example aligns its first two positions to the first two positions of target
entity `query`:

```python
from ofoldx.io import read_structure
from ofoldx.templates import Template, TemplateAlignment, chain_coordinates, load_templates, save_templates

structure, report = read_structure("template.cif")
coordinates = chain_coordinates(structure, ["A"])
alignment = TemplateAlignment(
    target_entity_id="query",
    target_instance_id=None,
    template_instance_id="A",
    residue_pairs=((0, 0), (1, 1)),
)
template = Template(coordinates=coordinates, alignments=(alignment,))
save_templates((template,), "templates.safetensors")
restored = load_templates("templates.safetensors")
```

Use the target's actual entity ID in the alignment. Pipelines whose concrete signature exposes `templates` accept
the documented `Template` sequence or native artifact path.

## Coordinates and Alignments

- `chain_coordinates` selects polymer chains by author chain name by default; `namespace="label"` selects label
  asym IDs. Ambiguous chain names, including names shared by several physical copies, require a more specific
  input selection.
- `PolymerCoordinates` groups chains that share one coordinate frame. Each `ChainCoordinates` snapshot retains
  component identities, sequence classes, atom names, coordinates, and coordinate availability.
- `TemplateAlignment.residue_pairs` contains zero-based `(target, template)` sequence positions, increasing on
  both axes. These positions are independent of author residue numbers and insertion codes.
- `target_instance_id=None` applies an alignment to every instance of its target entity. Set an instance ID to
  bind one physical occurrence. `resolve_template` checks those bindings and the family's accepted monomer types.

Coordinate masks retain unavailable atoms and positions. The template carrier describes aligned coordinates;
chemical topology remains in the source `Structure`. Alignment placement also determines which template ranges
can share a frame. The concrete validation and frame rules are documented on `Template`, `resolve_template`, and
`template_alignment_frames`.

Supply alignments obtained by your chosen alignment or template-search workflow. This module consumes that
evidence; model processors build their own template features from it.

## Related Modules

- [I/O](../io/README.md): read template structures, choose assemblies, and retain source reports.
- [MSA](../msa/README.md): sequence alignments and row plans.
- [Data](../data/README.md): canonical identities and coordinate evidence.
- [Models](../models/README.md): family-specific template inputs and capacity.
