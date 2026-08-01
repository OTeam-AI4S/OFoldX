import numpy as np
import pytest
import torch

from ofoldx.data import MolecularSystem, PolymerSequence, Template, TemplateAlignment
from ofoldx.data.coordinates import ChainCoordinates, PolymerCoordinates
from ofoldx.models.rfdiffusion.processor import _fold_features


@pytest.mark.parametrize(
    "target_position,template_chain,template_position",
    [
        pytest.param(0, "X", 0, id="same-source"),
        pytest.param(1, "X", 1, id="different-position"),
        pytest.param(1, "Y", 0, id="different-chain"),
    ],
)
def test_template_overlap(target_position, template_chain, template_position):
    source = MolecularSystem.from_entities(
        {"query": PolymerSequence.from_one_letter("AA")},
        {"A": "query", "B": "query"},
    )
    chain = ChainCoordinates(
        polymer_type="protein",
        component_ids=("ALA", "ALA"),
        atom_names=("N", "CA", "C") * 2,
        component_index=np.array([0, 0, 0, 1, 1, 1]),
        coordinates=np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 0, 0], [4, 0, 0], [4, 1, 0]],
            dtype=np.float32,
        ),
        coordinate_mask=np.ones(6, dtype=bool),
    )
    coordinates = PolymerCoordinates({"X": chain, "Y": chain})
    broadcast = TemplateAlignment("query", None, "X", ((0, 0),))
    exact = TemplateAlignment("query", "A", template_chain, ((target_position, template_position),))
    template = Template(coordinates=coordinates, alignments=(broadcast, exact))
    # Distinct target positions can resolve to the same output row.
    source_to_rows = {("A", 0): (0,), ("A", 1): (0,), ("B", 0): (1,)}
    output_instances = ("A", "B")

    if (template_chain, template_position) != ("X", 0):
        with pytest.raises(ValueError, match="overlap on one output row"):
            _fold_features(template, source, source_to_rows, output_instances)
        return

    baseline = Template(coordinates=coordinates, alignments=(broadcast,))
    expected = _fold_features(baseline, source, source_to_rows, output_instances)
    actual = _fold_features(template, source, source_to_rows, output_instances)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
