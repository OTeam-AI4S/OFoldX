from pathlib import Path

import pytest
import torch

from ofoldx.data import Structure, Template, TemplateAlignment, load_templates, save_templates
from ofoldx.io import read_structure, read_template_coordinates, read_template_structure


@pytest.fixture
def cif_text():
    return (Path(__file__).parents[1] / "io" / "fixtures" / "template.cif").read_text(encoding="utf-8")


@pytest.fixture
def cif_path(tmp_path, cif_text):
    path = tmp_path / "template.cif"
    path.write_text(cif_text, encoding="utf-8")
    return path


class TestTemplateFeatures:
    @pytest.mark.parametrize("representation", ["structure", "coordinates"])
    @pytest.mark.parametrize("altloc", ["first", "highest_occupancy"])
    def test_protein_features(self, cif_path, tmp_path, altloc, representation):
        from ofoldx.models.protenix.processor import _template_features

        full = read_structure(cif_path, altloc=altloc, include_hydrogens=False)
        fast = (
            read_template_structure(cif_path, "A", altloc=altloc)
            if representation == "structure"
            else read_template_coordinates(cif_path, ["A"], altloc=altloc)["A"]
        )
        query = Structure.from_sequence("AGM")
        target_entity = query.system.entity_id(0)
        alignment = TemplateAlignment(target_entity, None, "A", ((0, 0), (1, 1), (2, 2)))
        tokens = [{"instance_index": 0, "monomer_index": i, "component_index": i} for i in range(3)]
        expected = _template_features((Template.from_structure(full, (alignment,)),), tokens, query.system)
        template = (
            Template.from_structure(fast, (alignment,))
            if representation == "structure"
            else Template(coordinates=fast, alignments=(alignment,))
        )
        actual = _template_features((template,), tokens, query.system)
        for key, tensor in expected.items():
            torch.testing.assert_close(getattr(actual, key), tensor, rtol=0, atol=0)
        path = tmp_path / "test.templates.safetensors"
        save_templates((template,), path)
        restored = load_templates(path)[0]
        assert restored.alignments == (alignment,)
        reread = _template_features((restored,), tokens, query.system)
        for key, tensor in actual.items():
            torch.testing.assert_close(getattr(reread, key), tensor, rtol=0, atol=0)
        assert (
            restored.coordinates.chains["A"].parent_component_ids
            == template.coordinates.chains["A"].parent_component_ids
        )

    def test_rna_features(self, cif_text):
        from ofoldx.io import parse_source, read_polymer_coordinates
        from ofoldx.models.alphafold3.processor import _template_features as af3_features
        from ofoldx.models.boltzgen.processor import _template_channels as boltzgen_features

        text = cif_text.replace("'polypeptide(L)'", "polyribonucleotide")
        text = text.replace("ALA", "A").replace("GLY", "C").replace("MSE", "U").replace("U MET", "U ?")
        lines = []
        names = {"N": ("P", "P"), "CA": ("C1'", "C"), "C": ("C4", "C"), "CB": ("C4'", "C"), "SE": ("N1", "N")}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) == 20 and fields[6] == "A" and fields[3] in names:
                fields[3], fields[2] = names[fields[3]]
                if fields[5] == "U" and fields[3] == "C4":
                    fields[3] = "C2"
                fields[18] = fields[3]
                line = " ".join(fields)
            lines.append(line)
        source = parse_source("\n".join(lines) + "\n", format="mmcif")
        view = source.view(asym_ids=frozenset({"A"}))
        coordinates = read_polymer_coordinates(view)
        chain = coordinates.chains["A"]
        assert chain.polymer_type == "rna" and chain.component_ids == ("A", "C", "U")
        assert not chain.gather([1], ["C1'", "C2"])[1].any()
        query = Structure.from_sequence("ACU", polymer_type="rna")
        full = read_structure(view, include_hydrogens=False)
        alignment = TemplateAlignment(query.system.entity_id(0), None, "A", ((0, 0), (1, 1), (2, 2)))
        tokens = [{"instance_index": 0, "monomer_index": i, "component_index": i} for i in range(3)]
        expected = Template.from_structure(full, (alignment,))
        actual = Template(coordinates=coordinates, alignments=(alignment,))
        boltzgen = boltzgen_features(query.system, (actual,), tokens, 3)
        for expected_features, actual_features in (
            (af3_features((expected,), tokens, query.system), af3_features((actual,), tokens, query.system)),
            (boltzgen_features(query.system, (expected,), tokens, 3), boltzgen),
        ):
            for name, value in expected_features.items():
                torch.testing.assert_close(actual_features[name], value, rtol=0, atol=0, equal_nan=True)
        assert boltzgen["template_alpha_carbon_positions"].tolist() == [[[1, 0, 0], [0, 0, 0], [4, 0, 0]]]
        assert boltzgen["template_beta_carbon_positions"].tolist() == [[[2, 0, 0], [0, 0, 0], [5, 0, 0]]]
        assert boltzgen["template_beta_carbon_mask"].tolist() == [[1, 0, 1]]
