# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

import re
from html.parser import HTMLParser

import markdown
import pytest
from yaml import safe_load

from docs.gen_pages import ROOT, generated_model_pages, rewrite_readme_links


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.targets.update(value for key, value in attrs if key == "href")


def _links(text):
    links = _Links()
    links.feed(markdown.markdown(text, extensions=["tables", "fenced_code"]))
    return links.targets


class TestReadmeLinks:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("license.md", "../about/license.md"),
            ("license.md#conditions", "../about/license.md#conditions"),
            (
                "https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md",
                "../about/license.md",
            ),
            ("README.md", "../index.md"),
            ("ofoldx/chemistry", "../chemistry/index.md"),
            ("ofoldx/msa", "../msa/index.md"),
            ("ofoldx/templates", "../templates/index.md"),
            ("ofoldx/reference", "../reference-data/index.md"),
            ("docs", "https://github.com/OTeam-AI4S/OFoldX/tree/main/docs"),
            ("pyproject.toml", "https://github.com/OTeam-AI4S/OFoldX/blob/main/pyproject.toml"),
            ("https://example.org/guide", "https://example.org/guide"),
            ("#installation", "#installation"),
        ],
    )
    def test_readme_links(self, target, expected):
        result = rewrite_readme_links(f"[Guide]({target})", readme=ROOT / "README.md", page="guide/index.md")
        assert result == f"[Guide]({expected})"

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("license.md", "../about/license.md"),
            ("license-faq.md", "../about/license-faq.md"),
            ("license.md#conditions", "../about/license.md#conditions"),
            ("license-faq.md#weights", "../about/license-faq.md#weights"),
        ],
    )
    def test_model_card_legal_links(self, target, expected):
        readme = ROOT / "ofoldx" / "models" / "proteinmpnn" / "README.md"
        result = rewrite_readme_links(f"[Legal]({target})", readme=readme, page="models/proteinmpnn.md")
        assert result == f"[Legal]({expected})"


class TestModelDocumentation:
    def test_model_card_links(self):
        readme = ROOT / "ofoldx" / "models" / "proteinmpnn" / "README.md"
        source = readme.read_text()
        shared = {
            "https://github.com/OTeam-AI4S/OFoldX/blob/main/ofoldx/pipelines/README.md",
            "https://ofoldx.valtech.ai/models/proteinmpnn/"
            "#ofoldx.models.proteinmpnn.ProteinMpnnDesignAndPackPipeline",
        }
        assert shared | {"https://github.com/OTeam-AI4S/OFoldX/blob/main/license.md"} <= _links(source)

    def test_readme_api_references_render_as_links(self):
        for readme in (ROOT / "ofoldx").glob("**/README.md"):
            text = readme.read_text()
            references = set(re.findall(r"\]\[(ofoldx\.[^\]]+)\]", text))
            if not references:
                continue
            anchors = {link.partition("#")[2] for link in _links(text)}
            assert references <= anchors, readme

    def test_generated_page_resolves_api_references_with_mkdocs(self):
        readme = ROOT / "ofoldx" / "models" / "proteinmpnn" / "README.md"
        reference = "ofoldx.models.proteinmpnn.ProteinMpnnDesignAndPackPipeline"
        page = rewrite_readme_links(readme.read_text(), readme=readme, page="models/proteinmpnn.md")
        assert f"][{reference}]" in page
        assert not re.search(rf"(?m)^\[{re.escape(reference)}\]:", page)

    @pytest.mark.parametrize(
        ("family", "public_members"),
        [
            (
                "proteinmpnn",
                {"ProteinMpnnModel", "ProteinMpnnProcessor", "ProteinMpnnPipeline", "ProteinMpnnDesignAndPackPipeline"},
            ),
            ("boltzgen", {"BoltzGenPipeline", "diffusion_loss"}),
            ("membranempnn", {"MembraneMpnnPipeline", "ProteinMpnnInput", "ProteinMpnnOutput"}),
            ("oinvfold", {"OInvFoldPolymerPipeline", "OInvFoldLigandPipeline", "OInvFoldLoss"}),
            ("protenix", {"ProtenixPipeline", "ProtenixTfgGuidance"}),
        ],
    )
    def test_model_page_documents_user_interfaces(self, family, public_members):
        page = dict(generated_model_pages())[f"models/{family}.md"]
        directive = page.rsplit(f"::: ofoldx.models.{family}\n", 1)[1]
        members = safe_load(directive)["options"]["members"]
        assert public_members <= set(members)
        assert not {"MODELS", "PIPELINES"}.intersection(members)
        assert "ProteinMpnnCaFeatureEncoder" not in members
