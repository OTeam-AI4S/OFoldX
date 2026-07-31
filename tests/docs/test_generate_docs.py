import pytest

from docs.generate_docs import ROOT, rewrite_readme_links


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
            ("docs", "https://github.com/OTeam-AI4S/OFoldX/tree/main/docs"),
            ("pyproject.toml", "https://github.com/OTeam-AI4S/OFoldX/blob/main/pyproject.toml"),
            ("https://example.org/guide", "https://example.org/guide"),
            ("#installation", "#installation"),
        ],
    )
    def test_readme_links(self, target, expected):
        result = rewrite_readme_links(f"[Guide]({target})", readme=ROOT / "README.md", page="guide/index.md")
        assert result == f"[Guide]({expected})"
