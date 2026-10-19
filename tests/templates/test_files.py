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

"""Behaviour of ofoldx.templates.files: the native template artifact."""

from __future__ import annotations

import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from ofoldx.errors import ContradictionError
from ofoldx.templates import TemplateAlignment, load_templates, save_templates


def rewritten(edit):
    r"""An edit of a saved artifact: ``edit`` changes its header entries and tensors, which are then saved again."""

    def apply(path) -> None:
        with safe_open(str(path), framework="np") as artifact:
            metadata, arrays = artifact.metadata(), {key: artifact.get_tensor(key) for key in artifact.keys()}
        edit(metadata, arrays)
        save_file(arrays, path, metadata=metadata)

    return apply


ARTIFACT_EDITS = {
    "truncated": (lambda path: path.write_bytes(path.read_bytes()[:-1]), "header"),
    "not-json": (rewritten(lambda metadata, _: metadata.update(chains="[")), "chains"),
    "chain-facts-missing": (rewritten(lambda metadata, _: metadata.update(chains='[[{"name": "X"}]]')), "chains"),
    "entry-missing": (rewritten(lambda metadata, _: metadata.pop("alignments")), "alignments"),
    "tensor-missing": (
        rewritten(lambda _, arrays: arrays.pop("template.0.chain.0.coordinates")),
        "template.0.chain.0.coordinates",
    ),
}


class TestTemplateArtifact:
    def test_round_trip(self, tmp_path, template):
        alignments = (TemplateAlignment("1", None, "X", ((0, 0), (2, 1))), TemplateAlignment("1", "B", "X", ((1, 1),)))
        saved = template(*alignments)
        save_templates((saved, saved), tmp_path / "x.templates.safetensors")
        loaded = load_templates(tmp_path / "x.templates.safetensors")
        assert len(loaded) == 2 and loaded[1].alignments == alignments
        chain, original = loaded[0].coordinates.chains["X"], saved.coordinates.chains["X"]
        assert chain.component_ids == original.component_ids
        assert chain.parent_component_ids == original.parent_component_ids
        assert chain.gather([0, 1], ["CA", "CB"])[0].tolist() == original.gather([0, 1], ["CA", "CB"])[0].tolist()
        assert chain.token_class.tolist() == original.token_class.tolist()

    @pytest.mark.parametrize(("edit", "item"), ARTIFACT_EDITS.values(), ids=ARTIFACT_EDITS)
    def test_corrupt_refused(self, tmp_path, template, edit, item):
        path = tmp_path / "x.templates.safetensors"
        save_templates((template(TemplateAlignment("1", None, "X", ((0, 0),))),), path)
        edit(path)
        with pytest.raises(ContradictionError) as caught:
            load_templates(path)
        assert (caught.value.code, caught.value.refs[0].item) == ("native_corrupt", item)
