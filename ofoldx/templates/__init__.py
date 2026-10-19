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

r"""
Structural templates: polymer-chain coordinates projected from a read structure, target alignments with their
resolution and placement frames, and the native template artifact.
"""

from .alignment import (
    ResolvedTemplateAlignment,
    Template,
    TemplateAlignment,
    resolve_template,
    template_alignment_frames,
)
from .coordinates import ChainCoordinates, PolymerCoordinates, chain_coordinates, template_atoms
from .files import load_templates, save_templates

__all__ = [
    "ChainCoordinates",
    "PolymerCoordinates",
    "ResolvedTemplateAlignment",
    "Template",
    "TemplateAlignment",
    "chain_coordinates",
    "load_templates",
    "resolve_template",
    "save_templates",
    "template_alignment_frames",
    "template_atoms",
]
