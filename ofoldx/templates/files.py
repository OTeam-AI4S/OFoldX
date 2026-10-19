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
Native template artifacts: prepare template coordinates once and load them for reuse.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from ..errors import SourceRef, fail
from .alignment import Template, TemplateAlignment
from .coordinates import ChainCoordinates, PolymerCoordinates

#: The native artifact schema; another is ``InvalidRequestError(native_schema_version)``.
_SCHEMA = "ofoldx.templates/3"
_ARRAYS = ("token_class", "encoding", "atom_name", "component_index", "coordinates", "coordinate_mask")
_FACTS = (
    "polymer_type",
    "component_ids",
    "atom_symbols",
    "parent_component_ids",
)


def save_templates(templates: Sequence[Template], path: str | os.PathLike[str]) -> None:
    r"""
    Save templates as one safetensors artifact: per template its chains (arrays as columns at their stored precision,
    facts in the header) and its alignments in order.

    Args:
        templates: Template evidence to retain in order.
        path: Output native template artifact path.
    """

    arrays: dict[str, NDArray] = {}
    chains: list[list[dict[str, object]]] = []
    alignments = []
    for index, template in enumerate(templates):
        chains.append([])
        for position, (name, chain) in enumerate(template.coordinates.chains.items()):
            chains[-1].append({"name": name} | {fact: getattr(chain, fact) for fact in _FACTS})
            arrays |= {f"template.{index}.chain.{position}.{field}": getattr(chain, field) for field in _ARRAYS}
        alignments.append(
            [
                (a.target_entity_id, a.target_instance_id, a.template_instance_id, a.residue_pairs)
                for a in template.alignments
            ]
        )
    metadata = {"schema": _SCHEMA, "chains": json.dumps(chains), "alignments": json.dumps(alignments)}
    save_file({key: np.ascontiguousarray(value) for key, value in arrays.items()}, os.fspath(path), metadata=metadata)


def load_templates(path: str | os.PathLike[str]) -> tuple[Template, ...]:
    r"""
    The templates saved by ``save_templates``.

    Args:
        path: Native template artifact written by ``save_templates``.

    Returns:
        The ordered templates with their chain snapshots and target alignments restored.

    Raises:
        InvalidRequestError: ``native_schema_version`` for an artifact of another schema; ``value_invalid`` for a
            value that violates an invariant.
        ContradictionError: ``native_corrupt`` for recognised damage, the field named: a container that does not
            decode (its header, or a tensor's dtype, shape or offsets, with the decoder's reason), a header entry
            missing, no JSON or not of its type, or a tensor missing. An I/O error propagates as it is.
    """

    path = os.fspath(path)
    try:
        artifact = safe_open(path, framework="np")
    except SafetensorError as error:  # the container's decode; an I/O error is an OSError and propagates
        _corrupt(path, "header", reason=str(error))
    with artifact:
        metadata = artifact.metadata() or {}
        if metadata.get("schema") != _SCHEMA:
            fail("native_schema_version", SourceRef(path, 0, item="schema"), schema=metadata.get("schema"))
        arrays = {key: artifact.get_tensor(key) for key in artifact.keys()}
    stored = _header(path, metadata, "chains", lambda v: _lists(v, _chain))
    entries = zip(stored, _header(path, metadata, "alignments", lambda v: _lists(v, _alignment, len(stored))))
    templates = []
    for index, (chains, alignments) in enumerate(entries):
        coordinates = {}
        for position, chain in enumerate(chains):
            coordinates[chain["name"]] = ChainCoordinates(
                polymer_type=chain["polymer_type"],
                component_ids=tuple(chain["component_ids"]),
                atom_symbols=tuple(chain["atom_symbols"]),
                parent_component_ids=tuple(map(tuple, chain["parent_component_ids"])),
                **{field: _tensor(path, arrays, f"template.{index}.chain.{position}.{field}") for field in _ARRAYS},
            )
        templates.append(
            Template(
                PolymerCoordinates(MappingProxyType(coordinates)),
                tuple(
                    TemplateAlignment(entity, instance, source, tuple(map(tuple, pairs)))
                    for entity, instance, source, pairs in alignments
                ),
            )
        )
    return tuple(templates)


def _corrupt(path: str, item: str, **detail: object) -> NoReturn:
    fail("native_corrupt", SourceRef(path, 0, item=item), **detail)


def _header(path: str, metadata: Mapping[str, str], key: str, valid: Callable[[object], bool]) -> Any:
    r"""The JSON value of header entry ``key``: ``native_corrupt`` (item ``key``) when it is missing, no JSON (the
    decoder's reason named) or not ``valid``."""
    if key not in metadata:
        _corrupt(path, key)
    try:
        value = json.loads(metadata[key])
    except json.JSONDecodeError as error:
        _corrupt(path, key, reason=str(error))
    if not valid(value):
        _corrupt(path, key)
    return value


def _tensor(path: str, arrays: Mapping[str, NDArray], name: str) -> NDArray:
    r"""The tensor ``name``: ``native_corrupt`` (item ``name``) when it is missing."""
    if name not in arrays:
        _corrupt(path, name)
    return arrays[name]


def _lists(value: object, valid: Callable[[object], bool], length: int | None = None) -> bool:
    r"""Whether ``value`` is a list (of ``length`` entries when given) of lists of ``valid`` entries."""
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(isinstance(entries, list) and all(map(valid, entries)) for entries in value)
    )


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _chain(value: object) -> bool:
    r"""Whether ``value`` is one stored chain: its name and facts."""
    return (
        isinstance(value, dict)
        and set(value) == {"name", *_FACTS}
        and isinstance(value["name"], str)
        and (value["polymer_type"] is None or isinstance(value["polymer_type"], str))
        and _strings(value["component_ids"])
        and _strings(value["atom_symbols"])
        and isinstance(value["parent_component_ids"], list)
        and all(map(_strings, value["parent_component_ids"]))
    )


def _alignment(value: object) -> bool:
    r"""Whether ``value`` is one stored alignment: target entity, target instance (or null), template chain and
    position pairs."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and isinstance(value[0], str)
        and (value[1] is None or isinstance(value[1], str))
        and isinstance(value[2], str)
        and isinstance(value[3], list)
        and all(isinstance(pair, list) and len(pair) == 2 for pair in value[3])
    )
