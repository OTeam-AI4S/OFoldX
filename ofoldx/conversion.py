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
Save converted model repositories with their cards, licenses and source lineage.

Model-family READMEs supply the human model cards; public converter functions
provide the selected sources and options. This module combines them with the
saved Pipeline repository before an explicitly requested Hub upload. The model
configuration remains in model.json; conversion.json records source lineage.
See the model namespace README for the complete conversion and publication flow.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .hub import upload_repository

if TYPE_CHECKING:
    from .pipelines import Pipeline

SCHEMA = "ofoldx.conversion/1"


@dataclass(frozen=True)
class SourceCheckpoint:
    r"""
    A consumed checkpoint, source configuration or external repository reference.

    Attributes:
        path: Existing local checkpoint, shard index, source configuration or
            repository directory. None denotes an injected value or an external
            reference whose bytes are not part of this conversion's local input.
        reference: Known source identity, such as a Hub repository ID. A local
            filename does not establish an upstream URL or release identity.
        revision: Genuinely resolved source repository commit, when known.
            Requested tags belong in conversion options.
        members: Selected entries of a source archive.
        license_files: Source notices and attribution to retain beside the
            conversion. OFoldX's distributed artifacts carry the project AGPL
            license and FAQ.
    """

    path: str | PathLike[str] | None = None
    reference: str | None = None
    revision: str | None = None
    members: tuple[str, ...] = ()
    license_files: tuple[str | PathLike[str], ...] = ()


def _digest(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"size": path.stat().st_size, "sha256": digest}


def _source_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not any(part in {".git", "__pycache__"} for part in p.relative_to(path).parts)
        )
    paths = [path]
    if path.name.endswith(".safetensors.index.json"):
        index = json.loads(path.read_text(encoding="utf-8"))
        for name in sorted(set(index["weight_map"].values())):
            relative = Path(name)
            if not name or relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError(f"source shard must be repository-relative: {name!r}")
            shard = path.parent / relative
            if shard not in paths:
                paths.append(shard)
    return paths


def _source_record(
    source: SourceCheckpoint | str | PathLike[str] | None,
    cache: dict[Path, list[dict[str, object]]],
) -> dict[str, object]:
    source = source if isinstance(source, SourceCheckpoint) else SourceCheckpoint(source)
    record: dict[str, object] = {
        "reference": source.reference,
        "revision": source.revision,
        "members": list(source.members),
        "files": [],
    }
    if source.path is None:
        record["kind"] = "reference" if source.reference else "in_memory"
        return record
    path = Path(source.path).expanduser().absolute()
    # Resolve only to validate existence: names and shard paths belong to the
    # snapshot, not to the content-addressed target of a file symlink.
    path.resolve(strict=True)
    record["kind"] = "directory" if path.is_dir() else "file"
    record["reference"] = source.reference or str(path)
    if path not in cache:
        base = path if path.is_dir() else path.parent
        cache[path] = [{"name": p.relative_to(base).as_posix(), **_digest(p)} for p in _source_files(path)]
    record["files"] = cache[path]
    return record


def _component_records(pipeline: Pipeline, cache: dict[Path, list[dict[str, object]]]) -> dict[str, object]:
    result: dict[str, object] = {}

    def visit(component, prefix: str) -> None:
        source = getattr(component, "source", None)
        model = getattr(component, "model", None)
        entry: dict[str, object] = {"class": f"{type(component).__module__}.{type(component).__qualname__}"}
        if model is not None:
            entry["architecture"] = f"{type(model).__module__}.{type(model).__qualname__}"
            # These are references in the model configuration, not an assertion
            # that a requested tag was resolved to a particular source commit.
            config = dict(model.config)
            entry["external_references"] = {
                key: value
                for key, value in config.items()
                if key.endswith(("_repository", "_revision")) and value is not None
            }
        if source is not None:
            entry["source"] = _source_record(
                SourceCheckpoint(source.path, reference=source.repository, revision=source.commit), cache
            )
            entry["subdirectory"] = source.subdirectory
            lineage = source.path / "conversion.json"
            if lineage.is_file():
                entry["conversion"] = {"name": "conversion.json", **_digest(lineage)}
        result[prefix or "."] = entry
        children = getattr(component, "repository_components", None)
        entries = children() if children is not None else getattr(component, "components", {}).items()
        try:
            for role, child in entries:
                visit(child, f"{prefix}/{role}".lstrip("/"))
        finally:
            close = getattr(entries, "close", None)
            if close is not None:
                close()

    visit(pipeline, "")
    return result


def save_repository(
    pipeline: Pipeline,
    directory: str | PathLike[str],
    *,
    sources: Mapping[str, SourceCheckpoint | str | PathLike[str] | None],
    converter: str,
    options: Mapping[str, object] | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    private: bool = False,
) -> None:
    r"""
    Save a conversion, its family card, AGPL documents and explicit source lineage.

    Source checksums are computed once per distinct path within this save. Shard
    indexes include every referenced source shard. Local saving and document
    generation finish before any explicitly requested Hub upload. The model card
    is copied from the family README; existing licenses are preserved.
    This records provenance and makes no numerical or scientific parity claim.

    Args:
        pipeline: Concrete paired or composite workflow to save.
        directory: Local destination repository.
        sources: Consumed inputs keyed by their scientific or component role.
            Paths are local and must exist. SourceCheckpoint can additionally
            name a known external reference or an injected value.
        converter: Canonical module-qualified converter function name.
        options: JSON-serializable choices that identify the selected conversion.
        repo_id: Optional destination Hub model repository. None saves locally.
        token: Optional Hub token used only for an explicitly requested upload.
        private: Whether a newly created destination Hub repository is private.

    Raises:
        FileNotFoundError: A declared local source does not exist.
        TypeError: Conversion options cannot be represented as JSON.
        ValueError: A source shard name escapes its repository, or options
            contain a non-finite JSON value.
    """
    cache: dict[Path, list[dict[str, object]]] = {}
    source_records = {role: _source_record(source, cache) for role, source in sources.items()}
    try:
        release = version("ofoldx")
    except PackageNotFoundError:
        release = None
    module_name = converter.rpartition(".")[0]
    module = sys.modules.get(module_name)
    main = sys.modules.get("__main__")
    main_spec = getattr(main, "__spec__", None)
    if module_name == "__main__" and main_spec is not None:
        module_name = main_spec.name
        converter = f"{module_name}.{converter.rpartition('.')[2]}"
        module = main
    elif module is None and main_spec is not None and main_spec.name == module_name:
        module = main
    family = module_name.split(".")[2]
    card = files("ofoldx").joinpath("models", family, "README.md").read_bytes()
    module_file = getattr(module, "__file__", None)
    identity: dict[str, object] = {"callable": converter, "ofoldx_version": release}
    if module_file is not None and Path(module_file).is_file():
        identity["module_sha256"] = _digest(Path(module_file))["sha256"]
    identity["writer_sha256"] = _digest(Path(__file__))["sha256"]
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "license": "AGPL-3.0-or-later",
        "converter": identity,
        "options": dict(options or {}),
        "sources": source_records,
        "components": _component_records(pipeline, cache),
        "task": pipeline.task,
    }
    # Reject non-JSON options before changing an existing destination.
    json.dumps(record, allow_nan=False)
    destination = Path(directory).expanduser().resolve()
    pipeline.save(destination)
    marker = json.loads((destination / "pipeline.json").read_text(encoding="utf-8"))
    record["pipeline"] = marker["pipeline"]
    record["initial_artifacts"] = {
        p.relative_to(destination).as_posix(): _digest(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file()
        and (
            p.suffix == ".safetensors"
            or p.name in {"model.json", "processor.json", "pipeline.json", "model.safetensors.index.json"}
        )
    }
    resource = files("ofoldx").joinpath("licenses")
    for name in ("license.md", "license-faq.md"):
        target = destination / name
        if not target.exists():
            target.write_bytes(resource.joinpath(name).read_bytes())
    for role, source in sources.items():
        if not isinstance(source, SourceCheckpoint):
            continue
        for notice in source.license_files:
            source_notice = Path(notice).expanduser()
            safe_role = re.sub(r"[^A-Za-z0-9_.-]", "_", role)
            target = destination / "licenses" / safe_role / source_notice.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source_notice, target)
    (destination / "README.md").write_bytes(card)
    (destination / "conversion.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    if repo_id is not None:
        upload_repository(destination, repo_id, token=token, private=private)
