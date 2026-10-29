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
Build a versioned ``ofoldx.ccd/7`` dataset from a local or downloaded wwPDB CCD.

Every block is decoded by the same ``chem_comp*`` codec rows that read embedded definitions
(``ofoldx.io.mmcif.decode.decode_components``), defined once (``define_component``), its stereo labels checked against
its own conformer and InChI descriptor (``arbitrate_stereo``: the configuration against an existing
descriptor, no proof of the 3-D conformer), and defined again with the arbitration. The counts keep the keys of the
shipped provenance and add the port, finding, cap and conformer counts. Every count is a sum of per-component
counts, and the sidecar JSON lists them (``by_component``), so a rebuild that moves any count names its components; it
also names the atoms behind the arbitration counts and records the source digest and the build tool (the artifact holds
only its schema, content version and payload digest).

Usage::

    python -m ofoldx.reference.ccd build --source components.cif --content-version YOUR_CCD_VERSION --output OUT
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from os import PathLike
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from ...chemistry.definitions import (
    ARTIFACT_SCHEMA,
    ArtifactProvenance,
    ChemicalComponent,
    DefinitionRows,
    StereoBasis,
    component_findings,
    define_component,
    write_artifact,
)
from ...chemistry.fragments import TERMINAL_CAPS, PortCondition, PortStatus, ports
from ...chemistry.rdkit import arbitrate_stereo
from ...data.molecular import DefinitionScope, ReferenceSource
from ...errors import OFoldXError
from ...io.cif import read_cif
from ...io.mmcif.decode import decode_components
from .hub import ARTIFACT_FILENAME, PROVENANCE_FILENAME, validate


@dataclass
class Build:
    r"""The archive definitions of one source, their counts (the sum of ``by_component``), and the atoms, bonds and
    caps named behind the arbitration and cap counts."""

    components: list[ChemicalComponent] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    by_component: dict[str, dict[str, int]] = field(default_factory=dict)
    named: dict[str, list] = field(default_factory=dict)
    defects: list[str] = field(default_factory=list)

    def name(self, key: str, *entry: str) -> None:
        self.named.setdefault(key, []).append(list(entry) if len(entry) > 1 else entry[0])


def build(blocks: Iterable[tuple[str, DefinitionRows]]) -> Build:
    r"""
    Define every component of the decoded blocks. A block must hold exactly one component and no id may occur twice;
    a violation, or statements that cannot form one definition, is an input defect named in ``Build.defects`` (the
    caller writes nothing then).
    """

    result, seen = Build(), set()
    for block, rows in blocks:
        if len(rows.components) != 1:
            result.defects.append(f"block {block}: {len(rows.components)} components ({', '.join(rows.components)})")
            continue
        ((component_id, items),) = rows.components.items()
        if component_id in seen:
            result.defects.append(f"component {component_id} occurs in more than one block (again in {block})")
            continue
        seen.add(component_id)
        try:
            first = define_component(items, scope=DefinitionScope.ARCHIVE)
            arbitration = arbitrate_stereo(first, items.descriptors)
            component = define_component(items, scope=DefinitionScope.ARCHIVE, stereo=arbitration)
        except OFoldXError as error:
            result.defects.append(f"component {component_id}: {error}")
            continue
        result.components.append(component)
        counts = _count(result, component, arbitration, first.reference_source)
        result.by_component[component_id] = {key: value for key, value in sorted(counts.items()) if value}
        result.counts.update(counts)
    return result


def _count(result: Build, c: ChemicalComponent, arbitration, conformer: ReferenceSource) -> Counter:
    r"""The counts of one component (every key of the totals), naming its arbitrated atoms and bonds and its cap."""
    counts, cid = Counter(components=1), c.component_id
    labelled = [atom for atom, label in enumerate(c.cip_declared) if label in ("R", "S")]
    counts["labels_stated"] += len(labelled)
    counts["labels_non_tetrahedral"] += len(set(labelled) - set(c.stereo_centre.tolist()))
    stated = c.stereo_parity != 0
    counts["parities"] += int(stated.sum())
    counts["parity_conformer"] += int((stated & (c.stereo_basis == StereoBasis.CONFORMER)).sum())
    counts["parity_descriptor_verified"] += int((c.stereo_basis == StereoBasis.DESCRIPTOR_VERIFIED).sum())
    counts["components_with_parity"] += int(stated.any())
    counts["components_without_parity"] += int(not stated.any())
    counts["atoms_inverted_parity"] += len(arbitration.inverted)
    counts["atoms_refused_parity"] += len(arbitration.refused)
    counts["components_refused_parity"] += bool(arbitration.refused)
    for atom in arbitration.inverted:
        result.name("inverted_parity_atoms", cid, atom)
    for atom in arbitration.refused:
        result.name("refused_parity_atoms", cid, atom)
    for _atom, reason in arbitration.unverified:
        counts[f"labels_unverified.{reason}"] += 1
    if c.reference_source == ReferenceSource.CONTRADICTED:
        counts["components_reference_coordinates_contradicted"] += 1
        result.name("reference_coordinates_contradicted", cid)
    counts[f"conformer.{conformer.name.lower()}"] += 1  # the coherent conformer chosen, before arbitration
    counts["bond_labels_stated"] += sum(label is not None for label in c.bond_stereo_declared)
    counts["bond_configurations"] += int((c.bond_stereo_value != 0).sum())
    counts["bond_labels_disagreeing"] += len(arbitration.bonds_disagreeing)
    for pair in arbitration.bonds_disagreeing:
        result.name("bond_labels_disagreeing", cid, *pair)
    kind = c.chem_comp_type.strip().upper() if c.chem_comp_type is not None else "?"
    record = ports(c)
    for side, status, condition in zip(("previous", "next"), record.status, record.condition):
        key = f"{side}.{PortStatus(status).name}"
        if condition:
            key += f".{PortCondition(condition).name}"
        counts[f"port.{key}"] += 1
        counts[f"port_by_type.{kind}.{key}"] += 1
        if status not in (PortStatus.VERIFIED, PortStatus.NO_SIDE, PortStatus.NO_RULE):
            result.name(f"port.{key}", cid)
    for code in sorted({finding for finding, _ in component_findings(c)}):
        counts[f"finding.{code}"] += 1
        if code in ("leaving_flag_unusable", "two_fragment_port", "port_atom_absent", "chem_comp_type_unlisted"):
            result.name(f"finding.{code}", cid)
    if (cap := TERMINAL_CAPS.get(cid)) is not None:
        present = cap.atom in c.atom_names
        counts[f"caps.{cid}.atom_present"] = int(present)
        result.name("caps", cid, cap.atom, "present" if present else "absent")
    return counts


def _build_tool() -> str:
    """Identify the installed builder without requiring a Git checkout or executable."""
    try:
        release = version("ofoldx")
    except PackageNotFoundError:
        release = "uninstalled"
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return f"ofoldx.reference.ccd.build@{release}:sha256:{digest}"


def build_artifact(
    source: str | PathLike[str],
    output: str | PathLike[str],
    *,
    content_version: str,
) -> Path:
    r"""Build a CCD dataset directory from a source dictionary.

    Args:
        source: Local CIF or gzipped CIF path, or an HTTP(S) URL.
        output: New dataset directory. Its parent directories are created as needed.
        content_version: Caller-selected identity of the source snapshot. The build
            also records a SHA256 digest of the source bytes.

    Returns:
        The `components.safetensors` path inside the completed dataset directory.
        Its provenance sidecar and README card are saved beside it.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: The content version is empty, the source has no components, or
            component construction reports defects.

    Notes:
        Construction and validation use a temporary directory. The destination
        appears after the artifact, provenance, and card are complete and the artifact
        passes validation.
    """
    if not content_version.strip():
        raise ValueError("CCD content_version must be nonempty")
    destination = Path(output).expanduser().absolute()
    if destination.exists():
        raise FileExistsError(f"CCD output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference = str(source)
    remote = isinstance(source, str) and urlparse(source).scheme in {"http", "https"}
    with tempfile.TemporaryDirectory(prefix=".ofoldx-ccd-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        if remote:
            name = Path(urlparse(reference).path).name or "components.cif"
            download_directory = staging / "source"
            download_directory.mkdir()
            local_source = download_directory / name
            with urlopen(reference, timeout=60) as response, local_source.open("wb") as handle:
                resolved_source = response.geturl()
                shutil.copyfileobj(response, handle)
        else:
            local_source = Path(source).expanduser().resolve()
            reference = str(local_source)
        with local_source.open("rb") as handle:
            source_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        cif = read_cif(local_source)
        result = build((block.name, decode_components(block.categories, block=block.name)) for block in cif.blocks)
        if result.defects:
            raise ValueError("CCD build failed:\n" + "\n".join(result.defects))
        if not result.components:
            raise ValueError("CCD source contains no component definitions")
        dataset = staging / "dataset"
        dataset.mkdir()
        artifact = dataset / ARTIFACT_FILENAME
        write_artifact(result.components, artifact, ArtifactProvenance(ARTIFACT_SCHEMA, content_version, ""))
        with artifact.open("rb") as handle:
            artifact_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        record = {
            "artifact": ARTIFACT_FILENAME,
            "sha256": artifact_sha256,
            "source": reference,
            "source_sha256": source_sha256,
            "content_version": content_version,
            "build_tool": _build_tool(),
            "counts": dict(sorted(result.counts.items())),
            "named": {key: sorted(values) for key, values in sorted(result.named.items())},
            "by_component": dict(sorted(result.by_component.items())),
        }
        if remote:
            record["resolved_source"] = resolved_source
        (dataset / PROVENANCE_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (dataset / "README.md").write_text(
            "---\ntask_categories:\n  - other\npretty_name: OFoldX Chemical Component Dictionary\n---\n\n"
            "# OFoldX Chemical Component Dictionary\n\n"
            f"Content version: `{content_version}`. Format: `{ARTIFACT_SCHEMA}`.\n\n"
            "Derived from the wwPDB Chemical Component Dictionary; upstream data usage terms and citation: "
            "https://www.wwpdb.org/data/ccd .\n\n"
            f"Source: {reference}\n\nSource SHA256: `{source_sha256}`.\n\n"
            f"Artifact SHA256: `{artifact_sha256}`.\n\n"
            f"Builder: `{record['build_tool']}`. Components: {len(result.components)}.\n\n"
            f"`{ARTIFACT_FILENAME}` contains the canonical component definitions. "
            f"`{PROVENANCE_FILENAME}` records source identity and build findings. "
            "Definitions preserve unknown and contradicted facts; a successful build does not make all chemistry "
            "fully determined.\n\n"
            "OFoldX uses the cached `oteam/ccd` dataset by default. Download a chosen revision with "
            "`python -m ofoldx.reference.ccd download --repo-id oteam/ccd --revision COMMIT`, then set "
            "`OFOLDX_CCD_ARTIFACT` to the printed path.\n",
            encoding="utf-8",
        )
        validate(dataset)
        dataset.replace(destination)
    return destination / ARTIFACT_FILENAME
