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

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import tomllib
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType

import pytest

from ofoldx.chemistry.definitions import DefinitionRows, StereoBasis, component_findings, load_library
from ofoldx.chemistry.fragments import PortStatus, ports
from ofoldx.data.molecular import ReferenceSource
from ofoldx.errors import ContradictionError, InvalidRequestError
from ofoldx.reference.ccd import build_artifact, download, upload
from ofoldx.reference.ccd.build import build
from tests.chemistry.builders import SUBSET, make_items

ROOT = Path(__file__).resolve().parents[3]
IDS = tomllib.loads((SUBSET.parent / "MANIFEST.toml").read_text())["blocks"]
STEREO_COUNTS = (
    "components",
    "labels_stated",
    "parities",
    "parity_conformer",
    "parity_descriptor_verified",
    "atoms_inverted_parity",
    "atoms_refused_parity",
    "components_refused_parity",
    "components_reference_coordinates_contradicted",
    "components_with_parity",
    "components_without_parity",
)


def run(source: Path, output: Path) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "ofoldx.reference.ccd", "build", "--source", str(source)]
    command += ["--content-version", "subset", "--output", str(output)]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(command, capture_output=True, text=True, env=environment, check=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    output = tmp_path_factory.mktemp("artifact") / "subset.safetensors"
    result = run(SUBSET, output)
    assert result.returncode == 0, result.stderr
    artifact = output / "components.safetensors"
    return load_library(artifact), json.loads((output / "components.safetensors.provenance.json").read_text())


class TestBuild:
    def test_subset_matches_manifest(self):
        manifest = tomllib.loads((SUBSET.parent / "MANIFEST.toml").read_text())
        payload = gzip.decompress(SUBSET.read_bytes())
        headings = list(re.finditer(rb"(?m)^data_([^\r\n]+)\r?\n", payload))
        names = [match.group(1).decode("ascii") for match in headings]
        assert names == manifest["blocks"]
        for index, (name, match) in enumerate(zip(names, headings)):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(payload)
            actual = hashlib.sha256(payload[match.start() : end]).hexdigest()
            expected = manifest["fixture_block_sha256"].get(name, manifest["block_sha256"][name])
            assert actual == expected, name

    def test_counts_recount_artifact(self, built):
        library, sidecar = built
        counts = sidecar["counts"]
        components = [library.definition(x) for x in IDS]
        parity = [c.stereo_parity != 0 for c in components]
        recount = {
            "components": len(components),
            "labels_stated": sum(sum(label in ("R", "S") for label in c.cip_declared) for c in components),
            "parities": sum(int(p.sum()) for p in parity),
            "parity_descriptor_verified": sum(
                int((c.stereo_basis == StereoBasis.DESCRIPTOR_VERIFIED).sum()) for c in components
            ),
            "components_with_parity": sum(bool(p.any()) for p in parity),
            "components_reference_coordinates_contradicted": sum(
                c.reference_source == ReferenceSource.CONTRADICTED for c in components
            ),
            "atoms_refused_parity": sum(int((c.stereo_basis == StereoBasis.REFUSED).sum()) for c in components),
            "port.previous.VERIFIED": sum(ports(c).status[0] == PortStatus.VERIFIED for c in components),
        }
        assert set(STEREO_COUNTS) <= set(counts) and {key: counts[key] for key in recount} == recount

    def test_counts_sum_components(self, built):
        _, sidecar = built
        total = Counter()
        for counts in sidecar["by_component"].values():
            total.update(counts)
        assert sorted(sidecar["by_component"]) == sorted(IDS)
        assert dict(total) == {key: value for key, value in sidecar["counts"].items() if value}

    def test_sidecar_names_arbitrations(self, built):
        _, sidecar = built
        named = sidecar["named"]
        assert named["reference_coordinates_contradicted"] == ["C49"]
        assert named["inverted_parity_atoms"] == [["C49", "C1'"]]
        assert named["refused_parity_atoms"] == [["8D8", "C06"]]
        assert ["ACE", "C", "present"] in named["caps"] and ["NH2", "N", "present"] in named["caps"]

    def test_duplicate_block_refused(self, tmp_path):
        text = gzip.decompress(SUBSET.read_bytes()).decode()
        alanine = text[text.index("data_ALA\n") : text.index("data_GLY\n")]
        source = tmp_path / "twice.cif"
        source.write_text(alanine + alanine)
        result = run(source, tmp_path / "never.safetensors")
        assert result.returncode != 0 and "ALA" in result.stderr
        assert not (tmp_path / "never.safetensors").exists()

    def test_missing_cap_atom_counted(self):
        items = make_items({"id": "ACE", "type": "NON-POLYMER"}, atoms={"atom_id": ["CH3"], "type_symbol": ["C"]})
        result = build([("ACE", DefinitionRows(MappingProxyType({"ACE": items})))])
        assert "port_atom_absent" in dict(component_findings(result.components[0]))
        assert result.counts["caps.ACE.atom_present"] == 0

    def test_empty_source_refused_without_output(self, tmp_path):
        source = tmp_path / "empty.cif"
        source.touch()
        destination = tmp_path / "ccd"
        with pytest.raises(ValueError, match="no component"):
            build_artifact(source, destination, content_version="empty")
        assert not destination.exists()

    def test_complete_dataset_and_existing_destination(self, tmp_path):
        destination = tmp_path / "ccd"
        artifact = build_artifact(SUBSET, destination, content_version="subset")
        assert artifact == destination / "components.safetensors"
        assert (destination / "README.md").is_file()
        record = json.loads((destination / "components.safetensors.provenance.json").read_text())
        assert record["source_sha256"] == hashlib.sha256(SUBSET.read_bytes()).hexdigest()
        assert record["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert record["artifact"] == artifact.name
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        with pytest.raises(FileExistsError):
            build_artifact(SUBSET, destination, content_version="other")
        assert {p.name: p.read_bytes() for p in destination.iterdir()} == before

    def test_http_source(self, tmp_path):
        # A URL basename equal to our output staging directory is a valid source.
        (tmp_path / "dataset").write_bytes(gzip.decompress(SUBSET.read_bytes()))
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            source = f"http://127.0.0.1:{server.server_port}/dataset"
            artifact = build_artifact(source, tmp_path / "result", content_version="subset")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        record = json.loads((artifact.parent / "components.safetensors.provenance.json").read_text())
        assert record["source"] == record["resolved_source"] == source
        assert record["source_sha256"] == hashlib.sha256(gzip.decompress(SUBSET.read_bytes())).hexdigest()
        assert load_library(artifact).definition("ALA") is not None


@pytest.mark.parametrize("arguments", [[], ["build"], ["download"], ["upload"]])
def test_help_without_installed_dependencies(arguments):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "ofoldx.reference.ccd", *arguments, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.fixture
def dataset(tmp_path):
    artifact = build_artifact(SUBSET, tmp_path / "dataset", content_version="subset")
    return artifact.parent


class TestHub:
    def test_empty_revision_refused(self):
        with pytest.raises(InvalidRequestError) as raised:
            download(revision=" ")
        assert raised.value.code == "ccd_revision_invalid"

    def test_default_repository(self, dataset, monkeypatch):
        def snapshot(repo_id, *, repo_type, revision, **kwargs):
            assert (repo_id, repo_type, revision) == ("oteam/ccd", "dataset", "main")
            return str(dataset)

        monkeypatch.setattr("ofoldx.reference.ccd.source.snapshot_download", snapshot)
        assert download() == dataset / "components.safetensors"

    def test_download_and_configure(self, dataset, monkeypatch):
        def snapshot(repo_id, *, repo_type, revision, allow_patterns, local_files_only, token):
            assert repo_id == "example/ccd" and repo_type == "dataset" and revision == "fixed-release"
            assert local_files_only
            return str(dataset)

        monkeypatch.setattr("ofoldx.reference.ccd.source.snapshot_download", snapshot)
        artifact = download("example/ccd", revision="fixed-release", local_files_only=True)
        monkeypatch.setenv("OFOLDX_CCD_ARTIFACT", str(artifact))
        from ofoldx.chemistry.definitions import configured_library

        assert configured_library().definition("ALA") is not None

    @pytest.mark.parametrize("change", ["digest", "content_version"])
    def test_mismatched_sidecar_refused(self, dataset, monkeypatch, change):
        sidecar = dataset / "components.safetensors.provenance.json"
        record = json.loads(sidecar.read_text())
        record["sha256" if change == "digest" else "content_version"] = "wrong"
        sidecar.write_text(json.dumps(record))
        monkeypatch.setattr("ofoldx.reference.ccd.source.snapshot_download", lambda *args, **kwargs: str(dataset))
        with pytest.raises(ContradictionError, match="sha256|content_version"):
            download("example/ccd", revision="fixed-release")

    def test_upload_only_dataset_files(self, dataset, monkeypatch):
        import shutil
        from fnmatch import fnmatch
        from types import SimpleNamespace

        remote = dataset.parent / "remote"
        remote.mkdir()
        (remote / "unrelated.txt").write_text("retained")
        (dataset / "raw-download.cif").write_text("not published")

        class Api:
            def __init__(self, token=None):
                pass

            def create_repo(self, *, repo_id, repo_type, private, exist_ok):
                assert repo_id == "example/ccd" and repo_type == "dataset"

            def upload_folder(self, *, repo_id, repo_type, folder_path, allow_patterns, **kwargs):
                assert repo_type == "dataset"
                for path in Path(folder_path).iterdir():
                    if any(fnmatch(path.name, pattern) for pattern in allow_patterns):
                        shutil.copy2(path, remote / path.name)
                return SimpleNamespace(oid="published-commit")

        monkeypatch.setattr("huggingface_hub.HfApi", Api)
        assert upload(dataset, "example/ccd").oid == "published-commit"
        assert {p.name for p in remote.iterdir()} == {
            "components.safetensors",
            "components.safetensors.provenance.json",
            "README.md",
            "unrelated.txt",
        }
        assert (remote / "unrelated.txt").read_text() == "retained"
