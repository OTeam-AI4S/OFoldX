"""Convert Protenix SAbDab pickle samples to plain PDB files."""

from __future__ import annotations

import argparse
import gzip
import pickle
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .pkl.gz sample or a .tar.zst archive containing samples.")
    parser.add_argument("output", type=Path, help="Output PDB path.")
    parser.add_argument("--member", help="Archive member, e.g. sabdab_full_v2_bioassembly/1a6w.pkl.gz.")
    args = parser.parse_args()

    sample = _load_sample(args.input, member=args.member)
    atom_array = sample["atom_array"] if isinstance(sample, dict) else sample.atom_array
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_pdb(atom_array, args.output)


def _load_sample(path: Path, *, member: str | None) -> Any:
    if path.suffix == ".zst" or path.name.endswith(".tar.zst"):
        if member is None:
            raise ValueError("--member is required when input is a tar archive")
        with _open_tar_zst(path) as archive:
            for tar_member in archive:
                if tar_member.name != member:
                    continue
                file = archive.extractfile(tar_member)
                if file is None:
                    raise ValueError(f"archive member is not a regular file: {member}")
                return _loads_pickle(gzip.decompress(file.read()))
        raise ValueError(f"archive member not found: {member}")
    with gzip.open(path, "rb") as file:
        return _loads_pickle(file.read())


def _loads_pickle(data: bytes) -> Any:
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "unpickling this sample requires the source package on PYTHONPATH; "
            "run this script in the data-generation environment"
        ) from error


@contextmanager
def _open_tar_zst(path: Path) -> Iterator[tarfile.TarFile]:
    try:
        import zstandard
    except ImportError:
        zstd = shutil.which("zstd")
        if zstd is None:
            raise RuntimeError(
                "reading .tar.zst requires the zstandard Python package or the zstd executable"
            ) from None
        process = subprocess.Popen([zstd, "-dc", str(path)], stdout=subprocess.PIPE)
        if process.stdout is None:
            raise RuntimeError("failed to open zstd stdout")
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                yield archive
        finally:
            process.stdout.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"zstd failed with exit code {return_code}")
    else:
        with path.open("rb") as compressed:
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            try:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    yield archive
            finally:
                reader.close()


def _write_pdb(atom_array: Any, path: Path) -> None:
    coords = atom_array.coord
    chain_id = atom_array.get_annotation("chain_id")
    residue_number = atom_array.get_annotation("res_id")
    residue_name = atom_array.get_annotation("res_name")
    atom_name = atom_array.get_annotation("atom_name")
    element = atom_array.get_annotation("element")
    with path.open("w") as file:
        for index, coord in enumerate(coords, start=1):
            atom_index = index - 1
            file.write(
                _format_pdb_atom(
                    serial=index,
                    atom_name=str(atom_name[atom_index]).strip(),
                    residue_name=str(residue_name[atom_index]).strip(),
                    chain_id=str(chain_id[atom_index]).strip()[:1] or "A",
                    residue_number=int(residue_number[atom_index]),
                    x=float(coord[0]),
                    y=float(coord[1]),
                    z=float(coord[2]),
                    element=str(element[atom_index]).strip().upper()[:2],
                )
            )
        file.write("END\n")


def _format_pdb_atom(
    *,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:<4.4s} {residue_name:>3.3s} {chain_id:1.1s}"
        f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2.2s}\n"
    )


if __name__ == "__main__":
    main()
