"""
Download a small mmCIF debug set from the RCSB PDB.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
SOLVENTS = {"HOH", "WAT", "DOD", "H2O", "D2O", "SOL"}
CRYSTALLIZATION_AIDS = {
    "SO4",
    "GOL",
    "EDO",
    "PO4",
    "ACT",
    "PEG",
    "DMS",
    "TRS",
    "PGE",
    "PG4",
    "FMT",
    "EPE",
    "MPD",
    "MES",
    "CD",
    "IOD",
}
LIGAND_EXCLUSION = {
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCT",
    "BCN",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    "SEP",
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    "TPO",
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/debug/pdb_cifs"))
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raw", action="store_true", help="keep downloaded mmCIF files unfiltered")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in args.output.iterdir():
            if path.suffix.lower() in {".cif", ".mmcif"}:
                path.unlink()
    candidate_count = args.count if args.ids is not None or args.ids_file is not None else args.count * 2
    pdb_ids = collect_ids(candidate_count, ids=args.ids, ids_file=args.ids_file, timeout=args.timeout)
    print(f"downloading {args.count} mmCIF files to {args.output}", flush=True)

    downloaded = 0
    skipped = 0
    failed: list[str] = []
    rejected: list[str] = []
    selected: list[str] = []
    for pdb_id in pdb_ids:
        if len(selected) >= args.count:
            break
        destination = args.output / f"{pdb_id.lower()}.cif"
        if destination.exists() and not args.overwrite:
            if not has_polymer_atoms(destination):
                rejected.append(pdb_id)
                continue
            selected.append(pdb_id)
            skipped += 1
            continue
        try:
            download_cif(pdb_id, destination, timeout=args.timeout, retries=args.retries)
            if not args.raw:
                clean_cif(destination)
            if not has_polymer_atoms(destination):
                rejected.append(pdb_id)
                destination.unlink(missing_ok=True)
                print(f"skip {pdb_id}: no polymer ATOM records", flush=True)
                continue
            selected.append(pdb_id)
            downloaded += 1
            print(f"{len(selected):4d}/{args.count} {pdb_id} -> {destination}", flush=True)
        except (HTTPError, URLError, OSError) as error:
            failed.append(pdb_id)
            print(f"failed {pdb_id}: {error}", flush=True)

    write_manifest(args.output, pdb_ids=selected, failed=failed, rejected=rejected)
    print(
        f"done selected={len(selected)} downloaded={downloaded} skipped={skipped} "
        f"rejected={len(rejected)} failed={len(failed)}",
        flush=True,
    )
    if len(selected) < args.count or failed:
        raise SystemExit(1)


def collect_ids(
    count: int,
    *,
    ids: list[str] | None,
    ids_file: Path | None,
    timeout: float,
) -> list[str]:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if ids is not None and ids_file is not None:
        raise ValueError("use either --ids or --ids-file, not both")
    if ids is not None:
        return normalize_ids(ids)[:count]
    if ids_file is not None:
        return normalize_ids(ids_file.read_text().split())[:count]
    return search_protein_ids(count, timeout=timeout)


def search_protein_ids(count: int, *, timeout: float) -> list[str]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match",
                "value": "Protein (only)",
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": count},
            "sort": [{"sort_by": "rcsb_entry_info.deposited_atom_count", "direction": "asc"}],
        },
    }
    request = Request(
        SEARCH_URL,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    ids = [item["identifier"] for item in payload.get("result_set", [])]
    if len(ids) < count:
        raise RuntimeError(f"RCSB search returned {len(ids)} ids, expected {count}")
    return normalize_ids(ids)


def normalize_ids(ids: list[str]) -> list[str]:
    normalized = []
    seen = set()
    for pdb_id in ids:
        pdb_id = pdb_id.strip().upper()
        if not pdb_id or pdb_id in seen:
            continue
        seen.add(pdb_id)
        normalized.append(pdb_id)
    return normalized


def download_cif(pdb_id: str, destination: Path, *, timeout: float, retries: int) -> None:
    url = DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as response:
                destination.write_bytes(response.read())
            return
        except (HTTPError, URLError, OSError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def clean_cif(path: Path) -> None:
    lines = path.read_text().splitlines(keepends=True)
    ccd_atom_names, ccd_leaving_atoms = parse_chem_comp_atoms(lines)
    cleaned_lines: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            cleaned_lines.append(lines[index])
            index += 1
            continue
        headers, data_start, data_stop = read_loop(lines, index)
        if not headers or not headers[0].startswith("_atom_site."):
            cleaned_lines.extend(lines[index:data_stop])
            index = data_stop
            continue
        cleaned_lines.extend(lines[index:data_start])
        field_names = [header.removeprefix("_atom_site.") for header in headers]
        for row in lines[data_start:data_stop]:
            if keep_atom_site_row(row, field_names, ccd_atom_names, ccd_leaving_atoms):
                cleaned_lines.append(row)
        index = data_stop
    path.write_text("".join(cleaned_lines))


def parse_chem_comp_atoms(lines: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    atom_names: dict[str, set[str]] = {}
    leaving_atoms: dict[str, set[str]] = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        headers, data_start, data_stop = read_loop(lines, index)
        if headers and headers[0].startswith("_chem_comp_atom."):
            field_names = [header.removeprefix("_chem_comp_atom.") for header in headers]
            comp_index = _field_index(field_names, "comp_id")
            atom_index = _field_index(field_names, "atom_id")
            leaving_index = _field_index(field_names, "pdbx_leaving_atom_flag")
            for row in lines[data_start:data_stop]:
                values = split_cif_row(row)
                if len(values) <= max(comp_index, atom_index):
                    continue
                comp_id = _clean_cif_value(values[comp_index]).upper()
                atom_id = _clean_cif_value(values[atom_index])
                atom_names.setdefault(comp_id, set()).add(atom_id)
                if (
                    leaving_index >= 0
                    and len(values) > leaving_index
                    and _clean_cif_value(values[leaving_index]).upper() == "Y"
                ):
                    leaving_atoms.setdefault(comp_id, set()).add(atom_id)
        index = data_stop
    return atom_names, leaving_atoms


def read_loop(lines: list[str], loop_index: int) -> tuple[list[str], int, int]:
    index = loop_index + 1
    headers: list[str] = []
    while index < len(lines) and lines[index].lstrip().startswith("_"):
        headers.append(lines[index].strip().split()[0])
        index += 1
    data_start = index
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
            if stripped == "#":
                index += 1
            break
        index += 1
    return headers, data_start, index


def keep_atom_site_row(
    row: str,
    field_names: list[str],
    ccd_atom_names: dict[str, set[str]],
    ccd_leaving_atoms: dict[str, set[str]],
) -> bool:
    values = split_cif_row(row)
    if len(values) < len(field_names):
        return True
    residue_name = get_field(values, field_names, "label_comp_id", "auth_comp_id").upper()
    atom_name = get_field(values, field_names, "label_atom_id", "auth_atom_id")
    element = get_field(values, field_names, "type_symbol").upper()
    model = get_field(values, field_names, "pdbx_PDB_model_num")
    if model not in {"", ".", "?", "1"}:
        return False
    if residue_name in SOLVENTS or residue_name in CRYSTALLIZATION_AIDS or residue_name in LIGAND_EXCLUSION:
        return False
    if element in {"H", "D"} or atom_name.upper().startswith(("H", "D")):
        return False
    if residue_name in ccd_atom_names and atom_name not in ccd_atom_names[residue_name]:
        return False
    return atom_name not in ccd_leaving_atoms.get(residue_name, set())


def split_cif_row(row: str) -> list[str]:
    return shlex.split(row, posix=False)


def get_field(values: list[str], field_names: list[str], *names: str) -> str:
    for name in names:
        index = _field_index(field_names, name)
        if index >= 0 and index < len(values):
            return _clean_cif_value(values[index])
    return ""


def _field_index(field_names: list[str], name: str) -> int:
    try:
        return field_names.index(name)
    except ValueError:
        return -1


def _clean_cif_value(value: str) -> str:
    return value.strip().strip("'\"")


def has_polymer_atoms(path: Path) -> bool:
    with path.open() as file:
        return any(line.startswith("ATOM ") for line in file)


def write_manifest(output: Path, *, pdb_ids: list[str], failed: list[str], rejected: list[str]) -> None:
    manifest = {
        "source": "RCSB PDB",
        "count": len(pdb_ids),
        "download_url": DOWNLOAD_URL,
        "pdb_ids": pdb_ids,
        "failed": failed,
        "rejected": rejected,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
