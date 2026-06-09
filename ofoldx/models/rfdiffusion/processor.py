r"""
RfDiffusion input processing.

Holds the structure-to-[`RfDiffusionInput`][] processor, the RoseTTAFold atom14 PDB file readers
([`parse_pdb`][] / [`parse_pdb_lines`][] / [`read_pickle`][]), the PDB featurizer
([`process_target`][]), and the config spec-string parsers (``_parse_*``) the sampler consumes.
The readers resolve the atom14 layout from `.chemical`, so they are RfDiffusion-specific and live
with the model family rather than in the shared [`io`][ofoldx.io].
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...data.ccd import CcdProvider
from ...processors.mixin import ProcessorMixin
from ...processors.registry import PROCESSORS
from ...processors.utils import _PROTEIN_RESIDUE_NAMES, StructureInput, structure_from_input, tokenize_structure
from .chemical import ATOM14_NAMES, residue_token
from .model import RfDiffusionInput

PdbIndex = tuple[str, int]


def parse_pdb(filename: str | Path, *, parse_hetatom: bool = False, ignore_het_h: bool = True) -> dict[str, Any]:
    r"""
    Parse a PDB file into RFdiffusion's atom14 target representation.

    Thin file wrapper around [`parse_pdb_lines`][ofoldx.models.rfdiffusion.processor.parse_pdb_lines].

    Args:
        filename: Path to a PDB file.
        parse_hetatom: Whether to also collect HETATM (ligand) records.
        ignore_het_h: Whether to drop hydrogen HETATM atoms when ``parse_hetatom`` is set.

    Returns:
        The target dictionary described in
        [`parse_pdb_lines`][ofoldx.models.rfdiffusion.processor.parse_pdb_lines].

    Examples:
        >>> from ofoldx.models.rfdiffusion.processor import parse_pdb
        >>> target = parse_pdb("target.pdb")  # doctest: +SKIP
    """

    with Path(filename).open() as handle:
        return parse_pdb_lines(handle.readlines(), parse_hetatom=parse_hetatom, ignore_het_h=ignore_het_h)


def parse_pdb_lines(
    lines: Iterable[str],
    *,
    parse_hetatom: bool = False,
    ignore_het_h: bool = True,
) -> dict[str, Any]:
    r"""
    Parse PDB text lines into the official RFdiffusion inference target dictionary.

    Reads one residue per CA record (first occurrence wins), then scatters each ATOM into its atom14 slot for
    the residue's amino-acid type. Missing atoms stay masked; their coords are zero-filled. Optionally collects
    HETATM records for ligand-aware potentials.

    Args:
        lines: Iterable of raw PDB text lines.
        parse_hetatom: Whether to also collect HETATM (ligand) records into ``xyz_het``/``info_het``.
        ignore_het_h: Whether to drop hydrogen HETATM atoms when ``parse_hetatom`` is set.

    Returns:
        A dict with keys ``xyz`` (`[L, 14, 3]` coords), ``mask`` (`[L, 14]` atom-present mask), ``idx``
        (residue numbers), ``seq`` (residue ids), and ``pdb_idx`` (``(chain, number)`` pairs); plus
        ``xyz_het``/``info_het`` when ``parse_hetatom`` is set.

    Examples:
        >>> from ofoldx.models.rfdiffusion.processor import parse_pdb_lines
        >>> lines = [
        ...     "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
        ...     "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C",
        ... ]
        >>> target = parse_pdb_lines(lines)
        >>> target["xyz"].shape
        (1, 14, 3)
        >>> target["pdb_idx"]
        [('A', 1)]
    """

    line_list = list(lines)
    pdb_idx: list[PdbIndex] = []
    residue_names: list[str] = []
    seen: set[PdbIndex] = set()
    for line in line_list:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            chain_id = line[21:22].strip()
            residue_number = int(line[22:26].strip())
            key = (chain_id, residue_number)
            if key in seen:
                continue
            seen.add(key)
            pdb_idx.append(key)
            residue_names.append(line[17:20].strip().upper())

    seq = np.asarray([residue_token(name) for name in residue_names], dtype=np.int64)
    coords = np.full((len(pdb_idx), 14, 3), np.nan, dtype=np.float32)
    index_lookup = {key: index for index, key in enumerate(pdb_idx)}
    for line in line_list:
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21:22].strip()
        residue_number = int(line[22:26].strip())
        residue_idx = index_lookup.get((chain_id, residue_number))
        if residue_idx is None:
            continue
        aa = seq[residue_idx]
        atom_name = line[12:16].strip().upper()
        atom_names = ATOM14_NAMES[int(aa)]
        try:
            atom_idx = atom_names.index(atom_name)
        except ValueError:
            continue
        coords[residue_idx, atom_idx] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )

    mask = ~np.isnan(coords[..., 0])
    coords = np.nan_to_num(coords, nan=0.0)
    out: dict[str, Any] = {
        "xyz": coords,
        "mask": mask,
        "idx": np.asarray([residue_number for _chain, residue_number in pdb_idx], dtype=np.int64),
        "seq": seq,
        "pdb_idx": pdb_idx,
    }
    if parse_hetatom:
        hetatm_coords: list[list[float]] = []
        info_het: list[dict[str, Any]] = []
        for line in line_list:
            if not line.startswith("HETATM"):
                continue
            atom_type = line[76:78].strip()
            if ignore_het_h and atom_type == "H":
                continue
            hetatm_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            info_het.append(
                {
                    "idx": int(line[6:11]),
                    "atom_id": line[12:16],
                    "atom_type": atom_type,
                    "name": line[17:20],
                }
            )
        out["xyz_het"] = np.asarray(hetatm_coords, dtype=np.float32)
        out["info_het"] = info_het
    return out


def read_pickle(path: str | Path) -> object:
    r"""
    Load a pickled object from disk.

    Used to read the cached IGSO3 schedule tables the sampler precomputes.

    Args:
        path: Path to a pickle file.

    Returns:
        The unpickled object.

    Examples:
        >>> from ofoldx.models.rfdiffusion.processor import read_pickle
        >>> value = read_pickle("igso3_schedule.pkl")  # doctest: +SKIP
    """

    with Path(path).open("rb") as handle:
        return pickle.load(handle)


@PROCESSORS.register("rfdiffusion")
class RfDiffusionProcessor(ProcessorMixin):
    r"""
    Convert structure inputs into the RfDiffusion backbone-design schema.

    Args:
        preserve_input_motif: If ``True``, mark all input residues as motif constraints.
        hotspot_residues: Optional ``(chain_id, residue_number)`` residue ids to mark as hotspots.
        ccd: Accepted for API symmetry with structure processors; currently unused.
    """

    input_class = RfDiffusionInput

    def __init__(
        self,
        *,
        preserve_input_motif: bool = True,
        hotspot_residues: tuple[tuple[str, int], ...] = (),
        ccd: CcdProvider | None = None,
    ) -> None:
        self.preserve_input_motif = preserve_input_motif
        self.hotspot_residues = tuple(
            (str(chain_id), int(residue_number)) for chain_id, residue_number in hotspot_residues
        )
        self._ccd = ccd

    def __call__(self, structure: StructureInput) -> RfDiffusionInput:
        r"""
        Tokenize a structure into a batched RfDiffusion input.

        Reads one token per residue, records the representative-atom (CA) coords as the motif positions, and
        derives the token/motif/design masks from ``preserve_input_motif``. The returned tensors carry a
        leading batch dimension of 1.

        Args:
            structure: A structure input accepted by
                [`structure_from_input`][ofoldx.processors.utils.structure_from_input] (path, parsed
                structure, or mapping).

        Returns:
            An [`RfDiffusionInput`][ofoldx.models.rfdiffusion.model.RfDiffusionInput] batched to size 1.

        Raises:
            ValueError: If the structure yields no residue tokens.

        Examples:
            >>> from ofoldx import AutoProcessor  # doctest: +SKIP
            >>> processor = AutoProcessor.from_pretrained("oteam/rfdiffusion")  # doctest: +SKIP
            >>> batch = processor("target.cif")  # doctest: +SKIP
        """

        parsed = structure_from_input(structure)
        tokens = tokenize_structure(parsed)
        if not tokens:
            raise ValueError("RfDiffusion processing requires at least one residue token")
        residues = tuple(token.residue for token in tokens)
        chain_to_index = {chain_id: index for index, chain_id in enumerate(dict.fromkeys(r.chain_id for r in residues))}
        residue_type = torch.tensor(
            [_residue_type_id(parsed.atoms[token.atom_indices[0]].residue_name) for token in tokens],
            dtype=torch.long,
        )
        motif_positions = torch.tensor(
            [
                [
                    parsed.atoms[token.representative_atom_index].x,
                    parsed.atoms[token.representative_atom_index].y,
                    parsed.atoms[token.representative_atom_index].z,
                ]
                for token in tokens
            ],
            dtype=torch.float32,
        )
        token_mask = torch.ones(len(tokens), dtype=torch.bool)
        motif_mask = torch.full((len(tokens),), self.preserve_input_motif, dtype=torch.bool)
        design_mask = token_mask & ~motif_mask
        return RfDiffusionInput(
            token_mask=token_mask.unsqueeze(0),
            residue_index=torch.tensor([residue.residue_number for residue in residues], dtype=torch.long).unsqueeze(0),
            chain_index=torch.tensor(
                [chain_to_index[residue.chain_id] for residue in residues], dtype=torch.long
            ).unsqueeze(0),
            residue_type=residue_type.unsqueeze(0),
            motif_positions=motif_positions.unsqueeze(0),
            motif_mask=motif_mask.unsqueeze(0),
            design_mask=design_mask.unsqueeze(0),
            sequence_tokens=residue_type.unsqueeze(0),
        )

    def _processor_config(self) -> dict[str, object]:
        r"""
        Return the JSON-serializable processor configuration for saving.

        Raises:
            ValueError: If a custom CCD provider is attached, which cannot be serialized.
        """

        if self._ccd is not None:
            raise ValueError("processors with custom CCD providers cannot be saved")
        return {
            "preserve_input_motif": self.preserve_input_motif,
            "hotspot_residues": [list(value) for value in self.hotspot_residues],
        }


def _residue_type_id(residue_name: str) -> int:
    r"""Return the amino-acid type id for ``residue_name``; non-protein residues map to the unknown id (20)."""

    if residue_name not in _PROTEIN_RESIDUE_NAMES:
        return 20
    return _PROTEIN_ORDER[residue_name]


_PROTEIN_ORDER = {name: index for index, name in enumerate(sorted(_PROTEIN_RESIDUE_NAMES))}


def process_target(
    pdb_path: str | Path,
    *,
    parse_hetatom: bool = False,
    center: bool = True,
) -> dict[str, Any]:
    r"""
    Build RFdiffusion atom27 target features from a PDB file.

    Reads the PDB with [`parse_pdb`][ofoldx.models.rfdiffusion.processor.parse_pdb], optionally recenters
    on the mean CA, then lifts the raw atom14 arrays into the atom27 coord/mask tensors the sampler consumes.

    Args:
        pdb_path: Path to the target PDB file.
        parse_hetatom: Whether to carry HETATM (ligand) records through into the output.
        center: Whether to subtract the mean CA position from the coords.

    Returns:
        A dict with ``xyz_27`` (`[L, 27, 3]` coords), ``mask_27`` (`[L, 27]` mask), ``seq`` (residue ids), and
        ``pdb_idx``; plus ``xyz_het``/``info_het`` when ``parse_hetatom`` is set.

    Examples:
        >>> from ofoldx.models.rfdiffusion.processor import process_target
        >>> target = process_target("target.pdb")  # doctest: +SKIP
    """

    target = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)
    ca_center: float | np.ndarray = target["xyz"][:, :1, :].mean(axis=0, keepdims=True) if center else 0.0
    coords = torch.from_numpy(target["xyz"] - ca_center).float()
    seq = torch.from_numpy(target["seq"]).long()
    atom_mask = torch.from_numpy(target["mask"]).bool()
    coords27 = torch.full((coords.shape[0], 27, 3), float("nan"), dtype=torch.float32)
    coords27[:, :14] = coords[:, :14]
    mask_27 = torch.zeros((coords.shape[0], 27), dtype=torch.bool)
    mask_27[:, :14] = atom_mask
    out = {
        "xyz_27": coords27,
        "mask_27": mask_27,
        "seq": seq,
        "pdb_idx": target["pdb_idx"],
    }
    if parse_hetatom:
        out["xyz_het"] = target.get("xyz_het", np.empty((0, 3), dtype=np.float32))
        out["info_het"] = target.get("info_het", [])
    return out


def _parse_length(length: str | None) -> list[int] | None:
    r"""
    Parse a length spec string into a half-open ``[low, high)`` bound.

    Args:
        length: A single value (`"100"`) or a range (`"90-110"`); `None` passes through.

    Returns:
        A two-element ``[low, high)`` list, or `None`.
    """

    if length is None:
        return None
    if "-" not in length:
        value = int(length)
        return [value, value + 1]
    start, stop = length.split("-", maxsplit=1)
    return [int(start), int(stop) + 1]


def _parse_mask_segments(value: Sequence[str] | None) -> list[str] | None:
    r"""Flatten a sequence of slash-joined inpaint segment specs into a single list of segment tokens."""

    return "/".join(value).split("/") if value is not None else None


def _parse_range(value: object) -> list[int]:
    r"""
    Parse a scaffold length spec into an inclusive ``[low, high]`` pair.

    Args:
        value: A single value or a `"low-high"` range; a bare value ``n`` becomes ``[0, n]``.

    Returns:
        A two-element ``[low, high]`` list.
    """

    value_str = str(value)
    if "-" in value_str:
        start, stop = value_str.split("-", maxsplit=1)
        return [int(start), int(stop)]
    return [0, int(value_str)]
