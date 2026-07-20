"""Boltz-2 Processor lowering parity: MolecularSystem -> Boltz2Input vs the upstream Boltz-2 featurizer.

Closes the coverage gap where ``forward_parity`` feeds a *synthetic* feature batch and never exercises
:class:`~ofoldx.models.boltz2.processor.Boltz2Processor`. It mirrors
``scripts/parity/boltz1/processor_parity.py`` but drives the Boltz-2 whole-system folding Processor and, unlike
the boltz1 harness, compares the Boltz-2-specific token-conditioning tensors the boltz1 featurizer never emits:

* ``method_feature``     (experimental-method conditioning)
* ``modified``           (non-standard-residue flag)
* ``cyclic_period``      (cyclic-polymer conditioning)
* ``type_bonds``         (typed token-bond channel; boltz1 only had the binary ``token_bonds``)
* ``contact_conditioning`` / ``contact_threshold`` (pocket/contact conditioning)

It drives one real :class:`~ofoldx.data.MolecularSystem` (a two-residue ALA-GLY protein whose CCD heavy-atom
order coincides with upstream ``const.ref_atoms`` and whose residue-token indices coincide with
``const.token_ids``) through :class:`Boltz2Processor`, and compares the packed feature tensors against the
released Boltz-2 ``Boltz2Featurizer`` run on the equivalent hand-built ``StructureV2`` + ``Tokenized`` (v2
tokenizer, ``Boltz2Tokenizer``).

Also compared (shared with boltz1, deterministic): ``token_mask``, ``residue_type`` argmax, ``molecule_type``,
the 388-wide atom packing slices (``ref_charge`` idx3, ``ref_element`` idx4:132, ``ref_atom_name_chars``
idx132:388), ``atom_to_token`` argmax, and the query-only MSA row + ``msa_profile`` / ``deletion_mean``.

Excluded (non-deterministic or not lowered by the Processor):

* ``ref_pos`` (idx0:3) is a randomly roto-translated RDKit conformer (featurizerv2 ``center_random_augmentation``).
* Template tensors: the folding Processor does not lower templates yet (DESIGN 6.2); the upstream dummy-template
  block is all-zero, so there is nothing to compare.

The reference featurizer requires a per-residue RDKit-``Mol`` directory (boltz's canonical ``mols`` cache: one
``<COMP>.pkl`` per component). Point at it with ``--moldir`` or ``OFOLDX_BOLTZ_MOLDIR``.

    python -m scripts.parity.boltz2.processor_parity --moldir /data/users/z/.claude_bench/boltz2_realparity/ccd_moldir
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity.paths import reference_root  # noqa: E402

REFERENCE_SRC = reference_root("boltz") / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))

# The shared toy target: one protein chain of two standard residues.
RESIDUES = ("ALA", "GLY")
_DEFAULT_MOLDIR = "/data/users/z/.claude_bench/boltz2_realparity/ccd_moldir"


def compare(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-3) -> tuple[bool, float]:
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        print(f"[SHAPE] {name}: {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
        return False, float("inf")
    diff = (a - b).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    ok = max_abs < atol
    print(f"[{'ok ' if ok else 'BAD'}] {name}: max_abs={max_abs:.3e} shape={tuple(a.shape)}", flush=True)
    return ok, max_abs


def build_native():
    from ofoldx.data import MolecularSystem, PolymerSequence, get_component

    components = {residue: get_component(residue) for residue in RESIDUES}
    protein = PolymerSequence("protein", tuple(components[residue] for residue in RESIDUES))
    system = MolecularSystem(entities={"protein": protein}, instances={"A": "protein"})

    from ofoldx.models.boltz2.processor import Boltz2Processor

    return Boltz2Processor()(system)


def build_reference_features(native, moldir: Path) -> dict[str, torch.Tensor]:
    from boltz.data import const
    from boltz.data.feature.featurizerv2 import Boltz2Featurizer
    from boltz.data.parse.schema import convert_atom_name
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import (
        AtomV2,
        BondV2,
        Chain,
        Coords,
        Ensemble,
        Input,
        Interface,
        Record,
        Residue,
        StructureInfo,
        StructureV2,
    )

    # Build the atom table in the same CCD heavy-atom order the Processor emits. const.ref_atoms is the
    # canonical order (N, CA, C, O, CB ...); it equals the ofoldx get_component heavy-atom order, so the
    # comparison holds the two independent pipelines to the *same* atom layout (as the boltz1 harness does).
    atoms: list[tuple] = []
    residues: list[tuple] = []
    atom_cursor = 0
    for res_index, residue in enumerate(RESIDUES):
        ref_names = const.ref_atoms[residue]
        center_local = ref_names.index("CA")
        disto_local = ref_names.index("CB") if "CB" in ref_names else ref_names.index("CA")
        for local, atom_name in enumerate(ref_names):
            coords = (float(atom_cursor + local), 0.0, 0.0)
            # AtomV2: (name<U4, coords, is_present, bfactor, plddt)
            atoms.append((atom_name, coords, True, 0.0, 0.0))
        residues.append(
            (
                residue,  # name
                const.token_ids[residue],  # res_type
                res_index,  # res_idx
                atom_cursor,  # atom_idx
                len(ref_names),  # atom_num
                atom_cursor + center_local,  # atom_center
                atom_cursor + disto_local,  # atom_disto
                True,  # is_standard
                True,  # is_present
            )
        )
        atom_cursor += len(ref_names)

    atoms_arr = np.array(atoms, dtype=AtomV2)
    residues_arr = np.array(residues, dtype=Residue)
    chains_arr = np.array(
        [
            (
                "A",  # name
                const.chain_type_ids["PROTEIN"],  # mol_type
                0,  # entity_id
                0,  # sym_id
                0,  # asym_id
                0,  # atom_idx
                atom_cursor,  # atom_num
                0,  # res_idx
                len(RESIDUES),  # res_num
                0,  # cyclic_period
            )
        ],
        dtype=Chain,
    )
    coords_arr = np.array([(tuple(coord),) for coord in atoms_arr["coords"]], dtype=Coords)
    ensemble_arr = np.array([(0, atom_cursor)], dtype=Ensemble)
    structure = StructureV2(
        atoms=atoms_arr,
        bonds=np.array([], dtype=BondV2),
        residues=residues_arr,
        chains=chains_arr,
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
        coords=coords_arr,
        ensemble=ensemble_arr,
    )
    del convert_atom_name  # documents the shared atom-name encoding (ord(c)-32) both pipelines use

    record = Record(
        id="processor_parity",
        structure=StructureInfo(method=None),  # folding pass: upstream override_method is None
        chains=[],
        interfaces=[],
    )
    tokenized = Boltz2Tokenizer().tokenize(Input(structure=structure, msa={}, record=record))

    molecules = {}
    for residue in RESIDUES:
        with (moldir / f"{residue}.pkl").open("rb") as handle:
            molecules[residue] = pickle.load(handle)

    random = np.random.default_rng(0)  # pins ref_pos augmentation (positional slice is not compared)
    features = Boltz2Featurizer().process(
        tokenized,
        random=random,
        molecules=molecules,
        training=False,
        max_seqs=native.msa.shape[1] if native.msa is not None else 4096,
        max_atoms=native.atom_features.shape[1],
        max_tokens=native.token_mask.shape[1],
        pad_to_max_seqs=False,
        compute_frames=False,
        compute_symmetries=False,
        compute_constraint_features=False,
    )
    return features


def main() -> None:
    args = parse_args()
    moldir = Path(args.moldir).expanduser()
    if not moldir.is_dir():
        raise SystemExit(f"moldir not found: {moldir} (set --moldir or OFOLDX_BOLTZ_MOLDIR)")

    native = build_native()
    reference = build_reference_features(native, moldir)

    results: list[tuple[str, bool, float]] = []

    def record(name: str, a: torch.Tensor, b: torch.Tensor, *, atol: float = args.atol) -> None:
        ok, dev = compare(name, a, b, atol=atol)
        results.append((name, ok, dev))

    print("=== token features ===", flush=True)
    record("token_mask", native.token_mask.float(), reference["token_pad_mask"][None].float())
    ref_res = reference["res_type"]
    ref_res_argmax = ref_res.argmax(-1) if ref_res.dim() == 2 else ref_res
    record("residue_type argmax", native.residue_type.argmax(-1).float(), ref_res_argmax[None].float())
    record("molecule_type", native.molecule_type.float(), reference["mol_type"][None].float())

    print("\n=== boltz2-specific token conditioning ===", flush=True)
    record("method_feature", native.method_feature.float(), reference["method_feature"][None].float())
    record("modified", native.modified.float(), reference["modified"][None].float())
    record("cyclic_period", native.cyclic_period.float(), reference["cyclic_period"][None].float())
    record("token_bonds", native.token_bonds.float(), reference["token_bonds"][None].squeeze(-1).float())
    record("type_bonds", native.type_bonds.float(), reference["type_bonds"][None].float())
    record(
        "contact_conditioning argmax",
        native.contact_conditioning.argmax(-1).float(),
        reference["contact_conditioning"].argmax(-1)[None].float(),
    )
    record("contact_threshold", native.contact_threshold.float(), reference["contact_threshold"][None].float())

    print("\n=== atom features (388-wide packing) ===", flush=True)
    af = native.atom_features[0]  # [M, 388]
    natoms = int(native.atom_mask[0].sum())
    record("ref_charge (idx3)", af[:, 3], reference["ref_charge"])
    record("ref_element (idx4:132)", af[:natoms, 4:132], reference["ref_element"][:natoms].float())
    ref_name = reference["ref_atom_name_chars"][:natoms].float().reshape(natoms, 4 * 64)
    record("ref_atom_name_chars (idx132:388)", af[:natoms, 132:388], ref_name)
    record(
        "atom_to_token argmax",
        native.atom_to_token[0, :natoms].float(),
        reference["atom_to_token"][:natoms].argmax(-1).float(),
    )

    print("\n=== MSA features (query-only, no evidence) ===", flush=True)
    if native.msa is not None:
        ref_msa_query = reference["msa"][0]  # [N] query row token ids (already token ids, not one-hot)
        record("msa query row", native.msa[0, 0].float(), ref_msa_query.float())
        record("msa_profile", native.msa_profile[0].float(), reference["profile"].float())
        record("deletion_mean", native.deletion_mean[0].float(), reference["deletion_mean"].float())

    worst = max((dev for _, _, dev in results), default=0.0)
    ok = all(passed for _, passed, _ in results)
    failed = [name for name, passed, _ in results if not passed]
    print(f"\n[boltz2] processor worst max_abs = {worst:.3e}  tolerance = {args.atol:.1e}", flush=True)
    if failed:
        print(f"[boltz2] DIVERGENT tensors: {failed}", flush=True)
    raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--moldir", type=str, default=os.environ.get("OFOLDX_BOLTZ_MOLDIR", _DEFAULT_MOLDIR))
    return parser.parse_args()


if __name__ == "__main__":
    main()
