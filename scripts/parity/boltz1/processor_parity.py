"""Boltz-1 Processor lowering parity: MolecularSystem -> Boltz1Input vs the upstream featurizer.

Closes the gap where ``forward_parity`` fed synthetic features and bypassed ``Boltz1Processor``. It drives one
real :class:`~ofoldx.data.MolecularSystem` (a two-residue ALA-GLY protein whose CCD heavy-atom order and
residue-token indices coincide between the two independent pipelines) through
:class:`~ofoldx.models.boltz1.processor.Boltz1Processor`, and compares the packed feature tensors against the
released Boltz-1 ``BoltzFeaturizer`` run on the equivalent hand-built ``Structure`` + ``Tokenized``.

Compared (deterministic, conformer/augmentation-independent) tensors:

* ``token_mask``            <- upstream ``token_pad_mask``
* ``residue_type`` argmax   <- upstream ``res_type``
* ``molecule_type``         <- upstream ``mol_type``
* ``atom_features`` slices  <- upstream ``ref_charge`` (idx3), ``atom_pad_mask`` (idx4, the 389-wide pad
  channel), ``ref_element`` (idx5:133), ``ref_atom_name_chars`` (idx133:389)
* ``atom_to_token`` argmax  <- upstream one-hot ``atom_to_token``
* MSA query row + ``msa_profile``/``deletion_mean`` for the query-only (no-evidence) MSA

The upstream ``ref_pos`` (idx0:3) is a randomly roto-translated RDKit conformer (featurizer.py:840), so the
positional slice is intentionally excluded; only the deterministic packing is compared.

    python -m scripts.parity.boltz1.processor_parity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.parity.boltz1.forward_parity import _ensure_fairscale_stub  # noqa: E402
from scripts.parity.paths import reference_root  # noqa: E402

REFERENCE_SRC = reference_root("boltz") / "src"
if str(REFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(REFERENCE_SRC))

# The shared toy target: one protein chain of two standard residues.
RESIDUES = ("ALA", "GLY")


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
    from ofoldx.data import ChemicalBond, ChemicalComponent, MolecularSystem, Molecule, PolymerSequence, get_component

    components = {residue: get_component(residue) for residue in RESIDUES}
    # New canonical API: PolymerSequence(polymer_type, components) holding exact ChemicalComponent
    # occurrences directly; entity_id is the MolecularSystem mapping key, not a field.
    protein = PolymerSequence("protein", tuple(components[residue] for residue in RESIDUES))
    system = MolecularSystem(
        entities={"protein": protein},
        instances={"A": "protein"},
    )
    del ChemicalBond, ChemicalComponent, Molecule  # documentation of the resolved surface only

    from ofoldx.models.boltz1.processor import Boltz1Processor

    return Boltz1Processor()(system)


def build_reference_features(native) -> dict[str, torch.Tensor]:
    from boltz.data import const
    from boltz.data.feature.featurizer import BoltzFeaturizer, dummy_msa
    from boltz.data.parse.schema import convert_atom_name
    from boltz.data.tokenize.boltz import BoltzTokenizer
    from boltz.data.types import (
        Atom,
        Bond,
        Chain,
        Connection,
        Input,
        Interface,
        Residue,
        Structure,
    )

    # Build the atom table in the same CCD heavy-atom order the Processor emits (verified to match ref_atoms).
    atoms: list[tuple] = []
    residues: list[tuple] = []
    atom_cursor = 0
    for res_index, residue in enumerate(RESIDUES):
        ref_names = const.ref_atoms[residue]
        center_local = ref_names.index("CA")
        disto_local = ref_names.index("CB") if "CB" in ref_names else ref_names.index("CA")
        for local, atom_name in enumerate(ref_names):
            element = {"N": 7, "C": 6, "O": 8, "S": 16}[atom_name[0]]
            coords = (float(atom_cursor + local), 0.0, 0.0)
            atoms.append((convert_atom_name(atom_name), element, 0, coords, coords, True, 0))
        residues.append(
            (
                residue,
                const.token_ids[residue],
                res_index,
                atom_cursor,
                len(ref_names),
                atom_cursor + center_local,
                atom_cursor + disto_local,
                True,
                True,
            )
        )
        atom_cursor += len(ref_names)

    atoms_arr = np.array(atoms, dtype=Atom)
    residues_arr = np.array(residues, dtype=Residue)
    chains_arr = np.array(
        [("A", const.chain_type_ids["PROTEIN"], 0, 0, 0, 0, atom_cursor, 0, len(RESIDUES), 0)],
        dtype=Chain,
    )
    structure = Structure(
        atoms=atoms_arr,
        bonds=np.array([], dtype=Bond),
        residues=residues_arr,
        chains=chains_arr,
        connections=np.array([], dtype=Connection),
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
    )
    msa = {0: dummy_msa(residues_arr)}
    tokenized = BoltzTokenizer().tokenize(Input(structure=structure, msa=msa))
    np.random.seed(0)  # noqa: NPY002 -- pin the ref_pos random augmentation (positional slice is not compared)
    features = BoltzFeaturizer().process(
        tokenized,
        training=False,
        max_atoms=native.atom_features.shape[1],
        max_tokens=native.token_mask.shape[1],
        max_seqs=native.msa.shape[1] if native.msa is not None else 4096,
        pad_to_max_seqs=False,
    )
    return features


def main() -> None:
    args = parse_args()
    _ensure_fairscale_stub()
    native = build_native()
    reference = build_reference_features(native)

    results: list[tuple[bool, float]] = []

    print("=== token features ===", flush=True)
    results.append(
        compare("token_mask", native.token_mask.float(), reference["token_pad_mask"][None].float(), atol=args.atol)
    )
    ref_res_type = reference["res_type"]
    ref_res_argmax = ref_res_type.argmax(-1) if ref_res_type.dim() == 2 else ref_res_type
    results.append(
        compare(
            "residue_type argmax", native.residue_type.argmax(-1).float(), ref_res_argmax[None].float(), atol=args.atol
        )
    )
    results.append(
        compare("molecule_type", native.molecule_type.float(), reference["mol_type"][None].float(), atol=args.atol)
    )

    print("\n=== atom features (389-wide packing) ===", flush=True)
    af = native.atom_features[0]  # [M, 389]
    natoms = int(native.atom_mask[0].sum())
    results.append(compare("atom_pad_mask (idx4)", af[:, 4], reference["atom_pad_mask"], atol=args.atol))
    results.append(compare("ref_charge (idx3)", af[:, 3], reference["ref_charge"], atol=args.atol))
    results.append(
        compare("ref_element (idx5:133)", af[:natoms, 5:133], reference["ref_element"][:natoms].float(), atol=args.atol)
    )
    ref_name = reference["ref_atom_name_chars"][:natoms].float().reshape(natoms, 4 * 64)
    results.append(compare("ref_atom_name_chars (idx133:389)", af[:natoms, 133:389], ref_name, atol=args.atol))
    results.append(
        compare(
            "atom_to_token argmax",
            native.atom_to_token[0, :natoms].float(),
            reference["atom_to_token"][:natoms].argmax(-1).float(),
            atol=args.atol,
        )
    )

    print("\n=== MSA features (query-only, no evidence) ===", flush=True)
    if native.msa is not None:
        ref_msa_query = reference["msa"][0].argmax(-1)  # [N] query row token ids
        results.append(compare("msa query row", native.msa[0, 0].float(), ref_msa_query.float(), atol=args.atol))
        results.append(
            compare("msa_profile", native.msa_profile[0].float(), reference["profile"].float(), atol=args.atol)
        )
        results.append(
            compare(
                "deletion_mean", native.deletion_mean[0].float(), reference["deletion_mean"].float(), atol=args.atol
            )
        )

    worst = max((deviation for _, deviation in results), default=0.0)
    ok = all(passed for passed, _ in results)
    print(f"\n[boltz1] processor worst max_abs = {worst:.3e}  tolerance = {args.atol:.1e}", flush=True)
    raise SystemExit(0 if ok else 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
