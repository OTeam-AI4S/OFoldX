r"""Boltz-1 forward (trunk) numerical-parity harness.

Builds a small synthetic feature batch, runs the released Boltz-1 recycling trunk (input embedder ->
recycle -> MSA module -> Pairformer -> distogram, loaded module-by-module from the public
``boltz1_conf.ckpt``) and the OFoldX :class:`~ofoldx.models.boltz1.model.Boltz1Model` trunk (loaded via
:func:`~ofoldx.models.boltz1.convert.convert_boltz1_state_dict` from the same checkpoint), and compares the
``s_inputs``, post-trunk ``single``, post-trunk ``pair`` and ``distogram`` activations. This catches
op-order / masking / transpose bugs that structural load-parity cannot. This is the trunk diagnostic; it is
not public ``fold``/``FoldResult`` signoff.

Cross-environment split (RFdiffusion-proven): the reference Boltz-1 stack and the OFoldX-native stack do not
share a Python environment, so this harness runs each half separately and compares serialized golden
tensors:

    # reference half, in the environment that can import the boltz source checkout:
    python -m scripts.parity.boltz1.forward_parity --dump-reference ref.pt --device cpu

    # native half, in the zyc environment (CPU / fp32):
    python -m scripts.parity.boltz1.forward_parity --load-reference ref.pt --device cpu

Passing ``--reference-root`` (and neither ``--dump-reference`` nor ``--load-reference``) runs both halves in
one process when the boltz source checkout is importable. With no reference flags at all, only the OFoldX
native trunk runs and its output signature is printed for local golden-regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.parity.paths import reference_root, repo_root, weight_path

REPO = repo_root()
REFERENCE_ROOT = reference_root("boltz")
CHECKPOINT = weight_path("boltz1_conf.ckpt")

# batch, tokens, atoms (2 windows of 32), msa sequences
B, N, M, S = 1, 5, 64, 4
NUM_TOKENS, NUM_ELEMENTS = 33, 128
COMPARISONS = ("s_inputs", "single", "pair", "distogram")


def subset(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    r"""Return the sub-state-dict rooted at ``prefix`` with the prefix stripped."""

    return {key[len(prefix) + 1 :]: value for key, value in state.items() if key.startswith(prefix + ".")}


def build_feats(*, seed: int, device: torch.device) -> dict[str, Tensor]:
    r"""Build one deterministic synthetic Boltz-1 trunk feature batch."""

    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator)

    def randint(high: int, *shape: int) -> Tensor:
        return torch.randint(0, high, shape, generator=generator)

    def onehot(high: int, *shape: int) -> Tensor:
        return torch.nn.functional.one_hot(randint(high, *shape), high).float()

    feats = {
        # token-level residue features
        "res_type": onehot(NUM_TOKENS, B, N),
        "profile": rand(B, N, NUM_TOKENS).softmax(-1),
        "deletion_mean": rand(B, N).abs(),
        "pocket_feature": onehot(4, B, N),
        # atom-level reference features (Boltz-1 atom_feature_dim=389 includes the atom_pad_mask channel)
        "ref_pos": rand(B, M, 3),
        "ref_charge": rand(B, M),
        "ref_element": onehot(NUM_ELEMENTS, B, M),
        "ref_atom_name_chars": onehot(64, B, M, 4),
        "atom_pad_mask": torch.ones(B, M),
        "ref_space_uid": randint(N, B, M).float(),
        "atom_to_token": torch.nn.functional.one_hot(randint(N, B, M), N).float(),
        # relative-position / identity features
        "cyclic_period": torch.zeros(B, N),
        "mol_type": randint(4, B, N),
        "asym_id": randint(2, B, N),
        "entity_id": randint(2, B, N),
        "sym_id": randint(2, B, N),
        "residue_index": torch.arange(N).unsqueeze(0),
        "token_index": torch.arange(N).unsqueeze(0),
        "token_bonds": (rand(B, N, N) > 0).float(),
        "token_pad_mask": torch.ones(B, N),
        # MSA features
        "msa": randint(NUM_TOKENS, B, S, N),
        "has_deletion": (rand(B, S, N) > 0).float(),
        "deletion_value": rand(B, S, N).abs(),
        "msa_paired": (rand(B, S, N) > 0).float(),
        "msa_mask": torch.ones(B, S, N),
    }
    return {key: value.to(device) for key, value in feats.items()}


def allo_input(feats: dict[str, Tensor]) -> Any:
    r"""Assemble the OFoldX ``Boltz1Input`` from the shared golden features."""

    from ofoldx.models.boltz1.model import Boltz1Input

    device = feats["token_pad_mask"].device
    atom_features = torch.cat(
        (
            feats["ref_pos"],
            feats["ref_charge"].unsqueeze(-1),
            feats["atom_pad_mask"].unsqueeze(-1),
            feats["ref_element"],
            feats["ref_atom_name_chars"].reshape(B, M, 4 * 64),
        ),
        dim=-1,
    )
    return Boltz1Input(
        token_mask=feats["token_pad_mask"].bool(),
        residue_index=feats["residue_index"],
        chain_index=feats["asym_id"],
        entity_index=feats["entity_id"],
        symmetry_index=feats["sym_id"],
        token_index=feats["token_index"],
        atom_features=atom_features,
        atom_to_token=feats["atom_to_token"].argmax(-1),  # OFoldX index contract (reference keeps the one-hot)
        atom_mask=feats["atom_pad_mask"].bool(),
        reference_positions=feats["ref_pos"],
        reference_space_id=feats["ref_space_uid"],
        representative_atom_mask=torch.zeros(B, M, dtype=torch.bool, device=device),
        atom_token_index=torch.zeros(B, M, dtype=torch.long, device=device),
        token_to_rep_atom=torch.zeros(B, N, dtype=torch.long, device=device),
        residue_type=feats["res_type"],
        msa_profile=feats["profile"],
        deletion_mean=feats["deletion_mean"],
        token_bonds=feats["token_bonds"],
        molecule_type=feats["mol_type"],
        chain_type=feats["mol_type"],
        cyclic_period=feats["cyclic_period"],
        pocket_feature=feats["pocket_feature"],
        msa=feats["msa"],
        has_deletion=feats["has_deletion"].bool(),
        deletion_value=feats["deletion_value"],
        msa_paired=feats["msa_paired"],
        msa_mask=feats["msa_mask"].bool(),
    )


def _ensure_fairscale_stub() -> None:
    r"""Register a no-op ``fairscale.checkpoint_wrapper`` so the reference trunk imports without fairscale.

    The released Boltz-1 modules import ``checkpoint_wrapper`` at module load, but only *call* it when
    ``activation_checkpointing=True``. This harness always constructs those modules with activation
    checkpointing disabled (see the ``unused`` filter below), so the wrapper is never invoked and the stub
    is numerically inert -- it only satisfies the import in environments (e.g. zyc) that lack fairscale.
    """

    import types

    if "fairscale.nn.checkpoint.checkpoint_activations" in sys.modules:
        return
    fairscale = types.ModuleType("fairscale")
    nn = types.ModuleType("fairscale.nn")
    checkpoint = types.ModuleType("fairscale.nn.checkpoint")
    activations = types.ModuleType("fairscale.nn.checkpoint.checkpoint_activations")

    def checkpoint_wrapper(module: Any, **_: Any) -> Any:
        return module

    activations.checkpoint_wrapper = checkpoint_wrapper  # type: ignore[attr-defined]
    fairscale.nn = nn  # type: ignore[attr-defined]
    nn.checkpoint = checkpoint  # type: ignore[attr-defined]
    checkpoint.checkpoint_activations = activations  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "fairscale": fairscale,
            "fairscale.nn": nn,
            "fairscale.nn.checkpoint": checkpoint,
            "fairscale.nn.checkpoint.checkpoint_activations": activations,
        }
    )


def reference_trunk(
    state: dict[str, Tensor],
    hparams: dict[str, Any],
    feats: dict[str, Tensor],
    *,
    recycling_steps: int,
) -> dict[str, Tensor]:
    r"""Run the released Boltz-1 recycling trunk on the golden features."""

    _ensure_fairscale_stub()
    from boltz.data import const
    from boltz.model.modules.encoders import RelativePositionEncoder
    from boltz.model.modules.trunk import DistogramModule, InputEmbedder, MSAModule, PairformerModule
    from torch import nn

    device = feats["token_pad_mask"].device
    # The released Boltz-1 featurizer one-hots the MSA token ids to ``[B, S, N, num_tokens]`` before the
    # model sees them (boltz/data/feature/featurizer.py), whereas the OFoldX ``MsaModule`` one-hots the
    # integer ids inline. The shared golden batch carries the integer ids (native contract), so expand them
    # here for the reference MSA module. This rebinds only the local dict; the caller's integer ``msa`` (used
    # by ``allo_input`` for the native trunk and by ``--dump-reference`` serialization) is left untouched.
    feats = {
        **feats,
        "msa": torch.nn.functional.one_hot(feats["msa"].long(), const.num_tokens).to(feats["ref_pos"].dtype),
    }
    token_s = int(hparams["token_s"])
    token_z = int(hparams["token_z"])
    s_input_dim = token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)

    def build(module: nn.Module, prefix: str) -> nn.Module:
        module.load_state_dict(subset(state, prefix), strict=True)
        return module.to(device).eval()

    input_embedder = build(
        InputEmbedder(
            atom_s=int(hparams["atom_s"]),
            atom_z=int(hparams["atom_z"]),
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=int(hparams["atoms_per_window_queries"]),
            atoms_per_window_keys=int(hparams["atoms_per_window_keys"]),
            atom_feature_dim=int(hparams["atom_feature_dim"]),
            no_atom_encoder=bool(hparams.get("no_atom_encoder", False)),
            **hparams["embedder_args"],
        ),
        "input_embedder",
    )
    s_init = build(nn.Linear(s_input_dim, token_s, bias=False), "s_init")
    z_init_1 = build(nn.Linear(s_input_dim, token_z, bias=False), "z_init_1")
    z_init_2 = build(nn.Linear(s_input_dim, token_z, bias=False), "z_init_2")
    rel_pos = build(RelativePositionEncoder(token_z), "rel_pos")
    token_bonds = build(nn.Linear(1, token_z, bias=False), "token_bonds")
    s_norm = build(nn.LayerNorm(token_s), "s_norm")
    z_norm = build(nn.LayerNorm(token_z), "z_norm")
    s_recycle = build(nn.Linear(token_s, token_s, bias=False), "s_recycle")
    z_recycle = build(nn.Linear(token_z, token_z, bias=False), "z_recycle")
    unused = {"activation_checkpointing", "offload_to_cpu"}
    msa_args = {key: value for key, value in hparams["msa_args"].items() if key not in unused}
    msa = build(MSAModule(token_z=token_z, s_input_dim=s_input_dim, **msa_args), "msa_module")
    pairformer_args = {key: value for key, value in hparams["pairformer_args"].items() if key not in unused}
    pairformer = build(PairformerModule(token_s, token_z, **pairformer_args), "pairformer_module")
    distogram = build(DistogramModule(token_z, int(hparams["num_bins"])), "distogram_module")

    with torch.no_grad():
        s_inputs = input_embedder(feats)
        s_i = s_init(s_inputs)
        z_i = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]
        z_i = z_i + rel_pos(feats)
        z_i = z_i + token_bonds(feats["token_bonds"].float().unsqueeze(-1))
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s = torch.zeros_like(s_i)
        z = torch.zeros_like(z_i)
        for _ in range(recycling_steps + 1):
            s = s_i + s_recycle(s_norm(s))
            z = z_i + z_recycle(z_norm(z))
            z = z + msa(z, s_inputs, feats)
            s, z = pairformer(s, z, mask=mask, pair_mask=pair_mask)
        distogram_logits = distogram(z)
    return {
        "s_inputs": s_inputs.detach().cpu(),
        "single": s.detach().cpu(),
        "pair": z.detach().cpu(),
        "distogram": distogram_logits.detach().cpu(),
    }


def native_trunk(
    state: dict[str, Tensor],
    feats: dict[str, Tensor],
    *,
    device: torch.device,
    recycling_steps: int,
) -> dict[str, Tensor]:
    r"""Run the OFoldX Boltz-1 trunk on the golden features (loaded from the same checkpoint)."""

    from ofoldx.models.boltz1 import Boltz1Config
    from ofoldx.models.boltz1.convert import convert_boltz1_state_dict
    from ofoldx.models.boltz1.model import Boltz1Model

    with torch.device("meta"):
        model = Boltz1Model(Boltz1Config())
    model.load_state_dict(convert_boltz1_state_dict(state), strict=True, assign=True)
    model = model.to(device).eval().set_backend("torch")
    with torch.no_grad():
        representations = model._trunk(allo_input(feats), recycling_steps=recycling_steps)
    return {
        "s_inputs": representations.single_inputs.detach().cpu(),
        "single": representations.single.detach().cpu(),
        "pair": representations.pair.detach().cpu(),
        "distogram": representations.distogram_logits.detach().cpu(),
    }


def compare_tensor(actual: Tensor, expected: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    r"""Summarize the elementwise agreement of two tensors."""

    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    diff = (actual - expected).abs()
    max_abs = float(diff.max()) if diff.numel() else 0.0
    max_ref = float(expected.abs().max()) if expected.numel() else 0.0
    return {
        "ok": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_abs": max_abs,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "max_rel": max_abs / (max_ref + 1e-9),
        "shape": list(actual.shape),
    }


def signature(outputs: dict[str, Tensor]) -> dict[str, Any]:
    r"""Summarize one trunk output set for golden-regression."""

    return {
        name: {
            "shape": list(outputs[name].shape),
            "sum": float(outputs[name].sum()),
            "mean": float(outputs[name].mean()),
            "std": float(outputs[name].std(unbiased=False)),
        }
        for name in COMPARISONS
    }


def load_checkpoint(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    r"""Load the public Boltz-1 checkpoint's ``state_dict`` and ``hyper_parameters``."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["state_dict"], checkpoint.get("hyper_parameters", {})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.dump_reference is not None:
        state, hparams = load_checkpoint(args.checkpoint)
        feats = build_feats(seed=args.seed, device=device)
        outputs = reference_trunk(state, hparams, feats, recycling_steps=args.recycling_steps)
        args.dump_reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feats": {key: value.cpu() for key, value in feats.items()},
                "outputs": outputs,
                "seed": args.seed,
                "recycling_steps": args.recycling_steps,
                "checkpoint": str(args.checkpoint),
            },
            args.dump_reference,
        )
        print(f"wrote reference dump {args.dump_reference}")
        return

    dumped: dict[str, Any] | None = None
    if args.load_reference is not None:
        dumped = torch.load(args.load_reference, map_location="cpu", weights_only=False)
        if dumped.get("checkpoint") != str(args.checkpoint) or dumped.get("seed") != args.seed:
            raise SystemExit("reference dump checkpoint or seed does not match this invocation")
        if dumped.get("recycling_steps") != args.recycling_steps:
            raise SystemExit("reference dump recycling_steps does not match this invocation")
        feats = {key: value.to(device) for key, value in dumped["feats"].items()}
    else:
        feats = build_feats(seed=args.seed, device=device)

    state, hparams = load_checkpoint(args.checkpoint)
    native = native_trunk(state, feats, device=device, recycling_steps=args.recycling_steps)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "recycling_steps": args.recycling_steps,
        "device": str(device),
        "native": signature(native),
    }

    reference: dict[str, Tensor] | None = None
    if dumped is not None:
        reference = {name: tensor for name, tensor in dumped["outputs"].items()}
    elif args.reference_root is not None:
        if str(args.reference_root / "src") not in sys.path:
            sys.path.insert(0, str(args.reference_root / "src"))
        reference = reference_trunk(state, hparams, feats, recycling_steps=args.recycling_steps)

    if reference is not None:
        results = {
            name: compare_tensor(native[name], reference[name], atol=args.atol, rtol=args.rtol) for name in COMPARISONS
        }
        report["results"] = results
        report["ok"] = all(result["ok"] for result in results.values())

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)

    if "ok" in report and not report["ok"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="boltz source checkout for in-process compare",
    )
    parser.add_argument(
        "--dump-reference",
        type=Path,
        default=None,
        help="Run only the released Boltz-1 trunk (boltz env) and serialize golden feats+outputs to this .pt",
    )
    parser.add_argument(
        "--load-reference",
        type=Path,
        default=None,
        help="Run only the OFoldX-native trunk (zyc) and compare against this dumped reference .pt",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
