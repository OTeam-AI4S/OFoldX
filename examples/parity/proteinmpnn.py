"""Compare OFoldX ProteinMPNN against cloned dauparas/ProteinMPNN.

This is a local conversion check, not a CI test. It copies one randomly
initialized upstream state dict through the converter and compares teacher-forced
log-probabilities. No released checkpoints are downloaded.

Run::

    python examples/parity/proteinmpnn.py
    python examples/parity/proteinmpnn.py --backend eager oops
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ofoldx.data.structure import Structure
from ofoldx.models.proteinmpnn import ProteinMpnnConfig, ProteinMpnnInput, ProteinMpnnModel
from ofoldx.models.proteinmpnn.convert import convert_proteinmpnn_state_dict
from ofoldx.modules.backend._oops import OOPS_AVAILABLE

_DEFAULT_SOURCE = Path("/tmp/ofoldx-upstream/ProteinMPNN")
_SOURCE_URL = "https://github.com/dauparas/ProteinMPNN.git"


def _clone_source(source: Path) -> Path:
    if (source / "protein_mpnn_utils.py").is_file():
        return source
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        import subprocess

        subprocess.run(["git", "clone", "--depth", "1", _SOURCE_URL, str(source)], check=True)
    return source


def _load_upstream(source: Path):
    path = source / "protein_mpnn_utils.py"
    spec = importlib.util.spec_from_file_location("protein_mpnn_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ProteinMPNN from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ProteinMPNN


def _input(
    *,
    backbone: torch.Tensor,
    tokens: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_index: torch.Tensor,
) -> ProteinMpnnInput:
    batch, length = tokens.shape
    return ProteinMpnnInput(
        backbone_coordinates=backbone,
        sequence_tokens=tokens,
        residue_mask=residue_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        design_positions=torch.ones(batch, length, dtype=torch.bool),
        structure=Structure.from_sequence("A" * length),
        component_index=torch.arange(length).expand(batch, length),
        tied_group_ids=torch.arange(length).expand(batch, length),
    )


def _compare(
    backend: str,
    source: Path,
    *,
    seed: int,
    length: int,
    neighbors: int,
    ca_only: bool,
) -> None:
    ProteinMPNN = _load_upstream(_clone_source(source))
    torch.manual_seed(seed)
    hidden_size = 32
    upstream = ProteinMPNN(
        num_letters=21,
        node_features=hidden_size,
        edge_features=hidden_size,
        hidden_dim=hidden_size,
        num_encoder_layers=2,
        num_decoder_layers=2,
        vocab=21,
        k_neighbors=neighbors,
        augment_eps=0.0,
        dropout=0.1,
        ca_only=ca_only,
    )
    upstream.eval()

    converted = ProteinMpnnModel(
        ProteinMpnnConfig(
            hidden_size=hidden_size,
            num_encoder_blocks=2,
            num_decoder_blocks=2,
            num_neighbors=neighbors,
            dropout=0.1,
            ca_only=ca_only,
        )
    )
    converted.load_state_dict(convert_proteinmpnn_state_dict(upstream.state_dict()), strict=True, assign=True)
    converted.eval()
    converted.set_backend(backend)

    batch = 1
    atom_shape = (3,) if ca_only else (4, 3)
    backbone = torch.randn(batch, length, *atom_shape) + torch.arange(length).view(1, length, *([1] * len(atom_shape)))
    tokens = torch.randint(0, 20, (batch, length))
    residue_mask = torch.ones(batch, length)
    residue_index = torch.arange(length).expand(batch, length)
    chain_index = torch.zeros(batch, length, dtype=torch.long)
    decoding_ranks = torch.arange(length).expand(batch, length)
    decoding_order = torch.argsort(decoding_ranks, dim=-1)

    with torch.no_grad():
        expected = upstream(
            backbone,
            tokens,
            residue_mask,
            residue_mask,
            residue_index,
            chain_index,
            torch.ones(batch, length),
            use_input_decoding_order=True,
            decoding_order=decoding_order,
        )
        actual = converted(
            _input(
                backbone=backbone,
                tokens=tokens,
                residue_mask=residue_mask.bool(),
                residue_index=residue_index,
                chain_index=chain_index,
            ),
            decoding_ranks=decoding_ranks,
        )
        log_probs = F.log_softmax(actual.logits, dim=-1)

    torch.testing.assert_close(log_probs, expected, atol=1e-5, rtol=1e-5)
    diff = (log_probs - expected).abs()
    variant = "ca" if ca_only else "backbone"
    print(
        f"proteinmpnn/{variant}/{backend}: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    parser.add_argument("--backend", nargs="+", default=("eager", "oops"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args(argv)

    for backend in args.backend:
        if backend == "oops":
            if not OOPS_AVAILABLE:
                print(
                    "proteinmpnn/oops: skipped (import oops failed; PyPI ooperators 0.0.0 is a placeholder)",
                    flush=True,
                )
                continue
            import oops

            print(
                f"proteinmpnn/oops: {getattr(oops, '__version__', 'unknown')} from {getattr(oops, '__file__', 'unknown')}",
                flush=True,
            )
        for ca_only in (False, True):
            _compare(
                backend,
                args.source,
                seed=args.seed,
                length=args.length,
                neighbors=args.neighbors,
                ca_only=ca_only,
            )


if __name__ == "__main__":
    main()
