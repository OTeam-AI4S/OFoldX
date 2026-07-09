r"""
Verb CLI: ``ofoldx fold in.cif -m <model> -o out.pdb`` (and ``design`` / ``score`` / ``pack``).

Mirrors the adopted structure-tool CLIs (``colabfold_batch``, ``boltz predict``, ``chai-lab fold``): one
verb, an input path, a model, and an output. It is a thin wrapper over the Tier-0 facade
([`ofoldx.fold`][ofoldx.fold] etc.), so it stays out of the training ``otrain`` path.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

_STRUCTURE_VERBS = ("fold", "design")
_VERBS = ("fold", "design", "score", "pack")


def build_parser() -> argparse.ArgumentParser:
    r"""Build the ``ofoldx`` verb argument parser."""

    parser = argparse.ArgumentParser(prog="ofoldx", description="OFoldX task verbs: fold / design / score / pack.")
    subparsers = parser.add_subparsers(dest="verb", required=True)
    for verb in _VERBS:
        verb_parser = subparsers.add_parser(verb, help=f"{verb} a structure/sequence with a model")
        verb_parser.add_argument("input", help="input structure or FASTA path")
        verb_parser.add_argument("-m", "--model", required=True, help="model id or local pretrained directory")
        verb_parser.add_argument(
            "-o", "--output", default=None, help="output file (.pdb/.cif); prints a summary if omitted"
        )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    r"""Parse ``argv`` and run the requested task verb."""

    args = build_parser().parse_args(argv)

    import ofoldx

    result = getattr(ofoldx, args.verb)(args.input, model=args.model)
    if args.output and args.verb in _STRUCTURE_VERBS:
        if str(args.output).lower().endswith(".cif"):
            result.save_cif(args.output)
        else:
            result.save_pdb(args.output)
        print(f"wrote {args.output}")
    else:
        print(getattr(result, "raw", result))
