from __future__ import annotations

import pytest

from ofoldx.api.cli import build_parser


def test_cli_parses_fold_verb():
    args = build_parser().parse_args(["fold", "in.cif", "-m", "oteam/af3", "-o", "out.pdb"])
    assert (args.verb, args.input, args.model, args.output) == ("fold", "in.cif", "oteam/af3", "out.pdb")


@pytest.mark.parametrize("verb", ["fold", "design", "score", "pack"])
def test_cli_exposes_all_verbs(verb):
    args = build_parser().parse_args([verb, "x", "-m", "m"])
    assert args.verb == verb


def test_cli_requires_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fold", "in.cif"])


def test_cli_requires_a_verb():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
