from __future__ import annotations

import ofoldx


def test_training_namespace_holds_training_nouns():
    from ofoldx.training import Config, Runner

    assert Config is not None
    assert Runner is not None


def test_training_nouns_off_top_level():
    # arch: inference users never import a training noun from the top-level namespace
    assert not hasattr(ofoldx, "Runner")
    assert not hasattr(ofoldx, "Config")
