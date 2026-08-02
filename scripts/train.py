#!/usr/bin/env python
"""Compatibility wrapper for :mod:`ofoldx.runners.train`."""

from ofoldx.runners.train import build_config, main, parse_args

__all__ = ["build_config", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
