# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `ofoldx/`. Domain values and structures are in
`ofoldx/data/`, file-format handling in `ofoldx/io/`, reusable neural-network
components in `ofoldx/modules/`, model-family integrations in `ofoldx/models/`,
and task orchestration in `ofoldx/pipelines/` and `ofoldx/runners/`. Tests mirror
these areas under `tests/`. Documentation sources and configuration are under
`docs/`; maintenance and parity utilities belong in `scripts/`. Treat
`baseline/` and generated directories such as `build/`, `dist/`, and
`*.egg-info/` as external or derived material unless a task explicitly targets
them.

## Build, Test, and Development Commands

- `python -m pip install -e .` installs OFoldX and runtime dependencies for local
  development (Python 3.10 or newer).
- `python -m pip install pytest pytest-cov pre-commit build twine` installs the
  common development tools.
- `python -m pytest -m "not lfs" ofoldx tests` runs the standard CI test suite,
  including doctests and coverage.
- `pre-commit run --all-files` applies formatting, lint, type, spelling, and
  repository hygiene checks.
- `python -m mkdocs build --strict -f docs/mkdocs.yml` validates documentation
  after installing `docs/requirements.txt`.
- `python -m build && python -m twine check dist/*` builds and validates release
  artifacts.

## Coding Style & Naming Conventions

Use four-space indentation and a 120-character line limit. Black formats Python;
isort uses the Black profile. Flake8, pyupgrade, mypy, codespell, and Prettier
run through pre-commit. Use `snake_case` for modules, functions, and variables,
`PascalCase` for classes, and descriptive test names. Keep public model,
processor, and pipeline interfaces typed and family-specific code inside its
corresponding `ofoldx/models/<family>/` package.

## Testing Guidelines

Use pytest and name files `test_*.py` with functions `test_*`. Place tests beside
the mirrored subsystem in `tests/`; use parametrization for backend or shape
variants. Add regression tests for bug fixes and deterministic fixtures where
random tensors are involved. Project coverage is branch-aware and must remain at
or above 60%. Mark large-file-dependent tests with `lfs` so normal CI can exclude
them.

## Commit & Pull Request Guidelines

Follow the existing concise style: `scope: imperative summary`, for example
`io: reject non-finite coordinates`. Keep commits focused and avoid committing
generated artifacts. Pull requests should explain the behavior change and
scientific or architectural rationale, list verification commands, and link
relevant issues. Note checkpoint, compatibility, or data-format impacts
explicitly. Before requesting review, ensure pre-commit, tests, docs (when
changed), and package builds pass.
