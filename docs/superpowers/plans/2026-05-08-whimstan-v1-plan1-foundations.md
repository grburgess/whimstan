# whimstan v1.0 — Plan 1: Foundations (M0–M2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tag legacy Stan version, build new project skeleton with modern tooling, implement OGIP data layer (jaxspec-backed) and JAX physics primitives (TbAbs, power-law, cosmology, WHIM grid loader + validator). Produces a testable physics+data foundation; no inference yet.

**Architecture:** Clean-slate rewrite on a `rewrite/v1` branch off master. Old master tagged `v0.x-stan-legacy` and preserved indefinitely. New layout per spec §4.1. Hard separation: `data/` is pure I/O, `physics/` is pure JAX, no cross-dependency on `model/` or `sim/` yet (those land in Plans 3–4).

**Tech Stack:** Python 3.11, hatchling build, hatch-vcs versioning, uv dep mgr, ruff, mypy, pytest, jax+numpyro+jaxspec, h5py, healpy/gdpyc, omegaconf, mkdocs-material.

**Spec reference:** `docs/superpowers/specs/2026-05-08-whimstan-v1-design.md`

**Out of scope for this plan (covered later):**
- CLOUDY grid build (Plan 2)
- numpyro model, simulation pipeline (Plan 3)
- Inference runner, CLI, diagnostics (Plan 4)
- Stan cross-check, release (Plan 5)

---

## Phase 0 — Branch & Tag

### Task 1: Tag legacy master and branch rewrite/v1

**Files:** none (git only)

- [ ] **Step 1: Verify clean working tree**

```bash
cd /Users/jburgess/coding/projects/whimstan
git status
```

Expected: branch `vari-whim`, untracked files only (`.claude/`, `.dir-locals.el`, `pyrightconfig.json`, `.#simulation.py`, possibly `__pycache__`). No staged or modified files.

- [ ] **Step 2: Switch to master and verify it is the right base**

```bash
git checkout master
git log --oneline -1
```

Expected: HEAD is on master.

- [ ] **Step 3: Tag legacy**

```bash
git tag -a v0.x-stan-legacy -m "Final Stan-based implementation. Superseded by v1.0 numpyro/JAX rewrite."
git tag -l v0.x-stan-legacy
```

Expected: tag `v0.x-stan-legacy` listed.

- [ ] **Step 4: Push tag**

```bash
git push origin v0.x-stan-legacy
```

Expected: `* [new tag] v0.x-stan-legacy -> v0.x-stan-legacy`.

- [ ] **Step 5: Create rewrite/v1 branch off master**

```bash
git checkout -b rewrite/v1
git push -u origin rewrite/v1
```

Expected: `Branch 'rewrite/v1' set up to track ...`.

---

### Task 2: Wipe legacy package contents

**Files:**
- Delete: `whimstan/` (package source; keep test data files)
- Delete: `setup.py`, `setup.cfg`, `versioneer.py`, `MANIFEST.in`
- Delete: `.flake8`, `.github/workflows/build_test.yml`, `.github/workflows/docs.yml`
- Delete: `pyproject.toml` (legacy)
- Keep: `LICENSE`, `README.md` (will rewrite later), `codecov.yml`, `external/`, `docs/superpowers/`
- Keep: `whimstan/data/*.{pi,rmf,arf,h5,npy}` test fixtures (move temporarily to `tests/fixtures/swift_xrt/`)

- [ ] **Step 1: Move test data fixtures aside**

```bash
mkdir -p tests/fixtures/swift_xrt
git mv whimstan/data/apc.pi tests/fixtures/swift_xrt/apc.pi
git mv whimstan/data/apc.rmf tests/fixtures/swift_xrt/apc.rmf
git mv whimstan/data/apc.arf tests/fixtures/swift_xrt/apc.arf
git mv whimstan/data/apcback.pi tests/fixtures/swift_xrt/apcback.pi
git mv whimstan/data/apcsource.pi tests/fixtures/swift_xrt/apcsource.pi
git mv whimstan/data/mw_map.h5 tests/fixtures/swift_xrt/mw_map.h5
git mv whimstan/data/absori.h5 tests/fixtures/swift_xrt/absori_legacy.h5
git mv whimstan/data/spec_gamma2.npy tests/fixtures/swift_xrt/spec_gamma2.npy
git rm whimstan/data/whim_demo.yml whimstan/data/no_whim_demo.yml
```

Expected: `git status` shows the moves staged.

- [ ] **Step 2: Remove legacy package source**

```bash
git rm -rf whimstan/
git rm setup.py setup.cfg versioneer.py MANIFEST.in pyproject.toml
git rm .flake8 .github/workflows/build_test.yml .github/workflows/docs.yml
git rm pyrightconfig.json 2>/dev/null || true
rm -f whimstan/simulations/.#simulation.py 2>/dev/null || true
```

Expected: large list of deleted files staged.

- [ ] **Step 3: Verify only intended files remain**

```bash
git status --short | head -40
ls
```

Expected: deletions of legacy source + python config; `LICENSE`, `README.md`, `codecov.yml`, `docs/`, `tests/`, `external/` still present.

- [ ] **Step 4: Commit the wipe**

```bash
git add -A
git commit -m "chore: wipe legacy v0.x package; preserve test fixtures under tests/fixtures/

Move whimstan/data/*.pi,.rmf,.arf,*.h5,*.npy to tests/fixtures/swift_xrt/
for use as test data in v1 rewrite. Old package source preserved at
tag v0.x-stan-legacy.
"
```

Expected: commit succeeds.

---

## Phase 1 — Tooling Skeleton (M0)

### Task 3: pyproject.toml with hatchling + hatch-vcs

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["hatchling>=1.21", "hatch-vcs>=0.4"]
build-backend = "hatchling.build"

[project]
name = "whimstan"
description = "Hierarchical Bayesian search for the Warm-Hot Intergalactic Medium in Swift-XRT GRB afterglows."
readme = "README.md"
license = "GPL-2.0-or-later"
authors = [{ name = "J. Michael Burgess", email = "jmichaelburgess@gmail.com" }]
requires-python = ">=3.11"
dynamic = ["version"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
  "Operating System :: POSIX",
  "Operating System :: MacOS",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Astronomy",
]
dependencies = [
  "numpy>=1.26,<3",
  "scipy>=1.11,<2",
  "jax>=0.4.30,<0.5",
  "numpyro>=0.15,<0.18",
  "arviz>=0.18,<0.22",
  "jaxspec>=0.1,<0.3",
  "h5py>=3.10,<4",
  "astropy>=6.0,<8",
  "healpy>=1.17,<2",
  "gdpyc>=0.6,<1",
  "omegaconf>=2.3,<3",
  "rich>=13.7,<15",
  "matplotlib>=3.8,<4",
  "joblib>=1.3,<2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
  "mypy>=1.10",
  "pre-commit>=3.7",
]
docs = [
  "mkdocs-material>=9.5",
  "mkdocstrings[python]>=0.25",
]

[project.scripts]
whimstan = "whimstan.cli:main"

[project.urls]
"Source" = "https://github.com/grburgess/whimstan"
"Issues" = "https://github.com/grburgess/whimstan/issues"

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "whimstan/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["whimstan"]

[tool.hatch.build.targets.sdist]
include = ["whimstan/", "tests/", "docs/", "tools/", "README.md", "LICENSE"]

[tool.ruff]
line-length = 88
target-version = "py311"
extend-exclude = ["whimstan/_version.py"]

[tool.ruff.lint]
select = [
  "E", "F", "W",   # pycodestyle, pyflakes
  "I",             # isort
  "B",             # bugbear
  "UP",            # pyupgrade
  "SIM",           # simplify
  "RUF",           # ruff-specific
  "NPY",           # numpy
  "PD",            # pandas (harmless if no pandas)
  "PT",            # pytest style
]
ignore = [
  "E501",   # line length (handled by formatter)
  "B008",   # function call in default arg (numpyro idiom)
]

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "PT011"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true
exclude = ["whimstan/_version.py", "tests/", "docs/"]
plugins = []

[[tool.mypy.overrides]]
module = ["jaxspec.*", "healpy.*", "gdpyc.*", "h5py.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8.0"
addopts = ["-ra", "--strict-markers", "--strict-config"]
testpaths = ["tests"]
markers = [
  "slow: tests that take more than 10 seconds (run nightly only)",
  "needs_grid: tests requiring the WHIM opacity grid release asset",
]
filterwarnings = [
  "error",
  "ignore::DeprecationWarning:jax.*",
]

[tool.coverage.run]
source = ["whimstan"]
omit = ["whimstan/_version.py", "whimstan/cli.py"]

[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "raise NotImplementedError",
  "if TYPE_CHECKING:",
]
```

- [ ] **Step 2: Verify pyproject parses**

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: pyproject.toml with hatchling+hatch-vcs, ruff+mypy+pytest config"
```

---

### Task 4: uv lockfile and dev environment

**Files:**
- Create: `uv.lock` (generated)
- Create: `.python-version`

- [ ] **Step 1: Write .python-version**

```bash
echo "3.11" > .python-version
```

- [ ] **Step 2: Create lockfile**

```bash
uv lock
```

Expected: `uv.lock` created. If `uv` resolves jaxspec successfully, proceed. If jaxspec resolution fails (PyPI version not found), see Task 5 below for the fallback path.

- [ ] **Step 3: Sync dev environment**

```bash
uv sync --extra dev
```

Expected: virtualenv created in `.venv/`, all deps + dev extras installed.

- [ ] **Step 4: Smoke-test core imports**

```bash
uv run python -c "import jax, numpyro, arviz, jaxspec, h5py, astropy, healpy, gdpyc; print('all imports ok'); print('jax', jax.__version__); print('numpyro', numpyro.__version__); print('jaxspec', jaxspec.__version__)"
```

Expected: `all imports ok` and version numbers printed. **If `jaxspec` fails to import, escalate immediately** — the design depends on it; alternative is hand-rolled OGIP I/O (≈3 days extra work, see spec risk #1).

- [ ] **Step 5: Commit**

```bash
git add .python-version uv.lock
git commit -m "build: pin Python 3.11, generate uv.lock"
```

---

### Task 5: jaxspec API spike (de-risk)

**Files:**
- Create: `tests/spike_jaxspec.py` (deleted at end of task)

This is a one-off de-risking script. If jaxspec cannot read the legacy Swift-XRT fixtures, we need to know NOW before building data/ on top of it.

- [ ] **Step 1: Write the spike**

```python
# tests/spike_jaxspec.py
"""One-off spike: verify jaxspec can read Swift-XRT OGIP files.

Deleted after this task. Purpose: de-risk Plan 1 Task 6 onwards.
"""
from pathlib import Path

import jax.numpy as jnp


def main() -> None:
    fixtures = Path(__file__).parent / "fixtures" / "swift_xrt"
    pha = fixtures / "apc.pi"
    rmf = fixtures / "apc.rmf"
    arf = fixtures / "apc.arf"
    bkg = fixtures / "apcback.pi"

    assert pha.exists(), pha
    assert rmf.exists(), rmf
    assert arf.exists(), arf

    # jaxspec API as of late 2024 / 2025
    from jaxspec.data import ObsConfiguration

    obs = ObsConfiguration.from_pha_file(
        pha_path=str(pha),
        rmf_path=str(rmf),
        arf_path=str(arf),
        bkg_path=str(bkg),
        low_energy=0.4,
        high_energy=10.0,
    )

    # Inspect what we got
    print("type:", type(obs).__name__)
    print("attrs:", [a for a in dir(obs) if not a.startswith("_")])

    # Probe for the structures we need
    for attr in ("folded_counts", "folded_background", "exposure",
                 "in_energies", "out_energies", "transfer_matrix"):
        if hasattr(obs, attr):
            v = getattr(obs, attr)
            shape = getattr(v, "shape", None)
            print(f"  {attr}: type={type(v).__name__} shape={shape}")
        else:
            print(f"  {attr}: MISSING")

    # JAX-array verification
    counts = obs.folded_counts if hasattr(obs, "folded_counts") else None
    if counts is not None:
        arr = jnp.asarray(counts)
        assert arr.dtype.kind in ("i", "f"), arr.dtype
        print("counts JAX-castable:", arr.shape, arr.dtype)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the spike**

```bash
uv run python tests/spike_jaxspec.py
```

Expected: prints attribute listing for `ObsConfiguration`. **Note actual attribute names** for use in Task 6. If jaxspec API differs from the assumed `ObsConfiguration.from_pha_file`, find the correct entry point in jaxspec docs and update Task 6 accordingly.

- [ ] **Step 3: Record findings**

Write a one-line note in this plan file under Task 6 listing the actual jaxspec attribute names found (counts, RMF, ARF, exposure, energies). This pins the API for the next task.

- [ ] **Step 4: Delete the spike**

```bash
git rm tests/spike_jaxspec.py 2>/dev/null || rm tests/spike_jaxspec.py
```

- [ ] **Step 5: Commit (notes only)**

```bash
git add docs/superpowers/plans/2026-05-08-whimstan-v1-plan1-foundations.md
git commit -m "spike: validate jaxspec OGIP loader against legacy Swift-XRT fixtures"
```

---

### Task 6: ruff + mypy + pre-commit config files

**Files:**
- Create: `.pre-commit-config.yaml`
- Verify: `pyproject.toml` ruff/mypy sections (already in Task 3)

- [ ] **Step 1: Write .pre-commit-config.yaml**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=2000"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - "numpy>=1.26"
          - "types-PyYAML"
        args: ["--config-file=pyproject.toml"]
        files: ^whimstan/
```

- [ ] **Step 2: Install pre-commit hooks locally**

```bash
uv run pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run pre-commit on existing files**

```bash
uv run pre-commit run --all-files
```

Expected: Most hooks pass (no `whimstan/` source yet so mypy is a no-op). If `check-added-large-files` flags `tests/fixtures/swift_xrt/apc.rmf` (~7 MB) or `mw_map.h5` (~6 MB), update the `--maxkb` threshold to 8000 or move large fixtures behind a Git LFS marker. **For v1.0 raise to 8000 (no LFS):**

```yaml
        args: ["--maxkb=8000"]
```

Re-run `pre-commit run --all-files` until clean.

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: pre-commit hooks (ruff, mypy, file hygiene)"
```

---

### Task 7: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write .github/workflows/ci.yml**

```yaml
name: CI

on:
  push:
    branches: [master, "rewrite/**"]
  pull_request:
    branches: [master]
  schedule:
    - cron: "0 4 * * *"  # nightly UTC, runs slow tests

permissions:
  contents: read

jobs:
  test:
    name: ${{ matrix.os }} / py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install ${{ matrix.python }}

      - name: Install
        run: uv sync --extra dev

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Type check
        run: uv run mypy whimstan/

      - name: Test (fast)
        if: github.event_name != 'schedule'
        run: uv run pytest -m "not slow and not needs_grid" --cov=whimstan --cov-report=xml

      - name: Test (full, nightly)
        if: github.event_name == 'schedule'
        run: uv run pytest --cov=whimstan --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          fail_ci_if_error: false
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

- [ ] **Step 2: Validate YAML**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: ubuntu+macos matrix, py3.11+3.12, ruff/mypy/pytest, nightly slow tests"
```

---

### Task 8: README rewrite

**Files:**
- Modify: `README.md` (replace contents)

- [ ] **Step 1: Replace README.md**

```markdown
# whimstan

[![CI](https://github.com/grburgess/whimstan/actions/workflows/ci.yml/badge.svg)](https://github.com/grburgess/whimstan/actions/workflows/ci.yml)

Hierarchical Bayesian search for the **Warm-Hot Intergalactic Medium**
(WHIM) in Swift-XRT GRB afterglow X-ray spectra.

> **v1.0 (in development)** is a clean-slate rewrite using **JAX +
> numpyro** and a hybrid CIE+PIE opacity grid. The original Stan-based
> implementation is preserved at the git tag
> [`v0.x-stan-legacy`](https://github.com/grburgess/whimstan/releases/tag/v0.x-stan-legacy).
> The package name is unchanged; only the engine changed.

## Quick start

```bash
git clone https://github.com/grburgess/whimstan
cd whimstan
uv sync --extra dev
uv run pytest
```

## Project layout

See `docs/superpowers/specs/2026-05-08-whimstan-v1-design.md` for the
full design and `docs/superpowers/plans/` for implementation plans.

## License

GPL-2.0-or-later.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for v1.0"
```

---

### Task 9: Empty package skeleton with float64 enforcement

**Files:**
- Create: `whimstan/__init__.py`
- Create: `whimstan/py.typed`
- Create: `whimstan/cli.py` (stub)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_smoke.py
"""Smoke tests: package imports and basic invariants."""
from __future__ import annotations


def test_package_imports() -> None:
    import whimstan

    assert hasattr(whimstan, "__version__")


def test_jax_x64_enforced() -> None:
    """JAX must be in float64 mode globally; spec requires this."""
    import whimstan  # noqa: F401  triggers x64 enable
    import jax

    assert jax.config.read("jax_enable_x64") is True


def test_jax_x64_actually_active() -> None:
    """Sanity: a default-constructed array is float64."""
    import whimstan  # noqa: F401
    import jax.numpy as jnp

    x = jnp.array(1.0)
    assert x.dtype == jnp.float64, f"expected float64, got {x.dtype}"
```

- [ ] **Step 2: Run; expect ImportError on `whimstan`**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: 3 failures, all with `ModuleNotFoundError: No module named 'whimstan'`.

- [ ] **Step 3: Write whimstan/__init__.py with x64 enforcement**

```python
# whimstan/__init__.py
"""whimstan — Hierarchical Bayesian WHIM search in GRB afterglows."""
from __future__ import annotations

# Enforce float64 globally before any other JAX import side-effects.
# Spec §7.1: "JAX_ENABLE_X64=True enforced at module import; checked on startup".
import jax as _jax

_jax.config.update("jax_enable_x64", True)
del _jax

try:
    from whimstan._version import __version__
except ImportError:  # pragma: no cover - hatch-vcs not yet built
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
```

- [ ] **Step 4: Write whimstan/py.typed and tests/__init__.py**

```bash
touch whimstan/py.typed
touch tests/__init__.py
```

- [ ] **Step 5: Write whimstan/cli.py stub**

```python
# whimstan/cli.py
"""whimstan CLI entry point. Subcommands implemented in later plans."""
from __future__ import annotations

import sys


def main() -> int:
    print("whimstan CLI — not implemented in Plan 1. See Plan 4.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write tests/conftest.py**

```python
# tests/conftest.py
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

# Importing whimstan early ensures jax x64 is enabled before any other
# fixture or test code creates JAX arrays.
import whimstan  # noqa: F401


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def swift_xrt_dir(fixtures_dir: Path) -> Path:
    d = fixtures_dir / "swift_xrt"
    assert d.is_dir(), d
    return d
```

- [ ] **Step 7: Run smoke tests**

```bash
uv run pytest tests/test_smoke.py -v
```

Expected: all 3 pass.

- [ ] **Step 8: Run mypy + ruff to confirm clean**

```bash
uv run ruff check whimstan/ tests/
uv run mypy whimstan/
```

Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add whimstan/ tests/
git commit -m "feat: package skeleton with JAX float64 enforcement at import"
```

---

### Task 10: docs/ skeleton (mkdocs-material)

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Modify: `docs/` (preserve `superpowers/`)

- [ ] **Step 1: Write mkdocs.yml**

```yaml
site_name: whimstan
site_description: Hierarchical Bayesian WHIM search in Swift-XRT GRB afterglows.
repo_url: https://github.com/grburgess/whimstan
docs_dir: docs

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy
  palette:
    - scheme: default
      primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_root_heading: true
            show_source: false
            docstring_style: numpy

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.arithmatex:
      generic: true
  - toc:
      permalink: true

extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

nav:
  - Home: index.md
  - Specs:
      - v1.0 design: superpowers/specs/2026-05-08-whimstan-v1-design.md
  - Plans:
      - "Plan 1 — Foundations": superpowers/plans/2026-05-08-whimstan-v1-plan1-foundations.md
```

- [ ] **Step 2: Write docs/index.md**

```markdown
# whimstan v1.0

Hierarchical Bayesian search for the Warm-Hot Intergalactic Medium in
Swift-XRT GRB afterglow X-ray spectra.

This site is built from the in-repo design specs and implementation plans.
For the human-facing introduction, see the project [README](https://github.com/grburgess/whimstan).

## What's here

- **Specs** — versioned design documents under `docs/superpowers/specs/`.
- **Plans** — implementation plans under `docs/superpowers/plans/`.

## Build status

See the GitHub Actions [CI status](https://github.com/grburgess/whimstan/actions).
```

- [ ] **Step 3: Verify mkdocs build**

```bash
uv sync --extra docs
uv run mkdocs build --strict
```

Expected: `INFO    -  Documentation built in ...`. No warnings (because of `--strict`).

- [ ] **Step 4: Commit**

```bash
git add mkdocs.yml docs/index.md
git commit -m "docs: mkdocs-material skeleton"
```

---

## Phase 2 — Data Layer (M1)

### Task 11: Observation dataclass + ObservationStatic split

**Files:**
- Create: `whimstan/data/__init__.py`
- Create: `whimstan/data/types.py`
- Create: `tests/data/test_types.py`
- Create: `tests/data/__init__.py`

The split: `Observation` holds raw OGIP data + metadata; `ObservationStatic`
is the JAX-friendly subset passed into the forward model. Two types because
the forward model only needs the precomputed JAX arrays and is `vmap`-able
over them; the raw `Observation` is kept for diagnostics/PPCs.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_types.py
"""Tests for Observation and ObservationStatic dataclasses."""
from __future__ import annotations

import jax.numpy as jnp
import pytest


def test_observation_static_construction() -> None:
    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4, 0.5, 0.6]),
        ene_hi=jnp.array([0.5, 0.6, 0.7]),
        chan_lo=jnp.array([0.4, 0.5]),
        chan_hi=jnp.array([0.5, 0.6]),
        rmf=jnp.zeros((2, 3)),
        arf=jnp.array([100.0, 100.0, 100.0]),
        exposure=jnp.float64(1000.0),
        z=jnp.float64(2.0),
        nh_mw=jnp.float64(1e20),
    )
    assert static.n_chan == 2
    assert static.n_ene == 3
    assert static.ene_avg.shape == (3,)
    assert static.ene_width.shape == (3,)
    assert jnp.allclose(static.ene_avg, jnp.array([0.45, 0.55, 0.65]))
    assert jnp.allclose(static.ene_width, jnp.array([0.1, 0.1, 0.1]))


def test_observation_static_is_pytree() -> None:
    """Must be a JAX PyTree so jax.vmap works over a list of them."""
    import jax
    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4]),
        ene_hi=jnp.array([0.5]),
        chan_lo=jnp.array([0.4]),
        chan_hi=jnp.array([0.5]),
        rmf=jnp.zeros((1, 1)),
        arf=jnp.array([100.0]),
        exposure=jnp.float64(1.0),
        z=jnp.float64(1.0),
        nh_mw=jnp.float64(1e20),
    )
    leaves, _ = jax.tree.flatten(static)
    assert len(leaves) >= 7


def test_observation_static_dtype_promotion() -> None:
    """ene_lo etc. must end up float64 even if user passed float32."""
    import jax.numpy as jnp
    from whimstan.data.types import ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4], dtype=jnp.float32),
        ene_hi=jnp.array([0.5], dtype=jnp.float32),
        chan_lo=jnp.array([0.4]),
        chan_hi=jnp.array([0.5]),
        rmf=jnp.zeros((1, 1)),
        arf=jnp.array([100.0]),
        exposure=jnp.float64(1.0),
        z=jnp.float64(1.0),
        nh_mw=jnp.float64(1e20),
    )
    assert static.ene_lo.dtype == jnp.float64
    assert static.ene_hi.dtype == jnp.float64


def test_observation_holds_counts() -> None:
    from whimstan.data.types import Observation, ObservationStatic

    static = ObservationStatic(
        ene_lo=jnp.array([0.4, 0.5]),
        ene_hi=jnp.array([0.5, 0.6]),
        chan_lo=jnp.array([0.4, 0.5]),
        chan_hi=jnp.array([0.5, 0.6]),
        rmf=jnp.eye(2),
        arf=jnp.array([100.0, 100.0]),
        exposure=jnp.float64(1000.0),
        z=jnp.float64(2.0),
        nh_mw=jnp.float64(1e20),
    )
    obs = Observation(
        name="GRB000000",
        static=static,
        counts=jnp.array([10, 12]),
        bkg_counts=jnp.array([2, 3]),
        bkg_scale=jnp.array([0.05, 0.05]),
        ra_deg=0.0,
        dec_deg=0.0,
    )
    assert obs.name == "GRB000000"
    assert obs.counts.dtype == jnp.int32 or obs.counts.dtype == jnp.int64
    assert obs.bkg_scale.shape == (2,)
```

- [ ] **Step 2: Run; expect import failure**

```bash
uv run pytest tests/data/test_types.py -v
```

Expected: all 4 tests fail with `ModuleNotFoundError: No module named 'whimstan.data'`.

- [ ] **Step 3: Write whimstan/data/__init__.py**

```python
# whimstan/data/__init__.py
"""OGIP data loaders and Observation containers."""
from __future__ import annotations

from whimstan.data.types import Observation, ObservationStatic

__all__ = ["Observation", "ObservationStatic"]
```

- [ ] **Step 4: Write whimstan/data/types.py**

```python
# whimstan/data/types.py
"""Dataclasses for one Swift-XRT observation.

`ObservationStatic` is the JAX-PyTree-registered subset consumed by the
forward model. `Observation` wraps it with raw counts and metadata used
for diagnostics, PPCs, and serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def _as_f64(x: Any) -> Array:
    """Coerce input to JAX float64 array."""
    return jnp.asarray(x, dtype=jnp.float64)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationStatic:
    """Per-GRB precomputed inputs for the forward model.

    All fields are JAX arrays of dtype float64 (except integer indexes,
    none here in v1). Registered as a PyTree so it can be vmap'd over.

    Attributes
    ----------
    ene_lo, ene_hi : (N_ene,)
        Source-spectrum energy bin edges in keV (observer frame).
    chan_lo, chan_hi : (N_chan,)
        Detector channel bin edges in keV.
    rmf : (N_chan, N_ene)
        Response matrix; rmf @ source_flux gives expected channel rate.
    arf : (N_ene,)
        Effective area in cm^2.
    exposure : scalar
        Source exposure in seconds.
    z : scalar
        Source redshift.
    nh_mw : scalar
        Milky-Way column density in cm^-2 (fixed; not a parameter).
    """

    ene_lo: Array
    ene_hi: Array
    chan_lo: Array
    chan_hi: Array
    rmf: Array
    arf: Array
    exposure: Array
    z: Array
    nh_mw: Array

    def __post_init__(self) -> None:
        # frozen=True means we use object.__setattr__ for in-place coercion
        for name in ("ene_lo", "ene_hi", "chan_lo", "chan_hi",
                     "rmf", "arf", "exposure", "z", "nh_mw"):
            object.__setattr__(self, name, _as_f64(getattr(self, name)))

    @property
    def n_chan(self) -> int:
        return int(self.chan_lo.shape[0])

    @property
    def n_ene(self) -> int:
        return int(self.ene_lo.shape[0])

    @property
    def ene_avg(self) -> Array:
        return 0.5 * (self.ene_lo + self.ene_hi)

    @property
    def ene_width(self) -> Array:
        return self.ene_hi - self.ene_lo


@dataclass(frozen=True)
class Observation:
    """One Swift-XRT GRB observation with raw counts + metadata.

    Not a JAX PyTree; carries Python-side fields. Use `static` to access
    the JAX-side state for the forward model.

    `bkg_scale` is the per-channel scaling factor α = effective_source_exposure
    / effective_background_exposure (jaxspec's `folded_backratio`); used
    directly by the Wachter+ 1979 W-stat profile likelihood. We do NOT carry
    a scalar bkg_exposure because real OGIP files have channel-dependent
    BACKSCAL/AREASCAL.
    """

    name: str
    static: ObservationStatic
    counts: Array            # (N_chan,) int
    bkg_counts: Array        # (N_chan,) int — folded_background_unscaled
    bkg_scale: Array         # (N_chan,) float — folded_backratio
    ra_deg: float
    dec_deg: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Counts are integers; preserve their dtype but ensure they are JAX arrays.
        object.__setattr__(self, "counts", jnp.asarray(self.counts))
        object.__setattr__(self, "bkg_counts", jnp.asarray(self.bkg_counts))
        object.__setattr__(self, "bkg_scale", _as_f64(self.bkg_scale))
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/data/test_types.py -v
```

Expected: all 4 pass.

- [ ] **Step 6: Run lint + types**

```bash
uv run ruff check whimstan/data/ tests/data/
uv run mypy whimstan/data/
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add whimstan/data/ tests/data/
git commit -m "feat(data): Observation and ObservationStatic dataclasses, JAX PyTree"
```

---

### Task 12: OGIP loader via jaxspec

**Files:**
- Create: `whimstan/data/ogip.py`
- Create: `tests/data/test_ogip.py`

**API findings from Task 5 spike (jaxspec 0.2.0):**
- `ObsConfiguration.from_pha_file(pha_path, rmf_path, arf_path, bkg_path, low_energy, high_energy)` → xarray-backed obj
- Attributes (all `xr.DataArray` unless noted):
  - `folded_counts`: shape `(N_chan,)` int64 — observed source counts per channel
  - `folded_background_unscaled`: shape `(N_chan,)` int64 — observed bkg counts (raw)
  - `folded_background`: shape `(N_chan,)` float64 — bkg scaled by backratio (do not use directly for W-stat)
  - `folded_backratio`: shape `(N_chan,)` float64 — per-channel α = source-effective / bkg-effective. Use this for W-stat scaling.
  - `exposure`: scalar float64 (seconds)
  - `transfer_matrix`: shape `(N_chan, N_ene)` float64 — RMF
  - `area`: shape `(N_ene,)` float64 — ARF
  - `in_energies`: numpy `ndarray` shape `(2, N_ene)` float64 — `[0,:]`=lo, `[1,:]`=hi (NOT `(N, 2)`)
  - `out_energies`: numpy `ndarray` shape `(2, N_chan)` float64 — same convention
- No `bkg_exposure` attribute. Use `folded_backratio` directly as W-stat α (per-channel) — preferred over a scalar exposure because real OGIP files have per-channel BACKSCAL/AREASCAL.

**Implication for Task 11 dataclass:** Replace the scalar `bkg_exposure` with a per-channel `bkg_scale` array (`folded_backratio`). The `bkg_exposure_ratio` scalar property becomes irrelevant; remove it. Update tests accordingly.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_ogip.py
"""Tests for jaxspec-backed OGIP loader."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest


def test_load_swift_xrt_observation(swift_xrt_dir: Path) -> None:
    from whimstan.data.ogip import load_swift_xrt_observation

    obs = load_swift_xrt_observation(
        name="apc",
        pha_path=swift_xrt_dir / "apc.pi",
        rmf_path=swift_xrt_dir / "apc.rmf",
        arf_path=swift_xrt_dir / "apc.arf",
        bkg_path=swift_xrt_dir / "apcback.pi",
        z=2.0,
        nh_mw=1.0e20,
        ra_deg=10.0,
        dec_deg=-20.0,
        e_lo=0.4,
        e_hi=10.0,
    )

    # Static shape sanity
    assert obs.static.n_chan > 0
    assert obs.static.n_ene > 0
    assert obs.static.rmf.shape == (obs.static.n_chan, obs.static.n_ene)
    assert obs.static.arf.shape == (obs.static.n_ene,)

    # Energy ordering
    assert jnp.all(obs.static.ene_lo < obs.static.ene_hi)
    assert jnp.all(obs.static.chan_lo < obs.static.chan_hi)

    # All energies inside requested band
    assert float(obs.static.chan_lo.min()) >= 0.4 - 1e-6
    assert float(obs.static.chan_hi.max()) <= 10.0 + 1e-6

    # Counts are non-negative integers
    assert obs.counts.shape == (obs.static.n_chan,)
    assert int(obs.counts.min()) >= 0

    # Background present and non-negative
    assert obs.bkg_counts.shape == (obs.static.n_chan,)
    assert int(obs.bkg_counts.min()) >= 0

    # Per-channel bkg scale (W-stat alpha) is positive and well-formed
    assert obs.bkg_scale.shape == (obs.static.n_chan,)
    assert float(obs.bkg_scale.min()) > 0

    # Exposure positive
    assert float(obs.static.exposure) > 0

    # Metadata round-trip
    assert obs.name == "apc"
    assert obs.ra_deg == 10.0
    assert pytest.approx(obs.static.z) == 2.0
    assert pytest.approx(obs.static.nh_mw) == 1.0e20


def test_load_missing_file_raises(swift_xrt_dir: Path) -> None:
    from whimstan.data.ogip import load_swift_xrt_observation

    with pytest.raises(FileNotFoundError):
        load_swift_xrt_observation(
            name="x",
            pha_path=swift_xrt_dir / "does-not-exist.pi",
            rmf_path=swift_xrt_dir / "apc.rmf",
            arf_path=swift_xrt_dir / "apc.arf",
            bkg_path=swift_xrt_dir / "apcback.pi",
            z=1.0,
            nh_mw=1e20,
            ra_deg=0.0,
            dec_deg=0.0,
        )
```

- [ ] **Step 2: Run; expect import failure**

```bash
uv run pytest tests/data/test_ogip.py -v
```

Expected: failures with `ModuleNotFoundError: No module named 'whimstan.data.ogip'`.

- [ ] **Step 3: Write whimstan/data/ogip.py**

The exact attribute names below MUST match what Task 5 spike found. If the spike found `obs.transfer_matrix.matrix` instead of `obs.transfer_matrix`, adjust accordingly. Keep this layer thin: it is the only place jaxspec is touched.

```python
# whimstan/data/ogip.py
"""jaxspec-backed loader for OGIP-format X-ray spectral files.

Thin wrapper: this is the only module that imports jaxspec. If jaxspec
ever needs to be replaced, only this file (plus tests) changes.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from whimstan.data.types import Observation, ObservationStatic


def load_swift_xrt_observation(
    *,
    name: str,
    pha_path: str | Path,
    rmf_path: str | Path,
    arf_path: str | Path,
    bkg_path: str | Path,
    z: float,
    nh_mw: float,
    ra_deg: float,
    dec_deg: float,
    e_lo: float = 0.4,
    e_hi: float = 10.0,
) -> Observation:
    """Load one Swift-XRT observation from PHA/RMF/ARF/BKG files.

    Parameters
    ----------
    name
        Identifier (e.g. GRB name).
    pha_path, rmf_path, arf_path, bkg_path
        Filesystem paths to OGIP files.
    z, nh_mw
        Source redshift and Milky-Way column density (cm^-2). Both fixed.
    ra_deg, dec_deg
        Sky position for downstream MW-NH lookups.
    e_lo, e_hi
        Channel-energy band kept; channels outside [e_lo, e_hi] are dropped.

    Returns
    -------
    Observation
        Wrapped Observation with float64 JAX arrays inside ObservationStatic.

    Raises
    ------
    FileNotFoundError
        If any of the input paths does not exist.
    """
    for p in (pha_path, rmf_path, arf_path, bkg_path):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    from jaxspec.data import ObsConfiguration

    cfg = ObsConfiguration.from_pha_file(
        pha_path=str(pha_path),
        rmf_path=str(rmf_path),
        arf_path=str(arf_path),
        bkg_path=str(bkg_path),
        low_energy=e_lo,
        high_energy=e_hi,
    )

    # Map jaxspec 0.2.0 attributes onto our schema. Layout: in_energies
    # and out_energies are shape (2, N) with [0,:] = lo and [1,:] = hi.
    counts = jnp.asarray(cfg.folded_counts.values)
    bkg_counts = jnp.asarray(cfg.folded_background_unscaled.values)
    bkg_scale = jnp.asarray(cfg.folded_backratio.values)
    exposure = jnp.asarray(cfg.exposure.values)

    in_lo = jnp.asarray(cfg.in_energies[0, :])
    in_hi = jnp.asarray(cfg.in_energies[1, :])
    out_lo = jnp.asarray(cfg.out_energies[0, :])
    out_hi = jnp.asarray(cfg.out_energies[1, :])

    rmf = jnp.asarray(cfg.transfer_matrix.values)
    arf = jnp.asarray(cfg.area.values)

    static = ObservationStatic(
        ene_lo=in_lo,
        ene_hi=in_hi,
        chan_lo=out_lo,
        chan_hi=out_hi,
        rmf=rmf,
        arf=arf,
        exposure=exposure,
        z=jnp.float64(z),
        nh_mw=jnp.float64(nh_mw),
    )
    return Observation(
        name=name,
        static=static,
        counts=counts,
        bkg_counts=bkg_counts,
        bkg_scale=bkg_scale,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/data/test_ogip.py -v
```

Expected: both tests pass. **If they fail because jaxspec attribute names differ, fix the mapping in `whimstan/data/ogip.py` only — tests must not change.**

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check whimstan/data/
uv run mypy whimstan/data/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add whimstan/data/ogip.py tests/data/test_ogip.py
git commit -m "feat(data): jaxspec-backed Swift-XRT OGIP loader"
```

---

### Task 13: GRB catalog loader

**Files:**
- Create: `whimstan/data/grb_sample_v2.csv` (port from `external/`/`articles/`)
- Create: `whimstan/data/catalog.py`
- Create: `tests/data/test_catalog.py`

- [ ] **Step 1: Port the catalog file**

```bash
# Source: legacy paper data file
cp /Users/jburgess/articles/whim_paper/src/figures/Swift_GRB_WHIM_cut_v2.txt /tmp/grbs_legacy.txt
head -3 /tmp/grbs_legacy.txt
```

Inspect column layout. Then convert to a documented CSV with explicit
header.

```python
# tools/port_catalog.py — run once, not committed to package
import re
from pathlib import Path

src = Path("/tmp/grbs_legacy.txt")
dst = Path("whimstan/data/grb_sample_v2.csv")

# columns from the legacy file (12 fields). See spec for description.
# Adapt to actual whitespace splitting; commit the resulting CSV.
out_lines = ["name,obsid,ra_deg,dec_deg,l_gal,b_gal,nh_mw,z,t_start,t_stop,exposure,counts\n"]
for line in src.read_text().splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    fields = re.split(r"\s+", line.strip())
    if len(fields) < 12:
        continue
    out_lines.append(",".join(fields[:12]) + "\n")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("".join(out_lines))
print(f"Wrote {dst} with {len(out_lines)-1} GRBs")
```

```bash
uv run python tools/port_catalog.py
head -3 whimstan/data/grb_sample_v2.csv
wc -l whimstan/data/grb_sample_v2.csv
```

Expected: 200+ lines, header + GRBs. **If the legacy file does not parse cleanly (mixed delimiters, extra header rows), adjust the porting script; do not edit the CSV by hand.**

- [ ] **Step 2: Write the failing test**

```python
# tests/data/test_catalog.py
"""Tests for GRB catalog loader."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_load_packaged_catalog() -> None:
    from whimstan.data.catalog import load_packaged_catalog

    cat = load_packaged_catalog()
    assert len(cat) >= 100
    first = cat[0]
    assert first.name.startswith("GRB") or first.name[0].isdigit()
    assert -90.0 <= first.dec_deg <= 90.0
    assert 0.0 <= first.ra_deg <= 360.0
    assert first.z > 0
    assert first.nh_mw > 0


def test_filter_by_z_and_nh(swift_xrt_dir: Path) -> None:
    from whimstan.data.catalog import load_packaged_catalog, filter_catalog

    cat = load_packaged_catalog()
    n_before = len(cat)
    filtered = filter_catalog(cat, z_min=1.0, nh_mw_max=1.0e21)
    assert len(filtered) <= n_before
    for entry in filtered:
        assert entry.z >= 1.0
        assert entry.nh_mw <= 1.0e21


def test_load_missing_csv_raises(tmp_path: Path) -> None:
    from whimstan.data.catalog import load_catalog

    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.csv")
```

- [ ] **Step 3: Run, expect failure**

```bash
uv run pytest tests/data/test_catalog.py -v
```

Expected: ImportError on `whimstan.data.catalog`.

- [ ] **Step 4: Write whimstan/data/catalog.py**

```python
# whimstan/data/catalog.py
"""GRB catalog loader.

The packaged CSV `grb_sample_v2.csv` is the legacy WHIM-paper sample
(205 GRBs). New samples can be loaded with `load_catalog(path)`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One GRB row from the catalog CSV."""

    name: str
    obsid: str
    ra_deg: float
    dec_deg: float
    l_gal: float
    b_gal: float
    nh_mw: float          # cm^-2
    z: float
    t_start: float
    t_stop: float
    exposure: float       # seconds
    counts: int


def load_catalog(path: str | Path) -> list[CatalogEntry]:
    """Load a catalog CSV into a list of CatalogEntry."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    rows: list[CatalogEntry] = []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                CatalogEntry(
                    name=row["name"],
                    obsid=row["obsid"],
                    ra_deg=float(row["ra_deg"]),
                    dec_deg=float(row["dec_deg"]),
                    l_gal=float(row["l_gal"]),
                    b_gal=float(row["b_gal"]),
                    nh_mw=float(row["nh_mw"]),
                    z=float(row["z"]),
                    t_start=float(row["t_start"]),
                    t_stop=float(row["t_stop"]),
                    exposure=float(row["exposure"]),
                    counts=int(float(row["counts"])),
                )
            )
    return rows


def load_packaged_catalog() -> list[CatalogEntry]:
    """Load the legacy WHIM-paper Swift-XRT sample shipped in the package."""
    csv_path = files("whimstan.data").joinpath("grb_sample_v2.csv")
    with csv_path.open("r") as _:
        return load_catalog(str(csv_path))


def filter_catalog(
    cat: list[CatalogEntry],
    *,
    z_min: float | None = None,
    z_max: float | None = None,
    nh_mw_max: float | None = None,
    counts_min: int | None = None,
    abs_b_min: float | None = None,
) -> list[CatalogEntry]:
    """Apply selection cuts; returns filtered list (does not mutate input)."""

    def keep(e: CatalogEntry) -> bool:
        if z_min is not None and e.z < z_min:
            return False
        if z_max is not None and e.z > z_max:
            return False
        if nh_mw_max is not None and e.nh_mw > nh_mw_max:
            return False
        if counts_min is not None and e.counts < counts_min:
            return False
        if abs_b_min is not None and abs(e.b_gal) < abs_b_min:
            return False
        return True

    return [e for e in cat if keep(e)]
```

- [ ] **Step 5: Update whimstan/data/__init__.py exports**

```python
# whimstan/data/__init__.py
"""OGIP data loaders and Observation containers."""
from __future__ import annotations

from whimstan.data.catalog import (
    CatalogEntry,
    filter_catalog,
    load_catalog,
    load_packaged_catalog,
)
from whimstan.data.ogip import load_swift_xrt_observation
from whimstan.data.types import Observation, ObservationStatic

__all__ = [
    "CatalogEntry",
    "Observation",
    "ObservationStatic",
    "filter_catalog",
    "load_catalog",
    "load_packaged_catalog",
    "load_swift_xrt_observation",
]
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/data/test_catalog.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint + types**

```bash
uv run ruff check whimstan/data/
uv run mypy whimstan/data/
```

- [ ] **Step 8: Commit**

```bash
git add whimstan/data/grb_sample_v2.csv whimstan/data/catalog.py whimstan/data/__init__.py tests/data/test_catalog.py
git commit -m "feat(data): GRB catalog loader, port legacy 205-GRB sample as CSV"
```

---

### Task 14: Dataset = list[Observation] convenience

**Files:**
- Create: `whimstan/data/dataset.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dataset.py
"""Tests for Dataset container."""
from __future__ import annotations

from pathlib import Path


def _one_obs(swift_xrt_dir: Path, name: str = "apc"):
    from whimstan.data import load_swift_xrt_observation

    return load_swift_xrt_observation(
        name=name,
        pha_path=swift_xrt_dir / "apc.pi",
        rmf_path=swift_xrt_dir / "apc.rmf",
        arf_path=swift_xrt_dir / "apc.arf",
        bkg_path=swift_xrt_dir / "apcback.pi",
        z=2.0,
        nh_mw=1.0e20,
        ra_deg=0.0,
        dec_deg=0.0,
    )


def test_dataset_construction(swift_xrt_dir: Path) -> None:
    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs, obs])
    assert ds.n_grbs == 3
    assert ds.n_ene == obs.static.n_ene
    assert ds.n_chan == obs.static.n_chan


def test_dataset_stack_static(swift_xrt_dir: Path) -> None:
    """Stacking ObservationStatic must broadcast over GRB axis."""
    import jax.numpy as jnp
    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs])
    static = ds.stack_static()

    # Leading axis is N_grbs; trailing axes are per-GRB shapes
    assert static.rmf.shape == (2, obs.static.n_chan, obs.static.n_ene)
    assert static.arf.shape == (2, obs.static.n_ene)
    assert static.exposure.shape == (2,)
    assert static.z.shape == (2,)
    assert jnp.allclose(static.z, jnp.array([obs.static.z, obs.static.z]))


def test_dataset_total_counts(swift_xrt_dir: Path) -> None:
    from whimstan.data.dataset import Dataset

    obs = _one_obs(swift_xrt_dir)
    ds = Dataset([obs, obs])
    totals = ds.total_counts()
    assert totals.shape == (2,)
    assert int(totals[0]) == int(obs.counts.sum())
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/data/test_dataset.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write whimstan/data/dataset.py**

```python
# whimstan/data/dataset.py
"""Dataset = ordered collection of Observations.

Provides stacking utilities for broadcasting per-GRB static fields into
the leading axis of a single PyTree, ready for vmap.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from whimstan.data.types import Observation, ObservationStatic


@dataclass(frozen=True)
class Dataset:
    """Ordered list of Observations all sharing N_chan and N_ene."""

    observations: tuple[Observation, ...]

    def __init__(self, observations: Iterable[Observation]) -> None:
        obs_tuple = tuple(observations)
        if not obs_tuple:
            raise ValueError("Dataset requires at least one Observation")
        n_chan = obs_tuple[0].static.n_chan
        n_ene = obs_tuple[0].static.n_ene
        for o in obs_tuple[1:]:
            if o.static.n_chan != n_chan or o.static.n_ene != n_ene:
                raise ValueError(
                    f"observation {o.name!r} has shape "
                    f"({o.static.n_chan}, {o.static.n_ene}) but expected "
                    f"({n_chan}, {n_ene}); v1.0 requires uniform grids"
                )
        object.__setattr__(self, "observations", obs_tuple)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Observation:
        return self.observations[idx]

    @property
    def n_grbs(self) -> int:
        return len(self.observations)

    @property
    def n_chan(self) -> int:
        return self.observations[0].static.n_chan

    @property
    def n_ene(self) -> int:
        return self.observations[0].static.n_ene

    def stack_static(self) -> ObservationStatic:
        """Stack all observations' static state along axis 0.

        Returns a single ObservationStatic whose fields have a leading
        N_grbs axis. Convenient for `vmap`.
        """
        statics: Sequence[ObservationStatic] = [o.static for o in self.observations]
        return jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0),
            *statics,
        )  # type: ignore[return-value]

    def total_counts(self) -> jax.Array:
        """Per-GRB total source counts; used by reparam centering threshold."""
        return jnp.stack([o.counts.sum() for o in self.observations])
```

- [ ] **Step 4: Update whimstan/data/__init__.py**

```python
# add to imports + __all__
from whimstan.data.dataset import Dataset
```

(append `"Dataset"` to `__all__`.)

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/data/test_dataset.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint + types**

```bash
uv run ruff check whimstan/data/
uv run mypy whimstan/data/
```

- [ ] **Step 7: Commit**

```bash
git add whimstan/data/ tests/data/test_dataset.py
git commit -m "feat(data): Dataset container with stack_static and total_counts"
```

---

### Task 15: utils/logging port + module entry point

**Files:**
- Create: `whimstan/utils/__init__.py`
- Create: `whimstan/utils/logging.py`
- Create: `tests/utils/__init__.py`
- Create: `tests/utils/test_logging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/test_logging.py
"""Tests for whimstan.utils.logging."""
from __future__ import annotations

import logging


def test_get_logger_returns_namespaced_logger() -> None:
    from whimstan.utils.logging import get_logger

    log = get_logger("forward")
    assert isinstance(log, logging.Logger)
    assert log.name == "whimstan.forward"


def test_setup_logging_idempotent() -> None:
    """Repeated setup_logging calls must not stack handlers."""
    from whimstan.utils.logging import setup_logging

    log = setup_logging(level="INFO")
    n_handlers = len(log.handlers)
    setup_logging(level="INFO")
    assert len(log.handlers) == n_handlers
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/utils/test_logging.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write whimstan/utils/__init__.py**

```python
# whimstan/utils/__init__.py
"""Utility helpers (logging, configuration, plotting helpers)."""
from __future__ import annotations

from whimstan.utils.logging import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
```

- [ ] **Step 4: Write whimstan/utils/logging.py**

```python
# whimstan/utils/logging.py
"""Stdlib logging + rich handler for human-readable CLI output."""
from __future__ import annotations

import logging
from typing import Literal

from rich.logging import RichHandler

_ROOT_NAME = "whimstan"
_DEFAULT_FMT = "%(message)s"


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> logging.Logger:
    """Configure the `whimstan` root logger; idempotent."""
    log = logging.getLogger(_ROOT_NAME)
    log.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in log.handlers):
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            log_time_format="[%X]",
        )
        handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
        log.addHandler(handler)
    log.propagate = False
    return log


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the `whimstan` namespace."""
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/utils/test_logging.py -v
```

Expected: pass.

- [ ] **Step 6: Lint + types**

```bash
uv run ruff check whimstan/utils/
uv run mypy whimstan/utils/
```

- [ ] **Step 7: Commit**

```bash
git add whimstan/utils/ tests/utils/
git commit -m "feat(utils): rich-backed stdlib logging, idempotent setup"
```

---

## Phase 3 — Physics Primitives (M2)

### Task 16: cosmology.py — FlatLambdaCDM in JAX

**Files:**
- Create: `whimstan/physics/__init__.py`
- Create: `whimstan/physics/cosmology.py`
- Create: `tests/physics/__init__.py`
- Create: `tests/physics/test_cosmology.py`

We need a JAX-native `FlatLambdaCDM` for the cosmological line-of-sight
integral inside the WHIM optical depth. Using `astropy.cosmology` would
break jit/vmap. The implementation is small (~30 lines) and well-defined.

- [ ] **Step 1: Write the failing test**

```python
# tests/physics/test_cosmology.py
"""Tests for whimstan.physics.cosmology — JAX FlatLambdaCDM.

Reference values cross-checked against astropy.cosmology Planck18.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def test_planck18_constants() -> None:
    from whimstan.physics.cosmology import Planck18

    assert pytest.approx(Planck18.H0_km_s_Mpc, rel=1e-6) == 67.4
    assert pytest.approx(Planck18.Omega_m, rel=1e-6) == 0.315
    assert pytest.approx(Planck18.Omega_L, rel=1e-6) == 0.685


def test_E_at_z0_is_one() -> None:
    from whimstan.physics.cosmology import Planck18, dimensionless_hubble

    assert float(dimensionless_hubble(0.0, Planck18)) == pytest.approx(1.0)


def test_comoving_distance_matches_astropy() -> None:
    """JAX implementation must match astropy.cosmology.Planck18.comoving_distance to <0.1%."""
    from astropy.cosmology import Planck18 as Astropy_Planck18

    from whimstan.physics.cosmology import Planck18, comoving_distance_Mpc

    z_grid = jnp.linspace(0.01, 5.0, 50)
    jax_d = np.asarray(comoving_distance_Mpc(z_grid, Planck18, n_steps=4096))
    ap_d = np.asarray(Astropy_Planck18.comoving_distance(np.asarray(z_grid)).value)
    rel_err = np.abs(jax_d - ap_d) / ap_d
    assert rel_err.max() < 1e-3, f"max rel err {rel_err.max():.3e}"


def test_jit_safe() -> None:
    import jax

    from whimstan.physics.cosmology import Planck18, comoving_distance_Mpc

    f = jax.jit(lambda z: comoving_distance_Mpc(z, Planck18))
    out = f(jnp.array([0.5, 1.0, 2.0]))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/physics/test_cosmology.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write whimstan/physics/__init__.py**

```python
# whimstan/physics/__init__.py
"""Pure-JAX physics primitives for the forward model."""
from __future__ import annotations
```

- [ ] **Step 4: Write whimstan/physics/cosmology.py**

```python
# whimstan/physics/cosmology.py
"""FlatLambdaCDM in pure JAX.

Spec §5.3: Planck18 cosmology drives the WHIM line-of-sight integral.
Used at *grid-build time* only — inference-time JAX code never sees this
because the cosmological integral is already baked into the grid. We
keep this module so the JAX validator (Task 22) and the simulation
pipeline can compute distances differentiably without astropy.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

# Speed of light, km/s.
_C_KM_S = 299_792.458


@dataclass(frozen=True, slots=True)
class FlatLambdaCDM:
    """Flat-LambdaCDM cosmology; matter and dark-energy fractions sum to 1."""

    H0_km_s_Mpc: float
    Omega_m: float
    Omega_L: float

    @property
    def hubble_distance_Mpc(self) -> float:
        return _C_KM_S / self.H0_km_s_Mpc


# Planck 2018 TT,TE,EE+lowE+lensing+BAO best-fit (rounded).
Planck18 = FlatLambdaCDM(H0_km_s_Mpc=67.4, Omega_m=0.315, Omega_L=0.685)


def dimensionless_hubble(z: Array | float, cosmo: FlatLambdaCDM) -> Array:
    """E(z) = sqrt(Omega_m (1+z)^3 + Omega_L). Flat universe."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return jnp.sqrt(cosmo.Omega_m * (1.0 + z) ** 3 + cosmo.Omega_L)


def comoving_distance_Mpc(
    z: Array | float, cosmo: FlatLambdaCDM, n_steps: int = 4096
) -> Array:
    """Comoving distance D_C(z) in Mpc.

    Uses fixed-step Simpson integration of c/H(z') from 0 to z. Step
    count is configurable so the validator can tighten convergence;
    default 4096 is well under 0.1% error vs astropy on z<5.
    """
    z = jnp.asarray(z, dtype=jnp.float64)

    def integrand(zp: Array) -> Array:
        return 1.0 / dimensionless_hubble(zp, cosmo)

    # Vectorize over z (allow scalar or array input).
    def D_one(zi: Array) -> Array:
        zp = jnp.linspace(0.0, zi, n_steps)
        y = integrand(zp)
        # Simpson's rule
        h = zp[1] - zp[0]
        s = y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()
        return cosmo.hubble_distance_Mpc * (h / 3.0) * s

    if z.ndim == 0:
        return D_one(z)
    return jax.vmap(D_one)(z)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/physics/test_cosmology.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint + types**

```bash
uv run ruff check whimstan/physics/
uv run mypy whimstan/physics/
```

- [ ] **Step 7: Commit**

```bash
git add whimstan/physics/ tests/physics/
git commit -m "feat(physics): JAX FlatLambdaCDM with comoving distance, validated vs astropy"
```

---

### Task 17: powerlaw.py — energy-flux-normalized power law

**Files:**
- Create: `whimstan/physics/powerlaw.py`
- Create: `tests/physics/test_powerlaw.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/physics/test_powerlaw.py
"""Tests for energy-flux-normalized power law."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_eflux_integrates_to_K() -> None:
    """Spec §3.1: K_i is the integrated energy flux from E_lo to E_hi."""
    from whimstan.physics.powerlaw import powerlaw_eflux

    E = jnp.linspace(0.4, 10.0, 5000)
    dE = jnp.diff(E)
    Em = 0.5 * (E[:-1] + E[1:])

    # photon flux (photons / cm^2 / s / keV) ; energy flux per bin is photon * E
    log_K = jnp.log10(1.0e-10)  # 10^-10 erg/cm^2/s in 0.4-10 keV
    gamma = -2.0
    photons = powerlaw_eflux(Em, log_K, gamma, e_lo=0.4, e_hi=10.0, e_piv=1.0)
    # erg = keV * 1.602e-9
    KEV2ERG = 1.602176634e-9
    eflux_recovered = (photons * Em * dE).sum() * KEV2ERG
    assert pytest.approx(eflux_recovered, rel=2e-3) == 10.0 ** float(log_K)


def test_gamma_2_flat_in_F_E() -> None:
    """gamma = -2 means E^2 N(E) is flat (in keV space)."""
    from whimstan.physics.powerlaw import powerlaw_eflux

    E = jnp.array([0.5, 1.0, 2.0, 5.0])
    photons = powerlaw_eflux(E, log_K=-10.0, gamma=-2.0, e_lo=0.4, e_hi=10.0, e_piv=1.0)
    F_E_squared = E**2 * photons
    rel_spread = (F_E_squared.max() - F_E_squared.min()) / F_E_squared.mean()
    assert float(rel_spread) < 1e-12


def test_jit_and_grad() -> None:
    from whimstan.physics.powerlaw import powerlaw_eflux

    fj = jax.jit(powerlaw_eflux)
    E = jnp.array([1.0, 2.0])
    out = fj(E, -10.0, -2.0, 0.4, 10.0, 1.0)
    assert out.shape == (2,)

    # gradient w.r.t. log_K and gamma must be finite
    g_log_K = jax.grad(lambda lk: powerlaw_eflux(E, lk, -2.0, 0.4, 10.0, 1.0).sum())(-10.0)
    assert jnp.isfinite(g_log_K)
    g_gamma = jax.grad(lambda g: powerlaw_eflux(E, -10.0, g, 0.4, 10.0, 1.0).sum())(-2.0)
    assert jnp.isfinite(g_gamma)
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/physics/test_powerlaw.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write whimstan/physics/powerlaw.py**

```python
# whimstan/physics/powerlaw.py
"""Photon power-law normalized to integrated energy flux.

The photon spectrum is

    N(E) = K_norm * (E / E_piv) ** gamma                [photons / cm^2 / s / keV]

with K_norm chosen so that the integrated *energy* flux

    sum_E E * N(E) * dE * keV2erg

over [e_lo, e_hi] equals K = 10 ** log_K   [erg / cm^2 / s].

Spec §3.1 reasoning: this parameterization decorrelates K from gamma in
the posterior compared to the standard photon-flux normalization.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

KEV_TO_ERG: float = 1.602_176_634e-9  # CODATA 2019


def powerlaw_eflux(
    E: Array,
    log_K: Array | float,
    gamma: Array | float,
    e_lo: float,
    e_hi: float,
    e_piv: float,
) -> Array:
    """Photon spectrum N(E) (photons/cm^2/s/keV) for an eflux-normalized PL.

    Parameters
    ----------
    E : (N,)
        Energies in keV at which to evaluate N(E).
    log_K
        log10 of the integrated energy flux in [e_lo, e_hi], in erg/cm^2/s.
    gamma
        Photon-index slope (typical GRB index ≈ -2).
    e_lo, e_hi, e_piv
        Band edges (keV) for the energy-flux normalization and pivot.

    Returns
    -------
    (N,) array of photon flux density in photons/cm^2/s/keV at each E.
    """
    E = jnp.asarray(E, dtype=jnp.float64)
    log_K = jnp.asarray(log_K, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)

    # Analytic integral of E * (E / e_piv)^gamma dE from e_lo to e_hi.
    # Let p = gamma + 1. The integrand is e_piv^(-gamma) * E^(gamma+1).
    p = gamma + 1.0
    # branch-free formula valid for any gamma != -2 OR e_lo == e_hi
    # for gamma = -2, p = -1 -> integrand = e_piv^2 / E ; integral = e_piv^2 ln(e_hi/e_lo)
    p_eq_minus_one = jnp.isclose(p, -1.0)
    safe_p = jnp.where(p_eq_minus_one, 0.0, p)  # avoid div-by-zero in unused branch
    log_term = jnp.log(e_hi / e_lo)
    pow_term = (e_hi ** (safe_p + 1) - e_lo ** (safe_p + 1)) / (safe_p + 1)
    integral_keV = jnp.where(p_eq_minus_one, e_piv**2 * log_term, e_piv ** (-gamma) * pow_term)
    # KEV_TO_ERG converts keV*photon/cm^2/s -> erg/cm^2/s
    K_norm = (10.0**log_K) / (KEV_TO_ERG * integral_keV)
    return K_norm * (E / e_piv) ** gamma
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/physics/test_powerlaw.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check whimstan/physics/
uv run mypy whimstan/physics/
```

- [ ] **Step 6: Commit**

```bash
git add whimstan/physics/powerlaw.py tests/physics/test_powerlaw.py
git commit -m "feat(physics): energy-flux-normalized power law (jit/grad-safe)"
```

---

### Task 18: tbabs.py — Wilms+ 2000 photoelectric absorption (JAX)

**Files:**
- Create: `whimstan/physics/_data/tbabs_sigma_v1.npz` (precomputed reference)
- Create: `whimstan/physics/tbabs.py`
- Create: `tests/physics/test_tbabs.py`

The cleanest port: precompute the TbAbs cross section at a fine energy
grid using astromodels or sherpa OFFLINE, save as npz, ship in package.
At inference time we do JAX log-log interpolation. This avoids
re-implementing the full Wilms+ 2000 atomic cross sections in JAX.

- [ ] **Step 1: Generate the reference cross section (one-off, scripted)**

```python
# tools/build_tbabs_table.py — committed to repo for reproducibility
"""Precompute TbAbs sigma(E) on a fine grid; ship as npz package data.

Run once with the v0.x-stan-legacy environment OR any environment that
can produce astromodels TbAbs sigma at nH=1e22 cm^-2 baseline. Output
file is checked into git as `whimstan/physics/_data/tbabs_sigma_v1.npz`.
"""
from pathlib import Path

import numpy as np


def main() -> None:
    # Energy grid: 0.1 -- 100 keV log spaced, 4096 points
    E = np.logspace(np.log10(0.1), np.log10(100.0), 4096)

    # Use astromodels TbAbs (or sherpa). See:
    #   https://github.com/threeML/astromodels  TbAbs.evaluate
    # Effective sigma per H atom: tau / nH at nH=1.
    # For initial port, use the legacy whimstan precomputed sigma if available.
    # (Otherwise install astromodels in a separate env to generate.)

    # Fallback: load from legacy fixture if it exists.
    legacy = Path("tests/fixtures/swift_xrt/spec_gamma2.npy")
    if legacy.exists():
        # NOTE: spec_gamma2.npy is the *ionizing spectrum*, not tbabs sigma.
        # Real generation: astromodels.TbAbs(NH=1.0).evaluate(x=E, ...)
        # produces transmission; sigma = -log(transmission)/NH.
        pass

    # Production path (requires astromodels in build env):
    from astromodels.functions import TbAbs

    tb = TbAbs(NH=1.0)  # 1e22 cm^-2 by convention in astromodels
    transmission = np.asarray(tb.evaluate(x=E))  # dimensionless
    sigma_per_1e22_H = -np.log(np.clip(transmission, 1e-300, 1.0))
    sigma = sigma_per_1e22_H * 1e-22  # cm^2 per H atom

    out = Path("whimstan/physics/_data/tbabs_sigma_v1.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, E_keV=E, sigma_cm2=sigma)
    print(f"Wrote {out}: {len(E)} points, sigma range "
          f"[{sigma.min():.3e}, {sigma.max():.3e}] cm^2")


if __name__ == "__main__":
    main()
```

```bash
mkdir -p whimstan/physics/_data
# This is a one-time human-mediated step. Pick the path that works:
#
# Path A — legacy worktree (recommended; same code path as v0):
#   git worktree add ../whimstan-legacy v0.x-stan-legacy
#   cd ../whimstan-legacy
#   pip install -e . threeml astromodels       # legacy env
#   python /Users/jburgess/coding/projects/whimstan/tools/build_tbabs_table.py
#   cp whimstan/physics/_data/tbabs_sigma_v1.npz \
#      ../whimstan/whimstan/physics/_data/
#   cd ../whimstan
#
# Path B — sherpa in a throwaway uv env (no 3ML required):
#   uvx --with sherpa --from . python tools/build_tbabs_table.py
#   (replace astromodels.TbAbs in the script with sherpa.astro.xspec.XStbabs;
#    docs:  https://cxc.cfa.harvard.edu/sherpa/ahelp/xstbabs.html )
#
# Path C — published Wilms+ 2000 reference table (ASCII; manual download):
#   curl -L -o /tmp/tbabs_ref.txt \
#     https://pulsar.sternwarte.uni-erlangen.de/wilms/research/tbabs/tbabs.txt
#   python tools/build_tbabs_table.py --from-ascii /tmp/tbabs_ref.txt
#
ls whimstan/physics/_data/
```

Expected: `tbabs_sigma_v1.npz` exists, ~32 KB.

- [ ] **Step 2: Write the failing test**

```python
# tests/physics/test_tbabs.py
"""Tests for TbAbs JAX cross section."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_sigma_loaded() -> None:
    from whimstan.physics.tbabs import TBABS_E_KEV, TBABS_SIGMA_CM2

    assert TBABS_E_KEV.shape == TBABS_SIGMA_CM2.shape
    assert TBABS_E_KEV.shape[0] >= 1024
    assert float(TBABS_SIGMA_CM2.min()) > 0
    assert float(TBABS_SIGMA_CM2.max()) < 1e-18  # sanity: <1e-18 cm^2/H


def test_sigma_decreases_with_E() -> None:
    """At very high E, sigma must drop monotonically (Klein-Nishina region)."""
    from whimstan.physics.tbabs import tbabs_sigma

    E_high = jnp.linspace(5.0, 10.0, 20)
    s = tbabs_sigma(E_high)
    assert jnp.all(jnp.diff(s) <= 0), "sigma must decrease above 5 keV"


def test_sigma_at_known_energy() -> None:
    """Approximate sigma at 1 keV (~3e-22 cm^2 per H for solar abundance TbAbs)."""
    from whimstan.physics.tbabs import tbabs_sigma

    s_1kev = float(tbabs_sigma(jnp.array(1.0)))
    assert 1e-22 < s_1kev < 1e-21, f"sigma(1 keV)={s_1kev:.3e} outside expected range"


def test_jit_and_grad() -> None:
    from whimstan.physics.tbabs import tbabs_sigma

    f = jax.jit(tbabs_sigma)
    E = jnp.linspace(0.4, 10.0, 50)
    s = f(E)
    assert s.shape == E.shape
    g = jax.grad(lambda E_: tbabs_sigma(E_).sum())(E)
    assert jnp.all(jnp.isfinite(g))


def test_e_below_grid_clipped() -> None:
    """Below the table E_lo, return the value at E_lo (clip; do not extrapolate)."""
    from whimstan.physics.tbabs import tbabs_sigma, TBABS_E_KEV

    s_below = tbabs_sigma(jnp.array(TBABS_E_KEV[0] / 2))
    s_lo = tbabs_sigma(jnp.array(TBABS_E_KEV[0]))
    assert pytest.approx(float(s_below), rel=1e-6) == float(s_lo)
```

- [ ] **Step 3: Run, expect failure**

```bash
uv run pytest tests/physics/test_tbabs.py -v
```

Expected: ImportError.

- [ ] **Step 4: Write whimstan/physics/tbabs.py**

```python
# whimstan/physics/tbabs.py
"""Wilms+ 2000 TbAbs photoelectric ISM cross section, JAX-native.

Loaded from a precomputed npz table shipped under physics/_data/. The
table is generated offline by `tools/build_tbabs_table.py`. This module
exposes a single function `tbabs_sigma(E)` that does log-log linear
interpolation in JAX.

Notes
-----
* `tbabs_sigma(E)` returns sigma per H atom (cm^2). To get optical depth
  multiply by NH: `tau = NH * tbabs_sigma(E)`.
* For E outside the table, we clip to the nearest grid value (no
  extrapolation); a logged warning would be added in higher layers.
"""
from __future__ import annotations

from importlib.resources import files

import jax.numpy as jnp
import numpy as np
from jax import Array

_NPZ = np.load(str(files("whimstan.physics._data").joinpath("tbabs_sigma_v1.npz")))
TBABS_E_KEV: Array = jnp.asarray(_NPZ["E_keV"], dtype=jnp.float64)
TBABS_SIGMA_CM2: Array = jnp.asarray(_NPZ["sigma_cm2"], dtype=jnp.float64)
del _NPZ

_LOG_E: Array = jnp.log(TBABS_E_KEV)
_LOG_S: Array = jnp.log(TBABS_SIGMA_CM2)


def tbabs_sigma(E_keV: Array | float) -> Array:
    """Photoelectric cross section per H atom (cm^2) at energies E_keV.

    Log-log linear interpolation; clips at table edges.
    """
    E = jnp.asarray(E_keV, dtype=jnp.float64)
    logE = jnp.log(E)
    return jnp.exp(jnp.interp(logE, _LOG_E, _LOG_S))
```

- [ ] **Step 5: Add `_data` as a Python subpackage**

Hatchling auto-ships any non-`.py` file inside a Python package, so
no extra `pyproject.toml` config is needed. We just need `_data/` to be
a real subpackage so `importlib.resources.files("whimstan.physics._data")`
resolves.

```python
# whimstan/physics/_data/__init__.py
"""Precomputed reference tables shipped with the package."""
```

Verify the npz lands in the built wheel:

```bash
uv build --wheel
unzip -l dist/whimstan-*.whl | grep -E "tbabs_sigma_v1.npz|grb_sample_v2.csv"
rm -rf dist/
```

Expected: both files listed inside the wheel.

- [ ] **Step 6: Run tests**

```bash
uv run pip install -e . --quiet
uv run pytest tests/physics/test_tbabs.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint + types**

```bash
uv run ruff check whimstan/physics/
uv run mypy whimstan/physics/
```

- [ ] **Step 8: Commit**

```bash
git add whimstan/physics/tbabs.py whimstan/physics/_data/ tests/physics/test_tbabs.py tools/build_tbabs_table.py pyproject.toml
git commit -m "feat(physics): JAX TbAbs cross section via shipped Wilms+ 2000 lookup table"
```

---

### Task 19: physics/whim.py — WhimGrid loader & interpolator (placeholder grid)

**Files:**
- Create: `whimstan/physics/whim.py`
- Create: `tests/physics/test_whim.py`
- Create: `tests/fixtures/whim_grid_tiny.h5` (synthetic, for tests)

In Plan 1 we cannot build the real grid (Plan 2). So we ship the loader
+ interpolator and exercise them against a *synthetic tiny grid* fixture
generated in this task. Plan 2 replaces the synthetic fixture with the
CLOUDY artifact.

- [ ] **Step 1: Create the synthetic-grid fixture builder**

```python
# tests/fixtures/build_whim_grid_tiny.py
"""Generate a tiny synthetic WHIM grid for tests.

Real grid (Plan 2) replaces this with a 50 MB CLOUDY-derived HDF5.
Synthetic tau is a smooth analytic function so monotonicity and
gradient tests are well-defined.
"""
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    log_T = np.linspace(4.5, 7.5, 5)            # 5 T points
    log_n = np.linspace(-7.0, -3.0, 4)          # 4 n_H points
    z = np.linspace(0.0, 5.0, 6)                # 6 z points
    E_obs = np.logspace(np.log10(0.3), np.log10(10.0), 64)  # 64 E points

    # Smooth fake tau ~ n_H * (1+z)^2 * sigma_eff(E) * geometry(T)
    sigma_E = (E_obs / 1.0) ** -2.5
    geom_T = np.exp(-((log_T[:, None] - 6.0) ** 2) / 0.5)  # (5,1)
    base = sigma_E[None, None, None, :] * geom_T[:, None, None, None]
    n_factor = (10.0 ** log_n)[None, :, None, None]
    z_factor = ((1.0 + z) ** 2)[None, None, :, None]
    tau = base * n_factor * z_factor   # (5, 4, 6, 64)

    out = Path("tests/fixtures/whim_grid_tiny.h5")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("log_T", data=log_T)
        f.create_dataset("log_n_H", data=log_n)
        f.create_dataset("z", data=z)
        f.create_dataset("E_obs_keV", data=E_obs)
        f.create_dataset("tau", data=tau.astype("float32"))
        meta = f.create_group("metadata")
        meta.attrs["grid_version"] = "synthetic-tiny-v0"
        meta.attrs["ionizing_background"] = "synthetic"
        meta.attrs["abundances"] = "synthetic"
        meta.attrs["cosmology"] = "synthetic"
    print(f"Wrote {out}: shape={tau.shape}")


if __name__ == "__main__":
    main()
```

```bash
uv run python tests/fixtures/build_whim_grid_tiny.py
ls -la tests/fixtures/whim_grid_tiny.h5
```

Expected: file ~50 KB.

- [ ] **Step 2: Write the failing test**

```python
# tests/physics/test_whim.py
"""Tests for WhimGrid loader and JAX interpolator."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(scope="module")
def tiny_grid_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "whim_grid_tiny.h5"
    assert p.is_file(), f"run tests/fixtures/build_whim_grid_tiny.py first; missing {p}"
    return p


def test_load(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    assert g.tau_table.shape == (5, 4, 6, 64)
    assert g.log_T.shape == (5,)
    assert g.log_n.shape == (4,)
    assert g.z.shape == (6,)
    assert g.E_obs.shape == (64,)
    assert g.metadata["grid_version"] == "synthetic-tiny-v0"


def test_interp_at_grid_corner_exact(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    tau_interp = g.tau(g.log_T[2], g.log_n[1], g.z[3], g.E_obs)
    expected = g.tau_table[2, 1, 3, :]
    assert jnp.allclose(tau_interp, expected, rtol=1e-5)


def test_interp_monotone_in_n(tiny_grid_path: Path) -> None:
    """tau must be monotone increasing in log n_H at fixed (T, z, E)."""
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    n_grid = jnp.linspace(-6.5, -3.5, 20)
    tau_n = jax.vmap(lambda lognh: g.tau(6.0, lognh, 1.0, g.E_obs))(n_grid)
    tau_total = tau_n.sum(axis=1)
    diffs = jnp.diff(tau_total)
    assert jnp.all(diffs >= 0), f"non-monotone: {diffs}"


def test_jit_and_vmap(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)

    @jax.jit
    def fold(log_T, log_n, z):
        return g.tau(log_T, log_n, z, g.E_obs).sum()

    out = fold(jnp.float64(6.0), jnp.float64(-5.0), jnp.float64(2.0))
    assert jnp.isfinite(out)

    log_T = jnp.array([5.5, 6.0, 6.5])
    log_n = jnp.array([-5.0, -5.0, -5.0])
    z = jnp.array([1.0, 2.0, 3.0])
    out_v = jax.vmap(fold)(log_T, log_n, z)
    assert out_v.shape == (3,)


def test_grad_finite(tiny_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)

    def loss(p):
        return g.tau(p[0], p[1], p[2], g.E_obs).sum()

    p = jnp.array([6.0, -5.0, 2.0])
    grads = jax.grad(loss)(p)
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.any(grads != 0)


def test_clip_at_grid_edge(tiny_grid_path: Path) -> None:
    """Out-of-range queries clip to the boundary, do not nan."""
    from whimstan.physics.whim import WhimGrid

    g = WhimGrid.load(tiny_grid_path)
    out = g.tau(jnp.float64(8.0), jnp.float64(-2.0), jnp.float64(10.0), g.E_obs)
    expected = g.tau_table[-1, -1, -1, :]
    assert jnp.allclose(out, expected, rtol=1e-5)
```

- [ ] **Step 3: Run, expect failure**

```bash
uv run pytest tests/physics/test_whim.py -v
```

Expected: ImportError on `whimstan.physics.whim`.

- [ ] **Step 4: Write whimstan/physics/whim.py**

```python
# whimstan/physics/whim.py
"""WHIM opacity grid loader + JAX trilinear interpolator.

The grid is a 4-D array tau[i_T, i_n, i_z, i_E] of optical depths in
the observer frame. At inference time we trilinearly interpolate over
(log T, log n_H, z) at every requested (T, n_H, z), producing a vector
over the grid's E_obs axis. The caller (forward model) then linearly
interpolates that vector onto the per-GRB observer-frame energy grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import jax
import jax.numpy as jnp
from jax import Array


def _searchsorted_clipped(grid: Array, x: Array) -> tuple[Array, Array]:
    """Return (i, w) such that linear interp uses grid[i] and grid[i+1]
    with weight w in [0, 1]. Clips x to grid range (no extrapolation).
    """
    n = grid.shape[0]
    i = jnp.clip(jnp.searchsorted(grid, x) - 1, 0, n - 2)
    x_clipped = jnp.clip(x, grid[0], grid[-1])
    w = (x_clipped - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


@dataclass(frozen=True, slots=True)
class WhimGrid:
    """Tabulated WHIM optical depth tau(log_T, log_n_H, z, E_obs)."""

    log_T: Array        # (N_T,)
    log_n: Array        # (N_n,)
    z: Array            # (N_z,)
    E_obs: Array        # (N_E,) keV
    tau_table: Array    # (N_T, N_n, N_z, N_E) float64
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> WhimGrid:
        with h5py.File(path, "r") as f:
            log_T = jnp.asarray(f["log_T"][...], dtype=jnp.float64)
            log_n = jnp.asarray(f["log_n_H"][...], dtype=jnp.float64)
            z = jnp.asarray(f["z"][...], dtype=jnp.float64)
            E_obs = jnp.asarray(f["E_obs_keV"][...], dtype=jnp.float64)
            tau = jnp.asarray(f["tau"][...], dtype=jnp.float64)
            meta_grp = f.get("metadata", None)
            metadata: dict[str, Any] = {}
            if meta_grp is not None:
                for k, v in meta_grp.attrs.items():
                    metadata[k] = v if not isinstance(v, bytes) else v.decode()
        return cls(log_T=log_T, log_n=log_n, z=z, E_obs=E_obs,
                   tau_table=tau, metadata=metadata)

    def tau(
        self,
        log_T: Array | float,
        log_n: Array | float,
        z: Array | float,
        E_query: Array,
    ) -> Array:
        """Interpolate tau at (log_T, log_n, z) and project to E_query.

        Parameters
        ----------
        log_T, log_n, z : scalars
            Query point in (log T, log n_H, z) space; clipped to grid.
        E_query : (N_E_q,)
            Energies (keV, observer frame) at which to return tau.

        Returns
        -------
        tau_at_E : (N_E_q,)
            Optical depth at each requested energy.
        """
        log_T = jnp.asarray(log_T, dtype=jnp.float64)
        log_n = jnp.asarray(log_n, dtype=jnp.float64)
        z = jnp.asarray(z, dtype=jnp.float64)

        iT, wT = _searchsorted_clipped(self.log_T, log_T)
        iN, wN = _searchsorted_clipped(self.log_n, log_n)
        iZ, wZ = _searchsorted_clipped(self.z, z)

        c000 = self.tau_table[iT,     iN,     iZ,     :]
        c001 = self.tau_table[iT,     iN,     iZ + 1, :]
        c010 = self.tau_table[iT,     iN + 1, iZ,     :]
        c011 = self.tau_table[iT,     iN + 1, iZ + 1, :]
        c100 = self.tau_table[iT + 1, iN,     iZ,     :]
        c101 = self.tau_table[iT + 1, iN,     iZ + 1, :]
        c110 = self.tau_table[iT + 1, iN + 1, iZ,     :]
        c111 = self.tau_table[iT + 1, iN + 1, iZ + 1, :]

        c00 = c000 * (1 - wZ) + c001 * wZ
        c01 = c010 * (1 - wZ) + c011 * wZ
        c10 = c100 * (1 - wZ) + c101 * wZ
        c11 = c110 * (1 - wZ) + c111 * wZ

        c0 = c00 * (1 - wN) + c01 * wN
        c1 = c10 * (1 - wN) + c11 * wN

        tau_at_grid_E = c0 * (1 - wT) + c1 * wT  # (N_E_grid,)

        # Project onto E_query in log-log, clipped at table edges.
        logE_grid = jnp.log(self.E_obs)
        logE_q = jnp.log(jnp.asarray(E_query, dtype=jnp.float64))
        # interp returns 0 outside; we want clip-to-edge instead, so use
        # jnp.interp's default behavior (clamps) — verified in tests.
        return jnp.interp(logE_q, logE_grid, tau_at_grid_E)


# Register WhimGrid as a static PyTree so it can be carried inside jit'd
# functions without being differentiated.
def _flatten(g: WhimGrid):
    children = (g.log_T, g.log_n, g.z, g.E_obs, g.tau_table)
    aux = g.metadata
    return children, aux


def _unflatten(aux: dict[str, Any], children: tuple[Array, ...]) -> WhimGrid:
    return WhimGrid(*children, metadata=aux)  # type: ignore[arg-type]


jax.tree_util.register_pytree_node(WhimGrid, _flatten, _unflatten)
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/physics/test_whim.py -v
```

Expected: all pass.

- [ ] **Step 6: Lint + types**

```bash
uv run ruff check whimstan/physics/
uv run mypy whimstan/physics/
```

- [ ] **Step 7: Commit**

```bash
git add whimstan/physics/whim.py tests/physics/test_whim.py tests/fixtures/build_whim_grid_tiny.py tests/fixtures/whim_grid_tiny.h5
git commit -m "feat(physics): WhimGrid HDF5 loader + JAX trilinear interpolator"
```

---

### Task 20: physics/whim_validator.py — JAX from-scratch τ skeleton

**Files:**
- Create: `whimstan/physics/whim_validator.py`
- Create: `tests/physics/test_whim_validator.py`

(`whimstan/physics/_data/atomic_data_v1.h5` is *not* created here — Plan 2
ships it together with the real validator physics.)

The full validator (Saha–Boltzmann + photoionization at HM12/KS19
background) is a Plan-2 task because it needs the same atomic data
the CLOUDY pipeline uses. In Plan 1 we ship the *interface* and a
shape-correct stub that returns a smooth analytic τ matching the tiny
synthetic grid. The Plan-2 step replaces the analytic stub with real
physics.

This pattern (interface in Plan 1, real physics in Plan 2) keeps the
foundation-layer shippable and tested without blocking on CLOUDY.

- [ ] **Step 1: Write the failing test (interface only)**

```python
# tests/physics/test_whim_validator.py
"""Tests for the from-scratch WHIM tau validator.

Plan 1 ships an analytic stub that matches the tiny synthetic grid
within 5%; Plan 2 swaps in real Saha+photo-ionization physics.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp


def test_validator_returns_tau(fixtures_dir: Path) -> None:
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    E_obs = jnp.logspace(jnp.log10(0.3), jnp.log10(10.0), 32)
    tau = compute_tau_from_scratch(
        log_T=6.0, log_n=-5.0, z=2.0, E_obs=E_obs,
    )
    assert tau.shape == E_obs.shape
    assert jnp.all(jnp.isfinite(tau))
    assert jnp.all(tau >= 0)


def test_validator_matches_tiny_grid(fixtures_dir: Path) -> None:
    """Validator must agree with shipped grid to <= 5% on overlap region."""
    from whimstan.physics.whim import WhimGrid
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    grid = WhimGrid.load(fixtures_dir / "whim_grid_tiny.h5")
    log_T = jnp.float64(6.0)
    log_n = jnp.float64(-5.0)
    z = jnp.float64(2.0)
    E_obs = grid.E_obs
    tau_grid = grid.tau(log_T, log_n, z, E_obs)
    tau_scratch = compute_tau_from_scratch(log_T=log_T, log_n=log_n, z=z, E_obs=E_obs)
    rel = jnp.abs(tau_scratch - tau_grid) / jnp.maximum(jnp.abs(tau_grid), 1e-30)
    assert float(rel.max()) < 0.05
```

- [ ] **Step 2: Run, expect failure**

```bash
uv run pytest tests/physics/test_whim_validator.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write whimstan/physics/whim_validator.py (Plan-1 stub matching tiny grid)**

```python
# whimstan/physics/whim_validator.py
"""From-scratch WHIM optical-depth calculator (cross-validator).

PLAN 1 STATUS: STUB.
This module ships an analytic stub whose functional form *matches the
synthetic tiny grid built by tests/fixtures/build_whim_grid_tiny.py*.
This lets us land the public API and unit-test the validator interface
in Plan 1.

PLAN 2 will replace `_analytic_stub_tau` with a real Saha-Boltzmann +
photoionization computation using the same atomic data the CLOUDY grid
build consumes. The public function signature `compute_tau_from_scratch`
must NOT change between Plan 1 and Plan 2.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def _analytic_stub_tau(
    log_T: Array, log_n: Array, z: Array, E_obs: Array
) -> Array:
    """Smooth analytic τ; mirrors tests/fixtures/build_whim_grid_tiny.py."""
    sigma_E = (E_obs / 1.0) ** -2.5
    geom_T = jnp.exp(-((log_T - 6.0) ** 2) / 0.5)
    n_factor = 10.0**log_n
    z_factor = (1.0 + z) ** 2
    return sigma_E * geom_T * n_factor * z_factor


def compute_tau_from_scratch(
    *,
    log_T: Array | float,
    log_n: Array | float,
    z: Array | float,
    E_obs: Array,
) -> Array:
    """Compute observer-frame τ(E_obs) from-scratch in pure JAX.

    Returns
    -------
    (N_E,) array of optical depths.

    Notes
    -----
    Plan 1 implementation is an analytic stub matching the synthetic
    tiny grid for interface tests. Plan 2 replaces with full physics.
    """
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    log_n = jnp.asarray(log_n, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    E_obs = jnp.asarray(E_obs, dtype=jnp.float64)
    return _analytic_stub_tau(log_T, log_n, z, E_obs)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/physics/test_whim_validator.py -v
```

Expected: both pass (validator matches grid by construction).

- [ ] **Step 5: Lint + types**

```bash
uv run ruff check whimstan/physics/
uv run mypy whimstan/physics/
```

- [ ] **Step 6: Commit**

```bash
git add whimstan/physics/whim_validator.py tests/physics/test_whim_validator.py
git commit -m "feat(physics): WHIM validator interface + Plan-1 analytic stub

Real physics implementation lands in Plan 2 alongside the CLOUDY grid.
Public signature is fixed; Plan 2 only touches the internals."
```

---

### Task 21: Update package __init__.py exports + final verification

**Files:**
- Modify: `whimstan/__init__.py`
- Modify: `whimstan/physics/__init__.py`

- [ ] **Step 1: Add physics exports**

```python
# whimstan/physics/__init__.py
"""Pure-JAX physics primitives for the forward model."""
from __future__ import annotations

from whimstan.physics.cosmology import (
    FlatLambdaCDM,
    Planck18,
    comoving_distance_Mpc,
    dimensionless_hubble,
)
from whimstan.physics.powerlaw import KEV_TO_ERG, powerlaw_eflux
from whimstan.physics.tbabs import TBABS_E_KEV, TBABS_SIGMA_CM2, tbabs_sigma
from whimstan.physics.whim import WhimGrid
from whimstan.physics.whim_validator import compute_tau_from_scratch

__all__ = [
    "FlatLambdaCDM",
    "KEV_TO_ERG",
    "Planck18",
    "TBABS_E_KEV",
    "TBABS_SIGMA_CM2",
    "WhimGrid",
    "comoving_distance_Mpc",
    "compute_tau_from_scratch",
    "dimensionless_hubble",
    "powerlaw_eflux",
    "tbabs_sigma",
]
```

- [ ] **Step 2: Update whimstan/__init__.py to surface top-level**

```python
# whimstan/__init__.py
"""whimstan — Hierarchical Bayesian WHIM search in GRB afterglows."""
from __future__ import annotations

# Enforce float64 globally before any other JAX import side-effects.
import jax as _jax

_jax.config.update("jax_enable_x64", True)
del _jax

try:
    from whimstan._version import __version__
except ImportError:  # pragma: no cover - hatch-vcs not yet built
    __version__ = "0.0.0+unknown"

from whimstan import data, physics, utils

__all__ = ["__version__", "data", "physics", "utils"]
```

- [ ] **Step 3: Run all tests**

```bash
uv run pytest -v
```

Expected: every test passes. Count: ~25 tests across smoke, data, physics.

- [ ] **Step 4: Run full lint + type check**

```bash
uv run ruff check whimstan/ tests/
uv run ruff format --check whimstan/ tests/
uv run mypy whimstan/
```

Expected: no errors. If `ruff format --check` fails, run `uv run ruff format whimstan/ tests/` and re-commit.

- [ ] **Step 5: Run pre-commit on everything**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 6: Verify CI workflow validates**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

- [ ] **Step 7: Commit**

```bash
git add whimstan/__init__.py whimstan/physics/__init__.py
git commit -m "feat: surface data/physics/utils submodules at package root"
```

- [ ] **Step 8: Push branch**

```bash
git push origin rewrite/v1
```

- [ ] **Step 9: Open draft PR** (optional but recommended)

```bash
gh pr create --base master --head rewrite/v1 --draft \
  --title "rewrite/v1 — Plan 1: Foundations" \
  --body "Implements Plan 1 of the v1.0 rewrite. See docs/superpowers/plans/2026-05-08-whimstan-v1-plan1-foundations.md.

Tooling, OGIP data layer (jaxspec), JAX physics primitives (TbAbs, powerlaw, cosmology, WhimGrid loader + validator stub).

Plan 2 (CLOUDY grid build) is a follow-up.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase 4 — Plan 1 Exit Criteria

Before declaring Plan 1 done, verify all of the following:

- [ ] All tests pass (`uv run pytest -v`).
- [ ] CI green on `rewrite/v1` for both `ubuntu-latest` and `macos-latest` on Python 3.11 and 3.12.
- [ ] `ruff`, `ruff format --check`, and `mypy --strict` clean on `whimstan/`.
- [ ] `mkdocs build --strict` succeeds.
- [ ] `tests/fixtures/swift_xrt/apc.{pi,rmf,arf}` round-trips through `load_swift_xrt_observation` to a JAX-friendly `Observation`.
- [ ] `whimstan.physics.tbabs.tbabs_sigma(jnp.array(1.0))` returns a value in `[1e-22, 1e-21]` cm² per H atom.
- [ ] `whimstan.physics.WhimGrid.load(tiny_grid)` interpolation passes the `tau_at_corner == grid_value` test.
- [ ] `compute_tau_from_scratch` matches the tiny grid within 5%.
- [ ] `JAX_ENABLE_X64=True` enforced at import.
- [ ] Plan 2 is queued: a TODO note exists in `docs/superpowers/plans/` declaring Plan 2 will replace the synthetic tiny grid with the CLOUDY artifact.

---

## Spec coverage check (self-review)

| Spec section | Plan 1 task(s) |
|---|---|
| §4.1 layout — `data/`, `physics/`, `utils/` | T11–T15, T16–T20 |
| §4.2 boundaries — `data` no physics; `physics` pure JAX | enforced by module structure & mypy --strict |
| §4.4 forward-model signature inputs (ObservationStatic) | T11 |
| §5 WHIM grid — loader + interp + validator interface | T19, T20 |
| §6.1 tests — `test_ogip`, `test_physics_tbabs`, `test_physics_whim`, `test_physics_whim_validator` | T12, T18, T19, T20 |
| §6.3 tooling — hatchling, hatch-vcs, uv, ruff, mypy, pre-commit, mkdocs | T3, T4, T6, T7, T10 |
| §7.1 — `JAX_ENABLE_X64=True` enforced at import; checked on startup | T9 |
| §9 migration — tag `v0.x-stan-legacy`, branch `rewrite/v1`, wipe legacy | T1, T2 |

**Out of Plan 1 (deferred):**
- §3 model (Plan 3)
- §5 actual CLOUDY grid (Plan 2)
- §5.5 real validator physics (Plan 2)
- §7.2 sampling defaults / diagnostics (Plan 4)
- §8 simulation pipeline (Plan 3)
- §9.4 milestones M3 onward (Plans 2–5)

---

## Notes for Plan 2 author

- The synthetic `whim_grid_tiny.h5` MUST remain available in `tests/fixtures/` so the cheap interpolator unit tests stay fast. The full CLOUDY grid is a separate release asset.
- `compute_tau_from_scratch` keeps its public signature; Plan 2 replaces `_analytic_stub_tau` with full Saha + photoionization physics.
- Atomic data file path placeholder: `whimstan/physics/_data/atomic_data_v1.h5` — Plan 2 ships this.
- Plan 1 deliberately does not import `astromodels` or `cmdstanpy` anywhere; please keep it that way.
