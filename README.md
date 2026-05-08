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
