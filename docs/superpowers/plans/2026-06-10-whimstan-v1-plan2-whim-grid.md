# whimstan v1.0 — Plan 2: WHIM Opacity Grid (M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the production CLOUDY-derived hybrid CIE+PIE WHIM opacity grid (`whim_opacity_grid_v1.h5`, ~45 MB) and replace the Plan-1 validator stub with a real pure-JAX Saha/rate-balance + photoionization τ calculator that agrees with the CLOUDY grid to ≤5%.

**Architecture:** CLOUDY (c23.01, built from source under `tools/build_whim_grid/`, gitignored) computes ion fractions on a (log T × log n_local × z) grid using the KS19 ionizing background. `assemble.py` combines those fractions with Verner & Yakovlev 1995 (VY95) photoionization cross sections and Asplund+2009 abundances, performs the cosmological shell integral, and writes the HDF5 grid that `WhimGrid.load` (Plan 1) already reads. The JAX validator solves the same hybrid ionization balance from scratch using CHIANTI-derived recombination rates + CIE tables and the *same* VY95 σ and KS19 spectrum, so the 5% gate specifically cross-checks the ionization balance and the shell integral — the two places where bugs would corrupt science.

**Tech Stack:** CLOUDY c23.01, JAX, numpy, h5py, joblib, CHIANTI database (GitHub mirror, derived tables committed), VizieR (VY95 fit parameters, derived table committed).

**Spec:** `docs/superpowers/specs/2026-05-08-whimstan-v1-design.md` §5 (grid axes, physics inputs, build pipeline, validator, API).

---

## Decisions locked by this plan

These were not pinned by the spec; they are pinned here. Flag to the user if any looks wrong.

1. **Metallicity = 1.0 solar** (Asplund+2009 pattern *and* amplitude), recorded in grid
   metadata. For a hot WHIM the X-ray τ is metal-dominated, so τ scales ~linearly with Z
   and is therefore degenerate with n₀; choosing solar makes the n₀ posterior interpretable
   as "n₀ × (Z/Z_⊙)". Documented in the grid README and metadata.
2. **Ratio-ladder hybrid balance** for the validator. Per ionization rung *s* of element X
   (X^s+ ⇌ X^(s+1)+):
   `f_{s+1}/f_s = R_cie_s(T) + Γ_s(z) / (n_e · α_s(T))`
   where `R_cie` is the CIE ratio taken from CHIANTI's `chianti.ioneq` (so the CIE limit is
   exactly CHIANTI's), `α_s` = RR+DR recombination from CHIANTI's Badnell fit-parameter
   files, and `Γ_s` = photoionization rate from the KS19 spectrum × VY95 σ. This needs *no*
   collisional-ionization rates (they're implicit in `R_cie · α`) and no ChiantiPy/XUVTOP.
3. **Cross sections shared:** both `assemble.py` (grid build) and the validator use the
   committed VY95 per-shell fit table. CLOUDY's internal σ only enters via the ion
   fractions. The 5% gate therefore tests CLOUDY-vs-JAX *ionization balance* + integration,
   not cross-section provenance.
4. **UVB extracted from CLOUDY itself:** the JAX-side KS19 J_ν(z) table is produced by
   running CLOUDY with the chosen UVB command and `save incident continuum` per z. The
   validator and the grid see byte-identical radiation fields by construction.
5. **CLOUDY abundance file `solar_GASS10.abn`** (Grevesse/Asplund/Sauval/Scott 2010);
   differences from Asplund+2009 are ≤0.01 dex for our 12 elements — inside tolerance.
   The JAX side uses Asplund+2009 numbers per spec.
6. **CLOUDY local-density axis:** log n_local ∈ [−7.0, −0.5] step 0.25 (27 points). Needed
   because the shell at redshift z′ has n_local = n₀(1+z′)³ ≤ 10⁻³·(6)³ ≈ 10^−0.67.
7. **Sub-shell integration step Δz′ = 0.02** (251 points to z=5); the stored z_s axis
   (Δz=0.2) falls exactly on sub-shell nodes, so cumulative trapezoid gives the 26 stored
   τ(z_s) slices with no extra interpolation.
8. **5% gate masks τ_grid > 10⁻³**: relative error on negligible optical depths is
   meaningless for the likelihood (e^−τ ≈ 1 − τ); the mask is recorded in the test docstring.
9. **Grid-absent behavior = `pytest.skip`** with a clear message (spec said "xfail"; skip is
   the semantically correct pytest idiom for "required external asset missing" and shows up
   loudly in the `-ra` summary that `addopts` already enables).

## External data sources

| What | Primary source | Fallback | Committed artifact |
|---|---|---|---|
| CLOUDY c23.01 | `https://data.nublado.org/cloudy_releases/c23/c23.01.tar.gz` | — | nothing (gitignored build dir) |
| KS19 UVB | CLOUDY built-in `TABLE …` / shipped SED (Task 2 spike decides) | KS19 author files; else STOP and ask user (HM12 is a user decision) | `whimstan/physics/_data/ks19_uvb_v1.npz` |
| VY95 σ fits | VizieR `https://cdsarc.cds.unistra.fr/ftp/J/A+AS/109/125/` | `https://www.pa.uky.edu/~verner/dima/photo/phfit2.f` (DATA blocks) | `whimstan/physics/_data/vy95_phfit_v1.npz` |
| CHIANTI ioneq + rr/drparams | GitHub `chianti-atomic/CHIANTI` raw files | chiantidatabase.org tarball | `whimstan/physics/_data/ion_rates_v1.npz` |
| Asplund+2009 abundances | values inlined in this plan (verified) | — | `whimstan/physics/abundances.py` |

All `tools/` build scripts are committed and re-runnable; only their small derived `.npz`
outputs are committed under `whimstan/physics/_data/` (same pattern as Plan 1's
`tbabs_sigma_v1.npz`). Cite VY95, CHIANTI, KS19 in module docstrings.

## Element/ion registry (used everywhere — single source of truth in `atomic.py`)

12 elements: H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe (Z = 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 26).

- **140-layout** ("absorbing ions", have ≥1 electron): per element, stages s = 0..Z−1
  (s = electrons removed). Used by σ table, α table, R_cie table, Γ table.
  N_IONS = ΣZ = 140.
- **152-layout** ("all stages", includes bare nucleus): per element, stages s = 0..Z.
  Used by ion-fraction vectors. N_STAGES = Σ(Z+1) = 152.

Element order is the tuple order above, always.

---

# Phase 0 — CLOUDY infrastructure (Tasks 1–2)

### Task 1: CLOUDY download + build via Makefile

**Files:**
- Create: `tools/build_whim_grid/Makefile`
- Create: `tools/build_whim_grid/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the Makefile**

```makefile
# Builds CLOUDY c23.01 from source and provides grid-pipeline entry points.
# Everything heavyweight lives under this directory and is gitignored.

CLOUDY_VERSION := c23.01
CLOUDY_URL     := https://data.nublado.org/cloudy_releases/c23/$(CLOUDY_VERSION).tar.gz
CLOUDY_DIR     := $(CURDIR)/$(CLOUDY_VERSION)
CLOUDY_EXE     := $(CLOUDY_DIR)/source/cloudy.exe
export CLOUDY_DATA_PATH := $(CLOUDY_DIR)/data

NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc)

.PHONY: cloudy smoke grid assemble clean-runs

cloudy: $(CLOUDY_EXE)

$(CLOUDY_EXE):
	curl -fL -o $(CLOUDY_VERSION).tar.gz $(CLOUDY_URL)
	tar xzf $(CLOUDY_VERSION).tar.gz
	$(MAKE) -C $(CLOUDY_DIR)/source -j$(NPROC)

smoke: $(CLOUDY_EXE)
	printf 'test\n' | $(CLOUDY_EXE) && echo "CLOUDY SMOKE OK"

grid: $(CLOUDY_EXE)
	cd ../.. && uv run python tools/build_whim_grid/run_grid.py \
	    --cloudy-exe $(CLOUDY_EXE) --out tools/build_whim_grid/runs

assemble:
	cd ../.. && uv run python tools/build_whim_grid/assemble.py \
	    --runs tools/build_whim_grid/runs --out tools/build_whim_grid/dist

clean-runs:
	rm -rf runs dist
```

- [ ] **Step 2: Gitignore the heavy artifacts**

Append to `.gitignore`:

```
# CLOUDY build + grid pipeline outputs (regenerable via tools/build_whim_grid/Makefile)
tools/build_whim_grid/c23.*/
tools/build_whim_grid/*.tar.gz
tools/build_whim_grid/runs/
tools/build_whim_grid/dist/
```

- [ ] **Step 3: Write README.md**

Short doc: what the pipeline produces, the `make cloudy`, `make smoke`, `make grid`,
`make assemble` targets, expected runtimes (build ~10–20 min; grid ~2–4 h on 8 cores),
the metallicity note from "Decisions" §1, and the grid release-asset name.

- [ ] **Step 4: Build and smoke-test**

Run: `make -C tools/build_whim_grid cloudy && make -C tools/build_whim_grid smoke`
Expected: compile succeeds (clang on macOS is supported by c23.01), smoke prints
"Cloudy exited OK" in the test output and `CLOUDY SMOKE OK`.
Timeout note: download ~1 GB + compile ~15 min — run with a 600000 ms timeout, or
in the background.

- [ ] **Step 5: Commit**

```bash
git add tools/build_whim_grid/Makefile tools/build_whim_grid/README.md .gitignore
git commit -m "build(grid): CLOUDY c23.01 download+build Makefile for WHIM grid pipeline"
```

### Task 2: UVB reconnaissance spike → `uvb.py` + NOTES.md

The spec pins KS19 (Khaire & Srianand 2019). Whether c23.01 exposes it as a built-in
`TABLE` command or as a shipped SED file is unknown — resolve it empirically.

**Files:**
- Create: `tools/build_whim_grid/uvb.py`
- Create: `tools/build_whim_grid/NOTES.md`

- [ ] **Step 1: Search the CLOUDY tree**

```bash
ls tools/build_whim_grid/c23.01/data/SED/ | grep -iE "ks1|khaire"
grep -rli "khaire" tools/build_whim_grid/c23.01/data/ | head
grep -rli "khaire\|KS18\|KS19" tools/build_whim_grid/c23.01/docs/latex/hazy1/ 2>/dev/null | head
```

- [ ] **Step 2: Decide the UVB command, in priority order**

1. Built-in table (e.g. `table KS19 redshift {z}` / `table KS18 redshift {z}`) if Hazy
   documents one — redshift-interpolating, preferred.
2. Shipped SED file via `table SED "<file>" redshift {z}` if data/SED has a KS19 grid file.
3. Download the KS19 CLOUDY-format files from the authors (search: Khaire & Srianand 2019
   MNRAS 484, 4174 data; Vikram Khaire's site / GitHub) into
   `tools/build_whim_grid/c23.01/data/SED/` and use `table SED`.
4. None of the above works → **STOP, report NEEDS_CONTEXT**. Falling back to HM12 changes
   the science and is the user's call (spec §5.3 allows HM12 only as a comparison config).

- [ ] **Step 3: Verify with a one-point CLOUDY run**

Write `/tmp/uvbtest.in`:

```
title uvb smoke
<uvb command for z=2.0>
hden -5
constant temperature, t=6.0 log
stop zone 1
set dr 0
save incident continuum units keV "uvbtest.con" last
```

Run: `cd /tmp && CLOUDY_DATA_PATH=<repo>/tools/build_whim_grid/c23.01/data <CLOUDY_EXE> < uvbtest.in > uvbtest.out`
Expected: "Cloudy exited OK"; `uvbtest.con` has two columns (E [keV], 4πνJ_ν
[erg cm⁻² s⁻¹]) — confirm the exact header/units by inspection and record them.

- [ ] **Step 4: Write `uvb.py`** — the single place the UVB command string lives:

```python
"""UVB command used by every CLOUDY run in this pipeline.

Decided by the Task-2 spike (see NOTES.md). Both run_grid.py and
tools/build_uvb_table.py import this so the grid and the JAX validator
see the identical radiation field.
"""

UVB_NAME = "KS19"  # goes into grid metadata


def uvb_command(z: float) -> str:
    return f"table KS19 redshift {z:.4f}"  # ADJUST to the spike's finding
```

- [ ] **Step 5: Write NOTES.md** recording: what was found where, the exact command,
the `save incident continuum` column format, and the CLOUDY version banner line.

- [ ] **Step 6: Commit**

```bash
git add tools/build_whim_grid/uvb.py tools/build_whim_grid/NOTES.md
git commit -m "feat(grid): pin KS19 UVB command for CLOUDY pipeline (spike findings in NOTES)"
```

---

# Phase 1 — Atomic data foundation (Tasks 3–6)

### Task 3: VY95 photoionization cross sections → `physics/atomic.py`

**Files:**
- Create: `tools/build_vy95_table.py`
- Create: `whimstan/physics/_data/vy95_phfit_v1.npz`
- Create: `whimstan/physics/atomic.py`
- Test: `tests/physics/test_atomic.py`

- [ ] **Step 1: Inspect the VizieR table format first**

```bash
curl -fsSL https://cdsarc.cds.unistra.fr/ftp/J/A+AS/109/125/ReadMe
```

Expected columns (verify against ReadMe, adjust parser): Z, N (electrons in ion), n, l
(shell quantum numbers), E_th [eV], E_0 [eV], σ_0 [Mb], y_a, P, y_w. If VizieR is down,
fall back to parsing the `DATA` blocks of `phfit2.f` from the UKy URL in the sources table.

- [ ] **Step 2: Write `tools/build_vy95_table.py`**

```python
"""Build whimstan/physics/_data/vy95_phfit_v1.npz from Verner & Yakovlev 1995.

Source: VizieR J/A+AS/109/125 (per-shell analytic photoionization fits).
Run: uv run python tools/build_vy95_table.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np

ELEMENTS = (("H", 1), ("He", 2), ("C", 6), ("N", 7), ("O", 8), ("Ne", 10),
            ("Mg", 12), ("Si", 14), ("S", 16), ("Ar", 18), ("Ca", 20), ("Fe", 26))
MAX_SHELLS = 7  # neutral Fe: 1s 2s 2p 3s 3p 3d 4s

URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+AS/109/125/table1.dat"  # verify name in ReadMe


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=60).read().decode()
    # Parse fixed-width per ReadMe; build dict[(Z, n_elec)] -> list of shell rows
    # (l_q, E_th, E_0, sigma_0, y_a, P, y_w). The byte ranges below are
    # ADJUSTED FROM THE README in Step 1 before this script is run.
    rows: dict[tuple[int, int], list[tuple[float, ...]]] = {}
    for line in io.StringIO(raw):
        Z, ne = int(line[0:3]), int(line[3:6])
        l_q = int(line[9:12])  # n quantum number (cols 6:9) is not needed
        eth, e0, s0, ya, p, yw = (
            float(line[i:j])
            for i, j in ((12, 22), (22, 32), (32, 42), (42, 52), (52, 62), (62, 72))
        )
        rows.setdefault((Z, ne), []).append((l_q, eth, e0, s0, ya, p, yw))

    n_ions = sum(z for _, z in ELEMENTS)
    shape = (n_ions, MAX_SHELLS)
    out = {k: np.zeros(shape) for k in ("E_th_eV", "E0_eV", "sigma0_Mb", "y_a", "P", "y_w", "l_q")}
    mask = np.zeros(shape, dtype=bool)
    idx = 0
    for _, Z in ELEMENTS:
        for stage in range(Z):           # electrons removed
            shells = rows[(Z, Z - stage)]  # ion has Z-stage electrons
            assert 1 <= len(shells) <= MAX_SHELLS, (Z, stage, len(shells))
            for j, (l_q, eth, e0, s0, ya, p, yw) in enumerate(shells):
                out["l_q"][idx, j] = l_q
                out["E_th_eV"][idx, j] = eth
                out["E0_eV"][idx, j] = e0
                out["sigma0_Mb"][idx, j] = s0
                out["y_a"][idx, j] = ya
                out["P"][idx, j] = p
                out["y_w"][idx, j] = yw
                mask[idx, j] = True
            idx += 1

    dest = Path("whimstan/physics/_data/vy95_phfit_v1.npz")
    np.savez_compressed(dest, shell_mask=mask, **out)
    print(f"wrote {dest}: {idx} ions")


if __name__ == "__main__":
    main()
```

The fixed-width slices above are placeholders for the real ReadMe byte ranges — set them
from Step 1's output before running. Known checksum to assert in the script after parsing:
H I (Z=1, ne=1) 1s shell has E_th = 13.6 eV, σ_0 ≈ 5.475e4 Mb, y_a ≈ 32.88, P ≈ 2.963
(famous Verner H I values; tolerance 1%). If they don't match, the column mapping is wrong.

- [ ] **Step 3: Run it**

Run: `uv run python tools/build_vy95_table.py`
Expected: `wrote whimstan/physics/_data/vy95_phfit_v1.npz: 140 ions`

- [ ] **Step 4: Write the failing test**

```python
"""Tests for VY95 photoionization cross sections."""

from __future__ import annotations

import jax.numpy as jnp


def test_registry_shapes() -> None:
    from whimstan.physics.atomic import ELEMENTS, N_IONS, ion_index

    assert N_IONS == 140
    assert ion_index("H", 0) == 0
    assert ion_index("He", 1) == 2
    assert ion_index("Fe", 25) == 139


def test_hydrogen_threshold_value() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    E = jnp.array([0.0137])  # keV, just above 13.6 eV
    sig = sigma_ions(E)
    # sigma(H I, threshold) ~ 6.3e-18 cm^2
    assert 5.0e-18 < float(sig[ion_index("H", 0), 0]) < 8.0e-18


def test_hydrogen_highE_slope() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    E = jnp.array([0.05, 0.1])  # keV
    sig = sigma_ions(E)[ion_index("H", 0)]
    ratio = float(sig[1] / sig[0])
    assert abs(ratio - 2.0**-3) / 2.0**-3 < 0.25  # ~E^-3 far above threshold


def test_threshold_zero_below_edge() -> None:
    from whimstan.physics.atomic import ion_index, sigma_ions

    # O VII (He-like oxygen, stage 6): K edge at 739 eV.
    E = jnp.array([0.5, 0.80])
    sig = sigma_ions(E)[ion_index("O", 6)]
    assert float(sig[0]) == 0.0
    assert float(sig[1]) > 0.0


def test_all_finite_nonnegative() -> None:
    from whimstan.physics.atomic import sigma_ions

    E = jnp.logspace(jnp.log10(0.3), jnp.log10(60.0), 200)
    sig = sigma_ions(E)
    assert sig.shape == (140, 200)
    assert bool(jnp.all(jnp.isfinite(sig)))
    assert bool(jnp.all(sig >= 0.0))
```

Run: `uv run pytest tests/physics/test_atomic.py -v` — Expected: FAIL (no module).

- [ ] **Step 5: Write `whimstan/physics/atomic.py`**

```python
"""Per-ion photoionization cross sections (Verner & Yakovlev 1995).

sigma_nl(E) = sigma_0 * F(y),  y = E / E_0,
F(y) = [(y - 1)^2 + y_w^2] * y^{-Q} * (1 + sqrt(y / y_a))^{-P},
Q = 5.5 + l - 0.5 P,  valid above the shell threshold E_th.

Total per-ion sigma = sum over occupied shells. Fit parameters are
committed in _data/vy95_phfit_v1.npz (built by tools/build_vy95_table.py
from VizieR J/A+AS/109/125).

This module also owns the element/ion registry shared by the ionization
solver, the UVB Gamma table, and the grid assembler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import Array

ELEMENTS: tuple[tuple[str, int], ...] = (
    ("H", 1), ("He", 2), ("C", 6), ("N", 7), ("O", 8), ("Ne", 10),
    ("Mg", 12), ("Si", 14), ("S", 16), ("Ar", 18), ("Ca", 20), ("Fe", 26),
)

#: number of absorbing ions (>=1 electron): stages 0..Z-1 per element.
N_IONS = sum(z for _, z in ELEMENTS)  # 140
#: number of ionization stages incl. bare nucleus: 0..Z per element.
N_STAGES = sum(z + 1 for _, z in ELEMENTS)  # 152

#: start index of each element's block in the 140-layout / 152-layout.
ION_OFFSETS: dict[str, int] = {}
STAGE_OFFSETS: dict[str, int] = {}
_o = _s = 0
for _name, _z in ELEMENTS:
    ION_OFFSETS[_name] = _o
    STAGE_OFFSETS[_name] = _s
    _o += _z
    _s += _z + 1
del _o, _s, _name, _z


def ion_index(element: str, stage: int) -> int:
    """Flat 140-layout index for `element` with `stage` electrons removed."""
    z = dict(ELEMENTS)[element]
    if not 0 <= stage < z:
        raise ValueError(f"{element} stage {stage} has no bound electrons")
    return ION_OFFSETS[element] + stage


_DATA = Path(__file__).parent / "_data" / "vy95_phfit_v1.npz"
with np.load(_DATA) as _f:
    _ETH = jnp.asarray(_f["E_th_eV"])      # (140, 7)
    _E0 = jnp.asarray(_f["E0_eV"])
    _S0 = jnp.asarray(_f["sigma0_Mb"])
    _YA = jnp.asarray(_f["y_a"])
    _P = jnp.asarray(_f["P"])
    _YW = jnp.asarray(_f["y_w"])
    _LQ = jnp.asarray(_f["l_q"])
    _MASK = jnp.asarray(_f["shell_mask"])

# Guard against log(0)/0-division on padded shells.
_E0_SAFE = jnp.where(_MASK, _E0, 1.0)
_YA_SAFE = jnp.where(_MASK, _YA, 1.0)


def sigma_ions(E_keV: Array) -> Array:
    """Cross section per absorbing ion, cm^2.

    Parameters
    ----------
    E_keV : (N_E,) photon energies, keV.

    Returns
    -------
    (N_IONS, N_E) array; zero below each shell threshold.
    """
    E_eV = jnp.asarray(E_keV, dtype=jnp.float64) * 1.0e3
    y = E_eV[None, None, :] / _E0_SAFE[:, :, None]          # (140, 7, NE)
    q = 5.5 + _LQ - 0.5 * _P                                  # (140, 7)
    f = (
        ((y - 1.0) ** 2 + _YW[:, :, None] ** 2)
        * y ** (-q[:, :, None])
        * (1.0 + jnp.sqrt(y / _YA_SAFE[:, :, None])) ** (-_P[:, :, None])
    )
    sig = _S0[:, :, None] * 1.0e-18 * f                      # Mb -> cm^2
    keep = (E_eV[None, None, :] >= _ETH[:, :, None]) & _MASK[:, :, None]
    return jnp.where(keep, sig, 0.0).sum(axis=1)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/physics/test_atomic.py -v` — Expected: 5 PASS.

- [ ] **Step 7: Lint, typecheck, commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy whimstan
git add tools/build_vy95_table.py whimstan/physics/_data/vy95_phfit_v1.npz \
        whimstan/physics/atomic.py tests/physics/test_atomic.py
git commit -m "feat(physics): VY95 per-shell photoionization cross sections + ion registry"
```

### Task 4: Asplund+2009 abundances

**Files:**
- Create: `whimstan/physics/abundances.py`
- Test: `tests/physics/test_abundances.py`

- [ ] **Step 1: Failing test**

```python
"""Asplund+2009 abundance sanity checks."""


def test_abundance_values() -> None:
    from whimstan.physics.abundances import LOG_ABUND_ASPL09, n_relative_to_H

    assert set(LOG_ABUND_ASPL09) == {
        "H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Fe"
    }
    assert n_relative_to_H("H") == 1.0
    assert abs(n_relative_to_H("He") - 0.0851) / 0.0851 < 0.01
    assert abs(n_relative_to_H("O") - 4.90e-4) / 4.90e-4 < 0.01
    assert abs(n_relative_to_H("Fe") - 3.16e-5) / 3.16e-5 < 0.01
```

Run: `uv run pytest tests/physics/test_abundances.py -v` — Expected: FAIL.

- [ ] **Step 2: Implement**

```python
"""Solar abundances, Asplund et al. 2009 (ARA&A 47, 481), photospheric.

log eps_X = log10(n_X / n_H) + 12. Grid metallicity is pinned to 1.0 x
this set (amplitude and pattern); see grid metadata + README for the
n0 <-> Z degeneracy note.
"""

from __future__ import annotations

LOG_ABUND_ASPL09: dict[str, float] = {
    "H": 12.00, "He": 10.93, "C": 8.43, "N": 7.83, "O": 8.69, "Ne": 7.93,
    "Mg": 7.60, "Si": 7.51, "S": 7.12, "Ar": 6.40, "Ca": 6.34, "Fe": 7.50,
}


def n_relative_to_H(element: str) -> float:
    """Number density of `element` relative to hydrogen."""
    return 10.0 ** (LOG_ABUND_ASPL09[element] - 12.0)
```

- [ ] **Step 3: Run, lint, commit**

```bash
uv run pytest tests/physics/test_abundances.py -v
uv run ruff check . && uv run mypy whimstan
git add whimstan/physics/abundances.py tests/physics/test_abundances.py
git commit -m "feat(physics): Asplund+2009 abundances for the 12 WHIM elements"
```

### Task 5: CHIANTI-derived recombination + CIE-ratio tables

**Files:**
- Create: `tools/build_ion_rates.py`
- Create: `whimstan/physics/_data/ion_rates_v1.npz`
- Test: `tests/physics/test_ion_rates.py` (tests the npz directly; the JAX consumer comes in Task 7)

- [ ] **Step 1: Inspect the CHIANTI GitHub layout first**

```bash
gh api repos/chianti-atomic/CHIANTI/contents/ --jq '.[].name' | head -40
gh api repos/chianti-atomic/CHIANTI/contents/o/o_6 --jq '.[].name'
curl -fsSL https://raw.githubusercontent.com/chianti-atomic/CHIANTI/master/o/o_6/o_6.rrparams
curl -fsSL https://raw.githubusercontent.com/chianti-atomic/CHIANTI/master/ioneq/chianti.ioneq | head -8
```

Record in the script's docstring: directory layout, `.rrparams` line format (Badnell fit:
Z, ion, M, A, B, T0, T1 and optionally C, T2), `.drparams` format (one line of E_i, one of
c_i, up to 9 terms), `chianti.ioneq` format (T grid line, then rows of Z, ion, f(T)…).
**Convention check (critical):** CHIANTI names rate files by the *recombined* ion in
spectroscopic notation, but verify by computing α_RR for H at T = 10⁴ K from whichever
file (`h_1.rrparams` or `h_2.rrparams`) and requiring ≈ 4.18e-13 cm³ s⁻¹ (case-A
hydrogen RR — unambiguous). Wire that assert into the script.

- [ ] **Step 2: Write `tools/build_ion_rates.py`**

```python
"""Build whimstan/physics/_data/ion_rates_v1.npz from the CHIANTI database.

For each ladder rung s of element X (X^s+ <-> X^(s+1)+), stores on a
common log T grid (4.0..8.0, step 0.02, 201 points), 140-layout:

- alpha[(rung, T)]      : total recombination RR+DR for X^(s+1)+ + e -> X^s+
                          [cm^3 s^-1], from CHIANTI .rrparams/.drparams
                          (Badnell fit formulas).
- log10_R_cie[(rung,T)] : log10(f_{s+1}/f_s) in CIE from chianti.ioneq,
                          clipped to [-30, 30] with the tail rules below.

Tail rules for the CIE ratio (ioneq stores hard zeros in tails):
  f_{s+1} <= floor            -> R = 1e-30   (rung effectively closed)
  f_s <= floor < f_{s+1}      -> R = 1e+30   (rung saturated)
  floor = 1e-15.
These extremes never matter for tau (dominant-ion argument) but keep the
ladder product well-defined; the PIE term reopens closed rungs physically.

Badnell RR:  alpha(T) = A * [ sqrt(T/T0) * (1+sqrt(T/T0))^(1-B')
                              * (1+sqrt(T/T1))^(1+B') ]^-1,
             B' = B + C*exp(-T2/T) when C,T2 present, else B.
Badnell DR:  alpha(T) = T^(-3/2) * sum_i c_i * exp(-E_i/T).

Run: uv run python tools/build_ion_rates.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

ELEMENTS = (("h", 1), ("he", 2), ("c", 6), ("n", 7), ("o", 8), ("ne", 10),
            ("mg", 12), ("si", 14), ("s", 16), ("ar", 18), ("ca", 20), ("fe", 26))
RAW = "https://raw.githubusercontent.com/chianti-atomic/CHIANTI/master"
LOGT = np.arange(4.0, 8.0001, 0.02)
T = 10.0 ** LOGT
FLOOR = 1e-15


def fetch(path: str) -> str | None:
    try:
        return urllib.request.urlopen(f"{RAW}/{path}", timeout=60).read().decode()
    except Exception:
        return None


def alpha_rr(params: list[float]) -> np.ndarray:
    A, B, T0, T1 = params[0], params[1], params[2], params[3]
    Bp = B + (params[4] * np.exp(-params[5] / T) if len(params) >= 6 else 0.0)
    s0, s1 = np.sqrt(T / T0), np.sqrt(T / T1)
    return A / (s0 * (1.0 + s0) ** (1.0 - Bp) * (1.0 + s1) ** (1.0 + Bp))


def alpha_dr(e: np.ndarray, c: np.ndarray) -> np.ndarray:
    return T ** -1.5 * (c[:, None] * np.exp(-e[:, None] / T)).sum(axis=0)


def main() -> None:
    n_ions = sum(z for _, z in ELEMENTS)
    alpha = np.zeros((n_ions, LOGT.size))
    log10_r_cie = np.full((n_ions, LOGT.size), -30.0)

    ioneq = fetch("ioneq/chianti.ioneq")
    assert ioneq is not None
    # Parse per Step-1 findings into: t_ioneq (logT grid), frac[(Z, stage)] -> f(T).
    t_ioneq, frac = parse_ioneq(ioneq)  # implement per inspected format

    idx = 0
    for el, z in ELEMENTS:
        for stage in range(z):  # rung s: X^s+ <-> X^(s+1)+
            # Recombined ion X^s+ in spectroscopic notation is {el}_{stage+1};
            # VERIFY convention via the hydrogen check below.
            ion_name = f"{el}_{stage + 1}"
            rr = fetch(f"{el}/{ion_name}/{ion_name}.rrparams")
            dr = fetch(f"{el}/{ion_name}/{ion_name}.drparams")
            a = parse_and_eval_rr(rr) if rr else np.zeros_like(T)
            if dr:
                a = a + parse_and_eval_dr(dr)
            alpha[idx] = a

            f_lo = np.interp(LOGT, t_ioneq, frac[(z, stage)])
            f_hi = np.interp(LOGT, t_ioneq, frac[(z, stage + 1)])
            r = np.where(f_hi <= FLOOR, 1e-30,
                         np.where(f_lo <= FLOOR, 1e30, f_hi / np.maximum(f_lo, FLOOR)))
            log10_r_cie[idx] = np.clip(np.log10(r), -30.0, 30.0)
            idx += 1

    # Hydrogen convention check: alpha(H rung 0) at 1e4 K ~ 4.18e-13 (case A).
    a_h = np.interp(4.0, LOGT, alpha[0])
    assert 3.5e-13 < a_h < 5.0e-13, f"RR convention wrong: alpha_H(1e4K)={a_h:.3e}"

    dest = Path("whimstan/physics/_data/ion_rates_v1.npz")
    np.savez_compressed(dest, logT_grid=LOGT, alpha=alpha, log10_R_cie=log10_r_cie)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
```

`parse_ioneq`, `parse_and_eval_rr`, `parse_and_eval_dr` are written against the *actual*
formats inspected in Step 1 (they are simple fixed-format text). Missing `.drparams`
(e.g. recombination to neutral H has no DR channel) → DR = 0, expected and fine.

- [ ] **Step 3: Run it**

Run: `uv run python tools/build_ion_rates.py`
Expected: `wrote whimstan/physics/_data/ion_rates_v1.npz` (no assert trips). 140 × 2
small downloads + ioneq; a few minutes.

- [ ] **Step 4: Write tests against the committed npz**

```python
"""Sanity checks on the CHIANTI-derived rate/CIE tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NPZ = Path("whimstan/physics/_data/ion_rates_v1.npz")


def test_shapes() -> None:
    with np.load(_NPZ) as f:
        assert f["alpha"].shape == (140, f["logT_grid"].size)
        assert f["log10_R_cie"].shape == f["alpha"].shape
        assert np.all(np.isfinite(f["alpha"])) and np.all(f["alpha"] >= 0)


def test_hydrogen_caseA_rr() -> None:
    with np.load(_NPZ) as f:
        a = np.interp(4.0, f["logT_grid"], f["alpha"][0])
    assert 3.5e-13 < a < 5.0e-13


def test_cie_oxygen_o7_dominates_at_1e6K() -> None:
    """Reconstruct CIE fractions for O at logT=6 from the ratio ladder."""
    from whimstan.physics.atomic import ION_OFFSETS

    with np.load(_NPZ) as f:
        it = np.argmin(np.abs(f["logT_grid"] - 6.0))
        r = 10.0 ** f["log10_R_cie"][ION_OFFSETS["O"]:ION_OFFSETS["O"] + 8, it]
    logf = np.concatenate([[0.0], np.cumsum(np.log(r))])
    frac = np.exp(logf - logf.max())
    frac /= frac.sum()
    assert frac.argmax() == 6          # O VII (He-like) dominant
    assert frac[6] > 0.5


def test_cie_hydrogen_ionized_at_1e5K() -> None:
    with np.load(_NPZ) as f:
        it = np.argmin(np.abs(f["logT_grid"] - 5.0))
        r = 10.0 ** f["log10_R_cie"][0, it]
    assert r > 1e3  # f(HII)/f(HI) >> 1
```

Run: `uv run pytest tests/physics/test_ion_rates.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
uv run ruff check . && uv run mypy whimstan
git add tools/build_ion_rates.py whimstan/physics/_data/ion_rates_v1.npz \
        tests/physics/test_ion_rates.py
git commit -m "feat(physics): CHIANTI-derived recombination + CIE-ratio tables"
```

### Task 6: KS19 UVB table + photoionization rates Γ_s(z)

Depends on Tasks 1–3 (CLOUDY built, UVB command pinned, σ available).

**Files:**
- Create: `tools/build_uvb_table.py`
- Create: `whimstan/physics/_data/ks19_uvb_v1.npz`
- Test: `tests/physics/test_uvb.py`

- [ ] **Step 1: Write `tools/build_uvb_table.py`**

```python
"""Extract the KS19 UVB from CLOUDY and precompute photoionization rates.

For each z in 0..5 step 0.1, runs a minimal CLOUDY model whose only
radiation source is the UVB (no CMB), saves the incident continuum, and
computes for every absorbing ion s:

    Gamma_s(z) = int 4*pi*J_nu / (h*nu) * sigma_s(h*nu) dnu
               = int F(E) * sigma_s(E) / E_erg dlnE,

where F(E) = 4*pi*nu*J_nu [erg cm^-2 s^-1] is what CLOUDY's
`save incident continuum` provides (column format verified in Task 2,
see tools/build_whim_grid/NOTES.md).

Output npz: z (Nz,), gamma (140, Nz) [s^-1], plus provenance arrays
E_keV (NE,), log10_fourpiNuJnu (Nz, NE) on a common resampled grid.

Run: uv run python tools/build_uvb_table.py --cloudy-exe <path>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "build_whim_grid"))
from uvb import uvb_command  # noqa: E402

KEV_TO_ERG = 1.602176634e-9
Z_GRID = np.round(np.arange(0.0, 5.0001, 0.1), 2)
E_RESAMPLE_KEV = np.logspace(np.log10(13.6e-3 / 10.0), np.log10(60.0), 2048)

INPUT = """\
title uvb extraction z={z}
{uvb}
hden -5
constant temperature, t=6.0 log
stop zone 1
set dr 0
save incident continuum units keV "{stem}.con" last
"""


def run_one(cloudy_exe: str, data_path: str, z: float, workdir: Path) -> np.ndarray:
    stem = f"uvb_z{z:.2f}"
    (workdir / f"{stem}.in").write_text(
        INPUT.format(z=z, uvb=uvb_command(z), stem=stem)
    )
    subprocess.run(
        [cloudy_exe, "-r", stem], cwd=workdir, check=True,
        env={"CLOUDY_DATA_PATH": data_path, "PATH": "/usr/bin:/bin"},
    )
    e_kev, fnu = np.loadtxt(workdir / f"{stem}.con", usecols=(0, 1), unpack=True)
    order = np.argsort(e_kev)
    return np.interp(E_RESAMPLE_KEV, e_kev[order], fnu[order], left=0.0, right=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloudy-exe", required=True)
    args = ap.parse_args()
    data_path = str(Path(args.cloudy_exe).parent.parent / "data")

    f_grid = np.zeros((Z_GRID.size, E_RESAMPLE_KEV.size))
    with tempfile.TemporaryDirectory() as td:
        for i, z in enumerate(Z_GRID):
            f_grid[i] = run_one(args.cloudy_exe, data_path, float(z), Path(td))
            print(f"z={z:.2f} done")

    from whimstan.physics.atomic import sigma_ions  # jax; x64 on via whimstan import

    sig = np.asarray(sigma_ions(E_RESAMPLE_KEV))            # (140, NE)
    lnE = np.log(E_RESAMPLE_KEV)
    e_erg = E_RESAMPLE_KEV * KEV_TO_ERG
    gamma = np.array([
        # np.trapz: lockfile pins numpy 1.26 (no np.trapezoid until numpy 2)
        np.trapz(f_grid[i][None, :] * sig / e_erg[None, :], lnE, axis=1)
        for i in range(Z_GRID.size)
    ]).T                                                     # (140, Nz)

    dest = Path("whimstan/physics/_data/ks19_uvb_v1.npz")
    np.savez_compressed(
        dest, z=Z_GRID, gamma=gamma, E_keV=E_RESAMPLE_KEV,
        log10_fourpiNuJnu=np.log10(np.maximum(f_grid, 1e-300)),
    )
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
```

Note the `cloudy -r <stem>` invocation style (reads `<stem>.in`); if the built binary
wants stdin instead, use `subprocess.run([exe], stdin=...)` — the Task-1 smoke test
already established which works; follow NOTES.md.

- [ ] **Step 2: Run it**

Run: `uv run python tools/build_uvb_table.py --cloudy-exe tools/build_whim_grid/c23.01/source/cloudy.exe`
Expected: 51 z lines, then `wrote whimstan/physics/_data/ks19_uvb_v1.npz`. ~5–15 min.

- [ ] **Step 3: Tests**

```python
"""Sanity checks on the KS19-derived photoionization-rate table."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_NPZ = Path("whimstan/physics/_data/ks19_uvb_v1.npz")


def test_shapes_finite() -> None:
    with np.load(_NPZ) as f:
        assert f["gamma"].shape == (140, f["z"].size)
        assert np.all(np.isfinite(f["gamma"])) and np.all(f["gamma"] >= 0)


def test_hi_photoionization_rate_magnitude() -> None:
    """Gamma_HI(z=0) is measured to be a few x 1e-14 s^-1; KS19 ~4.6e-14."""
    with np.load(_NPZ) as f:
        g0 = np.interp(0.0, f["z"], f["gamma"][0])
        g2 = np.interp(2.0, f["z"], f["gamma"][0])
    assert 1e-14 < g0 < 2e-13
    assert g2 / g0 > 5.0  # UVB much harder/brighter at z~2
```

Run: `uv run pytest tests/physics/test_uvb.py -v` — Expected: 2 PASS.

- [ ] **Step 4: Lint, typecheck, commit**

```bash
uv run ruff check . && uv run mypy whimstan
git add tools/build_uvb_table.py whimstan/physics/_data/ks19_uvb_v1.npz tests/physics/test_uvb.py
git commit -m "feat(physics): KS19 UVB extraction + per-ion photoionization rate table"
```

---

# Phase 2 — JAX ionization solver + real validator (Tasks 7–9)

### Task 7: `physics/ionization.py` — hybrid CIE+PIE balance

**Files:**
- Create: `whimstan/physics/ionization.py`
- Test: `tests/physics/test_ionization.py`

- [ ] **Step 1: Failing tests**

```python
"""Tests for the hybrid CIE+PIE ionization balance."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def test_fractions_normalized_per_element() -> None:
    from whimstan.physics.atomic import ELEMENTS, STAGE_OFFSETS
    from whimstan.physics.ionization import ion_fractions

    f, n_e = ion_fractions(jnp.asarray(6.0), jnp.asarray(1e-5), jnp.asarray(0.5))
    assert f.shape == (152,)
    assert float(n_e) > 0.0
    for name, z in ELEMENTS:
        s = STAGE_OFFSETS[name]
        assert abs(float(f[s : s + z + 1].sum()) - 1.0) < 1e-8


def test_cie_limit_matches_chianti_at_high_density() -> None:
    """At n_H = 1 cm^-3 the UVB term is negligible: pure CIE."""
    from whimstan.physics.atomic import STAGE_OFFSETS
    from whimstan.physics.ionization import ion_fractions, ion_fractions_cie

    f, _ = ion_fractions(jnp.asarray(6.0), jnp.asarray(1.0), jnp.asarray(0.0))
    f_cie = ion_fractions_cie(jnp.asarray(6.0))
    s = STAGE_OFFSETS["O"]
    assert float(jnp.abs(f[s : s + 9] - f_cie[s : s + 9]).max()) < 0.05
    assert int(f[s : s + 9].argmax()) == 6  # O VII dominant at 1e6 K


def test_pie_dominates_at_low_density() -> None:
    """Photoionization keeps H ionized at low T, low density (classic PIE)."""
    from whimstan.physics.ionization import ion_fractions, ion_fractions_cie

    f, _ = ion_fractions(jnp.asarray(4.2), jnp.asarray(1e-7), jnp.asarray(0.0))
    f_cie = ion_fractions_cie(jnp.asarray(4.2))
    assert float(f[0]) < 0.01        # hybrid: H I rare
    assert float(f_cie[0]) > 0.05    # CIE alone: H I common at 16,000 K


def test_jit_and_grad_clean() -> None:
    from whimstan.physics.ionization import ion_fractions

    def first_frac(log_t: jax.Array) -> jax.Array:
        f, _ = ion_fractions(log_t, jnp.asarray(1e-5), jnp.asarray(1.0))
        return f[0]

    g = jax.grad(first_frac)(jnp.asarray(6.0))
    assert bool(jnp.isfinite(g))
    f_jit = jax.jit(
        lambda lt: ion_fractions(lt, jnp.asarray(1e-5), jnp.asarray(1.0))[0]
    )(jnp.asarray(6.0))
    assert f_jit.shape == (152,)
```

Run: `uv run pytest tests/physics/test_ionization.py -v` — Expected: FAIL (no module).

- [ ] **Step 2: Implement `whimstan/physics/ionization.py`**

```python
"""Hybrid CIE+PIE ionization balance, pure JAX.

Ladder ratio per rung s of element X (X^s+ <-> X^(s+1)+):

    f_{s+1} / f_s = R_cie_s(T) + Gamma_s(z) / (n_e * alpha_s(T))

R_cie : CIE ratio from CHIANTI chianti.ioneq (the CIE limit is exactly
        CHIANTI's equilibrium).
alpha : RR+DR recombination, CHIANTI Badnell fits (ion_rates_v1.npz).
Gamma : KS19 photoionization rates (ks19_uvb_v1.npz), z-interpolated.

n_e is solved self-consistently on H and He (metals contribute <0.1%
to n_e) with a damped fixed-point iteration, then all 12 elements are
evaluated at the converged n_e.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from whimstan.physics.abundances import n_relative_to_H
from whimstan.physics.atomic import ELEMENTS, ION_OFFSETS

_D = Path(__file__).parent / "_data"
with np.load(_D / "ion_rates_v1.npz") as _f:
    _LOGT = jnp.asarray(_f["logT_grid"])
    _ALPHA = jnp.asarray(_f["alpha"])            # (140, NT)
    _LOG10_RCIE = jnp.asarray(_f["log10_R_cie"])  # (140, NT)
with np.load(_D / "ks19_uvb_v1.npz") as _f:
    _ZGRID = jnp.asarray(_f["z"])
    _GAMMA = jnp.asarray(_f["gamma"])            # (140, NZ)

_A_HE = n_relative_to_H("He")


def _interp_rows(x: Array, xgrid: Array, table: Array) -> Array:
    """Interpolate each row of `table` (N, K) at scalar x -> (N,)."""
    return jax.vmap(lambda row: jnp.interp(x, xgrid, row))(table)


def _ladder_ratios(log_T: Array, z: Array, n_e: Array) -> Array:
    """f_{s+1}/f_s for all 140 rungs at (T, z, n_e)."""
    r_cie = 10.0 ** _interp_rows(log_T, _LOGT, _LOG10_RCIE)
    alpha = jnp.maximum(_interp_rows(log_T, _LOGT, _ALPHA), 1e-30)
    gamma = _interp_rows(z, _ZGRID, _GAMMA)
    return r_cie + gamma / (n_e * alpha)


def _fractions_from_ratios(r: Array) -> Array:
    """(Z,) rung ratios -> (Z+1,) normalized stage fractions, overflow-safe."""
    log_r = jnp.log(jnp.clip(r, 1e-30, 1e30))
    log_f = jnp.concatenate([jnp.zeros(1), jnp.cumsum(log_r)])
    f = jnp.exp(log_f - log_f.max())
    return f / f.sum()


def _all_fractions(ratios: Array) -> Array:
    """(140,) rung ratios -> (152,) stage fractions for all elements."""
    parts = []
    for name, z in ELEMENTS:
        o = ION_OFFSETS[name]
        parts.append(_fractions_from_ratios(ratios[o : o + z]))
    return jnp.concatenate(parts)


def ion_fractions(
    log_T: Array, n_H: Array, z: Array, n_iter: int = 12
) -> tuple[Array, Array]:
    """Hybrid stage fractions at (log10 T [K], n_H [cm^-3], z).

    Returns
    -------
    fractions : (152,) per-element-normalized stage fractions (152-layout).
    n_e       : electron density [cm^-3].
    """
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    n_H = jnp.asarray(n_H, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)

    def body(_: int, x: Array) -> Array:
        n_e = jnp.maximum(x, 1e-8) * n_H
        r = _ladder_ratios(log_T, z, n_e)
        f_h = _fractions_from_ratios(r[0:1])
        f_he = _fractions_from_ratios(r[1:3])
        x_new = f_h[1] + _A_HE * (f_he[1] + 2.0 * f_he[2])
        return 0.5 * (x + x_new)  # damped to avoid low-T oscillation

    x = jax.lax.fori_loop(0, n_iter, body, jnp.asarray(1.2, dtype=jnp.float64))
    n_e = jnp.maximum(x, 1e-8) * n_H
    return _all_fractions(_ladder_ratios(log_T, z, n_e)), n_e


def ion_fractions_cie(log_T: Array) -> Array:
    """CIE-only stage fractions (no UVB term); for tests and diagnostics."""
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    r = 10.0 ** _interp_rows(log_T, _LOGT, _LOG10_RCIE)
    return _all_fractions(r)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/physics/test_ionization.py -v` — Expected: 4 PASS.
If `test_pie_dominates_at_low_density` fails, debug by printing the three ratio terms
for the H rung — most likely an α-convention or Γ-units bug, not a solver bug.

- [ ] **Step 4: Lint, typecheck, commit**

```bash
uv run ruff check . && uv run mypy whimstan
git add whimstan/physics/ionization.py tests/physics/test_ionization.py
git commit -m "feat(physics): hybrid CIE+PIE ionization balance in pure JAX"
```

### Task 8: Real validator — replace the Plan-1 stub

**Files:**
- Modify: `whimstan/physics/whim_validator.py` (full rewrite, signature preserved)
- Modify: `tests/physics/test_whim_validator.py` (drop stub-era grid test, add physics tests)

- [ ] **Step 1: Rewrite the tests first**

Replace the whole file `tests/physics/test_whim_validator.py`:

```python
"""Tests for the from-scratch WHIM tau validator (real physics, Plan 2).

The decisive <=5% comparison against the CLOUDY production grid lives in
tests/physics/test_whim_vs_grid.py (marker: needs_grid). These tests
cover internal consistency that needs no grid asset.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _tau(log_t: float, log_n: float, z: float, n_e: int = 32) -> jax.Array:
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    e_obs = jnp.logspace(jnp.log10(0.3), 1.0, n_e)
    return compute_tau_from_scratch(log_T=log_t, log_n=log_n, z=z, E_obs=e_obs)


def test_shape_finite_nonnegative() -> None:
    tau = _tau(6.0, -5.0, 2.0)
    assert tau.shape == (32,)
    assert bool(jnp.all(jnp.isfinite(tau)))
    assert bool(jnp.all(tau >= 0.0))


def test_zero_at_z_zero() -> None:
    assert float(jnp.abs(_tau(6.0, -4.0, 0.0)).max()) == 0.0


def test_monotone_in_z_and_n() -> None:
    t1, t2 = _tau(6.0, -5.0, 1.0), _tau(6.0, -5.0, 3.0)
    assert bool(jnp.all(t2 >= t1))
    t3 = _tau(6.0, -4.0, 1.0)
    assert float(t3.max()) > float(t1.max())


def test_soft_band_more_absorbed_than_hard() -> None:
    tau = _tau(6.0, -4.0, 2.0, n_e=64)
    e = jnp.logspace(jnp.log10(0.3), 1.0, 64)
    assert float(tau[jnp.argmin(jnp.abs(e - 0.5))]) > float(
        tau[jnp.argmin(jnp.abs(e - 8.0))]
    )


def test_magnitude_plausible() -> None:
    """Solar-Z WHIM at n0=1e-5, z=2: peak tau should be ~1e-3..1, not 1e-12 or 1e3."""
    tau = _tau(6.0, -5.0, 2.0)
    assert 1e-4 < float(tau.max()) < 10.0


def test_jit_grad_clean() -> None:
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    e_obs = jnp.logspace(jnp.log10(0.3), 1.0, 16)

    def loss(params: jax.Array) -> jax.Array:
        return compute_tau_from_scratch(
            log_T=params[0], log_n=params[1], z=params[2], E_obs=e_obs
        ).sum()

    g = jax.grad(loss)(jnp.array([6.0, -5.0, 2.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
```

Run: `uv run pytest tests/physics/test_whim_validator.py -v` — Expected: FAIL
(stub returns wrong physics; e.g. magnitude test).

- [ ] **Step 2: Rewrite `whimstan/physics/whim_validator.py`**

```python
"""From-scratch WHIM optical-depth calculator (cross-validator).

Computes, in pure JAX with no CLOUDY involvement in the ionization
balance,

    tau(E_obs) = int_0^z n_H(z') sigma_eff(E_obs (1+z'); T, n_H(z'), z')
                 * c / ((1+z') H(z'))  dz',

with n_H(z') = n0 (1+z')^3 and

    sigma_eff(E) = sum_X A_X sum_s f_{X,s}(T, n_H, z') sigma_{X,s}(E),

f from physics.ionization (hybrid CIE+PIE; CHIANTI rates + KS19 UVB),
sigma from physics.atomic (VY95), A from physics.abundances (Asplund09,
solar metallicity — matches the grid build; see grid metadata).

Used by the needs_grid gate test to certify the CLOUDY grid to <=5%.
The public signature is frozen since Plan 1.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from whimstan.physics.abundances import n_relative_to_H
from whimstan.physics.atomic import ELEMENTS, N_IONS, STAGE_OFFSETS, sigma_ions
from whimstan.physics.cosmology import Planck18, dimensionless_hubble
from whimstan.physics.ionization import ion_fractions

_MPC_CM = 3.0856775814913673e24
_D_H_CM = Planck18.hubble_distance_Mpc * _MPC_CM

# Map 152-layout fractions onto the 140-layout sigma rows, with abundances.
_IDX152 = np.concatenate(
    [STAGE_OFFSETS[name] + np.arange(z) for name, z in ELEMENTS]
)
_ABUND140 = np.concatenate(
    [np.full(z, n_relative_to_H(name)) for name, z in ELEMENTS]
)
assert _IDX152.size == _ABUND140.size == N_IONS
_IDX152_J = jnp.asarray(_IDX152)
_ABUND140_J = jnp.asarray(_ABUND140)


def compute_tau_from_scratch(
    *,
    log_T: Array | float,
    log_n: Array | float,
    z: Array | float,
    E_obs: Array,
    n_shells: int = 257,
) -> Array:
    """Observer-frame WHIM tau(E_obs) computed from first principles.

    Parameters
    ----------
    log_T : log10 temperature [K] (z-independent, per spec).
    log_n : log10 n0 [cm^-3], comoving-mean H density at z=0.
    z     : source redshift; integration runs 0..z.
    E_obs : (N_E,) observer-frame energies, keV.
    n_shells : trapezoid nodes along the line of sight.

    Returns
    -------
    (N_E,) optical depths.
    """
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    log_n = jnp.asarray(log_n, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    e_obs = jnp.asarray(E_obs, dtype=jnp.float64)

    n0 = 10.0**log_n
    zz = jnp.linspace(0.0, z, n_shells)

    def shell(zp: Array) -> Array:
        n_loc = n0 * (1.0 + zp) ** 3
        f, _ = ion_fractions(log_T, n_loc, zp)
        w = _ABUND140_J * f[_IDX152_J]                    # (140,)
        sig = sigma_ions(e_obs * (1.0 + zp))               # (140, N_E)
        sigma_eff = w @ sig                                 # (N_E,)
        dl_dz = _D_H_CM / ((1.0 + zp) * dimensionless_hubble(zp, Planck18))
        return n_loc * sigma_eff * dl_dz

    integrand = jax.vmap(shell)(zz)                        # (n_shells, N_E)
    return jnp.trapezoid(integrand, zz, axis=0)
```

- [ ] **Step 3: Run the full physics test dir**

Run: `uv run pytest tests/physics -v`
Expected: all PASS, including the rewritten validator tests; the old
`test_validator_matches_tiny_grid` is gone (the tiny synthetic grid remains in use by
`test_whim.py` for interpolator mechanics only — do not delete the fixture).

- [ ] **Step 4: Lint, typecheck, commit**

```bash
uv run ruff check . && uv run mypy whimstan
git add whimstan/physics/whim_validator.py tests/physics/test_whim_validator.py
git commit -m "feat(physics): real hybrid CIE+PIE WHIM tau validator (replaces Plan-1 stub)"
```

### Task 9: Phase-2 quality gate

- [ ] **Step 1: Full suite + export check**

Add `atomic`, `abundances`, `ionization` exports to `whimstan/physics/__init__.py`
(`ELEMENTS`, `ion_index`, `sigma_ions`, `n_relative_to_H`, `ion_fractions`,
`ion_fractions_cie` + `__all__` entries, alphabetized like the existing file).

Run: `uv run pytest -v && uv run ruff check . && uv run ruff format --check . && uv run mypy whimstan`
Expected: everything green (37 Plan-1 tests minus 1 removed + new ones).

- [ ] **Step 2: Commit**

```bash
git add whimstan/physics/__init__.py
git commit -m "feat(physics): surface atomic/ionization primitives at physics namespace"
```

---

# Phase 3 — CLOUDY grid pipeline (Tasks 10–12)

### Task 10: CLOUDY input template + single-point format spike

**Files:**
- Create: `tools/build_whim_grid/axes.py`
- Create: `tools/build_whim_grid/cloudy_input/template.in`
- Modify: `tools/build_whim_grid/NOTES.md`

- [ ] **Step 1: Write `axes.py`** — spec §5.2 axes, single source of truth:

```python
"""Grid axes (spec §5.2) + CLOUDY-internal local-density axis."""

import numpy as np

LOG_T = np.linspace(4.5, 7.5, 25)              # stored axis
LOG_N0 = np.linspace(-7.0, -3.0, 17)           # stored axis (z=0 density)
Z_S = np.linspace(0.0, 5.0, 26)                # stored axis (source z)
E_OBS_KEV = np.logspace(np.log10(0.3), 1.0, 1024)  # stored axis

# CLOUDY runs sample *local* density n_loc = n0 (1+z')^3 <= 1e-3 * 6^3.
LOG_HDEN = np.arange(-7.0, -0.499, 0.25)       # 27 points
DZ_SUB = 0.02                                   # sub-shell integration step
ELEM_CLOUDY = {                                 # CLOUDY save-element spellings
    "H": "hydrogen", "He": "helium", "C": "carbon", "N": "nitrogen",
    "O": "oxygen", "Ne": "neon", "Mg": "magnesium", "Si": "silicon",
    "S": "sulphur", "Ar": "argon", "Ca": "calcium", "Fe": "iron",
}
```

- [ ] **Step 2: Write `cloudy_input/template.in`**

```
title whim grid logT={log_T:.4f} z={z:.2f}
cmb redshift {z:.4f}
{uvb_command}
hden -7.0 vary
grid range from -7.0 to -0.5 with 0.25 dex steps
constant temperature, t={log_T:.4f} log
abundances "solar_GASS10.abn"
no molecules
stop zone 1
set dr 0
save grid ".grd" last no hash
save element hydrogen ".elemH" last no hash
save element helium ".elemHe" last no hash
save element carbon ".elemC" last no hash
save element nitrogen ".elemN" last no hash
save element oxygen ".elemO" last no hash
save element neon ".elemNe" last no hash
save element magnesium ".elemMg" last no hash
save element silicon ".elemSi" last no hash
save element sulphur ".elemS" last no hash
save element argon ".elemAr" last no hash
save element calcium ".elemCa" last no hash
save element iron ".elemFe" last no hash
```

- [ ] **Step 3: Single-(T,z) spike run, reduced hden axis**

Render the template at log_T=6.0, z=0.2 with `grid range from -5 to -4 with 0.5 dex steps`
(3 points) into `/tmp/whimspike/spike.in`; run CLOUDY; then inspect:

```bash
head -20 /tmp/whimspike/spike.elemO
grep -c GRID_DELIMIT /tmp/whimspike/spike.elemO
tail -5 /tmp/whimspike/spike.out
```

Record in NOTES.md: column layout of `save element` (expect: depth column then one
ionization-fraction column per stage 1..Z+1; H may carry an H₂ column — note it), the
grid-point delimiter convention, and any warnings CLOUDY emits at hden −7 / logT 4.5
(run a second spike at the extreme corner log_T=4.5, z=5.0 to catch convergence issues
*now*, not 2 hours into the production run).

- [ ] **Step 4: Commit**

```bash
git add tools/build_whim_grid/axes.py tools/build_whim_grid/cloudy_input/template.in \
        tools/build_whim_grid/NOTES.md
git commit -m "feat(grid): CLOUDY input template + grid axes; output-format spike notes"
```

### Task 11: `run_grid.py` — joblib launcher + parser

**Files:**
- Create: `tools/build_whim_grid/run_grid.py`
- Create: `tools/build_whim_grid/parse_cloudy.py`
- Test: `tests/tools/test_parse_cloudy.py` + fixture `tests/fixtures/cloudy_elem_sample/`

- [ ] **Step 1: Copy a real spike output as a test fixture**

Copy the Task-10 spike's `.elemO` and `.elemH` (3 hden points, small files) to
`tests/fixtures/cloudy_elem_sample/`. Real-format fixtures make the parser test honest.

- [ ] **Step 2: Failing parser test**

```python
"""Parser tests against a real CLOUDY `save element` output sample."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "tools" / "build_whim_grid"))

FIX = Path(__file__).parents[1] / "fixtures" / "cloudy_elem_sample"


def test_parse_element_fractions() -> None:
    from parse_cloudy import parse_element_file

    frac = parse_element_file(FIX / "spike.elemO", n_stages=9, n_grid=3)
    assert frac.shape == (3, 9)
    assert np.all(frac >= 0.0) and np.all(frac <= 1.0)
    np.testing.assert_allclose(frac.sum(axis=1), 1.0, atol=1e-3)


def test_parse_hydrogen_renormalizes() -> None:
    from parse_cloudy import parse_element_file

    frac = parse_element_file(FIX / "spike.elemH", n_stages=2, n_grid=3)
    assert frac.shape == (3, 2)
    np.testing.assert_allclose(frac.sum(axis=1), 1.0, atol=1e-3)
```

Run: `uv run pytest tests/tools -v` — Expected: FAIL. (Create `tests/tools/__init__.py`.)

- [ ] **Step 3: Write `parse_cloudy.py`**

```python
"""Parse CLOUDY `save element` outputs (grid mode).

Format (verified in NOTES.md): per grid point, data rows of
depth + fraction-per-stage columns; points separated by GRID_DELIMIT
lines. With `stop zone 1` + `last` there is exactly one data row per
grid point. Extra trailing columns (e.g. H2 for hydrogen) are dropped
and the first n_stages columns renormalized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_element_file(path: Path, n_stages: int, n_grid: int) -> np.ndarray:
    """Returns (n_grid, n_stages) ionization fractions, renormalized."""
    blocks: list[np.ndarray] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if "GRID_DELIMIT" in line:
            if current:
                blocks.append(_last_row(current, n_stages))
                current = []
        elif line.strip() and not line.startswith("#"):
            current.append(line)
    if current:
        blocks.append(_last_row(current, n_stages))
    out = np.vstack(blocks)
    if out.shape[0] != n_grid:
        raise ValueError(f"{path}: expected {n_grid} grid points, got {out.shape[0]}")
    return out / out.sum(axis=1, keepdims=True)


def _last_row(rows: list[str], n_stages: int) -> np.ndarray:
    vals = np.array(rows[-1].split(), dtype=float)
    return np.clip(vals[1 : 1 + n_stages], 0.0, None)  # drop depth col
```

Adjust to the *actual* delimiter/row conventions in NOTES.md if they differ (e.g. some
CLOUDY versions put the delimiter *after* each block including the last — the fixture
test catches this).

- [ ] **Step 4: Write `run_grid.py`**

```python
"""Run the CLOUDY (T, z) grid: 25 x 26 runs, each an internal 27-point
hden grid. Resumable: completed runs (Cloudy exited OK) are skipped.

Run: uv run python tools/build_whim_grid/run_grid.py \
         --cloudy-exe <exe> --out tools/build_whim_grid/runs [--mini]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from axes import LOG_T, Z_S  # noqa: E402
from uvb import uvb_command  # noqa: E402

TEMPLATE = (Path(__file__).parent / "cloudy_input" / "template.in").read_text()
MINI_LOG_T = np.array([5.5, 6.0, 6.5])


def run_point(cloudy_exe: str, data_path: str, out: Path, it: int, iz: int,
              log_t: float, z: float) -> dict[str, object]:
    stem = f"run_T{it:02d}_z{iz:02d}"
    d = out / stem
    outfile = d / f"{stem}.out"
    if outfile.exists() and "Cloudy exited OK" in outfile.read_text():
        return {"stem": stem, "status": "cached"}
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.in").write_text(
        TEMPLATE.format(log_T=log_t, z=z, uvb_command=uvb_command(z))
    )
    subprocess.run(
        [cloudy_exe, "-r", stem], cwd=d, check=False,
        env={"CLOUDY_DATA_PATH": data_path, "PATH": "/usr/bin:/bin"},
    )
    ok = outfile.exists() and "Cloudy exited OK" in outfile.read_text()
    return {"stem": stem, "status": "ok" if ok else "FAILED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloudy-exe", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--mini", action="store_true",
                    help="3 T-points only (sanity grid for the validator gate)")
    args = ap.parse_args()
    data_path = str(Path(args.cloudy_exe).resolve().parent.parent / "data")

    log_t_axis = MINI_LOG_T if args.mini else LOG_T
    points = [
        (it, iz, float(lt), float(z))
        for it, lt in enumerate(log_t_axis)
        for iz, z in enumerate(Z_S)
    ]
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_point)(args.cloudy_exe, data_path, args.out, it, iz, lt, z)
        for it, iz, lt, z in points
    )
    manifest = {
        "mini": args.mini,
        "log_T": [float(x) for x in log_t_axis],
        "n_failed": sum(r["status"] == "FAILED" for r in results),
        "runs": results,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    failed = [r["stem"] for r in results if r["status"] == "FAILED"]
    if failed:
        raise SystemExit(f"{len(failed)} CLOUDY runs failed: {failed[:10]}")
    print(f"all {len(results)} runs OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run parser tests, lint, commit**

```bash
uv run pytest tests/tools -v && uv run ruff check .
git add tools/build_whim_grid/run_grid.py tools/build_whim_grid/parse_cloudy.py \
        tests/tools tests/fixtures/cloudy_elem_sample
git commit -m "feat(grid): joblib CLOUDY grid launcher + save-element parser with real fixture"
```

### Task 12: `assemble.py` — shell integral + HDF5 writer

**Files:**
- Create: `tools/build_whim_grid/assemble.py`
- Create: `tools/build_whim_grid/metadata.yaml.in`
- Test: `tests/tools/test_assemble.py`

- [ ] **Step 1: Write `metadata.yaml.in`**

```yaml
grid_version: "v1"
created: "@DATE@"
git_sha: "@GIT_SHA@"
cloudy_version: "c23.01"
ionizing_background: "@UVB_NAME@"
abundances: "CLOUDY solar_GASS10 (ion balance) / Asplund+2009 (opacity weights)"
metallicity_solar: 1.0
elements: "H He C N O Ne Mg Si S Ar Ca Fe"
cross_sections: "Verner & Yakovlev 1995 (vy95_phfit_v1)"
cosmology: "Planck18 flat: H0=67.4, Om=0.315, OL=0.685"
equilibrium: "hybrid CIE+PIE, constant temperature"
note: "tau scales ~linearly with metallicity; n0 posterior is n0 x (Z/Zsun)"
```

- [ ] **Step 2: Failing assemble test (synthetic mini run dir)**

```python
"""End-to-end assemble test on a fabricated 2x2x3 run directory.

Fractions are constant and concentrated in single known stages, so the
expected tau is an independent direct quadrature computed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "tools" / "build_whim_grid"))


@pytest.fixture
def fake_runs(tmp_path: Path) -> Path:
    """2 T-points x 2 z-points x 3 hden points; O all in stage 6, H all ionized,
    every other element parked in its bare stage (zero opacity)."""
    from assemble import write_fake_run_dir_for_tests

    return write_fake_run_dir_for_tests(
        tmp_path / "runs", log_t=[6.0, 6.5], z=[0.0, 5.0], log_hden=[-7.0, -4.0, -1.0]
    )


def test_assemble_matches_direct_quadrature(fake_runs: Path, tmp_path: Path) -> None:
    from assemble import assemble

    out = assemble(
        runs=fake_runs, out_dir=tmp_path / "dist",
        log_n0=np.array([-5.0, -4.0]), z_s=np.array([0.0, 2.5, 5.0]),
        e_obs=np.logspace(np.log10(0.3), 1.0, 32), dz_sub=0.05,
    )
    with h5py.File(out) as f:
        tau = f["tau"][...]
        assert f["tau"].dtype == np.float32
        assert tau.shape == (2, 2, 3, 32)
        assert np.all(tau[:, :, 0, :] == 0.0)          # z_s = 0
        assert np.all(np.diff(tau, axis=2) >= -1e-7)   # monotone in z_s
        assert "metadata" in f

    # Independent quadrature for (iT=0, in0=1, z_s=2.5): constant fractions
    # mean sigma_eff depends only on E_rest.
    import jax.numpy as jnp
    from whimstan.physics.abundances import n_relative_to_H
    from whimstan.physics.atomic import ion_index, sigma_ions
    from whimstan.physics.cosmology import Planck18, dimensionless_hubble

    e_obs = np.logspace(np.log10(0.3), 1.0, 32)
    zz = np.arange(0.0, 2.5001, 0.05)
    d_h_cm = Planck18.hubble_distance_Mpc * 3.0856775814913673e24
    integrand = np.zeros((zz.size, 32))
    for k, zp in enumerate(zz):
        sig_o6 = np.asarray(sigma_ions(jnp.asarray(e_obs * (1 + zp))))[
            ion_index("O", 6)
        ]
        n_loc = 1e-4 * (1 + zp) ** 3
        dl = d_h_cm / ((1 + zp) * float(dimensionless_hubble(zp, Planck18)))
        integrand[k] = n_loc * n_relative_to_H("O") * sig_o6 * dl
    expected = np.trapz(integrand, zz, axis=0)  # numpy 1.26 in lockfile
    np.testing.assert_allclose(tau[0, 1, 1], expected, rtol=2e-2)
```

Run: `uv run pytest tests/tools/test_assemble.py -v` — Expected: FAIL.

- [ ] **Step 3: Write `assemble.py`**

```python
"""Assemble the WHIM opacity HDF5 from CLOUDY run outputs.

tau[iT, in0, iz_s, iE] = int_0^{z_s} n_loc sigma_eff dl/dz' dz'

with n_loc = n0 (1+z')^3 and sigma_eff from CLOUDY ion fractions
(bilinear in (z', log n_loc) between run planes at fixed T), VY95 cross
sections and Asplund09 abundances. Independent numpy implementation —
deliberately NOT shared with the JAX validator, which cross-checks it.

Run: uv run python tools/build_whim_grid/assemble.py \
         --runs tools/build_whim_grid/runs --out tools/build_whim_grid/dist
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from axes import DZ_SUB, E_OBS_KEV, ELEM_CLOUDY, LOG_HDEN, LOG_N0, LOG_T, Z_S  # noqa: E402
from parse_cloudy import parse_element_file  # noqa: E402
from uvb import UVB_NAME  # noqa: E402

from whimstan.physics.abundances import n_relative_to_H  # noqa: E402
from whimstan.physics.atomic import ELEMENTS, STAGE_OFFSETS  # noqa: E402
from whimstan.physics.cosmology import Planck18  # noqa: E402

MPC_CM = 3.0856775814913673e24
D_H_CM = Planck18.hubble_distance_Mpc * MPC_CM
E_MASTER = np.logspace(np.log10(0.29), np.log10(61.0), 4096)


def _sigma_master() -> np.ndarray:
    """(140, 4096) VY95 sigma on the master rest-frame energy grid."""
    import jax.numpy as jnp

    from whimstan.physics.atomic import sigma_ions

    return np.asarray(sigma_ions(jnp.asarray(E_MASTER)))


def load_fractions(runs: Path, log_t_axis: np.ndarray) -> np.ndarray:
    """(N_T, N_z, N_hden, 152) stage fractions from all run dirs."""
    n_t, n_z, n_h = log_t_axis.size, Z_S.size, LOG_HDEN.size
    frac = np.zeros((n_t, n_z, n_h, sum(z + 1 for _, z in ELEMENTS)))
    for it in range(n_t):
        for iz in range(n_z):
            stem = f"run_T{it:02d}_z{iz:02d}"
            d = runs / stem
            for name, z_elem in ELEMENTS:
                block = parse_element_file(
                    d / f"{stem}.elem{name}", n_stages=z_elem + 1, n_grid=n_h
                )
                s = STAGE_OFFSETS[name]
                frac[it, iz, :, s : s + z_elem + 1] = block
    return frac


def _weights(frac_point: np.ndarray) -> np.ndarray:
    """152-layout fractions -> 140-layout abundance-weighted ion weights."""
    parts = []
    for name, z_elem in ELEMENTS:
        s = STAGE_OFFSETS[name]
        parts.append(n_relative_to_H(name) * frac_point[s : s + z_elem])
    return np.concatenate(parts)


def _e_z(zp: np.ndarray) -> np.ndarray:
    return np.sqrt(Planck18.Omega_m * (1.0 + zp) ** 3 + Planck18.Omega_L)


def tau_slab(frac_t: np.ndarray, sig: np.ndarray, n0: float,
             z_s_axis: np.ndarray, e_obs: np.ndarray, dz_sub: float) -> np.ndarray:
    """tau for one (T, n0): (N_zs, N_E). frac_t: (N_z, N_hden, 152)."""
    z_max = float(z_s_axis[-1])
    zz = np.round(np.arange(0.0, z_max + dz_sub / 2, dz_sub), 10)
    integrand = np.zeros((zz.size, e_obs.size))
    log_e_master = np.log(E_MASTER)
    for k, zp in enumerate(zz):
        n_loc = n0 * (1.0 + zp) ** 3
        # bilinear fraction lookup: z plane x log(hden), both clipped.
        f_point = _bilerp(frac_t, zp, np.log10(n_loc))
        w = _weights(f_point)                                    # (140,)
        # gather sigma at E_rest via linear interp on the master grid
        x = np.log(e_obs * (1.0 + zp))
        j = np.clip(np.searchsorted(log_e_master, x) - 1, 0, E_MASTER.size - 2)
        t = (x - log_e_master[j]) / (log_e_master[j + 1] - log_e_master[j])
        sig_at = sig[:, j] * (1.0 - t) + sig[:, j + 1] * t       # (140, N_E)
        sigma_eff = w @ sig_at
        integrand[k] = n_loc * sigma_eff * D_H_CM / ((1.0 + zp) * _e_z(np.asarray(zp)))
    cum = np.concatenate(
        [np.zeros((1, e_obs.size)),
         np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz_sub, axis=0)]
    )
    idx = np.rint(z_s_axis / dz_sub).astype(int)
    return cum[idx]


def _bilerp(frac_t: np.ndarray, zp: float, log_n_loc: float) -> np.ndarray:
    def lin(grid: np.ndarray, x: float) -> tuple[int, float]:
        i = int(np.clip(np.searchsorted(grid, x) - 1, 0, grid.size - 2))
        t = float(np.clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0.0, 1.0))
        return i, t

    iz, tz = lin(Z_S, zp)
    ih, th = lin(LOG_HDEN, log_n_loc)
    return (
        frac_t[iz, ih] * (1 - tz) * (1 - th)
        + frac_t[iz, ih + 1] * (1 - tz) * th
        + frac_t[iz + 1, ih] * tz * (1 - th)
        + frac_t[iz + 1, ih + 1] * tz * th
    )


def assemble(runs: Path, out_dir: Path, log_n0: np.ndarray = LOG_N0,
             z_s: np.ndarray = Z_S, e_obs: np.ndarray = E_OBS_KEV,
             dz_sub: float = DZ_SUB) -> Path:
    manifest = json.loads((runs / "manifest.json").read_text())
    log_t_axis = np.array(manifest["log_T"])
    frac = load_fractions(runs, log_t_axis)
    sig = _sigma_master()

    slabs = Parallel(n_jobs=-1, verbose=5)(
        delayed(tau_slab)(frac[it], sig, float(10.0 ** ln0), z_s, e_obs, dz_sub)
        for it in range(log_t_axis.size)
        for ln0 in log_n0
    )
    tau = np.array(slabs).reshape(log_t_axis.size, log_n0.size, z_s.size, e_obs.size)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "whim_opacity_grid_v1.h5"
    meta_src = (Path(__file__).parent / "metadata.yaml.in").read_text()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    meta_src = (meta_src.replace("@DATE@", datetime.date.today().isoformat())
                        .replace("@GIT_SHA@", sha)
                        .replace("@UVB_NAME@", UVB_NAME))
    with h5py.File(dest, "w") as f:
        f.create_dataset("log_T", data=log_t_axis)
        f.create_dataset("log_n_H", data=log_n0)
        f.create_dataset("z", data=z_s)
        f.create_dataset("E_obs_keV", data=e_obs)
        f.create_dataset("tau", data=tau.astype(np.float32), compression="gzip")
        meta = f.create_group("metadata")
        for line in meta_src.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta.attrs[k.strip()] = v.strip().strip('"')
    print(f"wrote {dest} shape={tau.shape}")
    return dest


def write_fake_run_dir_for_tests(out: Path, log_t: list[float], z: list[float],
                                 log_hden: list[float]) -> Path:
    """Fabricate run dirs with constant known fractions (O -> stage 6,
    every other element -> bare stage, zero opacity). Test-support only.

    Emits the same textual layout parse_cloudy reads (one data row per
    grid point, GRID_DELIMIT separators — match the Task-11 fixture if
    the real convention differs).
    """
    for it in range(len(log_t)):
        for iz in range(len(z)):
            stem = f"run_T{it:02d}_z{iz:02d}"
            d = out / stem
            d.mkdir(parents=True, exist_ok=True)
            for name, z_elem in ELEMENTS:
                frac = np.zeros(z_elem + 1)
                if name == "O":
                    frac[6] = 1.0       # all O in O VII (stage 6)
                else:
                    frac[z_elem] = 1.0  # bare nucleus: no opacity
                cols = "\t".join(f"{v:.4e}" for v in frac)
                block = f"#depth\tfractions\n1.000e+00\t{cols}\n"
                body = (block + "GRID_DELIMIT\n") * len(log_hden)
                (d / f"{stem}.elem{name}").write_text(body)
    (out / "manifest.json").write_text(json.dumps({
        "mini": True, "log_T": log_t, "z_s": z, "log_hden": log_hden,
        "n_failed": 0, "runs": [],
    }))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    assemble(args.runs, args.out)
```

`write_fake_run_dir_for_tests` must emit files byte-compatible with the real format
(copy the GRID_DELIMIT layout from the Task-11 fixture). Note: the test overrides the
module-level `Z_S`/`LOG_HDEN` lookup via fabricated axes — to support that, have
`load_fractions`/`tau_slab`/`_bilerp` take the axes from the manifest (add `"z_s"` and
`"log_hden"` keys to manifest.json in `run_grid.py` and to the fake dir) rather than
importing the globals. Implement it that way; the code above shows the math, the
manifest is the source of axes.

- [ ] **Step 4: Run, lint, commit**

```bash
uv run pytest tests/tools -v && uv run ruff check . && uv run mypy whimstan
git add tools/build_whim_grid/assemble.py tools/build_whim_grid/metadata.yaml.in \
        tests/tools/test_assemble.py tools/build_whim_grid/run_grid.py
git commit -m "feat(grid): assembler with cosmological shell integral + HDF5 writer"
```

---

# Phase 4 — Sanity grid, production grid, gate, release (Tasks 13–16)

### Task 13: Mini-grid end-to-end + validator cross-check (GO/NO-GO)

This is the moment of truth for the whole plan: CLOUDY ion fractions vs the JAX
ionization solver, integrated to τ, must agree ≤5%.

- [ ] **Step 1: Run the mini grid (3 T × 26 z = 78 CLOUDY runs)**

```bash
cd tools/build_whim_grid && make cloudy   # no-op if built
cd ../.. && uv run python tools/build_whim_grid/run_grid.py \
    --cloudy-exe tools/build_whim_grid/c23.01/source/cloudy.exe \
    --out tools/build_whim_grid/runs_mini --mini
```

Expected: `all 78 runs OK` (~30–60 min on 8 cores; run in background).

- [ ] **Step 2: Assemble the mini grid**

```bash
uv run python -c "
from pathlib import Path
import numpy as np, sys
sys.path.insert(0, 'tools/build_whim_grid')
from assemble import assemble
assemble(Path('tools/build_whim_grid/runs_mini'),
         Path('tools/build_whim_grid/dist_mini'),
         log_n0=np.array([-6.0, -5.0, -4.0]))
"
```

- [ ] **Step 3: Cross-check against the validator**

```bash
uv run python -c "
import numpy as np, jax.numpy as jnp
from whimstan.physics.whim import WhimGrid
from whimstan.physics.whim_validator import compute_tau_from_scratch
g = WhimGrid.load('tools/build_whim_grid/dist_mini/whim_opacity_grid_v1.h5')
worst = 0.0
for iT in range(3):
    for iN in range(3):
        for iz in (5, 13, 25):
            tg = np.asarray(g.tau(g.log_T[iT], g.log_n[iN], g.z[iz], g.E_obs))
            tv = np.asarray(compute_tau_from_scratch(
                log_T=g.log_T[iT], log_n=g.log_n[iN], z=g.z[iz], E_obs=g.E_obs))
            m = tg > 1e-3
            if m.any():
                r = float(np.max(np.abs(tv[m] - tg[m]) / tg[m]))
                worst = max(worst, r)
                print(f'T={float(g.log_T[iT]):.1f} n={float(g.log_n[iN]):.1f} '
                      f'z={float(g.z[iz]):.1f}  maxrel={r:.3f}')
print('WORST', worst)
"
```

Expected: `WORST` < 0.05.

- [ ] **Step 4 (only if >5%): localize the disagreement**

Compare *ion fractions directly* before touching τ: parse a CLOUDY `.elemO` block at a
known (T, n_loc, z) and print next to `ion_fractions(log_T, n_loc, z)` for the same
point. Three buckets:
1. Fractions agree, τ doesn't → integration/interp bug (check `_bilerp` axes, dz_sub,
   E_rest mapping).
2. Fractions disagree only at low n_loc → Γ or n_e treatment (check UVB units: factor
   4π, per-ν vs per-lnν; check that CLOUDY runs and the Γ table really use the identical
   UVB command).
3. Fractions disagree everywhere → α convention or R_cie tail rules (re-run the
   hydrogen case-A check; inspect rungs around the dominant stage).
Document whatever was found in NOTES.md. If agreement stalls between 5–10% only in
PIE-dominated corners (low n, low T) after honest effort, STOP and present the numbers
to the user — relaxing the gate vs adding physics (Auger/secondary ionization) is a
user decision per spec §10.

- [ ] **Step 5: Commit notes**

```bash
git add tools/build_whim_grid/NOTES.md
git commit -m "test(grid): mini-grid CLOUDY-vs-JAX cross-check results"
```

### Task 14: Production grid build

- [ ] **Step 1: Launch the full run (background, hours)**

```bash
uv run python tools/build_whim_grid/run_grid.py \
    --cloudy-exe tools/build_whim_grid/c23.01/source/cloudy.exe \
    --out tools/build_whim_grid/runs
```

650 runs × 27 hden points; ~2–4 h on 8 cores. Resumable on interruption.
Expected: `all 650 runs OK`.

- [ ] **Step 2: Assemble**

```bash
uv run python tools/build_whim_grid/assemble.py \
    --runs tools/build_whim_grid/runs --out tools/build_whim_grid/dist
```

Expected: `wrote .../whim_opacity_grid_v1.h5 shape=(25, 17, 26, 1024)`; file ~45 MB.

- [ ] **Step 3: Spot-check**

```bash
uv run python -c "
from whimstan.physics.whim import WhimGrid
import jax.numpy as jnp
g = WhimGrid.load('tools/build_whim_grid/dist/whim_opacity_grid_v1.h5')
print(g.metadata)
t = g.tau(6.0, -5.0, 2.0, jnp.array([0.5, 1.0, 5.0]))
print('tau(0.5,1,5 keV) =', t)
assert float(t[0]) > float(t[2]) > 0.0
"
```

### Task 15: `needs_grid` gate test + grid fetch helper

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/physics/test_whim_vs_grid.py`
- Modify: `whimstan/physics/whim.py` (add `WhimGrid.fetch`)
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: conftest fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def whim_grid_path() -> Path:
    """Production WHIM grid; skips (visibly, via -ra) when absent."""
    import os

    candidates = [
        os.environ.get("WHIMSTAN_GRID_PATH", ""),
        str(Path.home() / ".cache" / "whimstan" / "whim_opacity_grid_v1.h5"),
        str(Path(__file__).parents[1] / "tools" / "build_whim_grid" / "dist"
            / "whim_opacity_grid_v1.h5"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip(
        "WHIM opacity grid not available: set WHIMSTAN_GRID_PATH or download the "
        "whim-grid-v1 release asset (WhimGrid.fetch())."
    )
```

- [ ] **Step 2: The 5% gate test**

```python
"""The spec §5.5 gate: JAX validator vs CLOUDY production grid, <=5%.

Relative error is evaluated where tau_grid > 1e-3; below that the
absorption factor differs from 1 by <0.1% and the comparison is noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.needs_grid
def test_validator_agrees_with_cloudy_grid(whim_grid_path: Path) -> None:
    from whimstan.physics.whim import WhimGrid
    from whimstan.physics.whim_validator import compute_tau_from_scratch

    grid = WhimGrid.load(whim_grid_path)
    rng = np.random.default_rng(42)
    n_t, n_n, n_z = grid.log_T.shape[0], grid.log_n.shape[0], grid.z.shape[0]
    picks = {
        (int(rng.integers(0, n_t)), int(rng.integers(0, n_n)),
         int(rng.integers(1, n_z)))
        for _ in range(80)
    }
    worst = 0.0
    checked = 0
    for it, i_n, iz in sorted(picks):
        tau_g = np.asarray(
            grid.tau(grid.log_T[it], grid.log_n[i_n], grid.z[iz], grid.E_obs)
        )
        mask = tau_g > 1e-3
        if not mask.any():
            continue
        tau_v = np.asarray(compute_tau_from_scratch(
            log_T=grid.log_T[it], log_n=grid.log_n[i_n], z=grid.z[iz],
            E_obs=grid.E_obs,
        ))
        worst = max(worst, float(np.max(np.abs(tau_v[mask] - tau_g[mask]) / tau_g[mask])))
        checked += 1
    assert checked >= 50, f"only {checked} informative points"
    assert worst < 0.05, f"validator vs grid worst rel err {worst:.3f} >= 5%"
```

Run: `uv run pytest tests/physics/test_whim_vs_grid.py -v`
Expected: PASS (grid exists locally in `tools/build_whim_grid/dist/`).
Also run with `WHIMSTAN_GRID_PATH= mv dist dist.bak`-style absence to confirm the skip
message fires, then restore.

- [ ] **Step 3: Add `WhimGrid.fetch`**

Add to `whimstan/physics/whim.py` (after `load`):

```python
GRID_RELEASE_URL = (
    "https://github.com/grburgess/whimstan/releases/download/"
    "whim-grid-v1/whim_opacity_grid_v1.h5"
)
```

```python
    @classmethod
    def fetch(cls, cache_dir: str | Path | None = None) -> WhimGrid:
        """Download the v1 release grid (~45 MB) to a local cache and load it."""
        import urllib.request

        dest_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "whimstan"
        dest = dest_dir / "whim_opacity_grid_v1.h5"
        if not dest.is_file():
            dest_dir.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(GRID_RELEASE_URL, tmp)
            tmp.rename(dest)
        return cls.load(dest)
```

Unit test (no network): in `tests/physics/test_whim.py` add

```python
def test_fetch_uses_cache_without_network(tmp_path: Path, fixtures_dir: Path) -> None:
    import shutil

    from whimstan.physics.whim import WhimGrid

    cache = tmp_path / "cache"
    cache.mkdir()
    shutil.copy(fixtures_dir / "whim_grid_tiny.h5", cache / "whim_opacity_grid_v1.h5")
    grid = WhimGrid.fetch(cache_dir=cache)
    assert grid.tau_table.ndim == 4
```

- [ ] **Step 4: CI grid step**

In `.github/workflows/ci.yml`, before the pytest step add a best-effort asset download
(grid tests skip if it 404s — fine until Task 16 publishes):

```yaml
      - name: Fetch WHIM grid release asset (optional)
        continue-on-error: true
        run: |
          mkdir -p ~/.cache/whimstan
          curl -fL -o ~/.cache/whimstan/whim_opacity_grid_v1.h5 \
            https://github.com/grburgess/whimstan/releases/download/whim-grid-v1/whim_opacity_grid_v1.h5
```

- [ ] **Step 5: Full suite, lint, typecheck, commit**

```bash
uv run pytest -v && uv run ruff check . && uv run ruff format --check . && uv run mypy whimstan
git add tests/conftest.py tests/physics/test_whim_vs_grid.py tests/physics/test_whim.py \
        whimstan/physics/whim.py .github/workflows/ci.yml
git commit -m "test(physics): 5% CLOUDY-vs-validator gate + WhimGrid.fetch + CI asset pull"
```

### Task 16: Release the grid + docs (USER CONFIRMATION REQUIRED)

- [ ] **Step 1: STOP — confirm with the user before publishing.** Creating the GitHub
release `whim-grid-v1` on `grburgess/whimstan` is public and outward-facing. Present:
file size, metadata dump, gate-test result. Only proceed on explicit OK.

- [ ] **Step 2: Publish**

```bash
gh release create whim-grid-v1 \
  --title "WHIM opacity grid v1 (CIE+PIE, KS19, CLOUDY c23.01)" \
  --notes "Hybrid CIE+PIE WHIM opacity grid for whimstan v1.0. Axes: logT 4.5-7.5 (25), log n0 -7..-3 (17), z 0-5 (26), E_obs 0.3-10 keV (1024). Solar metallicity (Asplund+2009); tau scales ~linearly with Z. Built by tools/build_whim_grid; validated <=5% against an independent JAX ionization solver." \
  tools/build_whim_grid/dist/whim_opacity_grid_v1.h5
```

- [ ] **Step 3: Verify the gate via fetch path**

```bash
rm -f ~/.cache/whimstan/whim_opacity_grid_v1.h5
uv run python -c "from whimstan.physics.whim import WhimGrid; g = WhimGrid.fetch(); print(g.metadata)"
WHIMSTAN_GRID_PATH=$HOME/.cache/whimstan/whim_opacity_grid_v1.h5 \
  uv run pytest tests/physics/test_whim_vs_grid.py -v
```

Expected: download succeeds, gate PASSES on the released asset.

- [ ] **Step 4: README + docs touch-up**

Add a "WHIM opacity grid" section to `README.md`: what the asset is, `WhimGrid.fetch()`
one-liner, regeneration via `make -C tools/build_whim_grid cloudy grid assemble`, the
metallicity/n₀ degeneracy note, and citations (CLOUDY c23.01/Chatzikos+ 2023, KS19,
VY95, CHIANTI, Asplund+2009).

- [ ] **Step 5: Final commit + push**

```bash
uv run pytest -v && uv run ruff check . && uv run mypy whimstan
git add README.md
git commit -m "docs: WHIM opacity grid v1 release notes + fetch instructions"
git push origin vari-whim
```

---

## Exit criteria (M3, spec §9.4)

- [ ] `whim_opacity_grid_v1.h5` (25×17×26×1024, ~45 MB) published as release asset
      `whim-grid-v1` and loadable via `WhimGrid.fetch()`.
- [ ] `tests/physics/test_whim_vs_grid.py` gate: validator vs grid ≤5% (τ>10⁻³ mask),
      ≥50 informative grid points — PASSING against the released asset.
- [ ] Validator stub fully replaced; physics tests green without the grid present
      (skip, not fail).
- [ ] `uv run pytest -v`, `ruff check`, `ruff format --check`, `mypy whimstan` all green.
- [ ] Pipeline reproducible: `make cloudy && make grid && make assemble` documented in
      `tools/build_whim_grid/README.md`; NOTES.md records every format finding.

## Out of scope (deferred)

- Zenodo DOI hosting (spec §12.1 — deferred decision).
- HM12 comparison grid (config seam exists via `uvb.py`; build deferred).
- Auger/secondary-ionization physics in the validator (only if the 5% gate demands it).
- Forward-model consumption of the grid (Plan 3).
