# whimstan v1.0 — Design Spec

**Date:** 2026-05-08
**Status:** Approved (brainstorming complete; pending implementation plan)
**Supersedes:** Stan-based whimstan v0.x (tagged `v0.x-stan-legacy`)

---

## 1. Purpose

Search for the Warm-Hot Intergalactic Medium (WHIM) by hierarchical Bayesian
inference on Swift-XRT GRB afterglow X-ray spectra. Per-GRB nuisance
parameters (photon index, energy flux, host column density) are partially
pooled; the WHIM temperature $T$ and density $n_0$ are global parameters
shared across all GRBs.

This document specifies a clean-slate rewrite of the existing `whimstan`
package. The Stan-based v0.x is preserved at git tag `v0.x-stan-legacy`.

## 2. Scope

### In scope (v1.0)

- Single-instrument: Swift-XRT only.
- Single source class: long GRB afterglows, redshift known.
- Sample: Swift_GRB_WHIM_cut_v2 (205 GRBs as currently in the legacy paper
  data folder; selection cuts re-applied at runtime).
- Inference: numpyro NUTS on a laptop, ~30–60 min per fit at 50–200 GRBs.
- Forward model fully differentiable in JAX.
- Hybrid CIE+PIE WHIM opacity grid generated offline from CLOUDY, with a
  JAX-native validator.
- Population-recovery validation via end-to-end simulations.

### Out of scope (v1.0; extension seams only)

- AGN sources (paper "future work").
- XMM-Newton, Chandra, Athena (multi-instrument).
- GPU sampling (the architecture is GPU-friendly; not validated in v1.0).
- Redshift-evolving WHIM ($T(z)$, $n_0(z)$).
- Non-equilibrium ionization.
- Posterior surrogate / amortized inference.

### Non-goals

- Drop-in compatibility with the v0.x Stan package or 3ML.
- Cluster execution (everything must run on a MacBook M-series; cluster is
  a side-effect, not a target).

## 3. Scientific Model

### 3.1 Forward model

For GRB $i$ in observed energy bin $j$ (channel index $k$):

$$
\mu_{ij} = \sum_k R_{jk} \cdot A_i(E_k) \cdot S_i(E_k) \cdot \Delta E_k \cdot t_i
$$

with source spectrum

$$
S_i(E) = K_i \left(\frac{E}{E_{\rm piv}}\right)^{\gamma_i}
\cdot e^{-N_H^{MW,i}\,\sigma^{\rm TB}(E)}
\cdot e^{-N_H^{host,i}\,\sigma^{\rm TB}(E(1+z_i))}
\cdot e^{-\tau^{\rm WHIM}(E;\,T,\,n_0,\,z_i)}
$$

- $K_i$ = energy flux normalization, 0.4–10 keV. Reduces correlation with
  $\gamma_i$ vs. a photon-flux normalization.
- $E_{\rm piv}$ = pivot energy, 1 keV.
- $\sigma^{\rm TB}$ = TbAbs photoelectric cross section (Wilms+ 2000).
  Precomputed on the channel grid at load time.
- $N_H^{MW,i}$ = Milky Way column at the GRB sky position; **fixed** from
  HI4PI map (via `gdpyc`). Not a parameter.
- $\tau^{\rm WHIM}$ = line-of-sight optical depth from a hybrid CIE+PIE
  opacity grid (§5).
- $R_{jk}$ = RMF; $A_i(E_k)$ = ARF.

Observed counts likelihood: C-statistic (Cash 1979) with profile-likelihood
treatment of the background nuisance counts (Wachter, Leach & Kellogg
1979; standard W-stat formulation). C-stat is preferred over $\chi^2$ for
low-count Poisson regimes characteristic of XRT afterglow tails.

### 3.2 Hierarchy

| Parameter | Level | Prior |
|---|---|---|
| $\gamma_i$ | per-GRB | $\mathcal{N}(\mu_\gamma, \sigma_\gamma)$ |
| $\log_{10} K_i$ | per-GRB | $\mathcal{N}(\mu_K, \sigma_K)$ |
| $\log_{10} N_H^{host,i}$ | per-GRB | skew-$\mathcal{N}(\mu_N, \sigma_N, \alpha_N)$ |
| $\mu_\gamma$ | population | $\mathcal{N}(-2.0, 0.3)$ |
| $\sigma_\gamma$ | population | half-$\mathcal{N}(0, 0.5)$ |
| $\mu_K$ | population | $\mathcal{N}(\log K_{\rm offset}, 1.0)$ |
| $\sigma_K$ | population | half-$\mathcal{N}(0, 0.5)$ |
| $\mu_N$ | population | $\mathcal{N}(21.0, 1.0)$ (log₁₀ cm⁻²) |
| $\sigma_N$ | population | half-$\mathcal{N}(0, 0.5)$ |
| $\alpha_N$ | population | $\mathcal{N}(-1, 2)$ (left-skew prior) |
| $\log_{10} T$ | global (WHIM) | $\mathcal{N}(6.0, 0.75)$, bounded [4.5, 7.5] |
| $\log_{10} n_0$ | global (WHIM) | $\mathcal{N}(-5.0, 1.0)$, bounded [−7, −3] |
| $N_H^{MW,i}$ | per-GRB | fixed, from HI4PI |

The skew-normal prior on $\log_{10} N_H^{host}$ captures the observed and
theoretically motivated excess of low-density absorbing columns. Kept from
v0.x because simulations confirm the population follows it.

### 3.3 Partial centering reparameterization

Per-GRB parameters use **information-adaptive partial centering**
(Papaspiliopoulos, Roberts & Sköld 2007; Gorinova+ 2020). For each GRB $i$:

$$
c_i = \mathrm{sigmoid}\left(\frac{\log_{10}(N_{{\rm cnt},i}) - 2.7}{0.4}\right) \in (0, 1)
$$

with `numpyro.infer.reparam.LocScaleReparam(centered=c_i)` applied per
parameter group. Bright GRBs (~10⁴ counts) → $c_i \approx 0.99$ (centered);
faint GRBs (~10² counts) → $c_i \approx 0.05$ (non-centered). Avoids both
funnel pathology (faint) and posterior–hyperparam correlation (bright)
without an extra MCMC parameter. Threshold and scale are config-tunable.

The vector $\{c_i\}$ is logged into the InferenceData output for
transparency.

### 3.4 Model variants

The current v0.x repository has 10 separate Stan files for variant
combinations (with/without WHIM, MW, host; T fixed/free). v1.0 collapses
these into a single parameterized model controlled by config flags:

```yaml
model:
  whim:    {enabled: true,  fixed_T: null, fixed_n0: null}
  host:    {enabled: true}
  mw:      {enabled: true}
  partial_centering: {enabled: true, threshold_log10_counts: 2.7, scale: 0.4}
```

Setting `whim.enabled: false` produces the no-WHIM null model, used for
Bayes-factor / leave-one-out comparison against the WHIM-on model.

## 4. Architecture

### 4.1 Package layout

```
whimstan/
├── data/                    # OGIP I/O via jaxspec
│   ├── ogip.py              # PHA/RMF/ARF loaders → JAX arrays
│   ├── catalog.py           # GRB catalog (csv/parquet); MW nH lookup
│   └── dataset.py           # Dataset = list[Observation]; precomputed arrays
├── physics/                 # Forward-model physics, all JAX, pure functions
│   ├── tbabs.py             # Wilms+ 2000 ISM cross section, JAX
│   ├── whim.py              # Hybrid opacity: grid load + JAX trilinear interp
│   ├── whim_validator.py    # JAX from-scratch τ for cross-validation tests
│   ├── cosmology.py         # FlatLambdaCDM in JAX (no astropy at runtime)
│   └── powerlaw.py          # Energy-flux-normalized PL, JAX
├── model/                   # numpyro probabilistic models
│   ├── forward.py           # Pure JAX spectrum(θ) → predicted_counts
│   ├── likelihood.py        # C-stat (Cash 1979 + bkg profiling), JAX
│   ├── priors.py            # Hyperprior factory (configurable)
│   ├── reparam.py           # Counts-driven partial-centering helper
│   └── hbm.py               # Top-level HBM(spec) → numpyro model fn
├── sim/                     # Population synthesis (numpy/JAX, no popsynth)
│   ├── luminosity.py        # Schechter LF, SFR z-distribution
│   ├── host.py              # Ellipsoidal-cloud nH (ported from v0 cloud.py)
│   ├── selection.py         # Counts threshold, MW nH cut, |b| cut
│   └── generate.py          # Pipeline: pop → spectra → counts via forward.py
├── infer/                   # Inference orchestration
│   ├── config.py            # OmegaConf schema for fit/sim runs
│   ├── runner.py            # NUTS driver, warmup/sample, ArviZ output
│   └── diagnostics.py       # R̂/ESS/divergence gates; PPCs
├── utils/                   # logging, colors, dist plotters (light port from v0)
└── cli.py                   # `whimstan fit|simulate|ppc`

tools/
├── build_whim_grid/         # Out-of-tree CLOUDY pipeline, regenerable
│   ├── Makefile
│   ├── cloudy_input/
│   ├── run_grid.py
│   ├── assemble.py
│   └── metadata.yaml.in
└── stan_xcheck/             # One-time Stan cross-check; not on import path

tests/                       # see §6
docs/                        # mkdocs-material
```

### 4.2 Boundaries

Each unit must be testable in isolation:

- `data.ogip` knows nothing about physics or numpyro. Returns dataclasses
  + JAX arrays.
- `physics.*` are pure JAX functions. No I/O. No numpyro. Fully `jit`,
  `vmap`, `grad` compatible.
- `model.forward.spectrum(theta_i, observation_static, whim_grid)` is the
  **single source of truth** consumed by both `sim/generate.py` and
  `model/hbm.py`. There is no path by which simulation and inference can
  use different physics.
- `model.hbm(spec)` builds the numpyro model from a config. Spec encodes
  which terms are enabled.
- `infer.runner` knows nothing about physics. Invokes the model function,
  returns ArviZ.

### 4.3 Data flow (single fit)

```
  GRB catalog (csv) ─┐
  PHA/RMF/ARF       ─┼─► data.dataset.Dataset ──► model.forward.spectrum
  HI4PI MW nH map   ─┘                                ▲
                                                      │ θ (numpyro sample)
  WHIM grid (HDF5)  ──► physics.whim.WhimGrid ───────┘
                                                      │
                                                      ▼
                                          likelihood ──► NUTS ──► ArviZ ──► PPCs
```

### 4.4 Forward-model reference signature

```python
def spectrum(theta_i: PerGRBParams,
             static_i: ObservationStatic,
             whim_grid: WhimGrid) -> Array:
    """Predicted counts per channel for one GRB. Pure JAX."""
    E_avg, E_width, exposure, arf, rmf = static_i.unpack()
    pl       = powerlaw_eflux(E_avg, theta_i.log_K, theta_i.gamma,
                              E_lo=0.4, E_hi=10.0, E_piv=1.0)
    abs_mw   = jnp.exp(-static_i.NH_mw * tbabs_sigma(E_avg))
    abs_host = jnp.exp(-jnp.power(10.0, theta_i.log_NHh)
                       * tbabs_sigma(E_avg * (1 + static_i.z)))
    abs_whim = jnp.exp(-whim_grid.tau(theta_i.log_T,
                                      theta_i.log_n0,
                                      static_i.z, E_avg))
    flux_at_E    = pl * abs_mw * abs_host * abs_whim
    integral_flux = flux_at_E * E_width
    counts        = (rmf @ (arf * integral_flux)) * exposure
    return counts
```

Single jit boundary around the full likelihood; `vmap` over GRBs inside
the numpyro model.

## 5. WHIM Opacity Grid

### 5.1 Artifact

`whim_opacity_grid_v1.h5`, ~50 MB float32, distributed as a GitHub release
asset. Not committed to git.

### 5.2 Axes

| Axis | Range | Points | Spacing |
|---|---|---|---|
| $\log_{10} T$ [K] | 4.5 – 7.5 | 25 | uniform in log |
| $\log_{10} n_H$ [cm⁻³] | −7.0 – −3.0 | 17 | uniform in log |
| $z$ | 0 – 5 | 26 | uniform |
| $E_{\rm obs}$ [keV] | 0.3 – 10 | 1024 | uniform in log |

Stored quantity is the line-of-sight optical depth $\tau(E_{obs}, T, n_H, z)$
with the cosmological shell integral **already performed** at grid-build
time. Inference-time interpolation is one trilinear lookup over
$(T, n_H, z)$ producing a vector over $E_{obs}$. Trilinear is sufficient
given 5% physics tolerance and gives smooth gradients; cubic introduces
edge ringing and is rejected.

### 5.3 Physics inputs (pinned in metadata)

- **Ionizing background:** Khaire & Srianand 2019 (KS19). HM12 selectable
  by config flag for backwards-compatibility comparison against legacy
  literature.
- **Abundances:** Asplund+ 2009 ("aspl" in CLOUDY). Upgrade from v0's
  Anders & Grevesse 1989 ("angr").
- **Elements:** H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe (12 elements;
  v0 had 10).
- **Cosmology:** Planck18 ($\Omega_m=0.315$, $\Omega_\Lambda=0.685$,
  $H_0=67.4$).
- **Equilibrium:** full hybrid CIE + PIE at each grid point ("constant
  temperature" CLOUDY mode).

### 5.4 Build pipeline (`tools/build_whim_grid/`)

1. `Makefile` — pinned CLOUDY version (C23.01); `make grid` regenerates
   everything reproducibly.
2. `cloudy_input/` — parameterized CLOUDY input templates.
3. `run_grid.py` — joblib-parallel CLOUDY launcher across grid points.
4. `assemble.py` — collect outputs, integrate $\tau$ along the cosmological
   line of sight via redshift shells (cosmology fixed in §5.3), write HDF5.
5. `metadata.yaml.in` — versions, axes, abundance choice, ionizing
   background ID, git SHA, build date. Embedded inside the HDF5 so every
   fit logs the exact grid identity used.

### 5.5 JAX validator (`physics/whim_validator.py`)

Pure-JAX from-scratch implementation of Saha–Boltzmann + photoionization
balance using a small CHIANTI-derived atomic-data file (committed to
repo). Computes $\tau$ at a sparse subset of grid points (~50). Required
to agree with the CLOUDY grid to within 5% as a unit test gate; failure
blocks the build. Doubles as a sanity check on the interpolator.

### 5.6 Inference-time API

```python
grid = WhimGrid.load("whim_opacity_grid_v1.h5")  # JAX-backed dataclass
tau  = grid.tau(log_T, log_n0, z, E_obs)         # jit/vmap-safe
absorption = jnp.exp(-tau)
```

Edge handling: clip-to-boundary with a fit-time warning if any chain
leaves the grid.

## 6. Tests, CI, Tooling

### 6.1 Test inventory

| File | Purpose | Speed |
|---|---|---|
| `test_ogip.py` | jaxspec round-trip PHA/RMF/ARF, shape & units | <5s |
| `test_physics_tbabs.py` | TbAbs σ vs. Wilms+ 2000 reference | <1s |
| `test_physics_whim.py` | grid load, interpolation, gradient finiteness | <2s |
| `test_physics_whim_validator.py` | JAX validator vs. CLOUDY grid, ≤5% | <30s |
| `test_forward.py` | spectrum() shapes, jit/vmap clean, finite gradients | <5s |
| `test_likelihood.py` | C-stat against analytic test cases | <1s |
| `test_reparam.py` | partial-centering $c_i$ monotone in counts | <2s |
| `test_sim_population.py` | luminosity, host, selection samplers | <5s |
| `test_sim_recovery.py` | 10-GRB synthetic fit; truth in 90% CI | ~2 min |
| `test_diagnostics.py` | R̂/ESS/divergence gates fire as expected | <10s |

### 6.2 CI

`.github/workflows/ci.yml` — single workflow:

- ubuntu-latest + macos-latest, Python 3.11 + 3.12 matrix.
- Steps: `uv sync`, `ruff check`, `mypy`, `pytest -m "not slow"`, codecov.
- `test_sim_recovery.py` marked `@pytest.mark.slow`; runs nightly via
  scheduled cron, not on PRs.
- WHIM grid pulled as a workflow artifact / release asset. If absent,
  grid-dependent tests `xfail` with a clear message.

Stan cross-check (M7) is NOT in CI. It runs once before the v1.0 merge,
result archived in `docs/validation/`.

### 6.3 Tooling

| Concern | Choice |
|---|---|
| Build backend | `hatchling` |
| Versioning | `hatch-vcs` (git-tag-based; replaces versioneer) |
| Dep manager | `uv` lockfile, all runtime deps pinned |
| Lint/format | `ruff` (replaces black + isort + flake8) |
| Type check | `mypy --strict` on `whimstan/`, relaxed on `tests/` |
| Pre-commit | ruff, mypy, eof, trailing-whitespace |
| Docs | `mkdocs-material` (replaces sphinx) |
| Config | `omegaconf` (kept from v0) |
| Logging | stdlib `logging` + `rich` handler for CLI |
| Min Python | 3.11 |

### 6.4 Runtime dependencies (final)

`numpy`, `scipy`, `jax`, `numpyro`, `arviz`, `jaxspec`, `h5py`, `astropy`,
`healpy`, `gdpyc`, `omegaconf`, `rich`, `matplotlib`, `joblib`. All pinned.

**Dropped from v0.x:** `threeml`, `astromodels`, `bb_astromodels`,
`popsynth`, `cmdstanpy`, `numba`, `twopc`, `reproject`, `natsort`,
`versioneer`, `ipython`.

## 7. Sampling, Diagnostics, Output

### 7.1 NUTS defaults

- 4 chains.
- 1000 warmup + 1000 sampling.
- `target_accept = 0.9`.
- `max_tree_depth = 10`.
- `init_strategy = init_to_median`, warmed by a 10-second SVI run on the
  same model (helps NUTS find the typical set fast on a laptop).
- `JAX_ENABLE_X64=True` enforced at module import; checked on startup
  (raises if False).

### 7.2 Diagnostic gates

Reported prominently as warnings (not exceptions):

- $\hat R < 1.01$ on every parameter.
- $\mathrm{ESS}_{\rm bulk} > 400$ on every parameter.
- Divergences = 0.

Failure does not abort the fit; it is logged and surfaced in the JSON
summary. Re-fits are a user decision.

### 7.3 Output

For each fit:

- ArviZ `InferenceData` → NetCDF (`<run_id>.nc`).
- JSON summary: gates, runtime, grid identity, software versions, config
  digest, partial-centering vector $\{c_i\}$.
- PPC plots: per-GRB spectrum-vs-data, population-level posterior of
  hyperparameters.

## 8. Simulation Pipeline (validation)

### 8.1 Driver

```python
def simulate_population(cfg: SimConfig, key: PRNGKey) -> SimulatedDataset:
    truths   = sample_population(cfg, key)
    obs_args = build_observations(truths, cfg)
    counts   = vmap(spectrum)(truths, obs_args, cfg.whim_grid)
    counts_obs = poisson_sample(counts, key)
    bkg_obs    = poisson_sample(predicted_bkg(cfg), key)
    selected   = apply_selection(truths, counts_obs, cfg)
    return SimulatedDataset(truths, counts_obs, bkg_obs, mask=selected)
```

### 8.2 Sampled quantities

| Quantity | Distribution / source |
|---|---|
| $z$ | SFR-shaped, Cole+ 2001 form, $z_{\max}$ configurable |
| $L_{\rm iso}$ | Schechter LF, $L_*, \alpha$ from cfg |
| $K_i$ | derived from $(L_{\rm iso}, z, K\text{-correction})$ |
| $\gamma_i$ | $\mathcal{N}(\mu_\gamma^{\rm true}, \sigma_\gamma^{\rm true})$ |
| $N_H^{MW,i}$ | HI4PI lookup at sampled $(\ell, b)$, via healpy |
| $N_H^{host,i}$ | ellipsoidal-cloud (port from v0 `cloud.py`); log-normal fallback |
| $T, n_0$ (WHIM) | scalars from cfg (truths) |
| Selection | counts > 500, $N_H^{MW} < 10^{21}$, $\|b\| > 10°$ |

### 8.3 Output schema

Identical to real-data `Dataset` with an extra `truths` group. Inference
is dataset-source-agnostic: same code path runs on simulations and on
real data.

### 8.4 Recovery validation script

`whimstan/scripts/validate_recovery.py`:

1. Inject 100 GRBs with known $(T, n_0, \mu_\gamma, \sigma_\gamma, \mu_K,
   \sigma_K, \mu_N, \sigma_N, \alpha_N)$.
2. Fit the full HBM.
3. Assert each true population-level value lies within the 90% credible
   interval of its posterior.
4. Save truth-vs-posterior plots as a CI artifact.

## 9. Migration

### 9.1 Branching

1. Tag current master as `v0.x-stan-legacy` (preserves Stan version
   indefinitely).
2. Branch `rewrite/v1` off master.
3. Wipe `whimstan/` package contents and `.github/workflows/*.yml`
   (legacy CI). Keep `LICENSE`, `pyproject.toml` (rewrite), `README.md`
   (rewrite), and any non-package docs assets worth preserving. Replace
   docs from scratch under `docs/` with mkdocs-material.
4. Build v1 in milestones below.
5. Merge to master once §6 tests pass + recovery validation passes + Stan
   x-check archived.
6. Tag `v1.0.0`.

### 9.2 Reference checkout

```bash
git worktree add ../whimstan-legacy v0.x-stan-legacy
```

### 9.3 Files to port from v0

- `simulations/cloud.py` → `sim/host.py` (ellipsoid geometry; JAX-ify the
  path-length sampler).
- `utils/colors.py`, `utils/configuration.py` → `whimstan/utils/` (light
  cleanup, kept).
- `database/catalog.py` schema → `data/catalog.py` (column names, csv
  format).
- `articles/whim_paper/src/figures/Swift_GRB_WHIM_cut_v2.txt` →
  `whimstan/data/grb_sample_v2.csv` (canonical 205-GRB sample).

### 9.4 Milestones

| M | Scope | Exit criterion |
|---|---|---|
| M0 | Tooling: hatchling, uv, ruff, mypy, CI skeleton; drop versioneer | CI green on empty package |
| M1 | `data/`: jaxspec OGIP loaders, catalog, dataset; tests | Round-trip on 5 real Swift-XRT files |
| M2 | `physics/`: tbabs, powerlaw, cosmology, whim grid loader, JAX validator | All physics unit tests pass |
| M3 | CLOUDY pipeline + grid release | First grid HDF5 published as release asset |
| M4 | `model/`: forward, likelihood, priors, reparam, hbm | jit/vmap/grad clean on synthetic obs |
| M5 | `sim/`: population gen + recovery harness | 10-GRB synthetic recovery passes |
| M6 | `infer/`: runner, diagnostics, CLI | Full fit on 50 GRBs in <60 min produces InferenceData + PPCs |
| M7 | Stan cross-check (one-time, `tools/stan_xcheck/`) | <2σ MCSE difference on hyperparams; archived |
| M8 | Docs (mkdocs), tutorials, paper figures script | Docs deploy; figures regenerable |
| M9 | Tag v1.0.0; archive legacy tag in README | Release published |

Solo-developer estimate: M0–M2 ~1 wk; M3 ~1 wk (CLOUDY runtime
dominated); M4–M5 ~1.5 wk; M6 ~1 wk; M7 ~3 days; M8 ~4 days. **Total
≈ 5–6 weeks.**

## 10. Risks

| Risk | Mitigation |
|---|---|
| jaxspec API immaturity / breaking changes | Pin version; thin wrapper layer in `data/ogip.py` so swap to hand-rolled loader is mechanical |
| CLOUDY grid build fails or disagrees with JAX validator > 5% | Validator gate in CI catches; SPEX `hot`+`pion` is the fallback (~3 days extra work) |
| numpyro vs. Stan posterior mismatch on M7 | Bayeux-compatible model interface kept as escape hatch; if persistent, escalate to Stan as primary |
| WHIM grid memory at scale (multi-instrument later) | float32 storage + lazy slicing on z; out of scope for v1.0 |
| Real Swift-XRT data not in repo | Distribute as a separate data release; CI uses synthetic-only |
| Partial-centering threshold poorly tuned | Counts threshold + scale are config; recovery test catches gross misconfiguration |

## 11. Glossary

- **WHIM** — Warm-Hot Intergalactic Medium. Diffuse ionized gas, $T \sim
  10^{5.5}$–$10^{7}$ K, $n_H \sim 10^{-6}$–$10^{-4}$ cm⁻³.
- **CIE** — Collisional Ionization Equilibrium.
- **PIE** — Photoionization Equilibrium.
- **HBM** — Hierarchical Bayesian Model.
- **C-stat** — Cash 1979 likelihood for Poisson data.
- **OGIP** — NASA standard for X-ray spectral file formats (PHA, RMF, ARF).
- **TbAbs** — Tübingen-Boulder ISM absorption model (Wilms+ 2000).
- **NUTS** — No-U-Turn Sampler (Hoffman & Gelman 2014).
- **PPC** — Posterior Predictive Check.
- **SBC** — Simulation-Based Calibration (Talts+ 2018).
- **HM12 / KS19** — Haardt-Madau 2012 / Khaire-Srianand 2019 ionizing
  backgrounds.
- **Asplund+ 2009 / "aspl"** — modern solar abundance set.

## 12. Unresolved questions

(Per project convention; deferred, not blocking.)

1. Grid hosting (release asset vs. Zenodo DOI) — deferred.
2. Authorship of any rewrite-derived paper — deferred.
3. License — deferred (current GPL-2+ retained for now).
