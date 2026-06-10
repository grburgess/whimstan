# WHIM Opacity Grid Builder

This pipeline produces `whim_opacity_grid_v1.h5`, a ~45 MB HDF5 file distributed
as a release asset, which encodes the X-ray photoelectric opacity of the warm-hot
intergalactic medium (WHIM) as a function of temperature, density, and redshift.
The grid is computed from first principles using the photoionization code CLOUDY
c23.01 under the Khaire & Srianand 2019 (KS19) ionizing background, and it is the
primary opacity table consumed by the JAX/numpyro inference engine in whimstan.

## Prerequisites

A C++ compiler (clang on macOS, gcc on Linux) and standard POSIX tools (`curl`,
`make`, `tar`). The Python steps require `uv`.

## Build targets

```
make cloudy     # download (~1 GB) and compile CLOUDY c23.01 (10–20 min)
make smoke      # one-line sanity check — should print "CLOUDY SMOKE OK"
make grid       # run all CLOUDY grid points (2–4 h on 8 cores)
make assemble   # collect per-point outputs into whim_opacity_grid_v1.h5
make clean-runs # remove runs/ and dist/ but leave the CLOUDY binary intact
```

Run these targets from the `tools/build_whim_grid/` directory.  The CLOUDY source
tree and all intermediate outputs are gitignored; only the Makefile, the Python
drivers, and the final assembled HDF5 file (when promoted to a release) are
tracked.

## Metallicity note

The grid is built at solar metallicity (Asplund+2009 pattern and amplitude).
X-ray τ is metal-dominated, so τ scales ~linearly with Z and is degenerate with
n₀: the inferred n₀ should be read as n₀ × (Z/Z_⊙).

It is therefore important to note that fits which do not fix the metallicity are
underdetermined; the pipeline intentionally fixes Z = Z_⊙ to reduce the
dimensionality of the inference problem, and we leave a generalization to variable
Z for a future study.

## Output format

`whim_opacity_grid_v1.h5` contains a single dataset `tau` of shape
`(N_z, N_T, N_nH, N_E)` plus coordinate axes `redshift`, `log_T`, `log_nH`, and
`energy_keV`.  The JAX trilinear interpolator in `whimstan/physics/whim_grid.py`
loads this file at inference time.
