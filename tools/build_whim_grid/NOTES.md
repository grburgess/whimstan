# Task-2 Spike: UVB Reconnaissance Notes

## CLOUDY version

Build performed 2026-06-10 from `c23.01.tar.gz` on macOS 25.3 (Apple Silicon,
clang++ 1700). No build adjustments were required; the standard `make -j$(nproc)`
in `c23.01/source/` completed without errors.

Version banner from the smoke test:

```
Cloudy (vari, whim, v0.x, stan, legacy, modified)
Cloudy compiled on Jun 10 2026 in OS Apple MacOS using the clang++ 1700 compiler.
Mode I32LP64, denormalized float: T double: T.
```

## KS19 data files

The c23.01 data directory ships seven CLOUDY-format KS19 SED grids, one per Q
value (Q14–Q20):

```
c23.01/data/ks19_q14.ascii
c23.01/data/ks19_q15.ascii
c23.01/data/ks19_q16.ascii
c23.01/data/ks19_q17.ascii
c23.01/data/ks19_q18.ascii
c23.01/data/ks19_q19.ascii
c23.01/data/ks19_q20.ascii
```

These are loaded by the built-in `TABLE KS19` command; no external download is
needed.

## UVB command pinned

```
table KS19 redshift {z:.4f}
```

**Q parameter:** omitted (defaults to Q=18, the fiducial model of
Khaire & Srianand 2019).

**What was tried first:** The Hazy1 documentation (cont-shp.tex) documents the
command as `TABLE KS18`, but `parse_table.cpp` in c23.01 prints
`"The TABLE KS18 command has been replaced by the TABLE KS19 command."` and
exits. The rename was introduced silently between documentation and binary; the
correct command in this version is `TABLE KS19`.

## CLOUDY invocation style

CLOUDY reads input from a named file, not from stdin in production use.  The
invocation pattern is:

```
cd <rundir>
CLOUDY_DATA_PATH=<repo>/tools/build_whim_grid/c23.01/data \
    <repo>/tools/build_whim_grid/c23.01/source/cloudy.exe -r <stem>
```

CLOUDY reads `<stem>.in` and writes `<stem>.out` plus any save files (e.g.
`<stem>.con`).  Output files are written to the current working directory;
subdirectory paths in `save` filenames are not supported by c23.01 (the binary
prints "writing in subdirectories is not supported" and exits).

The stdin pipe form (`printf 'test\n' | cloudy.exe`) works for the built-in
self-test only.

## `save incident continuum` column format

Command used:

```
save incident continuum units keV "uvbtest.con" last
```

Output header: `#Incident Continuum, Enr<TAB>nFn<TAB>Occ Num`

| Column | Name    | Units                            | Notes                                      |
|--------|---------|----------------------------------|--------------------------------------------|
| 1      | Enr     | keV                              | Photon energy (AnuUnit with `units keV`)   |
| 2      | nFn     | erg cm⁻² s⁻¹                     | νFν = EF_E, 4π-integrated (= 4πEJ_E)      |
| 3      | Occ Num | dimensionless                    | Photon occupation number                   |

The `nFn` column is computed as `ν² × (photon flux / Δν) × h × 4πr²_inner`
(see `flxCell` in `save_do.cpp`), which gives the 4π-steradians-integrated
specific intensity νJ_ν × 4π = νF_ν in erg cm⁻² s⁻¹.  The grid pipeline uses
only column 2 (νFν) for the ionization parameter; column 3 is not used.

Energy grid runs from ~4 × 10⁻¹¹ keV (far IR) to ~9.6 × 10⁴ keV (9244 rows
for the z=2 KS19 Q18 field).

## z=2 verification run output (first 4 data rows)

```
#Incident Continuum, Enr    nFn         Occ Num
4.14257e-11    1.8065e-15    9.6849e+02
4.15640e-11    1.8247e-15    9.6526e+02
4.17028e-11    1.8430e-15    9.6205e+02
4.18420e-11    1.8615e-15    9.5885e+02
```

## Build notes

No compiler flags or source patches were needed on Apple Silicon (arm64) with
Xcode clang++ 1700.  The Makefile uses `$(shell sysctl -n hw.ncpu)` to detect
core count (returns 16 on the build machine).  The `$(MAKE) -C ... -j$(NPROC)`
build completed in approximately 14 minutes.
