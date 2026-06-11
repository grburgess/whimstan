"""Build VY95 per-shell photoionization fit-parameter table shipped with whimstan.

Downloads VizieR J/A+AS/109/125 table1, parses the byte-by-byte format
described in the ReadMe, and writes ``whimstan/physics/_data/vy95_phfit_v1.npz``
containing arrays of shape (140, 7):

* ``E_th_eV``   -- subshell threshold energy, eV
* ``E0_eV``     -- fit parameter E_0, eV
* ``sigma0_Mb`` -- fit parameter sigma_0, Mb
* ``y_a``       -- fit parameter y_a
* ``P``         -- fit parameter P
* ``y_w``       -- fit parameter y_w
* ``l_q``       -- orbital quantum number l (float, for Q = 5.5 + l - 0.5*P)
* ``shell_mask``-- bool, True where a shell entry exists

Row ordering: ELEMENTS tuple order (H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe),
stages 0..Z-1 within each element (stage = electrons removed, so Z-stage electrons).
MAX_SHELLS = 7 (neutral Fe: 1s 2s 2p 3s 3p 3d 4s).

Run::

    uv run python tools/build_vy95_table.py

Source::

    Verner & Yakovlev 1995, A&AS 109, 125
    https://cdsarc.cds.unistra.fr/ftp/J/A+AS/109/125/table1
"""

from __future__ import annotations

import io
import pathlib
import urllib.request
from collections import defaultdict
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TABLE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+AS/109/125/table1"

# Elements and atomic numbers (Z).
ELEMENTS: tuple[tuple[str, int], ...] = (
    ("H", 1),
    ("He", 2),
    ("C", 6),
    ("N", 7),
    ("O", 8),
    ("Ne", 10),
    ("Mg", 12),
    ("Si", 14),
    ("S", 16),
    ("Ar", 18),
    ("Ca", 20),
    ("Fe", 26),
)

# Total absorbing ions: stages 0..Z-1 per element.
N_IONS = sum(z for _, z in ELEMENTS)  # 140
MAX_SHELLS = 7

OUT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "whimstan/physics/_data/vy95_phfit_v1.npz"
)

# ---------------------------------------------------------------------------
# ReadMe byte ranges (1-based, inclusive):
#   1- 2  Z
#   4- 5  N (electrons)
#   7- 7  n (principal q.n.)
#   9- 9  l (orbital q.n.)
#  11-20  E_th (eV)
#  22-31  E_0  (eV)
#  33-42  sigma_0 (Mb)
#  44-53  y_a
#  55-64  P
#  66-75  y_w
# ---------------------------------------------------------------------------


class ShellRow(NamedTuple):
    E_th: float
    E0: float
    sigma0: float
    y_a: float
    P: float
    y_w: float
    l_orb: int


def fetch_table() -> list[str]:
    """Download table1 and return list of lines."""
    print(f"Downloading {TABLE_URL} …")
    with urllib.request.urlopen(TABLE_URL, timeout=60) as resp:
        raw = resp.read()
    return io.TextIOWrapper(io.BytesIO(raw), encoding="ascii").readlines()


def parse_table(lines: list[str]) -> dict[tuple[int, int], list[ShellRow]]:
    """Parse VY95 table1; return {(Z, N): [ShellRow, ...]}.

    Skips blank/comment lines robustly. Byte ranges per ReadMe (0-indexed
    slices): Z=0:2, N=3:5, n=6:7, l=8:9, E_th=10:20, E0=21:31,
    sigma0=32:42, y_a=43:53, P=54:64, y_w=65:75.
    """
    data: dict[tuple[int, int], list[ShellRow]] = defaultdict(list)
    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if len(line) < 20:
            continue
        try:
            z_val = int(line[0:2])
            n_val = int(line[3:5])
            # principal q.n. n = int(line[6:7])  -- not stored; unused in formula
            l_orb = int(line[8:9])
            e_th = float(line[10:20])
            e0 = float(line[21:31])
            sigma0 = float(line[32:42])
            y_a = float(line[43:53])
            p_fit = float(line[54:64])
            y_w = float(line[65:75])
        except (ValueError, IndexError) as exc:
            print(f"  skipping line {lineno}: {exc!r}  [{line!r}]")
            continue
        data[(z_val, n_val)].append(ShellRow(e_th, e0, sigma0, y_a, p_fit, y_w, l_orb))
    return data


def build_arrays(
    data: dict[tuple[int, int], list[ShellRow]],
) -> dict[str, np.ndarray]:
    """Build (140, 7) arrays from parsed data."""
    shape = (N_IONS, MAX_SHELLS)
    arr_eth = np.zeros(shape, dtype=np.float64)
    arr_e0 = np.zeros(shape, dtype=np.float64)
    arr_sigma0 = np.zeros(shape, dtype=np.float64)
    arr_ya = np.zeros(shape, dtype=np.float64)
    arr_p = np.zeros(shape, dtype=np.float64)
    arr_yw = np.zeros(shape, dtype=np.float64)
    arr_lq = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)

    row_idx = 0
    for name, z in ELEMENTS:
        for stage in range(z):  # stage = electrons removed; ion has z-stage electrons
            n_elec = z - stage  # electrons in ion
            key = (z, n_elec)
            shells = data.get(key, [])
            if not shells:
                raise RuntimeError(
                    f"No shells found for {name} (Z={z}, stage={stage}, N={n_elec})"
                )
            n_shells = len(shells)
            if n_shells > MAX_SHELLS:
                raise RuntimeError(
                    f"{name} stage={stage} has {n_shells} shells > MAX_SHELLS={MAX_SHELLS}"
                )
            for col, sh in enumerate(shells):
                arr_eth[row_idx, col] = sh.E_th
                arr_e0[row_idx, col] = sh.E0
                arr_sigma0[row_idx, col] = sh.sigma0
                arr_ya[row_idx, col] = sh.y_a
                arr_p[row_idx, col] = sh.P
                arr_yw[row_idx, col] = sh.y_w
                arr_lq[row_idx, col] = float(sh.l_orb)
                mask[row_idx, col] = True
            row_idx += 1

    return {
        "E_th_eV": arr_eth,
        "E0_eV": arr_e0,
        "sigma0_Mb": arr_sigma0,
        "y_a": arr_ya,
        "P": arr_p,
        "y_w": arr_yw,
        "l_q": arr_lq,
        "shell_mask": mask,
    }


def run_checksums(
    arrays: dict[str, np.ndarray],
    data: dict[tuple[int, int], list[ShellRow]],
) -> None:
    """Assert H I 1s shell parameters against known values (1% tolerance).

    H I = Z=1, N=1.  E_th=13.6 eV, sigma0≈5.475e4 Mb, y_a≈32.88, P≈2.963.
    Also verifies all 140 ions have 1..MAX_SHELLS shells, and O VII K-edge ~739 eV.
    """
    hi_shells = data[(1, 1)]
    assert len(hi_shells) == 1, f"Expected 1 shell for H I, got {len(hi_shells)}"
    sh = hi_shells[0]

    def check(param: str, val: float, expected: float, rtol: float = 0.01) -> None:
        err = abs(val - expected) / abs(expected)
        assert err < rtol, (
            f"H I checksum FAILED for {param}: got {val:.6g}, expected {expected:.6g}"
            f" (rel err {err:.3%})"
        )

    check("E_th", sh.E_th, 13.6)
    check("sigma0", sh.sigma0, 5.475e4)
    check("y_a", sh.y_a, 32.88)
    check("P", sh.P, 2.963)
    print(
        f"  H I checksum OK: E_th={sh.E_th:.4g} eV, "
        f"sigma0={sh.sigma0:.4g} Mb, y_a={sh.y_a:.4g}, P={sh.P:.4g}"
    )

    # Verify every ion has >=1 and <=MAX_SHELLS occupied shells.
    shell_mask = arrays["shell_mask"]
    n_shells_per_ion = shell_mask.sum(axis=1)
    assert bool((n_shells_per_ion >= 1).all()), "Some ions have 0 shells"
    assert bool((n_shells_per_ion <= MAX_SHELLS).all()), "Some ions exceed MAX_SHELLS"
    print(
        f"  Shell count range: [{int(n_shells_per_ion.min())}, "
        f"{int(n_shells_per_ion.max())}] across {N_IONS} ions — OK"
    )

    # O VII (Z=8, N=2 → stage=6) → row = H(1)+He(2)+C(6)+N(7)+O_offset(6) = 22.
    o7_row = 22
    o7_eth = arrays["E_th_eV"][o7_row]
    occupied = arrays["shell_mask"][o7_row]
    o7_eth_occupied = o7_eth[occupied]
    assert any(
        abs(float(e) - 739.3) / 739.3 < 0.01 for e in o7_eth_occupied
    ), f"O VII K-edge not near 739 eV; found {o7_eth_occupied}"
    print(f"  O VII (row {o7_row}) E_th occupied: {o7_eth_occupied} eV — OK")


def main() -> None:
    lines = fetch_table()
    print(f"  {len(lines)} lines read")

    data = parse_table(lines)
    print(f"  {len(data)} (Z, N) entries parsed")

    arrays = build_arrays(data)
    run_checksums(arrays, data)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT}: {N_IONS} ions")


if __name__ == "__main__":
    main()
