"""GRB catalog loader.

The packaged CSV `grb_sample_v2.csv` is the legacy WHIM-paper sample
(~205 GRBs). New samples can be loaded with `load_catalog(path)`.
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
    nh_mw: float  # cm^-2
    z: float
    t_start: float
    t_stop: float
    exposure: float  # seconds
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
        return not (abs_b_min is not None and abs(e.b_gal) < abs_b_min)

    return [e for e in cat if keep(e)]
