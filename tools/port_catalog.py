"""Port the legacy WHIM-paper Swift-XRT catalog (Swift_GRB_WHIM_cut_v2.txt)
to a packaged CSV with explicit headers.

Run once:
    uv run python tools/port_catalog.py

Produces whimstan/data/grb_sample_v2.csv (committed to git).

Source-of-truth columns (12) per plan:
    name, obsid, ra_deg, dec_deg, l_gal, b_gal,
    nh_mw, z, t_start, t_stop, exposure, counts

Legacy file layout (12 whitespace-separated fields, no header):
    name obsid ra_deg dec_deg b_gal nh_mw z <unknown> t_start t_stop exposure counts

The legacy file lacks `l_gal`; we compute it from (ra, dec) via astropy.
The 8th column (unknown quantity, possibly an unused legacy field) is discarded.
"""

from __future__ import annotations

import re
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

SRC = Path("/Users/jburgess/articles/whim_paper/src/figures/Swift_GRB_WHIM_cut_v2.txt")
DST = Path("whimstan/data/grb_sample_v2.csv")


def main() -> None:
    if not SRC.is_file():
        raise FileNotFoundError(SRC)

    out_lines = [
        "name,obsid,ra_deg,dec_deg,l_gal,b_gal,nh_mw,z,t_start,t_stop,exposure,counts\n"
    ]
    n_kept = 0
    for raw in SRC.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = re.split(r"\s+", line)
        if len(fields) < 12:
            print(f"skipping short row ({len(fields)} fields): {raw!r}")
            continue
        name = fields[0]
        obsid = fields[1]
        ra_deg = float(fields[2])
        dec_deg = float(fields[3])
        b_gal = float(fields[4])
        nh_mw = float(fields[5])
        z = float(fields[6])
        # fields[7] is an unused legacy column - discarded.
        t_start = float(fields[8])
        t_stop = float(fields[9])
        exposure = float(fields[10])
        counts = int(float(fields[11]))

        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        l_gal = float(coord.galactic.l.deg)

        out_lines.append(
            f"{name},{obsid},{ra_deg},{dec_deg},{l_gal:.6f},{b_gal},"
            f"{nh_mw},{z},{t_start},{t_stop},{exposure},{counts}\n"
        )
        n_kept += 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text("".join(out_lines))
    print(f"Wrote {DST} with {n_kept} GRBs")


if __name__ == "__main__":
    main()
