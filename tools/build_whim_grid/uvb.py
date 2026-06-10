"""UVB command used by every CLOUDY run in this pipeline.

Decided by the Task-2 spike (see NOTES.md). Both run_grid.py and
tools/build_uvb_table.py import this so the grid and the JAX validator
see the identical radiation field.
"""

UVB_NAME = "KS19"  # goes into grid metadata


def uvb_command(z: float) -> str:
    """Return the CLOUDY TABLE command for the KS19 UVB at redshift z.

    Uses the default Q=18 model, the fiducial recommended by
    Khaire & Srianand 2019 (MNRAS 484, 4174).
    """
    return f"table KS19 redshift {z:.4f}"
