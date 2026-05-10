"""Ask for data by run + step + variable, get a remote handle. No URL juggling."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

from spin4d_explorer.array import RemoteArray
from spin4d_explorer.remote import DEFAULT_BASE_URL, load_manifest, open_remote_h5

# variable -> subdomain index, per http://dtn-itc.ifa.hawaii.edu/spin4d/DR1/
MURAM_VARIABLES: dict[str, int] = {
    "rho": 0,
    "vx": 1,
    "vy": 2,
    "vz": 3,
    "eint": 4,
    "Bx": 5,
    "By": 6,
    "Bz": 7,
    "T": 8,
    "P": 9,
    "ne": 10,
    "tau500": 11,
}

# Per-case MURaM cube shapes (raw float32). The 25x25x8 Mm runs are
# (1536, 1536, 128); the Large run (50x50x8 Mm) is 4x in horizontal.
# Source: spin4d-data notebook does
#   np.fromfile(..., dtype=np.float32).reshape((1536, 1536, 128))
MURAM_SHAPES: dict[str, tuple[int, int, int]] = {
    "SPIN4D_SSD":       (1536, 1536, 128),
    "SPIN4D_SSD_100G":  (1536, 1536, 128),
    "SPIN4D_SSD_200G":  (1536, 1536, 128),
    "SPIN4D_SSD_50G":   (1536, 1536, 128),
    "SPIN4D_SSD_50G_V": (1536, 1536, 128),
    "SPIN4D_SSD_Large": (3072, 3072, 128),
}

STOKES_WAVELENGTHS: tuple[int, ...] = (6302, 15648)


def cube(
    run: str,
    step: str | int,
    var: str,
    *,
    shape: tuple[int, ...] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> RemoteArray:
    """Open a MURaM cube by physical variable name. Pass shape= to override the per-run lookup."""
    if var not in MURAM_VARIABLES:
        raise ValueError(
            f"unknown variable {var!r}; pick one of {sorted(MURAM_VARIABLES)}"
        )
    if shape is None:
        if run not in MURAM_SHAPES:
            raise ValueError(
                f"unknown run {run!r}; pass shape= or use one of {sorted(MURAM_SHAPES)}"
            )
        shape = MURAM_SHAPES[run]
    file_name = f"subdomain_{MURAM_VARIABLES[var]}.{_format_step(step)}"
    url = f"{base_url}/{run}/{file_name}"
    return RemoteArray(url, shape=shape, dtype=np.float32)


def stokes(
    run: str,
    step: str | int,
    *,
    wavelength: int,
    base_url: str = DEFAULT_BASE_URL,
) -> h5py.File:
    """Open the Stokes profile file. Wavelength is 6302 (630.2 nm) or 15648 (1564.8 nm)."""
    if wavelength not in STOKES_WAVELENGTHS:
        raise ValueError(f"wavelength must be one of {STOKES_WAVELENGTHS}")
    file_name = f"stokes-{_format_step(step)}-{wavelength}.h5"
    return open_remote_h5(f"{base_url}/{run}/{file_name}")


def list_runs(*, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    return sorted(load_manifest(base_url=base_url)["run"].unique().tolist())


def list_steps(run: str, *, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    df = load_manifest(base_url=base_url)
    return sorted({_format_step(s) for s in df.loc[df["run"] == run, "step"]})


def list_variables() -> list[str]:
    return list(MURAM_VARIABLES)


def list_wavelengths() -> list[int]:
    return list(STOKES_WAVELENGTHS)


def files_for(
    run: str, step: str | int, *, base_url: str = DEFAULT_BASE_URL
) -> pd.DataFrame:
    """Manifest rows for one (run, step) snapshot."""
    s = _format_step(step)
    df = load_manifest(base_url=base_url)
    mask = (df["run"] == run) & (df["step"].astype(str).str.zfill(6) == s)
    return df.loc[mask].reset_index(drop=True)


def _format_step(step: str | int) -> str:
    if isinstance(step, int):
        return f"{step:06d}"
    s = str(step).strip()
    return s.zfill(6) if s.isdigit() else s
