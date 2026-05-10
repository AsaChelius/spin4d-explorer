"""High-level convenience API for the SPIn4D-DR1 dataset.

Hides the file-naming conventions documented at
``http://dtn-itc.ifa.hawaii.edu/spin4d/DR1/`` so callers can ask for data by
physical concept (run + timestep + variable / wavelength) rather than by
constructing URLs by hand.

Examples
--------
>>> from spin4d_explorer import cube, stokes
>>> bx = cube("SPIN4D_SSD", "031544", "Bx").read_slab(0, 32)
>>> with stokes("SPIN4D_SSD", "031544", wavelength=6302) as f:
...     stokes_i = f["stokes/I"][16:32, 16:32, :]
"""

from __future__ import annotations

import h5py
import pandas as pd

from spin4d_explorer.npy import RemoteNpy, open_remote_npy
from spin4d_explorer.remote import DEFAULT_BASE_URL, load_manifest, open_remote_h5

# MURaM equation-of-state variables, mapping physical name → subdomain index.
# Source: http://dtn-itc.ifa.hawaii.edu/spin4d/DR1/ (variable key table).
MURAM_VARIABLES: dict[str, int] = {
    "rho": 0,      # density
    "vx": 1,       # velocity_x
    "vy": 2,       # velocity_y
    "vz": 3,       # velocity_z
    "eint": 4,     # internal electron pressure
    "Bx": 5,       # magnetic field_x
    "By": 6,       # magnetic field_y
    "Bz": 7,       # magnetic field_z
    "T": 8,        # temperature
    "P": 9,        # pressure
    "ne": 10,      # number of electrons
    "tau500": 11,  # opacity at 500 nm
}

# Stokes profile wavelengths available in DR1 (Fe I lines, integer nm * 10).
STOKES_WAVELENGTHS: tuple[int, ...] = (6302, 15648)


def cube(
    run: str,
    step: str | int,
    var: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> RemoteNpy:
    """Open a MURaM cube for the given case, timestep, and physical variable.

    Parameters
    ----------
    run : str
        Simulation case name, e.g. ``"SPIN4D_SSD"`` or ``"SPIN4D_SSD_100G"``.
    step : str or int
        Timestep, e.g. ``"031544"`` or ``31544``. Integers are zero-padded.
    var : str
        Physical variable name. One of :data:`MURAM_VARIABLES` keys
        (``"rho"``, ``"vx"``, ..., ``"Bz"``, ``"T"``, ``"tau500"``).
    base_url : str, optional
        Override the data root.

    Returns
    -------
    RemoteNpy
        Streaming handle to the corresponding ``subdomain_<i>.<step>`` file.
    """
    if var not in MURAM_VARIABLES:
        raise ValueError(
            f"unknown MURaM variable {var!r}; expected one of "
            f"{sorted(MURAM_VARIABLES)}"
        )
    step_str = _format_step(step)
    file_name = f"subdomain_{MURAM_VARIABLES[var]}.{step_str}"
    url = f"{base_url}/{run}/{file_name}"
    return open_remote_npy(url)


def stokes(
    run: str,
    step: str | int,
    *,
    wavelength: int,
    base_url: str = DEFAULT_BASE_URL,
) -> h5py.File:
    """Open the Stokes profile file for a case, timestep, and Fe I wavelength.

    Parameters
    ----------
    run : str
        Simulation case name.
    step : str or int
        Timestep, zero-padded to 6 digits if int.
    wavelength : int
        Fe I wavelength code: ``6302`` (630.2 nm) or ``15648`` (1564.8 nm).
    base_url : str, optional
        Override the data root.

    Returns
    -------
    h5py.File
        Open HDF5 file handle backed by an HTTP-streaming file object.
    """
    if wavelength not in STOKES_WAVELENGTHS:
        raise ValueError(
            f"wavelength must be one of {STOKES_WAVELENGTHS}, got {wavelength}"
        )
    step_str = _format_step(step)
    file_name = f"stokes-{step_str}-{wavelength}.h5"
    url = f"{base_url}/{run}/{file_name}"
    return open_remote_h5(url)


def list_runs(*, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """List the simulation case names available in DR1."""
    df = load_manifest(base_url=base_url)
    return sorted(df["run"].unique().tolist())


def list_steps(run: str, *, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """List the available timesteps for a given simulation case (zero-padded)."""
    df = load_manifest(base_url=base_url)
    steps = df.loc[df["run"] == run, "step"]
    return sorted({_format_step(s) for s in steps})


def list_variables() -> list[str]:
    """List the supported MURaM variable names."""
    return list(MURAM_VARIABLES)


def list_wavelengths() -> list[int]:
    """List the available Stokes profile wavelength codes."""
    return list(STOKES_WAVELENGTHS)


def files_for(
    run: str,
    step: str | int,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Return the manifest rows for one (run, step) — one snapshot's files.

    Useful for inspection / debugging when you want to know exactly which
    files exist for a given snapshot before opening any of them.
    """
    step_str = _format_step(step)
    df = load_manifest(base_url=base_url)
    mask = (df["run"] == run) & (df["step"].astype(str).str.zfill(6) == step_str)
    return df.loc[mask].reset_index(drop=True)


def _format_step(step: str | int) -> str:
    """Coerce a timestep to the canonical zero-padded 6-digit string."""
    if isinstance(step, int):
        return f"{step:06d}"
    s = str(step).strip()
    # Already padded? leave alone. Otherwise zero-pad to 6.
    return s.zfill(6) if s.isdigit() else s
