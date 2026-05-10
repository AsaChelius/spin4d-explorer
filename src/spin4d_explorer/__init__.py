"""Stream the SPIn4D solar magnetism dataset over HTTP."""

from __future__ import annotations

from spin4d_explorer.dataset import (
    MURAM_VARIABLES,
    STOKES_WAVELENGTHS,
    cube,
    files_for,
    list_runs,
    list_steps,
    list_variables,
    list_wavelengths,
    stokes,
)
from spin4d_explorer.npy import NpyHeader, RemoteNpy, open_remote_npy
from spin4d_explorer.remote import (
    DEFAULT_BASE_URL,
    DatasetInfo,
    file_url,
    inspect_h5,
    load_manifest,
    open_remote_h5,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_BASE_URL",
    "DatasetInfo",
    "MURAM_VARIABLES",
    "NpyHeader",
    "RemoteNpy",
    "STOKES_WAVELENGTHS",
    "__version__",
    "cube",
    "file_url",
    "files_for",
    "inspect_h5",
    "list_runs",
    "list_steps",
    "list_variables",
    "list_wavelengths",
    "load_manifest",
    "main",
    "open_remote_h5",
    "open_remote_npy",
    "stokes",
]


def main() -> None:
    print(f"spin4d-explorer {__version__}")
    print(f"data root: {DEFAULT_BASE_URL}")
