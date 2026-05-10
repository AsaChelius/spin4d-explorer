"""spin4d-explorer: stream the SPIn4D solar magnetism dataset without local downloads.

Public API
----------
- :func:`open_remote_h5` — open a remote HDF5 file for streaming reads
- :func:`inspect_h5` — return the structure of a remote HDF5 file (metadata only)
- :func:`load_manifest` — load the SPIn4D-DR1 file manifest as a pandas DataFrame
- :func:`file_url` — construct a full URL for a DR1 file given run + filename
- :data:`DEFAULT_BASE_URL` — the IfA Hawaii data root
- :class:`DatasetInfo` — dataclass describing one HDF5 dataset
"""

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
    """CLI entry point — print package version and a brief overview."""
    print(f"spin4d-explorer {__version__}")
    print("Stream the SPIn4D solar magnetism dataset without local downloads.")
    print(f"Data root: {DEFAULT_BASE_URL}")
    print()
    print("Quick start:")
    print("  from spin4d_explorer import inspect_h5, file_url")
    print("  url = file_url('SPIN4D_SSD', 'stokes-031544-6302.h5')")
    print("  for ds in inspect_h5(url):")
    print("      print(ds.path, ds.shape, ds.dtype)")
