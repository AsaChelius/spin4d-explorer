"""Remote access to SPIn4D-DR1 datasets without local downloads.

Streams HDF5 (Stokes profile) files from the IfA Hawaii data server via HTTP
range requests, so the 13.7 TB DR1 release can be inspected and partially read
without downloading whole 9 GB files.

The data server lives at ``http://dtn-itc.ifa.hawaii.edu/spin4d/DR1`` and
supports ``Accept-Ranges: bytes``, which is what makes streaming possible.

Examples
--------
>>> from spin4d_explorer.remote import inspect_h5, file_url
>>> url = file_url("SPIN4D_SSD", "stokes-031544-6302.h5")
>>> for ds in inspect_h5(url):
...     print(ds.path, ds.shape, ds.dtype)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fsspec
import h5py
import pandas as pd

DEFAULT_BASE_URL = "http://dtn-itc.ifa.hawaii.edu/spin4d/DR1"
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024  # 4 MB read-ahead


@dataclass(frozen=True)
class DatasetInfo:
    """Summary of one dataset inside an HDF5 file."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    chunks: tuple[int, ...] | None
    compression: str | None


def open_remote_h5(
    url: str,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> h5py.File:
    """Open a remote HDF5 file for streaming reads via HTTP range requests.

    The returned ``h5py.File`` behaves like a normal local file: indexing and
    slicing pull only the bytes needed (rounded up to ``block_size`` chunks),
    so you can read a single 2-D slice out of a 9 GB cube without downloading
    the whole file.

    Parameters
    ----------
    url : str
        Full HTTP(S) URL to the .h5 file.
    block_size : int, optional
        Read-ahead chunk size in bytes for the underlying fsspec HTTP file.
        Larger values reduce request count for contiguous reads; smaller
        values reduce wasted bandwidth for sparse access. Defaults to 4 MiB.

    Returns
    -------
    h5py.File
        Open file handle. Use as a context manager or call ``.close()``.
    """
    fs = fsspec.filesystem("http")
    fileobj = fs.open(url, "rb", block_size=block_size)
    return h5py.File(fileobj, "r")


def inspect_h5(url: str) -> list[DatasetInfo]:
    """Return a structural summary of every dataset in a remote HDF5 file.

    This reads only metadata (file header, group structure, dataset
    descriptors) — typically a few hundred KB regardless of the file's actual
    size — so it's safe to call against multi-gigabyte cubes.

    Parameters
    ----------
    url : str
        Full HTTP(S) URL to the .h5 file.

    Returns
    -------
    list of DatasetInfo
        One entry per dataset, in HDF5 traversal order.
    """
    items: list[DatasetInfo] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            items.append(
                DatasetInfo(
                    path=name,
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    size_bytes=int(obj.nbytes),
                    chunks=tuple(obj.chunks) if obj.chunks else None,
                    compression=obj.compression,
                )
            )

    with open_remote_h5(url) as f:
        f.visititems(visit)

    return items


def load_manifest(base_url: str = DEFAULT_BASE_URL) -> pd.DataFrame:
    """Load the SPIn4D-DR1 file manifest as a DataFrame.

    The manifest enumerates every file in DR1 with columns
    ``run, step, file_type, is_flipped, file_name``.

    Parameters
    ----------
    base_url : str, optional
        Override the data root if the dataset is mirrored elsewhere.

    Returns
    -------
    pandas.DataFrame
    """
    url = f"{base_url}/spin4d-dr1-manifest.csv"
    return pd.read_csv(url)


def file_url(run: str, file_name: str, base_url: str = DEFAULT_BASE_URL) -> str:
    """Construct the full URL for a single DR1 file.

    Parameters
    ----------
    run : str
        Simulation case name, e.g. ``"SPIN4D_SSD"`` or ``"SPIN4D_SSD_100G"``.
    file_name : str
        File name as it appears in the manifest, e.g.
        ``"stokes-031544-6302.h5"`` or ``"subdomain_5.031544"``.
    base_url : str, optional
        Override the data root.

    Returns
    -------
    str
        Full URL suitable for ``open_remote_h5`` or any HTTP client.
    """
    return f"{base_url}/{run}/{file_name}"
