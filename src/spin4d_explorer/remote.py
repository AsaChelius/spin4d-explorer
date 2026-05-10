"""Stream SPIn4D HDF5 files over HTTP range requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import fsspec
import h5py
import pandas as pd

DEFAULT_BASE_URL = "http://dtn-itc.ifa.hawaii.edu/spin4d/DR1"
DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    chunks: tuple[int, ...] | None
    compression: str | None


def open_remote_h5(url: str, *, block_size: int = DEFAULT_BLOCK_SIZE) -> h5py.File:
    """Open a remote .h5 file. Slicing pulls only the bytes you ask for."""
    fs = fsspec.filesystem("http")
    fileobj = fs.open(url, "rb", block_size=block_size)
    return h5py.File(fileobj, "r")


def inspect_h5(url: str) -> list[DatasetInfo]:
    """Return one DatasetInfo per dataset in the file. Metadata only, no payload."""
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
    """Load the DR1 manifest CSV (run, step, file_type, is_flipped, file_name)."""
    return pd.read_csv(f"{base_url}/spin4d-dr1-manifest.csv")


def file_url(run: str, file_name: str, base_url: str = DEFAULT_BASE_URL) -> str:
    return f"{base_url}/{run}/{file_name}"
