"""Remote NumPy ``.npy`` file access via HTTP range requests.

The SPIn4D-DR1 MURaM simulation cubes are stored as raw ``.npy`` files (one
state variable per file, e.g. ``subdomain_5.<timestep>`` for ``B_x``). The
files are large (multi-GB) but the on-disk layout is simple: a small text
header followed by a contiguous block of float bytes, in either C or
Fortran order.

This module exposes :class:`RemoteNpy`, a thin wrapper that:

1. Range-requests the first few KB to read the header (``shape``, ``dtype``,
   memory order, data offset).
2. Lets the caller read the full array (``.read()``) or a contiguous slab
   along the slowest-varying axis (``.read_slab(start, stop)``) using a
   single targeted range request — no local download of the whole file.

For arbitrary fancy slicing, fall back to :meth:`RemoteNpy.read` and slice
in memory. Future work: arbitrary ``__getitem__`` that issues the minimum
set of range requests for the access pattern.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Self

import fsspec
import numpy as np
from numpy.lib import format as npyfmt

DEFAULT_HEADER_PROBE = 8192  # bytes; .npy headers are typically < 1 KB


@dataclass(frozen=True)
class NpyHeader:
    """Parsed metadata from a ``.npy`` file header."""

    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    data_offset: int  # byte offset where the raw array data begins


_HEADER_READERS = {
    (1, 0): npyfmt.read_array_header_1_0,
    (2, 0): npyfmt.read_array_header_2_0,
}
# numpy>=1.23 added 3.0
if hasattr(npyfmt, "read_array_header_3_0"):
    _HEADER_READERS[(3, 0)] = npyfmt.read_array_header_3_0


def _parse_npy_header(header_bytes: bytes) -> NpyHeader:
    """Parse a ``.npy`` header from raw bytes.

    Raises ``ValueError`` if ``header_bytes`` is too short to contain the
    full header — the caller should retry with a larger probe.
    """
    f = io.BytesIO(header_bytes)
    version = npyfmt.read_magic(f)
    try:
        reader = _HEADER_READERS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported .npy format version {version}") from exc
    shape, fortran_order, dtype = reader(f)
    return NpyHeader(
        shape=tuple(shape),
        dtype=dtype,
        fortran_order=fortran_order,
        data_offset=f.tell(),
    )


class RemoteNpy:
    """A remote ``.npy`` file accessed via HTTP range requests.

    The header is read at construction (~8 KB by default). Subsequent reads
    issue range requests for only the bytes needed.

    Parameters
    ----------
    url : str
        Full HTTP(S) URL to the ``.npy`` file.
    header_probe_bytes : int, optional
        Initial number of bytes to fetch when reading the header. The
        constructor automatically retries with a larger probe if the
        header is longer than this. Default 8192.
    """

    def __init__(
        self, url: str, *, header_probe_bytes: int = DEFAULT_HEADER_PROBE
    ) -> None:
        self._url = url
        self._fs = fsspec.filesystem("http")
        self._fileobj = self._fs.open(url, "rb")
        try:
            self._header = _parse_npy_header(self._fileobj.read(header_probe_bytes))
        except ValueError:
            # Header didn't fit — re-read with a much larger probe.
            self._fileobj.seek(0)
            self._header = _parse_npy_header(
                self._fileobj.read(header_probe_bytes * 16)
            )

    @property
    def url(self) -> str:
        return self._url

    @property
    def shape(self) -> tuple[int, ...]:
        return self._header.shape

    @property
    def dtype(self) -> np.dtype:
        return self._header.dtype

    @property
    def fortran_order(self) -> bool:
        return self._header.fortran_order

    @property
    def ndim(self) -> int:
        return len(self._header.shape)

    @property
    def size(self) -> int:
        n = 1
        for d in self._header.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def data_offset(self) -> int:
        return self._header.data_offset

    def read(self) -> np.ndarray:
        """Read and return the entire array.

        Downloads :attr:`nbytes` bytes — fine for small files, expensive for
        multi-GB cubes. Prefer :meth:`read_slab` when you only need part of
        the volume.
        """
        self._fileobj.seek(self._header.data_offset)
        raw = self._fileobj.read(self.nbytes)
        order = "F" if self._header.fortran_order else "C"
        return np.frombuffer(raw, dtype=self._header.dtype).reshape(
            self._header.shape, order=order
        )

    def read_slab(self, start: int, stop: int) -> np.ndarray:
        """Read a contiguous slab along the slowest-varying axis.

        For C-order arrays this is ``arr[start:stop, ...]`` (slowest axis is 0).
        For Fortran-order arrays this is ``arr[..., start:stop]`` (slowest is -1).
        Either way, the result lives in one contiguous range of bytes on disk,
        which we fetch with a single HTTP range request.

        Parameters
        ----------
        start, stop : int
            Half-open interval along the slab axis.

        Returns
        -------
        numpy.ndarray
            View-shaped slab of the original array.
        """
        slab_axis = -1 if self._header.fortran_order else 0
        slab_dim = self._header.shape[slab_axis]
        if not (0 <= start < stop <= slab_dim):
            raise IndexError(
                f"slab [{start}:{stop}] out of bounds for axis size {slab_dim}"
            )

        # Number of elements per unit step along the slab axis.
        other_dims = (
            self._header.shape[:-1]
            if self._header.fortran_order
            else self._header.shape[1:]
        )
        elements_per_step = 1
        for d in other_dims:
            elements_per_step *= d
        bytes_per_step = elements_per_step * self.dtype.itemsize

        byte_offset = self._header.data_offset + start * bytes_per_step
        nbytes = (stop - start) * bytes_per_step

        self._fileobj.seek(byte_offset)
        raw = self._fileobj.read(nbytes)

        result_shape = list(self._header.shape)
        result_shape[slab_axis] = stop - start
        order = "F" if self._header.fortran_order else "C"
        return np.frombuffer(raw, dtype=self._header.dtype).reshape(
            tuple(result_shape), order=order
        )

    def close(self) -> None:
        self._fileobj.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_remote_npy(
    url: str, *, header_probe_bytes: int = DEFAULT_HEADER_PROBE
) -> RemoteNpy:
    """Open a remote ``.npy`` file for streaming reads.

    Parameters
    ----------
    url : str
        Full HTTP(S) URL to the ``.npy`` file.
    header_probe_bytes : int, optional
        Initial probe size for the header. Default 8192.

    Returns
    -------
    RemoteNpy
    """
    return RemoteNpy(url, header_probe_bytes=header_probe_bytes)
