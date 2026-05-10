"""Stream SPIn4D MURaM .npy cubes over HTTP range requests.

The .npy format is a tiny header (shape, dtype, order) followed by raw bytes,
so once you know the header you can compute the byte offset for any contiguous
slab and pull it with a single range request.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Self

import fsspec
import numpy as np
from numpy.lib import format as npyfmt

DEFAULT_HEADER_PROBE = 8192


@dataclass(frozen=True)
class NpyHeader:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    data_offset: int


_HEADER_READERS = {
    (1, 0): npyfmt.read_array_header_1_0,
    (2, 0): npyfmt.read_array_header_2_0,
}
if hasattr(npyfmt, "read_array_header_3_0"):
    _HEADER_READERS[(3, 0)] = npyfmt.read_array_header_3_0


def _parse_npy_header(header_bytes: bytes) -> NpyHeader:
    f = io.BytesIO(header_bytes)
    version = npyfmt.read_magic(f)
    try:
        reader = _HEADER_READERS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported .npy version {version}") from exc
    shape, fortran_order, dtype = reader(f)
    return NpyHeader(tuple(shape), dtype, fortran_order, f.tell())


class RemoteNpy:
    """A remote .npy file. Read the whole thing or pull a slab via range request."""

    def __init__(
        self, url: str, *, header_probe_bytes: int = DEFAULT_HEADER_PROBE
    ) -> None:
        self._url = url
        self._fs = fsspec.filesystem("http")
        self._fileobj = self._fs.open(url, "rb")
        try:
            self._header = _parse_npy_header(self._fileobj.read(header_probe_bytes))
        except ValueError:
            # header was longer than our probe; retry bigger
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
        """Pull the whole array."""
        self._fileobj.seek(self._header.data_offset)
        raw = self._fileobj.read(self.nbytes)
        order = "F" if self._header.fortran_order else "C"
        return np.frombuffer(raw, dtype=self._header.dtype).reshape(
            self._header.shape, order=order
        )

    def read_slab(self, start: int, stop: int) -> np.ndarray:
        """arr[start:stop, ...] for C-order, arr[..., start:stop] for F-order.

        Either way it's one contiguous chunk of bytes on disk, so one HTTP
        request gets it.
        """
        slab_axis = -1 if self._header.fortran_order else 0
        slab_dim = self._header.shape[slab_axis]
        if not (0 <= start < stop <= slab_dim):
            raise IndexError(
                f"slab [{start}:{stop}] out of bounds for axis size {slab_dim}"
            )

        other_dims = (
            self._header.shape[:-1]
            if self._header.fortran_order
            else self._header.shape[1:]
        )
        elements_per_step = 1
        for d in other_dims:
            elements_per_step *= d
        bytes_per_step = elements_per_step * self.dtype.itemsize

        self._fileobj.seek(self._header.data_offset + start * bytes_per_step)
        raw = self._fileobj.read((stop - start) * bytes_per_step)

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
    return RemoteNpy(url, header_probe_bytes=header_probe_bytes)
