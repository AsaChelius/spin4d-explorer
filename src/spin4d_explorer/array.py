"""Read a remote raw-binary array (no header) over HTTP range requests.

The SPIn4D MURaM cubes are stored as plain np.float32 dumped via np.tofile,
not as .npy. Caller supplies shape + dtype; we range-request the bytes.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from spin4d_explorer._http import RangeFile


class RemoteArray:
    """A raw binary array on an HTTP server. Headerless; caller supplies the layout."""

    def __init__(
        self,
        url: str,
        shape: tuple[int, ...],
        dtype: np.dtype | str,
        *,
        fortran_order: bool = False,
        offset: int = 0,
    ) -> None:
        self._url = url
        self._fileobj = RangeFile(url)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.fortran_order = fortran_order
        self.offset = offset

    @property
    def url(self) -> str:
        return self._url

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    def read(self) -> np.ndarray:
        """Pull the whole array."""
        self._fileobj.seek(self.offset)
        raw = self._fileobj.read(self.nbytes)
        order = "F" if self.fortran_order else "C"
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape, order=order)

    def read_slab(self, start: int, stop: int) -> np.ndarray:
        """arr[start:stop, ...] for C-order, arr[..., start:stop] for F-order.

        One contiguous chunk of bytes on disk, one HTTP request.
        """
        slab_axis = -1 if self.fortran_order else 0
        slab_dim = self.shape[slab_axis]
        if not (0 <= start < stop <= slab_dim):
            raise IndexError(
                f"slab [{start}:{stop}] out of bounds for axis size {slab_dim}"
            )

        other_dims = self.shape[:-1] if self.fortran_order else self.shape[1:]
        elements_per_step = 1
        for d in other_dims:
            elements_per_step *= d
        bytes_per_step = elements_per_step * self.dtype.itemsize

        self._fileobj.seek(self.offset + start * bytes_per_step)
        raw = self._fileobj.read((stop - start) * bytes_per_step)

        result_shape = list(self.shape)
        result_shape[slab_axis] = stop - start
        order = "F" if self.fortran_order else "C"
        return np.frombuffer(raw, dtype=self.dtype).reshape(
            tuple(result_shape), order=order
        )

    def close(self) -> None:
        self._fileobj.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def open_remote_array(
    url: str,
    shape: tuple[int, ...],
    dtype: np.dtype | str,
    *,
    fortran_order: bool = False,
    offset: int = 0,
) -> RemoteArray:
    return RemoteArray(
        url, shape, dtype, fortran_order=fortran_order, offset=offset
    )
