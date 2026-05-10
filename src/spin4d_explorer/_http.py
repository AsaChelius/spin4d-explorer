"""Minimal sync HTTP file-like object backed by Range requests."""

from __future__ import annotations

import urllib.request
from typing import Self

DEFAULT_BLOCK_SIZE = 4 * 1024 * 1024


class RangeFile:
    """File-like wrapper around an HTTP URL using Range requests.

    seek/tell/read in bytes, no whole-file download. Reads are buffered in
    one block of `block_size` bytes; consecutive small reads in the same
    region hit the buffer instead of the network.
    """

    def __init__(self, url: str, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.url = url
        self.block_size = block_size
        self._pos = 0
        self._size = self._fetch_size()
        self._buf = b""
        self._buf_start = -1

    def _fetch_size(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            if cl is None:
                raise OSError(f"server did not return Content-Length for {self.url}")
            ar = r.headers.get("Accept-Ranges", "").lower()
            if ar != "bytes":
                # not fatal — many servers omit the header but still honor Range
                pass
            return int(cl)

    @property
    def size(self) -> int:
        return self._size

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        else:
            raise ValueError(f"bad whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""

        buf_end = self._buf_start + len(self._buf)
        if 0 <= self._buf_start <= self._pos and self._pos + n <= buf_end:
            off = self._pos - self._buf_start
            data = self._buf[off : off + n]
            self._pos += len(data)
            return data

        fetch_n = max(n, self.block_size)
        end = min(self._pos + fetch_n - 1, self._size - 1)
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            self._buf = r.read()
        self._buf_start = self._pos

        data = self._buf[:n]
        self._pos += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def close(self) -> None:
        self._buf = b""
        self._buf_start = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
