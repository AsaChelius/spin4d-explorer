"""Local raw-binary array smoke test. C-order and F-order, full + slab."""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from spin4d_explorer.array import open_remote_array


def _start_local_server(directory: Path, port: int) -> socketserver.TCPServer:
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *_a, **_kw) -> None:
            return

    server = socketserver.TCPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.1)
    return server


def main() -> None:
    rng = np.random.default_rng(seed=42)
    arr_c = rng.standard_normal((48, 32, 64), dtype=np.float32)
    arr_f = np.asfortranarray(arr_c.copy())

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        c_path = td_path / "subdomain_5.000123"
        f_path = td_path / "subdomain_5.000456"
        # np.tofile always writes C-order bytes regardless of memory layout,
        # so for the F-order case write tobytes(order='F') explicitly.
        arr_c.tofile(c_path)
        with open(f_path, "wb") as fh:
            fh.write(arr_c.tobytes(order="F"))

        port = 8767
        server = _start_local_server(td_path, port)
        try:
            cases = [
                ("C-order", c_path, arr_c, False),
                ("F-order", f_path, arr_f, True),
            ]
            for label, path, ref, fortran in cases:
                url = f"http://127.0.0.1:{port}/{path.name}"
                print(f"--- {label} ({path.name}) ---")
                with open_remote_array(
                    url, shape=ref.shape, dtype=np.float32, fortran_order=fortran
                ) as r:
                    print(
                        f"  shape={r.shape}  dtype={r.dtype}  "
                        f"fortran_order={r.fortran_order}  nbytes={r.nbytes}"
                    )

                    full = r.read()
                    assert full.shape == ref.shape, (full.shape, ref.shape)
                    assert np.array_equal(full, ref), "full read mismatch"
                    print(f"  full read OK  ({full.nbytes} bytes)")

                    slab = r.read_slab(10, 25)
                    expected = ref[..., 10:25] if fortran else ref[10:25, ...]
                    assert slab.shape == expected.shape, (slab.shape, expected.shape)
                    assert np.array_equal(slab, expected), "slab mismatch"
                    print(
                        f"  slab read OK  shape={slab.shape}  "
                        f"({slab.nbytes} bytes streamed)"
                    )
                print()

            print("[OK] RemoteArray validated for both C-order and Fortran-order.")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
