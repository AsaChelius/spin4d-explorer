"""Local .npy smoke test. Both C-order and F-order, slab + full read."""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from spin4d_explorer.npy import open_remote_npy


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
    # MURaM-flavoured: 3D float32 cube with shape (z, y, x).
    arr_c = rng.standard_normal((48, 32, 64), dtype=np.float32)
    arr_f = np.asfortranarray(arr_c.copy())

    from numpy.lib.format import write_array

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Real SPIn4D MURaM files have no extension (e.g. subdomain_5.031544),
        # so use write_array directly to avoid np.save auto-appending ".npy".
        c_path = td_path / "subdomain_5.000123"  # c-order
        f_path = td_path / "subdomain_5.000456"  # fortran-order
        with open(c_path, "wb") as fh:
            write_array(fh, arr_c, allow_pickle=False)
        with open(f_path, "wb") as fh:
            write_array(fh, arr_f, allow_pickle=False)

        port = 8767
        server = _start_local_server(td_path, port)
        try:
            for label, path, ref in (("C-order", c_path, arr_c), ("F-order", f_path, arr_f)):
                url = f"http://127.0.0.1:{port}/{path.name}"
                print(f"--- {label} ({path.name}) ---")
                with open_remote_npy(url) as r:
                    print(
                        f"  shape={r.shape}  dtype={r.dtype}  "
                        f"fortran_order={r.fortran_order}  nbytes={r.nbytes}  "
                        f"data_offset={r.data_offset}"
                    )

                    full = r.read()
                    assert full.shape == ref.shape, (full.shape, ref.shape)
                    assert full.dtype == ref.dtype
                    assert np.array_equal(full, ref), "full read mismatch"
                    print(f"  full read OK  ({full.nbytes} bytes)")

                    slab = r.read_slab(10, 25)
                    expected = (
                        ref[..., 10:25] if r.fortran_order else ref[10:25, ...]
                    )
                    assert slab.shape == expected.shape, (slab.shape, expected.shape)
                    assert np.array_equal(slab, expected), "slab mismatch"
                    print(
                        f"  slab read OK  shape={slab.shape}  "
                        f"({slab.nbytes} bytes streamed)"
                    )
                print()

            print("[OK] RemoteNpy validated for both C-order and Fortran-order.")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
