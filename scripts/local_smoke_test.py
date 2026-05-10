"""Local smoke test for the streaming architecture.

Creates a small synthetic HDF5 file that mimics the SPIn4D Stokes layout,
serves it from a local HTTP server, and verifies that ``inspect_h5`` and
``open_remote_h5`` work end-to-end against it. No external network needed.

If this passes, the package's streaming layer is correctly wired regardless of
whether the production data server happens to be reachable at the moment.
"""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from spin4d_explorer.remote import inspect_h5, open_remote_h5


def make_synthetic_stokes_h5(path: Path) -> None:
    """Write a small h5 file that mimics SPIn4D Stokes layout."""
    rng = np.random.default_rng(seed=0)
    with h5py.File(path, "w") as f:
        for stokes in ("I", "Q", "U", "V"):
            f.create_dataset(
                f"stokes/{stokes}",
                data=rng.standard_normal((64, 64, 21), dtype=np.float32),
                chunks=(32, 32, 21),
                compression="gzip",
            )
        f.create_dataset("wavelength_nm", data=np.linspace(630.0, 630.5, 21))
        f.attrs["instrument"] = "synthetic"
        f.attrs["wavelength_nm_central"] = 630.2


def _start_local_server(directory: Path, port: int) -> socketserver.TCPServer:
    """Start an HTTP server bound to localhost serving ``directory``."""

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *_a, **_kw) -> None:  # silence access log
            return

    server = socketserver.TCPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # tiny delay so the listening socket is definitely up
    time.sleep(0.1)
    return server


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        h5_path = td_path / "synthetic-stokes.h5"
        make_synthetic_stokes_h5(h5_path)
        size_kb = h5_path.stat().st_size / 1024
        print(f"Synthetic file: {h5_path.name} ({size_kb:.1f} KB on disk)\n")

        port = 8765
        server = _start_local_server(td_path, port)
        try:
            url = f"http://127.0.0.1:{port}/{h5_path.name}"
            print(f"Inspecting via HTTP: {url}\n")

            items = inspect_h5(url)
            print(f"Found {len(items)} datasets:")
            for d in items:
                print(
                    f"  {d.path:24s}  shape={d.shape}  dtype={d.dtype}  "
                    f"size={d.size_bytes/1024:.1f}KB  chunks={d.chunks}"
                )

            # Verify we can actually read a slice — exercises h5py + fsspec end-to-end.
            with open_remote_h5(url) as f:
                stokes_i_slice = f["stokes/I"][16:32, 16:32, :]
                wavelengths = f["wavelength_nm"][:]

            print(
                f"\nRead slice from stokes/I: shape={stokes_i_slice.shape}, "
                f"mean={stokes_i_slice.mean():+.4f}, std={stokes_i_slice.std():.4f}"
            )
            print(f"Wavelengths: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm "
                  f"({len(wavelengths)} points)")
            print("\n[OK] Streaming architecture validated end-to-end.")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
