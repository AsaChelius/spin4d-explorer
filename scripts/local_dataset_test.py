"""End-to-end test of cube/stokes/list_runs against a synthetic DR1 tree."""

from __future__ import annotations

import http.server
import socketserver
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from numpy.lib.format import write_array

from spin4d_explorer.dataset import (
    MURAM_VARIABLES,
    cube,
    files_for,
    list_runs,
    list_steps,
    stokes,
)


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


def _build_synthetic_dr1(root: Path) -> None:
    """Write a tiny SPIn4D-shaped tree with one run and two timesteps."""
    rng = np.random.default_rng(0)
    run = "SPIN4D_SSD"
    steps = ["031544", "032100"]
    wavelengths = [6302, 15648]

    (root / run).mkdir(parents=True)

    manifest_rows: list[dict] = []
    for step in steps:
        # 12 MURaM cube files per snapshot
        for var, idx in MURAM_VARIABLES.items():
            arr = rng.standard_normal((16, 16, 8), dtype=np.float32)
            path = root / run / f"subdomain_{idx}.{step}"
            with open(path, "wb") as fh:
                write_array(fh, arr, allow_pickle=False)
            manifest_rows.append(
                {"run": run, "step": step, "file_type": "MURaM",
                 "is_flipped": "-", "file_name": path.name}
            )
        # 2 Stokes h5 files per snapshot
        for wl in wavelengths:
            path = root / run / f"stokes-{step}-{wl}.h5"
            with h5py.File(path, "w") as f:
                for s in ("I", "Q", "U", "V"):
                    f.create_dataset(
                        f"stokes/{s}",
                        data=rng.standard_normal((16, 16, 11), dtype=np.float32),
                        chunks=(8, 8, 11),
                    )
            manifest_rows.append(
                {"run": run, "step": step, "file_type": "SIR",
                 "is_flipped": "Y", "file_name": path.name}
            )

    pd.DataFrame(manifest_rows).to_csv(root / "spin4d-dr1-manifest.csv", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _build_synthetic_dr1(td_path)
        print(f"Synthetic DR1 built at {td_path}")
        print(f"  manifest: {(td_path / 'spin4d-dr1-manifest.csv').stat().st_size} bytes")

        port = 8768
        server = _start_local_server(td_path, port)
        try:
            base_url = f"http://127.0.0.1:{port}"

            # 1. list_runs / list_steps via manifest
            runs = list_runs(base_url=base_url)
            print(f"\nlist_runs(): {runs}")
            assert runs == ["SPIN4D_SSD"]

            steps = list_steps("SPIN4D_SSD", base_url=base_url)
            print(f"list_steps('SPIN4D_SSD'): {steps}")
            assert steps == ["031544", "032100"]

            # 2. files_for one snapshot
            snap = files_for("SPIN4D_SSD", "031544", base_url=base_url)
            print(f"files_for('SPIN4D_SSD', '031544'): {len(snap)} files "
                  f"({(snap['file_type'] == 'MURaM').sum()} MURaM, "
                  f"{(snap['file_type'] == 'SIR').sum()} Stokes)")
            assert len(snap) == 14  # 12 MURaM + 2 Stokes

            # 3. cube() — by physical variable name
            with cube("SPIN4D_SSD", "031544", "Bx", base_url=base_url) as bx:
                bx_arr = bx.read()
                bx_slab = bx.read_slab(0, 4)
                print(f"\ncube('SPIN4D_SSD', '031544', 'Bx'): "
                      f"shape={bx.shape}, dtype={bx.dtype}, "
                      f"slab[0:4].shape={bx_slab.shape}")
                assert bx.shape == (16, 16, 8)
                assert bx_slab.shape == (4, 16, 8)
                assert np.array_equal(bx_arr[0:4], bx_slab)

            # Same path via integer step
            with cube("SPIN4D_SSD", 31544, "T", base_url=base_url) as t:
                assert t.shape == (16, 16, 8)
                print(f"cube(..., step=31544 [int], 'T'): shape={t.shape}  "
                      f"(int step coerced to '031544')")

            # 4. stokes() — by wavelength
            with stokes("SPIN4D_SSD", "031544", wavelength=6302,
                        base_url=base_url) as f:
                keys = list(f.keys())
                stokes_i_slice = f["stokes/I"][4:8, 4:8, :]
                print(f"\nstokes('SPIN4D_SSD', '031544', wavelength=6302): "
                      f"top-level keys={keys}, "
                      f"stokes/I[4:8,4:8,:].shape={stokes_i_slice.shape}")
                assert "stokes" in keys
                assert stokes_i_slice.shape == (4, 4, 11)

            # 5. error handling
            try:
                cube("SPIN4D_SSD", "031544", "not_a_variable", base_url=base_url)
            except ValueError as e:
                print(f"\nbad var name correctly rejected: {e}")
            else:
                raise AssertionError("expected ValueError for unknown variable")

            try:
                stokes("SPIN4D_SSD", "031544", wavelength=999, base_url=base_url)
            except ValueError as e:
                print(f"bad wavelength correctly rejected: {e}")
            else:
                raise AssertionError("expected ValueError for bad wavelength")

            print("\n[OK] High-level dataset API validated end-to-end.")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
