"""End-to-end smoke test against the real SPIn4D-DR1 server.

Run when ``http://dtn-itc.ifa.hawaii.edu/spin4d/DR1/`` is reachable.
Exercises every public API entry point without downloading any whole 9 GB
file: lists runs/steps from the manifest, opens one MURaM cube and reads
a small slab, opens one Stokes file and inspects + slices it.

Total network traffic: ~tens of MB at most. No local cache file.
"""

from __future__ import annotations

import time

from spin4d_explorer import (
    DEFAULT_BASE_URL,
    cube,
    inspect_h5,
    list_runs,
    list_steps,
    stokes,
)


def _human(n: int) -> str:
    if n < 1e6:
        return f"{n / 1e3:.1f} KB"
    if n < 1e9:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    print(f"SPIn4D-DR1 base: {DEFAULT_BASE_URL}\n")

    # 1. Manifest-driven enumeration
    t = time.perf_counter()
    runs = list_runs()
    print(f"list_runs() -> {runs}  ({time.perf_counter() - t:.2f}s)")
    if not runs:
        raise RuntimeError("no runs returned — manifest empty or unreachable")

    target_run = runs[0]
    t = time.perf_counter()
    steps = list_steps(target_run)
    print(
        f"list_steps({target_run!r}) -> {len(steps)} steps "
        f"(first 3: {steps[:3]})  ({time.perf_counter() - t:.2f}s)\n"
    )
    if not steps:
        raise RuntimeError(f"no steps for {target_run}")

    target_step = steps[0]

    # 2. MURaM cube — open and slab-read
    print(f"cube({target_run!r}, {target_step!r}, 'Bx') ...")
    t = time.perf_counter()
    with cube(target_run, target_step, "Bx") as bx:
        print(
            f"  shape={bx.shape}  dtype={bx.dtype}  "
            f"fortran_order={bx.fortran_order}  "
            f"total_size={_human(bx.nbytes)}"
        )
        slab = bx.read_slab(0, min(4, bx.shape[-1 if bx.fortran_order else 0]))
        print(
            f"  slab shape={slab.shape}  "
            f"streamed={_human(slab.nbytes)}  "
            f"mean={slab.mean():+.3e}  std={slab.std():.3e}"
        )
    print(f"  total time: {time.perf_counter() - t:.2f}s\n")

    # 3. Stokes profile file — open, inspect, read a small slice
    print(f"stokes({target_run!r}, {target_step!r}, wavelength=6302) ...")
    t = time.perf_counter()
    with stokes(target_run, target_step, wavelength=6302) as f:
        keys = list(f.keys())
        print(f"  top-level keys: {keys}")
        # Walk to find the first real dataset for a slice read
        first_path: str | None = None

        def visit(name: str, obj: object) -> None:
            nonlocal first_path
            import h5py as _h5

            if first_path is None and isinstance(obj, _h5.Dataset):
                first_path = name

        f.visititems(visit)
        if first_path is None:
            raise RuntimeError("no datasets in Stokes file")
        ds = f[first_path]
        # tiny slice so we don't pull a multi-GB read
        slc = tuple(slice(0, min(8, d)) for d in ds.shape)
        chunk = ds[slc]
        print(
            f"  first dataset {first_path!r}: shape={ds.shape}  "
            f"dtype={ds.dtype}  total_size={_human(ds.nbytes)}"
        )
        print(
            f"  read slice {slc} -> shape={chunk.shape}  "
            f"streamed={_human(chunk.nbytes)}"
        )
    print(f"  total time: {time.perf_counter() - t:.2f}s\n")

    # 4. Low-level inspect — just metadata, no array reads
    from spin4d_explorer.dataset import _format_step

    file_name = f"stokes-{_format_step(target_step)}-6302.h5"
    url = f"{DEFAULT_BASE_URL}/{target_run}/{file_name}"
    print(f"inspect_h5({url!r}) ...")
    t = time.perf_counter()
    items = inspect_h5(url)
    print(f"  {len(items)} datasets, total payload {_human(sum(i.size_bytes for i in items))}")
    for i in items[:6]:
        print(
            f"    {i.path}: shape={i.shape}  dtype={i.dtype}  "
            f"size={_human(i.size_bytes)}  chunks={i.chunks}"
        )
    if len(items) > 6:
        print(f"    ... ({len(items) - 6} more)")
    print(f"  total time: {time.perf_counter() - t:.2f}s")

    print("\n[OK] Full API verified against the real SPIn4D-DR1 server.")


if __name__ == "__main__":
    main()
