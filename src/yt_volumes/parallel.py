"""Row-band parallelism for per-pixel numpy work on full frames."""

import os

import numpy as np

_pool = None


def for_row_bands(n_rows, fn, bands=16):
    """Call `fn(rows)` with slices covering `range(n_rows)`, in a thread pool.
    numpy releases the GIL in ufuncs, so bands of a frame run in parallel;
    `fn` must only write its own rows.
    """
    global _pool
    if n_rows < 64:
        fn(slice(0, n_rows))
        return
    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor

        _pool = ThreadPoolExecutor(min(8, os.cpu_count() or 1))
    edges = np.linspace(0, n_rows, bands + 1).astype(int)
    list(_pool.map(fn, [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]))
