"""Decoder worker for stream_decode.py (a separate module so spawned processes can import it)."""

from __future__ import annotations

import os
import sys
import time

import numpy as np


def decoder_worker(so_dir: str, cfg: tuple, inq, outq) -> None:
    """One CPU decoder: its own QecMemory (same circuit, same DEM), single-threaded decoding."""
    sys.path.insert(0, so_dir)
    import qumo_py

    code, d, rounds, p = cfg
    mem = qumo_py.qec_memory(code, d, rounds, p)
    outq.put(("ready", os.getpid()))
    while True:
        item = inq.get()
        if item is None:
            break
        bid, emitted, dets, obs, shots = item
        t_start = time.perf_counter()
        pred, secs, events = mem.decode(dets, shots, 1)
        done = time.perf_counter()
        o = np.frombuffer(bytes.fromhex(obs), dtype=np.uint8)
        pr = np.frombuffer(bytes.fromhex(pred), dtype=np.uint8)
        errors = int(np.count_nonzero(o != pr))
        outq.put(("done", bid, emitted, t_start, done, float(secs), int(events), errors, pred))
