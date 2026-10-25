"""One-time reclamation of only the calling Windows worker's resident pages.

This does not free committed tensors, alter model data, or touch other processes.
Subsequent page faults remain part of measured inference/episode wall time.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import os
import time


def reclaim_own_startup_pages():
    import psutil
    process = psutil.Process()

    def snapshot():
        memory = process.memory_info()
        return {"rss_bytes": memory.rss, "private_bytes": getattr(memory, "private", None),
                "system_available_bytes": psutil.virtual_memory().available}

    started = time.perf_counter()
    collected = gc.collect()
    before = snapshot()
    record = {"scope": "calling worker only via GetCurrentProcess; no other process handle",
              "pid": os.getpid(), "created": process.create_time(), "gc_objects_collected": collected, "before": before,
              "phase": "once after model load, before simulator construction and first inference", "calls": 1}
    if os.name != "nt":
        record.update(supported=False, success=None, after=snapshot(), seconds=time.perf_counter() - started)
        return record
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
    success = bool(kernel.K32EmptyWorkingSet(kernel.GetCurrentProcess()))
    record.update(supported=True, success=success, winerror=0 if success else ctypes.get_last_error(),
                  after=snapshot(), seconds=time.perf_counter() - started,
                  limitation="working-set eviction, not tensor deallocation or a steady-state memory guarantee")
    return record
