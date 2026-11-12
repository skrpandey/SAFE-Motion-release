"""Boundary-event log for the selection-repair reruns (protocol repair_selection_v14).

log_event appends one tab-separated line to the file named by SM_EVENT_LOG (inherited by DataLoader
workers) and does nothing when it is unset. It draws no random numbers, so RNG streams are unchanged.
"""
import os
import time


def log_event(kind, **info):
    path = os.environ.get("SM_EVENT_LOG")
    if not path:
        return
    with open(path, "a") as f:
        f.write("\t".join([f"{time.time():.3f}", str(os.getpid()), kind] +
                          [f"{k}={v}" for k, v in sorted(info.items())]) + "\n")
